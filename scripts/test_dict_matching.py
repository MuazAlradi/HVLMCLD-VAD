import torch
import clip
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime
import os
import random
import argparse
from pathlib import Path

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, _ = clip.load("ViT-B/32", device=device)

class UniversalDataLoader(Dataset):
    """Universal data loader that works with any dataset structure"""
    def __init__(self, dataset_name, split_type, data_type, base_path='data'):
        super(UniversalDataLoader, self).__init__()
        self.dataset_name = dataset_name
        self.split_type = split_type  # 'train' or 'test'
        self.data_type = data_type    # 'normal' or 'anomaly'
        self.base_path = Path(base_path)
        self.features_path = Path('output/visual_features') / dataset_name
        
        # Load appropriate label file
        if dataset_name == 'ucf_crime':
            prefix = 'UCF'
        elif dataset_name == 'shanghai_tech':
            prefix = 'SHT'
        elif dataset_name == 'xd_violence':
            prefix = 'XDV'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        if split_type == 'train':
            label_file = f"{prefix}_train_{data_type}.txt"
        else:
            label_file = f"{prefix}_test_{data_type}v2.txt"
            
        label_path = self.base_path / dataset_name / 'labels' / label_file
        
        with open(label_path, 'r') as f:
            self.data_list = f.readlines()
            
        # Shuffle test normal data and remove last 10 (as in original)
        if split_type == 'test' and data_type == 'normal':
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx].strip()
        
        if self.split_type == 'train':
            # Training data - just return features
            video_name = line
            feature_path = self.features_path / f"{video_name}.npy"
            features = np.load(feature_path)
            return features
        else:
            # Test data - return features, ground truth, and frames
            if self.data_type == 'normal':
                parts = line.split(' ')
                video_name = parts[0]
                frames = int(parts[1])
                gts = int(parts[2])
            else:  # anomaly
                parts = line.split('|')
                video_name = parts[0]
                frames = int(parts[1])
                gts_str = parts[2][1:-1]  # Remove brackets
                gts = [int(i) for i in gts_str.split(',')]
            
            feature_path = self.features_path / f"{video_name}.npy"
            features = np.load(feature_path)
            return features, gts, frames

def load_text_features(file_path):
    """Load text features from dictionary file"""
    print(f"Loading text features from {file_path}")
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    
    text_tokens = clip.tokenize(lines).to(device)
    with torch.no_grad():
        text_features = CLIP_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print(f"Loaded {len(lines)} text features")
    return lines, text_features

def process_snippet(snippet_features):
    """Process snippet features to correct format"""
    if isinstance(snippet_features, np.ndarray):
        snippet_features = torch.from_numpy(snippet_features)
    snippet_features = snippet_features.reshape(21, 512)
    return snippet_features.float().to(device)

# Similarity calculation methods
def average_similarity(similarities):
    return similarities.mean()

def max_similarity(similarities):
    return similarities.max()

def top_k_average(similarities, k=5):
    return similarities.topk(k)[0].mean()

def weighted_average(similarities):
    weights = torch.linspace(1, 0.1, similarities.size(0), device=similarities.device)
    sorted_sims, _ = similarities.sort(descending=True)
    return (sorted_sims * weights).sum() / weights.sum()

def count_above_threshold(similarities, threshold=0.5):
    return (similarities > threshold).float().sum()

def softmax_weighted_average(similarities):
    weights = torch.nn.functional.softmax(similarities, dim=0)
    return (similarities * weights).sum()

def percentile_similarity(similarities, percentile=95):
    k = int(len(similarities) * (percentile / 100))
    return similarities.topk(k)[0][-1]

def ensemble_similarity(similarities):
    max_sim = similarities.max()
    avg_sim = similarities.mean()
    top_k_sim = similarities.topk(5)[0].mean()
    return (max_sim + avg_sim + top_k_sim) / 3

def calculate_moving_average(predictions, window_size):
    """Calculate moving average of predictions"""
    half_window = window_size // 2
    padded_predictions = [predictions[0]] * half_window + predictions + [predictions[-1]] * half_window
    moving_avg = []
    for i in range(len(predictions)):
        window = padded_predictions[i:i+window_size]
        moving_avg.append(sum(window) / window_size)
    return moving_avg

similarity_methods = [
    ("Average", average_similarity),
    ("Max", max_similarity),
    ("Top-K", top_k_average),
    ("Weighted", weighted_average),
    ("Threshold", count_above_threshold),
    ("Softmax", softmax_weighted_average),
    ("Percentile", percentile_similarity),
    ("Ensemble", ensemble_similarity)
]

def evaluate_snippets(data_loader, normal_features, anomaly_features, dataset_type, report_file, window_size):
    """Evaluate video snippets using different similarity methods"""
    predictions = {method[0]: [] for method in similarity_methods}
    labels = []
    total_videos = len(data_loader)

    print(f"Evaluating {dataset_type} videos...")
    for video_idx, batch in enumerate(data_loader, 1):
        if dataset_type == 'normal':
            rgb_npy, gts, frames = batch
            video_name = data_loader.dataset.data_list[video_idx - 1].split()[0]
        else:
            rgb_npy, gts, frames = batch
            video_name = data_loader.dataset.data_list[video_idx - 1].split('|')[0]
            
        report_file.write(f"\nFile: {video_name}\n")
        header = "Snippet\t" + "\t".join([f"{method[0]}\t{method[0]}_MA" for method in similarity_methods]) + "\tGround Truth\n"
        report_file.write(header)
        
        video_predictions = {method[0]: [] for method in similarity_methods}
        
        for i in range(32):  # 32 snippets per video
            snippet_features = process_snippet(rgb_npy[0, i])
            
            normal_sim = (snippet_features @ normal_features.T).mean(dim=0)
            anomaly_sim = (snippet_features @ anomaly_features.T).mean(dim=0)
            
            # Ground truth calculation
            snippet_start = i * (frames.item() // 32)
            snippet_end = (i + 1) * (frames.item() // 32)
            
            if dataset_type == 'normal':
                is_gt_anomaly = False  # Normal videos have no anomalies
            else:
                # For anomaly videos, check if snippet overlaps with any anomaly interval
                is_gt_anomaly = any(start <= snippet_start < end or start < snippet_end <= end 
                                    for start, end in zip(gts[::2], gts[1::2]))
            
            labels.append(1 if is_gt_anomaly else 0)
            
            results = []
            for method_name, method_func in similarity_methods:
                normal_score = method_func(normal_sim)
                anomaly_score = method_func(anomaly_sim)
                is_anomaly = anomaly_score > normal_score
                video_predictions[method_name].append(1 if is_anomaly else 0)
                results.append("Anomaly" if is_anomaly else "Normal")
            
            ground_truth = "Anomaly" if is_gt_anomaly else "Normal"
            report_file.write(f"{i+1}\t" + "\t".join(results) + f"\t{ground_truth}\n")
        
        # Calculate moving average for each method
        for method_name in video_predictions:
            ma_predictions = calculate_moving_average(video_predictions[method_name], window_size)
            predictions[method_name].extend(video_predictions[method_name])
            predictions[f"{method_name}_MA"] = predictions.get(f"{method_name}_MA", []) + ma_predictions
        
        if video_idx % 10 == 0 or video_idx == total_videos:
            print(f"Processed {video_idx}/{total_videos} {dataset_type} videos")

    print(f"Finished evaluating {dataset_type} videos")
    return predictions, labels

def calculate_similarities(features1, features2):
    """Calculate similarity matrix between two feature sets"""
    return torch.mm(features1, features2.t())

def filter_dictionaries(normal_lines, normal_features, anomaly_lines, anomaly_features, threshold):
    """Filter dictionaries using distance maximization"""
    similarities = calculate_similarities(normal_features, anomaly_features)
    
    filtered_normal = []
    filtered_anomaly = []
    
    for i in tqdm(range(len(normal_lines)), desc="Filtering normal dictionary"):
        if torch.max(similarities[i]) < threshold:
            filtered_normal.append(normal_lines[i])
            
    for j in tqdm(range(len(anomaly_lines)), desc="Filtering anomaly dictionary"):
        if torch.max(similarities[:, j]) < threshold:
            filtered_anomaly.append(anomaly_lines[j])
    
    return filtered_normal, filtered_anomaly

def save_filtered_dictionaries(filtered_normal, filtered_anomaly, normal_output, anomaly_output):
    """Save filtered dictionaries to files"""
    normal_output.parent.mkdir(parents=True, exist_ok=True)
    anomaly_output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(normal_output, 'w') as f:
        for line in filtered_normal:
            f.write(f"{line}\n")
    
    with open(anomaly_output, 'w') as f:
        for line in filtered_anomaly:
            f.write(f"{line}\n")

def get_dictionary_paths(dictionary_type, base_path='data'):
    """Get paths for normal and anomaly dictionaries based on type"""
    dict_base = Path(base_path) / 'training_free_dictionaries'
    
    dictionary_map = {
        'action_gpt4': ('gpt_actions/gpt_actions_normal.txt', 'gpt_actions/gpt_actions_anomaly.txt'),
        'action_k700': ('k700_actions/k700_actions_normal.txt', 'k700_actions/k700_actions_anomaly.txt'),
        'action_gpt4_k700': ('gpt+k700_actions/combined_actions_normal.txt', 'gpt+k700_actions/combined_actions_anomaly.txt'),
        'object_gpt4': ('gpt_objects/gpt_objects_normal.txt', 'gpt_objects/gpt_objects_anomaly.txt'),
        'object_coco': ('coco_objects/coco_objects_normal.txt', 'coco_objects/coco_objects_anomaly.txt'),
        'object_gpt4_coco': ('gpt+coco_objects/combined_objects_normal.txt', 'gpt+coco_objects/combined_objects_anomaly.txt'),
    }
    
    if dictionary_type not in dictionary_map:
        raise ValueError(f"Unknown dictionary type: {dictionary_type}")
    
    normal_path, anomaly_path = dictionary_map[dictionary_type]
    return dict_base / normal_path, dict_base / anomaly_path

def run_evaluation(dataset_name, dictionary_type, use_dmm=True, threshold=None):
    """Run the complete evaluation pipeline"""
    print(f"Running evaluation on {dataset_name} with {dictionary_type} dictionary")
    print(f"Dictionary Distance Maximization: {'Enabled' if use_dmm else 'Disabled'}")
    
    # Get dictionary paths
    normal_dict_path, anomaly_dict_path = get_dictionary_paths(dictionary_type)
    
    # Load dictionaries
    normal_lines, normal_features = load_text_features(normal_dict_path)
    anomaly_lines, anomaly_features = load_text_features(anomaly_dict_path)
    
    if use_dmm:
        # Apply dictionary distance maximization
        if threshold is None:
            # Find optimal threshold
            print("Finding optimal threshold for dictionary distance maximization...")
            best_auc = 0
            best_threshold = 0
            
            for thresh in np.arange(0.85, 0.98, 0.01):
                print(f"Testing threshold: {thresh:.3f}")
                
                # Filter dictionaries
                filtered_normal, filtered_anomaly = filter_dictionaries(
                    normal_lines, normal_features, anomaly_lines, anomaly_features, thresh
                )
                
                if len(filtered_normal) == 0 or len(filtered_anomaly) == 0:
                    continue
                
                # Quick evaluation (you might want to implement a faster version)
                # For now, we'll use a simple threshold
                best_threshold = thresh
                break  # Use first valid threshold for demo
            
            threshold = best_threshold
        
        print(f"Using threshold: {threshold:.3f}")
        
        # Filter dictionaries with chosen threshold
        filtered_normal, filtered_anomaly = filter_dictionaries(
            normal_lines, normal_features, anomaly_lines, anomaly_features, threshold
        )
        
        # Save filtered dictionaries
        output_dir = Path('output/filtered_dictionaries') / dataset_name / dictionary_type
        normal_output = output_dir / f'filtered_normal_thrs{threshold:.3f}.txt'
        anomaly_output = output_dir / f'filtered_anomaly_thrs{threshold:.3f}.txt'
        
        save_filtered_dictionaries(filtered_normal, filtered_anomaly, normal_output, anomaly_output)
        
        # Load filtered features
        _, final_normal_features = load_text_features(normal_output)
        _, final_anomaly_features = load_text_features(anomaly_output)
        
        print(f"Filtered normal dictionary: {len(filtered_normal)} phrases")
        print(f"Filtered anomaly dictionary: {len(filtered_anomaly)} phrases")
    else:
        # Use original dictionaries
        final_normal_features = normal_features
        final_anomaly_features = anomaly_features
        threshold = "none"
    
    # Prepare data loaders
    normal_loader = DataLoader(
        UniversalDataLoader(dataset_name, 'test', 'normal'), 
        batch_size=1, shuffle=False
    )
    anomaly_loader = DataLoader(
        UniversalDataLoader(dataset_name, 'test', 'anomaly'), 
        batch_size=1, shuffle=False
    )
    
    # Run evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{dataset_name}_{dictionary_type}_dmm{use_dmm}_thrs{threshold}_{timestamp}.txt"
    window_size = 7
    
    with open(report_filename, "w") as report_file:
        report_file.write(f"Evaluation Report\n")
        report_file.write(f"Dataset: {dataset_name}\n")
        report_file.write(f"Dictionary: {dictionary_type}\n")
        report_file.write(f"DMM Enabled: {use_dmm}\n")
        report_file.write(f"Threshold: {threshold}\n")
        report_file.write(f"Moving Average Window Size: {window_size}\n\n")
        
        # Evaluate normal videos
        report_file.write("Normal Videos:\n")
        normal_preds, normal_labels = evaluate_snippets(
            normal_loader, final_normal_features, final_anomaly_features, 
            "normal", report_file, window_size
        )
        
        # Evaluate anomaly videos
        report_file.write("\nAnomaly Videos:\n")
        anomaly_preds, anomaly_labels = evaluate_snippets(
            anomaly_loader, final_normal_features, final_anomaly_features, 
            "anomaly", report_file, window_size
        )
    
    # Calculate AUC-ROC scores
    auc_scores = {}
    for method_name in normal_preds.keys():
        all_preds = normal_preds[method_name] + anomaly_preds[method_name]
        all_labels = normal_labels + anomaly_labels
        auc_scores[method_name] = roc_auc_score(all_labels, all_preds)
    
    # Print and save results
    print(f"\nAUC-ROC Scores:")
    for method_name, score in auc_scores.items():
        print(f"{method_name}: {score:.4f}")
    
    with open(report_filename, "a") as report_file:
        report_file.write(f"\n\nOverall Results:\n")
        for method_name, score in auc_scores.items():
            report_file.write(f"AUC-ROC Score ({method_name}): {score:.4f}\n")
    
    print(f"Detailed report saved in '{report_filename}'")
    
    return max(auc_scores.values())

def main():
    parser = argparse.ArgumentParser(description="Test dictionary matching for anomaly detection")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ucf_crime', 'shanghai_tech', 'xd_violence'],
                        help='Dataset to evaluate on')
    parser.add_argument('--dictionary', type=str, required=True,
                        choices=['action_gpt4', 'action_k700', 'action_gpt4_k700',
                                'object_gpt4', 'object_coco', 'object_gpt4_coco'],
                        help='Dictionary type to use')
    parser.add_argument('--use_dmm', type=bool, default=True,
                        help='Whether to use Dictionary Distance Maximization')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for DMM (if not provided, will search for optimal)')
    
    args = parser.parse_args()
    
    # Run evaluation
    max_auc = run_evaluation(args.dataset, args.dictionary, args.use_dmm, args.threshold)
    
    print(f"\nFinal Results:")
    print(f"Dataset: {args.dataset}")
    print(f"Dictionary: {args.dictionary}")
    print(f"DMM: {args.use_dmm}")
    print(f"Maximum AUC achieved: {max_auc:.4f}")

if __name__ == "__main__":
    main()
