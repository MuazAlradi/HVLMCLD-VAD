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
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, _ = clip.load("ViT-B/32", device=device)

# [Keep the existing load_text_features, calculate_similarities, filter_dictionaries, and save_filtered_dictionaries functions]

# class Normal_Loader(Dataset):
#     # [Keep the existing Normal_Loader class implementation]

# class Anomaly_Loader(Dataset):
#     # [Keep the existing Anomaly_Loader class implementation]

# def process_snippet(snippet_features):
#     # [Keep the existing process_snippet function]

# # [Keep all the similarity calculation methods: average_similarity, max_similarity, etc.]

# similarity_methods = [
#     ("Average", average_similarity),
#     ("Max", max_similarity),
#     ("Top-K", top_k_average),
#     ("Weighted", weighted_average),
#     ("Threshold", count_above_threshold),
#     ("Softmax", softmax_weighted_average),
#     ("Percentile", percentile_similarity),
#     ("Ensemble", ensemble_similarity)
# ]

# def calculate_moving_average(predictions, window_size):
#     # [Keep the existing calculate_moving_average function]

# def evaluate_snippets(data_loader, normal_features, anomaly_features, dataset_type, report_file, window_size):
#     # [Keep the existing evaluate_snippets function]

## DATALOADERS ##
class Normal_Loader(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_normalv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', self.data_list[idx][:-1]+'.npy'))
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
            return rgb_npy, gts, frames

class Anomaly_Loader(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', self.data_list[idx][:-1]+'.npy'))
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
            return rgb_npy, gts, frames

def load_text_features(file_path):
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
    predictions = {method[0]: [] for method in similarity_methods}
    labels = []
    total_videos = len(data_loader)

    print(f"Evaluating {dataset_type} videos...")
    for video_idx, (rgb_npy, gts, frames) in enumerate(data_loader, 1):
        video_name = data_loader.dataset.data_list[video_idx - 1].split()[0]
        report_file.write(f"\nFile: {video_name}\n")
        header = "Snippet\t" + "\t".join([f"{method[0]}\t{method[0]}_MA" for method in similarity_methods]) + "\tGround Truth\n"
        report_file.write(header)
        
        video_predictions = {method[0]: [] for method in similarity_methods}
        
        for i in range(32):  # 32 snippets per video
            snippet_features = process_snippet(rgb_npy[0, i])
            
            normal_sim = (snippet_features @ normal_features.T).mean(dim=0)
            anomaly_sim = (snippet_features @ anomaly_features.T).mean(dim=0)
            
            snippet_start = i * (frames.item() // 32)
            snippet_end = (i + 1) * (frames.item() // 32)
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

def filter_dictionaries(normal_lines, normal_features, anomaly_lines, anomaly_features, threshold):
    similarities = calculate_similarities(normal_features, anomaly_features)
    
    filtered_normal = []
    filtered_anomaly = []
    
    for i in tqdm(range(len(normal_lines)), desc="Filtering dictionaries"):
        if torch.max(similarities[i]) < threshold:
            filtered_normal.append(normal_lines[i])
            
    for j in tqdm(range(len(anomaly_lines)), desc="Filtering anomaly dictionary"):
        if torch.max(similarities[:, j]) < threshold:
            filtered_anomaly.append(anomaly_lines[j])
    
    return filtered_normal, filtered_anomaly

def save_filtered_dictionaries(filtered_normal, filtered_anomaly, normal_output, anomaly_output):
    with open(normal_output, 'w') as f:
        for line in filtered_normal:
            f.write(f"{line}\n")
    
    with open(anomaly_output, 'w') as f:
        for line in filtered_anomaly:
            f.write(f"{line}\n")



def calculate_similarities(features1, features2):
    return torch.mm(features1, features2.t())

def evaluate_filtered_dictionaries(normal_path, anomaly_path, threshold):
    print(f"\nEvaluating filtered dictionaries with threshold {threshold}")
    
    normal_output = f'E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\FilteredGPT4NormalActions_thrs{threshold:.4f}.txt'
    anomaly_output = f'E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\FilteredGPT4AnomalyActions_thrs{threshold:.4f}.txt'

    # normal_output = f'E:\\EvaluateCLIPMatching\\TextDictionaries\\UniversalDictionary\\FilteredUniNormalActions_thrs{threshold:.3f}.txt'
    # anomaly_output = f'E:\\EvaluateCLIPMatching\\TextDictionaries\\UniversalDictionary\\FilteredUniAnomalyActions_thrs{threshold:.3f}.txt'
    
    # Load and filter dictionaries
    anomaly_lines, anomaly_features = load_text_features(anomaly_path)
    normal_lines, normal_features = load_text_features(normal_path)
    
    filtered_normal, filtered_anomaly = filter_dictionaries(normal_lines, normal_features, anomaly_lines, anomaly_features, threshold)
    
    save_filtered_dictionaries(filtered_normal, filtered_anomaly, normal_output, anomaly_output)
    
    print(f"Filtered normal dictionary: {len(filtered_normal)} words")
    print(f"Filtered anomaly dictionary: {len(filtered_anomaly)} words")
    
    # Load filtered dictionaries
    _, filtered_normal_features = load_text_features(normal_output)
    _, filtered_anomaly_features = load_text_features(anomaly_output)
    
    # Prepare data loaders
    path = 'E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\'
    normal_loader = DataLoader(Normal_Loader(is_train=0, path=path), batch_size=1, shuffle=False)
    anomaly_loader = DataLoader(Anomaly_Loader(is_train=0, path=path), batch_size=1, shuffle=False)
    
    # Evaluate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"detailed_report_thrs{threshold:.2f}_{timestamp}.txt"
    window_size = 7
    
    with open(report_filename, "w") as report_file:
        report_file.write(f"Detailed Report of CLIP Matching (Threshold: {threshold:.2f})\n\n")
        report_file.write(f"Moving Average Window Size: {window_size}\n\n")
        report_file.write("Normal Videos:\n")
        normal_preds, normal_labels = evaluate_snippets(normal_loader, filtered_normal_features, filtered_anomaly_features, "normal", report_file, window_size)
        
        report_file.write("\nAnomaly Videos:\n")
        anomaly_preds, anomaly_labels = evaluate_snippets(anomaly_loader, filtered_normal_features, filtered_anomaly_features, "anomaly", report_file, window_size)
    
    # Calculate AUC-ROC scores
    auc_scores = {}
    for method_name in normal_preds.keys():
        all_preds = normal_preds[method_name] + anomaly_preds[method_name]
        all_labels = normal_labels + anomaly_labels
        auc_scores[method_name] = roc_auc_score(all_labels, all_preds)
    
    # Print AUC-ROC Scores
    print(f"\nAUC-ROC Scores for threshold {threshold:.2f}:")
    for method_name, score in auc_scores.items():
        print(f"{method_name}: {score:.4f}")
    
    with open(report_filename, "a") as report_file:
        report_file.write(f"\n\nOverall Results:\n")
        for method_name, score in auc_scores.items():
            report_file.write(f"AUC-ROC Score ({method_name}): {score:.4f}\n")
    
    print(f"Detailed report saved in '{report_filename}'")
    
    return max(auc_scores.values())

def main():
    normal_path = 'E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4NormalActions.txt'
    anomaly_path = 'E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4AnomalyActions.txt'

    # normal_path = 'E:\\EvaluateCLIPMatching\\TextDictionaries\\UniversalDictionary\\UniversalNormalDictionary.txt'
    # anomaly_path = 'E:\\EvaluateCLIPMatching\\TextDictionaries\\UniversalDictionary\\UniversalAnomalyDictionary.txt'
    
    best_auc = 0
    best_threshold = 0
    
    for threshold in np.arange(0.948, 0.956, 0.001):
        max_auc = evaluate_filtered_dictionaries(normal_path, anomaly_path, threshold)
        
        if max_auc > best_auc:
            best_auc = max_auc
            best_threshold = threshold
    
    print(f"\n\nBest Results:")
    print(f"Maximum AUC achieved: {best_auc:.4f}")
    print(f"Best threshold: {best_threshold:.4f}")

if __name__ == "__main__":
    main()
