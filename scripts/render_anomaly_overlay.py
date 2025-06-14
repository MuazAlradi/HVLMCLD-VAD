import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

class Learner(nn.Module):
    def __init__(self, input_dim=10752, drop_p=0.0):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = 0.6
        self.weight_init()
        self.vars = nn.ParameterList()

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        return self.vars

class VideoAnomalyDataset(Dataset):
    def __init__(self, dataset_name, data_root='./data', video_filter=None):
        super(VideoAnomalyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.dataset_path = self.data_root / dataset_name
        
        # Define label file names and video extensions based on dataset
        dataset_configs = {
            'ucf_crime': {
                'anomaly_file': 'UCF_test_anomalyv2.txt',
                'normal_file': 'UCF_test_normalv2.txt',
                'video_ext': '.mp4',
                'video_subdir': 'videos'
            },
            'shanghai_tech': {
                'anomaly_file': 'SHT_test_anomalyv2.txt',
                'normal_file': 'SHT_test_normalv2.txt',
                'video_ext': '.avi',
                'video_subdir': 'videos'
            },
            'xd_violence': {
                'anomaly_file': 'XDV_test_anomalyv2.txt',
                'normal_file': 'XDV_test_normalv2.txt',
                'video_ext': '.mp4',
                'video_subdir': 'videos'
            }
        }
        
        if dataset_name not in dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(dataset_configs.keys())}")
        
        self.config = dataset_configs[dataset_name]
        
        # Load video data
        self.video_data = []
        self._load_video_data()
        
        # Apply video filter
        if video_filter:
            self.video_data = self._apply_filter(video_filter)
        
        print(f"Total videos to process: {len(self.video_data)}")

    def _load_video_data(self):
        """Load video data from label files"""
        # Load anomaly videos
        anomaly_path = self.dataset_path / 'labels' / self.config['anomaly_file']
        if anomaly_path.exists():
            with open(anomaly_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if '|' in line:
                        parts = line.split('|')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = [int(i) for i in parts[2][1:-2].split(',')]
                            self.video_data.append({
                                'name': name,
                                'frames': frames,
                                'gts': gts,
                                'type': 'anomaly'
                            })
        
        # Load normal videos
        normal_path = self.dataset_path / 'labels' / self.config['normal_file']
        if normal_path.exists():
            with open(normal_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if ' ' in line:
                        parts = line.split(' ')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = int(parts[2])
                            self.video_data.append({
                                'name': name,
                                'frames': frames,
                                'gts': gts,
                                'type': 'normal'
                            })

    def _apply_filter(self, video_filter):
        """Apply video filtering based on user input"""
        if video_filter['type'] == 'specific':
            video_name = video_filter['value']
            filtered_data = [item for item in self.video_data if video_name in item['name']]
            if not filtered_data:
                print(f"Warning: No videos found matching '{video_name}'")
            return filtered_data
        
        elif video_filter['type'] == 'random':
            num_videos = min(video_filter['value'], len(self.video_data))
            return random.sample(self.video_data, num_videos)
        
        elif video_filter['type'] == 'all':
            return self.video_data
        
        else:
            raise ValueError(f"Unknown filter type: {video_filter['type']}")

    def __len__(self):
        return len(self.video_data)

    def __getitem__(self, idx):
        video_info = self.video_data[idx]
        
        # Get video path
        video_name = video_info['name']
        video_path = self._get_video_path(video_name)
        
        # Get feature path
        feature_path = self._get_feature_path(video_name)
        
        return {
            'video_path': video_path,
            'feature_path': feature_path,
            'video_info': video_info
        }

    def _get_video_path(self, video_name):
        """Get the full path to the video file"""
        video_base = Path(video_name).stem
        
        # Try different possible paths
        possible_paths = [
            self.dataset_path / self.config['video_subdir'] / f"{video_name}",
            self.dataset_path / self.config['video_subdir'] / f"{video_base}{self.config['video_ext']}",
        ]
        
        # For UCF-Crime, videos are organized in category folders
        if self.dataset_name == 'ucf_crime':
            category = video_name.split('_')[0] if '_' in video_name else video_name.split('/')[0]
            possible_paths.extend([
                self.dataset_path / self.config['video_subdir'] / category / f"{video_name}",
                self.dataset_path / self.config['video_subdir'] / category / f"{video_base}{self.config['video_ext']}",
            ])
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Video file not found for: {video_name}")

    def _get_feature_path(self, video_name):
        """Get the full path to the feature file"""
        video_base = Path(video_name).stem
        
        # Try different possible paths
        possible_paths = [
            self.data_root / 'visual_features' / self.dataset_name / f"{video_base}.npy",
            self.data_root / 'visual_features' / self.dataset_name / f"{video_name}.npy",
        ]
        
        # For UCF-Crime, features might be organized in category folders
        if self.dataset_name == 'ucf_crime':
            category = video_name.split('_')[0] if '_' in video_name else video_name.split('/')[0]
            possible_paths.extend([
                self.data_root / 'visual_features' / self.dataset_name / category / f"{video_base}.npy",
                self.data_root / 'visual_features' / self.dataset_name / category / f"{video_name}.npy",
            ])
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Feature file not found for: {video_name}")

def compute_anomaly_scores(feature_path, model, device):
    """Compute anomaly scores for a video"""
    # Load visual features
    rgb_npy = np.load(feature_path)
    inputs = torch.from_numpy(rgb_npy).view(-1, rgb_npy.shape[-1]).to(device)
    
    with torch.no_grad():
        scores = model(inputs)
        scores = scores.cpu().detach().numpy()
    
    return scores

def interpolate_scores_to_frames(scores, total_frames):
    """Interpolate segment-based scores to frame-level scores"""
    score_list = np.zeros(total_frames)
    
    # Create mapping from segments to frames (assuming 16 frames per segment)
    segments_per_video = len(scores)
    if segments_per_video > 0:
        step = np.round(np.linspace(0, total_frames // 16, segments_per_video + 1))
        
        for j in range(segments_per_video):
            start_frame = int(step[j]) * 16
            end_frame = min(int(step[j+1]) * 16, total_frames)
            if start_frame < total_frames:
                score_list[start_frame:end_frame] = scores[j]
    
    return score_list

def create_ground_truth_mask(video_info):
    """Create ground truth mask for anomaly videos"""
    total_frames = video_info['frames']
    gt_mask = np.zeros(total_frames)
    
    if video_info['type'] == 'anomaly' and isinstance(video_info['gts'], list):
        gts = video_info['gts']
        for k in range(len(gts) // 2):
            start = max(0, gts[k*2] - 1)  # Convert to 0-indexed
            end = min(gts[k*2+1], total_frames)
            gt_mask[start:end] = 1
    
    return gt_mask

def render_video_with_overlay(video_path, feature_path, video_info, model, device, output_path, 
                            show_ground_truth=True, score_threshold=0.5):
    """Render video with anomaly score overlay"""
    
    # Compute anomaly scores
    scores = compute_anomaly_scores(feature_path, model, device)
    frame_scores = interpolate_scores_to_frames(scores, video_info['frames'])
    
    # Create ground truth mask
    gt_mask = create_ground_truth_mask(video_info)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure frame count matches
    if len(frame_scores) != total_frames:
        print(f"Warning: Score length ({len(frame_scores)}) != video frames ({total_frames})")
        # Resize frame_scores to match video length
        if len(frame_scores) < total_frames:
            # Pad with last score
            frame_scores = np.pad(frame_scores, (0, total_frames - len(frame_scores)), 
                                mode='constant', constant_values=frame_scores[-1] if len(frame_scores) > 0 else 0)
        else:
            # Truncate
            frame_scores = frame_scores[:total_frames]
        
        # Do the same for gt_mask
        if len(gt_mask) < total_frames:
            gt_mask = np.pad(gt_mask, (0, total_frames - len(gt_mask)), mode='constant', constant_values=0)
        else:
            gt_mask = gt_mask[:total_frames]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc=f"Processing {video_info['name']}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(frame_scores):
            score = frame_scores[frame_idx]
            is_anomaly_gt = gt_mask[frame_idx] > 0.5
            is_anomaly_pred = score > score_threshold
            
            # Create overlay text
            score_text = f"Anomaly Score: {score:.3f}"
            
            # Determine colors based on score and ground truth
            if video_info['type'] == 'anomaly' and show_ground_truth:
                if is_anomaly_gt:
                    # Ground truth anomaly region
                    if is_anomaly_pred:
                        color = (0, 255, 0)  # Green - Correct detection
                        status = "TP"
                    else:
                        color = (0, 165, 255)  # Orange - Missed detection
                        status = "FN"
                else:
                    # Ground truth normal region
                    if is_anomaly_pred:
                        color = (0, 0, 255)  # Red - False alarm
                        status = "FP"
                    else:
                        color = (255, 255, 255)  # White - Correct normal
                        status = "TN"
                
                gt_text = f"GT: {'Anomaly' if is_anomaly_gt else 'Normal'} ({status})"
            else:
                # For normal videos or when not showing ground truth
                if is_anomaly_pred:
                    color = (0, 0, 255)  # Red for predicted anomalies
                else:
                    color = (255, 255, 255)  # White for normal
                gt_text = f"Predicted: {'Anomaly' if is_anomaly_pred else 'Normal'}"
            
            # Add semi-transparent background for better readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Add text overlay
            cv2.putText(frame, score_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, gt_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add score bar visualization
            bar_width = 200
            bar_height = 10
            bar_x = width - bar_width - 20
            bar_y = 20
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Score bar
            score_width = int(bar_width * score)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + score_width, bar_y + bar_height), color, -1)
            # Threshold line
            thresh_x = bar_x + int(bar_width * score_threshold)
            cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_height), (255, 255, 0), 2)
            
            # Bar labels
            cv2.putText(frame, "0.0", (bar_x - 20, bar_y + bar_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "1.0", (bar_x + bar_width - 10, bar_y + bar_height + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Render anomaly score overlays on videos')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ucf_crime', 'shanghai_tech', 'xd_violence'],
                       help='Dataset to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='./output/visualizations',
                       help='Directory to save annotated videos')
    parser.add_argument('--save_video', type=str, default='True',
                       help='Set to True to export videos')
    parser.add_argument('--input_dim', type=int, default=10752,
                       help='Input dimension for the model')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Threshold for anomaly classification')
    parser.add_argument('--show_ground_truth', action='store_true', default=True,
                       help='Show ground truth comparison for anomaly videos')
    
    # Video selection options (mutually exclusive)
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument('--video_name', type=str,
                           help='Specific video name to process')
    video_group.add_argument('--num_random', type=int,
                           help='Number of random videos to process')
    video_group.add_argument('--all_videos', action='store_true',
                           help='Process all videos')
    
    args = parser.parse_args()
    
    # Parse save_video argument
    save_video = args.save_video.lower() == 'true'
    if not save_video:
        print("save_video is False. No videos will be saved.")
        return
    
    # Determine video filter
    if args.video_name:
        video_filter = {'type': 'specific', 'value': args.video_name}
    elif args.num_random:
        video_filter = {'type': 'random', 'value': args.num_random}
    elif args.all_videos:
        video_filter = {'type': 'all', 'value': None}
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = Learner(input_dim=args.input_dim, drop_p=0.0).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {args.model_path}")
    
    # Create dataset
    dataset = VideoAnomalyDataset(args.dataset, args.data_root, video_filter)
    
    if len(dataset) == 0:
        print("No videos to process. Exiting.")
        return
    
    # Create output directory
    output_path = Path(args.output_dir) / args.dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    print(f"Processing {len(dataset)} videos...")
    
    for i in range(len(dataset)):
        data = dataset[i]
        video_info = data['video_info']
        
        # Create output filename
        video_name = Path(video_info['name']).stem
        output_filename = f"{video_name}_overlay.mp4"
        output_video_path = output_path / output_filename
        
        try:
            print(f"\nProcessing video {i+1}/{len(dataset)}: {video_info['name']}")
            render_video_with_overlay(
                data['video_path'], 
                data['feature_path'], 
                video_info, 
                model, 
                device, 
                output_video_path,
                show_ground_truth=args.show_ground_truth,
                score_threshold=args.score_threshold
            )
            print(f"Saved: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {video_info['name']}: {str(e)}")
            continue
    
    print(f"\nAll videos processed. Outputs saved to: {output_path}")

if __name__ == "__main__":
    main()