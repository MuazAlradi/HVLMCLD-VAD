import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import random
import argparse
from pathlib import Path

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

class AnomalyDataset(Dataset):
    def __init__(self, dataset_name, data_root='./data', video_filter=None):
        super(AnomalyDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.dataset_path = self.data_root / dataset_name
        
        # Define label file names based on dataset
        label_files = {
            'ucf_crime': ('UCF_test_anomalyv2.txt', 'UCF_test_normalv2.txt'),
            'shanghai_tech': ('SHT_test_anomalyv2.txt', 'SHT_test_normalv2.txt'),
            'xd_violence': ('XDV_test_anomalyv2.txt', 'XDV_test_normalv2.txt')
        }
        
        if dataset_name not in label_files:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(label_files.keys())}")
        
        anomaly_file, normal_file = label_files[dataset_name]
        
        # Load anomaly videos
        self.anomaly_data = []
        anomaly_path = self.dataset_path / 'labels' / anomaly_file
        if anomaly_path.exists():
            with open(anomaly_path, 'r') as f:
                for line in f.readlines():
                    if dataset_name == 'ucf_crime':
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = [int(i) for i in parts[2][1:-2].split(',')]
                            self.anomaly_data.append((name, frames, gts, 'anomaly'))
                    else:
                        # Adapt for other datasets as needed
                        parts = line.strip().split('|')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = [int(i) for i in parts[2][1:-2].split(',')]
                            self.anomaly_data.append((name, frames, gts, 'anomaly'))
        
        # Load normal videos
        self.normal_data = []
        normal_path = self.dataset_path / 'labels' / normal_file
        if normal_path.exists():
            with open(normal_path, 'r') as f:
                for line in f.readlines():
                    if dataset_name == 'ucf_crime':
                        parts = line.strip().split(' ')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = int(parts[2])
                            self.normal_data.append((name, frames, gts, 'normal'))
                    else:
                        # Adapt for other datasets as needed
                        parts = line.strip().split(' ')
                        if len(parts) >= 3:
                            name = parts[0]
                            frames = int(parts[1])
                            gts = int(parts[2])
                            self.normal_data.append((name, frames, gts, 'normal'))
        
        # Combine all data
        self.all_data = self.anomaly_data + self.normal_data
        
        # Apply video filter
        if video_filter:
            self.all_data = self._apply_filter(video_filter)
        
        print(f"Loaded {len(self.anomaly_data)} anomaly and {len(self.normal_data)} normal videos")
        print(f"Total videos to process: {len(self.all_data)}")

    def _apply_filter(self, video_filter):
        """Apply video filtering based on user input"""
        if video_filter['type'] == 'specific':
            # Filter for specific video name
            video_name = video_filter['value']
            filtered_data = [item for item in self.all_data if video_name in item[0]]
            if not filtered_data:
                print(f"Warning: No videos found matching '{video_name}'")
            return filtered_data
        
        elif video_filter['type'] == 'random':
            # Select random number of videos
            num_videos = min(video_filter['value'], len(self.all_data))
            return random.sample(self.all_data, num_videos)
        
        elif video_filter['type'] == 'all':
            # Return all videos
            return self.all_data
        
        else:
            raise ValueError(f"Unknown filter type: {video_filter['type']}")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        name, frames, gts, video_type = self.all_data[idx]
        
        # Load visual features
        feature_path = self.data_root / 'visual_features' / self.dataset_name / f"{name}.npy"
        
        if not feature_path.exists():
            # Try alternative path structure
            category = name.split('_')[0] if '_' in name else name.split('/')[0] if '/' in name else ''
            if category:
                feature_path = self.data_root / 'visual_features' / self.dataset_name / category / f"{name}.npy"
        
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
        rgb_npy = np.load(feature_path)
        
        return rgb_npy, gts, frames, name, video_type

def plot_anomaly_scores(data_loader, model, save_dir, dataset_name):
    """Generate and save anomaly score plots"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, (inputs, gts, frames, name, video_type) in enumerate(data_loader):
            print(f"Processing video {i+1}/{len(data_loader)}: {name[0]}")
            
            frames = frames.item()
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            score = model(inputs)
            score = score.cpu().detach().numpy()
            
            # Create score list for all frames
            score_list = np.zeros(frames)
            step = np.round(np.linspace(0, torch.div(frames, 16, rounding_mode='floor').item(), 33))
            
            for j in range(32):
                start_frame = int(step[j]) * 16
                end_frame = int(step[j+1]) * 16
                if start_frame < frames:
                    end_frame = min(end_frame, frames)
                    score_list[start_frame:end_frame] = score[j]
            
            # Create ground truth list for anomaly videos
            gt_list = np.zeros(frames)
            is_anomaly = video_type[0] == 'anomaly'
            
            if is_anomaly and isinstance(gts, list):
                gts_list = gts if isinstance(gts, list) else gts.tolist()
                for k in range(len(gts_list) // 2):
                    s = gts_list[k*2]
                    e = min(gts_list[k*2+1], frames)
                    gt_list[s-1:e] = 1
            
            # Create plot
            plt.figure(figsize=(12, 4))
            plt.plot(score_list, color='orange', linewidth=2, label='Anomaly Score')
            
            if is_anomaly:
                plt.fill_between(range(len(gt_list)), 0, 1, where=gt_list, 
                               color='red', alpha=0.3, label='Ground Truth Anomaly')
            
            plt.ylim(0, 1)
            plt.xlabel('Frame')
            plt.ylabel('Anomaly Score')
            plt.legend()
            
            video_name = Path(name[0]).stem
            video_type_str = video_type[0]
            plt.title(f'Anomaly Score for {video_type_str.title()} Video: {video_name}')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_filename = f'{video_type_str}_{video_name}.png'
            plt.savefig(save_path / plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot: {plot_filename}")

def main():
    parser = argparse.ArgumentParser(description='Visualize anomaly scores for video datasets')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ucf_crime', 'shanghai_tech', 'xd_violence'],
                       help='Dataset to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='./output/anomaly_plots',
                       help='Directory to save plots')
    parser.add_argument('--input_dim', type=int, default=10752,
                       help='Input dimension for the model')
    
    # Video selection options (mutually exclusive)
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument('--video_name', type=str,
                           help='Specific video name to process')
    video_group.add_argument('--num_random', type=int,
                           help='Number of random videos to process')
    video_group.add_argument('--all_videos', action='store_true',
                           help='Process all videos')
    
    args = parser.parse_args()
    
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
    
    # Create dataset and data loader
    dataset = AnomalyDataset(args.dataset, args.data_root, video_filter)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    if len(dataset) == 0:
        print("No videos to process. Exiting.")
        return
    
    # Create output directory
    output_path = Path(args.output_dir) / args.dataset
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"Generating plots for {len(dataset)} videos...")
    plot_anomaly_scores(data_loader, model, output_path, args.dataset)
    
    print(f"All plots saved to: {output_path}")

if __name__ == "__main__":
    main()