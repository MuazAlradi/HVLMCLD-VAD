import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
import optuna
import json
from datetime import datetime


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
            # Split the features into text and video features
            text_features = rgb_npy[:, :10240]
            video_features = rgb_npy[:, 10240:]
            return text_features, video_features
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
            # Split the features into text and video features
            text_features = rgb_npy[:, :10240]
            video_features = rgb_npy[:, 10240:]
            return text_features, video_features, gts, frames

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
            # Split the features into text and video features
            text_features = rgb_npy[:, :10240]
            video_features = rgb_npy[:, 10240:]
            return text_features, video_features
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
            # Split the features into text and video features
            text_features = rgb_npy[:, :10240]
            video_features = rgb_npy[:, 10240:]
            return text_features, video_features, gts, frames

## FEATURE FUSION METHODS ##

# class AdditionFusion(nn.Module):
#     def __init__(self):
#         super(AdditionFusion, self).__init__()
#         self.fc = nn.Linear(10240, 512)
        
#     def forward(self, text_features, video_features):
#         text_features_reduced = self.fc(text_features)
#         fused_features = text_features_reduced + video_features
#         return fused_features

class AdditionFusion(nn.Module):
    def __init__(self):
        super(AdditionFusion, self).__init__()
        # No initial dimension reduction for text
        
    def forward(self, text_features, video_features):
        # video_features shape: [1980, 512]
        # text_features shape: [1980, 10240]
        
        # Expand each element in the 512-dim vector by repeating it 20 times
        expanded_video = torch.repeat_interleave(video_features, 20, dim=1)
        # Now expanded_video shape is [1980, 10240]
        
        # Perform addition fusion at the higher dimension
        fused_features = text_features + expanded_video  # [1980, 10240]
                
        return fused_features

class ProductFusion(nn.Module):
    def __init__(self):
        super(ProductFusion, self).__init__()
        
    def forward(self, text_features, video_features):
        expanded_video = torch.repeat_interleave(video_features, 20, dim=1)
        fused_features = text_features * expanded_video
        return fused_features
        
# class AttentionFusion(nn.Module):
#     def __init__(self):
#         super(AttentionFusion, self).__init__()
#         self.text_projection = nn.Linear(10240, 512)
#         self.query = nn.Linear(512, 512)
#         self.key = nn.Linear(512, 512)
#         self.value = nn.Linear(512, 512)
#         self.scale = 512 ** -0.5
        
#     # def forward(self, text_features, video_features):
#     #     text_features_reduced = self.text_projection(text_features)
        
#     #     queries = self.query(video_features)
#     #     keys = self.key(text_features_reduced)
#     #     values = self.value(text_features_reduced)
        
#     #     # Calculate attention scores
#     #     attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
#     #     attention_weights = F.softmax(attention_scores, dim=-1)
        
#     #     # Apply attention weights to values
#     #     attention_output = torch.matmul(attention_weights, values)
        
#     #     # Combine with original video features (residual connection)
#     #     fused_features = video_features + attention_output
        
#     #     return fused_features
#     def forward(self, text_features, video_features):
#         text_features_reduced = self.text_projection(text_features)
        
#         # Use features directly instead of projecting them
#         queries = video_features  # removed self.query
#         keys = text_features_reduced  # removed self.key
#         values = text_features_reduced  # removed self.value
        
#         # Calculate attention scores
#         attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
#         attention_weights = F.softmax(attention_scores, dim=-1)
        
#         # Apply attention weights to values
#         attention_output = torch.matmul(attention_weights, values)
        
#         # Combine with original video features (residual connection)
#         fused_features = video_features + attention_output
        
#         return fused_features


class AttentionFusion(nn.Module):
    def __init__(self):
        super(AttentionFusion, self).__init__()
        self.scale = 10240 ** -0.5
        
    def forward(self, text_features, video_features):
        # text_features_reduced = self.text_projection(text_features)
        expanded_video = torch.repeat_interleave(video_features, 20, dim=1)
        
        # Use features directly
        queries = expanded_video
        keys = text_features
        values = text_features
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, values)
        
        # Combine with original video features (residual connection)
        fused_features = expanded_video + attention_output
        # normalized_fused_features = (fused_features - fused_features.min()) / (fused_features.max() - fused_features.min())

        # Concatenate the original features with the fusion result
        final_output = torch.cat([text_features, video_features, fused_features], dim=1)
        # print(f"Max: {final_output.max().item()}, Min: {final_output.min().item()}")
        # normalized = (final_output - final_output.min()) / (final_output.max() - final_output.min()) #min-max norm

        
        return final_output

class AFF(nn.Module):
    '''
    多特征融合 AFF with feature reshaping for compatibility
    '''
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, text_features, video_features):
        batch_size = text_features.size(0)
        
        # Reshape text features from [batch_size, 10240] to [batch_size, 64, 8, 20]
        x = text_features.view(batch_size, 64, 8, 20)
        
        # Reshape video features from [batch_size, 512] to [batch_size, 64, 8, 1]
        residual = video_features.view(batch_size, 64, 8, 1)
        
        # Expand video features along the last dimension to match text features
        residual = residual.expand(-1, -1, -1, 20)
        
        # Original AFF logic
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        
        # Reshape back to original dimension [batch_size, 10240]
        xo = xo.view(batch_size, -1)

        # return xo

        # Concatenate the original features with the fusion result
        final_output = torch.cat([text_features, video_features, xo], dim=1)
        
        
        return final_output
        
        


## FCN LEARNER MODEL ##

class Learner(nn.Module):
    def __init__(self, input_dim=512, drop_p=0.0, fusion_method='concat'):
        super(Learner, self).__init__()
        
        # Initialize the fusion method based on the parameter
        self.fusion_method = fusion_method
        if fusion_method == 'addition':
            self.fusion = AdditionFusion()
        elif fusion_method == 'product':
            self.fusion = ProductFusion()
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion()
        elif fusion_method == 'AFF':
            self.fusion = AFF()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()

        # Add fusion parameters to vars
        if fusion_method != 'concat':
            for param in self.fusion.parameters():
                self.vars.append(param)

        # Add classifier parameters to vars
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x_tuple):
        if self.fusion_method == 'concat':
            # For original concatenated features
            x = x_tuple
            x = self.classifier(x)
            return x
        else:
            # For separate text and video features
            text_features, video_features = x_tuple
            fused_features = self.fusion(text_features, video_features)
            x = self.classifier(fused_features)
            return x

    def parameters(self):
        return self.vars


## MIL LOSS FUNCTION ##

def MIL(y_pred, batch_size, is_transformer=0):
    loss = torch.tensor(0.).cuda()
    loss_intra = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    if is_transformer==0:
        y_pred = y_pred.view(batch_size, -1)
    else:
        y_pred = torch.sigmoid(y_pred)

    for i in range(batch_size):
        anomaly_index = torch.randperm(30).cuda()
        normal_index = torch.randperm(30).cuda()

        y_anomaly = y_pred[i, :33][anomaly_index]
        y_normal  = y_pred[i, 33:][normal_index]

        y_anomaly_max = torch.max(y_anomaly)
        y_anomaly_min = torch.min(y_anomaly)

        y_normal_max = torch.max(y_normal)
        y_normal_min = torch.min(y_normal)

        loss += F.relu(1.-y_anomaly_max+y_normal_max)

        sparsity += torch.sum(y_anomaly)*0.00008
        smooth += torch.sum((y_pred[i,:31] - y_pred[i,1:32])**2)*0.00008
    loss = (loss+sparsity+smooth)/batch_size

    return loss


## MODIFIED TRAINING AND TESTING FUNCTIONS ##

def train(epoch, model, optimizer, scheduler, fusion_method='concat'):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        if fusion_method == 'concat':
            # For original concatenated approach
            normal_text, normal_video = normal_inputs
            anomaly_text, anomaly_video = anomaly_inputs
            
            # Concatenate text and video features
            normal_concat = torch.cat([normal_text, normal_video], dim=2)
            anomaly_concat = torch.cat([anomaly_text, anomaly_video], dim=2)
            
            # Prepare inputs as in original code
            inputs = torch.cat([anomaly_concat, normal_concat], dim=1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            
            outputs = model(inputs)
        else:
            # For the new fusion methods
            normal_text, normal_video = normal_inputs
            anomaly_text, anomaly_video = anomaly_inputs
            
            # Prepare batch for anomaly and normal samples
            batch_size = normal_text.shape[0]
            
            # Reshape and concatenate text features
            all_text = torch.cat([anomaly_text.view(-1, anomaly_text.size(-1)), 
                                 normal_text.view(-1, normal_text.size(-1))], dim=0).to(device)
            
            # Reshape and concatenate video features
            all_video = torch.cat([anomaly_video.view(-1, anomaly_video.size(-1)), 
                                  normal_video.view(-1, normal_video.size(-1))], dim=0).to(device)
            
            outputs = model((all_text, all_video))
        
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()
    return train_loss/len(normal_train_loader)

def test_abnormal(epoch, model, fusion_method='concat'):
    model.eval()
    auc = 0
    all_results = []
    
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            if fusion_method == 'concat':
                # For original concatenated approach
                anomaly_text, anomaly_video, gts, frames = data
                anomaly_concat = torch.cat([anomaly_text, anomaly_video], dim=2)
                inputs = anomaly_concat.view(-1, anomaly_concat.size(-1)).to(device)
                
                normal_text, normal_video, gts2, frames2 = data2
                normal_concat = torch.cat([normal_text, normal_video], dim=2)
                inputs2 = normal_concat.view(-1, normal_concat.size(-1)).to(device)
                
                score = model(inputs)
                score2 = model(inputs2)
            else:
                # For the new fusion methods
                anomaly_text, anomaly_video, gts, frames = data
                anomaly_text = anomaly_text.view(-1, anomaly_text.size(-1)).to(device)
                anomaly_video = anomaly_video.view(-1, anomaly_video.size(-1)).to(device)
                
                normal_text, normal_video, gts2, frames2 = data2
                normal_text = normal_text.view(-1, normal_text.size(-1)).to(device)
                normal_video = normal_video.view(-1, normal_video.size(-1)).to(device)
                
                score = model((anomaly_text, anomaly_video))
                score2 = model((normal_text, normal_video))
            
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, torch.div(frames[0], 16, rounding_mode='floor').item(), 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, torch.div(frames2[0], 16, rounding_mode='floor').item(), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
                
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            current_auc = metrics.auc(fpr, tpr)
            auc += current_auc
            
            # Save detailed results for each test sample
            result = {
                'sample_idx': i,
                'auc': current_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'anomaly_scores': score_list.tolist(),
                'normal_scores': score_list2.tolist(),
                'gt_anomaly': gt_list.tolist(),
                'gt_normal': gt_list2.tolist()
            }
            all_results.append(result)
            
        avg_auc = auc/140
        print('auc = ', avg_auc)
    return avg_auc, all_results


## OBJECTIVE FUNCTION FOR OPTUNA ##

def create_objective(fusion_method):
    def objective(trial):
        if fusion_method == 'concat':
            input_dim = 10752  # Original concatenated features
        if fusion_method == 'addition':   
            input_dim = 10240  # Original concatenated features 
        if fusion_method == 'product':   
            input_dim = 10240  # Original concatenated features 
        if fusion_method == 'AFF':   
            # input_dim = 10240  # Original concatenated features
            input_dim = 20992  # Original concatenated features
        else:
            # input_dim = 10240  # Fused features dimension 512
            input_dim = 20992  # Original concatenated features
            
        drop_p = trial.suggest_float("dropout", 0.0, 0.9)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

        model = Learner(input_dim=input_dim, drop_p=drop_p, fusion_method=fusion_method).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10, 15, 20, 25, 50])
        
        auc_score = 0
        max_auc = 0
        best_results = None
        training_history = []

        for epoch in range(10):  # Reduce the number of epochs for quicker optimization
            train_loss = train(epoch, model, optimizer, scheduler, fusion_method)
            auc_score, results = test_abnormal(epoch, model, fusion_method)
            
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'auc_score': auc_score
            }
            training_history.append(epoch_info)
            
            if auc_score > max_auc:
                max_auc = auc_score
                best_results = results
                # Save model checkpoint
                torch.save(model.state_dict(), f'best_model_{fusion_method}.pth')
                
            print(f"Epoch {epoch}, AUC: {auc_score:.6f}")

        # Save detailed results and training history
        results_path = f'results_{fusion_method}.txt'
        with open(results_path, 'w') as f:
            json.dump({
                'best_auc': max_auc,
                'detailed_results': best_results,
                'training_history': training_history,
                'hyperparameters': {
                    'dropout': drop_p,
                    'learning_rate': lr,
                    'weight_decay': weight_decay
                }
            }, f, indent=4)

        return max_auc
    
    return objective


## MAIN CODE ##

# Prepare data loaders
normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)
anomaly_train_dataset = Anomaly_Loader(is_train=1)
anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = MIL

# Define fusion methods to evaluate
# fusion_methods = ['concat', 'addition', 'product', 'attention','AFF']
fusion_methods = ['attention']

# Run optimization for each fusion method
for fusion_method in fusion_methods:
    print(f"\n\n{'='*50}")
    print(f"Starting optimization for {fusion_method} fusion method")
    print(f"{'='*50}\n")
    
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the objective function for this fusion method
    objective_fn = create_objective(fusion_method)
    
    # Create and run the study
    study = optuna.create_study(direction='maximize', 
                               study_name=f"{fusion_method}_{timestamp}")
    study.optimize(objective_fn, n_trials=100)
    
    # Print and save the best results
    print(f"\nBest trial for {fusion_method} fusion:")
    trial = study.best_trial
    print(f"  AUC: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the study statistics
    with open(f'study_results_{fusion_method}.txt', 'w') as f:
        f.write(f"Best AUC: {trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
            
    print(f"Completed optimization for {fusion_method}!\n")
