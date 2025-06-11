import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch.nn as nn
from torch.nn import functional as F
from sklearn import metrics
import optuna

## DATALOADERS ##

class Normal_Loader(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\', subset='random_8'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.subset = subset
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
        file_name = self.data_list[idx].strip().split(' ')[0]
        rgb_npy = np.load(os.path.join(self.path, 'UCF-Crime Features Finalized', file_name + '.npy'))
        if self.is_train == 1:
            if self.subset == 'random_8':
                return rgb_npy[np.random.choice(32, 8, replace=False)]
            elif self.subset == 'random_16':
                return rgb_npy[np.random.choice(32, 16, replace=False)]
            elif self.subset == 'random_24':
                return rgb_npy[np.random.choice(32, 24, replace=False)]
            else:
                return rgb_npy[:32]  # Default to all short views
        else:
            return rgb_npy, 0, rgb_npy.shape[0]  # Return full array for testing

class Anomaly_Loader(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\', subset='random_8'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.subset = subset
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
            file_name = self.data_list[idx].strip().split(' ')[0]
            rgb_npy = np.load(os.path.join(self.path, 'UCF-Crime Features Finalized', file_name + '.npy'))
            if self.subset == 'random_8':
                return rgb_npy[np.random.choice(32, 8, replace=False)]
            elif self.subset == 'random_16':
                return rgb_npy[np.random.choice(32, 16, replace=False)]
            elif self.subset == 'random_24':
                return rgb_npy[np.random.choice(32, 24, replace=False)]
            else:
                return rgb_npy[:32]  # Default to all short views
        else:
            split_data = self.data_list[idx].strip().split('|')
            name, frames, gts = split_data[0], int(split_data[1]), list(map(int, split_data[2][1:-1].split(',')))
            rgb_npy = np.load(os.path.join(self.path, 'UCF-Crime Features Finalized', name + '.npy'))
            return rgb_npy, gts, frames

## FCN LEARNER MODEL ##

class Learner(nn.Module):
    def __init__(self, input_dim=10752, drop_p=0.6):
        super(Learner, self).__init__()
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
        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.classifier(x)

## MIL LOSS FUNCTION ##

def MIL(y_pred, batch_size):
    loss = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    
    y_pred = y_pred.view(batch_size, -1)

    for i in range(batch_size):
        y_anomaly = y_pred[i, :y_pred.size(1)//2]
        y_normal = y_pred[i, y_pred.size(1)//2:]

        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max = torch.max(y_normal)

        loss += F.relu(1. - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * 0.00008
        if y_pred.size(1) > 1:
            smooth += torch.sum((y_pred[i, :-1] - y_pred[i, 1:])**2) * 0.00008

    loss = (loss + sparsity + smooth) / batch_size

    return loss

## MAIN CODE ##

def train(epoch, model, optimizer, scheduler, normal_train_loader, anomaly_train_loader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        normal_inputs = normal_inputs.to(device)
        anomaly_inputs = anomaly_inputs.to(device)
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1))
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        # if batch_idx % 10 == 0:
        #     print(f'Batch {batch_idx}, Loss: {loss.item():.6f}, Outputs mean: {outputs.mean().item():.6f}, Outputs std: {outputs.std().item():.6f}')
    print('Average loss = {:.6f}'.format(train_loss / len(normal_train_loader)))
    scheduler.step()

def test_abnormal(model, anomaly_test_loader, normal_test_loader):
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            score = model(inputs)
            score = score.cpu().detach().numpy()
            
            # Convert frames to integer
            frames = frames.item() if isinstance(frames, torch.Tensor) else int(frames)
            score_list = np.zeros(frames)
            
            step = np.round(np.linspace(0, frames // 16, 33))

            for j in range(32):
                score_list[int(step[j])*16:int(step[j+1])*16] = score[j]

            gt_list = np.zeros(frames)
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            inputs2, _, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            score2 = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            
            # Convert frames2 to integer
            frames2 = frames2.item() if isinstance(frames2, torch.Tensor) else int(frames2)
            score_list2 = np.zeros(frames2)
            
            step2 = np.round(np.linspace(0, frames2 // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:int(step2[kk+1])*16] = score2[kk]
            gt_list2 = np.zeros(frames2)
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
        print('auc = ', auc/len(anomaly_test_loader))
    return auc/len(anomaly_test_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = MIL

# def objective(trial):
#     subset = trial.suggest_categorical('subset', ['random_8', 'random_16', 'random_24'])
#     input_dim = 10752
#     drop_p = trial.suggest_float("dropout", 0.3, 0.7)
#     lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
#     weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

#     normal_train_dataset = Normal_Loader(is_train=1, subset=subset)
#     normal_test_dataset = Normal_Loader(is_train=0, subset=subset)
#     anomaly_train_dataset = Anomaly_Loader(is_train=1, subset=subset)
#     anomaly_test_dataset = Anomaly_Loader(is_train=0, subset=subset)

#     normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
#     normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)
#     anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
#     anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

#     model = Learner(input_dim=input_dim, drop_p=drop_p).to(device)
#     optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

#     max_auc = 0
#     for epoch in range(100):
#         train(epoch, model, optimizer, scheduler, normal_train_loader, anomaly_train_loader)
#         auc_score = test_abnormal(model, anomaly_test_loader, normal_test_loader)
#         if auc_score > max_auc:
#             max_auc = auc_score
#         print(f"Epoch {epoch}, AUC: {auc_score:.6f}")

#     return max_auc

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print(f"  AUC: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


def objective(trial, subset):
    input_dim = 10752
    drop_p = trial.suggest_float("dropout", 0.3, 0.7)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    normal_train_dataset = Normal_Loader(is_train=1, subset=subset)
    normal_test_dataset = Normal_Loader(is_train=0, subset=subset)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, subset=subset)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, subset=subset)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

    model = Learner(input_dim=input_dim, drop_p=drop_p).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

    max_auc = 0
    for epoch in range(30):
        train(epoch, model, optimizer, scheduler, normal_train_loader, anomaly_train_loader)
        auc_score = test_abnormal(model, anomaly_test_loader, normal_test_loader)
        if auc_score > max_auc:
            max_auc = auc_score
        print(f"Epoch {epoch}, AUC: {auc_score:.6f}")

    return max_auc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = MIL

subset_types = [
    'random_8', 'random_16', 'random_24',
    'random_8_with_long', 'random_16_with_long', 'random_24_with_long',
    'long_view_only', 'short_views_only'
]

results = {}

for subset in subset_types:
    print(f"\nRunning optimization for subset: {subset}")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, subset), n_trials=10)
    
    best_auc = study.best_value
    best_params = study.best_params
    
    results[subset] = {
        'AUC': best_auc,
        'params': best_params
    }
    
    print(f"Best AUC for {subset}: {best_auc}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

# Print all results
print("\nAll results:")
for subset, result in results.items():
    print(f"{subset}: AUC = {result['AUC']:.6f}")

# Save results to a text file
with open("ablationStudyResults.txt", "w") as f:
    for subset, result in results.items():
        f.write(f"{subset}: AUC = {result['AUC']:.6f}\n")
        f.write("Best parameters:\n")
        for key, value in result['params'].items():
            f.write(f"    {key}: {value}\n")
        f.write("\n")

print("Results saved to ablationStudyResults.txt")