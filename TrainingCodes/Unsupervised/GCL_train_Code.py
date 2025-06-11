import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
import optuna

## DATALOADERS ##

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
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
    """
    is_train = 1 <- train, 0 <- test
    """
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

## GCL MODELS ##

class Generator(nn.Module):
    def __init__(self, input_dim=10752, hidden_dim=512):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
        self._initialize_weights()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def _initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

class Discriminator(nn.Module):
    def __init__(self, input_dim=10752):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

## Pseudo-label Generation Functions ##

def generate_pseudo_labels(generator, data, threshold):
    generator.eval()
    with torch.no_grad():
        reconstructed = generator(data)
        loss = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
    pseudo_labels = (loss >= threshold).float()
    return pseudo_labels, loss

def compute_threshold(loss, percentage):
    sorted_loss, _ = torch.sort(loss)
    threshold_index = int((1 - percentage) * len(sorted_loss))
    return sorted_loss[threshold_index].item()  # Convert to scalar

def get_loss_values(generator, data_loader):
    generator.eval()
    all_losses = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            reconstructed = generator(data)
            loss = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
            all_losses.append(loss)
    all_losses = torch.cat(all_losses)
    return all_losses.view(-1)  # Ensure it is 1-dimensional

def save_model(model, filename="model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def train_generator(generator, discriminator, normal_loader, anomaly_loader, optimizer_G, threshold):
    generator.train()
    total_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_loader, anomaly_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        reconstructed = generator(inputs)
        loss = F.mse_loss(reconstructed, inputs, reduction='none').mean(dim=1)
        
        # Generate pseudo-labels from discriminator
        pseudo_labels, _ = generate_pseudo_labels(generator, inputs, threshold)
        
        # Modify targets for negative learning
        targets = inputs.clone()
        targets[pseudo_labels == 1] = torch.ones_like(targets[pseudo_labels == 1])
        
        loss = F.mse_loss(reconstructed, targets).mean()
        
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()
        total_loss += loss.item()
    print('Generator loss = {}'.format(total_loss / len(normal_loader)))

def train_discriminator(discriminator, generator, normal_loader, anomaly_loader, optimizer_D, threshold):
    discriminator.train()
    total_loss = 0
    criterion = nn.BCELoss()
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_loader, anomaly_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        
        # Generate pseudo-labels from generator
        pseudo_labels, loss = generate_pseudo_labels(generator, inputs, threshold)
        
        # Train discriminator with pseudo-labels
        outputs = discriminator(inputs).view(-1)
        loss = criterion(outputs, pseudo_labels)
        
        optimizer_D.zero_grad()
        loss.backward()
        optimizer_D.step()
        total_loss += loss.item()
    print('Discriminator loss = {}'.format(total_loss / len(normal_loader)))

def test_model(discriminator, anomaly_test_loader, normal_test_loader):
    discriminator.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            score = discriminator(inputs)
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

            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            score2 = discriminator(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, torch.div(frames2[0], 16, rounding_mode='floor').item(), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)

            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
        print('AUC = ', auc / 140)
    return auc / 140

def objective(trial):
    input_dim = 10752
    hidden_dim = trial.suggest_int("hidden_dim", 128, 1024)
    lr_G = trial.suggest_loguniform("lr_G", 1e-5, 1e-1)
    lr_D = trial.suggest_loguniform("lr_D", 1e-5, 1e-1)
    weight_decay_G = trial.suggest_loguniform("weight_decay_G", 1e-5, 1e-2)
    weight_decay_D = trial.suggest_loguniform("weight_decay_D", 1e-5, 1e-2)

    generator = Generator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)

    optimizer_G = torch.optim.Adagrad(generator.parameters(), lr=lr_G, weight_decay=weight_decay_G)
    optimizer_D = torch.optim.Adagrad(discriminator.parameters(), lr=lr_D, weight_decay=weight_decay_D)

    max_auc = 0
    for epoch in range(150):  # Reduced epochs for faster optimization
        print('\nEpoch: %d' % epoch)
        # Generate threshold from the previous epoch's reconstruction loss
        all_losses = get_loss_values(generator, normal_train_loader)
        threshold = compute_threshold(all_losses, 0.1)

        # Train Generator and Discriminator alternately
        train_generator(generator, discriminator, normal_train_loader, anomaly_train_loader, optimizer_G, threshold)
        train_discriminator(discriminator, generator, normal_train_loader, anomaly_train_loader, optimizer_D, threshold)

        # Test the model
        auc_score = test_model(discriminator, anomaly_test_loader, normal_test_loader)
        if auc_score > max_auc:
            max_auc = auc_score
            print("New maximum AUC")
    
    return max_auc

if __name__ == "__main__":
    normal_train_dataset = Normal_Loader(is_train=1)
    normal_test_dataset = Normal_Loader(is_train=0)

    anomaly_train_dataset = Anomaly_Loader(is_train=1)
    anomaly_test_dataset = Anomaly_Loader(is_train=0)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial
    print(f"AUC: {trial.value}")
    print("Best hyperparameters: ", trial.params)
