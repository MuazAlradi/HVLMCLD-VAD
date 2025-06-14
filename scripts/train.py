import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import argparse
import json
import logging
from sklearn import metrics
from sklearn.decomposition import PCA
import optuna
from scipy.optimize import minimize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## UNIVERSAL DATALOADERS ##

class Normal_Loader(Dataset):
    def __init__(self, is_train=1, data_path='./data', dataset='ucf_crime'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.dataset = dataset
        
        if self.is_train == 1:
            data_list = os.path.join(data_path, dataset, 'labels', f'{self._get_prefix()}_train_normal.txt')
        else:
            data_list = os.path.join(data_path, dataset, 'labels', f'{self._get_prefix()}_test_normalv2.txt')
            
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()
            
        if self.is_train == 0:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def _get_prefix(self):
        if self.dataset == 'ucf_crime':
            return 'UCF'
        elif self.dataset == 'shanghai_tech':
            return 'SHT'
        elif self.dataset == 'xd_violence':
            return 'XDV'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            feature_path = os.path.join(self.data_path, 'output', 'visual_features', 
                                      self.dataset, self.data_list[idx].strip() + '.npy')
            rgb_npy = np.load(feature_path)
            return rgb_npy
        else:
            parts = self.data_list[idx].split(' ')
            name, frames, gts = parts[0], int(parts[1]), int(parts[2].strip())
            feature_path = os.path.join(self.data_path, 'output', 'visual_features', 
                                      self.dataset, name + '.npy')
            rgb_npy = np.load(feature_path)
            return rgb_npy, gts, frames

class Anomaly_Loader(Dataset):
    def __init__(self, is_train=1, data_path='./data', dataset='ucf_crime'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.data_path = data_path
        self.dataset = dataset
        
        if self.is_train == 1:
            data_list = os.path.join(data_path, dataset, 'labels', f'{self._get_prefix()}_train_anomaly.txt')
        else:
            data_list = os.path.join(data_path, dataset, 'labels', f'{self._get_prefix()}_test_anomalyv2.txt')
            
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def _get_prefix(self):
        if self.dataset == 'ucf_crime':
            return 'UCF'
        elif self.dataset == 'shanghai_tech':
            return 'SHT'
        elif self.dataset == 'xd_violence':
            return 'XDV'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            feature_path = os.path.join(self.data_path, 'output', 'visual_features', 
                                      self.dataset, self.data_list[idx].strip() + '.npy')
            rgb_npy = np.load(feature_path)
            return rgb_npy
        else:
            parts = self.data_list[idx].split('|')
            name, frames = parts[0], int(parts[1])
            gts = [int(i) for i in parts[2][1:-2].split(',')]
            feature_path = os.path.join(self.data_path, 'output', 'visual_features', 
                                      self.dataset, name + '.npy')
            rgb_npy = np.load(feature_path)
            return rgb_npy, gts, frames

## GODS ALGORITHM (OCC) ##

class GODS:
    def __init__(self, d, k):
        self.d = d
        self.k = k
        self.W1 = np.random.randn(d, k)
        self.W2 = np.random.randn(d, k)
        self.b1 = np.random.randn(k)
        self.b2 = np.random.randn(k)

    def fit(self, X):
        def objective(params):
            W1 = params[:self.d * self.k].reshape(self.d, self.k)
            W2 = params[self.d * self.k:self.d * self.k * 2].reshape(self.d, self.k)
            b1 = params[self.d * self.k * 2:self.d * self.k * 2 + self.k]
            b2 = params[self.d * self.k * 2 + self.k:]

            term1 = np.sum(np.linalg.norm(W1.T @ X.T + b1[:, None], axis=0)**2)
            term2 = np.sum(np.linalg.norm(W2.T @ X.T + b2[:, None], axis=0)**2)
            term3 = np.sum((b1 - b2)**2)
            term4 = np.sum(np.maximum(0, 1 - np.min(W1.T @ X.T + b1[:, None], axis=0))**2)
            term5 = np.sum(np.maximum(0, 1 + np.max(W2.T @ X.T + b2[:, None], axis=0))**2)

            return term1 + term2 + term3 + term4 + term5

        params_init = np.concatenate([self.W1.ravel(), self.W2.ravel(), self.b1, self.b2])
        result = minimize(objective, params_init, method='L-BFGS-B')
        params_opt = result.x

        self.W1 = params_opt[:self.d * self.k].reshape(self.d, self.k)
        self.W2 = params_opt[self.d * self.k:self.d * self.k * 2].reshape(self.d, self.k)
        self.b1 = params_opt[self.d * self.k * 2:self.d * self.k * 2 + self.k]
        self.b2 = params_opt[self.d * self.k * 2 + self.k:]

    def decision_function(self, X):
        scores = np.min(self.W1.T @ X.T + self.b1[:, None], axis=0) + np.max(self.W2.T @ X.T + self.b2[:, None], axis=0)
        return scores

## UNSUPERVISED MODELS ##

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

## WEAKLY SUPERVISED MODEL ##

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
        self.drop_p = drop_p
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

## TRAINING CLASSES ##

class OCCTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.setup_data_loaders()

    def setup_data_loaders(self):
        normal_train_dataset = Normal_Loader(is_train=1, data_path=self.config['data_path'], 
                                           dataset=self.config['dataset'])
        normal_test_dataset = Normal_Loader(is_train=0, data_path=self.config['data_path'], 
                                          dataset=self.config['dataset'])
        anomaly_test_dataset = Anomaly_Loader(is_train=0, data_path=self.config['data_path'], 
                                            dataset=self.config['dataset'])

        self.normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
        self.normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
        self.anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    def train_gods(self, gods, train_loader, pca):
        train_data = []
        for batch_idx, normal_inputs in enumerate(train_loader):
            train_data.append(normal_inputs.numpy())
        train_data = np.vstack(train_data)
        train_data = train_data.reshape(train_data.shape[0], -1)
        train_data = pca.transform(train_data)
        gods.fit(train_data)

    def test_gods(self, gods, normal_test_loader, anomaly_test_loader, pca):
        anomaly_scores = []
        normal_scores = []

        for data, gts, frames in anomaly_test_loader:
            data = data.numpy().reshape(data.shape[0], -1)
            data = pca.transform(data)
            scores = gods.decision_function(data)
            anomaly_scores.extend(scores)

        for data, gts, frames in normal_test_loader:
            data = data.numpy().reshape(data.shape[0], -1)
            data = pca.transform(data)
            scores = gods.decision_function(data)
            normal_scores.extend(scores)

        flat_true_labels = [1] * len(anomaly_scores) + [0] * len(normal_scores)
        scores = np.array(anomaly_scores + normal_scores)

        fpr, tpr, thresholds = metrics.roc_curve(flat_true_labels, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    def objective(self, trial):
        k = trial.suggest_int("k", 1, 10)
        n_components = trial.suggest_int("n_components", 1, 30)

        pca = PCA(n_components=n_components)
        sample_data = next(iter(self.normal_train_loader)).numpy()
        sample_data = sample_data.reshape(sample_data.shape[0], -1)
        pca.fit(sample_data)

        d = n_components
        gods = GODS(d=d, k=k)
        
        self.train_gods(gods, self.normal_train_loader, pca)
        max_auc = self.test_gods(gods, self.normal_test_loader, self.anomaly_test_loader, pca)

        return max_auc

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.config['n_trials'])
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  AUC: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        return study.best_trial

class UnsupervisedTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.setup_data_loaders()

    def setup_data_loaders(self):
        normal_train_dataset = Normal_Loader(is_train=1, data_path=self.config['data_path'], 
                                           dataset=self.config['dataset'])
        normal_test_dataset = Normal_Loader(is_train=0, data_path=self.config['data_path'], 
                                          dataset=self.config['dataset'])
        anomaly_train_dataset = Anomaly_Loader(is_train=1, data_path=self.config['data_path'], 
                                             dataset=self.config['dataset'])
        anomaly_test_dataset = Anomaly_Loader(is_train=0, data_path=self.config['data_path'], 
                                            dataset=self.config['dataset'])

        self.normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
        self.normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
        self.anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
        self.anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    def generate_pseudo_labels(self, generator, data, threshold):
        generator.eval()
        with torch.no_grad():
            reconstructed = generator(data)
            loss = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
        pseudo_labels = (loss >= threshold).float()
        return pseudo_labels, loss

    def compute_threshold(self, loss, percentage):
        sorted_loss, _ = torch.sort(loss)
        threshold_index = int((1 - percentage) * len(sorted_loss))
        return sorted_loss[threshold_index].item()

    def get_loss_values(self, generator, data_loader):
        generator.eval()
        all_losses = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                reconstructed = generator(data)
                loss = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
                all_losses.append(loss)
        all_losses = torch.cat(all_losses)
        return all_losses.view(-1)

    def train_generator(self, generator, discriminator, normal_loader, anomaly_loader, optimizer_G, threshold):
        generator.train()
        total_loss = 0
        for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_loader, anomaly_loader)):
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
            inputs = inputs.view(-1, inputs.size(-1)).to(self.device)
            reconstructed = generator(inputs)
            
            pseudo_labels, _ = self.generate_pseudo_labels(generator, inputs, threshold)
            
            targets = inputs.clone()
            targets[pseudo_labels == 1] = torch.ones_like(targets[pseudo_labels == 1])
            
            loss = F.mse_loss(reconstructed, targets).mean()
            
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            total_loss += loss.item()

    def train_discriminator(self, discriminator, generator, normal_loader, anomaly_loader, optimizer_D, threshold):
        discriminator.train()
        total_loss = 0
        criterion = nn.BCELoss()
        for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_loader, anomaly_loader)):
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
            inputs = inputs.view(-1, inputs.size(-1)).to(self.device)
            
            pseudo_labels, _ = self.generate_pseudo_labels(generator, inputs, threshold)
            
            outputs = discriminator(inputs).view(-1)
            loss = criterion(outputs, pseudo_labels)
            
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()
            total_loss += loss.item()

    def test_model(self, discriminator, anomaly_test_loader, normal_test_loader):
        discriminator.eval()
        auc = 0
        with torch.no_grad():
            for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
                inputs, gts, frames = data
                inputs = inputs.view(-1, inputs.size(-1)).to(self.device)
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
                inputs2 = inputs2.view(-1, inputs2.size(-1)).to(self.device)
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
        return auc / 140

    def objective(self, trial):
        input_dim = self.config['input_dim']
        hidden_dim = trial.suggest_int("hidden_dim", 128, 1024)
        lr_G = trial.suggest_loguniform("lr_G", 1e-5, 1e-1)
        lr_D = trial.suggest_loguniform("lr_D", 1e-5, 1e-1)
        weight_decay_G = trial.suggest_loguniform("weight_decay_G", 1e-5, 1e-2)
        weight_decay_D = trial.suggest_loguniform("weight_decay_D", 1e-5, 1e-2)

        generator = Generator(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        discriminator = Discriminator(input_dim=input_dim).to(self.device)

        optimizer_G = torch.optim.Adagrad(generator.parameters(), lr=lr_G, weight_decay=weight_decay_G)
        optimizer_D = torch.optim.Adagrad(discriminator.parameters(), lr=lr_D, weight_decay=weight_decay_D)

        max_auc = 0
        for epoch in range(self.config['epochs']):
            all_losses = self.get_loss_values(generator, self.normal_train_loader)
            threshold = self.compute_threshold(all_losses, 0.1)

            self.train_generator(generator, discriminator, self.normal_train_loader, 
                               self.anomaly_train_loader, optimizer_G, threshold)
            self.train_discriminator(discriminator, generator, self.normal_train_loader, 
                                   self.anomaly_train_loader, optimizer_D, threshold)

            auc_score = self.test_model(discriminator, self.anomaly_test_loader, self.normal_test_loader)
            if auc_score > max_auc:
                max_auc = auc_score
        
        return max_auc

    def train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.config['n_trials'])
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"AUC: {trial.value}")
        logger.info("Best hyperparameters: ", trial.params)
        
        return study.best_trial

class WeaklySupervisedTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.setup_data_loaders()

    def setup_data_loaders(self):
        normal_train_dataset = Normal_Loader(is_train=1, data_path=self.config['data_path'], 
                                           dataset=self.config['dataset'])
        normal_test_dataset = Normal_Loader(is_train=0, data_path=self.config['data_path'], 
                                          dataset=self.config['dataset'])
        anomaly_train_dataset = Anomaly_Loader(is_train=1, data_path=self.config['data_path'], 
                                             dataset=self.config['dataset'])
        anomaly_test_dataset = Anomaly_Loader(is_train=0, data_path=self.config['data_path'], 
                                            dataset=self.config['dataset'])

        self.normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
        self.normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
        self.anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
        self.anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    def MIL_loss(self, y_pred, batch_size, is_transformer=0):
        loss = torch.tensor(0.).to(self.device)
        sparsity = torch.tensor(0.).to(self.device)
        smooth = torch.tensor(0.).to(self.device)
        
        if is_transformer == 0:
            y_pred = y_pred.view(batch_size, -1)
        else:
            y_pred = torch.sigmoid(y_pred)

        for i in range(batch_size):
            anomaly_index = torch.randperm(30).to(self.device)
            normal_index = torch.randperm(30).to(self.device)

            y_anomaly = y_pred[i, :33][anomaly_index]
            y_normal = y_pred[i, 33:][normal_index]

            y_anomaly_max = torch.max(y_anomaly)
            y_anomaly_min = torch.min(y_anomaly)
            y_normal_max = torch.max(y_normal)
            y_normal_min = torch.min(y_normal)

            loss += F.relu(1. - y_anomaly_max + y_normal_max)
            sparsity += torch.sum(y_anomaly) * 0.00008
            smooth += torch.sum((y_pred[i, :31] - y_pred[i, 1:32])**2) * 0.00008
        
        loss = (loss + sparsity + smooth) / batch_size
        return loss

    def train_epoch(self, model, optimizer, scheduler):
        model.train()
        train_loss = 0
        for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(self.normal_train_loader, self.anomaly_train_loader)):
            inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(self.device)
            outputs = model(inputs)
            loss = self.MIL_loss(outputs, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

    def test_model(self, model):
        model.eval()
        auc = 0
        with torch.no_grad():
            for i, (data, data2) in enumerate(zip(self.anomaly_test_loader, self.normal_test_loader)):
                inputs, gts, frames = data
                inputs = inputs.view(-1, inputs.size(-1)).to(self.device)
                score = model(inputs)
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
                inputs2 = inputs2.view(-1, inputs2.size(-1)).to(self.device)
                score2 = model(inputs2)
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
        return auc / 140

    def objective(self, trial):
        input_dim = self.config['input_dim']
        drop_p = trial.suggest_float("dropout", 0.0, 0.9)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

        model = Learner(input_dim=input_dim, drop_p=drop_p).to(self.device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10, 15, 20, 25, 50])

        max_auc = 0
        for epoch in range(self.config['epochs']):
            self.train_epoch(model, optimizer, scheduler)
            auc_score = self.test_model(model)
            if auc_score > max_auc:
                max_auc = auc_score

        return max_auc

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.config['n_trials'])
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  AUC: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
        
        return study.best_trial

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def save_model(model, save_path):
    """Save the trained model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), save_path)
    else:
        # For non-PyTorch models like GODS
        torch.save(model, save_path)
    logger.info(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Universal Anomaly Detection Training')
    parser.add_argument('--config', type=str, required=True, 
                       choices=['weakly_sup', 'unsup', 'occ'],
                       help='Training configuration to use')
    parser.add_argument('--save_dir', type=str, default='./output',
                       help='Directory to save outputs')
    parser.add_argument('--save_model', type=bool, default=False,
                       help='Whether to save the trained model')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for the saved model file')
    parser.add_argument('--data_path', type=str, default='./data',
                       help='Path to the data directory')
    parser.add_argument('--dataset', type=str, default='ucf_crime',
                       choices=['ucf_crime', 'shanghai_tech', 'xd_violence'],
                       help='Dataset to use for training')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = f'configs/{args.config}.json'
    config = load_config(config_path)
    
    # Override config with command line arguments
    config['data_path'] = args.data_path
    config['dataset'] = args.dataset
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize trainer based on config
    if args.config == 'occ':
        trainer = OCCTrainer(config, device)
    elif args.config == 'unsup':
        trainer = UnsupervisedTrainer(config, device)
    elif args.config == 'weakly_sup':
        trainer = WeaklySupervisedTrainer(config, device)
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Train the model
    logger.info(f"Starting {args.config} training...")
    best_trial = trainer.train()
    
    # Save model if requested
    if args.save_model and args.model_name:
        checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
        model_path = os.path.join(checkpoint_dir, args.model_name)
        
        # Note: For full model saving, you'd need to recreate the best model
        # This is a simplified version - you might want to modify trainer classes
        # to return the best model for saving
        logger.info(f"Model saving requested but requires implementation of best model recreation")
        logger.info(f"Best parameters: {best_trial.params}")
        logger.info(f"Best AUC: {best_trial.value}")

if __name__ == "__main__":
    main()
