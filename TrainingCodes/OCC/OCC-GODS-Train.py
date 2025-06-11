import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.decomposition import PCA
import optuna
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    def __init__(self, is_train=0, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 0:
            data_list = os.path.join(path, 'test_anomalyv2.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
        gts = [int(i) for i in gts]
        rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
        return rgb_npy, gts, frames

## GODS ALGORITHM IMPLEMENTATION ##

class GODS:
    def __init__(self, d, k):
        self.d = d  # Dimensionality of input data
        self.k = k  # Number of subspaces
        self.W1 = np.random.randn(d, k)
        self.W2 = np.random.randn(d, k)
        self.b1 = np.random.randn(k)
        self.b2 = np.random.randn(k)
        # logger.debug(f'Initialized GODS with W1 shape {self.W1.shape}, W2 shape {self.W2.shape}, b1 shape {self.b1.shape}, b2 shape {self.b2.shape}')

    def fit(self, X):
        def objective(params):
            W1 = params[:self.d * self.k].reshape(self.d, self.k)
            W2 = params[self.d * self.k:self.d * self.k * 2].reshape(self.d, self.k)
            b1 = params[self.d * self.k * 2:self.d * self.k * 2 + self.k]
            b2 = params[self.d * self.k * 2 + self.k:]

            # logger.debug(f'W1 shape: {W1.shape}, W2 shape: {W2.shape}, b1 shape: {b1.shape}, b2 shape: {b2.shape}, X shape: {X.shape}')

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

        # logger.debug(f'Optimized GODS with W1 shape {self.W1.shape}, W2 shape {self.W2.shape}, b1 shape {self.b1.shape}, b2 shape {self.b2.shape}')

    def decision_function(self, X):
        scores = np.min(self.W1.T @ X.T + self.b1[:, None], axis=0) + np.max(self.W2.T @ X.T + self.b2[:, None], axis=0)
        # logger.debug(f'Decision function input X shape: {X.shape}, scores shape: {scores.shape}')
        return scores

## TRAINING AND TESTING FUNCTIONS ##

def train_gods(gods, train_loader, pca):
    train_data = []
    for batch_idx, normal_inputs in enumerate(train_loader):
        train_data.append(normal_inputs.numpy())
    train_data = np.vstack(train_data)
    train_data = train_data.reshape(train_data.shape[0], -1)  # Flatten the data if needed
    train_data = pca.transform(train_data)
    # logger.debug(f'Training data shape after PCA: {train_data.shape}')
    gods.fit(train_data)

def test_gods(gods, normal_test_loader, anomaly_test_loader, pca):
    anomaly_scores = []
    normal_scores = []

    for data, gts, frames in anomaly_test_loader:
        data = data.numpy().reshape(data.shape[0], -1)  # Flatten the data
        data = pca.transform(data)
        scores = gods.decision_function(data)
        anomaly_scores.extend(scores)

    for data, gts, frames in normal_test_loader:
        data = data.numpy().reshape(data.shape[0], -1)  # Flatten the data
        data = pca.transform(data)
        scores = gods.decision_function(data)
        normal_scores.extend(scores)

    # logger.debug(f'Anomaly scores length: {len(anomaly_scores)}, Normal scores length: {len(normal_scores)}')

    flat_true_labels = [1] * len(anomaly_scores) + [0] * len(normal_scores)
    scores = np.array(anomaly_scores + normal_scores)

    fpr, tpr, thresholds = metrics.roc_curve(flat_true_labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    logger.info(f'AUC = {auc}')
    return auc

## MAIN CODE ##

normal_train_dataset = Normal_Loader(is_train=1)
normal_test_dataset = Normal_Loader(is_train=0)

anomaly_test_dataset = Anomaly_Loader(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

def objective(trial):
    k = trial.suggest_int("k", 1, 10)
    n_components = trial.suggest_int("n_components", 1, 30)  # Number of PCA components
    logger.info(f'Trial {trial.number}: k={k}, n_components={n_components}')

    # Fit PCA
    pca = PCA(n_components=n_components)
    sample_data = next(iter(normal_train_loader)).numpy()
    sample_data = sample_data.reshape(sample_data.shape[0], -1)  # Flatten the data
    pca.fit(sample_data)

    d = n_components  # Dimensionality after PCA
    gods = GODS(d=d, k=k)
    
    train_gods(gods, normal_train_loader, pca)
    max_auc = test_gods(gods, normal_test_loader, anomaly_test_loader, pca)

    return max_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

print("Best trial:")
trial = study.best_trial
print(f"  AUC: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
