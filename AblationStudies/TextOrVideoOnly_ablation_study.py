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


## DATALOADERS ##

class Normal_Loader_VidOnly(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI'):
        super(Normal_Loader_VidOnly, self).__init__()
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
            # print("Shape before: ", rgb_npy.shape)
            # rgb_npy = rgb_npy[: , -512:] # Video Only
            # rgb_npy = rgb_npy[: , :-512] # text Only
            rgb_npy = rgb_npy # text Only (has size 10240, no video features)
            # print("Shape after: ", rgb_npy.shape)
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path+'UCF-Crime Features Finalized', name + '.npy'))
            # print("Shape before: ", rgb_npy.shape)
            # rgb_npy = rgb_npy[: , -512:] # Video Only
            # rgb_npy = rgb_npy[: , :-512] # text Only
            rgb_npy = rgb_npy # text Only (has size 10240, no video features)
            return rgb_npy, gts, frames

class Anomaly_Loader_VidOnly(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI'):
        super(Anomaly_Loader_VidOnly, self).__init__()
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
            rgb_npy = np.load(os.path.join(self.path+'AllFeaturesBLIP2Only', self.data_list[idx][:-1]+'.npy'))
            # rgb_npy = rgb_npy[: , -512:] # Video Only
            # rgb_npy = rgb_npy[: , :-512] # text Only
            rgb_npy = rgb_npy # text Only (has size 10240, no video features)
            return rgb_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path+'AllFeaturesBLIP2Only', name + '.npy'))
            # rgb_npy = rgb_npy[: , -512:] # Video Only
            # rgb_npy = rgb_npy[: , :-512] # text Only
            rgb_npy = rgb_npy # text Only (has size 10240, no video features)
            return rgb_npy, gts, frames







# class Normal_Loader_TextOnly(Dataset):
#     def __init__(self, is_train=1, path='E:\\Llama 3\\'):
#         super(Normal_Loader_TextOnly, self).__init__()
#         self.is_train = is_train
#         self.path = path
#         if self.is_train == 1:
#             # data_list = os.path.join(path, 'train_normal.txt')
#             data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\train_normal.txt"
#             with open(data_list, 'r') as f:
#                 self.data_list = f.readlines()
#         else:
#             # data_list = os.path.join(path, 'test_normalv2.txt')
#             data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\test_normalv2.txt"
#             with open(data_list, 'r') as f:
#                 self.data_list = f.readlines()
#             random.shuffle(self.data_list)
#             self.data_list = self.data_list[:-10]
#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         if self.is_train == 1:
#             rgb_npy = np.load(os.path.join(self.path+'Features', self.data_list[idx][:-1]+'.npy'))
#             # print("Shape before: ", rgb_npy.shape)
#             # rgb_npy = rgb_npy[: , :-512] # Text Only
#             rgb_npy = rgb_npy # text Only (has size 10240, no video features)
#             # print("Shape after: ", rgb_npy.shape)
#             return rgb_npy
#         else:
#             name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
#             rgb_npy = np.load(os.path.join(self.path+'Features', name + '.npy'))
#             # print("Shape before: ", rgb_npy.shape)
#             # rgb_npy = rgb_npy[: , :-512] # Text Only
#             rgb_npy = rgb_npy # text Only (has size 10240, no video features)
#             return rgb_npy, gts, frames

class Normal_Loader_TextOnly(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI'):
        super(Normal_Loader_TextOnly, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\train_normal.txt"
        else:
            data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\test_normalv2.txt"
            
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()
        if not self.is_train:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            # Clean up the file path
            file_name = self.data_list[idx].strip()  # Remove newline
            # Split the path to get the category and filename
            parts = file_name.split('/')
            if len(parts) == 2:
                category, filename = parts
            else:
                filename = parts[0]
                category = filename.split('_')[0]  # Extract category from filename
                
            full_path = os.path.join(self.path, 'UCF-Crime Features Finalized', category, filename + '.npy')
            full_path = os.path.normpath(full_path)  # Normalize path separators
            
            # Debug print
            # print(f"Loading file: {full_path}")
            
            try:
                rgb_npy = np.load(full_path).astype(np.float32)
                # print(rgb_npy.shape)
                # rgb_npy = rgb_npy[: , 512:] # text Only
                rgb_npy = rgb_npy[: , :512] # video Only
                return rgb_npy
            except FileNotFoundError:
                print(f"File not found: {full_path}")
                raise
                
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            
            # Similar path handling for test data
            parts = name.split('/')
            if len(parts) == 2:
                category, filename = parts
            else:
                filename = parts[0]
                category = filename.split('_')[0]
                
            full_path = os.path.join(self.path, 'UCF-Crime Features Finalized', category, filename + '.npy')
            full_path = os.path.normpath(full_path)
            
            # Debug print
            # print(f"Loading file: {full_path}")
            
            try:
                rgb_npy = np.load(full_path).astype(np.float32)
                # rgb_npy = rgb_npy[: , 512:] # text Only
                rgb_npy = rgb_npy[: , :512] # video Only
                return rgb_npy, gts, frames
            except FileNotFoundError:
                print(f"File not found: {full_path}")
                raise

# class Anomaly_Loader_TextOnly(Dataset):
#     def __init__(self, is_train=1, path='E:\\Llama 3\\'):
#         super(Anomaly_Loader_TextOnly, self).__init__()
#         self.is_train = is_train
#         self.path = path
#         if self.is_train == 1:
#             # data_list = os.path.join(path, 'train_anomaly.txt')
#             data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\train_anomaly.txt"
#             with open(data_list, 'r') as f:
#                 self.data_list = f.readlines()
#         else:
#             # data_list = os.path.join(path, 'test_anomalyv2.txt')
#             data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\test_anomalyv2.txt"
#             with open(data_list, 'r') as f:
#                 self.data_list = f.readlines()

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         if self.is_train == 1:
#             rgb_npy = np.load(os.path.join(self.path+'Features', self.data_list[idx][:-1]+'.npy'))
#             # rgb_npy = rgb_npy[: , :-512] # Text Only
#             rgb_npy = rgb_npy # text Only (has size 10240, no video features)
#             return rgb_npy
#         else:
#             name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
#             gts = [int(i) for i in gts]
#             rgb_npy = np.load(os.path.join(self.path+'Features', name + '.npy'))
#             # rgb_npy = rgb_npy[: , :-512] # Text Only
#             rgb_npy = rgb_npy # text Only (has size 10240, no video features)
#             # return rgb_npy, gts, frames


class Anomaly_Loader_TextOnly(Dataset):
    def __init__(self, is_train=1, path='E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\'):
        super(Anomaly_Loader_TextOnly, self).__init__()
        self.is_train = is_train
        self.path = path
        if self.is_train == 1:
            data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\train_anomaly.txt"
        else:
            data_list = "E:\\Anomaly Detection MAXI MultiScale\\New Training Code UCF-Crime Features Finalized and ANOMALY-MAXI\\UCF-Crime_ablations_studies\\test_anomalyv2.txt"
            
        with open(data_list, 'r') as f:
            self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            # Clean up the file path
            file_name = self.data_list[idx].strip()  # Remove newline
            # Split the path to get the category and filename
            parts = file_name.split('/')
            if len(parts) == 2:
                category, filename = parts
            else:
                filename = parts[0]
                category = filename.split('_')[0]  # Extract category from filename
                
            full_path = os.path.join(self.path, 'UCF-Crime Features Finalized', category, filename + '.npy')
            full_path = os.path.normpath(full_path)
            
            # Debug print
            # print(f"Loading file: {full_path}")
            
            try:
                rgb_npy = np.load(full_path).astype(np.float32)
                # rgb_npy = rgb_npy[: , 512:] # text Only
                rgb_npy = rgb_npy[: , :512] # video Only
                return rgb_npy
            except FileNotFoundError:
                print(f"File not found: {full_path}")
                raise
                
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            
            # Similar path handling for test data
            parts = name.split('/')
            if len(parts) == 2:
                category, filename = parts
            else:
                filename = parts[0]
                category = filename.split('_')[0]
                
            full_path = os.path.join(self.path, 'UCF-Crime Features Finalized', category, filename + '.npy')
            full_path = os.path.normpath(full_path)
            
            # Debug print
            # print(f"Loading file: {full_path}")
            
            try:
                rgb_npy = np.load(full_path).astype(np.float32)
                # rgb_npy = rgb_npy[: , 512:] # text Only
                rgb_npy = rgb_npy[: , :512] # video Only
                return rgb_npy, gts, frames
            except FileNotFoundError:
                print(f"File not found: {full_path}")
                raise


## FCN LEARNER MODEL ##

# class Learner(nn.Module):
#     def __init__(self, input_dim=10752, drop_p=0.0):
#         super(Learner, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(512, 32),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#         self.drop_p = 0.6
#         self.weight_init()
#         self.vars = nn.ParameterList()

#         for i, param in enumerate(self.classifier.parameters()):
#             self.vars.append(param)

#     def weight_init(self):
#         for layer in self.classifier:
#             if type(layer) == nn.Linear:
#                 nn.init.xavier_normal_(layer.weight)

#     def forward(self, x, vars=None):
#         if vars is None:
#             vars = self.vars
#         x = F.linear(x, vars[0], vars[1])
#         x = F.relu(x)
#         x = F.dropout(x, self.drop_p, training=self.training)
#         x = F.linear(x, vars[2], vars[3])
#         x = F.dropout(x, self.drop_p, training=self.training)
#         x = F.linear(x, vars[4], vars[5])
#         return torch.sigmoid(x)

#     def parameters(self):
#         return self.vars

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

        # Ensure parameters are float32
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param.float())

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        # Ensure input is float32
        x = x.float()
        x = F.linear(x, vars[0].float(), vars[1].float())
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[2].float(), vars[3].float())
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        x = F.linear(x, vars[4].float(), vars[5].float())
        return torch.sigmoid(x)

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


## MAIN CODE ##

# def train(epoch, model, optimizer, scheduler):
#     print('\nEpoch: %d' % epoch)
#     model.train()
#     train_loss = 0
#     for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
#         inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
#         batch_size = inputs.shape[0]
#         inputs = inputs.view(-1, inputs.size(-1)).to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, batch_size)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     print('loss = {}', train_loss/len(normal_train_loader))
#     scheduler.step()

def train(epoch, model, optimizer, scheduler):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        # Ensure inputs are float32
        normal_inputs = normal_inputs.float()
        anomaly_inputs = anomaly_inputs.float()
        
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('loss = {}', train_loss/len(normal_train_loader))
    scheduler.step()

def test_abnormal(epoch, model):
    model.eval()
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))
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
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(torch.device('cuda'))
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
        print('auc = ', auc/140)
    return auc/140


# normal_train_dataset = Normal_Loader_VidOnly(is_train=1)
# normal_test_dataset = Normal_Loader_VidOnly(is_train=0)

# anomaly_train_dataset = Anomaly_Loader_VidOnly(is_train=1)
# anomaly_test_dataset = Anomaly_Loader_VidOnly(is_train=0)

# normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
# normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

# anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
# anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# criterion = MIL


def objectiveVidOnly(trial):
    # input_dim = 10752
    input_dim = 512 # Video Only
    drop_p = trial.suggest_float("dropout", 0.0, 0.9)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

    model = Learner(input_dim=input_dim, drop_p=drop_p).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10, 15, 20, 25, 50])
    criterion = MIL

    auc_score = 0
    max_auc = 0

    for epoch in range(50):  # Reduce the number of epochs for quicker optimization
        train(epoch, model, optimizer, scheduler)
        auc_score = test_abnormal(epoch, model)
        if auc_score > max_auc:
            max_auc = auc_score
        print(f"Epoch {epoch}, AUC: {auc_score:.6f}")

    return max_auc


# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveVidOnly, n_trials=50)

# print("Best trial:")
# trial = study.best_trial
# print(f"  AUC: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")


normal_train_dataset = Normal_Loader_TextOnly(is_train=1)
normal_test_dataset = Normal_Loader_TextOnly(is_train=0)

anomaly_train_dataset = Anomaly_Loader_TextOnly(is_train=1)
anomaly_test_dataset = Anomaly_Loader_TextOnly(is_train=0)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = MIL


# def objectiveTextOnly(trial):
#     # input_dim = 10752
#     input_dim = 10240 # Text Only
#     drop_p = trial.suggest_float("dropout", 0.0, 0.9)
#     lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)

#     model = Learner(input_dim=input_dim, drop_p=drop_p).to(device)
#     optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10, 15, 20, 25, 50])
#     criterion = MIL

#     auc_score = 0
#     max_auc = 0

#     for epoch in range(50):  # Reduce the number of epochs for quicker optimization
#         train(epoch, model, optimizer, scheduler)
#         auc_score = test_abnormal(epoch, model)
#         if auc_score > max_auc:
#             max_auc = auc_score
#         print(f"Epoch {epoch}, AUC: {auc_score:.6f}")

#     return max_auc

def objectiveTextOnly(trial):
    # input_dim = 10240  # Text Only
    input_dim = 512  # Text Only
    drop_p = trial.suggest_float("dropout", 0.0, 0.9)
    # Update deprecated methods
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    model = Learner(input_dim=input_dim, drop_p=drop_p).to(device)
    # Ensure model parameters are float32
    model.double().float()  # Convert to double then back to float to ensure proper conversion
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 10, 15, 20, 25, 50])

    auc_score = 0
    max_auc = 0

    try:
        for epoch in range(50):
            train(epoch, model, optimizer, scheduler)
            auc_score = test_abnormal(epoch, model)
            if auc_score > max_auc:
                max_auc = auc_score
            print(f"Epoch {epoch}, AUC: {auc_score:.6f}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 0.0

    return max_auc



studyText = optuna.create_study(direction='maximize')
studyText.optimize(objectiveTextOnly, n_trials=50)

print("Video Only")
print("Best trial:")
trial = studyText.best_trial
print(f"  AUC: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# print("Video Only")
# print("Best trial:")
# trial = study.best_trial
# print(f"  AUC: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")