# import torch
# import clip
# from PIL import Image
# import os
# import cv2
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)
# CLIP_model.eval()

# def process_snippet(snippet_frames):
#     image_features = []
#     for frame in snippet_frames:
#         frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         frame_preprocessed = CLIP_preprocess(frame).unsqueeze(0).to(device)
#         with torch.no_grad():
#             image_feature = CLIP_model.encode_image(frame_preprocessed)
#             image_features.append(image_feature)
    
#     if len(image_features) == 0:
#         return None
#     image_features = torch.cat(image_features, dim=0)
#     image_features = torch.mean(image_features, dim=0)
#     return image_features

# def split_video(video_path, num_snippets=32):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames_per_snippet = total_frames // num_snippets
#     snippets = []

#     for i in range(num_snippets):
#         snippet_frames = []
#         for _ in range(frames_per_snippet):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             snippet_frames.append(frame)
#         snippets.append(snippet_frames)

#     cap.release()
#     return snippets

# def load_text_features(file_path):
#     with open(file_path, 'r') as file:
#         lines = [line.strip() for line in file.readlines()]
#     text_tokens = clip.tokenize(lines).to(device)
#     with torch.no_grad():
#         text_features = CLIP_model.encode_text(text_tokens).float()
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#     return lines, text_features

# # def evaluate_snippets(base_dir, lines, text_features, label, video_list, is_normal=False):
# #     predictions = []
# #     labels = []
# #     for video_info in video_list:
# #         if is_normal:
# #             video_path, total_frames, _ = video_info.split(" ")
# #         else:
# #             video_path, total_frames, anomalous_periods = video_info.split("|")
# #         full_video_path = os.path.join(base_dir, video_path)
# #         print("loaded video: " , video_path)
# #         snippets = split_video(full_video_path)
# #         for snippet in snippets:
# #             snippet_features = process_snippet(snippet)
# #             if snippet_features is not None:
# #                 snippet_features = snippet_features.float()
# #                 probs = (100.0 * snippet_features @ text_features.T).softmax(dim=-1)
# #                 max_prob_index = probs.argmax().item()
# #                 predicted_label = lines[max_prob_index]
# #                 print("predicted_label: ", predicted_label)
# #                 predictions.append(predicted_label)
# #                 labels.append(label)
        
# #         # print("predictions: ", predictions)
# #         print("Finished video: " , video_path)
# #     return predictions, labels

# def evaluate_snippets(base_dir, lines, text_features, video_list, is_normal=False):
#     predictions = []
#     labels = []
#     for video_info in video_list:
#         if is_normal:
#             video_path, total_frames, _ = video_info.split(" ")
#             total_frames = int(total_frames)
#             anomalous_periods = []
#         else:
#             video_path, total_frames, anomalous_periods = video_info.split("|")
#             total_frames = int(total_frames)
#             anomalous_periods = eval(anomalous_periods)
        
#         full_video_path = os.path.join(base_dir, video_path)
#         snippets = split_video(full_video_path)
#         snippet_length = total_frames // len(snippets)
        
#         for i, snippet in enumerate(snippets):
#             snippet_features = process_snippet(snippet)
#             if snippet_features is not None:
#                 snippet_features = snippet_features.float()
#                 probs = (100.0 * snippet_features @ text_features.T).softmax(dim=-1)
#                 max_prob_index = probs.argmax().item()
#                 predicted_label = lines[max_prob_index]
#                 predictions.append(predicted_label)
                
#                 # Determine if this snippet is anomalous
#                 snippet_start = i * snippet_length
#                 snippet_end = snippet_start + snippet_length
#                 is_anomalous = any(start <= snippet_start <= end or start <= snippet_end <= end for start, end in anomalous_periods)
#                 labels.append(1 if is_anomalous else 0)
#     return predictions, labels


# def get_ground_truth_labels(video_list, anomaly_intervals):
#     ground_truth_labels = []
#     for video_info in video_list:
#         video_path, total_frames, anomalous_periods = video_info.split("|")
#         total_frames = int(total_frames)
#         anomalous_periods = eval(anomalous_periods)
#         labels = np.zeros(total_frames)
#         for start, end in anomalous_periods:
#             labels[start:end+1] = 1
#         ground_truth_labels.extend(labels.tolist())
#     return ground_truth_labels

# def load_video_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file.readlines()]

# # Load text data and corresponding features
# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700AnomalyActions.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700NormalActions.txt')

# # # Evaluate normal videos
# # normal_video_list = load_video_list('test_normalv2smallSplit.txt')
# # normal_preds, normal_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', normal_lines, normal_features, 'Normal', normal_video_list, True)

# # # Evaluate anomaly videos
# # anomaly_video_list = load_video_list('test_anomalyv2smallSplit.txt')
# # anomaly_preds, anomaly_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', anomaly_lines, anomaly_features, 'Anomaly', anomaly_video_list, False)

# # # Combine predictions and labels
# # all_preds = normal_preds + anomaly_preds
# # all_labels = normal_labels + anomaly_labels

# # # Map predictions to binary labels for ROC-AUC calculation
# # binary_preds = [1 if pred in anomaly_lines else 0 for pred in all_preds]
# # binary_labels = [1 if label == 'Anomaly' else 0 for label in all_labels]

# # # Calculate AUC-ROC score
# # auc_roc = roc_auc_score(binary_labels, binary_preds)

# # print(f"AUC-ROC Score: {auc_roc}")


# # Evaluate normal videos
# normal_video_list = load_video_list('test_normalv2smallSplit.txt')
# normal_preds, normal_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', normal_lines, normal_features, normal_video_list, True)

# # Evaluate anomaly videos
# anomaly_video_list = load_video_list('test_anomalyv2smallSplit.txt')
# anomaly_preds, anomaly_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', anomaly_lines, anomaly_features, anomaly_video_list, False)

# # Combine predictions and labels
# all_preds = normal_preds + anomaly_preds
# all_labels = normal_labels + anomaly_labels

# # Map predictions to binary labels for ROC-AUC calculation
# binary_preds = [1 if pred in anomaly_lines else 0 for pred in all_preds]

# # Calculate AUC-ROC score
# auc_roc = roc_auc_score(all_labels, binary_preds)

# print(f"AUC-ROC Score: {auc_roc}")


########## VERSION 2 ############


# import torch
# import clip
# from PIL import Image
# import os
# import cv2
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Load the CLIP model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)
# CLIP_model.eval()

# def process_snippet(snippet_frames):
#     image_features = []
#     for frame in snippet_frames:
#         frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         frame_preprocessed = CLIP_preprocess(frame).unsqueeze(0).to(device)
#         with torch.no_grad():
#             image_feature = CLIP_model.encode_image(frame_preprocessed)
#             image_features.append(image_feature)
    
#     if len(image_features) == 0:
#         return None
#     image_features = torch.cat(image_features, dim=0)
#     image_features = torch.mean(image_features, dim=0)
#     return image_features

# def split_video(video_path, num_snippets=32):
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames_per_snippet = total_frames // num_snippets
#     snippets = []

#     for i in range(num_snippets):
#         snippet_frames = []
#         for _ in range(frames_per_snippet):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             snippet_frames.append(frame)
#         snippets.append(snippet_frames)

#     cap.release()
#     return snippets

# def load_text_features(file_path):
#     with open(file_path, 'r') as file:
#         lines = [line.strip() for line in file.readlines()]
#     text_tokens = clip.tokenize(lines).to(device)
#     with torch.no_grad():
#         text_features = CLIP_model.encode_text(text_tokens).float()
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#     return lines, text_features

# def evaluate_snippets(base_dir, lines, text_features, video_list, is_normal=False):
#     predictions = []
#     labels = []
#     for video_info in video_list:
#         if is_normal:
#             video_path, total_frames, _ = video_info.split(" ")
#             total_frames = int(total_frames)
#             anomalous_periods = []
#         else:
#             video_path, total_frames, anomalous_periods = video_info.split("|")
#             total_frames = int(total_frames)
#             anomalous_periods = eval(anomalous_periods)
#             if isinstance(anomalous_periods[0], int):
#                 # Convert to list of tuples if the format is [start1, end1, start2, end2, ...]
#                 anomalous_periods = [(anomalous_periods[i], anomalous_periods[i+1]) for i in range(0, len(anomalous_periods), 2)]

#         full_video_path = os.path.join(base_dir, video_path)
#         print("loaded video: " , video_path)
#         snippets = split_video(full_video_path)
#         snippet_length = total_frames // len(snippets)

#         for i, snippet in enumerate(snippets):
#             snippet_features = process_snippet(snippet)
#             if snippet_features is not None:
#                 snippet_features = snippet_features.float()
#                 probs = (100.0 * snippet_features @ text_features.T).softmax(dim=-1)
#                 max_prob_index = probs.argmax().item()
#                 predicted_label = lines[max_prob_index]
#                 print("predicted_label: ", predicted_label)
#                 predictions.append(predicted_label)

#                 # Determine if this snippet is anomalous
#                 snippet_start = i * snippet_length
#                 snippet_end = snippet_start + snippet_length
#                 is_anomalous = any(start <= snippet_start <= end or start <= snippet_end <= end for start, end in anomalous_periods)
#                 labels.append(1 if is_anomalous else 0)
#     return predictions, labels


# def load_video_list(file_path):
#     with open(file_path, 'r') as file:
#         return [line.strip() for line in file.readlines()]

# # Load text data and corresponding features
# # anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700AnomalyActions.txt')
# # normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700NormalActions.txt')

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4AnomalyActions.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4NormalActions.txt')

# # anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4 and K700\\GPT4andK700AnomalyActions.txt')
# # normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4 and K700\\GPT4andK700NormalActions.txt')


# # Evaluate normal videos
# normal_video_list = load_video_list('test_normalv2smallSplit.txt')
# normal_preds, normal_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', normal_lines, normal_features, normal_video_list, True)

# # Evaluate anomaly videos
# anomaly_video_list = load_video_list('test_anomalyv2smallSplit.txt')
# anomaly_preds, anomaly_labels = evaluate_snippets('E:\\Anomaly Detection MAXI MultiScale\\UCFCrime', anomaly_lines, anomaly_features, anomaly_video_list, False)

# # Combine predictions and labels
# all_preds = normal_preds + anomaly_preds
# all_labels = normal_labels + anomaly_labels

# # Map predictions to binary labels for ROC-AUC calculation
# binary_preds = [1 if pred in anomaly_lines else 0 for pred in all_preds]

# # Calculate AUC-ROC score
# auc_roc = roc_auc_score(all_labels, binary_preds)

# print(f"AUC-ROC Score: {auc_roc}")






##########  VERSION 3 #########

import torch
import clip
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time 
start_time = time.time()

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)
CLIP_model.eval()

def process_snippet(snippet_frames):
    image_features = []
    for frame in snippet_frames:
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_preprocessed = CLIP_preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = CLIP_model.encode_image(frame_preprocessed)
            image_features.append(image_feature)
    
    if len(image_features) == 0:
        return None
    image_features = torch.cat(image_features, dim=0)
    image_features = torch.mean(image_features, dim=0)
    return image_features

def split_video(video_path, num_snippets=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_snippet = total_frames // num_snippets
    snippets = []

    for i in range(num_snippets):
        snippet_frames = []
        for _ in range(frames_per_snippet):
            ret, frame = cap.read()
            if not ret:
                break
            snippet_frames.append(frame)
        snippets.append(snippet_frames)

    cap.release()
    return snippets

def load_text_features(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    text_tokens = clip.tokenize(lines).to(device)
    with torch.no_grad():
        text_features = CLIP_model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return lines, text_features

def evaluate_snippets(base_dir, lines, text_features, video_list, is_normal=False):
    predictions = []
    labels = []
    for video_info in video_list:
        if is_normal:
            video_path, total_frames, _ = video_info.split(" ")
            total_frames = int(total_frames)
            anomalous_periods = []
        else:
            video_path, total_frames, anomalous_periods = video_info.split("|")
            total_frames = int(total_frames)
            anomalous_periods = eval(anomalous_periods)
            if isinstance(anomalous_periods[0], int):
                # Convert to list of tuples if the format is [start1, end1, start2, end2, ...]
                anomalous_periods = [(anomalous_periods[i], anomalous_periods[i+1]) for i in range(0, len(anomalous_periods), 2)]

        full_video_path = os.path.join(base_dir, video_path)
        print("loaded video: " , video_path)
        snippets = split_video(full_video_path)
        snippet_length = total_frames // len(snippets)

        for i, snippet in enumerate(snippets):
            snippet_features = process_snippet(snippet)
            if snippet_features is not None:
                snippet_features = snippet_features.float()
                probs = (100.0 * snippet_features @ text_features.T).softmax(dim=-1)
                max_prob_index = probs.argmax().item()
                predicted_label = lines[max_prob_index]
                print("predicted_label: ", predicted_label)
                predictions.append(predicted_label)

                # Determine if this snippet is anomalous
                snippet_start = i * snippet_length
                snippet_end = snippet_start + snippet_length
                is_anomalous = any(start <= snippet_start <= end or start <= snippet_end <= end for start, end in anomalous_periods)
                labels.append(1 if is_anomalous else 0)
    return predictions, labels


def load_video_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

### Action Dictionaries ###

# # Load text data and corresponding features
# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700AnomalyActions.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\K700\\K700NormalActions.txt')

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4AnomalyActions.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4\\GPT4NormalActions.txt')

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4 and K700\\GPT4andK700AnomalyActions.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TextDictionaries\\GPT4 and K700\\GPT4andK700NormalActions.txt')

### Object Dictionaries ###

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\COCO\\COCOAnomaly.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\COCO\\COCONormal.txt')

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\GPT4\\GPT4Anomaly.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\GPT4\\GPT4Normal.txt')

# anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\COCO and GPT4\\GPT4andCOCOAnomaly.txt')
# normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\ObjectTextDictionaries\\COCO and GPT4\\GPT4andCOCONormal.txt')

anomaly_lines, anomaly_features = load_text_features('E:\\EvaluateCLIPMatching\\TrainingFreeResults\\filteredDictionaries\\ActionGPT4\\window7\\FilteredGPT4AnomalyActions_thrs0.94.txt')
normal_lines, normal_features = load_text_features('E:\\EvaluateCLIPMatching\\TrainingFreeResults\\filteredDictionaries\\ActionGPT4\\window7\\FilteredGPT4NormalActions_thrs0.94.txt')

# Combine normal and anomaly text features and lines
all_lines = normal_lines + anomaly_lines
all_features = torch.cat([normal_features, anomaly_features], dim=0)

# Evaluate normal videos
# normal_video_list = load_video_list('test_normalv2smallSplit.txt')
normal_video_list = load_video_list('E:\\Anomaly Detection MAXI MultiScale\\XDViolance\\XDVGENtest_normalv2.txt')
normal_preds, normal_labels = evaluate_snippets('F:\\videos\\videos', all_lines, all_features, normal_video_list, True)

# Evaluate anomaly videos
# anomaly_video_list = load_video_list('test_anomalyv2smallSplit.txt')
anomaly_video_list = load_video_list('E:\\Anomaly Detection MAXI MultiScale\\XDViolance\\XDVGENtest_anomalyv2.txt')
anomaly_preds, anomaly_labels = evaluate_snippets('F:\\videos\\videos', all_lines, all_features, anomaly_video_list, False)

# Combine predictions and labels
all_preds = normal_preds + anomaly_preds
all_labels = normal_labels + anomaly_labels

# Map predictions to binary labels for ROC-AUC calculation
binary_preds = [1 if pred in anomaly_lines else 0 for pred in all_preds]

# Calculate AUC-ROC score
auc_roc = roc_auc_score(all_labels, binary_preds)

print(f"AUC-ROC Score: {auc_roc}")
print(f"Execution time: {time.time() - start_time} seconds")

