import torch
import clip
from PIL import Image
import csv
import nltk
from nltk.corpus import wordnet
import random
from googletrans import Translator, LANGUAGES
import numpy as np
import os
import cv2
import json
import argparse
from lavis.models import load_model_and_preprocess
from pathlib import Path

class TextBagGenerator:
    def __init__(self, device, data_dir, use_blip=True):
        self.device = device
        self.use_blip = use_blip
        self.data_dir = data_dir
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # Load BLIP-2 model if requested
        if use_blip:
            try:
                self.blip_model, self.blip_preprocess, _ = load_model_and_preprocess(
                    name="blip2_t5", 
                    model_type="pretrain_flant5xxl", 
                    is_eval=True, 
                    device=device
                )
            except Exception as e:
                print(f"Warning: Could not load BLIP model: {e}")
                self.use_blip = False
        
        # Load action and object dictionaries
        self.load_dictionaries(data_dir)
        
        # Load GPT descriptions
        self.load_gpt_descriptions(data_dir)
        
    def load_dictionaries(self, data_dir):
        """Load action and object dictionaries from files"""
        keyword_dir = os.path.join(data_dir, "keyword_dictionaries")
        
        try:
            # Load Kinetics actions
            action_file = os.path.join(keyword_dir, "K700_action_keywords.txt")
            with open(action_file, 'r') as file:
                self.action_labels = [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            print("Warning: K700_action_keywords.txt not found. Using default actions.")
            self.action_labels = ["walking", "running", "sitting", "standing"]
            
        try:
            # Load COCO objects
            object_file = os.path.join(keyword_dir, "COCO_object_keywords.txt")
            with open(object_file, 'r') as file:
                self.object_labels = [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            print("Warning: COCO_object_keywords.txt not found. Using default objects.")
            self.object_labels = ["person", "car", "chair", "table"]

    def load_gpt_descriptions(self, data_dir):
        """Load GPT-4 expanded descriptions"""
        gpt_dir = os.path.join(data_dir, "gpt_descriptions")
        
        self.gpt_actions = {}
        self.gpt_objects = {}
        
        # Load GPT action descriptions
        action_file = os.path.join(gpt_dir, "gpt_action_expanded.txt")
        if os.path.exists(action_file):
            with open(action_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        action, description = line.strip().split(':', 1)
                        self.gpt_actions[action.strip()] = description.strip()
        else:
            print("Warning: gpt_action_expanded.txt not found.")
        
        # Load GPT object descriptions
        object_file = os.path.join(gpt_dir, "gpt_object_expanded.txt")
        if os.path.exists(object_file):
            with open(object_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        obj, description = line.strip().split(':', 1)
                        self.gpt_objects[obj.strip()] = description.strip()
        else:
            print("Warning: gpt_object_expanded.txt not found.")

    def get_synonyms(self, word):
        """Get synonyms of a word using WordNet"""
        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except:
            pass
        return list(synonyms)

    def synonym_replacement(self, sentence, max_replacements=2):
        """Replace random words in the sentence with their synonyms"""
        words = sentence.split()
        new_words = words.copy()
        
        # Find words that have synonyms
        replaceable_words = [word for word in words if wordnet.synsets(word)]
        random.shuffle(replaceable_words)
        
        num_replaced = 0
        for word in replaceable_words:
            if num_replaced >= max_replacements:
                break
                
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                num_replaced += 1
        
        return ' '.join(new_words)

    def extract_clip_features(self, image_features):
        """Extract CLIP-based action and object matches"""
        # Average features across frames
        avg_features = torch.mean(image_features, dim=0).float().to(self.device)
        
        # Match actions
        action_text = clip.tokenize(self.action_labels).to(self.device)
        with torch.no_grad():
            action_features = self.clip_model.encode_text(action_text).float()
            action_features /= action_features.norm(dim=-1, keepdim=True)
            action_probs = (100.0 * avg_features @ action_features.T).softmax(dim=-1)
        
        matched_action_idx = action_probs.flatten().argmax()
        matched_action = self.action_labels[matched_action_idx]
        
        # Match objects (top 5)
        object_text = clip.tokenize(self.object_labels).to(self.device)
        with torch.no_grad():
            object_features = self.clip_model.encode_text(object_text).float()
            object_features /= object_features.norm(dim=-1, keepdim=True)
            object_probs = (100.0 * avg_features @ object_features.T).softmax(dim=-1).cpu().numpy()
        
        top_5_indices = object_probs.flatten().argsort()[-5:][::-1]
        matched_objects = [self.object_labels[i] for i in top_5_indices]
        
        return matched_action, matched_objects

    def generate_blip_captions(self, frames, num_frames=5):
        """Generate captions using BLIP-2"""
        if not self.use_blip:
            return []
            
        captions = []
        selected_frames = random.sample(frames, min(num_frames, len(frames)))
        
        for frame in selected_frames:
            try:
                blip_input = self.blip_preprocess["eval"](frame).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Generate different types of captions
                    action_caption = self.blip_model.generate({
                        "image": blip_input, 
                        "prompt": "Question: What action is this? Answer:"
                    })
                    captions.extend(action_caption)
                    
                    reason_caption = self.blip_model.generate({
                        "image": blip_input, 
                        "prompt": "Question: Why did this happen? Answer:"
                    })
                    captions.extend(reason_caption)
                    
                    free_caption = self.blip_model.generate(
                        {"image": blip_input}, 
                        use_nucleus_sampling=True,
                        num_captions=1,
                        temperature=0.7,
                        repetition_penalty=1.3,
                        min_length=8,
                        max_length=32
                    )
                    captions.extend(free_caption)
            except Exception as e:
                print(f"BLIP caption generation failed: {e}")
                continue
                
        return captions

    def generate_text_bag(self, image_features, frames):
        """Generate comprehensive text bag from visual features"""
        # Extract CLIP matches
        matched_action, matched_objects = self.extract_clip_features(image_features)
        
        # Initialize text bag
        text_bag = [matched_action] + matched_objects
        
        # Add GPT descriptions if available
        if hasattr(self, 'gpt_actions') and matched_action in self.gpt_actions:
            gpt_action_desc = self.gpt_actions[matched_action]
            if gpt_action_desc and gpt_action_desc not in text_bag:
                text_bag.append(gpt_action_desc)
            
        for obj in matched_objects:
            if hasattr(self, 'gpt_objects') and obj in self.gpt_objects:
                gpt_obj_desc = self.gpt_objects[obj]
                if gpt_obj_desc and gpt_obj_desc not in text_bag:
                    text_bag.append(gpt_obj_desc)
        
        # Generate BLIP captions
        blip_captions = self.generate_blip_captions(frames)
        text_bag.extend(blip_captions)
        
        # Apply text augmentations
        augmented_descriptions = []
        for text in text_bag[:10]:  # Limit augmentation to first 10 items
            if isinstance(text, str) and len(text.split()) > 2:
                augmented = self.synonym_replacement(text)
                if augmented != text:
                    augmented_descriptions.append(augmented)
        
        text_bag.extend(augmented_descriptions)
        
        # Clean and deduplicate
        text_bag = [str(item) for item in text_bag if item]  # Ensure all strings
        text_bag = list(dict.fromkeys(text_bag))  # Remove duplicates while preserving order
        
        return text_bag, matched_action, matched_objects

    def extract_visual_features(self, image_features, text_bag):
        """Extract combined visual and textual features"""
        # Average image features
        avg_image_features = torch.mean(image_features, dim=0)
        
        # Select top textual features based on similarity
        text_inputs = clip.tokenize(text_bag[:77]).to(self.device)  # CLIP limit
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarities = (100.0 * avg_image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
        
        # Get top 20 most similar texts
        top_indices = np.argsort(similarities.flatten())[-20:][::-1]
        filtered_texts = [text_bag[i] for i in top_indices if i < len(text_bag)]
        
        # Extract features for filtered texts
        if filtered_texts:
            filtered_text_inputs = clip.tokenize(filtered_texts).to(self.device)
            with torch.no_grad():
                filtered_text_features = self.clip_model.encode_text(filtered_text_inputs)
                filtered_text_features /= filtered_text_features.norm(dim=-1, keepdim=True)
        else:
            filtered_text_features = torch.empty(0, 512).to(self.device)
        
        # Combine features
        combined_features = torch.cat([
            avg_image_features.unsqueeze(0), 
            filtered_text_features
        ], dim=0)
        
        return combined_features.flatten().cpu().numpy()

def load_video(video_path):
    """Load video frames as PIL Images"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []
        
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            frame_count += 1
        except Exception as e:
            print(f"Error processing frame {frame_count} from {video_path}: {e}")
            continue
    
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames

def split_video(frames, n_views):
    """Split video into n_views segments"""
    if len(frames) == 0:
        return []
        
    segment_length = max(1, len(frames) // n_views)
    segments = []
    
    for i in range(n_views):
        start = i * segment_length
        end = min(start + segment_length, len(frames))
        if start < len(frames):
            segments.append(frames[start:end])
    
    return segments

def process_video(video_path, generator, views=32, long_view=True, extract_visual=True):
    """Process a single video file"""
    print(f"Processing: {video_path}")
    
    # Load video
    frames = load_video(video_path)
    if not frames:
        print(f"No frames loaded from {video_path}")
        return None, None
    
    print(f"Loaded {len(frames)} frames")
    
    # Split into segments
    segments = split_video(frames, views)
    
    # Process each segment
    all_features = []
    all_text_bags = []
    
    for i, segment in enumerate(segments):
        if not segment:
            continue
            
        # Extract CLIP features for segment
        image_features = []
        for frame in segment:
            frame_preprocessed = generator.clip_preprocess(frame).unsqueeze(0).to(generator.device)
            with torch.no_grad():
                feature = generator.clip_model.encode_image(frame_preprocessed)
                image_features.append(feature)
        
        if image_features:
            image_features = torch.cat(image_features, dim=0)
            
            # Generate text bag
            text_bag, matched_action, matched_objects = generator.generate_text_bag(image_features, segment)
            all_text_bags.append({
                'segment': i,
                'text_bag': text_bag,
                'matched_action': matched_action,
                'matched_objects': matched_objects
            })
            
            # Extract visual features if requested
            if extract_visual:
                visual_features = generator.extract_visual_features(image_features, text_bag)
                all_features.append(visual_features)
    
    # Process long view if requested
    if long_view and frames:
        print("Processing long view...")
        image_features = []
        sample_frames = frames[::max(1, len(frames)//32)]  # Sample frames for long view
        
        for frame in sample_frames:
            frame_preprocessed = generator.clip_preprocess(frame).unsqueeze(0).to(generator.device)
            with torch.no_grad():
                feature = generator.clip_model.encode_image(frame_preprocessed)
                image_features.append(feature)
        
        if image_features:
            image_features = torch.cat(image_features, dim=0)
            text_bag, matched_action, matched_objects = generator.generate_text_bag(image_features, sample_frames)
            
            all_text_bags.append({
                'segment': 'long_view',
                'text_bag': text_bag,
                'matched_action': matched_action,
                'matched_objects': matched_objects
            })
            
            if extract_visual and all_features:
                visual_features = generator.extract_visual_features(image_features, text_bag)
                all_features.append(visual_features)
    
    return all_text_bags, np.array(all_features) if all_features else None

def get_dataset_structure(dataset, data_dir):
    """Get video files and their labels for the specified dataset"""
    video_dir = os.path.join(data_dir, dataset, "videos")
    label_dir = os.path.join(data_dir, dataset, "labels")
    
    video_files = []
    
    if dataset == "ucf_crime":
        # UCF-Crime has class subdirectories with .mp4 files
        for class_dir in os.listdir(video_dir):
            class_path = os.path.join(video_dir, class_dir)
            if os.path.isdir(class_path):
                for video_file in os.listdir(class_path):
                    if video_file.endswith('.mp4'):
                        video_files.append({
                            'path': os.path.join(class_path, video_file),
                            'class': class_dir,
                            'filename': video_file
                        })
    
    elif dataset == "shanghai_tech":
        # Shanghai Tech has flat structure with .avi files
        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                if video_file.endswith('.avi'):
                    video_files.append({
                        'path': os.path.join(video_dir, video_file),
                        'class': 'shanghai_tech',  # Use dataset name as class
                        'filename': video_file
                    })
    
    elif dataset == "xd_violence":
        # XD-Violence has flat structure with .mp4 files
        if os.path.exists(video_dir):
            for video_file in os.listdir(video_dir):
                if video_file.endswith('.mp4'):
                    # Extract class from filename (A for anomaly, G for normal based on pattern)
                    if 'label_A' in video_file:
                        class_name = 'anomaly'
                    elif 'label_G' in video_file:
                        class_name = 'normal'
                    else:
                        class_name = 'unknown'
                    
                    video_files.append({
                        'path': os.path.join(video_dir, video_file),
                        'class': class_name,
                        'filename': video_file
                    })
    
    return video_files

def main():
    parser = argparse.ArgumentParser(description="Generate text bags and visual features for video datasets")
    parser.add_argument("--dataset", choices=["ucf_crime", "shanghai_tech", "xd_violence"], 
                       required=True, help="Dataset to process")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--save_txt", type=bool, default=True, help="Save text bags as .txt files")
    parser.add_argument("--save_json", type=bool, default=True, help="Save text bags as .json files")
    parser.add_argument("--extract_visual", type=bool, default=True, help="Extract visual features")
    parser.add_argument("--views", type=int, default=32, help="Number of segments per video")
    parser.add_argument("--long_view", type=bool, default=True, help="Enable long-range temporal context")
    parser.add_argument("--use_blip", type=bool, default=True, help="Use BLIP-2 for caption generation")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize generator
    generator = TextBagGenerator(device, args.data_dir, use_blip=args.use_blip)
    
    # Create output directories
    output_dir = "output"
    text_output_dir = os.path.join(output_dir, "text_bags", args.dataset)
    feature_output_dir = os.path.join(output_dir, "features", args.dataset)
    
    os.makedirs(text_output_dir, exist_ok=True)
    if args.extract_visual:
        os.makedirs(feature_output_dir, exist_ok=True)
    
    # Get video files
    video_files = get_dataset_structure(args.dataset, args.data_dir)
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print(f"No video files found for dataset {args.dataset} in {args.data_dir}")
        print("Please check the directory structure and file formats.")
        return
    
    # Process videos
    for video_info in video_files:
        try:
            # Process video
            text_bags, visual_features = process_video(
                video_info['path'], 
                generator, 
                views=args.views, 
                long_view=args.long_view,
                extract_visual=args.extract_visual
            )
            
            if text_bags:
                # Create class subdirectory
                class_text_dir = os.path.join(text_output_dir, video_info['class'])
                os.makedirs(class_text_dir, exist_ok=True)
                
                # Save text bags
                base_filename = os.path.splitext(video_info['filename'])[0]
                
                if args.save_txt:
                    txt_path = os.path.join(class_text_dir, f"{base_filename}.txt")
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        for i, segment_data in enumerate(text_bags):
                            f.write(f"Segment {segment_data['segment']}:\n")
                            f.write(f"Action: {segment_data['matched_action']}\n")
                            f.write(f"Objects: {', '.join(segment_data['matched_objects'])}\n")
                            for text in segment_data['text_bag']:
                                f.write(f"{text}\n")
                            f.write("\n")
                
                if args.save_json:
                    json_path = os.path.join(class_text_dir, f"{base_filename}.json")
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(text_bags, f, indent=2, ensure_ascii=False)
                
                # Save visual features
                if args.extract_visual and visual_features is not None:
                    class_feat_dir = os.path.join(feature_output_dir, video_info['class'])
                    os.makedirs(class_feat_dir, exist_ok=True)
                    feat_path = os.path.join(class_feat_dir, f"{base_filename}.npy")
                    np.save(feat_path, visual_features)
                
                print(f"✓ Processed: {video_info['filename']}")
            
        except Exception as e:
            print(f"✗ Error processing {video_info['filename']}: {e}")
            continue
    
    print("Processing complete!")

if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    main()
