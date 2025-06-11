import os
import torch
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig
from PIL import Image
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Pre-allocate list for better performance
    frames = [None] * total_frames
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        frames[frame_idx] = pil_image
        frame_idx += 1
    
    cap.release()
    # Remove any None elements if we didn't fill the entire list
    frames = [f for f in frames if f is not None]
    return frames, total_frames

def split_video(video_frames, n_snippets):
    snippet_length = len(video_frames) // n_snippets
    
    # More efficient list comprehension approach
    return [
        video_frames[i * snippet_length : (i + 1) * snippet_length if i < n_snippets - 1 else len(video_frames)]
        for i in range(n_snippets)
    ]

def get_uniform_frames(snippet, num_frames, snippet_start_idx):
    snippet_length = len(snippet)
    if snippet_length <= num_frames:
        frame_indices = range(snippet_length)
    else:
        step = snippet_length / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
    
    selected_frames = [snippet[idx] for idx in frame_indices]
    absolute_indices = [snippet_start_idx + idx for idx in frame_indices]
    
    return selected_frames, absolute_indices

def batch_process_frames(frames, absolute_indices, model, processor, batch_size=4):
    """
    Process frames in batches to reduce GPU overhead
    
    This function processes smaller batches of frames at a time to avoid CUDA
    out-of-memory issues and improve throughput.
    """
    """Process frames in batches to reduce GPU overhead"""
    all_captions = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_indices = absolute_indices[i:i+batch_size]
        
        # Prepare all inputs at once
        batch_inputs1 = processor(images=batch_frames, text=["Question: What action is this? Answer:"] * len(batch_frames), return_tensors="pt", padding=True)
        batch_inputs2 = processor(images=batch_frames, text=["Question: Why did this happen? Answer:"] * len(batch_frames), return_tensors="pt", padding=True)
        batch_inputs3 = processor(images=batch_frames, text=["Describe this scene:"] * len(batch_frames), return_tensors="pt", padding=True)
        
        # Move inputs to GPU
        for inputs in [batch_inputs1, batch_inputs2, batch_inputs3]:
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if 'pixel_values' in k:
                        inputs[k] = v.cuda().to(dtype=torch.float16)
                    else:
                        inputs[k] = v.cuda()
        
        # Generate all captions
        with torch.no_grad():
            caption1_ids = model.generate(**batch_inputs1, max_length=50, num_beams=5, min_length=5)
            caption2_ids = model.generate(**batch_inputs2, max_length=50, num_beams=5, min_length=5)
            caption3_ids = model.generate(**batch_inputs3, max_length=50, num_beams=5, min_length=5, 
                                       temperature=0.7, do_sample=True, repetition_penalty=1.3)
            
            caption1_texts = processor.batch_decode(caption1_ids, skip_special_tokens=True)
            caption2_texts = processor.batch_decode(caption2_ids, skip_special_tokens=True)
            caption3_texts = processor.batch_decode(caption3_ids, skip_special_tokens=True)
        
        # Collect results
        for idx, frame_idx in enumerate(batch_indices):
            all_captions.append({
                'frame': frame_idx,
                'captions': [caption1_texts[idx].strip(), caption2_texts[idx].strip(), caption3_texts[idx].strip()]
            })
    
    return all_captions

def generate_captions_for_video(video_path, model, processor, n_snippets=32, frames_per_snippet=20, batch_size=8):
    """
    Generate captions for a video by splitting it into snippets and processing frames
    
    Args:
        video_path: Path to the video file
        model: The BLIP2 model
        processor: Processor instance (thread-local)
        n_snippets: Number of snippets to divide the video into
        frames_per_snippet: Number of frames to process per snippet
        batch_size: Number of frames to process in a single forward pass
    """
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    video_frames, total_frames = load_video(video_path)
    if not video_frames:
        print("No frames loaded.")
        return []

    snippets = split_video(video_frames, n_snippets)
    all_caption_data = []
    
    for snippet_idx, snippet in enumerate(snippets):
        snippet_start_idx = snippet_idx * (len(video_frames) // n_snippets)
        selected_frames, absolute_indices = get_uniform_frames(snippet, frames_per_snippet, snippet_start_idx)
        
        # Batch process frames for this snippet
        caption_results = batch_process_frames(selected_frames, absolute_indices, model, processor, batch_size)
        
        # Add snippet index to all results
        for result in caption_results:
            result['snippet'] = snippet_idx
            all_caption_data.append(result)
            
        # Print progress
        print(f"Processed snippet {snippet_idx+1}/{n_snippets}, {len(caption_results)} frames")
    
    elapsed_time = time.time() - start_time
    print(f"Video processing completed in {elapsed_time:.2f} seconds")
    return all_caption_data

def process_video_wrapper(args):
    """Wrapper function for parallel processing"""
    video_path, output_path, model_name = args
    
    # Create a new processor instance for this thread
    local_processor = Blip2Processor.from_pretrained(model_name)
    
    # Access the shared model through global space
    # This avoids creating multiple copies of the large model in memory
    
    captions = generate_captions_for_video(video_path, model, local_processor)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for caption_data in captions:
            line = f"snippet_{caption_data['snippet']}, frame_{caption_data['frame']}, {caption_data['captions'][0]}, {caption_data['captions'][1]}, {caption_data['captions'][2]}\n"
            f.write(line)
    
    print(f"Processed {video_path} -> {output_path}")
    return output_path

def process_videos(input_dir, output_dir, model_name, max_workers=2):
    """
    Process videos in parallel with thread-local processor instances
    
    Args:
        input_dir: Directory containing videos
        output_dir: Directory for output files
        model_name: Name of the model for processor initialization
        max_workers: Number of concurrent workers
    """
    video_tasks = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                
                # Create output path maintaining directory structure
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                output_file = os.path.join(output_subdir, os.path.splitext(file)[0] + '.txt')
                
                # Only pass the model_name, not the model or processor instances
                video_tasks.append((video_path, output_file, model_name))
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_video_wrapper, video_tasks))
    
    print(f"Processed {len(results)} videos")

# Define the model as a global variable to be shared across threads
model = None

if __name__ == "__main__":
    # Configure quantization
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Model name to be used for processor initialization in each thread
    model_name = "Salesforce/blip2-flan-t5-xxl"
    
    # Load the model (this will be shared across threads)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=nf4_config,
        torch_dtype=torch.float16
    )
    
    # Enable flash attention if available (for newer PyTorch/Transformers versions)
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"
    
    # Enable better transformer if available
    if hasattr(torch, "set_autocast_enabled"):
        torch.set_autocast_enabled(True)
    
    # Set your input and output directories
    input_dir = "F:\\UCFCrimeNormal\\AllVideos"
    output_dir = "E:\\BLIP-2\\Outputs"
    
    # We no longer pass the model instance - it's available as a global
    process_videos(input_dir, output_dir, model_name, max_workers=2)