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
from lavis.models import load_model_and_preprocess


def get_synonyms(word):
    """Get synonyms of a word"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def synonym_replacement(sentence):
    """Replace random words in the sentence with their synonyms"""
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= 2:  # for more replacements, increase this number
            break

    sentence = ' '.join(new_words)
    return sentence

def back_translate(text, intermediate_lang='es'):
    """
    Translate text to an intermediate language and back to the original language.
    Default intermediate language is Spanish ('es').
    """
    translator = Translator()
    # Translate to intermediate language
    translated = translator.translate(text, dest=intermediate_lang).text
    # Translate back to original language
    back_translated = translator.translate(translated, dest='en').text
    return back_translated

def text_bag_extraction(image_features):

    # Actions recognized
    # Read the lines from the file





    with open('K700new.txt', 'r') as file:
        linesK700 = [line.strip() for line in file.readlines()]

    # Tokenize the lines
    text = clip.tokenize(linesK700).to(device)
        
    # Calculate features
    with torch.no_grad():

        image_features = torch.from_numpy(image_features).float().to(device)

        # # Here we obtain the CLIP text features

        text_features = model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # # Here we calculate the logits and probabilities

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # print("Label probs:", probs)
    max_prob_index = probs.flatten().argmax()
    matched_text = linesK700[max_prob_index]
    print("Matched Text: ", matched_text)

    # Read the action descriptions from the CSV file and find the descriptions for the matched text
    action_descriptions = []
    with open('action_descriptions.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == matched_text:
                action_descriptions = row[1:6]
                break

    print("Action Descriptions:", action_descriptions)
    print("-----------------------")


    # Objects recognized

    # Read the lines from the file
    with open('COCO_objects.txt', 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    # Tokenize the lines
    text = clip.tokenize(lines).to(device)


    with torch.no_grad():

        # # Here we obtain the CLIP image features

        #image_features = torch.from_numpy(image_features).float().to(device)

        # # Here we obtain the CLIP text features

        text_features = model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # # Here we calculate the logits and probabilities

        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()


    # Find the indices of the top 5 probabilities
    top_5_indices = probs.flatten().argsort()[-5:][::-1]

    # Matched Objects
    matched_Objects = [lines[i] for i in top_5_indices]
    print("Matched Objects: ", matched_Objects)


    # Read the object descriptions from the CSV file
    object_descriptions = {}
    with open('COCO_descriptions.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for matched_obj in matched_Objects:  # Iterate through each matched text
            found = False
            csvfile.seek(0)  # Reset CSV file pointer to the beginning for each new object
            for row in reader:
                if row[0].strip().lower() == matched_obj.strip().lower():
                    object_descriptions[matched_obj] = row[1:5]  # Get the first 4 description columns
                    found = True
                    break  # Found the match, break the loop
            if not found:
                print(f"No descriptions found for: {matched_obj}")

    print("Object Descriptions:", object_descriptions)

    


    print("-------------------")
    print("Synonym replacing: ")

    synonyms_augmented_sentences = [synonym_replacement(sentence) for sentence in action_descriptions]
    for original, augmented in zip(action_descriptions, synonyms_augmented_sentences):
        print(f"Original: {original}")
        print(f"Augmented: {augmented}\n")

    

    print("-------------------")
    print("Back translation: ")


    # Perform back translation
    backtrans_augmented_sentences = [back_translate(sentence) for sentence in action_descriptions]
    for original, augmented in zip(action_descriptions, backtrans_augmented_sentences):
        print(f"Original: {original}")
        print(f"Augmented: {augmented}\n")



    # Flatten the object descriptions (since it's a dictionary of lists)
    flattened_object_descriptions = [desc for sublist in object_descriptions.values() for desc in sublist]

    # Combine action descriptions with matched text
    text_bag = [matched_text] + action_descriptions + matched_Objects + flattened_object_descriptions + synonyms_augmented_sentences + backtrans_augmented_sentences

    print("Text Bag:", text_bag)

    return text_bag



    # # Tokenize and encode each string in the text bag
    # text_inputs = clip.tokenize(text_bag).to(device)

    # with torch.no_grad():
    #     # Encode the texts
    #     text_features = model.encode_text(text_inputs).float()
    #     text_features /= text_features.norm(dim=-1, keepdim=True)

    #     # Calculate the similarity between image features and each text feature
    #     similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    # # Get the indices of the top 10 similarities
    # top_10_indices = np.argsort(similarities.flatten())[-10:][::-1]

    # # Select the top 10 text strings
    # filtered_text_bag = [text_bag[i] for i in top_10_indices]

    # print("Filtered Text Bag:", filtered_text_bag)

    # return filtered_text_bag

def process_video_files(data_dir, output_overall_feat_dir, CLIP_model, CLIP_preprocess, BLIP_model, BLIP_preprocess, device):
    """
    Process video files in subdirectories, perform feature extraction and save the features.

    :param data_dir: Directory containing subdirectories with video files.
    :param output_overall_feat_dir: Directory where output features will be saved.
    :param model: The loaded CLIP model.
    :param device: The device (cpu or cuda) where the model is loaded.
    """

    # Iterate through each subdirectory and files within
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if the file is a video file (e.g., .mp4)
            if file.endswith('.mp4'):
                # Construct the full path to the video file
                video_path = os.path.join(subdir, file)

                # # Replace the file extension from .mp4 to .npy
                # npy_file_name = file.replace('.mp4', '.npy')

                # Construct the full path for the .npy file in the output directory
                output_npy_path = os.path.join(output_overall_feat_dir, os.path.basename(subdir), npy_file_name)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

                # load the video
                video_frames = load_video(video_path)


                # Display the dimensions of video_frames
                print(f"Video frames dimensions: {video_frames.shape}")  # [N_frames, height, width, N_channels]

                # Split the video into 32 snippets
                snippets = split_video(video_frames, 32) #CONTINUE FROM HERE

                short_view_features = torch.zeros((11, 512, number_of_batches)).to(device)

                long_view_features = torch.zeros((11, 512, number_of_batches)).to(device)

                processed_snippets = [process_snippet(snippet, CLIP_model, CLIP_preprocess, BLIP_model, BLIP_preprocess, device) for snippet in snippets]
                print("processed_snippets :  " , processed_snippets.shape)

                # Save the overall_features tensor as a .npy file
                np.save(output_npy_path, processed_snippets.cpu().numpy())

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return np.array(frames)

def split_video(video_frames, n_snippets=32):
    snippet_length = len(video_frames) // n_snippets
    snippets = []

    for i in range(n_snippets):
        start = i * snippet_length
        end = start + snippet_length
        snippets.append(video_frames[start:end])

    return snippets


# Function to process a snippet and extract CLIP features
def process_snippet(snippet, CLIP_model, CLIP_preprocess, BLIP_model, BLIP_preprocess, device):


    CLIP_model.eval()  # Set the CLIP_model to evaluation mode
    image_features = []

    for frame in snippet:
        # Convert frame to PIL Image and preprocess
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_preprocessed = preprocess(frame).unsqueeze(0).to(device)

        # Extract features
        with torch.no_grad():  # Disable gradient calculation
            image_feature = CLIP_model.encode_image(frame_preprocessed)
            image_features.append(image_feature)


        


    selected_frames = random.sample(list(snippet), 5)

    for frame in selected_frames:
        # Preprocess the frame for BLIP caption generation
        blip_input = BLIP_preprocess["eval"](frame_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            # Generate captions using BLIP model
            # Update the following line according to your BLIP model's API for generating captions
            caption = BLIP_model.generate({"image": blip_input, "prompt": "Question: what action is this? Answer:"})
            captions.append(caption)



    # Concatenate features
    image_features = torch.cat(image_features, dim=0)

    print("image_features: " , image_features.shape)

    text_bag_out  = text_bag_extraction(image_features)

    text_bag_out = text_bag_out + captions

    # Tokenize and encode each string in the text bag
    text_inputs = clip.tokenize(text_bag).to(device)

    with torch.no_grad():
        # Encode the texts
        text_features = CLIP_model.encode_text(text_inputs).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate the similarity between image features and each text feature
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    # Get the indices of the top 10 similarities
    top_10_indices = np.argsort(similarities.flatten())[-10:][::-1]

    # Select the top 10 text strings
    filtered_text_bag = [text_bag[i] for i in top_10_indices]

    print("Filtered Text Bag:", filtered_text_bag)

    # Initialize an empty tensor to store concatenated text features for the current batch
    concatenated_text_features = torch.empty(0, 512).to(device)

    # Loop through each string in text_bag_out
    for text_str in text_bag_out:
        # Tokenize the string
        text_input = CLIP_model.tokenize([text_str]).to(device)
        
        # Encode the text
        with torch.no_grad():
            text_feature = CLIP_model.encode_text(text_input)
        
        # Normalize the text feature vector
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        # Concatenate the result to the concatenated_text_features tensor
        concatenated_text_features = torch.cat((concatenated_text_features, text_feature), dim=0)

    image_features_flat = image_features.flatten(start_dim=1)
    concatenated_text_features_flat = concatenated_text_features.flatten(start_dim=1)

    # Concatenate the flattened tensors along the feature dimension
    # Assuming both tensors have the same first dimension (e.g., batch size)
    combined_features = torch.cat((image_features_flat, concatenated_text_features_flat), dim=1)

    print("Combined features shape:", combined_features.shape)

    # # Ensure image_features is a 2D tensor of shape [1, 512]
    # image_features_reshaped = torch.from_numpy(image_features).float().unsqueeze(0).to(device)


    return combined_features.cpu().numpy()






##### STARTING FROM HERE IS THE CODE FOR A SET OF FILES #####

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_model, CLIP_preprocess = clip.load("ViT-B/32", device=device)

# loads BLIP-2 pre-trained model
BLIP_model, BLIP_preprocess, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)


# Example usage
data_dir = "E:\\Anomaly Detection MAXI MultiScale\\CLIP features\\ucf\\ucf\\features\\smallcopyTest"  # Replace with actual data directory path
output_overall_feat_dir = "E:\\Anomaly Detection MAXI MultiScale\\Combined features\\ucf\\ucf\\features\\smallcopyTest"  # Replace with actual output directory path

# Assuming the CLIP model and device are already set up
process_video_files(data_dir, output_overall_feat_dir, CLIP_model, CLIP_preprocess, BLIP_model, BLIP_preprocess, device)


print("---------------------------------")
print("Processing of all videos is done.")


##### UNTIL HERE IS THE CODE FOR A SET OF FILES #####