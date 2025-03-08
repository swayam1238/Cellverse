import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import open_clip
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import faiss

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.to(device).eval()

# Extract frames from video using OpenCV
def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(image).unsqueeze(0))
        count += 1
    cap.release()
    return frames

# Convert frames to embeddings using CLIP
def get_video_embeddings(video_path):
    frames = extract_frames(video_path)
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            embedding = model.encode_image(frame.to(device))
            embeddings.append(embedding.cpu().numpy().flatten())  # Flatten the embedding to 1D
    return np.array(embeddings)  # Ensure the result is a 2D array

# Perform semantic search using FAISS
def search_similar_videos(query_embedding, embeddings, top_k=3):
    # Index the embeddings using FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance (Euclidean)
    index.add(embeddings)  # Add embeddings to the index
    
    # Search for the top-k most similar embeddings to the query
    D, I = index.search(np.array([query_embedding]), top_k)  # D = distances, I = indices
    return I, D

# Summarization using an LLM (Hugging Face's BART model)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Full pipeline
def process_video_with_summary(video_path, text_query):
    print("Extracting embeddings...")
    embeddings = get_video_embeddings(video_path)
    
    # Use the first frame as the query
    query_embedding = embeddings[0]
    
    print("Performing search...")
    indices, distances = search_similar_videos(query_embedding, embeddings)
    
    print("Generating summary...")
    summary = generate_summary(text_query)

    print("\nResults:")
    print(f"Top FAISS search results (indices): {indices}")
    print(f"Distances: {distances}")
    print(f"Generated Summary: {summary}")

# Example usage
video_path = "/video/case_2004.mp4"
text_query = "A cataract operation or surgery."
process_video_with_summary(video_path, text_query)