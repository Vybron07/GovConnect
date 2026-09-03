from sentence_transformers import SentenceTransformer
import os

if __name__ == "__main__":
    print("Pre-downloading SentenceTransformer model for Render Build...")
    # This downloads the model to the local cache so it doesn't happen at runtime
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    print("Model successfully downloaded and cached!")

