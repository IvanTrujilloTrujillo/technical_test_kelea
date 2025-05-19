import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, PreTrainedTokenizerFast
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict
import datetime
from dotenv import load_dotenv


load_dotenv()


# Define request and response models
class ReviewRequest(BaseModel):
    review_text: str

class SimilarReview(BaseModel):
    text: str
    sentiment: str
    similarity: float

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    similar_reviews: List[SimilarReview]


# Create FastAPI app
api_version = "1.0.0"
app = FastAPI(title="Movie Review Sentiment Analysis API",
              description="API for analyzing sentiment in movie reviews using Llama 3 with LoRA fine-tuning",
              version=api_version)


# Global variables to store model and data
device = "cuda" if torch.cuda.is_available() else "cpu"
global_model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
global_tokenizer = AutoTokenizer.from_pretrained("tokenizer")
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
global_embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
global_embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)


def preprocess_data(df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_length: int = 512) -> List[Dict]:
    """Preprocess data for training by formatting prompts and tokenizing.

    Args:
        df (Dataframe): Dataframe containing the data.
        tokenizer (PreTrainedTokenizerFast): Tokenizer to use for preprocessing.
        max_length (int, optional): Maximum length of the input sequence. Defaults to 512.

    Returns:
        (List[Dict]): List of preprocessed data items.
    """
    processed_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing data"):
        text = row['text']
        label = "positive" if row['label'] == 1 else "negative"

        # Create the prompt in the format Llama 3 expects
        prompt = (f"<|begin_of_text|>Analyze the sentiment of the movie review, "
                  f"determine if it is positive, or negative, and return the answer as "
                  f"the corresponding sentiment label 'positive' or 'negative'. "
                  f"---------------------------------"
                  f"Review: {text}\nSentiment:{label}<|end_of_text|>")
        # Tokenize
        encodings = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

        processed_data.append({
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone(),
            "original_text": text,
            "original_label": row['label']
        })

    return processed_data


dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset['train'])
global_train_data = preprocess_data(train_df, global_tokenizer)

def generate_embedding(text: str, model: any, tokenizer: PreTrainedTokenizerFast) -> np.ndarray:
    """Generate embeddings for a given text using the provided model.

    Args:
        text (str): The text to generate embeddings for
        model (any): The model to use for generating embeddings
        tokenizer (PreTrainedTokenizerFast): The tokenizer for the embedding model

    Returns:
        (np.ndarray): The embedding vector
    """
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.cpu().numpy()


# Initialize Chroma vector database
# Get the dimension of embeddings by generating a sample embedding
sample_embedding = generate_embedding(
    global_train_data[0]["original_text"],
    global_embedding_model,
    global_embedding_tokenizer
)
embedding_dim = sample_embedding.shape[1]


# Create a Chroma client and collection
chroma_client = chromadb.Client()
global_vector_db = chroma_client.create_collection(
    name="movie_reviews",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)


# Pre-compute embeddings for all training data and add to Chroma
global_embedding_to_data_map = {}  # Map to store the relationship between index and original data


# Lists to collect data for batch addition
ids = []
embeddings = []
metadatas = []

for i, item in enumerate(tqdm(global_train_data, desc="Generating embeddings")):
    embedding = generate_embedding(
        item["original_text"],
        global_embedding_model,
        global_embedding_tokenizer
    )

    # Prepare data for the vector database
    ids.append(str(i))
    embeddings.append(embedding[0].tolist())  # Convert numpy array to list

    # Store metadata
    metadata = {
        "text": item["original_text"] if type(item["original_text"]) == str else item["original_text"][0],
        "sentiment": "positive" if item["original_label"] == 1 else "negative"
    }
    metadatas.append(metadata)

    # Store mapping from index to original data
    global_embedding_to_data_map[i] = metadata

# Add all embeddings to Chroma in a batch
global_vector_db.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Review Sentiment Analysis API"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": api_version,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": global_model is not None
    }


def predict_sentiment(text: str, model: any, tokenizer: PreTrainedTokenizerFast, max_length: int = 512) -> Dict[str, str | float]:
    """Predict sentiment for a given text. Also returns the confidence score.

    Args:
        text (str): The text to be analyzed.
        model (any): The model to be used for prediction.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for preprocessing.
        max_length (int, optional): The maximum length of the input sequence. Defaults to 512.

    Returns:
        (Dict[str, str | float]): A dictionary containing the predicted sentiment and confidence score.
    """
    # Create prompt
    prompt = (f"<|begin_of_text|>Analyze the sentiment of the movie review, "
                  f"determine if it is positive, or negative, and return the answer as "
                  f"the corresponding sentiment label 'positive' or 'negative'. "
                  f"---------------------------------"
                  f"Review: {text}\nSentiment:")

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True
    ).to(device)

    # Generate one token, returning scores
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        temperature=0.0,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True
    )

    # Extract the generated token id and score logits
    next_token_id = gen_outputs.sequences[0, -1].item()
    # scores is a tuple with one element per generated step; here only 1 step
    step_logits = gen_outputs.scores[0]  # shape: (batch_size=1, vocab_size)

    # Compute probabilities
    probs = torch.nn.functional.softmax(step_logits, dim=-1)
    confidence = probs[0, next_token_id].item()

    # Decode token to string
    token_str = tokenizer.decode([next_token_id]).strip().lower()

    # Determine sentiment label
    if token_str == "positive":
        sentiment = "positive"
    elif token_str == "negative":
        sentiment = "negative"
    else:
        # Fallback heuristic
        sentiment = "unknown"

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }


def find_similar_reviews(text: str, top_n: int = 2) -> List[Dict]:
    """Find similar reviews in the training data using the vector database.

    Args:
        text (str): The review text.
        top_n (int, optional): The number of top results. Defaults to 2.

    Returns:
        (List[Dict]): A list of dictionaries containing the result.
    """
    # Generate embedding for the input text
    query_embedding = generate_embedding(
        text,
        global_embedding_model,
        global_embedding_tokenizer
    )

    # Query the Chroma vector database for similar embeddings
    results = global_vector_db.query(
        query_embeddings=query_embedding[0].tolist(),
        n_results=top_n
    )

    # Convert results to the expected format
    similar_reviews = []

    # Chroma returns results in a dictionary with 'ids', 'distances', and 'metadatas' keys
    for i in range(len(results['ids'][0])):
        # Get the ID, distance, and metadata
        idx = results['ids'][0][i]
        # Chroma returns cosine distances directly (1 is most similar, 0 is least similar)
        similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
        item_data = results['metadatas'][0][i]

        similar_reviews.append({
            "text": item_data["text"],
            "sentiment": item_data["sentiment"],
            "similarity": float(similarity_score)
        })

    return similar_reviews


def complete_prediction(text: str, model: any, tokenizer: PreTrainedTokenizerFast, train_data: List[Dict]) -> Dict[str, any]:
    """Complete prediction function that returns sentiment, confidence, and similar reviews.

    Args:
        text (str): The review text.
        model (any): The model to be used.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for preprocessing.
        train_data (List[Dict]): The training data.

    Returns:
        (Dict[str, any]): A dictionary containing the result.
    """
    # Get sentiment prediction
    result = predict_sentiment(text, model, tokenizer)

    # Find similar reviews
    similar_reviews = find_similar_reviews(text, train_data)

    return {
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "similar_reviews": similar_reviews
    }


@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: ReviewRequest):
    try:
        # Check if the text is provided
        if not request.review_text or len(request.review_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Review text cannot be empty")

        # Get prediction
        result = complete_prediction(
            text=request.review_text,
            model=global_model,
            tokenizer=global_tokenizer,
            train_data=global_train_data
        )

        # Format similar reviews for response
        similar_reviews = [
            SimilarReview(
                text=review["text"],
                sentiment=review["sentiment"],
                similarity=review["similarity"]
            )
            for review in result["similar_reviews"]
        ]

        # Return response
        return SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            similar_reviews=similar_reviews
        )
    except Exception as e:
        # Change to a log error in a production environment
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
