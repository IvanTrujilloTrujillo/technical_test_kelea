# AI-Powered Sentiment Analysis for Movie Reviews

## Overview
This project implements an AI-powered solution for sentiment analysis of movie reviews using the IMDB dataset. It leverages fine-tuned language models to analyze and classify movie reviews as positive or negative, and provides similar reviews using vector similarity search.

## Features
- Sentiment analysis of movie reviews (positive/negative classification)
- Confidence score for predictions
- Similar review recommendations using vector embeddings
- RESTful API for easy integration

## Project Structure
```
├── ai_engineer_technical_assesment 1.ipynb  # Main notebook with all code and explanations
├── fastapi_main.py                          # Fastapi app for demostration purpose
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git ignore file
├── fine_tuned_model/                        # Directory containing the fine-tuned model
└── tokenizer/                               # Directory containing the tokenizer
```

**NOTE**: Fine-tuned model and tokenizer are not included in the repository due to file size limitations.

## Tasks Implemented
1. **Model Implementation**: Fine-tuned Llama 3 8B model with LoRA for sentiment analysis
2. **API Implementation**: FastAPI-based REST API for sentiment analysis
3. **Testing and Performance**: Evaluation of model performance with metrics
4. **Deployment Strategy**: Containerization and cloud deployment plan

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for inference)
- 16GB+ RAM

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Hugging Face token:
```
HUGGINGFACE_TOKEN=your_token_here
```

## Usage

### Running the Jupyter Notebook
The main implementation is in the Jupyter notebook. You can run it to:
- Load and preprocess the IMDB dataset
- Fine-tune the Llama 3 model with LoRA
- Evaluate model performance
- Test the sentiment analysis functionality

```bash
jupyter notebook "ai_engineer_technical_assesment 1.ipynb"
```

### Using the API
The project includes a FastAPI implementation for sentiment analysis:

1. Start the API server:
```bash
python fastapi_main.py
```

2. Send requests to the API:
```bash
curl -X POST "http://localhost:8000/analyze" \
    -H "Content-Type: application/json" \
    -d '{"review_text": "This movie was absolutely fantastic! The acting was superb."}'
```

Python example:
```python
import requests
import json

url = "http://localhost:8000/analyze"
data = {"review_text": "This movie was absolutely fantastic! The acting was superb."}
response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

## Performance
The project evaluates three models:
1. Fine-tuned Llama 3 with LoRA
2. Original Llama 3 (without fine-tuning)
3. Baseline BERT model

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- Inference time

## Dependencies
- pandas: Data manipulation
- datasets: Dataset loading and processing
- fastapi: API implementation
- scikit-learn: Metrics calculation
- python-dotenv: Environment variable management
- torch: PyTorch for deep learning
- transformers: Hugging Face Transformers for models
- peft: Parameter-Efficient Fine-Tuning
- tqdm: Progress bars
- matplotlib: Visualization
- uvicorn: ASGI server for FastAPI
- bitsandbytes: Quantization for efficient inference
- chromadb: Vector database for similar reviews
- nlpaug: Data augmentation
