import os
import json
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

import nltk
import numpy as np
import torch
import torch.nn as nn

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ChatBotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatBotAssistant:
    def __init__(self, intents_path):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file '{self.intents_path}' not found!")
        
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))

        self.vocabulary = sorted(set(self.vocabulary))
        print(f"Loaded {len(self.intents)} intents and {len(self.vocabulary)} vocabulary words")

    def load_model(self, model_path, dimension_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        if not os.path.exists(dimension_path):
            raise FileNotFoundError(f"Dimension file '{dimension_path}' not found!")
        
        with open(dimension_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print("Model loaded successfully!")

    def process_message(self, input_message):
        if not self.model:
            raise ValueError("Model not loaded!")
            
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if predicted_intent in self.intents_responses and self.intents_responses[predicted_intent]:
            return {
                "response": random.choice(self.intents_responses[predicted_intent]),
                "intent": predicted_intent
            }
        else:
            return {
                "response": "I'm not sure how to respond to that.",
                "intent": "unknown"
            }

# FastAPI Setup
chatbot = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot
    
    # Initialize and load the chatbot
    try:
        print("Loading chatbot...")
        chatbot = ChatBotAssistant('intents.json')
        chatbot.parse_intents()
        chatbot.load_model('chatbot_model.pth', 'dimensions.json')
        print("✅ Chatbot loaded and ready!")
    except Exception as e:
        print(f"❌ Error loading chatbot: {e}")
        chatbot = None
    
    yield
    # Shutdown - cleanup if needed

app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# Pydantic model for request
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint that uses your custom model"""
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot is not loaded. Please check server logs.")
    
    try:
        result = chatbot.process_message(request.message)

        reply = result['response']

        return {"reply": reply}
        
    except Exception as e:
        print(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail="Error processing your message")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if chatbot else "unhealthy", 
        "chatbot_loaded": chatbot is not None,
        "intents_loaded": len(chatbot.intents) > 0 if chatbot else False
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chatbot API is running!", "status": "active"}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
