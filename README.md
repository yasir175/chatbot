# 🤖 Simple ChatBot

A full-stack AI-powered chatbot built with a custom-trained PyTorch neural network backend and a clean, responsive web frontend. The bot understands natural language through intent classification and is trained on a rich dataset covering topics like machine learning, deep learning, NLP, computer vision, and general conversation.

---

## 📁 Project Structure

```
chatbot/
├── back_end/
│   ├── main.py               # FastAPI server & inference logic
│   ├── train.py              # Model training script
│   ├── intents.json          # Training data (patterns & responses)
│   ├── chatbot_model.pth     # Pre-trained model weights
│   ├── dimensions.json       # Saved model dimensions & vocabulary metadata
│   └── requirements.txt      # Python dependencies
└── front_end/
    ├── index.html            # Chat UI markup
    ├── script.js             # Frontend logic (fetch, message rendering)
    ├── style.css             # Messenger-style chat styling
    └── logo.jpg              # Bot avatar/logo
```

---

## 🧠 How It Works

### Natural Language Understanding
The chatbot uses a **bag-of-words + feedforward neural network** approach:

1. User input is **tokenized and lemmatized** using NLTK.
2. Tokens are mapped to a **881-word vocabulary** to produce a binary bag-of-words vector.
3. The vector is passed to a trained **3-layer neural network** that classifies the message into one of **67 intent categories**.
4. A **random response** from the matched intent's response pool is returned to the user.

### Neural Network Architecture
```
Input Layer  →  128 neurons (ReLU + Dropout 0.5)
Hidden Layer →  64 neurons  (ReLU + Dropout 0.5)
Output Layer →  67 neurons  (one per intent class)
```

---

## 🗂️ Intent Categories

The model is trained on 67 intents across a wide range of topics:

| Category | Intents |
|---|---|
| **General** | greeting, goodbye, name, age, help, customer_service |
| **ML/AI** | ml, dl, nlp, ds, ml_algorithms, dl_architectures, dl_vs_ml, overfitting_and_underfitting, hyperparameter_tuning, cross_validation, bias_variance_tradeoff, ensemble_learning, feature_selection, gradient_descent, gradient_ascent |
| **Algorithms** | logistic_regression, linear_regression, naive_bayes, decision_trees, random_forests, support_vector_machines, k_nearest_neighbors, neural_networks, binary_classification |
| **Frameworks & Tools** | keras, tensorflow, pytorch, nltk, spacy, pandas, numpy, ml_frameworks, nlp_tools, ds_tools |
| **NLP** | nlp_applications, nlp_preprocessing, abstractive_summarization, extractive_summarization |
| **Computer Vision** | image_processing, image_classification, object_detection, image_segmentation, facial_recognition, computer_vision, image_recognition |
| **3D / Graphics** | computer_graphics, 3d_modeling, rendering, blender, blender_tutorials, 3d_animation |
| **Systems** | computer_systems, memory_management, hardware_software |
| **Data Science** | ds_process, ds_visualization, dl_applications, dl_limitations, dl_vs_traditional, ml_evaluation |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A modern web browser

### 1. Install Dependencies

```bash
cd back_end
pip install -r requirements.txt
```

**Dependencies:**
```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
torch==2.1.0
numpy==1.24.0
nltk==3.8.1
python-multipart==0.0.6
```

### 2. Run the Backend Server

The project ships with pre-trained weights (`chatbot_model.pth`), so you can start the server immediately without retraining:

```bash
cd back_end
python main.py
```

The API will be available at `http://127.0.0.1:8000`.

### 3. Open the Frontend

Simply open `front_end/index.html` in your browser. No build step required.

> **Note:** The frontend sends requests to `http://127.0.0.1:8000/chat`, so the backend must be running locally.

---

## 🔁 Retraining the Model

To retrain the model on the existing (or modified) `intents.json`:

```bash
cd back_end
python train.py
```

**Training configuration (defaults):**
| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 8 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |

After training completes, `chatbot_model.pth` and `dimensions.json` will be updated automatically.

### Customizing Intents

To add new topics or responses, edit `intents.json` following this structure:

```json
{
  "intents": [
    {
      "tag": "your_intent_name",
      "patterns": [
        "example user message",
        "another way to phrase it"
      ],
      "responses": [
        "Bot reply option 1",
        "Bot reply option 2"
      ],
      "context_set": ""
    }
  ]
}
```

Then retrain the model using `python train.py`.

---

## 🌐 API Reference

### `POST /chat`
Send a message to the chatbot.

**Request body:**
```json
{ "message": "What is machine learning?" }
```

**Response:**
```json
{ "reply": "Machine learning is a subset of AI that enables systems to learn from data." }
```

### `GET /health`
Check server and model status.

```json
{
  "status": "healthy",
  "chatbot_loaded": true,
  "intents_loaded": true
}
```

### `GET /`
Basic liveness check.

```json
{ "message": "Chatbot API is running!", "status": "active" }
```

---

## 🎨 Frontend

The chat interface is styled to resemble a modern messaging app (inspired by Facebook Messenger):

- **Messenger-style bubbles** — blue for user, grey for bot
- **Bot avatar** displayed alongside each bot message
- **Send on Enter** or button click
- **Auto-scroll** to the latest message
- **Disabled send button** while awaiting a response
- **Font Awesome** icons for the send button

---

## 📊 Model Info

| Property | Value |
|---|---|
| Input size (vocabulary) | 881 |
| Output classes (intents) | 67 |
| Model file size | ~493 KB |
| Inference device | CPU |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch |
| NLP Preprocessing | NLTK (tokenization + lemmatization) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Data Format | JSON (intents) |
