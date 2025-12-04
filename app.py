import os
import io
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from collections import Counter

from Preliminary_Results.top_features.features import LocalLLM

# --------------------- CONFIG ---------------------
BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings.pkl')
PRODUCTS_INDEX_PATH = os.path.join(BASE_DIR, 'products_index.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
INPUT_SIZE = 300
MAX_LEN = 200  # For sentiment tokenization
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------- FLASK APP ---------------------
app = Flask(__name__)

# --------------------- LOAD PRECOMPUTED EMBEDDINGS ---------------------
with open(EMBEDDINGS_PATH, 'rb') as f:
    embeddings = pickle.load(f)
products_df = pd.read_csv(PRODUCTS_INDEX_PATH)
print(f"Loaded {len(products_df)} products and embeddings.")

# --------------------- LOAD IMAGE MODEL ---------------------
print("🔥 Loading EfficientNetB3 model...")
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
print("✅ EfficientNetB3 loaded.")

# --------------------- LOAD SENTIMENT MODEL ---------------------
print("🔥 Loading Sentiment Model...")
sentiment_model = load_model("sentiment_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
df_path = "cleaned_dataset.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"{df_path} not found.")
df_binary = pd.read_csv(df_path)

# --------------------- LOAD EMOTION MODEL ---------------------
print("🔥 Loading Emotion Model...")
emotion_model = None
emotion_tokenizer = None
try:
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    local_path = "./emotion_model"
    if os.path.exists(local_path):
        print("🔄 Loading emotion model from local cache...")
        emotion_tokenizer = AutoTokenizer.from_pretrained(local_path)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(local_path)
    else:
        print("📥 Downloading emotion model...")
        emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
        emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        emotion_tokenizer.save_pretrained(local_path)
        emotion_model.save_pretrained(local_path)
    print("✅ Emotion model loaded successfully!")
except Exception as e:
    print(f"⚠️ Failed to load emotion model: {e}")

# --------------------- HELPER FUNCTIONS ---------------------

# Image embedding & prediction
def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.expand_dims(preprocess_input(np.array(img, dtype=np.float32)), axis=0)
    emb = base_model.predict(arr, verbose=0)
    return np.mean(emb, axis=(1, 2)).reshape(1, -1)

def predict_product(img_path, top_k=1):
    img_emb = get_embedding(img_path)
    product_names = products_df['name'].tolist()
    product_embs = np.array([embeddings[name] for name in product_names])
    sims = cosine_similarity(img_emb, product_embs).flatten()
    top_idxs = sims.argsort()[::-1][:top_k]
    top_products = [(product_names[i], float(sims[i])) for i in top_idxs]
    top_brand = products_df.loc[products_df['name'] == top_products[0][0], 'brand'].values[0]
    return top_brand, top_products[0]

# Sentiment analysis
def analyze_brand_sentiment(brand_name):
    brand_reviews = df_binary[df_binary["brand"].str.lower() == brand_name.lower()]["cleaned_review_text"]
    if brand_reviews.empty:
        return None
    sequences = tokenizer.texts_to_sequences(brand_reviews)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
    preds = sentiment_model.predict(padded, verbose=0)
    pred_labels = encoder.inverse_transform(np.argmax(preds, axis=1))
    counts = pd.Series(pred_labels).value_counts().to_dict()
    return {"Positive": int(counts.get("Positive", 0)), "Negative": int(counts.get("Negative", 0))}

# Emotion analysis
def analyze_emotion(brand_name):
    subset = df_binary[df_binary["brand"].str.lower() == brand_name.lower()]
    if subset.empty or emotion_model is None:
        return "Unknown"
    reviews = subset["review_text"].dropna().tolist()[:20]
    emotions = []
    for review in reviews:
        review_text = str(review).strip()
        if len(review_text) < 10:
            continue
        try:
            inputs = emotion_tokenizer(review_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = emotion_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
            emotions.append(emotion_labels[predicted_class])
        except Exception as e:
            continue
    if not emotions:
        return "Unknown"
    dominant = Counter(emotions).most_common(1)[0][0]
    mapping = {'joy': 'enjoyment', 'love': 'satisfaction', 'sadness': 'disappointment',
               'anger': 'anger', 'fear': 'confusion', 'surprise': 'surprise'}
    return mapping.get(dominant, dominant.capitalize())

# --------------------- FLASK ROUTES ---------------------

# Route: image → product
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = file.filename or "uploaded_image.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    try:
        brand, top_product = predict_product(file_path)
        return jsonify({
            'brand': brand,
            'product_name': top_product[0],
            'similarity': top_product[1]
        })
    except Exception as e:
        return jsonify({'error': f"Backend crashed: {str(e)}"}), 500

# Route: brand → sentiment + emotion
@app.route("/analyze", methods=["POST"])
def analyze_route():
    brand_name = request.form.get("brand", "Redmi")
    sentiment_result = analyze_brand_sentiment(brand_name)
    if sentiment_result is None:
        return jsonify({"error": f"No reviews found for brand '{brand_name}'."}), 404
    total = sum(sentiment_result.values()) or 1
    percentages = {k: round((v / total) * 100, 2) for k, v in sentiment_result.items()}
    subset = df_binary[df_binary["brand"].str.lower() == brand_name.lower()]
    avg_rating = round(float(subset["rating"].mean()), 2)
    emotion = analyze_emotion(brand_name)
    return jsonify({
        "average_rating": avg_rating,
        "sentiment_percentages": percentages,
        "dominant_emotion": emotion
    })

llm = LocalLLM()

# Route for features
@app.route("/generate-features", methods=["POST"])
def generate_features():
    data = request.get_json()

    product_name = data.get("product_name")
    brand = data.get("brand")

    if not product_name or not brand:
        return jsonify({"error": "Missing 'product_name' or 'brand'"}), 400

    result = llm.generate_features(product_name, brand)
    return jsonify(result)


# --------------------- START SERVER ---------------------
if __name__ == "__main__":
    print("🚀 Flask backend running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
