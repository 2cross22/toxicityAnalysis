from flask import Flask, render_template, request, jsonify
import joblib
import re
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model and vectorizer on startup
model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        print("Starting model loading process...")
        
        # Check if files exist
        if not os.path.exists("tfidf_vectorizer.pkl"):
            print("ERROR: tfidf_vectorizer.pkl not found!")
            return False
            
        print("Loading vectorizer...")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✓ Vectorizer loaded successfully")
        
        print("Loading neural network model (this may take a moment)...")
        import tensorflow as tf
        import signal
        
        # Suppress TensorFlow warnings for cleaner output
        tf.get_logger().setLevel('ERROR')
        
        # Set a timeout for model loading
        def timeout_handler(signum, frame):
            raise TimeoutError("Model loading timed out")
        
        # Set 60 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            # Try loading Keras native format first (recommended)
            if os.path.exists("toxic_model_nn.keras"):
                print("Loading from Keras native format (.keras)...")
                model = load_model("toxic_model_nn.keras")
                print("✓ Model loaded from .keras format")
            elif os.path.exists("toxic_model_nn.h5"):
                print("Loading from H5 format (.h5)...")
                model = load_model("toxic_model_nn.h5", compile=False)
                # Recompile the model for H5 format
                model.compile(loss='binary_crossentropy', 
                             optimizer='adam', 
                             metrics=['accuracy'])
                print("✓ Model loaded from .h5 format and recompiled")
            else:
                print("ERROR: No model file found! Please run trainModel.py first.")
                print("Looking for: toxic_model_nn.keras or toxic_model_nn.h5")
                return False
        except TimeoutError:
            print("ERROR: Model loading timed out after 60 seconds")
            return False
        finally:
            signal.alarm(0)  # Cancel the alarm
        
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        model = None
        vectorizer = None
        return False

# Load models at startup
print("=" * 50)
print("BIAS PREDICTOR MODEL - WEBAPP STARTING")
print("=" * 50)
print("Starting server immediately... Models will load in background.")
print("=" * 50)

# Start with models not loaded, they'll load in background
models_loaded = False
model = None
vectorizer = None

# Load models in background
import threading
def load_models_background():
    global model, vectorizer, models_loaded
    models_loaded = load_models()
    if models_loaded:
        print("✓ Models loaded successfully in background!")
    else:
        print("✗ Failed to load models in background")

# Start background loading
threading.Thread(target=load_models_background, daemon=True).start()

# Clean text function (same as training)
def clean_text(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# Toxicity labels (only accurate categories)
labels = ['toxic', 'obscene', 'insult']

def analyze_comment(comment, confidence_threshold=0.5):
    """Analyze a single comment for toxicity"""
    if not model or not vectorizer:
        return {"error": "Model not loaded. Please restart the application or check model files."}
    
    cleaned = clean_text(comment)
    
    # Vectorize and predict
    X = vectorizer.transform([cleaned])
    X_dense = X.toarray()
    pred_prob_all = model.predict(X_dense, verbose=0)[0]
    
    # Only use the accurate categories (indices 0, 2, 4 from original model)
    pred_prob = pred_prob_all[[0, 2, 4]]  # toxic, obscene, insult
    pred_binary = (pred_prob > confidence_threshold).astype(int)
    
    # Process results
    detected_labels = []
    all_scores = {}
    
    for label, prob, binary in zip(labels, pred_prob, pred_binary):
        all_scores[label] = {
            'probability': float(prob),
            'flagged': bool(binary)
        }
        if binary == 1:
            detected_labels.append({'label': label, 'confidence': float(prob)})
    
    # Risk assessment
    max_score = float(max(pred_prob))
    if max_score < 0.2:
        risk_level = "LOW"
    elif max_score < 0.5:
        risk_level = "MEDIUM"
    elif max_score < 0.8:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"
    
    return {
        "comment": comment,
        "is_toxic": len(detected_labels) > 0,
        "detected_labels": detected_labels,
        "all_scores": all_scores,
        "risk_level": risk_level,
        "max_score": max_score
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return "Server is working! Models loaded: " + str(model is not None and vectorizer is not None)

@app.route('/analyze', methods=['POST'])
def analyze():
    comment = request.form.get('comment', '').strip()
    confidence_threshold = float(request.form.get('threshold', 0.5))
    
    if not comment:
        return render_template('index.html', error="Please enter a comment to analyze")
    
    try:
        result = analyze_comment(comment, confidence_threshold)
        if "error" in result:
            return render_template('index.html', error=result["error"])
        return render_template('index.html', result=result, threshold=confidence_threshold)
    except Exception as e:
        return render_template('index.html', error=f"Analysis error: {str(e)}")

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic access"""
    data = request.get_json()
    comment = data.get('comment', '').strip()
    confidence_threshold = data.get('threshold', 0.5)
    
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    
    try:
        result = analyze_comment(comment, confidence_threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """Entry point for the bias-predictor command"""
    print("🚀 Starting Flask server...")
    print("📱 Access the webapp at: http://localhost:5000")
    print("📱 Or from another device at: http://0.0.0.0:5000")
    if not models_loaded:
        print("⚠️  WARNING: Models failed to load - predictions won't work!")
        print("   The webapp will still start, but you'll see error messages when trying to analyze text.")
    print("=" * 50)
    try:
        # For production (Render), use environment port
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == '__main__':
    main() 