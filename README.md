# Toxicity Detection System

A machine learning web application that detects toxic, obscene, and insulting content in text using a neural network model. Built with TensorFlow/Keras and Flask.

## ⚠️ Python Version Requirement

**This project requires Python 3.12** for TensorFlow compatibility. Python 3.13 is not yet supported by TensorFlow.

- ✅ **Python 3.12** - Fully supported
- ❌ **Python 3.13** - Not supported (TensorFlow compatibility issues)
- ❌ **Python 3.11 or earlier** - May work but not tested

## Dataset

**Important**: The training dataset is required for model training. Download it from:
- [Get It Here](https://drive.google.com/file/d/1TMSVTKbGcVHXjsqss9jgLrmlMuNJZ30W/view?usp=sharing)

### How to Add the Dataset:
1. Download the dataset file from the link above
2. Place the `dataset.csv` file in the root directory of this project
3. The file should be in the same folder as `trainModel.py`
4. Ensure the file is named exactly `dataset.csv`

**File Structure After Adding Dataset:**
```
BiasPredictorModel/
├── dataset.csv           # ← Place the downloaded file here
├── trainModel.py
├── app.py
└── ... (other files)
```

## Overview

This system analyzes text input and provides toxicity scores across three categories:
- **Toxic**: General toxic content
- **Obscene**: Profane or vulgar language  
- **Insult**: Personal attacks or insults

The model achieves 97%+ accuracy and provides detailed confidence scores and risk assessments.

## Architecture

### Neural Network Model
- **Type**: Dense Neural Network (Multi-layer Perceptron)
- **Input**: TF-IDF vectorized text features
- **Layers**: 
  - Dense layers with ReLU activation
  - Dropout layers for regularization
  - Final sigmoid activation for binary classification
- **Output**: 3 binary classifiers (toxic, obscene, insult)

### Text Processing Pipeline
1. **Text Cleaning**: Remove URLs, special characters, normalize whitespace
2. **TF-IDF Vectorization**: Convert text to numerical features
3. **Neural Network Prediction**: Generate toxicity probabilities
4. **Threshold Filtering**: Apply confidence threshold to determine flags
5. **Risk Assessment**: Categorize overall risk level

## Performance Metrics

- **Training Accuracy**: 97.68% (Epoch 10)
- **Validation Accuracy**: 99.41%
- **Training Loss**: 0.0443
- **Validation Loss**: 0.0581

The model shows excellent generalization with high validation accuracy and low overfitting.

## Technologies & Libraries

### Core ML Libraries
- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural network API
- **scikit-learn**: TF-IDF vectorization and model evaluation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation

### Web Framework
- **Flask**: Lightweight web server
- **Jinja2**: HTML templating

### Data Processing
- **re**: Regular expressions for text cleaning
- **joblib**: Model serialization

## Project Structure

```
BiasPredictorModel/
├── app.py                 # Flask web application
├── trainModel.py          # Model training script
├── dataset.csv           # Training dataset (required for training)
├── templates/
│   └── index.html        # Web interface
├── pyproject.toml        # Python project configuration
├── render.yaml           # Render deployment config
└── .gitignore           # Git ignore rules
```

# Generated files (after training):
├── toxic_model_nn.keras  # Trained neural network model
└── tfidf_vectorizer.pkl  # TF-IDF vectorizer
```

## Quick Start

### 1. Environment Setup

**Option A: Using Python 3.12 (Recommended)**

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies (using pyproject.toml)
pip install -e .

# Or install with development dependencies:
# pip install -e ".[dev]"
```

**Option B: Direct Python 3.12 Installation (Windows)**

If you have Python 3.12 installed in a specific location:

```bash
# Install directly with Python 3.12
& "C:\Users\yourusername\AppData\Local\Programs\Python\Python312\python.exe" -m pip install -e .

# Run the application
& "C:\Users\yourusername\AppData\Local\Programs\Python\Python312\python.exe" app.py
```

**Option C: Virtual Environment with Python 3.12 (Windows)**

```bash
# Create virtual environment with specific Python version
& "C:\Users\yourusername\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv312

# Activate it
venv312\Scripts\activate

# Install the project
pip install -e .

# Run the app
python app.py
```

### 2. Train the Model

Since pre-trained models are not included, you must train the model first:

```bash
# Using Python 3.12
python trainModel.py

# Or with specific Python installation (Windows)
& "C:\Users\yourusername\AppData\Local\Programs\Python\Python312\python.exe" trainModel.py
```

**Training Details:**
- Takes 10-20 minutes depending on your computer
- Requires at least 4GB RAM
- Saves `toxic_model_nn.keras` and `tfidf_vectorizer.pkl`
- Shows progress with accuracy metrics

### 3. Run the Web Application

```bash
python app.py
```

The server will start at:
- **Local**: http://127.0.0.1:5000
- **Network**: http://0.0.0.0:5000

### 4. Use the Application

1. Open your browser to `http://127.0.0.1:5000`
2. Enter text in the input field
3. Adjust confidence threshold (0.0-1.0)
4. Click "Analyze" to get toxicity scores

## Configuration

### Confidence Threshold
- **0.2-0.4**: Sensitive detection (flags potential issues)
- **0.5**: Balanced approach (default)
- **0.7-0.9**: Strict moderation (only obvious toxicity)

### Model Loading
The app automatically loads models in this order:
1. `toxic_model_nn.keras` (Keras 3 native format)
2. `tfidf_vectorizer.pkl` (TF-IDF vectorizer)

**Important**: You must train the model first using `python trainModel.py` before running the web application.

## Model Details

### Training Process
- **Dataset**: Toxic comment classification dataset
- **Text Cleaning**: Remove URLs, special characters, normalize case
- **Feature Extraction**: TF-IDF vectorization (max 10,000 features)
- **Architecture**: 3-layer dense network with dropout
- **Optimization**: Adam optimizer, binary crossentropy loss
- **Training**: 20 epochs with early stopping

### Neural Network Architecture
```
Input Layer: TF-IDF features (10,000 dimensions)
    ↓
Dense Layer 1: 512 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 2: 256 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 3: 128 units + ReLU + Dropout(0.3)
    ↓
Output Layer: 3 units + Sigmoid (toxic, obscene, insult)
```

## API Usage

### REST API Endpoint

```bash
POST /api/analyze
Content-Type: application/json

{
    "comment": "Your text here",
    "threshold": 0.5
}
```

### Response Format

```json
{
    "comment": "Your text here",
    "is_toxic": true,
    "detected_labels": [
        {"label": "toxic", "confidence": 0.85}
    ],
    "all_scores": {
        "toxic": {"probability": 0.85, "flagged": true},
        "obscene": {"probability": 0.12, "flagged": false},
        "insult": {"probability": 0.23, "flagged": false}
    },
    "risk_level": "HIGH",
    "max_score": 0.85
}
```

## Use Cases

- **Content Moderation**: Filter toxic comments on platforms
- **Social Media**: Detect harmful content in posts
- **Customer Support**: Flag inappropriate messages
- **Educational Platforms**: Monitor student interactions
- **Online Communities**: Maintain healthy discussion environments

## Risk Levels

- **LOW**: Max score < 0.2
- **MEDIUM**: Max score 0.2-0.5
- **HIGH**: Max score 0.5-0.8
- **VERY HIGH**: Max score > 0.8

## Error Handling

The application includes robust error handling for:
- Missing model files
- Invalid input text
- Model loading failures
- Network connectivity issues

## Project Configuration

This project uses `pyproject.toml` for modern Python project management, providing:

### ✅ **Benefits of pyproject.toml**
- **Reproducible builds** - Exact dependency versions for everyone
- **Python version enforcement** - Automatically requires Python 3.12
- **Professional packaging** - Modern Python standards (PEP 517/518)
- **Development tools** - Optional testing, linting, and formatting tools
- **Easy installation** - One command: `pip install -e .`

### 📦 **Dependencies**
- **Python 3.12** (required for TensorFlow compatibility)
- **TensorFlow 2.16.1** - Deep learning framework
- **Flask 3.1.1** - Web framework
- **scikit-learn 1.4.0** - Machine learning utilities
- **pandas 2.2.0** - Data manipulation
- **numpy 1.26.4** - Numerical computations
- **joblib 1.3.2** - Model serialization
- **gunicorn 21.2.0** - Production WSGI server

### 🛠️ **Development Tools** (Optional)
Install with `pip install -e ".[dev]"` to get:
- **pytest** - Testing framework
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking

See `pyproject.toml` for complete configuration and exact versions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

---

**Note**: This system is designed for educational and research purposes. Always review flagged content manually for critical applications. 
