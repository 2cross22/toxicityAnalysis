# Toxicity Detection System

A machine learning web application that detects toxic, obscene, and insulting content in text using a neural network model. Built with TensorFlow/Keras and Flask.

## Quick Start

### Prerequisites

- **Python 3.12** (required for TensorFlow compatibility)
- At least 4GB RAM for model training
- Internet connection for downloading dependencies

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/toxicity-detection-system.git
cd toxicity-detection-system

# Create virtual environment with Python 3.12
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Step 2: Download Dataset

1. Download the training dataset from [Google Drive](https://drive.google.com/file/d/1TMSVTKbGcVHXjsqss9jgLrmlMuNJZ30W/view?usp=sharing)
2. Place `dataset.csv` in the project root directory (same folder as `trainModel.py`)
3. Ensure the file is named exactly `dataset.csv`

### Step 3: Train the Model

```bash
# Train the neural network model
python trainModel.py
```

**Training Details:**
- Takes 10-20 minutes depending on your computer
- Shows real-time progress with accuracy metrics
- Saves `toxic_model_nn.keras` and `tfidf_vectorizer.pkl`
- Final accuracy: ~97%+ on validation set

### Step 4: Run the Web Application

```bash
# Start the Flask web server
python app.py
```

**Access the application:**
- **Local**: http://127.0.0.1:5000
- **Network**: http://0.0.0.0:5000

### Step 5: Use the Application

1. Open your browser to `http://127.0.0.1:5000`
2. Enter text in the input field
3. Adjust confidence threshold (0.0-1.0)
4. Click "Analyze" to get toxicity scores

## What It Detects

The system analyzes text for three types of harmful content:

- **Toxic**: General toxic or harmful content
- **Obscene**: Profane or vulgar language
- **Insult**: Personal attacks or insults

Each category gets a confidence score (0.0-1.0) and risk assessment.

## System Architecture

### Text Processing Pipeline

```
Input Text → Text Cleaning → TF-IDF Vectorization → Neural Network → Toxicity Scores
```

1. **Text Cleaning**: Remove URLs, special characters, normalize whitespace
2. **TF-IDF Vectorization**: Convert text to 10,000-dimensional feature vector
3. **Neural Network Prediction**: Generate toxicity probabilities
4. **Threshold Filtering**: Apply confidence threshold to determine flags
5. **Risk Assessment**: Categorize overall risk level

### Neural Network Architecture

The model uses a **Multi-Layer Perceptron (MLP)** with the following structure:

```
Input Layer: 10,000 TF-IDF features
    ↓
Dense Layer 1: 512 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 2: 256 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 3: 128 units + ReLU + Dropout(0.3)
    ↓
Output Layer: 3 units + Sigmoid (toxic, obscene, insult)
```

**Key Features:**
- **Dense Layers**: Fully connected layers for feature learning
- **ReLU Activation**: Introduces non-linearity for complex patterns
- **Dropout (0.3)**: Prevents overfitting by randomly deactivating neurons
- **Sigmoid Output**: Produces probabilities between 0 and 1

### Model Performance

- **Training Accuracy**: 97.68% (Epoch 10)
- **Validation Accuracy**: 99.41%
- **Training Loss**: 0.0443
- **Validation Loss**: 0.0581

The model shows excellent generalization with high validation accuracy and minimal overfitting.

## Configuration

### Confidence Thresholds

- **0.2-0.4**: Sensitive detection (flags potential issues)
- **0.5**: Balanced approach (default)
- **0.7-0.9**: Strict moderation (only obvious toxicity)

### Risk Levels

- **LOW**: Max score < 0.2
- **MEDIUM**: Max score 0.2-0.5
- **HIGH**: Max score 0.5-0.8
- **VERY HIGH**: Max score > 0.8

## Technical Details

### Dependencies

This project uses `pyproject.toml` for modern Python project management:

**Core ML Libraries:**
- **TensorFlow 2.16.1**: Deep learning framework
- **Keras 3.10.0**: High-level neural network API
- **scikit-learn 1.4.0**: TF-IDF vectorization and model evaluation
- **NumPy 1.26.4**: Numerical computations
- **Pandas 2.2.0**: Data manipulation

**Web Framework:**
- **Flask 3.1.1**: Lightweight web server
- **Jinja2 3.1.6**: HTML templating

**Utilities:**
- **joblib 1.3.2**: Model serialization
- **gunicorn 21.2.0**: Production WSGI server

### Project Structure

```
toxicity-detection-system/
├── app.py                 # Flask web application
├── trainModel.py          # Model training script
├── dataset.csv           # Training dataset (download separately)
├── templates/
│   └── index.html        # Web interface
├── pyproject.toml        # Project configuration
├── render.yaml           # Render deployment config
└── .gitignore           # Git ignore rules

# Generated files (after training):
├── toxic_model_nn.keras  # Trained neural network model
└── tfidf_vectorizer.pkl  # TF-IDF vectorizer
```

### Training Process

1. **Data Loading**: Load and preprocess the toxic comments dataset
2. **Text Cleaning**: Remove URLs, special characters, normalize case
3. **Feature Extraction**: TF-IDF vectorization with 10,000 features
4. **Model Training**: 20 epochs with early stopping
5. **Model Saving**: Save in Keras native format (.keras)

### API Usage

**REST API Endpoint:**
```bash
POST /api/analyze
Content-Type: application/json

{
    "comment": "Your text here",
    "threshold": 0.5
}
```

**Response Format:**
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

## Development

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes:
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
```

## Error Handling

The application includes robust error handling for:
- Missing model files
- Invalid input text
- Model loading failures
- Network connectivity issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

---

**Important Note**: This system is designed for educational and research purposes. Always review flagged content manually for critical applications. 
