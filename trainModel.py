import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

def train_and_save_model():
    """Train and save the model - can be called from other modules"""
    # Load and clean data
    df = pd.read_csv("dataset.csv")

    def clean_text(text):
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.lower()

    df['cleaned_text'] = df['comment_text'].apply(clean_text)

    # Features and labels
    X = df['cleaned_text']
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_vectorized = vectorizer.fit_transform(X)

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Convert sparse matrix to dense for Keras
    X_train = X_train.toarray()
    X_val = X_val.toarray()

    # Neural network model (reduced complexity to prevent overfitting)
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))  # 6 labels

    # Add early stopping to prevent overfitting
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model with early stopping and lower learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    model.fit(X_train, y_train.values, epochs=20, batch_size=128, 
              validation_data=(X_val, y_val.values), callbacks=[early_stopping], verbose=1)

    # Save model and vectorizer
    print("Saving neural network model...")
    # Save in Keras native format (recommended for Keras 3)
    model.save("toxic_model_nn.keras")
    print("Neural network model saved as toxic_model_nn.keras (Keras native format)")

    # Also save as H5 for backward compatibility
    try:
        model.save("toxic_model_nn.h5")
        print("Backup H5 model saved as toxic_model_nn.h5")
    except Exception as e:
        print(f"H5 save failed (this is OK): {e}")

    print("Saving TF-IDF vectorizer...")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Vectorizer saved as tfidf_vectorizer.pkl")

    # Predict and report
    print("Generating predictions for evaluation...")
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_binary, target_names=y.columns))

    print("\nTraining completed! Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
