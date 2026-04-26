import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load drilling data from CSV file"""
    data_path = os.path.join("data", "drilling_data.csv")
    print(f"Loading data from {data_path}...")
    return pd.read_csv(data_path)

def prepare_data(df):
    """Prepare features and target for training"""
    # Separate features and target
    X = df.drop("quality", axis=1)
    y = df["quality"]
    
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def train_model(X_train, y_train):
    """Train a Random Forest classifier"""
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Convert predictions back to original labels for reporting
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    return accuracy

def save_model(model, label_encoder):
    """Save the trained model and label encoder"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "drilling_model.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    
    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    
    # Save label encoder
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_path}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("Drilling Quality Prediction - Model Training")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    print(f"Classes: {df['quality'].unique()}")
    
    # Prepare data
    X, y, label_encoder = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    save_model(model, label_encoder)
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
