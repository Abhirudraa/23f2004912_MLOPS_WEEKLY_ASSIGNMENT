import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os

def train_model(data_path, model_path):
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics = pd.DataFrame({'accuracy': [accuracy]})
    metrics.to_csv('metrics.csv', index=False)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = "models/model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    train_model(data_path, model_path)
