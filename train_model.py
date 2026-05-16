import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from model_utils import load_and_preprocess_data, build_model

def train_and_evaluate(filepath="diabetes.csv", model_path="diabetes_model.keras", scaler_path="scaler.pkl"):
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, _ = load_and_preprocess_data(filepath)
    
    print("Building model...")
    model = build_model(input_dim=X_train.shape[1])
    
    print("Training model...")
    history = model.fit(
        X_train, y_train, 
        epochs=150, 
        batch_size=10, 
        validation_split=0.2, 
        verbose=0
    )
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Predict probabilities and classes
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        roc_auc = None
        print(f"Could not calculate ROC-AUC: {e}")
        
    print(f"\nSaving model to {model_path} and scaler to {scaler_path}...")
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    return model, history, scaler, (X_test, y_test, y_pred, y_pred_prob, roc_auc, cm)

if __name__ == "__main__":
    train_and_evaluate()
