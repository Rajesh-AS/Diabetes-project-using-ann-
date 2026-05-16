import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def load_and_preprocess_data(filepath="diabetes.csv"):
    """
    Loads dataset, handles missing/invalid zeros, and splits into train/test sets.
    """
    # 1. Load dataset with Pima Indians Diabetes column names
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(filepath, header=None, names=columns)
    
    # 2. Handle missing or invalid zero values for medical features
    medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    # Replace zeros with NaN
    df[medical_cols] = df[medical_cols].replace(0, np.nan)
    
    # Impute NaNs with median of the respective columns
    for col in medical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 3. Features and labels
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # 4. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df

def build_model(input_dim=8):
    """
    Builds and compiles the Sequential ANN model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(12, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    # Compile the model
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    
    return model
