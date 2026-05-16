import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """# Deep Learning Diabetes Prediction Project
This notebook contains the complete training pipeline for the Artificial Neural Network (ANN) that predicts diabetes based on medical records.
"""

code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle"""

text_2 = """## 1. Data Loading and Preprocessing
We handle missing values by replacing zeroes with NaNs and filling them with the median of each respective column."""

code_2 = """# Load dataset
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv("diabetes.csv", header=None, names=columns)

# Handle missing/invalid zero values
medical_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[medical_cols] = df[medical_cols].replace(0, np.nan)
df[medical_cols] = df[medical_cols].fillna(df[medical_cols].median())

df.head()"""

text_3 = """## 2. Train-Test Split and Standardization"""

code_3 = """X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for the Streamlit app
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)"""

text_4 = """## 3. Building the ANN Model"""

code_4 = """model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(12, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()"""

text_5 = """## 4. Training the Model"""

code_5 = """history = model.fit(
    X_train_scaled, y_train, 
    epochs=150, 
    batch_size=10, 
    validation_split=0.2, 
    verbose=1
)"""

text_6 = """## 5. Evaluation and Metrics"""

code_6 = """# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate on test set
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(e)"""

text_7 = """## 6. Saving the Model"""

code_7 = """model.save("diabetes_model.keras")
print("Model saved as diabetes_model.keras")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(text_7),
    nbf.v4.new_code_cell(code_7)
]

with open('diabetes_model_training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
