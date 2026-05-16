import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve
from model_utils import load_and_preprocess_data
from train_model import train_and_evaluate

# --- Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "diabetes_model.keras"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "diabetes.csv"

# --- Styling ---
st.markdown("""
<style>
    .main-title { font-size: 40px; font-weight: bold; color: #2C3E50; }
    .sub-title { font-size: 24px; color: #34495E; margin-bottom: 20px; }
    .footer { text-align: center; color: grey; margin-top: 50px; font-size: 14px; }
    .prediction-card { padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px; }
    .risk-high { background-color: #FADBD8; color: #C0392B; border: 2px solid #E74C3C; }
    .risk-low { background-color: #D5F5E3; color: #27AE60; border: 2px solid #2ECC71; }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        _, _, _, _, _, df = load_and_preprocess_data(DATA_PATH)
        return df
    return None

df = get_data()

# --- Navigation ---
st.sidebar.title("🩺 Navigation")
menu = ["Home", "Data Overview", "Model Training", "Prediction", "Batch Prediction"]
choice = st.sidebar.radio("Select a module:", menu)

# --- Resource Loading ---
@st.cache_resource
def load_ml_resources():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

# --- Pages ---

if choice == "Home":
    st.markdown('<p class="main-title">Deep Learning Diabetes Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">An end-to-end ANN project for medical diagnosis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Diabetes Prediction App**! 
    This application uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict the likelihood of a patient having diabetes based on their medical records.
    
    ### Project Objectives
    1. **Data Preprocessing**: Handle missing zero values in medical features (Glucose, Blood Pressure, BMI, etc.) using median imputation.
    2. **Model Architecture**: Train a Sequential Neural Network with optimized layers and activation functions.
    3. **Interactive Demo**: Provide an easy-to-use interface for single predictions and batch processing.
    
    Use the sidebar to navigate through the data, view the training dashboard, or test out the prediction engine!
    """)
    
elif choice == "Data Overview":
    st.markdown('<p class="main-title">📊 Data Overview</p>', unsafe_allow_html=True)
    
    if df is not None:
        st.write("Dataset: **Pima Indians Diabetes Database** (Preprocessed)")
        
        st.subheader("Data Sample")
        st.dataframe(df.head(15), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
        with col2:
            st.subheader("Outcome Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x="Outcome", data=df, palette="viridis", ax=ax)
            ax.set_xticklabels(["Not Diabetic (0)", "Diabetic (1)"])
            ax.set_ylabel("Count")
            st.pyplot(fig)
    else:
        st.error(f"Dataset not found at `{DATA_PATH}`.")

elif choice == "Model Training":
    st.markdown('<p class="main-title">⚙️ Model Training Dashboard</p>', unsafe_allow_html=True)
    st.write("Train the Artificial Neural Network on the dataset and visualize its performance metrics.")
    
    if st.button("🚀 Train Model Now", use_container_width=True):
        with st.spinner("Training ANN model... This may take a moment."):
            if not os.path.exists(DATA_PATH):
                st.error("Dataset not found!")
            else:
                model, history, scaler, metrics = train_and_evaluate(DATA_PATH, MODEL_PATH, SCALER_PATH)
                X_test, y_test, y_pred, y_pred_prob, roc_auc, cm = metrics
                
                st.success("Model trained and saved successfully!")
                
                # Metrics Dashboard
                st.markdown("### Training Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
                    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
                    ax1.set_title("Model Accuracy")
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig1)
                    
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(history.history['loss'], label='Train Loss', color='red')
                    ax2.plot(history.history['val_loss'], label='Val Loss', color='green')
                    ax2.set_title("Model Loss")
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    ax2.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig2)
                    
                st.markdown("### Evaluation Metrics")
                st.info(f"**ROC-AUC Score:** {roc_auc:.4f}")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Confusion Matrix**")
                    fig3, ax3 = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                               xticklabels=["Not Diabetic", "Diabetic"], 
                               yticklabels=["Not Diabetic", "Diabetic"])
                    st.pyplot(fig3)
                    
                with col4:
                    st.markdown("**ROC Curve**")
                    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                    fig4, ax4 = plt.subplots(figsize=(5, 4))
                    ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC area = {roc_auc:.2f}')
                    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax4.set_xlabel('False Positive Rate')
                    ax4.set_ylabel('True Positive Rate')
                    ax4.set_title('Receiver Operating Characteristic')
                    ax4.legend(loc="lower right")
                    ax4.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig4)

elif choice == "Prediction":
    st.markdown('<p class="main-title">🔮 Patient Prediction</p>', unsafe_allow_html=True)
    st.write("Enter the patient's medical details below to assess their diabetes risk.")
    
    model, scaler = load_ml_resources()
    if model is None or scaler is None:
        st.warning("⚠️ Model or Scaler not found! Please go to the 'Model Training' tab and train the model first.")
    else:
            
        with st.form("prediction_form"):
            st.markdown("### Patient Diagnostics")
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1,
                                             help="Number of times pregnant")
                glucose = st.number_input("Glucose", min_value=1.0, max_value=300.0, value=120.0,
                                         help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=1.0, max_value=200.0, value=70.0,
                                                help="Diastolic blood pressure (mm Hg)")
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0,
                                                help="Triceps skin fold thickness (mm)")
                
            with col2:
                insulin = st.number_input("Insulin (IU/mL)", min_value=1.0, max_value=1000.0, value=79.0,
                                         help="2-Hour serum insulin (mu U/ml)")
                bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=32.0,
                                     help="Body mass index (weight in kg/(height in m)^2)")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.01, max_value=3.0, value=0.5,
                                     help="Diabetes pedigree function")
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=33, step=1)
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit = st.form_submit_button("Run Diagnostic Prediction", use_container_width=True)
            
            if submit:
                # Prepare data
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
                input_scaled = scaler.transform(input_data)
                
                # Predict
                prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
                prediction_class = 1 if prediction_prob > 0.5 else 0
                risk_percentage = prediction_prob * 100
                
                # Display Result
                st.markdown("---")
                if prediction_class == 1:
                    st.markdown(f"""
                    <div class="prediction-card risk-high">
                        <h2>⚠️ High Risk: Diabetic (Class 1)</h2>
                        <h4>Diabetes Risk Probability: {risk_percentage:.1f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card risk-low">
                        <h2>✅ Low Risk: Not Diabetic (Class 0)</h2>
                        <h4>Diabetes Risk Probability: {risk_percentage:.1f}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.progress(float(prediction_prob))

elif choice == "Batch Prediction":
    st.markdown('<p class="main-title">📁 Batch Prediction</p>', unsafe_allow_html=True)
    st.write("Upload a CSV file containing multiple patient records for bulk prediction.")
    
    model, scaler = load_ml_resources()
    if model is None or scaler is None:
        st.warning("⚠️ Model or Scaler not found! Please train the model first.")
    else:
        st.info("The uploaded CSV should contain the 8 features in order: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.")
        uploaded_file = st.file_uploader("Upload Patient Data (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Assuming no header or similar header
                test_df = pd.read_csv(uploaded_file)
                st.write("### Data Preview")
                st.dataframe(test_df.head(), use_container_width=True)
                
                if st.button("Run Batch Prediction", use_container_width=True):
                    
                    if len(test_df.columns) >= 8:
                        # Extract first 8 columns
                        X_batch = test_df.iloc[:, :8].values
                        
                        # Note: User may upload raw data with 0s that need imputation,
                        # but for simplicity in batch prediction we'll assume they upload ready-to-scale data.
                        
                        X_batch_scaled = scaler.transform(X_batch)
                        probs = model.predict(X_batch_scaled, verbose=0)
                        preds = (probs > 0.5).astype(int).flatten()
                        
                        result_df = test_df.copy()
                        result_df["Predicted_Class"] = preds
                        result_df["Probability (%)"] = np.round(probs.flatten() * 100, 2)
                        
                        st.markdown("### Prediction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download Button
                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="batch_predictions_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"Expected at least 8 columns. Found {len(test_df.columns)}.")
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

# --- Footer ---
st.markdown('<p class="footer">Deep Learning Diabetes Prediction App | Developed with Streamlit & TensorFlow/Keras</p>', unsafe_allow_html=True)
