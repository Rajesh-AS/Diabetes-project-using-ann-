# Deep Learning Diabetes Prediction App 🩺

An end-to-end Machine Learning pipeline and interactive web application for predicting diabetes risk based on patient medical records. The application uses an Artificial Neural Network (ANN) built with TensorFlow/Keras and provides an intuitive, dynamic interface via Streamlit.

## Features

- **Data Preprocessing**: Automatically handles invalid zero values in critical medical features (Glucose, Blood Pressure, BMI, etc.) by using median imputation.
- **Model Architecture**: A multi-layer Artificial Neural Network (Sequential) utilizing `relu` and `sigmoid` activation functions, optimized with the Adam optimizer for binary classification.
- **Interactive Web Interface**:
  - **Data Overview**: Visualize the dataset structure, statistical summaries, and outcome distributions.
  - **Model Training Dashboard**: Train the model dynamically from the UI, visualize loss/accuracy graphs, view the confusion matrix, and analyze ROC-AUC curves.
  - **Patient Prediction**: Input individual patient metrics to get a real-time risk assessment.
  - **Batch Prediction**: Upload a CSV file of multiple patients for bulk diagnostic processing and download the results.

## Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) / Keras
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning Utilities**: Scikit-Learn (Scaling, Metrics)
- **Data Visualization**: Matplotlib, Seaborn

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rajesh-AS/Diabetes-project-using-ann-.git
   cd Diabetes-project-using-ann-
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**:
   ```bash
   python -m streamlit run app.py
   ```
   *(The application will open automatically in your browser at `http://localhost:8501`)*

## Project Structure

- `app.py`: The main Streamlit web application.
- `train_model.py`: Script to train the model and save the outputs (`.keras` model and `.pkl` scaler).
- `model_utils.py`: Utility functions for building the ANN and preprocessing data.
- `create_nb.py`: Helper script that generates a Jupyter Notebook (`diabetes_model_training.ipynb`) summarizing the full training pipeline.
- `requirements.txt`: Python package dependencies.
- `diabetes.csv`: The Pima Indians Diabetes Database.

## Model Details

- **Input Features**: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- **Layers**: 
  - Input Layer (8 features)
  - Hidden Layer 1 (12 neurons, ReLU)
  - Hidden Layer 2 (8 neurons, ReLU)
  - Output Layer (1 neuron, Sigmoid)

---
*Developed with Streamlit & TensorFlow/Keras*
