import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

def preprocess_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y')
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['MONTH'] = df['DATE'].dt.month
    df['DAY'] = df['DATE'].dt.day
    df['WITHDRAWAL_AMT'] = df[' WITHDRAWAL AMT '].replace({',': ''}, regex=True).astype(float)
    return df[['MONTH', 'DAY', 'WITHDRAWAL_AMT']].values

def predict_fraud(model, data, scaler):
    data = scaler.transform(data)
    data = data.reshape((data.shape[0], 1, data.shape[1]))
    predictions = model.predict(data)
    return predictions.flatten()

model_path = "fraud_detection_model_final.h5"  
model = load_model(model_path)

new_data_path = "dataset.csv" 
new_data = pd.read_csv(new_data_path)

new_data_processed = preprocess_data(new_data)

scaler_path = "scaler.pkl" 
scaler = joblib.load(scaler_path)

predictions = predict_fraud(model, new_data_processed, scaler)

for i, pred in enumerate(predictions):
    print(f"Transaction {i + 1}: Probability of Fraud - {pred:.4f}")
