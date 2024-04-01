import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

file_path = "train_data.csv"
df = pd.read_csv(file_path)
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%b-%y')
df = df.sort_values(by='DATE')
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
df['MONTH'] = df['DATE'].dt.month
df['DAY'] = df['DATE'].dt.day
df['WITHDRAWAL_AMT'] = df[' WITHDRAWAL AMT '].replace({',': ''}, regex=True).astype(float)

sequences = df[['MONTH', 'DAY', 'WITHDRAWAL_AMT']].values
scaler = MinMaxScaler()
sequences = scaler.fit_transform(sequences)

scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)

df['LABEL'] = np.where(df['WITHDRAWAL_AMT'] > 1000000, 1, 0)
labels = df['LABEL'].values
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("fraud_detection_model.h5", save_best_only=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=2)

model.save("fraud_detection_model_final.h5")

_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f' % (accuracy * 100))
