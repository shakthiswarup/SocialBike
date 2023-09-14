import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import glob
from datetime import timedelta
import tensorflow as tf

def read_merged_data_files(root_dir):
    merged_data_files = glob.glob(os.path.join(root_dir, '**', 'merged_data.csv'), recursive=True)
    return merged_data_files

# Data Collection
root_dir = r"C:\Cycling Data\0002"
merged_data_files = read_merged_data_files(root_dir)

dataset_list = []
for file in merged_data_files:
    dataset = pd.read_csv(file)
    dataset['Time'] = pd.to_datetime(dataset['Time'], format='%H:%M:%S')  # Specifying the format
    dataset_list.append(dataset)

dataset = pd.concat(dataset_list)

scaler = MinMaxScaler()
dataset[['HR averages', 'WL averages']] = scaler.fit_transform(dataset[['HR averages', 'WL averages']])

# Splitting into input (X) and output (y) sequences
lookback = 5
X, y, timestamps = [], [], []
for i in range(len(dataset) - lookback):
    X.append(dataset[['HR averages', 'WL averages']].values[i:i+lookback])
    y.append(dataset[['HR averages', 'WL averages']].values[i+lookback])
    timestamps.append(dataset['Time'].values[i+lookback])

X, y, timestamps = np.array(X), np.array(y), np.array(timestamps)

# Model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(lookback, 2)))
model.add(Dense(2))

# Modify the loss function to mean squared error between input and output
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Model Training
model.compile(loss=custom_loss, optimizer='adam')

# Modify target data to match the input shape
y_target = X[:, -1, :]  # Use the last timestamp of each sequence as target data
model.fit(X, y_target, epochs=50, batch_size=32)

# Maximum Heart Rate and Threshold Limits
max_heart_rate = dataset['HR averages'].max()
ideal_heart_rate = max_heart_rate * 0.9  # 90% of maximum heart rate

session_data = []

# Fetch the input data for prediction
input_data = X[-1].reshape(1, lookback, 2)  # Adjusted reshape to (1, lookback, 2)

# Set the time interval in seconds (5 seconds)
time_interval = timedelta(seconds=5)

max_iterations = 100  # Maximum number of iterations for the prediction loop
iterations = 0

previous_workload = input_data[0, -1, 1]  # Get the previous workload value

while iterations < max_iterations:
    # Predict and scale back the data
    predicted_data = model.predict(input_data)
    predicted_data = scaler.inverse_transform(predicted_data)
    predicted_heart_rate, predicted_workload = predicted_data[0]

    # Append the predicted values to the session_data list
    session_data.append([timestamps[-1], predicted_heart_rate, predicted_workload])

    # Update the input_data for the next prediction
    input_data = np.roll(input_data, -1, axis=1)
    input_data[0, -1, :] = [predicted_heart_rate, predicted_workload]

    # Modify workload if predicted heart rate is higher than ideal heart rate
    if predicted_heart_rate > ideal_heart_rate:
        difference = np.random.uniform(-2, 2)  # Randomly select a value between -5 and 5
        input_data[0, -1, 1] += difference

    # Check for NaN or infinite values in the predicted data
    if np.isnan(predicted_heart_rate) or np.isnan(predicted_workload) or np.isinf(predicted_heart_rate) or np.isinf(predicted_workload):
        break

    iterations += 1

# Create a DataFrame with the session data
session_df = pd.DataFrame(session_data, columns=['Timestamp', 'Predicted Heart Rate', 'Predicted Workload'])
session_df['Timestamp'] = session_df['Timestamp'].dt.strftime('%m-%d-%Y %H:%M:%S')
session_df = session_df.round({'Predicted Heart Rate': 5, 'Predicted Workload': 5})

# Save the session data to a CSV file
session_df.to_csv('session_data4.csv', index=False)
