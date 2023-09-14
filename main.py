import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import timedelta
import glob
import os

# Data Collection
def read_merged_data_files(root_dir):
    merged_data_files = glob.glob(os.path.join(root_dir, '**', 'merged_data.csv'), recursive=True)
    return merged_data_files

# Load the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=1))  # Input dimension should be 1 for workload
model.add(Dense(1))  # Output layer with 1 neuron (predicted heart rate)

# Function to predict heart rates for a new session
def predict_heart_rates(model):
    #the time interval (5 seconds) and session duration in total is (15 minutes)
    time_interval = timedelta(seconds=5)
    session_duration = timedelta(minutes=15)

    # Initializing variables for time and workload
    current_time = timedelta(seconds=0)
    initial_workload = 30
    workload_increase_rate = 10
    current_workload = initial_workload

    # lists to store timestamps, workload values, and predicted heart rates
    timestamps = []
    workload_values = []
    predicted_hr_values = []

    while current_time < session_duration:
        # Appending the current timestamp and workload value
        timestamps.append(current_time)
        workload_values.append(current_workload)

        # Creating input sequence for prediction
        X_pred = np.array([current_workload]).reshape(1, 1)

        # Predicting heart rate for the current workload
        predicted_hr = model.predict(X_pred).flatten()
        predicted_hr_values.append(predicted_hr[0])

        # Updating the time and workload based on intervals
        current_time += time_interval
        if current_time.total_seconds() % (2 * 60) == 0:
            current_workload += workload_increase_rate

    # DataFrame with predicted output including timestamps and workload
    output_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Workload': workload_values,
        'Predicted Heart Rate': predicted_hr_values
    })

    return output_df

# Data Collection
root_dir = r"C:\Cycling Data\0002"
merged_data_files = read_merged_data_files(root_dir)

# Loop through merged data files and predict heart rates
for file in merged_data_files:
    dataset = pd.read_csv(file)

    # Train the model on the dataset
    X_train = dataset['WL averages'].values.reshape(-1, 1)
    y_train = dataset['HR averages'].values
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10)

    # Predict heart rates for a new session
    predicted_output = predict_heart_rates(model)

    # Saving predicted output to a CSV file
    output_filename = "prediction-heartrate10.csv"
    predicted_output.to_csv(output_filename, index=False)
