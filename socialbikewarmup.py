import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
dataset = pd.read_csv(r'C:\Users\mandalapu\PycharmProjects\Cyclingdatao2\merged_data.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])  # Convert timestamp column to datetime
dataset = dataset[['Timestamp', 'HR averages', 'WL averages']].copy()

# Step 1: Warm-up data
warmup_data = np.array([[warmup_hr_1, warmup_wl_1],
                        [warmup_hr_2, warmup_wl_2],
                        ...
                        [warmup_hr_n, warmup_wl_n]])

# Step 2: ML with data from Warm-up
step2_data = np.array([[step2_hr_1, step2_wl_1],
                       [step2_hr_2, step2_wl_2],
                       ...
                       [step2_hr_n, step2_wl_n]])

# Step 3: ML with data from Warm-up + Step 2
step3_data = np.array([[step3_hr_1, step3_wl_1],
                       [step3_hr_2, step3_wl_2],
                       ...
                       [step3_hr_n, step3_wl_n]])

# Step 4: ML with data from Warm-up + Step 2 + Step 3
step4_data = np.array([[step4_hr_1, step4_wl_1],
                       [step4_hr_2, step4_wl_2],
                       ...
                       [step4_hr_n, step4_wl_n]])

# Step 5: ML with data from Warm-up + Step 2 + Step 3 + Step 4
step5_data = np.array([[step5_hr_1, step5_wl_1],
                       [step5_hr_2, step5_wl_2],
                       ...
                       [step5_hr_n, step5_wl_n]])

# Concatenate all the data for training
train_data = np.concatenate((warmup_data, step2_data, step3_data, step4_data, step5_data))

# Prepare the input and target data
input_data = train_data[:, :-1]  # HR and WL inputs
target_data = train_data[:, -1]  # Target WL

# Reshape the input data for LSTM (samples, timesteps, features)
input_data = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, input_data.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(input_data, target_data, epochs=100, batch_size=32, verbose=0)

# Predict HR and WL for a new data point
new_data = np.array([[new_hr, new_wl]])
new_data = new_data.reshape((1, 1, new_data.shape[1]))
prediction = model.predict(new_data)

print("Predicted WL:", prediction)
