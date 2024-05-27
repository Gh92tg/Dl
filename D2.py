import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the CSV file
file_path = 'path_to_your_file/date.csv'
df = pd.read_csv(file_path)

# Convert dates to datetime objects
df['HireDate'] = pd.to_datetime(df['HireDate'], format='%d-%m-%Y', errors='coerce')
df['TerminationDate'] = pd.to_datetime(df['TerminationDate'], format='%d-%m-%Y', errors='coerce')
df['BirthDate'] = pd.to_datetime(df['BirthDate'], format='%d-%m-%Y', errors='coerce')

# Melt the dataframe to long format
melted_df = df.melt(var_name='Column', value_name='Date').dropna()

# Extract features from the dates
melted_df['Year'] = melted_df['Date'].dt.year
melted_df['Month'] = melted_df['Date'].dt.month
melted_df['Day'] = melted_df['Date'].dt.day
melted_df['DayOfWeek'] = melted_df['Date'].dt.dayofweek
melted_df['DayOfYear'] = melted_df['Date'].dt.dayofyear
melted_df['IsLeapYear'] = melted_df['Date'].dt.is_leap_year.astype(int)

# Encode the labels
label_encoder = LabelEncoder()
melted_df['ColumnEncoded'] = label_encoder.fit_transform(melted_df['Column'])

# Prepare features and labels
X = melted_df[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'IsLeapYear']]
y = to_categorical(melted_df['ColumnEncoded'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the deep learning model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=0)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Model Accuracy: {accuracy}")
