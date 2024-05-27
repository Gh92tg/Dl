import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
file_path = '/mnt/data/date.csv'
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
y = melted_df['ColumnEncoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

accuracy, report
