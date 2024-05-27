import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Load data
file_path = '/mnt/data/pc.csv'
data = pd.read_csv(file_path)

# Preprocessing
categorical_columns = ['FirstName', 'LastName', 'ADEmail', 'BusinessUnit', 'EmployeeStatus', 'State', 'GenderCode', 'MaritalStatus', 'SSN']
numerical_columns = ['EmpID']

# Define the preprocessing steps for categorical and numerical columns
categorical_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessor, numerical_columns),
        ('cat', categorical_preprocessor, categorical_columns)
    ])

# Generate dummy labels for each row (cyclically assign column names as labels)
dummy_labels = [data.columns[i % len(data.columns)] for i in range(data.shape[0])]

# Encode labels to numerical values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(dummy_labels)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)
rf_predictions = rf_model.predict(X_test_preprocessed)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_preprocessed, y_train)
xgb_predictions = xgb_model.predict(X_test_preprocessed)

# Combine predictions (simple averaging ensemble)
final_predictions = (rf_predictions + xgb_predictions) // 2

# Evaluate the model
accuracy = accuracy_score(y_test, final_predictions)
print(f'Ensemble Model Test Accuracy: {accuracy:.2f}')