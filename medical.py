import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv('project-2/healthcare_dataset.csv', header=0)
print(df.head())
print(df.columns)




columns_to_encode = ['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition',
                     'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
                     'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date',
                     'Medication', 'Test Results']


label_encoders = {col: LabelEncoder() for col in columns_to_encode}


for column in columns_to_encode:
    df[column] = label_encoders[column].fit_transform(df[column])


print(df.head())
print(df.columns)


X = df.drop('Medical Condition', axis=1)
y = df['Medical Condition']




smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"""**Train a Classification Model with Hyperparameter Tuning**"""

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Define a parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with StratifiedKFold
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=stratified_kfold, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

"""**Evaluate the Model**"""

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

"""**Making Predictions with New Data**"""

# Example of making predictions with new data
new_data = {
    'Name': 'Bobby JacksOn',
    
}

# Encode the new data
new_data_encoded = {}
for column, value in new_data.items():
    if column in label_encoders:
        new_data_encoded[column] = label_encoders[column].transform([value])[0]
    else:
        new_data_encoded[column] = value

# Convert the encoded new data into a DataFrame with the same columns as X_train
new_data_df = pd.DataFrame([new_data_encoded], columns=X.columns)

# Make a prediction
prediction = best_model.predict(new_data_df)
# Decode the prediction
prediction_decoded = label_encoders['Medical Condition'].inverse_transform(prediction)
print(f'Prediction: {prediction_decoded[0]}')
