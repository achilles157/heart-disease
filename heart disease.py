import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np

# Read dataset from CSV file
df = pd.read_csv('heart_disease_uci.csv')

# Drop columns that are not required
data = df.drop(columns=['id'])

# Apply one-hot encoding with pd.get_dummies()
data = pd.get_dummies(data)

# Drop rows with NaN values
dc = data.dropna()

# Split dataset into features (X) and target (y)
X = dc.drop('num', axis=1)
y = dc['num']

# Create a scaler object
scaler = StandardScaler()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert X_test into a DataFrame
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters for Random Forest: {best_params}")

# Use the best parameters to train the model
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

# Step tambahan: Menggunakan SMOTE untuk menyeimbangkan data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Retrain the Random Forest model with balanced data
best_rf_model.fit(X_train_balanced, y_train_balanced)
best_rf_predictions = best_rf_model.predict(X_test)


# Evaluasi model Random Forest setelah balanccing
rf_accuracy = accuracy_score(y_test, best_rf_predictions)
rf_report = classification_report(y_test, best_rf_predictions, zero_division=1)
rf_conf_matrix = confusion_matrix(y_test, best_rf_predictions)

print(f'Random Forest Accuracy (with SMOTE): {rf_accuracy * 100:.2f}%')
print(rf_report)
print("Confusion Matrix:\n", rf_conf_matrix)

# Save the tuned Random Forest model
joblib.dump(best_rf_model, 'best_random_forest_model_balanced.pkl')

# Create model Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Combine the predictions from the Logistic Regression and the tuned Random Forest models
results_df = pd.DataFrame({'Logistic_Predictions': logistic_predictions, 'Random_Forest_Predictions': best_rf_predictions})

# Save the predictions to a CSV file
results_df.to_csv('model_predictions.csv', index=False)

# Combine the predictions as additional features for the Hybrid model
hybrid = pd.concat([X_test_df, pd.DataFrame({'Logistic_Predictions': logistic_predictions, 'Random_Forest_Predictions': best_rf_predictions})], axis=1)

# Remove rows with NaN values
hybrid_features = hybrid.dropna()

# Create Hybrid model (using Logistic Regression)
hybrid_model = LogisticRegression()
hybrid_model.fit(hybrid_features, y_test)
hybrid_predictions = hybrid_model.predict(hybrid_features)

# Melakukan prediksi dengan model Hybrid
hybrid_predictions = hybrid_model.predict(hybrid_features)

# Save the Hybrid model
joblib.dump(hybrid_model, 'hybrid_model.pkl')

# Evaluasi model Hybrid
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
hybrid_report = classification_report(y_test, hybrid_predictions, zero_division=1)
hybrid_conf_matrix = confusion_matrix(y_test, hybrid_predictions)

print(f'Hybrid Model Accuracy (with SMOTE): {hybrid_accuracy * 100:.2f}%')
print(hybrid_report)
print("Confusion Matrix:\n", hybrid_conf_matrix)

# Save the evaluation results to a text file
with open('hybrid_model_report_balanced.txt', 'w') as f:
    f.write(f'Akurasi Model Hybrid (with SMOTE): {hybrid_accuracy * 100:.2f}%\n\n')
    f.write(hybrid_report)
    f.write(f"\nConfusion Matrix:\n{hybrid_conf_matrix}")
