import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Read dataset from CSV file (make sure to download the dataset before running the code)
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

# Create model Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

# Create model Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100)
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)

# Combine the predictions from the Logistic Regression and Random Forest models
results_df = pd.DataFrame({'Logistic_Predictions': logistic_predictions, 'Random_Forest_Predictions': random_forest_predictions})

# Combine the predictions as additional features for the Hybrid model
hybrid = pd.concat([X_test_df, results_df], axis=1)

# Remove rows with NaN values
hybrid_features = hybrid.dropna()

# Buat model Hybrid (contoh menggunakan Logistic Regression)
hybrid_model = LogisticRegression()
hybrid_model.fit(hybrid_features, y_test)

# Melakukan prediksi dengan model Hybrid
hybrid_predictions = hybrid_model.predict(hybrid_features)

# Evaluasi model Hybrid, misalnya dengan akurasi
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, hybrid_predictions)
print(f'Akurasi Model Hybrid: {accuracy * 100:.2f}%')

# Evaluasi model Hybrid
from sklearn.metrics import accuracy_score, classification_report
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)
hybrid_report = classification_report(y_test, hybrid_predictions)

print("Akurasi Model Hybrid:", hybrid_accuracy)
print(hybrid_report)