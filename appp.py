from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from cassandra.cluster import Cluster
import uuid

app = Flask(__name__)

# Load your diabetes dataset or replace it with your data
diabetes = pd.read_csv('diabetes.csv')

# Use a smaller subset of data for experimentation (adjust as needed)
diabetes_subset = diabetes.sample(frac=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_subset.loc[:, diabetes_subset.columns != 'Outcome'],
    diabetes_subset['Outcome'],
    stratify=diabetes_subset['Outcome'],
    random_state=66
)

# Apply SMOTE to the training data only
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create individual classifiers and a scaler
scaler = MinMaxScaler()
rf = RandomForestClassifier(random_state=0, n_jobs=-1)  # Use all available cores for training
dt = DecisionTreeClassifier(random_state=0)
svm = SVC(probability=True, random_state=0)
ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)

# Scale the resampled training data and testing data
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest with a reduced search space
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train_resampled)

# Get the best Random Forest model from the search
best_rf_model = grid_search.best_estimator_

# Create a VotingClassifier with the best models
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('dt', dt),
    ('svm', svm),
    ('ann', ann)
], voting='hard')

# Train the VotingClassifier
voting_clf.fit(X_train_scaled, y_train_resampled)

# Use fuzzy logic for post-processing or refining predictions
def apply_fuzzy_logic(predictions, threshold=80):
    fuzzy_predictions = []

    for prediction in predictions:
        # Apply fuzzy matching with a threshold
        similarity = fuzz.ratio(str(prediction), 'Your_Target_Label')
        if similarity >= threshold:
            fuzzy_predictions.append('Your_Target_Label')
        else:
            fuzzy_predictions.append(prediction)

    return fuzzy_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree_function']),
            float(request.form['age'])
        ]
        input_data_reshaped = np.asarray(input_data).reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = voting_clf.predict(std_data)[0]

        # Connect to Cassandra
        cluster = Cluster(['127.0.0.1'])
        session = cluster.connect()

        # Use your existing keyspace
        session.set_keyspace('diabetes1')

        # Save the prediction to Cassandra
        id = uuid.uuid4()
        input_data_str = ','.join(map(str, input_data))  # Convert input_data to a comma-separated string
        prediction_str = 'Positive' if prediction == 1 else 'Negative'  # Convert prediction to a string
        session.execute(
            "INSERT INTO predictions (id, input_data, prediction) VALUES (%s, %s, %s);",
            (id, input_data_str, prediction_str)
        )

        # Close the Cassandra session and cluster
        session.shutdown()
        cluster.shutdown()

        return render_template('result copy.html', prediction=prediction_str)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
