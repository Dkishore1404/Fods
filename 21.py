# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'weight': [150, 130, 120, 160, 140, 130],
    'color': ['red', 'green', 'yellow', 'red', 'green', 'yellow'],
    'texture': ['smooth', 'rough', 'smooth', 'rough', 'smooth', 'rough'],
    'type': ['apple', 'orange', 'banana', 'apple', 'orange', 'banana']
}

df = pd.DataFrame(data)

# Separate features and target variable
X = df[['weight', 'color', 'texture']]
y = df['type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Use ColumnTransformer to apply different preprocessing steps to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['weight']),
        ('cat', OneHotEncoder(), ['color', 'texture'])
    ])

# Create a kNN classifier pipeline
knn_classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3))  # You can adjust 'n_neighbors' here
])

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
