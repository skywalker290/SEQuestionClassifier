import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Importing Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# Function to preprocess data (optional scaling for certain models)
def preprocess_data(X, scale=False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X

# Logistic Regression
def logistic_regression(X, y):
    # X = preprocess_data(X, scale=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Decision Tree Classifier
def decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Random Forest Classifier
def random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Support Vector Machine (SVM)
def support_vector_machine(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', probability=True)  # Enable progress output
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# k-Nearest Neighbors (k-NN)
def knn(X, y, k=5):
    # X = preprocess_data(X, scale=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Naïve Bayes
def naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Gradient Boosting Classifier (XGBoost)
def xgboost_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)

# Multi-Layer Perceptron (MLP - Neural Network)
# def mlp_classifier(X, y):
#     # X = preprocess_data(X, scale=True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def mlp_classifier(X, y, epochs=500, lr=0.01):
    # Normalize data without centering (for sparse matrices)
    scaler = StandardScaler(with_mean=False)  # Fix applied here
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = torch.tensor(X_train.toarray(), dtype=torch.float32), torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

    # Define MLP model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.fc3 = nn.Linear(hidden_sizes[1], output_size)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Initialize model
    model = MLP(input_size=X.shape[1], hidden_sizes=[100, 50], output_size=len(set(y))).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

    # Return classification report and accuracy score
    return classification_report(y_test.cpu().numpy(), y_pred), accuracy_score(y_test.cpu().numpy(), y_pred)

# Example usage:
# report, acc = logistic_regression(X, y)
# print("Accuracy:", acc)
# print(report)
