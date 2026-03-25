import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

def main():
    print("Loading healthcare dataset...")
    # 1. Load dataset (using breast cancer dataset as a proxy for healthcare classification tasks)
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Simulate saving raw data
    raw_data = X.copy()
    raw_data['target'] = y
    
    # 2. Clean and preprocess
    print("Preprocessing data...")
    # Check for missing values (none in this sklearn dataset, but simulated logic here)
    X.fillna(X.mean(), inplace=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaled numerical data for MATLAB (Module 4) PCA analysis
    X_scaled_df.to_csv('data/healthcare_data_scaled.csv', index=False)
    print("Saved scaled features to data/healthcare_data_scaled.csv for MATLAB PCA.")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    # 3. Train models
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # 4. Evaluate
    print("\n--- Logistic Regression Results ---")
    print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
    print(classification_report(y_test, lr_pred, target_names=data.target_names))
    
    print("\n--- Random Forest Results ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(classification_report(y_test, rf_pred, target_names=data.target_names))
    
    # Visualize feature importances from RF
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Random Forest Feature Importances")
    plt.bar(range(10), importances[indices][:10], align="center")
    plt.xticks(range(10), X.columns[indices][:10], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/module3_rf_importances.png')
    print("Saved visualization to output/module3_rf_importances.png")

if __name__ == "__main__":
    main()
