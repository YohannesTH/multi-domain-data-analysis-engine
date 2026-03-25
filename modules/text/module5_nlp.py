import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

def main():
    print("Loading text dataset (20 newsgroups)...")
    # Using 3 categories to make it run faster
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics']
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    
    print("Applying TF-IDF vectorization...")
    # Keep max features reasonable for exporting to MATLAB clustering later
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    
    X_train_tfidf = vectorizer.fit_transform(data_train.data)
    X_test_tfidf = vectorizer.transform(data_test.data)
    
    y_train = data_train.target
    y_test = data_test.target
    
    # Save a subset of features for MATLAB (Module 6) clustering
    print("Saving TF-IDF features for MATLAB clustering...")
    # Convert sparse matrix to dense
    dense_features = X_train_tfidf.toarray()
    
    # Save as CSV
    df_features = pd.DataFrame(dense_features, columns=vectorizer.get_feature_names_out())
    # Save a small sample (e.g. 500 rows) to keep the CSV file size small
    df_features.head(500).to_csv('data/text_features_tfidf.csv', index=False)
    print("Saved text features to data/text_features_tfidf.csv")
    
    # Train Naive Bayes
    print("Training Naive Bayes classifier...")
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    
    # Predict and evaluate
    pred = clf.predict(X_test_tfidf)
    print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=data_test.target_names))
    
    # Log some top keywords
    feature_names = vectorizer.get_feature_names_out()
    with open('output/module5_text_analysis_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, pred, target_names=data_test.target_names))
        
    print("Saved text analysis report to output/module5_text_analysis_report.txt")

if __name__ == "__main__":
    main()
