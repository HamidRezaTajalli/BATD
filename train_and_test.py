import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

def main():
    X, y = fetch_covtype(return_X_y=True)
    
    # Keep the same random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Increased iterations, adjusted learning_rate, etc.
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=1,
        verbose=100  # Show logs every 100 iterations
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
