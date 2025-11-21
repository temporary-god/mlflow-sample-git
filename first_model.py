"""
Sample Classifier Model with MLflow
Demonstrates MLflow tracking, logging, and model registry
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Set MLflow tracking URI (optional - defaults to local ./mlruns)
mlflow.set_tracking_uri("http://10.0.11.179:5001")

# Set experiment name
mlflow.set_experiment("iris_classification_v2")

def train_model(n_estimators=100, max_depth=5, random_state=42, register_model=False):
    """
    Train a Random Forest classifier with MLflow tracking
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees
        random_state: Random seed for reproducibility
        register_model: Whether to register model in MLflow Model Registry
    """
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_classifier"):
        
        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log feature importances (sanitize feature names for MLflow)
        for i, importance in enumerate(model.feature_importances_):
            # Replace invalid characters with underscores
            safe_name = iris.feature_names[i].replace('(', '').replace(')', '').replace(' ', '_')
            mlflow.log_metric(f"feature_importance_{safe_name}", importance)
        
        # Log model (register only if requested)
        if register_model:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="iris_random_forest"
            )
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Log dataset info as tags
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return model, mlflow.active_run().info.run_id


def load_and_predict(run_id, sample_data=None):
    """
    Load a logged model and make predictions
    
    Args:
        run_id: MLflow run ID
        sample_data: Optional sample data for prediction
    """
    
    # Load model
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    # Use sample data or create default
    if sample_data is None:
        sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    # Make prediction
    prediction = loaded_model.predict(sample_data)
    
    print(f"\nPrediction for sample {sample_data}: {prediction}")
    return prediction


def compare_runs():
    """
    Compare multiple runs with different hyperparameters
    """
    
    configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
    ]
    
    print("Training multiple models with different configurations...\n")
    
    for config in configs:
        print(f"Training with config: {config}")
        train_model(**config)
        print("-" * 50)


if __name__ == "__main__":
    # Train a single model (without registering)
    print("Training single model...")
    model, run_id = train_model(n_estimators=100, max_depth=5, register_model=False)
    
    # Load and test the model
    print("\nLoading and testing model...")
    load_and_predict(run_id)
    
    # Uncomment to compare multiple runs
    # print("\n" + "="*50)
    # compare_runs()
    
    # When you find the best model, register it:
    # model, run_id = train_model(n_estimators=100, max_depth=5, register_model=True)
    
    print("\n" + "="*50)
    print("To view results in MLflow UI, navigate to:")
    print("http://10.0.11.179:5001")