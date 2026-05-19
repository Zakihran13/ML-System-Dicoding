import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random

# 1. Data preparation
data = load_iris()
# Convert to MLflow dataset for input tracking
logdata = mlflow.data.from_numpy(data.data, source="data.csv")
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

input_example = X_train[0:5]

# 2. Set Tracking URI (Ensure 'mlflow ui' is running in terminal)
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# 3. Enable Autologging before the run starts
mlflow.autolog()

# 4. Start tracking
with mlflow.start_run():
    
    # Log the input dataset specifically
    mlflow.log_input(logdata, context="training")
    
    n_estimators = random.randint(10, 100) # Adjusted range for practical RF training
    max_depth = random.randint(1, 50)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    # Train model - Autolog will automatically capture parameters, metrics, and the model here
    model.fit(X_train, y_train)
    
    # Evaluate model - Autolog will capture this score
    accuracy = model.score(X_test, y_test)
    
    print(f"Run completed with Accuracy: {accuracy}")