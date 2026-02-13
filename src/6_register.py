from mlflow.tracking import MlflowClient
import mlflow


client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

run_id = "8f5c867668ec430e9f5460fe1dbf3e83"
model_path = "model"  # This should match the 'name' parameter used in log_model()
model_name = 'water_potability_model'

model_uri = f"runs:/{run_id}/{model_path}"
reg = mlflow.register_model(model_uri, model_name)