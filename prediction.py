from json import load
import mlflow.pyfunc
import pandas as pd
import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.DataFrame({
    'ph': 3.716,
    'Hardness': 204.890,
    'Solids': 20791.0,
    'Chloramines': 7.0,
    'Sulfate': 300.0,
    'Conductivity': 500.0,
    'Organic_carbon': 10.0,
    'Trihalomethanes': 80.0,
    'Turbidity': 5.0
}, index=[0])

logger_model = 'runs:/8f5c867668ec430e9f5460fe1dbf3e83/model'

loaded_model = mlflow.pyfunc.load_model(logger_model)
loaded_model.predict(data)
print("Prediction is ", loaded_model.predict(data))