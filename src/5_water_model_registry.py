from matplotlib import axis
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment("water_exp_model_registry")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv('data/water_potability.csv')
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

# X_train = train_processed_data.iloc[:,0:-1].values
# y_train = train_processed_data.iloc[:,-1].values

X_train = train_processed_data.drop(columns=['Potability'], axis= 1)
y_train = train_processed_data['Potability']

n_estimators = 1000

rf  = RandomForestClassifier(n_estimators=n_estimators)

param_dist = {
    'n_estimators': [100, 200, 500, 820, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5,]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)

with mlflow.start_run(run_name="RF_model_registry_3") as parent_run:
    random_search.fit(X_train, y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"RandomForest_Run_{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])

    print("Best Hyperparameters:", random_search.best_params_)

    mlflow.log_params(random_search.best_params_)
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    pickle.dump(best_rf,open("model.pkl","wb"))

    # X_test = test_processed_data.iloc[:,0:-1].values
    # y_test = test_processed_data.iloc[:,-1].values

    X_test = test_processed_data.drop(columns=['Potability'], axis= 1)
    y_test = test_processed_data['Potability']

    model = pickle.load(open('model.pkl',"rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_score = f1_score(y_test,y_pred)

    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score",f1_score)

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train_data")
    mlflow.log_input(test_df, "test_data")

    mlflow.log_artifact(__file__)
    signature = infer_signature(train_processed_data.drop(columns=['Potability'], axis= 1), best_rf.predict(train_processed_data.drop(columns=['Potability'], axis= 1)))
    
    mlflow.sklearn.log_model(best_rf, name="model", signature=signature)


    print("Accuracy",acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score",f1_score)