import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    features = data.drop(columns=['Speed_Motor_Activation', 'Torque_Motor_Activation', 'Performance_Score'])
    labels = data[['Speed_Motor_Activation', 'Torque_Motor_Activation']]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
def train_ensemble_model(X_train, y_train):
    model_speed = GradientBoostingClassifier()
    model_torque = RandomForestClassifier()

    model_speed.fit(X_train, y_train['Speed_Motor_Activation'])
    model_torque.fit(X_train, y_train['Torque_Motor_Activation'])

    return model_speed, model_torque
def evaluate_models(model_speed, model_torque, X_test, y_test):
    pred_speed = model_speed.predict(X_test)
    pred_torque = model_torque.predict(X_test)

    print("Speed Motor Prediction:")
    print(classification_report(y_test['Speed_Motor_Activation'], pred_speed))

    print("\nTorque Motor Prediction:")
    print(classification_report(y_test['Torque_Motor_Activation'], pred_torque))

    return pred_speed, pred_torque
