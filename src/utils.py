import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """Save an object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {e}", sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate multiple models using GridSearchCV and return R2 scores."""
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {})
            if not para:
                raise CustomException(f"No parameters found for model '{model_name}'", sys)

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"Model: {model_name} | Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}")
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(f"Error during model evaluation: {e}", sys)


def load_object(file_path):
    """Load an object from a file using pickle."""
    try:
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}", sys)
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {e}", sys)
