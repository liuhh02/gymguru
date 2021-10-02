import streamlit as st
from models.utils import model_imports, model_infos, model_urls
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

from models.paramselector import (
    lr_params, dt_params, 
    rf_params, gb_params,
    nn_params, knn_params)

def model_selector():
    model_training_container = st.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "k-Nearest Neighbors",
                "Linear Regression"
            ),
        )

        if model_type == "Linear Regression":
            model = lr_params()
            
        elif model_type == "Decision Tree":
            model = dt_params()
            
        elif model_type == "Random Forest":
            model = rf_params()

        elif model_type == "Gradient Boosting":
            model = gb_params()
            
        elif model_type == "Neural Network":
            model = nn_params()
            
        elif model_type == "k-Nearest Neighbors":
            model = knn_params()
            
    return model_type, model

def generate_snippet(model, model_type):
    model_import = model_imports[model_type]
    model_text_rep = repr(model)
    snippet = f"""
    >>> # Import necessary libraries
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> df = pd.read_csv('./data.csv')
    >>> data = df.values
    >>> # X columns contain data incuding whether the chosen date is during the start of
    >>> # the semester / during school breaks, as well as whether it is a weekday etc. to
    >>> # help predict occupancy levels
    >>> X = data[:, 1:]
    >>> # Y column contains the crowd level at the corresponding point in time 
    >>> # (which is what we want to predict)
    >>> y = data[:, 0]
    >>> # Split dataset into training & test so we can evaluate the model performance
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    >>> {model_import}
    >>> model = {model_text_rep}
    >>> model.fit(X_train, y_train)

    >>> train_score = model.score(X_train, y_train)
    >>> test_score = model.score(X_test, y_test)
    """
    return snippet

def load_dataset():
    df = pd.read_csv('./data.csv')
    df = df.drop("date", axis=1)
    df = df.drop("is_holiday", axis=1)
    data = df.values
    X = data[:, 1:]
    y = data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_model(model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - t0

    train_score = np.round(model.score(X_train, y_train), 3)

    test_score = np.round(model.score(X_test, y_test), 3)


    return model, train_score, test_score, duration

def get_model_info(model_type):
    model_tips = model_infos[model_type]
    return model_tips

def get_model_url(model_type):
    model_url = model_urls[model_type]
    text = f"**Link to scikit-learn official documentation [here]({model_url}) 💻 **"
    return text

def testmodel(model):
    model_testing_container = st.expander("Test my model!", True)
    with model_testing_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "k-Nearest Neighbors",
                "Linear Regression"
            ),
        )

        if model_type == "Linear Regression":
            model = lr_param_selector()
            
        elif model_type == "Decision Tree":
            model = dt_param_selector()
            
        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()
            
        elif model_type == "Neural Network":
            model = nn_param_selector()
            
        elif model_type == "k-Nearest Neighbors":
            model = knn_param_selector()
            
    return model_type, model

