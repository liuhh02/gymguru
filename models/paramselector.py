import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

def lr_param_selector():
    model = LinearRegression()
    return model

def dt_param_selector():
    #max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", ["auto", "sqrt", "log2"])

    params = {
        #"max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeRegressor(**params)
    return model

def rf_param_selector():

    n_estimators = st.number_input("n_estimators", 50, 300, 100, 50)
    #max_depth = st.number_input("max_depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("min_samples_split", 1, 20, 2, 1)
    max_features = st.selectbox("max_features", ["auto", "sqrt", "log2"])

    params = {
        "n_estimators": n_estimators,
        #"max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    return model

def gb_param_selector():
    learning_rate = st.slider("learning_rate", 0.001, 0.5, 0.1, 0.005)
    n_estimators = st.number_input("n_estimators", 10, 500, 100, 10)
    max_depth = st.number_input("max_depth", 3, 30, 3, 1)

    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    model = GradientBoostingRegressor(**params)
    return model

def nn_param_selector():
    number_hidden_layers = st.number_input("number of hidden layers", 1, 5, 1)

    hidden_layer_sizes = []

    for i in range(number_hidden_layers):
        n_neurons = st.number_input(
            f"Number of neurons at layer {i+1}", 2, 200, 100, 25
        )
        hidden_layer_sizes.append(n_neurons)

    hidden_layer_sizes = tuple(hidden_layer_sizes)
    params = {"hidden_layer_sizes": hidden_layer_sizes}

    model = MLPRegressor(**params)
    return model

def knn_param_selector():

    n_neighbors = st.number_input("n_neighbors", 5, 20, 5, 1)
    weights = st.selectbox(
        "weights", ("uniform", "distance")
    )
    metric = st.selectbox(
        "metric", ("minkowski", "euclidean", "manhattan", "chebyshev", "mahalanobis")
    )

    params = {"n_neighbors": n_neighbors, "weights": weights, "metric": metric}

    model = KNeighborsRegressor(**params)
    return model