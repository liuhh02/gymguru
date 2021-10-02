model_imports = {
    "Linear Regression": "from sklearn.linear import LinearRegression",
    "Decision Tree": "from sklearn.tree import DecisionTreeRegressor",
    "Random Forest": "from sklearn.ensemble import RandomForestRegressor",
    "Gradient Boosting": "from sklearn.ensemble import GradientBoostingRegressor",
    "Neural Network": "from sklearn.neural_network import MLPRegressor",
    "k-Nearest Neighbors": "from sklearn.neighbors import KNeighborsRegressor",
}


model_urls = {
    "Linear Regression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
    "Decision Tree": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
    "Random Forest": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    "Gradient Boosting": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
    "Neural Network": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html",
    "k-Nearest Neighbors": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
}


model_infos = {
    "Linear Regression": """
        - Fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation
        - Computationally fast and interpretable by design
    """,
    "Decision Tree": """
        - Simple to understand and intrepret
        - Prone to overfitting when they are deep (high variance)
    """,
    "Random Forest": """
        - Consists of multiple decision trees voting together
        - Have a lower risk of overfitting compared to decision trees
        - More robust to outliers
        - Computationally intensive on large datasets 
        - Not as easily interpretable as decision trees
    """,
    "Gradient Boosting": """
        - Combines decision trees in an additive fashion from the start
        - Builds one tree at a time sequentially
        - If carefully tuned, gradient boosting can result in better performance than random forests
    """,
    "Neural Network": """
        - Have great representational power but may overfit on small datasets if not properly regularized
        - Have many parameters that require fine-tuning
        - Computationally intensive on large datasets
    """,
    "k-Nearest Neighbors": """
        - Intuitive and simple
        - Become very slow as the dataset size grows
    """,
}