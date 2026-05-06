from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def get_model(model_name):
    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "logistic_regression": LogisticRegression(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }
    return models[model_name]

def train_and_evaluate(model_name, X_train, y_train, X_test, y_test):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    score = f1_score(y_test, predictions)
    
    return model, score
