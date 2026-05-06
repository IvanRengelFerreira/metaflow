from metaflow import FlowSpec, step

from preprocess import load_churn_data, split_and_scale_data
from model import train_and_evaluate

class ChurnPredictionFlow(FlowSpec):

    @step
    def start(self):
        X, y = load_churn_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_and_scale_data(X, y)
        
        self.model_names = ["random_forest", "logistic_regression", "gradient_boosting"]
        self.next(self.train_models, foreach="model_names")

    @step
    def train_models(self):
        self.model_name = self.input
        
        self.model, self.score = train_and_evaluate(
            self.model_name, 
            self.X_train, 
            self.y_train, 
            self.X_test, 
            self.y_test
        )
        
        self.next(self.join_models)

    @step
    def join_models(self, inputs):
        self.results = [(inp.model_name, inp.score, inp.model) for inp in inputs]
        self.best_result = max(self.results, key=lambda x: x[1])
        
        self.best_model_name = self.best_result[0]
        self.best_score = self.best_result[1]
        self.best_model = self.best_result[2]
        
        self.next(self.end)

    @step
    def end(self):
        print(f"Best model: {self.best_model_name} with F1-Score: {self.best_score:.4f}")

if __name__ == "__main__":
    ChurnPredictionFlow()
