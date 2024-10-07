import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

class LinearRegressionModel:
    def __init__(self, db_path, query, predictors, target):
        self.db_path = db_path
        self.query = query
        self.predictors = predictors
        self.target = target
        self.model = LinearRegression()
        self.df = self.load_data()

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(self.query, conn)
        conn.close()
        return df
    
    def train_model(self):
        X = self.df[self.predictors]
        y = self.df[self.target]
        self.model.fit(X, y)
        self.df['Predicted_Price'] = self.model.predict(X)
    
    def evaluate_model(self):
        X = self.df[self.predictors]
        y = self.df[self.target]
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2
    
    def plot_predictions(self):
        y = self.df[self.target]
        predictions = self.df['Predicted_Price']
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, edgecolors=(0, 0, 0))
        plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=4)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Prices')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.ticklabel_format(style='plain', axis='y')
        
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.ticklabel_format(style='plain', axis='x')
        plt.show()
    
    def get_coefficients(self):
        return pd.DataFrame(self.model.coef_, self.predictors, columns=['Coefficient'])


