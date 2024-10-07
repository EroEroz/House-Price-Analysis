import pandas as pd
class RealEstateStats:
    def __init__(self, df):
        self.df = df

    def describe_price(self):
        pd.options.display.float_format = '{:,.2f}'.format
        return self.df['Price'].describe()

    def mean_price(self):
        return self.df['Price'].mean()

    def median_price(self):
        return self.df['Price'].median()

    def std_price(self):
        return self.df['Price'].std()

    def min_price(self):
        return self.df['Price'].min()

    def max_price(self):
        return self.df['Price'].max()

    def percentile_25(self):
        return self.df['Price'].quantile(0.25)

    def percentile_75(self):
        return self.df['Price'].quantile(0.75)

    def coefficient_of_variation(self):
        return self.std_price() / self.mean_price()

    def count_by_area(self):
        return self.df['Location'].value_counts()

    def outlier_analysis(self):
        q1 = self.df['Price'].quantile(0.25)
        q3 = self.df['Price'].quantile(0.75)
        iqr = q3 - q1
        return self.df[(self.df['Price'] < (q1 - 1.5 * iqr)) | (self.df['Price'] > (q3 + 1.5 * iqr))]

    def price_frequency(self):
        return self.df['Price'].value_counts().sort_index()

    def correlation_price_area(self):
        if 'Area' in self.df.columns:
            return self.df['Price'].corr(self.df['Area'])
        else:
            return "No 'Area' column in DataFrame"
        