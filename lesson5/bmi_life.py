from sklearn.linear_model import LinearRegression
import pandas as pd

# read data
bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

# train
bmi_life_model = LinearRegression()
bmi_life_model.fit(x_values, y_values)

# check
laos_life_exp = bmi_life_model.predict([[21.07931]])
