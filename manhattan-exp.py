import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv")

#prints first 5 rows of the dataframe
print(df.head())

#split into training (to fit data) & testing (to test model!)
#need 2 brackets when multiple columns from dataframe

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]
y = df["rent"]

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=6)

print(x_train, x_test, y_train, y_test)

# this is the model!
mlr = LinearRegression()

# finds the slopes and the intercept value
mlr.fit(x_train,y_train)

# takes values calculated by .fit() and the x test values, plugs them into the multiple linear regression equation, and calculates the predicted y values
#formula for mlr: y = m1x1 + m2x2 + m3x3 + ... + b
y_predict = mlr.predict(x_test)


my_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(my_apartment)
print(f"the price of the house is {predict}")

