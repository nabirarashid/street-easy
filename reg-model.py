#we're using multiple linear regression
#we have multiple factors that can affect the price of a house

#imports!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

streeteasy = pd.read_csv("streeteasy.csv")
df = pd.DataFrame(streeteasy)

x = df["size_sqft","building_age_yrs"]
y = df["rent"]

#train_test_split is used to split the data into training (80%) and testing data (20%)
#outputs are x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, test_size = 0.2, randome_state = 6)