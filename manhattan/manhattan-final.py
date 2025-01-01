import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv")

#prints first 5 rows of the dataframe
# print(df.head())

#split into training (to fit data) & testing (to test model!)
#need 2 brackets when multiple columns from dataframe

x = df[["size_sqft"]]
y = df["rent"]

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=6)

# print(x_train, x_test, y_train, y_test)

# this is the model!
mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)

results = pd.DataFrame(x_test)
results['y_test'] = y_test.values
results['y_predict'] = y_predict

# changing x values to 1D array
sns.scatterplot(x=x_test.squeeze(), y=y_test, label="actual data", color="red")
sns.lineplot(x=x_test.squeeze(), y=y_predict, label="predicted data", color="blue")
plt.xlabel("Size sqft")
plt.xlabel("Rent")
plt.title("Actual Rent vs Predicted Rent")
plt.show()
