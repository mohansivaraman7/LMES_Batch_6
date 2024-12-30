import pickle
from sklearn.linear_model import LinearRegression

# Train your Linear Regression model
import pandas

data = {"Year":[2000,2001,2002,2003,2004,2005],"Price":[1000,2000,3000,4000,5000,6000]}

"""since above data is normal data, we have to convert it into dataframe for our model usage"""
df = pandas.DataFrame(data)

x = df[["Year"]] #x data should be alwyas given double square braces
y = df["Price"] #y data should be alwauys in single braces , since it will only one column

"""below is the snytax for model creation"""
linear_data_model = LinearRegression()
linear_data_model.fit(x,y)


# Save the model
with open("linear_regression_model.pkl", "wb") as obj:
    pickle.dump(linear_data_model, obj)

