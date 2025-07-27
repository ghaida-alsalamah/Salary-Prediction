import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load dataset
data = pd.read_csv('Salary_Data.csv')

# Explore basic info
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Histogram of Salary
plt.figure(figsize=(8, 5))
plt.hist(data['Salary'], bins=5, color='skyblue', edgecolor='black')
plt.title('Salary Distribution')
plt.grid(axis='y')
plt.xlabel('Salary ($)')
plt.ylabel('Number of Employees')
plt.show()

# Histogram of Years of Experience
plt.figure(figsize=(8, 5))
plt.hist(data['YearsExperience'], bins=5, color='skyblue', edgecolor='black')
plt.title('Years of Experience Distribution')
plt.grid(axis='y')
plt.xlabel('Years of Experience')
plt.ylabel('Number of Employees')
plt.show()

# Boxplot of Salary
plt.figure(figsize=(8, 5))
sns.boxplot(y='Salary', data=data, color='skyblue')
plt.title('Salary Boxplot')
plt.grid(axis='y')
plt.show()

# Regression line with scatter (Seaborn)
plt.figure(figsize=(8, 5))
sns.regplot(x='YearsExperience', y='Salary', data=data, color='skyblue', line_kws={'color': 'red'})
plt.title('Experience VS Salary (Seaborn Regression)')
plt.grid()
plt.show()

# Scatter Plot of all data
plt.figure(figsize=(8, 5))
plt.scatter(data['YearsExperience'], data['Salary'], color='skyblue')
plt.title('Experience VS Salary (Scatter Only)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.grid()
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.title('Correlation Matrix')
plt.show()

# Feature selection
X = data[['YearsExperience']]
y = data['Salary']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Linear Regression Metrics
MAE = mean_absolute_error(y_test, predictions)
MSE = mean_squared_error(y_test, predictions)
RMSE = np.sqrt(MSE)
r2 = r2_score(y_test, predictions)

print("Linear Regression Results")
print("MAE:", MAE)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R2 Score:", r2)

# Polynomial Regression
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(x_train, y_train)
predictions_poly = poly_model.predict(x_test)

# Polynomial Regression Metrics
MAE_poly = mean_absolute_error(y_test, predictions_poly)
MSE_poly = mean_squared_error(y_test, predictions_poly)
RMSE_poly = np.sqrt(MSE_poly)
r2_poly = r2_score(y_test, predictions_poly)

print("\nPolynomial Regression Results")
print("MAE:", MAE_poly)
print("MSE:", MSE_poly)
print("RMSE:", RMSE_poly)
print("R2 Score:", r2_poly)

# Plot Linear Regression (Training)
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='skyblue', label='Training data')  # Training data scatter
plt.plot(x_train, model.predict(x_train), color='red', label='Regression line')  # Linear line
plt.title('Experience VS Salary (Linear - Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()

# Plot Polynomial Regression (Training)
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='skyblue', label='Training data')  # Training data scatter
x_sorted_train = np.sort(x_train, axis=0)
plt.plot(x_sorted_train, poly_model.predict(x_sorted_train), color='red', label='Regression line')  # Polynomial curve
plt.title('Experience VS Salary (Polynomial - Training)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()

# Plot Linear Regression (Testing)
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color='skyblue', label='Testing data')  # Testing data scatter
plt.plot(x_test, predictions, color='red', label='Regression line')  # Linear line
plt.title('Experience VS Salary (Linear - Testing)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()

# Plot Polynomial Regression (Testing)
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color='skyblue', label='Testing data')  # Testing data scatter
x_sorted_test = np.sort(x_test, axis=0)
plt.plot(x_sorted_test, poly_model.predict(x_sorted_test), color='red', label='Regression line')  # Polynomial curve
plt.title('Experience VS Salary (Polynomial - Testing)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.grid()
plt.show()