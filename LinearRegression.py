# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your local path after downloading from Kaggle)
# Kaggle dataset link: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
data = pd.read_csv("train.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# Select relevant features (example: square footage, bedrooms, bathrooms)
# For this dataset: 'GrLivArea' = Living area square footage, 'BedroomAbvGr' = Bedrooms, 'FullBath' = Bathrooms
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

# Prepare data
X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot predicted vs actual prices
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color='purple')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Example prediction
sample_house = np.array([[2000, 3, 2]])  # 2000 sqft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(sample_house)
print(f"Predicted Price for 2000 sqft, 3BR, 2BA: ${predicted_price[0]:,.2f}")
