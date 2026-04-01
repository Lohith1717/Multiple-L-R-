import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Area': [2600, 3000, 3200, 3600, 4000, 2800, 3400, 2000],
    'Bedrooms': [3, 4, None, 3, 5, 4, None, 2],
    'Age': [20, 15, 18, 30, 8, 12, 10, 25],
    'Price': [550000, 565000, 610000, 595000, 760000, 580000, 620000, 500000]
}

df = pd.DataFrame(data)

print(" Original Dataset:\n", df)

df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].mean())

print("\n After Handling Missing Values:\n", df)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n Predictions:", y_pred)
print(" Actual:", y_test.values)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\n Model Details:")
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2, b3):", model.coef_)

new_house = [[2500, 4, 5]]  # Area, Bedrooms, Age
predicted_price = model.predict(new_house)

print("\n Predicted Price for new house:", predicted_price[0])

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()