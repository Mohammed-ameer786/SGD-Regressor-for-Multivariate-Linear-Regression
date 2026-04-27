# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset, clean column names, and select features (Size, Bedrooms) with targets (Price, Occupants).
2. Scale the input features using StandardScaler.
3. Train two SGDRegressor models separately for Price and Occupants using the scaled data.
4. Take user input, scale it, and predict house price and occupants using the trained models.

## Program:
```
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("house.csv")

data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]


y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))

```
Developed by: Mohammed Ameer F
RegisterNumber:  212225040248


## Output:
```

<img width="288" height="51" alt="Screenshot 2026-04-27 142245" src="https://github.com/user-attachments/assets/7aa8c3e7-445a-426b-b027-f3fafea94056" />


```
```

<img width="333" height="43" alt="Screenshot 2026-04-27 142254" src="https://github.com/user-attachments/assets/221db889-0124-4ff3-81ab-6cecb219c419" />


```


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
