Boston Housing Price Prediction: Build a machine learning model that can predict the price of a house in Boston based on various features of the house.
# Step 1: Import packges
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Step 2: Data processing 
!gdown --id 1shBZAw8s7b2jse4rmD8RhJ1WF-5zxfB7

#Lets load the dataset and sample some
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

df.head()
df.shape
df.describe()
df.info()

# Check For Missing Values
info = pd.DataFrame(df.isnull().sum(),columns=["IsNull"])
info.insert(1,"IsNa",df.isna().sum(),True)
info.insert(2,"Duplicate",df.duplicated().sum(),True)
info.insert(3,"Unique",df.nunique(),True)
info.T

# Step 3: Data Visualization
# Heatmap of correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='Reds')
plt.xticks(rotation=90)
plt.show()

# Histogram and Kernel Density Plot for Multiple Variables

for variable in df.columns:
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(df[variable], bins=30, density=True)
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.title(f'Histogram of {variable}')

    # Kernel Density Plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(df[variable])
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.title(f'Kernel Density Plot of {variable}')

    plt.tight_layout()
    plt.show()

# Step 4: LinearRegression model:
X = df.drop('MEDV', axis=1)
y = df['MEDV']

scaler = MinMaxScaler()
X= scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(r2)

# Comparison of Actual vs. Predicted Housing Prices (Linear Regression)
column_names =['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

for idx, feature in enumerate(X_test.T):
    plt.scatter(X_test[:, idx], y_test, color='green', alpha=0.5)
    plt.scatter(X_test[:, idx], y_pred, color='blue', alpha=0.5)
    plt.xlabel(column_names[idx])  # Use the column name as the x-label
    plt.ylabel('MEDV')
    plt.title(f'Linear Regression Model for MEDV Estimation ({column_names[idx]})')
    plt.show()

# Step 5: # Use cross-validation folds (e.g., 5-fold cross-validation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and obtain the performance metrics (e.g., R-squared)
scores = cross_val_score(lr, X, y, scoring='r2', cv=kf)
scores

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
