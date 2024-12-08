# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Sample data
data = pd.DataFrame({
    'Age': [25, 30, None, 35, 28],
    'Income': [50000, 60000, 75000, None, 55000],
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Loan_Status': ['Approved', 'Rejected', 'Approved', 'Approved', 'Rejected']
})

print("Original Data:")
print(data)

# Handling missing values with mean imputation
imputer = SimpleImputer(strategy='mean')
data[['Age','Income']] = imputer.fit_transform(data[['Age','Income']])

print("\nMissing Data replaced with mean:")
print(data)

# Encoding categorical variables (Gender and Loan_Status)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

print("\n1-hot encoding categorical data:")
print(data)

# Scaling numerical features (Age and Income) using StandardScaler
scaler = StandardScaler()
data[['Age', 'Income']] = scaler.fit_transform(data[['Age', 'Income']])

# Display the preprocessed and cleansed data
print("\nScaling numerical features using StandardScaler:")
print(data)