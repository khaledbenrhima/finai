import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as pr_auc

# Load the dataset
url = 'https://raw.githubusercontent.com/Mun-Min/Credit_Risk_Classification/main/Resources/lending_data.csv?raw=true'
data = pd.read_csv(url)

# Print the top and bottom 5 rows
print(data.head(5))
print(data.tail(5))
print("")

# Split the data into features (independent variables) and the target variable (default or not)
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict probabilities for the test data
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Calculate precision-recall curve and AUC
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc_score = pr_auc(recall, precision)

# Calculate Gini coefficient
gini = 2 * roc_auc - 1

# Calculate KS statistic
ks = max(tpr - fpr)

# Print the results
print(f"ROC AUC: {roc_auc:.2f}")
print(f"PR AUC: {pr_auc_score:.2f}")
print(f"Gini: {gini:.2f}")
print(f"KS: {ks:.2f}")

# Create a lift chart
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, 1, len(tpr)), tpr, label='Cumulative Response')
plt.plot(np.linspace(0, 1, len(fpr)), fpr, label='Cumulative Non-Response')
plt.xlabel('Fraction of Sample')
plt.ylabel('Cumulative Percentage')
plt.title('Lift Chart')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
y_pred = (y_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("")
print("Confusion Matrix:")
print(confusion)
print("")
print("Classification Report:")
print(classification_rep)