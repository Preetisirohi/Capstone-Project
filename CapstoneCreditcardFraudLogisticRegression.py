# First step is to install Libraries in terminal and then import them in Py file
import pandas as pd
import numpy as np
# Below 2 libraries are for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score, recall_score
from sklearn.metrics import auc
# Read the data set, provide tha local path of the file
df = pd.read_csv('/Users/preetisirohi/Documents/Capstone/Dataset/Preeti_creditcard_2023.csv')
print("------ Print the top 5 rows of the dataset ------")
print(df.head())

# To check basic information about tha data
print("------ Basic information about the dataset ------")
df.info()

# Check missing values
print("------ Checking for missing values ------")
print(df.isnull().sum())

print("------ Checking for duplicate values ------")
print(df.duplicated().sum())
# Divide the dataset into features(input variable) and target(output variable)
x = df.drop(['id', 'Class'], axis=1, errors='ignore')
y = df['Class']
print(x.columns.tolist())

# Split the dataset into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
print("------ After split # of rows and column in the training and test dataset ------")
print(x_train.shape)
print(x_test.shape)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# To check class distribution in training dataset
print("------ Class distribution of Training dataset ------")
print(pd.Series(y_train).value_counts(normalize=True))

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
print("------ 5 folds cross validation score for Logistic Regression model ------")
cv_scores = cross_val_score(lr_model, x_train_scaled, y_train, cv=5, scoring='f1')
print(" cross validation F1 score:", cv_scores)
print(" Average F1 score:", np.mean(cv_scores))

# Fit the model on entire training dataset
lr_model.fit(x_train_scaled, y_train)

# Model Prediction
print('------ Logistic Regression Model performance and Accuracy ------')
y_pred = lr_model.predict(x_test_scaled)
print(classification_report(y_test, y_pred, digits=3))

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = x.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.xlabel('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# ROC Curve
y_pred_proba = lr_model.predict_proba(x_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Visualise the ROC AUC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()