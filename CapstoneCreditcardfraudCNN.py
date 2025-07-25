import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve
# Load dataset
data = pd.read_csv("/Users/preetisirohi/Documents/Capstone/Dataset/Preeti_creditcard_2023.csv")  # Replace with your file

# Step 1: Feature/target split
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 2: Split the dataset into a training and a test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
print(y.value_counts(normalize=True))

print("Test set class distribution:")
print(y_test.value_counts(normalize=True))

# Step 3: Feature scaling to make sure each feature contributes equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Reshape for CNN (samples, features, 1)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Step 5: Build CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train model on entire training dataset
history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1)

# Step 7: Model prediction
y_pred_proba = model.predict(X_test_cnn)
y_pred = (y_pred_proba >= 0.5).astype(int)
print(np.unique(y_pred))

print("\n------ CNN model Performance and Accuracy------\n", classification_report(y_test, y_pred, digits= 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Step 8:Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Step 9:Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.xlabel('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Step 10: Plot Model accuracy curve
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy over Epochs')
plt.show()

# Calculate FPR and TPR and Thershold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC AUC Curve for CNN
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange',label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
