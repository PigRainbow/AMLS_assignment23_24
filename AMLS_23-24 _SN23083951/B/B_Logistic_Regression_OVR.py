#Import required libraries
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

print()
print("Starting Task B OVR Logistic Regression:")
print()

# Loading pathmnist dataset
pathmnist_path = os.path.join(os.path.dirname(__file__), '../Datasets/pathmnist.npz') 
TASK_B = np.load(pathmnist_path)

# Extracting features and target labels
X_train = TASK_B['train_images'].reshape(TASK_B['train_images'].shape[0], -1)  # transform the image data into two-dimensional arrays
y_train = TASK_B['train_labels'].ravel()  # flatten the label arrays to one-dimensional arrays
X_val = TASK_B['val_images'].reshape(TASK_B['val_images'].shape[0], -1)
y_val = TASK_B['val_labels'].ravel()
X_test = TASK_B['test_images'].reshape(TASK_B['test_images'].shape[0], -1)
y_test = TASK_B['test_labels'].ravel()

# Dimensionality reduction
n_dimension = 40
pca = PCA(n_components=n_dimension)
pca.fit(X_train, y_train)
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Training the logistic regression on the training sets by One-vs-Rest
log_reg_ovr = OneVsRestClassifier(LogisticRegression(C=1.0,  max_iter=5000))
log_reg_ovr.fit(X_train, y_train)

# Prediction on the test sets
y_pred_ovr = log_reg_ovr.predict(X_test)

# Final evaluation on the model
test_accuracy = accuracy_score(y_test, y_pred_ovr)
print(f"Logistic regression for test set prediction accuracy: {test_accuracy}")
print(classification_report(y_test,y_pred_ovr))#text report showing the main classification metrics
print(confusion_matrix(y_test, y_pred_ovr))  # TP FN/ FP TN

print()
print("Task B OVR Logistic Regression ends")
print()