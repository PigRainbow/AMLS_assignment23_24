#Import required libraries
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

print()
print("Starting Task A Logistic Regression:")
print()

# Loading PneumoniaMNIST dataset
pneumoniamnist_path = os.path.join(os.path.dirname(__file__), '../Datasets/pneumoniamnist.npz')
TASK_A = np.load(pneumoniamnist_path)

# Extracting features and target labels
X_train = TASK_A['train_images'].reshape(TASK_A['train_images'].shape[0], -1)  # transform the image data into two-dimensional arrays
y_train = TASK_A['train_labels'].ravel()  # flatten the label arrays to one-dimensional arrays
X_val = TASK_A['val_images'].reshape(TASK_A['val_images'].shape[0], -1)
y_val = TASK_A['val_labels'].ravel()
X_test = TASK_A['test_images'].reshape(TASK_A['test_images'].shape[0], -1)
y_test = TASK_A['test_labels'].ravel()

# PCA reduction
n_dimension = 50  # For analysis, dimensionality will change to 10, 100, 300 seperately to discuss the effect on prediction accuracy
pca = PCA(n_components=n_dimension)
pca.fit(X_train, y_train)
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Training the logistic regression on the training sets
log_reg = LogisticRegression(C=1.0,  max_iter=5000) # For analysis, max_iter will change to 50, 100, 1000 seperately to discuss the effect on prediction accuracy
                                                    # For analysis, C will change to 0.1, 0.5, 2.0 seperately to discuss the effect on prediction accuracy
log_reg.fit(X_train, y_train)

# Prediction on the test sets
y_pred = log_reg.predict(X_test)

# Final evaluation on the model
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic regression for test set prediction accuracy: {test_accuracy}")
print(classification_report(y_test,y_pred))#text report showing the main classification metrics
print(confusion_matrix(y_test, y_pred))  # TP FN/ FP TN

# Define the learning curve
train_sizes, train_scores, val_scores = learning_curve(
    log_reg , 
    X_train, y_train, 
    cv=10,
    n_jobs=-1,
    train_sizes = np.linspace(0.1, 1.0, 20)
)

# Calculating mean accuracy
train_accuracy_average = np.mean(train_scores, axis=1)
val_accuracy_average = np.mean(val_scores, axis=1)

print()
print("Task A Logistic Regression ends")
print()

# Plot the learning curve
plt.figure()
plt.xlim(right=4500)
plt.plot(train_sizes, train_accuracy_average, label='Training accuracy')
plt.plot(train_sizes, val_accuracy_average, label='Validation accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


