#Import required libraries
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

print()
print("Starting Task B Random Forest:")
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

# Training the random forest on the training sets
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1)  # For analysis, n_estimators will change to 1, 10, 500 seperately to discuss the effect on prediction accuracy
                                                                            # For analysis, max_depth will change to 5, 10, 25 seperately to discuss the effect on prediction accuracy
rf_clf.fit(X_train, y_train)

# Prediction on the test sets
y_pred = rf_clf.predict(X_test)

# Final evaluation on the model
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Random forest for test set prediction accuracy: {test_accuracy}")
print(classification_report(y_test,y_pred))#text report showing the main classification metrics
print(confusion_matrix(y_test, y_pred))  # TP FN/ FP TN

# Define the learning curve
train_sizes, train_scores, val_scores = learning_curve(
    rf_clf,
    X_train, y_train,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.01, 1.0, 10)
)

# Calculating mean accuracy
train_accuracy_average = np.mean(train_scores, axis=1)
val_accuracy_average = np.mean(val_scores, axis=1)

print()
print("Task B Random Forest ends")
print()

# Plot the learning curve
plt.plot(train_sizes, train_accuracy_average, label='Training accuracy')
plt.plot(train_sizes, val_accuracy_average, label='Validation accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

