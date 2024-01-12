# Description: 

This project contains two classification tasks. Task A is to do binary classification for PneumoniaMNIST dataset and Task B is to do multi-class classification for PathMNIST dataset to divid into 9 different types. Task A is solved by logistic regression with image flattening and PCA data preprocessing techniques. Task B is solved by one of ensamble methods called random forest, which is the combination of decision tree and bagging strategy. Due to the bad performance on random forest in Task B, OVO logistic regression and OVR logistic regression are involved to compare with random forest. 


# Role of each file:

* **A_Logistic_Regression.py**: Express how to implement logistic regression model with image flattening and PCA to do binary classification for PneumoniaMNIST dataset in python
* **B_Random_Forest.py**: Express how to implement random forest model with image flattening to do multi-class classification for PathMNIST dataset in python
* **B_Logistic_Regression_OVO.py**: Express how to implement One-vs-One logistic regression model with image flattening and PCA to do multi-class classification for PathMNIST dataset in python
* **B_Logistic_Regression_OVR.py**: Express how to implement One-vs-Rest logistic regression model with image flattening and PCA to do multi-class classification for PathMNIST dataset in python
* **main.py**: Run each of machine learning code together


# Required packages:

numpy, os, scikit-learn, matplotlib 