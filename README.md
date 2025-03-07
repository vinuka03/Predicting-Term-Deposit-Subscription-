Bank Marketing Subscription Prediction
This project focuses on predicting whether a client will subscribe to a term deposit based on the Bank Marketing Dataset from the UCI Machine Learning Repository. The classification task is performed using Neural Networks and Random Forest Classification, with extensive data preprocessing and class imbalance handling before training the models.

Project Overview
Financial institutions use marketing campaigns to promote term deposits. The goal of this project is to build machine learning models that predict a client’s likelihood of subscribing to a term deposit based on past marketing campaign data.

Dataset
The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository. It contains client-related, campaign-related, and social/economic attributes.

Steps in the Project
Data Preprocessing

Handle missing values
Encode categorical variables
Scale numerical features
Handling Class Imbalance

The dataset is imbalanced, with more non-subscribers than subscribers.
Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
Model Training & Evaluation

Random Forest Classifier
Neural Network (Multi-Layer Perceptron - MLP)
Compared model performances using accuracy, precision, recall, and F1-score.
Results
Evaluated model performance with metrics such as ROC-AUC Score, Confusion Matrix, and Precision-Recall Curve.
The models provide insights into factors influencing client subscription decisions.
Technologies Used
Python
Pandas, NumPy
Scikit-learn
TensorFlow/Keras (for Neural Networks)
Imbalanced-learn (for SMOTE)
Matplotlib, Seaborn (for data visualization)
