# supervised-learning
Module 20 Challenge - Credit Risk Classification - May 2023

****ANALYSIS REPORT****

## Overview of the Analysis

The goal was to create a model that would correctly identify the creditworthiness of borrowers. Data used was from historical lending activity from a peer-to-peer lending services company. 

The label - the column/value being affected - was loan status. A loan_status value of 0 means that the loan is healthy and a value of 1 means that the loan has a high risk of defaulting. 

The features - the columns/factors that may affect the label - were loan size, interest rate,	borrower income, debt to income, number of accounts, derogatory marks, and total debt.

First, the data set was divided into Y (the label) and x (the features). For the initial dataset, 96.7% (75036 observations) were of healthy loans leaving less than 4% of the dataset for the minority class, high-risk loans. This was observed using value_counts.

The Y and x data was then split into testing and training groups using train_test_split from sklearn. 

Using LogisticRegression, a logistic regression model was created based on the training data. That model was used to generate predictions using the testing data. The success of the model was assessed by looking at the balanced accuracy score, confusion matrix, and classification report. These were generated using sklearn.metrics.

Lastly, random over-sampling was performed on the initial dataset using RandomOverSampler from imblearn.over_sampling. It increased the number of high-risk loan observations in the dataset to match the number of healthy loan observations - 75036 of each.

The new dataset was divided into Y and x. This could have been split into testing and training groups but was not done to allign with the assignment request. 

Another logistic regression model was created based on the new training data. That model was used to generate predictions using the original testing data. The success of the revised model was assessed by again looking at the balanced accuracy score, confusion matrix, and classification report generated using sklearn.metrics.

## Results

* Accuracy is how often the model is correct — the ratio of correctly predicted observations to the total number of observations. 
* Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
* Recall is the ratio of correctly predicted positive observations to all predicted observations for that class. 


* Machine Learning Model 1 - Logistic Regression with Original Data:
  * Original Data: 75036 healthy loans (0) and 2500 high-risk loans (1) - calculated using value_counts.
  * Balanced Accuracy Score: 0.952 - calculated using balanced_accuracy_score
  * Precision Score: 1.00 (0), 0.85 (1) - read from the classification report
  * Recall Score: 0.99(0), 0.91 (1) - read from the classification report

The logistic regression model did very well. The balanced accuracy score, the average of recall obtained on each class, was 0.952 out of a best score of 1.0. Recall for the healthy loan class is the number of loans correctly identified as healthy divided by the total number of loans that actually are healthy. (The same holds for the high-risk class). 
From the confusion matrix, the 0/0 - 1/1 diagonal is extremely high in value compared to the 0/1 - 1/0 diagonal showing that the model was highly accurate in its predictions. However, about 10% of the correctly predicted high-risk loans were incorrectly predicted as healthy and only about 0.5% of the correctly predicted healthy loans were incorrectly predicted as high-risk. So the model was very good at predicting healthy loans and not as good at predicting high-risk loans. Lastly, looking at the Classification Report, the model's predictions for the healthy loans received precision, recall, and F1 scores of 1.00, 0.99, and 1.00 respectively, while the scores for the predictions for the high-risk loans were lower, 0.85, 0.91, and 0.88. This confirms that the model is nearly 100% successful at predicting the healthy loans but less successful at predicting the high-risk loans.

* Machine Learning Model 2 - Logistic Regression with Resampled Training Data:
  * Resampled Data: 75036 healthy loans (0) and 75036 high-risk loans (1) - calculated using value_counts.
  * Balanced Accuracy Score: 0.994
  * Precision Score: 1.00 (0), 0.84 (1) 
  * Recall Score: 0.99(0), 0.99 (1)

The model fit with oversampled data seemed to perform better than the model fit with the original data. The balanced accuracy score increased from 0.952 to 0.994. The number of incorrect predictions for the actual healthy loans (0) increased slightly from 102 to 116, but the model still performed very well with this class. The precision, recall, and F1 scores did not change - 1.00, 0.99, 1.00.
With the actual high-risk loans (1), the number of incorrect predictions went down from 56 to 4. The precision score went down  slightly from 0.85 to 0.84 while the recall and F1 scores went up from 0.91, 0.88 to 0.99, 0.91.

## Summary

Machine Learning Model 2 performed better overall (higher balanced accuracy score) and in predicting high-risk loans (higher recall score). However, it is important to understand how the model would be used before recommending the model. Model 2 could be recommended if the goal is to identify high-risk loans and it is acceptable that not all such loans are identified by the model. In addition, it must be acceptable that some healthy loans are mistakenly identified as high-risk. There should be a second check of the high-risk loans identified to ensure they are in fact high-risk.


****INSTRUCTIONS:****
The instructions for this Challenge are divided into the following subsections:
1. Split the Data into Training and Testing Sets
2. Create a Logistic Regression Model with the Original Data
3. Predict a Logistic Regression Model with Resampled Training Data
4. Write a Credit Risk Analysis Report

1. Split the Data into Training and Testing Sets
Open the starter code notebook and use it to complete the following steps:
- Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.
- Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

NOTE
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

- Split the data into training and testing datasets by using train_test_split.

2. Create a Logistic Regression Model with the Original Data (and 3.)
Use your knowledge of logistic regression to complete the following steps:
- Fit a logistic regression model by using the training data (X_train and y_train).
- Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
- Evaluate the model’s performance by doing the following:
    - Calculate the accuracy score of the model.
    - Generate a confusion matrix.
    - Print the classification report.
- Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

4. Write a Credit Risk Analysis Report
Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:
- An overview of the analysis: Explain the purpose of this analysis.
- The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.
- A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.

