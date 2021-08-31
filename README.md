# Term Deposit Marketing

# Data Science Project

The objective of the project was to build a machine learning classifier to predict whether a customer will subscribe (yes/no) to a term deposit.  
We are also interested in finding customers who are more likely to buy the investment product, and in doing so identify the segment(s) of customers our client should prioritise.
We will identify the features that the company should be focusing on and as a result identify what makes customers but the term deposit.


# Background and Problem Statement

ACME is a small startup focusing mainly on providing machine learning solutions in the European banking market. The company  works on a variety of problems including fraud detection, sentiment classification and customer intention prediction and classification.
ACME is interested in developing a robust machine learning system that leverages information held from call center data.
The company is looking for ways to improve the success rate for calls made to customers for any product that their clients offer. Towards this goal they are working on designing an ever evolving machine learning product that offers high success outcomes while offering interpretability for our clients to make informed decisions.

# Project overview

The project was carried out using data from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. 
The data consists of 40,000 rows and 14 columns, with one column indicating whether a customer subscribed to a term deposit or not. Refer to the data dictionary for more details on the information given in each column. The dataset was cleaned and transformed using feature engineering where necessary and split into training (70%) and testing (30%) test sets.
In this project python was used to build predictive models. Classification techniques were used to predict the likelihood of a customer subscribing to a term deposit or not using.  Various modelling techniques such as Logistic Regression, Naive Bayes, Decision Trees,  ensemble methods AdaBoost and Gradient Boosting and Random forest  were used to create models and make predictions using these models. 

The main challenge of this project was dealing with an imbalanced dataset. The dataset shows  customers with no term deposit customers of 93% vs term deposit customers of 7%. 
As the dataset was imbalanced, this needed to be addressed to obtain meaningful predictions. 
Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. A classification problem may be a little skewed, such as one with a slight imbalance or alternatively, the classification problem may have a severe imbalance similar to this one where we are predicting term deposit subscriptions.
Imbalanced classifications result in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class rather than the majority class.
Another challenge was the metrics used to evaluate performance. In this situation, the client wants us to use accuracy as the evaluating metrics. However, for  imbalanced datasets, using simpler metrics like accuracy can be misleading. In a dataset with highly unbalanced classes, if the classifier always "predicts"  the most common class without performing any analysis of the features, it will still have a high accuracy rate, which is obviously illusory.

The models were first assessed without considering the imbalance in the dataset. From the model results  we noted that the models all performed well, giving accuracy scores above 90% in all cases. However, as the dataset is imbalanced with 93% in the majority class and 7% in the minority class, this performance is misleading as accuracy is simply predicting the negative class all the time, achieving a score that naively looks good, but in practice is meaningless in predicting the minority class.
To overcome this, I have explored other popular methods used to deal with imbalanced datasets. These are outlined below.
* 1.Undersampling of the majority class
* 2.Oversampling of the minority class
* 3.SMOTE
* 4.XGBoost

# Prerequisites

To run the code, there were a number of Python libraries that needed to be installed. These are as follows:

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Logistic RegressionCV
* DecisionTree Classifier
* Bagging Classifier
* StandardScaler
* Metrics
* Train_test_split, cross_val_score, GridSearchCV
* Accuracy_score
* AdaBoostClassifier 
* GradientBoostingClassifier
* GaussianNB
* Under_sampling
* Over_sampling
* SMOTE
* XGBoost

# Summary/Findings

We have used the accuracy as the metric to evaluate the performance of the models, as the client requests the use of this metrics, however as a general rule, accuracy is not the best metrics for use in assessing model performance where you have imbalanced datasets and I would not recommend the use of this metrics in this instance.
Accuracy is not an effective measure since, among other things, it does not take into account the distribution of the misclassification among classes nor the marginal distributions. Other more subtle measures have been introduced in the multi-class setting to address this issue, improving efficiency and class discrimination power.
 
The XGBoost model was the best performing model, using accuracy as the metric. This model achieved 94% accuracy on the test set. Accuracy was used as this was requested by the client. However, as the dataset is an imbalanced one, Accuracy is not the best evaluation metric to use and in this case precision, recall or F1-score would be better metrics to use.

To determine what makes a customer buy term deposits, the company should focus on the following:
* Duration - The duration of the last contact in seconds is the most important feature across all the models. The duration of the last contact in seconds for the majority of term deposit holders fall between 1 and 999. Duration above 1000 seconds are less likely to result in a customer buying a term deposit
* Balance - This is the average yearly balance of term deposit in euros. This points to the amount of money the customer is able to place on term deposit.
* Day - The last contact day of the month. The results show that there are certain contact days that are most likely to result in the customer buying a term deposit.
* Age - The age of the customer based on their date of birth. The EDA showed that persons between ages 20 and 50 were more likely to invest in term deposits with the 30-39 age group being the most likely of the group.
* Campaign - This is the number of contacts performed during this campaign and for this client.The EDA shows that customers most likely to invest in term deposits had 5 or less contacts during the campaign.
* Month - The last contact month of the year. This is also very important and models show that June, March, October, July, November, February and May are the months that customers are most likely to buy term deposits.


