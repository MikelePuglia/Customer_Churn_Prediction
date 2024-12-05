# Customer-Churn-Prediction

https://miro.medium.com/v2/resize:fit:1400/format:webp/1*WZdoYPpmiIk1AcPQ1YHWug.png

**Introduction**

The dataset focuses on analyzing customer churn for ABC Multistate Bank, capturing cases where customers stop using the bank's services or close their accounts. Customer attrition represents a significant challenge for the company, both financially and reputationally. Proactively addressing churn risks can improve customer satisfaction and enable the development of effective retention strategies.

This project leverages Generalized Linear Models (GLMs), specifically logistic regression, to predict the probability of a customer churning based on behavioral and demographic factors. The objective is to identify key drivers of churn and enable the bank to implement targeted interventions.

**Logistic Regression**

Logistic regression is a predictive model for binary response variables, used to estimate the probability of an event occurring (e.g., churn vs. no churn). It employs a link function, such as logit, probit, or clog-log, to capture the relationship between predictor variables and the target probability.

**Dataset Overview**

The data was acquired from Kaggle: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset). It includes 10,000 observations capturing customer behaviors and demographic details, with the following 12 variables:

- **Credit Score**: Customer's creditworthiness score.
- **Country**: Country of residence (Germany, France, or Spain).
- **Gender**: Gender of the customer (1=Female, 0=Male).
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Customer's bank account balance (in Euros).
- **Products Number**: Number of bank products owned by the customer.
- **Credit Card**: Indicates if the customer owns a credit card (1=Yes, 0=No).
- **Active Member**: Indicates if the customer is an active member (1=Yes, 0=No).
- **Estimated Salary**: Estimated annual salary (in Euros).
- **Churn**: Binary target variable indicating customer churn (1=Churn, 0=No churn).

**Main Steps**

This project follows 8 structured steps:

1. **Descriptive Analysis:** Summarize key statistics to understand the dataset's structure and features.
   
2. **Pre-processing:** Clean and prepare the dataset by handling missing values, encoding categorical variables, and normalizing numerical features.

3. **Data Visualization:** Use visual tools to identify trends, patterns, and potential predictors of churn.

4. **Correlation Analysis:** Examine relationships between predictors and the churn variable.

5. **Data Splitting:** Divide the dataset into training and testing sets to evaluate model performance on unseen data.

6. **GLM Construction:** Build and optimize a logistic regression model suitable for binary outcomes.

7. **Model Validation:** Assess model accuracy using metrics such as Area Under the Curve (AUC), Precision, Recall, and F1 Score.

8. **Conclusion:** Present findings, highlight key drivers of churn, and provide recommendations to improve customer retention.

**Dataset Link**

[Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

**Outcome**

By accurately modeling customer churn, this project enables the bank to implement proactive retention strategies, minimizing attrition and enhancing customer satisfaction.
