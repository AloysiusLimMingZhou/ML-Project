# Project Description
**This Project will be focusing on House Prices Prediction using dataset from Kaggle.**\
In this project, it covers a complete Machine Learning Workflow as below:
1. Libraries import
2. Feature Engineering
3. Exploratory Data Analysis
4. ML Model Selection
5. ML Models Train Test
6. Metrics Evaluation Comparison between ML Models

Lastly, this project will be using Linear Regression model from a custom-built algorithm and comparing it with Scikit-Learn's Linear Regression model. We will be comparing their metrics evaluation using the same dataset and workflow processes

# Machine Learning Model Selection
It is essential for us to identify the suitable ML Algorithm used for our dataset.

Since our Housing Dataset has a continuous value, where our target, y represents the house price carries actual number values(house price:$1000000) instead of categorical data(yes, no), we should use a regression algorithm to predict continuous value(house prices)

Thus, Linear Regression will be our best option here.

# a) Comparison between Linear and Logistic Regression
**1. Linear Regression**
The formula for Linear Regression can be as follows:
$$y = \theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n} + b$$
**Where:**\
y = Actual value\
x = Value for each feature\
$\theta_{n}$ = Weights for each feature (1-n)\
b = Bias

- The value of the target y, can be anywhere from 0 to ∞ (i.e. Predicted house prices: $2340000)
- It takes in the input value of each feature and multiply it with the weights, and add with a bias, to produce a predicted y value(house prices).

**2. Logistic Regression**
The formula for Logistic Regression can be as follows:
$$\frac{1}{1+e^{-(b_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{m}x_{m})}}\\= \frac{1}{1+e^{-z}}$$

**Where:**\
z = $y = \theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n} + b$\
$\theta_{m}$ = Weights for mth column (mth feature)\
$x_{m}$ = Value for the mth column (mth feature)\
$b_{0}$ = Bias

- The value of the target y represents categorical data (i.e. 0 - not spam; 1 - spam)
- It takes in the input value of each feature and multiply it with the weights and add with a bias, to produce a continuous output.
- The output is then converted into probabilistic value using activation function like sigmoid (i.e. 0.5635634).
- Finally, by implementing a certain treshold (i.e. y > 0.5), the probabilistic value is then converted into a class label which represents the model's prediction (i.e. 0.65785 -> 1 (Spam email))

In short, though both Linear and Logistic Regression falls under the same Linear Model family, Linear Regression directly predicts an actual number, while Logistic Regression takes a step further to convert the actual number into probability value, and round the probability value off to the nearest categorical value.

In the end, Linear Regression is more suitable for housing price prediction as we want an actual number as output (regression), not class label (categorical)

# b) Use case of Linear and Logistic Regression:
**1. Linear Regression:**\
a) House Pricing\
b) Student Grade\

**2. Logistic Regression:**\
a) Spam Email (0 - true, 1 - false)
b) Heart Disease (0 - true, 1 - false)

# Comparison between Custom Linear Regression and Scikit-Learn Linear Regression Model
In this part, we will be training and testing 2 Linear Regression models, to determine the proof-of-concept of our custom LR model and compare its performance with industry standard LR model:
1. Custom Linear Regression Model
2. Scikit-Learn Linear Regression Model

In my Custom Linear Regression Model, it includes the following blocks of mathematical concepts that forms together:
1. Linear Regression Formula
```math
y = \theta_{1}x_{1}+\theta_{2}x_{2}+...+\theta_{n}x_{n} + b
```
**Where:**\
y = Actual value\
x = Value for each feature\
$\theta_{n}$ = Weights for each feature (1-n)\
b = Bias

2. Mean Square Error Loss
**Formula:**\
```math
\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2}
```
**Where:**\
n = Number of total rows (Total dataset count)\
$\hat{y}$ = Predicted value\
y = Actual value

3. L1 Lasso Regularisation
**Formula:**\
```math
\lambda\sum_{i=1}^{m}|\theta_{i}|
```
**Where:**\
$\lambda$ = l1 penalty constant (recommended: 0.0001)\
m = Number of total columns (Total features in a dataset)\
$w_{i}$ = Weights for each feature (from 1 - m)\

**Combining L1 (Lasso) Regularisation with MSE:**\
```math
\frac{1}{n}\sum_{i=1}^{n}(\hat{y_{i}}-y_{i})^{2} + \lambda\sum_{i=1}^{m}|\theta_{i}|
```

4. Root Mean Square Error Formula
**Formula:**
```math
\sqrt{\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}}
```

6. R-Square Formula
```math
1 - \frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y}_{i})^{2}}
```

**Where:**\
$y_{i}$ = Actual value for ith index\
$\hat{y}\_{i}$ = Predicted value for ith index\
$\bar{y}\_{i}$ = Mean of the actual value

**For detailed explanation, of the formula, you may refer to the algorithmic explanation from my ML-Algorithm Repository for Linear Regression**
c) Credit Card Fraud (0 - true, 1 - false)

# Conclusion

In this project, we have implemented a custom Linear Regression model from scratch using Gradient Descent optimization methods and evaluated it against Scikit-Learn’s built-in `LinearRegression` model. The objective is to validate the accuracy of the custom model's theoretical concepts, mathematical formulation and overall implementation by comparing its performance with an industry-standard model.

The performance comparison between the two models is summarized as below:

| Model                     | R² Score | RMSE |
|---------------------------|---------:|-------------:|
| Custom Linear Regression  | 0.628    | 1,371,525 |
| Scikit-Learn Linear Regression| 0.653    | 1,324,507 |

Based on the results, our custom model achieves similar performance metrics with Scikit-Learn’s implementation. Although the Scikit-Learn model attains slightly better results, this is expected due to its more advanced numerical optimizations. Nevertheless, our custom Linear Regression model has demonstrated accuracy and completeness that even rivals an industry-standard model.

This outcome validates that the mathematical foundations which includes gradient loss derivations, l1-ridge regularization, and metrics calculation used in the custom model are sound and correctly implemented. Most importantly, it proves to us that a manually implemented optimization-based linear regression model can effectively approximate the performance of a production-grade library when proper data preprocessing and training procedures are followed.

Overall, this project successfully demonstrates a complete machine learning workflow from feature engineering and exploratory data analysis, to model implementation, evaluation, and comparison, all while reinforcing a deep understanding of linear regression and gradient descent-based optimization models.

# Future Directions
We believe that there's none existence of a "perfect" project, and every project requires continuous improvements to be more enhanced and mature over time. Here are some considerations that can be done in order to boost the performance of the models.
1. **Better feature creation**: The current feature derivations are not complete and some might be irrelevant. To boost the `R^2 value` and the model's understanding, more concrete combination of other features like `hotwaterheating` should be connected in order to solidify each feature's relationship with each other in the dataset.
2. **Implementing more sophisicated Machine Learning models**: As of now Linear Regression is great at capturing dataset who has linear relationship with the features. However, the Housing dataset is more complex and some of their relationship might be non-linear. Thus, implementing other algorithms like `Polynomial Regression` and `Random Forests` might be more suitable in handling dataset with complex feature's relationships which help boost its performance.
