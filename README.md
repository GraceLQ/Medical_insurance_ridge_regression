# Medical Insurance Premium Prediction Using Ridge Regression Model
This project aims to identify the most important factors influencing medical insurance prices and to build robust machine learning models for prediction. By leveraging data sourced from Kaggle, the ultimate goal is to explore how predictive models can enhance the efficiency and profitability of health insurance companies through better risk assessment and cost management.

Data source: https://www.kaggle.com/datasets/harishkumardatalab/medical-insurance-price-prediction?resource=download

## Technical Skills Highlighted
- Conducted data preprocessing, including cleaning datasets, handling missing values, and performing EDA with visualizations to identify key attributes.
- Developed and evaluated single-variable, multi-variable, and Ridge regression models, leveraging polynomial transformations and hyperprameter tuning to improve predictions and reduce overfitting.
- Adept at model evaluation using metrics such as R square and MSE to measure predictive accuracy and ensure robustness.

### Correlation Analysis
Correlation analysis shows a strong positive correlation (0.79) between smoker and premium, indicating that smoking status significantly influences medical expenses. In comparison, age and BMI show weaker correlations with charges, at 0.3 and 0.2 respectively, though they are still stronger compared to correlations of other variables.
![alt_text](heat1.png)

### Model Development and Refinement

1. Linear Regression 

   Z = df[['age','bmi','gender','no_of_child','region','smoker']]

   lm.fit(Z,Y)

   **R2_score with all attributes is: 0.75)**
   
2. Adjust the model with polynomial and standard scaler using pipeline.

   input = [('scale',StandardScaler()),('Polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

   pipe = Pipeline(input)

   pipe.fit(Z,Y)

   ypipe = pipe.predict(Z)

   **The R2 score increases from 0.75 to 0.845.**

3. Use Ridge Regression Model (alpha=0.1)
