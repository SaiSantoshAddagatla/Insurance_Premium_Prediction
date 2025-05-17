# Insurance Premium Prediction Project

## Problem Statement

The goal of this project is to develop a machine learning model that accurately predicts insurance premium prices based on user-specific information such as age, BMI, and health conditions. This project also involves deploying the model through a web-based application that allows real-time predictions.

## Target Metric

The primary evaluation metric for model selection was **Root Mean Squared Error (RMSE)**, with supporting metrics like **MAE** and **Rsquare** to validate overall performance.

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed age, BMI, and medical conditions and their relationship with insurance premiums.
- Used correlation heatmaps, distribution plots, and box plots for insights.

### 2. Hypothesis Testing
- Tested relationships between health conditions (e.g., diabetes, surgeries) and premium prices.
- Applied statistical tests to validate significant predictors.

### 3. Feature Engineering
- Calculated BMI from height and weight.
- Encoded categorical features and scaled numerical features for model training.

### 4. Machine Learning Modeling
- Trained multiple models: Linear Regression, Random Forest, and Gradient Boosting.
- Tuned Random Forest gave the best results.

### 5. Model Evaluation

| Model             | MAE      | RMSE     |Rsquare|
|-------------------|--------- |----------|-------|
| Linear Regression | ₹2586.18 | ₹3494.41 | 0.71  |
| Random Forest     | ₹1038.13 | ₹2141.25 | 0.89  |
| Gradient Boosting | ₹1523.09 | ₹2382.59 | 0.86  |
| Tuned RF (final)  | ₹984.13  | ₹2053.67 | 0.90  |


## Best Performing Model

- Model:	Tuned Random Forest Regressor  
- RMSE: 	2053.67  
- Rsquare:	0.90
- MAE:		984


- Selected for deployment in the web-based calculator

---

## Insights & Recommendations

- Age and BMI are strong predictors of premium cost.
- Chronic diseases, surgeries, and diabetes also significantly impact pricing.



## Live Project Links

- Tableau Dashboard: [Link to Tableau](https://public.tableau.com/views/MedicalInsurancePremiumAnalysis_17469490918310/Dashboard2-PremiumPricing?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)  
- Google Colab Notebook: [Link to Colab](https://colab.research.google.com/drive/1OKnvltVOXv9tVCdHsW45PjS4TXWgjON-?usp=sharing)  
- Streamlit App: Can be run locally using the steps below (https://insurancepremiumprediction-sugdbsxopwvm5gnput2uw6.streamlit.app)
- Medium Technical Blog: [Link to Medium](https://medium.com/@saisantosh24898/predicting-insurance-premiums-using-machine-learning-a-complete-technical-journey-953137f1e348)


---

## Deployment

The final model is deployed using **Streamlit**, allowing users to enter personal and medical details to receive a real-time insurance premium estimate.

### How to Run the App Locally:

```bash
git clone https://github.com/your_username/insurance-premium-prediction.git
cd insurance-premium-prediction
pip install -r requirements.txt
streamlit run app.py



