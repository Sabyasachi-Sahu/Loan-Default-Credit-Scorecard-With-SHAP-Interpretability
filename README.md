# Loan Default Prediction & Credit Scorecard System

This repository contains an end-to-end Machine Learning project designed to predict the probability of loan default and generate a standardized credit score for applicants. The project covers data generation, exploratory data analysis (EDA), model training, and deployment via a Flask API.

## 🚀 Project Overview

Financial institutions use credit scoring to determine the risk of lending to a customer. This project simulates that environment by:
1. **Generating Synthetic Data**: Using `Faker` to create a realistic dataset of 227,000 loan applications.
2. **Analysis**: Performing EDA using SQL (SQLite) and visualization libraries to find risk patterns.
3. **Machine Learning**: Training an XGBoost Classifier to predict default risk.
4. **Scoring Engine**: Converting default probabilities into a credit score (300-900 range) using industry-standard scaling logic.
5. **Deployment**: Serving the model through a REST API for real-time decisioning.

## 📂 Project Structure

* `Data Generation with Faker.ipynb`: Initial script to generate synthetic loan data.
* `Data Generation with Faker Advanced.ipynb`: Advanced data generation with logical correlations (e.g., credit history dependent on age).
* `Project 2 Loan Default Credit Scorecard.ipynb`: The main pipeline including SQL-based EDA, model training (XGBoost), and scoring logic.
* `credit_model_app.py`: Flask web server to serve the trained model as an API.
* `models/`: Directory containing the saved `xgb_credit_model.pkl`.

## 📊 Features & Data

The model utilizes several key financial and demographic attributes:
* **Demographics**: Age, Gender, Marital Status, Education, Dependents.
* **Financials**: Annual Income, Loan Amount, Monthly EMI, Existing Loans.
* **Ratios**: Debt-to-Income (DTI), Loan-to-Income.
* **Credit History**: Credit Score (internal/external), Years of Credit History.
* **Property**: Property Area (Urban/Semi-Urban/Rural).

## 🛠️ Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/loan-default-scorecard.git](https://github.com/your-username/loan-default-scorecard.git)
    cd loan-default-scorecard
    ```

2.  **Install dependencies**:
    ```bash
    pip install flask joblib pandas numpy xgboost matplotlib seaborn faker
    ```

3.  **Generate the data and train the model**:
    Run the `Project 2 Loan Default Credit Scorecard.ipynb` notebook to process the raw data and save the model to the `models/` folder.

## 💻 Usage (API Deployment)

To start the local prediction server:

```bash
python credit_model_app.py
```



The API will be available at http://127.0.0.1:5000/predict.

#### Example Request
You can send a POST request with applicant data in JSON format:

JSON

{
    "age": 35,
    "annual_income_lakhs": 15.5,
    "loan_amount_lakhs": 5.0,
    "loan_tenure_months": 36,
    "credit_score": 720,
    "debt_to_income": 0.15,
    "existing_loans": 1,
    "has_co_applicant": 1,
    "employment_type": 1
}


#### Example Response
The API returns a credit score, risk category, and an automated decision:

JSON

{
    "credit_score": 742,
    "default_probability": 0.0245,
    "risk_category": "Low",
    "decision": "Approve"
}


## 📈 Model Performance
The XGBoost model was chosen for its high performance with tabular data.

Scoring Logic: Probabilities are scaled to a score between 300 and 900.

Thresholds:

Score >= 650: Approve

550 - 649: Review

< 550: Decline
