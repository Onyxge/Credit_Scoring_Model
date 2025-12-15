# Credit Risk Scoring System

## Project Overview

This project implements an **end-to-end Credit Risk Scoring System** designed to predict whether a customer is **High Risk** or **Low Risk** based on historical transaction behavior. The system follows a full machine learning lifecycle, from data analysis and feature engineering to model training, deployment, and CI/CD automation.

The solution is built with **business decision-making** in mind, focusing on explainability, robustness, and deployability rather than only predictive accuracy.

---

## Business Objective

Financial institutions face significant losses due to customer default and fraudulent behavior. The goal of this project is to:

* Identify customers with **higher probability of default or risky behavior**
* Enable **early risk mitigation** strategies
* Support **data-driven credit decisions**
* Provide an interpretable and deployable ML solution

The output of the system is:

* A **binary risk label** (High Risk / Low Risk)
* A **risk probability** (Probability of Default)
* A derived **credit score** (300–850 range)

---

## Dataset Description

The dataset consists of customer-level transaction records including:

* Transaction timestamps
* Transaction amounts (debits and credits)
* Product categories
* Channel identifiers
* Customer identifiers

The raw transactional data is transformed into **customer-level behavioral features** suitable for credit risk modeling.

---

## Exploratory Data Analysis (EDA)

EDA was conducted to understand customer behavior patterns and risk indicators.

### Key Findings

* Total customers analyzed: **3,742**
* Average transactions per customer: **25.56**
* Most common product category: **Financial Services**
* Approximately **40%** of transactions are negative (refunds or adjustments)
* Peak transaction activity occurs around **16:00 (4 PM)**

### Visual Insights

The following visualizations were generated and included in the report:

* Transaction amount distributions
* Transactions over time
* Product category usage
* Channel usage frequency
* RFM metric distributions

(EDA figures are stored under the `reports/figures/` directory.)

---

## Feature Engineering

Feature engineering was performed at the **customer level**.

### RFM Features (Proxy Target Creation)

* **Recency**: Days since last transaction
* **Frequency**: Number of transactions
* **Monetary**: Total positive transaction value

### Behavioral Aggregations

* Product category usage counts
* Channel usage counts
* Average and standard deviation of transaction hour

### Behavioral Diversity Metrics

* **Channel Diversity**: Measures how diversified customer channel usage is
* **Category Diversity**: Measures how varied customer product usage is

### Engagement Score

A composite engagement score was engineered to reflect:

* Activity consistency
* Behavioral diversity
* Transaction frequency

These features help capture **customer stability vs volatility**, which is critical for risk modeling.

---

## Target Variable Engineering

Since no explicit default label existed, a **proxy target** was created using clustering.

### Approach

* RFM features were log-transformed and scaled
* **KMeans clustering (k=3)** was applied
* Clusters were analyzed based on:

  * High Recency
  * Low Frequency
  * Low Monetary value

The cluster exhibiting the riskiest behavior was labeled as **High Risk (1)**. Others were labeled **Low Risk (0)**.

This approach aligns with real-world scenarios where explicit default labels are unavailable.

---

## Model Development

Two models were trained and evaluated:

### Baseline Model

* **Logistic Regression**
* Used for interpretability and benchmarking

### Challenger Model

* **Random Forest Classifier**
* Captures nonlinear interactions and behavioral patterns

### Hyperparameter Tuning

Hyperparameter tuning was performed to improve model stability and generalization:

* `n_estimators`
* `max_depth`
* `min_samples_split`

Tuning focused on balancing **performance**, **interpretability**, and **overfitting control**, rather than maximizing complexity.

---

## Model Evaluation

Models were evaluated using:

* F1-score
* ROC-AUC
* Precision and Recall

### Results

* Logistic Regression: **F1 ≈ 0.98**, **ROC-AUC ≈ 0.98**
* Random Forest: **F1 ≈ 0.98**, **ROC-AUC ≈ 0.98**

The Random Forest model was selected as the final model due to its superior ability to model complex behavioral patterns.

---

## System Architecture

### High-Level Flow

```
Raw Transaction Data
        ↓
Data Processing & Feature Engineering
        ↓
Proxy Target Generation (Clustering)
        ↓
Model Training & Evaluation
        ↓
FastAPI Prediction Service
        ↓
Streamlit User Interface
```

---

## API Service (FastAPI)

A REST API was built using **FastAPI**.

### Key Endpoints

* `GET /` – Health check
* `POST /predict` – Risk prediction

### API Output

* Risk label (High / Low)
* Risk probability
* Credit score (300–850)

The API automatically aligns incoming features with training features, ensuring robustness against missing inputs.

---

## Streamlit Dashboard

A Streamlit web application was built as an optional enhancement.

### Features

* User-friendly input form
* Real-time risk prediction
* Visual display of:

  * Risk label
  * Default probability
  * Credit score

This component demonstrates how the model can be consumed by non-technical stakeholders.

---

## Dockerization

The system is fully containerized using Docker:

* FastAPI backend container
* Streamlit frontend container
* Docker Compose for orchestration

This ensures reproducibility and easy deployment across environments.

---

## CI/CD Pipeline

A CI pipeline was implemented using **GitHub Actions**.

### Pipeline Steps

* Code checkout
* Dependency installation
* Code linting using **flake8**
* Unit testing using **pytest**
* Docker image build validation

The pipeline fails automatically if linting or tests fail, ensuring code quality and stability.

---

## Project Structure

```
credit-risk-model/
│
├── .github/
│   └── workflows/
│       └── workflow.yml
│
├── Data/
│   ├── Raw/
│   └── Processed/
│
├── models/                  # model artifacts only
│   ├── model.pkl
│   ├── rfm_kmeans.pkl
│   └── rfm_scaler.pkl
│
├── reports/
│   ├── figures/
│   └── interim_report_w4.md
│
├── notebooks/
│   ├── eda.ipynb
│   └── test.ipynb
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI app
│   │
│   ├── dashboard/
│   │   └── app.py           # Streamlit app
│   │
│   ├── data_processing.py
│   ├── train_model.py
│   ├── predict.py
│   └── __init__.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_data_processing.py
│
├── Dockerfile
├── DockerfileStreamlit
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Key Takeaways

* Behavioral transaction data is highly predictive of credit risk
* RFM-based proxy labeling is effective when default labels are unavailable
* Engagement and diversity metrics add meaningful signal
* End-to-end ML systems require strong engineering, not just modeling

---

## Future Improvements

* Incorporate SHAP for model explainability
* Real-time transaction ingestion
* Model monitoring and drift detection
* Integration with production-grade databases

---

## Author

**Yonatan** -
ML Engineer

Credit Risk Modeling Project

