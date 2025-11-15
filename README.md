# Jamboree Admission Prediction 🎓

This repository contains a data-driven business case analysis for **Jamboree Education**. The project focuses on building a predictive model to determine a student's **"Chance of Admit"** based on their academic and personal profile.

The analysis dives into the data to understand which factors are most important for admission success, ultimately building several regression models to find the most accurate predictor.

---

## 🎯 Project Goal

The primary objective is to develop a machine learning model that accurately predicts a student's likelihood of admission. This model helps answer a key business question: **"What are the most significant factors driving a successful university admission?"**

## 🛠️ Tech Stack

This project leverages the core Python data science and machine learning stack:

* **Data Analysis:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Statistical Modeling:** `statsmodels`
* **Machine Learning:** `scikit-learn`

---

## 🚀 Project Workflow

This analysis follows a structured machine learning workflow:

### 1. Data Cleaning & Exploration (EDA)
* Loaded the `Jamboree_Admission.csv` dataset.
* Checked for and handled any missing values or duplicate entries.
* Performed a deep dive into the data using visualizations:
    * **Distributions:** Used histograms and box plots to understand the spread of features like `GRE Score`, `TOEFL Score`, and `CGPA`.
    * **Relationships:** Used scatter plots and a comprehensive pair plot to see how variables interact.
    * **Correlations:** Generated a heatmap to quantify the linear relationships between features.

### 2. Preprocessing & Feature Engineering
* Identified and treated outliers to ensure they didn't skew the model.
* Split the dataset into training and testing sets to prepare for model building.
* Checked for **multicollinearity** using VIF (Variance Inflation Factor) to ensure model stability.

### 3. Model Building
Several regression models were built and trained to find the best fit:

* **OLS (Ordinary Least Squares):** A statistical model from `statsmodels` to get detailed insights into feature importance, p-values, and R-squared.
* **Linear Regression:** The standard `scikit-learn` model as a baseline.
* **Ridge Regression (L2):** A regularized model to prevent overfitting by penalizing large coefficients.
* **Lasso Regression (L1):** A regularized model that can also perform feature selection by shrinking irrelevant feature coefficients to zero.

### 4. Model Evaluation & Selection
* All models were compared based on two key metrics:
    * **R-squared ($R^2$):** Measures how much of the variance in the "Chance of Admit" is explained by the model.
    * **Root Mean Squared Error (RMSE):** Measures the average error of the model's predictions.
* The model with the best balance of high R-squared and low RMSE was selected as the final, most reliable predictor.

---

## 📂 How to Use

To run this analysis yourself:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook "Jamboree Business Case.ipynb"
    ```
