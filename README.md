# Walmart weekly sales prediction model

> A Machine Learning project to forecast the weekly sales of Walmart stores. This project compares the effectiveness of a **Linear Regression** model against a **Neural Network (MLP)**, featuring a detailed analysis of how holidays and seasonality impact revenues.

## Project goal

The primary objective of this project is to develop a robust predictive model to estimate the **Weekly\_Sales** for 45 different Walmart stores.
The analysis aims to identify the key factors influencing sales, giving particular emphasis to:
1.  **Seasonality:** the effect of temporal features like week of the year and month.
2.  **Holidays:** the sharp impact of major US holiday weeks (e.g., Thanksgiving, Christmas) on revenue.
3.  **Macroeconomic variables:** the influence of factors such as inflation (CPI), fuel price, and the unemployment rate on consumer spending.

## Core implementation details

The `Prediction_model.py` script implements the full Machine Learning pipeline, ensuring a comparison between a baseline statistical model and a deep learning approach.

### Feature engineering and preprocessing

* **Date transformation:** the `Date` column is converted into a datetime object to extract crucial temporal features: `WeekOfYear` and `Month`.
* **Store encoding:** the categorical variable `Store` is processed using **One-Hot Encoding** to prevent the model from inferring incorrect ordinal relationships.
* **Data scaling:** the continuous features (including CPI, unemployment, and the newly engineered features) are scaled using `StandardScaler`. This is a critical step to ensure optimal performance, especially for the **MLP Regressor**.

### Models compared 

Two models were trained and evaluated to assess performance and complexity trade-offs:

| Model | Type | Rationale |
| :--- | :--- | :--- |
| **Linear Regression** | Benchmark statistical model | Provides a transparent baseline to understand the linear relationships between features and sales |
| **MLP Regressor** | Neural Network | Used to capture complex, non-linear interactions within the dataset, potentially leading to higher accuracy |

## Data utilized

The project relies on the `Walmart_Store_sales.csv` dataset, which covers weekly sales and relevant economic data from 2010 to 2012.

| Column | Type | Role |
| :--- | :--- | :--- |
| **Weekly\_Sales** | Float | **Target** Variable |
| **Store** | Int | Store ID |
| **Date** | Date | Sales date (used for feature engineering) |
| **Holiday\_Flag** | Int | 1 if it is a holiday week, 0 otherwise |
| **Temperature** | Float | Average temperature in the region |
| **Fuel\_Price** | Float | Fuel price in the region |
| **CPI** | Float | Consumer price index (inflation) |
| **Unemployment** | Float | Unemployment rate |

---

## Getting started

Follow these steps to configure and run the project locally.

### Prerequisites

Ensure you have Python (version 3.x) installed and are using a virtual environment (e.g., Conda) to isolate dependencies.

### Environment Setup

1. **Clone the GitHub repository:**
    ```bash
    git clone [https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git](https://github.com/biondomarisol/Walmart-Sales-Prediction-Project.git)
    ```
2. **Navigate to the project directory:**
    ```bash
    cd Walmart-Sales-Prediction-Project
    ```
3. **Activate the virtual environment (e.g., Conda):**
    ```bash
    conda activate [your_environment_name]
    ```
4. **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    # Note: TensorFlow is required for the MLP Neural Network model.
    pip install tensorflow
    ```

## Execution

Run the main Python script to perform the data processing, model training, and evaluation:

```bash
python Prediction_model.py
