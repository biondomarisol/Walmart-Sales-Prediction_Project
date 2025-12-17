# ------------------------------------------------------------
# PROJECT: Walmart Weekly Sales Prediction
# MODELS: Linear Regression and Multi-Layer Perceptron (MLP)
# ------------------------------------------------------------

# INSTALL LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. DATA LOADING AND PREPARATION 
# ------------------------------------------------------------

# Dataset reading
data = pd.read_csv("Walmart_Store_sales.csv")
# Save a copy for the charts (Time Series)
data_eda = data.copy()

# Convert date into a datetime object
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Feature engineering: extract WeekOfYear, Month, DayOfWeek
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek

# One-Hot Encoding: create dummy variables for 'Store'
data = pd.get_dummies(data, columns=['Store'], drop_first=True)

# Continuous colums to be standardized
continuous_cols = [
    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'WeekOfYear', 'Month', 'DayOfWeek'
]

# All feature used by the models 
feature_cols = ['Holiday_Flag'] + continuous_cols + \
               [c for c in data.columns if c.startswith('Store_')]

# Prepare X (the features) and Y (the target)
X = data[feature_cols].copy()
y = np.log1p(data['Weekly_Sales'])     # log transformation to reduce variability 

# Standardize continuos features 
scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

# Store the final feature names after encoding
final_features = X.columns.tolist()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# 2. MODELS TRAINING
# ------------------------------------------------------------

# 2a. Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# 2b. Multi-Layer Perceptron (MLP)
model_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
model_mlp.fit(X_train, y_train)

# ------------------------------------------------------------
# 3. MODELS EVALUATION
# ------------------------------------------------------------

# Convert log-transformed back to real values for evaluation
y_test_real = np.expm1(y_test)
y_train_real = np.expm1(y_train)

# Test set predictions
pred_lr_test = np.expm1(model_lr.predict(X_test))
pred_mlp_test = np.expm1(model_mlp.predict(X_test))

# Training set predictions
pred_lr_train = np.expm1(model_lr.predict(X_train))
pred_mlp_train = np.expm1(model_mlp.predict(X_train))

print("\n--- LINEAR REGRESSION RESULTS  ---")
print("R² (Test):", r2_score(y_test_real, pred_lr_test))
print("MSE (Test):", mean_squared_error(y_test_real, pred_lr_test))
print("R² (Train):", r2_score(y_train_real, pred_lr_train))
print("MSE (Train):", mean_squared_error(y_train_real, pred_lr_train))

print("\n--- MLP RESULTS ---")
print("R² (Test):", r2_score(y_test_real, pred_mlp_test))
print("R² (Train):", r2_score(y_train_real, pred_mlp_train))
print("MSE (Test):", mean_squared_error(y_test_real, pred_mlp_test))
print("MSE (Train):", mean_squared_error(y_train_real, pred_mlp_train))

# ------------------------------------------------------------
# 4. LINEAR REGRESSION COEFFICIENTS OUTPUT
# ------------------------------------------------------------

coef_df = pd.DataFrame({
    'Feature': final_features,
    'Coefficient': model_lr.coef_
})

print("\n--- LINEAR MODEL COEFFICIENTS OUTPUT ---")
print(coef_df.to_string(index=False))
print("\nIntercept:", model_lr.intercept_)

# ------------------------------------------------------------
# 5. PREDICTION FUNCTION FOR NEW DATA
# ------------------------------------------------------------

def predict_new(input_dict, model_to_use):
    """
    input_dict = new observation values dictionary
    model_to_use = model to use (model_lr o model_mlp)
    """

    new = pd.DataFrame([input_dict])

    # One-Hot Encoding: create dummy variables for 'Store'
    new = pd.get_dummies(new, columns=['Store'], prefix='Store')

    # Add missing columns with 0 values
    for col in final_features:
        if col not in new.columns:
            new[col] = 0

    # Reorder colums to match training data
    new = new[final_features]

    # Standardize continuos features
    new[continuous_cols] = scaler.transform(new[continuous_cols])

    # Prediction
    pred_log = model_to_use.predict(new)
    return np.expm1(pred_log)[0]

# ------------------------------------------------------------
# 6. EXAMPLE OF PREDICTION (INTERACTIVE)
# ------------------------------------------------------------

print("\n--- ENTER VALUES FOR PREDICTION ---")

# Initialize the dictionary for the new data
new_example = {}

try:
    # Collect input values from the user
    new_example['Store'] = int(input("Store number (es. 1, 2, ...): "))
    new_example['Holiday_Flag'] = int(input("Is it a holiday week? (1 for Yes, 0 for No): "))
    new_example['Temperature'] = float(input("Average temperature (°F): "))
    new_example['Fuel_Price'] = float(input("Fuel price (USD/gallon): "))
    new_example['CPI'] = float(input("CPI (Consumer Price Index): "))
    new_example['Unemployment'] = float(input("Unemployment rate (%): "))
    
    # For simplicity, ask the user for the full date and extract features automatically
    date_str = input("Week date (format dd-mm-yyyy, e.g. 10-03-2012): ")
    
    # Extract date-related features
    input_date = pd.to_datetime(date_str, format='%d-%m-%Y')
    new_example['WeekOfYear'] = input_date.isocalendar().week
    new_example['Month'] = input_date.month
    new_example['DayOfWeek'] = input_date.dayofweek # Monday=0, Sunday=6
    
    # Convert WeekOfYear from scalar to standard integer (required for scaling)
    new_example['WeekOfYear'] = int(new_example['WeekOfYear'])
    
    print("\nProcessing...")
    
    # Example prediction with the entered values
    print(f"Prediction (Linear Regression) for Store {new_example['Store']}: {predict_new(new_example, model_lr):,.2f}")
    print(f"Prediction (MLP) for Store {new_example['Store']}: {predict_new(new_example, model_mlp):,.2f}")

except ValueError:
    print("\nERROR: Make sure to enter numeric values for all numeric fields and use the correct date format.")

# ------------------------------------------------------------
# 7. CHARTS FOR MODEL ACCURACY AND DATA INSIGHTS
# ------------------------------------------------------------

# ---- 1. ACTUAL vs PREDICTED VALUES PLOTS ----
# Find the minimum and maximum value to define the ideal prediction line (we use a fixed max for consistency)
min_val = y_test_real.min()
# Define a maximum limit for the X and Y axes (Zoom) to improve readability of the plot 
max_display_val = max(y_test_real.max(), pred_lr_test.max(), pred_mlp_test.max())
# Set a zoom limit (e.g., 2.5 million) for better visualization
ZOOM_LIMIT = 2.5e6

plt.figure(figsize=(14, 6))

# 1a SUBPLOT: Linear Regression (LR) 
plt.subplot(1, 2, 1)
# Draw the ideal prediction line (y = x)
plt.plot([min_val, max_display_val], [min_val, max_display_val], color='red', linestyle='--', label='Ideal Prediction')
# Scatter plot for LR
plt.scatter(y_test_real, pred_lr_test, label='Linear Regression', alpha=0.7, color='skyblue')
# Apply zoom for readability
plt.xlim(0, ZOOM_LIMIT)
plt.ylim(0, ZOOM_LIMIT)
plt.xlabel("Actual Values ($)")
plt.ylabel("Predicted Values ($)")
plt.title("Prediction Accuracy: Linear Regression")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# 1b SUBPLOT: Neural Network (MLP) 
plt.subplot(1, 2, 2)
# Draw the ideal prediction line (y = x)
plt.plot([min_val, max_display_val], [min_val, max_display_val], color='red', linestyle='--', label='Ideal Prediction')
# Scatter plot per MLP
plt.scatter(y_test_real, pred_mlp_test, label='MLP', alpha=0.7, color='green')
# Apply zoom for readability
plt.xlim(0, ZOOM_LIMIT)
plt.ylim(0, ZOOM_LIMIT)
plt.xlabel("Actual Values ($)")
plt.ylabel("Predicted Values ($)")
plt.title("Prediction Accuracy: Neural Network (MLP)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()


# ---- 2. RESIDUAL PLOTS ----
plt.figure(figsize=(12, 5))

# 2a Residuals for Linear Regression (LR)
residuals_lr = y_test_real - pred_lr_test # Calculate residuals: Actual - Predicted
plt.subplot(1, 2, 1)
plt.scatter(pred_lr_test, residuals_lr, alpha=0.5, color='skyblue') # Scatter plot: Predicted values (X) vs Residuals (Y)
plt.hlines(0, pred_lr_test.min(), pred_lr_test.max(), color='red', linestyle='--') # Draw the Y=0 reference line
plt.title('Residuals vs Predictions (Linear Regression)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# 2b. Residuals for Multi-Layer Perceptron (MLP)
residuals_mlp = y_test_real - pred_mlp_test # Calculate residuals: Actual - Predicted
plt.subplot(1, 2, 2)
plt.scatter(pred_mlp_test, residuals_mlp, alpha=0.5, color='green') # Scatter plot: Predicted values (X) vs Residuals (Y)
plt.hlines(0, pred_mlp_test.min(), pred_mlp_test.max(), color='red', linestyle='--') # Draw the Y=0 reference line
plt.title('Residuals vs Predictions (MLP)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.tight_layout()
plt.show()


# ---- 3. DATA INSIGHT CHARTS: Comparison of Average Sales by Holiday_Flag ----
# Group the average sales by the holiday flag
holiday_comparison = data_eda.groupby('Holiday_Flag')['Weekly_Sales'].mean()
plt.figure(figsize=(8, 6))
# Rename labels for clarity
labels = ['Normal Week (0)', 'Holiday Week (1)']
bars = plt.bar(labels, holiday_comparison.values, color=['skyblue', 'lightcoral'])
# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    # Position the text $100,000 above the x-axis to be visible inside the bar
    plt.text(bar.get_x() + bar.get_width()/2, 100000,
             f'${yval:,.0f}',
             ha='center', va='bottom', fontsize=12, color='black', weight='bold')
plt.title('Average Weekly Sales: Holiday vs. Normal Weeks')
plt.xlabel('Holiday_Flag')
plt.ylabel('Average Weekly Sales ($)')
# # Set the y-axis limit to provide more space above the bars
plt.ylim(0, holiday_comparison.max() * 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ---- 4. DATA INSIGHT CHARTS: Seasonal Sales Trend ----

# Map to convert the week number to an approximate month name (for labels)
month_mapping = {
    4: 'JAN', 8: 'Feb', 13: 'Mar', 17: 'Apr', 22: 'MaY', 26: 'Jun',
    30: 'Jul', 35: 'Aug', 39: 'Seo', 44: 'Oct', 48: 'Nov', 52: 'Dec'
}
# Group the average sales by the Week of the Year
seasonal_sales = data.groupby('WeekOfYear')['Weekly_Sales'].mean().reset_index()

plt.figure(figsize=(15, 6))
plt.plot(seasonal_sales['WeekOfYear'], seasonal_sales['Weekly_Sales'],
         marker='o', linestyle='-', color='darkgreen')

# Set the X-axis labels with the Months
week_ticks = list(month_mapping.keys())
month_labels = list(month_mapping.values())
plt.xticks(week_ticks, month_labels, rotation=0)

plt.title('Seasonal Sales Trend (Average Sales per Week)')
plt.xlabel('Period of the Year')
plt.ylabel('Average Weekly Sales ($)')
plt.grid(True, linestyle='--', alpha=0.7)

#Highlight peaks with HOLIDAY NAMES for immediate understanding
# NOTE: These are the standard weeks for major US holidays in the period (WalMart data)
peak_annotations = {
    # Circa 47: Thanksgiving
    47: {'name': 'Thanksgiving', 'color': 'red'},
    # Circa 50: Black Friday/Pre-Christmas)
    50: {'name': 'Black Friday', 'color': 'red'},
    # Circa 51: Christmas/New Year
    51: {'name': 'Christmas Week', 'color': 'red'}
}
for week, info in peak_annotations.items():
    # Find the sales value for that specific week
    peak_value = seasonal_sales[seasonal_sales['WeekOfYear'] == week]['Weekly_Sales'].iloc[0]

    plt.scatter(week, peak_value, color=info['color'], s=150, zorder=5)
    plt.annotate(
        info['name'],
        (week, peak_value),
        textcoords="offset points",
        xytext=(0, 15), 
        ha='center',
        color=info['color'],
        weight='bold',
        fontsize=10
    )

plt.show()
