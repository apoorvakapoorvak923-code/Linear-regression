# Linear-regression
Overview

This Python script demonstrates a simple/multiple linear regression workflow on a housing dataset. 
It is designed to handle small or medium-sized datasets and includes feature scaling, missing value handling, model evaluation, and plotting.
The script predicts a numeric target variable (e.g., house prices) using available numeric features from the dataset.

Features
Automatic target detection: Supports common target column names like median_house_value, SalePrice, price, Price, target.
Numeric feature selection: Uses all numeric columns (excluding the target) as input features.
Missing value handling: Replaces missing values with the median of each column.
Train/test split:
Standard split (80/20) for datasets larger than 5 rows.
For very small datasets (≤5 rows), uses all data for training and evaluation.
Feature scaling: Applies StandardScaler to normalize feature ranges and improve model stability.
Linear Regression:
Fits a linear regression model to the data.Calculates predictions on the test set.Evaluation metrics:
MAE (Mean Absolute Error)
MSE (Mean Squared Error)
R² (Coefficient of Determination)
Coefficient analysis: Displays top features sorted by absolute value of their coefficients.

Visualization:
Single feature: scatter plot with actual vs predicted values.
Multiple features: scatter plot of predicted vs actual values with reference line.
Results saving: Plots are saved to the results/ folder automatically.

Requirements
Python 3.8+
Libraries:
pandas
numpy
matplotlib
scikit-learn

Install missing libraries with:
pip install pandas numpy matplotlib scikit-learn
Usage
Place your dataset in the data/ folder, e.g., data/housing.csv.
Edit the DATA_PATH variable in the script if your file has a different name or location.
Run the script:
python linear_regression.py

Output:
Console:
Dataset shape and columns.
Numeric features used.
Evaluation metrics (MAE, MSE, R²).
Top coefficients (sorted by absolute value).
Plots saved to results/:
simple_regression.png (if single numeric feature)
pred_vs_actual.png (if multiple features)


Top coefficients (abs sorted):
households            3.124688e+06
total_bedrooms       -2.275477e+06
population           -8.583588e+05
longitude             3.522756e+05
housing_median_age    3.157217e+05
median_income        -2.666006e+05
latitude              2.302795e+05
total_rooms          -1.562314e+04
Plots are saved in results/ as pred_vs_actual.png or simple_regression.png.

Notes
For tiny datasets, metrics like R² may be unreliable.
If features are highly correlated (multicollinearity), coefficient signs may seem counterintuitive.
Scaling features is crucial to avoid huge coefficients and improve model stability.
You can customize the target variable by editing the TARGET_CANDIDATES list or setting target manually.

Future Improvements
Feature selection for small datasets (use only top predictors).
Cross-validation for robust evaluation metrics.
Handling categorical features using one-hot encoding.
Hyperparameter tuning for regularized models (Ridge, Lasso) to reduce overfitting.


