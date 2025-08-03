# Time Series Forecasting with Ensemble Models
This project uses a Python to forecast a future numerical value based on historical time-series data. It uses an ensemble of several machine learning and statistical models to capture different aspects of the data, such as trends and cyclical patterns. The final prediction is visualized alongside the historical data.

# Overview
The python project reads a dataset from a CSV file (results.csv), which contains dates and corresponding numerical values. The data is use to train multiple models on this data to predict a value for a user-defined future date. The core of this project is its hybrid approach, combining predictions from different models to produce a more robust forecast.

# Features
- Data Handling: Loads and preprocesses time-series data using pandas.
- Multi-Model Approach: Utilizes several models to analyze the data from different perspectives:
- Bayesian Ridge Regression: To identify the long-term linear trend in the data.
- Custom Cyclical Analysis: To detect and leverage weekly, monthly, and yearly patterns.
- Holt-Winters Exponential Smoothing: As a fallback for the cyclical analysis.
- Gaussian Naive Bayes: A calibrated probabilistic model trained on the data.
- Ensemble Prediction: Combines outputs from the linear and cyclical models using a weighted formula to generate a final forecast.
- Visualization: Plots the historical data, model predictions, and the final forecast using matplotlib for easy comparison and analysis.

# Actual model predictions
To compare the effectiveness of the Naive Bayes model, Linear Trend, and a Weighted Average combination for predicting future values.

# Dependencies
~~~~~~~~~~~~

requires scikit-learn:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

User installation
~~~~~~~~~~~~~~~~~
Install pandas is using ::

    pip install pandas
    or
    sudo apt-get install python3-pandas

Install matplotlib is using ::

    pip install matplotlib
    or
    sudo apt-get install python3-matplotlib

Install statsmodels is using ::

    pip install statsmodels
    or
    sudo apt-get install python3-statsmodels

Install scikit-learn is using ::


    pip install -U scikit-learn
    or
    sudo apt-get install python3-sklearn

Testing
~~~~~~~

After installation, you can run the test data::

    python actual_model_predictions.py

Example results.csv:

Code snippet

Date,Numbers
01-Jan-24,1500
02-Jan-24,1520

Bash
python your_script_name.py
The script will display a plot and print the detailed prediction values to the console.
