import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model as lm
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configuration constants
PREDICTED_DATE = "20-Nov-24"
FILE_PATH = "results.csv"

def load_and_prepare_data(file_path) -> tuple:
    """
    Load data from CSV and prepare it for analysis

    Parameter:
        file_path (str): Path to the CSV file

    Returns:
        tuple: DataFrame and prepared X, y data for modeling
    """
    # Read and parse dates
    df = pd.read_csv(file_path, dtype={'Date': 'object'})
    df['date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

    # Prepare training data
    X = df['date_ordinal'].values.reshape(-1, 1)
    y = df['Numbers'].values

    return df, X, y

def train_naive_bayes_model(X, y) -> tuple:
    """
    Train and calibrate a Gaussian Naive Bayes model

    Parameter:
        X (numpy.array): Feature matrix
        y (numpy.array): Target values

    Returns:
        tuple: Trained GNB model and calibrated model
    """
    gnb = GaussianNB()
    gnb.fit(X, y)
    model = CalibratedClassifierCV(gnb, cv='prefit')
    model.fit(X, y)
    return gnb, model

def generate_predictions(df: DataFrame, model, gnb) -> tuple:
    """
    Generate predictions for existing dates

    Parameter:
        df (DataFrame): Input data
        model (CalibratedClassifierCV): Trained calibrated model
        gnb (GaussianNB): Trained GNB model

    Returns:
        tuple: Lists of predicted and actual values
    """
    predicted_list = []
    actual_list = []

    for date_ordinal in df['date_ordinal']:
        predicted_proba = model.predict_proba([[date_ordinal]])
        predicted_value = np.dot(predicted_proba, gnb.classes_)
        predicted_list.append(predicted_value[0])
        actual_value = df.loc[df['date_ordinal'] == date_ordinal, 'Numbers'].iloc[0]
        actual_list.append(int(actual_value))

    return predicted_list, actual_list

def train_linear_model(X, y) -> tuple:
    """
    Train a Bayesian Ridge regression model

    Parameter:
        X (numpy.array): Feature matrix
        y (numpy.array): Target values

    Returns:
        tuple: Trained model and slope
    """
    linear_model = lm.BayesianRidge()
    linear_model.fit(X, y)
    slope = linear_model.coef_[0]
    return linear_model, slope

def calculate_cyclical_adjustments(data) -> float:
    """
    Calculate cyclical adjustments based on historical patterns

    Parameter:
        data (list): List of tuples containing dates and values

    Returns:
        float: Calculated cyclical adjustment value
    """
    dates = [datetime.strptime(d, '%d-%b-%y') for d, _ in data]
    numbers = [int(v) for _, v in data]
    dates = np.array(dates)
    values = np.array(numbers)
    time_diffs = np.array([(dates[-1] - date).days for date in dates])

    cycles = [7, 30, 365]  # weekly, monthly, yearly
    predictions = []

    for cycle in cycles:
        cycle_indices = np.where(time_diffs % cycle == 0)[0]
        if len(cycle_indices) > 0:
            cycle_prediction = np.mean(values[cycle_indices])
            predictions.append(cycle_prediction)

    if predictions:
        return int(np.mean(predictions))
    else:
        ex_model = ExponentialSmoothing(values, seasonal='add', seasonal_periods=12).fit()
        forecast = ex_model.forecast(steps=1)
        return forecast.iloc[0]

def plot_results(result_df, next_date, next_value_weighted_avg, next_value_linear, cyclical_adjustment):
    """
    Create and display visualization of results

    Parameter:
        result_df (DataFrame): Results data
        next_date (datetime): Future prediction date
        next_value_weighted_avg (float): Weighted average prediction
        next_value_linear (float): Linear trend prediction
        cyclical_adjustment (float): Cyclical adjustment prediction
    """
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['ADate'], result_df['Actual'], color='purple', label='Actual', marker='o')
    plt.plot(result_df['PDate'], result_df['Predicted_LinearTrend'], color='pink',
             label='Predicted (Linear Trend)', linestyle='-.', marker='^')
    plt.plot(result_df['PDate'], result_df['Predicted_CyclicalAdjustment'], color='green',
             label='Predicted (Cyclical Adjustment)', linestyle='--', marker='*')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Actual and Value Prediction')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Annotate predictions
    plt.axvline(x=next_date, color='red', linestyle='--')
    plt.annotate(f'{int(next_value_weighted_avg):,}', (next_date, next_value_weighted_avg),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='purple')
    plt.annotate(f'{int(next_value_linear[0]):,}', (next_date, next_value_linear),
                 textcoords="offset points", xytext=(0, -20), ha='center', color='pink')
    plt.annotate(f'{int(cyclical_adjustment):,}', (next_date, cyclical_adjustment),
                 textcoords="offset points", xytext=(0, 20), ha='center', color='green')
    plt.show()

def print_predictions(next_date, next_value, next_value_linear, cyclical_adjustment, next_value_weighted_avg):
    """
    Print prediction results

    Parameter:
        next_date (datetime): Future prediction date
        next_value (float): Naive Bayes prediction
        next_value_linear (float): Linear trend prediction
        cyclical_adjustment (float): Cyclical adjustment prediction
        next_value_weighted_avg (float): Weighted average prediction
    """
    print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Naive Bayes): {int(next_value[0]):,}")
    print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Linear Trend): {int(next_value_linear[0]):,}")
    print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Cyclical Adjustment): {int(cyclical_adjustment):,}")
    print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Weighted Avg with Cyclical Adjustment): {int(next_value_weighted_avg):,}")

def main() -> None:
    """
    Main function to for the prediction process
    """

    # Model weights
    weight_naive_bayes = 0.99973895177364354734851780911949
    weight_cyclical_patterns = 8970990 #cyclical_adjustment
    weight_linear_offset = -763860
    weight_cyclical = 0.1

    # Load and prepare data
    df, X, y = load_and_prepare_data(FILE_PATH)

    # Train models
    gnb, calibrated_model = train_naive_bayes_model(X, y)
    linear_model, slope = train_linear_model(X, y)

    # Generate predictions for existing dates
    predicted_list, actual_list = generate_predictions(df, calibrated_model, gnb)

    # Predict for future date
    next_date = datetime.strptime(PREDICTED_DATE, '%d-%b-%y')
    next_date_ordinal = next_date.toordinal()
    next_proba = calibrated_model.predict_proba([[next_date_ordinal]])
    next_value = np.dot(next_proba, gnb.classes_)
    next_value_linear = linear_model.predict([[next_date_ordinal]])

    # Calculate cyclical adjustment
    data_for_cyclical_adjustment = list(zip(df['Date'], df['Numbers']))
    cyclical_adjustment = calculate_cyclical_adjustments(data_for_cyclical_adjustment)

    """
    # Calculate weighted average prediction
    next_value_weighted_avg = (
            weight_naive_bayes * cyclical_adjustment +
            weight_linear_offset +
            slope * next_value_linear[0]
    )
    """
    # Calculate Weighted Average prediction
    next_value_weighted_avg = (
            weight_naive_bayes * weight_cyclical_patterns + weight_linear_offset +  # next_value[0] +
            slope * next_value_linear[0]
    )

    # Prepare results DataFrame
    predicted_date_list = list(pd.to_datetime(df['Date'], format='%d-%b-%y')) + [next_date]
    actual_date_list = list(pd.to_datetime(df['Date'], format='%d-%b-%y')) + [next_date]

    predicted_list.append(next_value[0])
    actual_list.append(next_value_weighted_avg)

    result_df = pd.DataFrame({
        'ADate': actual_date_list,
        'Actual': actual_list,
        'PDate': predicted_date_list,
        'Predicted_NaiveBayes': predicted_list,
        'Predicted_LinearTrend': list(predicted_list[:-1]) + [next_value_linear[0]],
        'Predicted_CyclicalAdjustment': list(predicted_list[:-1]) + [cyclical_adjustment]
    })

    # Visualize and print results
    plot_results(result_df, next_date, next_value_weighted_avg, next_value_linear, cyclical_adjustment)
    print_predictions(next_date, next_value, next_value_linear, cyclical_adjustment, next_value_weighted_avg)

main()