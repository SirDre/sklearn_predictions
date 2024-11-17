import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

# Default variables
predicted_date = "20-Nov-24"
predicted_list = []
actual_list = []

# Read the data from the CSV file and parse dates
df = pd.read_csv("results.csv", dtype={'Date': 'object'})

# Convert data to DataFrame
df['date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')

# Convert dates to ordinal for regression
df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

# Prepare training data
X = df['date_ordinal'].values.reshape(-1, 1)
y = df['Numbers'].values

# Train the Gaussian Naive Bayes model with calibration
gnb = GaussianNB()
gnb.fit(X, y)

# Calibrate the model
model = CalibratedClassifierCV(gnb, cv="prefit")
model.fit(X, y)

# loop
for date_ordinal in df['date_ordinal']:
    predicted_proba = model.predict_proba([[date_ordinal]])
    predicted_value = np.dot(predicted_proba, gnb.classes_)  # Calculate the expected value
    predicted_list.append(predicted_value[0])  # Append the expected value, not the entire array
    actual_value = df.loc[df['date_ordinal'] == date_ordinal, 'Numbers'].iloc[0]
    actual_list.append(int(actual_value))

# Predict for a future date using the Naive Bayes model
next_date = datetime.strptime(predicted_date, '%d-%b-%y')
next_date_ordinal = next_date.toordinal()
next_proba = model.predict_proba([[next_date_ordinal]])
next_value = np.dot(next_proba, gnb.classes_)  # Calculate the expected value for the next date

# Train the model Linear Regression to predict the next value using the linear trend
# linear_model = lm.LinearRegression()
# linear_model = lm.TheilSenRegressor()
# linear_model = lm.LogisticRegression(C=1.0)
# linear_model = lm.GammaRegressor()
# linear_model = RandomForestRegressor(n_estimators=9, random_state=0)
linear_model = lm.BayesianRidge()
linear_model.fit(X, y)

# Predict using the linear trend model
next_value_linear = linear_model.predict([[next_date_ordinal]])

# Calculate the slope predicted by the linear regression model
slope = linear_model.coef_[0]

# A sine function to the data for cyclical adjustments
def sinusoidal_model(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Wrapper to convert dates
x_data = df['date_ordinal']
y_data = df['Numbers']

# Fit the curve
params, _ = curve_fit(sinusoidal_model, x_data, y_data, p0=[np.std(y), 2 * np.pi / 365, 0, np.mean(y)])

# Predicting the cyclical adjustment for the given future date
cyclical_adjustment = sinusoidal_model(next_date_ordinal, *params)

# Calculate Weighted Average prediction
weight_naive_bayes = 0.99973895177364354734851780911949  # you can adjust these
weight_linear_trend = slope
weight_linear_offset = 494509  # you can adjust these
weight_cyclical_patterns = cyclical_adjustment  # you can adjust these
weight_cyclical = 0.1
"""
next_value_weighted_avg = (
        weight_naive_bayes * next_value[0] +
        weight_linear_trend * next_value_linear[0] +
        weight_cyclical * cyclical_adjustment
)
"""
next_value_weighted_avg = (
        weight_naive_bayes * weight_cyclical_patterns + weight_linear_offset +
        weight_linear_trend * next_value_linear[0]
)

# Add the next prediction
predicted_list.append(next_value[0])
actual_list.append(next_value_weighted_avg)  # guess an actual value for the future date

# Append the future date to the date list as datetime objects
predicted_date = list(pd.to_datetime(df['Date'], format='%d-%b-%y')) + [next_date]
actual_date = list(pd.to_datetime(df['Date'], format='%d-%b-%y')) + [next_date]

# Create a DataFrame with the actual and predicted values
result_df = pd.DataFrame({
    'ADate': actual_date,
    'Actual': actual_list,
    'PDate': predicted_date,
    'Predicted_NaiveBayes': predicted_list,
    'Predicted_LinearTrend': list(predicted_list[:-1]) + [next_value_linear[0]],  # Use linear trend for the last prediction
    'Predicted_CyclicalAdjustment': list(predicted_list[:-1]) + [cyclical_adjustment]  # Use cyclical adjustment for another prediction
})

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(result_df['ADate'], result_df['Actual'], color='purple', label='Actual', marker='o')
plt.plot(result_df['PDate'], result_df['Predicted_LinearTrend'], color='pink', label='Predicted (Linear Trend)', linestyle='-.', marker='^')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual and Value Prediction')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Annotate the predicted points
plt.axvline(x=next_date, color='red', linestyle='--')
plt.annotate(f'{int(next_value_weighted_avg):,}', (next_date, next_value_weighted_avg), textcoords="offset points", xytext=(0, 10), ha='center', color='orange')
plt.annotate(f'{int(next_value_linear[0]):,}', (next_date, next_value_linear), textcoords="offset points", xytext=(0, -20), ha='center', color='red')
plt.annotate(f'{int(cyclical_adjustment):,}', (next_date, cyclical_adjustment), textcoords="offset points", xytext=(0, 20), ha='center', color='green')

plt.show()

# Print the predicted next values
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Naive Bayes): {int(next_value[0]):,}")
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Linear Trend): {int(next_value_linear[0]):,}")
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Cyclical Adjustment): {int(cyclical_adjustment):,}")
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Weighted Avg with Cyclical Adjustment): {int(next_value_weighted_avg):,}")
