import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from datetime import datetime, timedelta

# Data preparation
data = {
    "date":["22-Jun-24","26-Jun-24","29-Jun-24","03-Jul-24","06-Jul-24","10-Jul-24","13-Jul-24","17-Jul-24","20-Jul-24","24-Jul-24","27-Jul-24","31-Jul-24","03-Aug-24"],
    "value":[13437644,2310428,7032157,1760406,5854964,6439818,8204344,6746572,10834075,748018,6849481,11459902,4082858]
}

# Convert data to DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')

# Convert dates to ordinal for regression
df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())

# Prepare training data
X = df['date_ordinal'].values.reshape(-1, 1)
y = df['value'].values

# Train the Gaussian Naive Bayes model with calibration
gnb = GaussianNB()
gnb.fit(X, y)

model = CalibratedClassifierCV(gnb, cv="prefit")
model.fit(X, y)

# Loop through the dates to create a list of predicted values
predicted_list = []
actual_list = []

for date_ordinal in df['date_ordinal']:
    predicted_proba = model.predict_proba([[date_ordinal]])
    predicted_value = np.dot(predicted_proba, gnb.classes_)  # Calculate the expected value
    predicted_list.append(predicted_value[0])  # Append the expected value, not the entire array
    actual_value = df.loc[df['date_ordinal'] == date_ordinal, 'value'].iloc[0]
    actual_list.append(int(actual_value))

# Predict for a future date using the Naive Bayes model
next_date = datetime.strptime("07-Aug-24", '%d-%b-%y')
next_date_ordinal = next_date.toordinal()
next_proba = model.predict_proba([[next_date_ordinal]])
next_value = np.dot(next_proba, gnb.classes_)  # Calculate the expected value for the next date

# Train the model Linear Regression to predict the next value using the linear trend
linear_model = linear_model.BayesianRidge()
linear_model.fit(X, y)

# Predict using the linear trend model
next_value_linear = linear_model.predict([[next_date_ordinal]])

# Calculate Weighted Average prediction
weight_naive_bayes = 0.8  # Adjust the weights
weight_linear_trend = 0.4
next_value_weighted_avg = (
        weight_naive_bayes * next_value[0] +
        weight_linear_trend * next_value_linear[0]
)

# Add the next prediction
predicted_list.append(next_value[0])
actual_list.append(next_value_weighted_avg)  # Guess an actual value for the future date

# Append the future date to the date list
predicted_date = list(df['date']) + [next_date]
actual_date = list(df['date']) + [next_date]

# Create a DataFrame with the actual and predicted values
result_df = pd.DataFrame({
    'ADate': actual_date,
    'Actual': actual_list,
    'PDate': predicted_date,
    'Predicted_NaiveBayes': predicted_list,
    'Predicted_LinearTrend': list(predicted_list[:-1]) + [next_value_linear[0]]  # Use linear trend for the last prediction
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
plt.annotate(f'{int(next_value[0]):,}', (next_date, next_value), textcoords="offset points", xytext=(0, 10), ha='center', color='orange')
plt.annotate(f'{int(next_value_linear[0]):,}', (next_date, next_value_linear), textcoords="offset points", xytext=(0, -20), ha='center', color='red')

# Show plotted chart
plt.show()

# Print the predicted next values
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Naive Bayes): {int(next_value[0]):,}")
print(f"Predicted value for {next_date.strftime('%d-%b-%y')} (Linear Trend): {int(next_value_linear[0]):,}")
