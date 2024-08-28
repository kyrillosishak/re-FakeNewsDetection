:::{.cell .markdown}
# Temporal Leakage
:::
:::{.cell .markdown}

Time series data is a collection of data points arranged in chronological order. This type of data is common across various fields, such as finance, where stock prices fluctuate over time, and weather forecasting, where temperature readings are recorded in sequence.

<img src="https://github.com/kyrillosishak/re-FakeNewsDetection/raw/main/assets/TimeSeries_DataLeaked.png" height = 300>

Imagine you have 1,000 samples of Microsoft’s (MSFT) stock price throughout a single day. You might be tempted to randomly select 500 samples for your training set and reserve the rest for validation. However, this approach can make your model appear astonishingly accurate, as though it possesses an uncanny ability to predict Microsoft’s future stock prices. The reason? Your model has inadvertently seen the future—the training set includes data from both before and after nearly every point in the validation set. From the perspective of the validation data, information from the future has leaked into your model, undermining the integrity of your evaluation.

To illustrate the idea consider this Scenario:

* In which an e-commerce platform collects data on users, including their profiles, purchase histories, and product ratings. The goal is to demonstrate how data leakage can occur when building a machine learning model to predict product ratings.

* In this scenario, the platform has data on users’ ages, gift card balances, and purchase quantities over time. Additionally, each user has given ratings to certain products. 
* *we will create synthetic datasets for user profiles, purchase histories, and product ratings*.

* *The key point of the scenario is that the purchase quantities are time-dependent features, meaning they change over time. When trying to predict product ratings using these features, it is crucial to ensure that the model does not have access to information from the future, such as purchases made after the rating was given.*

Then we create two different datasets:

1. With Leakage: This dataset includes purchase information from both before and after the rating was given, leading to potential leakage of future information into the model.
2. Without Leakage: This dataset carefully removes any purchase data that occurs after the rating was given, ensuring no future information is leaked into the model.


:::

:::{.cell .code}
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 user profiles
n_samples = 50000
user_ids = np.arange(1, n_samples + 1)
ages = np.random.randint(18, 70, size=n_samples)
gift_card_balances = np.random.randint(0, 2000, size=n_samples)

user_profile_data = pd.DataFrame({
    'user_id': user_ids,
    'age': ages,
    'gift_card_balance': gift_card_balances
})

# Generate purchase history with multiple entries per user
purchase_times = pd.date_range(start='2022-01-01', end='2022-12-31', periods=n_samples*3)
purchase_quantities = np.random.randint(1, 300, size=n_samples*3)
purchase_user_ids = np.random.choice(user_ids, size=n_samples*3)

user_purchase_history = pd.DataFrame({
    'user_id': purchase_user_ids,
    'purchase_time': np.random.choice(purchase_times, size=n_samples*3),
    'purchase_quantity': purchase_quantities
})

# Generate observation data (product ratings)
event_times = pd.date_range(start='2022-01-01', end='2022-12-31', periods=n_samples)
# Introduce a strong correlation between future purchases and product rating
product_ratings = np.random.randint(1, 6, size=n_samples) + \
                  np.array([np.sum(user_purchase_history[
                    (user_purchase_history['user_id'] == user_id) &
                    (user_purchase_history['purchase_time'] > event_time)]['purchase_quantity']) / 100
                    for user_id, event_time in zip(user_ids, event_times)])

# Clip ratings to be between 1 and 5
product_ratings = np.clip(product_ratings, 1, 5)

user_observation_data = pd.DataFrame({
    'user_id': user_ids,
    'event_time': event_times,
    'Product_rating': product_ratings
})

```
:::

:::{.cell .code}
```python
# Merge user profile data with observation data
merged_data = pd.merge(user_observation_data, user_profile_data, on='user_id')

# Introduce potential leakage by joining purchase history without time restriction
merged_data_leak = pd.merge(merged_data, user_purchase_history, on='user_id', how='left')

# Filter to remove future purchases in no-leakage scenario
merged_data_no_leak = merged_data_leak[merged_data_leak['purchase_time'] <= merged_data_leak['event_time']]

# Aggregate purchase quantities for no-leakage scenario
agg_data_no_leak = merged_data_no_leak.groupby(['user_id', 'event_time', 'age', 'gift_card_balance']).agg({
    'purchase_quantity': 'sum'
}).reset_index()

# Aggregate purchase quantities for leakage scenario (includes future data)
agg_data_leak = merged_data_leak.groupby(['user_id', 'event_time', 'age', 'gift_card_balance']).agg({
    'purchase_quantity': 'sum'
}).reset_index()

# Merge with the original ratings
agg_data_no_leak = pd.merge(agg_data_no_leak, user_observation_data, on=['user_id', 'event_time'])
agg_data_leak = pd.merge(agg_data_leak, user_observation_data, on=['user_id', 'event_time'])

# Extract features and labels
X_leak = agg_data_leak[['age', 'gift_card_balance', 'purchase_quantity']]
y_leak = agg_data_leak['Product_rating']

X_no_leak = agg_data_no_leak[['age', 'gift_card_balance', 'purchase_quantity']]
y_no_leak = agg_data_no_leak['Product_rating']

```
:::
:::{.cell .markdown}

Then we train two linear regression models, one on each dataset, and evaluates their performance using Mean Squared Error (MSE). The results highlight how data leakage can lead to misleadingly good model performance during training, which may not translate to real-world accuracy. This underscores the importance of handling time-dependent features carefully to avoid introducing bias into the model.

:::

:::{.cell .code}
```python
# Initialize Linear Regression models
model_leak = LinearRegression()
model_no_leak = LinearRegression()

# Train the model with leakage
model_leak.fit(X_leak, y_leak)

# Train the model without leakage
model_no_leak.fit(X_no_leak, y_no_leak)

# Make predictions
y_pred_leak = model_leak.predict(X_leak)
y_pred_no_leak = model_no_leak.predict(X_no_leak)

# Evaluate models
mse_leak = mean_squared_error(y_leak, y_pred_leak)
mse_no_leak = mean_squared_error(y_no_leak, y_pred_no_leak)

print(f'MSE with Leakage: {mse_leak}')
print(f'MSE without Leakage: {mse_no_leak}')
```
:::
