:::{.cell .markdown}

## Using a feature that is a proxy for target variable, that will not be available at inference time

:::

:::{.cell .markdown}

One specific form of data leakage involves selecting features that directly influence the target variable, thereby giving the model access to information it would not have in a real-world scenario. For example, if a feature is chosen based on its direct correlation with the target variable—perhaps through data that includes future outcomes or results—this feature will unfairly boost the model’s accuracy during training. When the model is evaluated on unseen data, it may not perform as well because it has been trained with features that directly reveal the target variable. This scenario creates an unrealistic impression of the model's capabilities and can lead to misleading conclusions about its effectiveness, ultimately compromising its reliability and generalization to new, real-world data.

:::

:::{.cell .markdown}

### Consider this example:

:::

:::{.cell .markdown}

<img src="https://github.com/kyrillosishak/re-FakeNewsDetection/raw/main/Assets/features.png" height=130>

If the feature `took_antibiotic_medicine` is updated based on whether a patient had pneumonia `got_pneumonia`, it creates target leakage. This is because the model learns that if took_antibiotic_medicine is False, the patient didn't have pneumonia. During validation, this pattern repeats, leading to misleadingly high performance. However, in real-world scenarios, predictions are made before knowing if a patient has received antibiotics, leading to poor model accuracy. To avoid this, exclude features updated or created after the target variable is known.

*To illustrate the problem we are considering `Credit Card Data from book "Econometric Analysis"` Dataset from Kaggle, it has this features:*
* `card`: Dummy variable, 1 if application for credit card accepted, 0 if not
  reports: Number of major derogatory reports
* `age`: Age n years plus twelfths of a year
* `income`: Yearly income (divided by 10,000)
* `share`: Ratio of monthly credit card expenditure to yearly income
* `expenditure`: Average monthly credit card expenditure
* `owner`: 1 if owns their home, 0 if rent
* `selfempl`: 1 if self employed, 0 if not.
* `dependents`: 1 + number of dependents
* `months`: Months living at current address
* `majorcards`: Number of major credit cards held
* `active`: Number of active credit accounts

The feature `expenditure` is ambiguous: it is unclear whether it refers to spending on the current credit card application or on other cards used before applying.

To investigate this, we conduct the following experiment:

```python
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))
```

The results show that all individuals who did not receive a card had no expenditures, while only 2% of cardholders had no expenditures. This suggests that expenditure likely refers to spending on the card applied for, indicating potential target leakage.

Since share is derived from expenditure, it should also be excluded to prevent leakage from influencing the model.

:::

:::{.cell .code}
```python
!wget -q https://github.com/kyrillosishak/re-FakeNewsDetection/raw/main/Data/CreditCardDatafrombook%22EconometricAnalysis%22.zip
!unzip -q "/content/CreditCardDatafrombook\"EconometricAnalysis\".zip"
```
:::

:::{.cell .code}
```python
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Read the data
data = pd.read_csv('AER_credit_card_data.csv', true_values = ['yes'], false_values = ['no'])
# Select target
y = data.card
# Select predictors
X = data.drop(['card'], axis=1)

rf = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("Cross-validation accuracy (Data Leakage):", cv_scores.mean()*100)

# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(rf, X2, y, cv=5, scoring='accuracy')
print("Cross-validation accuracy (without Data Leakage):", cv_scores.mean()*100)
```
:::

:::{.cell .markdown}

This analysis demonstrates the impact of data leakage on model performance. Initially, a machine learning model is trained using all available features, including those that are likely to cause leakage such as `expenditure` and `share`. The model achieves an impressive accuracy of over 98% during cross-validation, but this high accuracy is misleading because it includes information that wouldn't be available in a real-world scenario. By excluding these leaky predictors, which are derived from the target variable or are closely related to it, the model's cross-validation accuracy drops to around 84%. This significant decrease highlights how the model was initially overfitting to information that wouldn't be accessible at the time of prediction.

:::
