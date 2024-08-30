:::{.cell .markdown}

# A Soft Two-Layers Voting Model for Fake News Detection

:::

:::{.cell .markdown}

In this series of notebooks, we aim to reproduce the results of [A Soft T A Soft Two-Layers Voting Model for F oting Model for Fake News Detection](https://digitalcommons.aaru.edu.jo/cgi/viewcontent.cgi?article=1548&context=erjeng). The paper explores the application of advanced ensemble techniques, specifically a two-layered weighted voting approach, in conjunction with multiple machine learning algorithms for fake news classification. Our objective is to rigorously validate the findings of the paper, ensuring transparency and reproducibility in our approach. Additionally, we will investigate the impact of data leakage caused by wrongly doing feature extraction. Data Leakage can cause wrong performance estimation. By identifying and addressing these issues, we aim to demonstrate how such leakage can impact model performance and interpretation of results.

**🏆 Objectives of these notebooks:**

1. Identify the specific claims related to classification accuracy, performance improvements over existing methods, and the efficacy of the proposed two-layered weighted voting methodology.
2. Assess the methodology, datasets (LIAR and fake or real news), and the problem statement presented.
2. Define specific experiments needed to validate each identified claim, including the performance of individual classifiers and voting ensembles.
Obtain the LIAR and fake or real news datasets and perform data cleaning and preprocessing steps as mentioned in the paper.
3. Implement the two-layered weighted voting classifier with the same base models used in the paper (Random Forest, SVM, SGD, Logistic Regression, Extra Trees) and train the models using the preprocessed data.
4. Evaluate the models using the same metrics reported in the paper (accuracy, precision, recall, F1 score) and compare the results with those of individual classifiers and existing methods.
5. Investigate the impact of different feature extraction techniques (Count Vectorizer and TF-IDF) on model performance.
6. Analyze the effect of classifier combinations and selective weighting on the ensemble's overall accuracy.
7. Implement and evaluate the stacking approach to refine predictions in the final layer of the model.
8. Compare the results obtained from our reproduced models with the results reported in the original paper, analyzing any differences or discrepancies in performance and identifying potential reasons for variations.
9. Fixing the data leakage problem caused by the authors' approach.
---

**🔍 In this notebook, we will:**

* Identify the specific claims.
* Define specific experiments.
* Obtain the `LIAR` dataset and `Fake or Real` dataset.
* Use the same model used in the paper for fake news classification.
* Compare the results obtained from your reproduced models with the original paper's results.
* Fix the data leakage problem in the authors' approach.

**🗣️ Claims :**

1. The authors claim that the proposed method shows 4.2% and 5% accuracy improvements over single classifier or simple voting methods on the LIAR and fake or real news datasets respectively.

**🧪Experiment from the paper:**

> pre-processing steps applied in this work are: Tokenization...Lower casing...Removal of special characters and punctuation...Removal of numbers...Stop words removal

> Two types of vectorization methods used in the proposed system which 
include Count Vectorizer (CV) and Term Frequency-Inverse Document Frequency (TF-IDF)

> After feature extraction, data is split into training and testing sets. The training set is used to train machine learning models, and the testing set is used to evaluate the models' performance. The testing set helps to 
assess how well models generalize to unseen data.

>  automated methods like GridSearchCV were employed in this study for tuning

> two-layered weighted voting ensemble classifier, carefully selecting base 
classifiers through a rigorous evaluation process to maximize accuracy and robustness.

> In our proposed model, we utilize a weighted voting 
ensemble method that involves nine machine learning 
classifiers to enhance the accuracy of fake news detection.

> In our model, there are five base classifiers 
are used such as Random Forest, Support Vector Machine, 
Stochastic Gradient Descent, Logistic Regression and Extra 
Trees.

> we test many pairs of classifiers to use in each voting. We select the best pair for voting according to their accuracy. Finally, we 
use final selected pair of classifiers for our weighted voting 
ensemble model.

:::

:::{.cell .markdown}

## Data Loading and preprocessing

:::

:::{.cell .markdown}

> pre-processing steps applied in this work are: Tokenization...Lower casing...Removal of special characters and punctuation...Removal of numbers...Stop words removal

:::

:::{.cell .code}
```python
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download nltk data
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Removal of special characters and punctuation
    tokens = [re.sub(r'\W', '', word) for word in tokens if word.isalpha()]
    # Removal of numbers
    tokens = [word for word in tokens if not word.isdigit()]
    # Stop words removal
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)
```
:::

:::{.cell .markdown}

### LIAR dataset

:::

:::{.cell .markdown}

The `LIAR` dataset is a comprehensive benchmark dataset used for detecting fake news and evaluating automated fact-checking systems. Introduced in 2017 by Wang, the dataset consists of 12,836 short statements labeled with one of six truthfulness levels: "pants-fire," "false," "barely-true," "half-true," "mostly-true," and "true." These statements are collected from the fact-checking website PolitiFact, where professional fact-checkers have analyzed and labeled them. Each statement in the LIAR dataset is accompanied by additional metadata, including the speaker’s name, job title, party affiliation, the context or subject of the statement, and a justification for the truthfulness rating. The dataset covers a wide range of topics, including politics, health, education, and more, making it a valuable resource for researchers in natural language processing and machine learning who are working on developing models for automated fact-checking and detecting misinformation.

:::

:::{.cell .markdown}

#### Downloading the data

:::

:::{.cell .code}
```python
!wget -q https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
!unzip -q liar_dataset.zip
```
:::

:::{.cell .code}
```python
liar_train = pd.read_csv('/content/train.tsv', sep='\t', header=None)
liar_val = pd.read_csv('/content/test.tsv', sep='\t', header=None)
liar_test = pd.read_csv('/content/valid.tsv', sep='\t', header=None)
liar = pd.concat([liar_train, liar_val, liar_test], ignore_index = True)
```
:::

:::{.cell .code}
```python
def convert_label(label):
    label_mapping = {'pants-fire': 0, 'false': 0, 'barely-true': 0, 'half-true': 1, 'mostly-true': 1, 'true': 1}
    return label_mapping[label]
liar[1] = liar[1].apply(convert_label)
```
:::

:::{.cell .markdown}

#### Preprocessing

:::

:::{.cell .code}
```python
liar[2] = liar[2].apply(preprocess_text)
X_liar = liar[2]
y_liar = liar[1]
```
:::

:::{.cell .markdown}

### Fake or Real dataset

:::

:::{.cell .markdown}

The Fake or Real dataset is a collection of news articles curated to support the development and evaluation of models for fake news detection. This dataset contains a balanced set of articles, with each article labeled as either "fake" or "real." It includes a variety of news stories covering different topics, such as politics, entertainment, sports, and more, providing a diverse range of content for analysis.

:::

:::{.cell .markdown}

#### Downloading the data

:::

:::{.cell .code}
```python
!wget -q https://raw.githubusercontent.com/kyrillosishak/re-FakeNewsDetection/main/data/fake_or_real_news.csv
```
:::

:::{.cell .code}
```python
fake_real = pd.read_csv('fake_or_real_news.csv')
```
:::

:::{.cell .code}
```python
def convert_label2(label):
    label_mapping = {'FAKE': 0, 'REAL': 1}
    return label_mapping[label]
fake_real['label'] = fake_real['label'].apply(convert_label2)
```
:::

:::{.cell .markdown}

#### Preprocessing

:::

:::{.cell .code}
```python
fake_real['text'] = fake_real['text'].apply(preprocess_text)
X_fake_real = fake_real['text']
y_fake_real = fake_real['label']
```
:::

:::{.cell .markdown}

## Feature extraction

:::

:::{.cell .markdown}

<img src = "https://github.com/kyrillosishak/re-FakeNewsDetection/raw/main/assets/featureExtraction.png" height = 400>

> Two types of vectorization methods used in the proposed system which
include Count Vectorizer (CV) and Term Frequency-Inverse Document Frequency (TF-IDF)

> After feature extraction, data is split into training and testing sets. The training set is used to train machine learning models, and the testing set is used to evaluate the models' performance. The testing set helps to
assess how well models generalize to unseen data.

:::

:::{.cell .code}
```python
def tfidf(X,y):
    # Feature extraction
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Splitting
    X_train_tfidf, X_test_tfidf,y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf,y.array, test_size=0.3, random_state=42)
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    return X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf
```
:::

:::{.cell .code}
```python
def CV(X,y):
    # Feature extraction
    cv = CountVectorizer()
    X_cv = cv.fit_transform(X)

    # Splitting
    X_train_cv, X_test_cv,y_train_cv, y_test_cv = train_test_split(X_cv,y.array, test_size=0.3, random_state=42)
    print(f"Vocabulary size: {len(cv.vocabulary_)}")
    return X_train_cv, y_train_cv, X_test_cv, y_test_cv
```
:::

:::{.cell .code}
```python
X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf = tfidf(X_liar,y_liar)
X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf = tfidf(X_fake_real,y_fake_real)
```
:::

:::{.cell .code}
```python
X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv = CV(X_liar,y_liar)
X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv = CV(X_fake_real,y_fake_real)
```
:::

:::{.cell .markdown}

## Training & Exploring best parameters

:::

:::{.cell .markdown}

>  automated methods like GridSearchCV were employed in this study for tuning

> In our model, there are five base classifiers
are used such as Random Forest, Support Vector Machine,
Stochastic Gradient Descent, Logistic Regression and Extra
Trees.

:::

:::{.cell .code}
```python
def run_with_grid_search(X_train, y_train, X_test, y_test, clf, param_grid, name):
    # Initialize variables for tracking the best parameters and performance
    best_params = {}
    best_score = 0

    # Create a ParameterGrid object for the grid search
    grid = ParameterGrid(param_grid)

    # Calculate the total number of parameter combinations and folds
    num_combinations = len(grid)
    num_folds = 10  # Number of folds
    total_evaluations = num_combinations * num_folds

    # Initialize tqdm for tracking overall progress
    with tqdm(total=total_evaluations, desc=f"Grid Search {name}", unit="folds") as pbar:
        # Perform grid search manually
        for params in grid:
            # Set parameters
            clf.set_params(**params)

            # Evaluate on training data using KFold
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

            for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
                X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_test_fold = np.array(y_train)[train_index], np.array(y_train)[test_index]

                # Train and evaluate model
                clf.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)
                acc = accuracy_score(y_test_fold, y_pred)
                best_params[acc] = clf.get_params()

                # Update progress bar for each fold evaluation
                pbar.set_postfix({"Fold Accuracy": acc})
                pbar.update(1)

    # Final model training with best parameters
    clf.set_params(**best_params[max(best_params)])
    clf.fit(X_train, y_train)

    # Test accuracy
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy with {name}: {test_acc}")
    return clf
```
:::

:::{.cell .code}
```python
param_grids = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'kernel': ['linear', 'rbf'], 'C': [1, 10]},
    'ExtraTrees': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
    'SGD': {'loss': ['log_loss'], 'penalty': ['l2', 'l1']},
    'LogisticRegression': {'C': [0.1, 1, 10]},
}
classifiers = {
    'SGD': SGDClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=5000),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'ExtraTrees': ExtraTreesClassifier()
}
```
:::

:::{.cell .code}
```python
run_with_grid_search(X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf, classifiers['SVM'], param_grids['SVM'],'SVM')
run_with_grid_search(X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf, classifiers['LogisticRegression'], param_grids['LogisticRegression'], 'Logistic Regression')
run_with_grid_search(X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf, classifiers['SGD'], param_grids['SGD'], 'SGD')
run_with_grid_search(X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf, classifiers['RandomForest'], param_grids['RandomForest'], 'Random Forest')
run_with_grid_search(X_train_liar_tfidf, y_train_liar_tfidf, X_test_liar_tfidf, y_test_liar_tfidf, classifiers['ExtraTrees'], param_grids['ExtraTrees'], 'Extra Trees')
```
:::

:::{.cell .code}
```python
run_with_grid_search(X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf, classifiers['SVM'], param_grids['SVM'],'SVM')
run_with_grid_search(X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf, classifiers['LogisticRegression'], param_grids['LogisticRegression'], 'Logistic Regression')
run_with_grid_search(X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf, classifiers['SGD'], param_grids['SGD'], 'SGD')
run_with_grid_search(X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf, classifiers['RandomForest'], param_grids['RandomForest'], 'Random Forest')
run_with_grid_search(X_train_fr_tfidf, y_train_fr_tfidf, X_test_fr_tfidf, y_test_fr_tfidf, classifiers['ExtraTrees'], param_grids['ExtraTrees'], 'Extra Trees')
```
:::

:::{.cell .code}
```python
run_with_grid_search(X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv, classifiers['SVM'], param_grids['SVM'],'SVM')
run_with_grid_search(X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv, classifiers['LogisticRegression'], param_grids['LogisticRegression'], 'Logistic Regression')
run_with_grid_search(X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv, classifiers['SGD'], param_grids['SGD'], 'SGD')
run_with_grid_search(X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv, classifiers['RandomForest'], param_grids['RandomForest'], 'Random Forest')
run_with_grid_search(X_train_liar_cv, y_train_liar_cv, X_test_liar_cv, y_test_liar_cv, classifiers['ExtraTrees'], param_grids['ExtraTrees'], 'Extra Trees')
```
:::

:::{.cell .code}
```python
run_with_grid_search(X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv, classifiers['SVM'], param_grids['SVM'],'SVM')
run_with_grid_search(X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv, classifiers['LogisticRegression'], param_grids['LogisticRegression'], 'Logistic Regression')
run_with_grid_search(X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv, classifiers['SGD'], param_grids['SGD'], 'SGD')
run_with_grid_search(X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv, classifiers['RandomForest'], param_grids['RandomForest'], 'Random Forest')
run_with_grid_search(X_train_fr_cv, y_train_fr_cv, X_test_fr_cv, y_test_fr_cv, classifiers['ExtraTrees'], param_grids['ExtraTrees'], 'Extra Trees')
```
:::

:::{.cell .markdown}

## Identify and Explain data leakage impact

:::