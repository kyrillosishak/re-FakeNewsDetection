::: {.cell .markdown}

# Toy Example
:::

::: {.cell .markdown}
## Introduction
:::

::: {.cell .markdown}

In machine learning, ensuring that models generalize well to unseen data is paramount. However, a subtle yet pervasive issue known as data leakage can severely compromise this goal. Data leakage occurs when information from outside the training dataset slips into the model, artificially boosting performance during evaluation. This can lead to overly optimistic results that fail in real-world scenarios. Among the various forms of data leakage, feature selection is particularly insidious, as it can occur at multiple stages of the machine learning pipeline and often goes unnoticed.

✨ The objective of this notebook is to:

* Understand the concept of data leakage, with a focus on feature selection.
  Examine three specific cases:

    * Feature selection by TF-IDF before splitting data.
    * Random splitting of temporal data leading to temporal leakage.
    * Target data leakage caused by selecting features that illegitimately influence the target variable.
* Illustrate these concepts with synthetic and real-world examples.
* Train machine learning models and evaluate their performance, highlighting the impact of these types of data leakage.
* Critically analyze how different types of feature selection data leakage can affect model outcomes.

🔍 In this notebook, we will explore:

* A synthetic data example to demonstrate feature selection before data splitting.
* An analysis of temporal leakage due to random splitting in temporal datasets.
* An investigation into target data leakage with a focus on illegitimate features.

---
:::

::: {.cell .markdown}

## Feature extraction using TF-IDF vectorizer

:::

::: {.cell .markdown}

### What is TF-IDF?

:::

::: {.cell .markdown}

TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The TF-IDF score is composed of two parts:

* Term Frequency (TF): This measures how frequently a term appears in a document. The term frequency of a term `t` in a document `d` is given by:

    $TF(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}$


* Inverse Document Frequency (IDF): This measures how important a term is. While computing TF, all terms are considered equally important. IDF, on the other hand, decreases the weight of terms that appear very frequently in the document set and increases the weight of terms that appear rarely. The IDF of a term `t` is given by:
    
    $IDF(t,D) = \log{(\frac{\text{Total number of documents } (N)}{\text{Number of documents with term t in it+1 }})}$
    
    *The addition of 1 to the denominator is to prevent division by zero.*


* The TF-IDF score for a term `t` in a document `d` is then calculated as:

  $TF-IDF(t,d,D) = TF(t,d)×IDF(t,D)$

*Consider `D` is total dataset (e.g. X) and `d` is an entry in the dataset (e.g. X[0])*

:::

::: {.cell .markdown}

### What can cause Data Leakage?

:::

::: {.cell .markdown}

Data leakage refers to the situation where information from outside the training dataset is used to create the model. This leads to overly optimistic performance estimates and poor generalization to new data.

This can happen when TF-IDF is computed on the entire dataset before splitting it into training and testing sets

:::

::: {.cell .markdown}

### Why Does This Cause Data Leakage?

:::

::: {.cell .markdown}

* The IDF part of TF-IDF takes into account the frequency of terms across all documents in the dataset. If you compute IDF using the entire dataset, the resulting IDF values will be influenced by the documents that end up in the test set. This means that information from the test set leaks into the training process, giving the model a peek into the test data distribution. The model might perform well in cross-validation or on the training set but fail to generalize to truly unseen data.


* When you split the dataset after computing TF-IDF, both the training and test sets will share the same IDF values, which are based on the entire dataset. However, in a real-world scenario, the test data is unknown during the training phase, and its distribution might be different from the training data. By using IDF values that include test data, you're effectively giving your model information it shouldn't have during training.

:::

:::{.cell .markdown}

### Examples Illustrating the Problem:

:::

:::{.cell .markdown}

1. Word Appears in Test Set but Not in Training Set:
   * Consider a scenario where a word appears in the test set but not in the training set. If TF-IDF is calculated using the entire dataset (including both training and test sets), the vectorizer will include this word in its vocabulary, and its TF-IDF representation will be based on its presence in the entire dataset. Although the model won’t use this word's TF-IDF value for classification during testing, the word's presence can affect the TF-IDF values of other words, potentially influencing classification outcomes.
   * In the correct approach, where TF-IDF is computed only on the training set, this word won’t be included in the vocabulary used to transform the test set. The TF-IDF values of other words remain unaffected by this word, resulting in predictions based purely on the TF-IDF values without any unintended bias from the test data. This prevents the model from making biased predictions based on information it shouldn’t have had access to during training.

2. No Overlap Between Training and Testing Vocabulary:
   * Imagine another scenario where we intentionally remove all overlapping words between the training and test sets, making their vocabularies completely distinct. One might assume that in this case, vectorizing the entire dataset (train + test) is similar to vectorizing just the training set. While the number of words in the vocabulary might be the same, the TF-IDF values for each word will differ because the IDF values depend on the total number of documents. This difference can affect classification outcomes.

:::

:::{.cell .markdown}

To demonstrate the effect of vectorization (specifically TF-IDF), we will conduct two scenarios, each with two experiments:

1. **Scenario 1: Overlapping Words Between Training and Test Sets**
   * Experiment 1: With Data Leakage
   * Experiment 2: Without Data Leakage
2. **Scenario 2: No Overlapping Words Between Training and Test Sets**
   * Experiment 1: With Data Leakage
   * Experiment 2: Without Data Leakage

:::

:::{.cell .code}
```python
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")
```
:::

:::{.cell .markdown}
```python
# These words is our corpus
common_words = [
    "time", "person", "year", "way", "day", "thing", "man", "world", "life",
    "hand", "part", "child", "eye", "woman", "place", "work", "week", "case",
    "point", "government", "company", "number", "group", "problem", "fact",
    "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
    "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
    "work", "seem", "feel", "try", "leave", "call", "good", "new", "first",
    "last", "long", "great", "little", "own", "other", "old", "right",
    "big", "high", "different", "small", "large", "next", "early", "young",
    "few", "public", "bad", "same", "able", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among",
    "important"
]
```
:::

:::{.cell .code}
```python
# Label logic: Sentences containing certain "positive" words are labeled as 1; others are labeled as 0.
positive_words = {'good', 'great', 'important', 'large', 'first', 'like'}  # Example positive words
def assign_labels(sentences, positive_words):
    labels = []
    for sentence in sentences:
        if any(word in sentence for word in positive_words):
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)

# Generate random sentences with a length of 10 words
def generate_sentences(vocab, n_sentences=100, sentence_length=10):
    return [' '.join(np.random.choice(vocab, sentence_length, replace=False)) for _ in range(n_sentences)]

# Analyze feature importance (TF-IDF values and model coefficients)
def analyze_word_importance(vectorizer, model, feature_names):
    coef = model.coef_[0]
    tfidf_values = vectorizer.transform(test_sentences).toarray()
    importance = {feature_names[i]: coef[i] for i in range(len(coef))}

    # Sort by importance
    importance_sorted = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))

    # Show top features
    df = pd.DataFrame(list(importance_sorted.items()), columns=['Word', 'Coefficient'])
    #df['TF-IDF'] = tfidf_values.mean(axis=0)  # Average TF-IDF score across the test set

    return df
```
:::

:::{.cell .code}
```python
np.random.seed(91)
np.random.shuffle(common_words)
```
:::

:::{.cell .markdown}

### Overlapping Words Between Training and Test Sets

:::

:::{.cell .code}
```python
train_vocab = np.array(common_words[:80])

# you can adjust how much words from train set
unique_vocab = np.random.choice(common_words[80:], 20, replace=False)
overlap_vocab = np.random.choice(train_vocab, 20, replace=False)
test_vocab = np.concatenate((unique_vocab, overlap_vocab))
```
:::

:::{.cell .code}
```python
train_sentences = generate_sentences(train_vocab)
test_sentences = generate_sentences(test_vocab)
```
:::

:::{.cell .code}
```python
y_train = assign_labels(train_sentences, positive_words)
y_test = assign_labels(test_sentences, positive_words)
```
:::

:::{.cell .code}
```python
# Scenario 1: Incorrect Vectorization
vectorizer_in = TfidfVectorizer()
X = vectorizer_in.fit_transform(train_sentences + test_sentences)
y = np.concatenate([y_train, y_test])

# Split the dataset (simulating the issue)
X_train, X_test, y_train_split, y_test_split = train_test_split(X, y, test_size=0.5, stratify=y)

model_in = LogisticRegression()
model_in.fit(X_train, y_train_split)
y_pred = model_in.predict(X_test)

print("Incorrect Vectorization Results")
print(classification_report(y_test_split, y_pred))

# Scenario 2: Correct Vectorization
vectorizer_cr = TfidfVectorizer()
X_train = vectorizer_cr.fit_transform(train_sentences)
X_test = vectorizer_cr.transform(test_sentences)

model_cr = LogisticRegression()
model_cr.fit(X_train, y_train)
y_pred = model_cr.predict(X_test)

print("Correct Vectorization Results")
print(classification_report(y_test, y_pred))
```
:::

:::{.cell .code}
```python
# Analyze the incorrect vectorization
feature_names_incorrect = vectorizer_in.get_feature_names_out()
importance_incorrect = analyze_word_importance(vectorizer_in, model_in, feature_names_incorrect)
print("\nWord Importance (Incorrect Vectorization):")
print(importance_incorrect.head(10))

# Analyze the correct vectorization
feature_names_correct = vectorizer_cr.get_feature_names_out()
importance_correct = analyze_word_importance(vectorizer_cr, model_cr, feature_names_correct)
print("\nWord Importance (Correct Vectorization):")
print(importance_correct.head(10))
```
:::

:::{.cell .markdown}

For this random seed : 
```python
Word Importance (Incorrect Vectorization):
        Word  Coefficient
0  important     1.711376
1      first     1.317084
2       like     1.253898
3      large     1.037566
4    between    -0.898636
5         be     0.790213
6       good     0.757860
7     during     0.642729
8      about    -0.638115
9    without    -0.613821

Word Importance (Correct Vectorization):
        Word  Coefficient
0  important     2.041620
1      first     1.702450
2       like     1.594894
3      great     0.946408
4    without    -0.739197
5    between    -0.681667
6      group    -0.651607
7       seem    -0.634980
8         in    -0.593599
9     person     0.582079
```

1- Incorrect Vectorization (With Data Leakage):
  
   In the first experiment, where TF-IDF vectorization was applied to the entire dataset before splitting, the coefficients of words show a particular pattern:

  * The word "important" has a high positive coefficient (1.711376), indicating strong importance in the model's decision-making process.
  * Other words like "first" (1.317084), "like" (1.253898), and "large" (1.037566) also have significant positive coefficients, contributing heavily to the model's predictions.
  * Negative coefficients, such as for "between" (-0.898636) and "about" (-0.638115), suggest that these words negatively influence the model's predictions.


2- Correct Vectorization (Without Data Leakage):
   
   In the second experiment, where TF-IDF was correctly applied only to the training set, the coefficients differ:

  * The importance of the word "important" increases significantly (2.041620), suggesting that without leakage, the model identifies this word as even more influential.
  * The word "first" also shows increased importance (1.702450) compared to the scenario with leakage.
  * New words like "great" (0.946408) and "person" (0.582079) appear in the top 10, which were not present in the incorrect vectorization output.
  * Negative coefficients, like for "without" (-0.739197) and "between" (-0.681667), are still present but with slightly different magnitudes.
  * In this specific case, the accuracy of the correct vectorization approach scores higher than the approach with data leakage (even though it's considered data leakage, it does not always introduce overoptimism).

:::

:::{.cell .markdown}

### No Overlapping Words Between Training and Test Sets
:::

:::{.cell .code}
```python
train_vocab = np.array(common_words[:80])

# you can adjust how much words from train set
unique_vocab = np.random.choice(common_words[80:], 20, replace=False)
overlap_vocab = np.random.choice(train_vocab, 0, replace=False)  # No Overlap
test_vocab = np.concatenate((unique_vocab, overlap_vocab))
```
:::

:::{.cell .code}
```python
train_sentences = generate_sentences(train_vocab)
test_sentences = generate_sentences(test_vocab)
```
:::

:::{.cell .code}
```python
y_train = assign_labels(train_sentences, positive_words)
y_test = assign_labels(test_sentences, positive_words)
```
:::

:::{.cell .code}
```python
# Scenario 1: Incorrect Vectorization
vectorizer_in = TfidfVectorizer()
X = vectorizer_in.fit_transform(train_sentences + test_sentences)
y = np.concatenate([y_train, y_test])

# Split the dataset (simulating the issue)
X_train, X_test, y_train_split, y_test_split = train_test_split(X, y, test_size=0.5, stratify=y)

model_in = LogisticRegression()
model_in.fit(X_train, y_train_split)
y_pred = model_in.predict(X_test)

print("Incorrect Vectorization Results")
print(classification_report(y_test_split, y_pred))

# Scenario 2: Correct Vectorization
vectorizer_cr = TfidfVectorizer()
X_train = vectorizer_cr.fit_transform(train_sentences)
X_test = vectorizer_cr.transform(test_sentences)

model_cr = LogisticRegression()
model_cr.fit(X_train, y_train)
y_pred = model_cr.predict(X_test)

print("Correct Vectorization Results")
print(classification_report(y_test, y_pred))
```
:::

:::{.cell .code}
```python
# Analyze the incorrect vectorization
feature_names_incorrect = vectorizer_in.get_feature_names_out()
importance_incorrect = analyze_word_importance(vectorizer_in, model_in, feature_names_incorrect)
print("\nWord Importance (Incorrect Vectorization):")
print(importance_incorrect.head(10))

# Analyze the correct vectorization
feature_names_correct = vectorizer_cr.get_feature_names_out()
importance_correct = analyze_word_importance(vectorizer_cr, model_cr, feature_names_correct)
print("\nWord Importance (Correct Vectorization):")
print(importance_correct.head(10))
```
:::

:::{.cell .markdown}

Although we expected that removing overlapping words from the test set would reduce data leakage, it turns out that the incorrect vectorization approach still impacts classification performance.

For some random seed : 
```python
Word Importance (Incorrect Vectorization):
        Word  Coefficient
0      large     1.095835
1       good     1.022922
2       like     0.984725
3      great     0.982889
4  important     0.896704
5       know    -0.690697
6        ask    -0.677369
7       part    -0.668816
8      group    -0.662566
9      other    -0.626573

Word Importance (Correct Vectorization):
        Word  Coefficient
0  important     1.578938
1      great     1.437431
2       good     1.376976
3       like     1.343495
4      large     1.190552
5       know    -0.794442
6       fact    -0.732356
7    against     0.598915
8         on    -0.594957
9      first     0.582358

```
:::
