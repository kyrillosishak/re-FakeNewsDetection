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

Data leakage refers to the situation where information from outside the training dataset is used to create the model. This leads to wrong performance estimates and poor generalization to new data.

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

### Explaining data leakage

:::

:::{.cell .markdown}

#### How TF-IDF Reduces the Importance of Common Words
TF-IDF reduces the importance of words that are common across many documents:

- **High Document Frequency:** When a word appears frequently across different documents in a dataset, its document frequency (DF) increases. This results in a lower IDF score.
- **Lower TF-IDF Score:** The multiplication of TF (high for common words within a document) and IDF (low for common words across documents) results in a lower TF-IDF score for these words. Thus, common words like "the", "is", and "and" are assigned lower importance because they do not provide meaningful information to differentiate between documents.

#### Effect of Data Leakage on TF-IDF
Data leakage can occur if the TF-IDF feature extraction process uses information from the test set during training, improperly influencing the IDF calculation. This leads to a biased assessment of word importance and can impact model performance :

1. Word is Less Frequent in the Training Set and More Frequent in the Test Set:
  - **Effect of Data Leakage:** If the TF-IDF calculation mistakenly includes test set data, the IDF for this word will be lower than it should be, as the word appears more frequently in the test set.
  - **Impact on Model:** The model will de-emphasize the word during training, assigning it a lower TF-IDF score. When this word appears frequently in the test set, the model might fail to recognize its significance, leading to poor predictions.

  **Illustration:** Consider the word "innovation" that is rare in the training set but common in the test set. Due to data leakage, the IDF calculated during training would reflect the high frequency of "innovation" in the test set, reducing its importance. As a result, the model might not predict well for test documents where "innovation" is actually a critical term.

2. Word is More Frequent in the Training Set and Less Frequent in the Test Set:
  - **Effect of Data Leakage:** The TF-IDF model might overestimate the importance of this word during training if the test set is improperly included in the IDF calculation.
  - **Impact on Model:** The model will emphasize this word too much, increasing its TF-IDF score in training. When the test set rarely contains this word, the model might perform poorly as it over-relies on a feature that is not prevalent in unseen data.

  **Illustration:** Imagine the word "reliable" is common in the training set but rare in the test set. Due to data leakage, the IDF value calculated might overemphasize "reliable," making the model overly dependent on it. When this word doesn’t appear as often in the test set, the model’s predictions may degrade.

:::

:::{.cell .markdown}

### Exploring the behavior of the data leakage 

:::

:::{.cell .markdown}

In this 2 example, we explore the effect of data leakage on the TF-IDF feature extraction process using a simple text classification task. The dataset consists of a small training set (`train_docs`) and a test set (`test_docs`). We are particularly interested in understanding how the TF-IDF scores for the word "banana" differ when there is data leakage versus when there is no data leakage. We designated "banana" as an important word for labeling (label 1 : if "banana" exist in sentence, label 0 : if "banana" does not exist)

:::

:::{.cell .code}
```python
# Function to print TF-IDF scores for a specific word
def print_tfidf_scores(word, vectorizer, train_tfidf, test_tfidf):
    word_index = vectorizer.vocabulary_.get(word)
    if word_index is not None:
        print(f"TF-IDF scores for '{word}':")
        print("Training set:\t", train_tfidf[:, word_index].toarray().flatten())
        print("Test set:\t", test_tfidf[:, word_index].toarray().flatten())
    else:
        print(f"Word '{word}' not found in vocabulary.")

```
:::

:::{.cell .markdown}

#### Case 1 : Word is Less Frequent in the Training Set and More Frequent in the Test Set

:::

:::{.cell .code}
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
train_docs = ['apple apple apple',
 'orange orange orange',
 'banana orange apple',
 'apple orange orange']

train_labels = [0, 0, 1, 0]
test_docs = ['apple banana banana', 'apple apple banana', 'banana banana orange']
test_labels = [1, 1, 1]

# Initialize the TF-IDF vectorizer
vectorizer_with_leakage = TfidfVectorizer()
vectorizer_without_leakage = TfidfVectorizer()


X_combined = vectorizer_with_leakage.fit_transform(train_docs + test_docs)

# Transform the training and test sets separately
X_train_leakage = X_combined[:len(train_docs)]
X_test_leakage = X_combined[len(train_docs):]

# Fit the vectorizer without data leakage (fit on training set only)
vectorizer_without_leakage.fit(train_docs)
X_train_without_leakage = vectorizer_without_leakage.transform(train_docs)
X_test_without_leakage = vectorizer_without_leakage.transform(test_docs)


# Compare the TF-IDF scores for the word 'banana' with and without data leakage
print("With Data Leakage:")
print_tfidf_scores('banana', vectorizer_with_leakage, X_train_leakage, X_test_leakage)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_leakage, train_labels)
preds = clf.predict(X_test_leakage)
dist, idx = clf.kneighbors(X_test_leakage)
print("predictions: ", preds)
print("True labels: ", np.array(test_labels))
print(f"Accuracy:  {np.mean(preds == np.array(test_labels))*100}%")

print("\nWithout Data Leakage:")
print_tfidf_scores('banana', vectorizer_without_leakage, X_train_without_leakage, X_test_without_leakage)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_without_leakage, train_labels)
preds = clf.predict(X_test_without_leakage)
dist, idx = clf.kneighbors(X_test_without_leakage)
print("predictions: ", preds)
print("True labels: ", np.array(test_labels))
print(f"Accuracy:  {np.mean(preds == np.array(test_labels))*100}%")
```
:::

:::{.cell .markdown}

##### Analysis of the Results
1. With Data Leakage:

  - **Training Set Scores:** Scores of "banana" across all train set (`[0.0,0.0,0.0.60113188,0.0]`)

  - **Test Set Scores:**  Scores of "banana" across all test set (`[0.91599378, 0.49572374, 0.89442719]`)
  
2. Without Data Leakage:
  - **Training Set Scores:** Scores of "banana" across all train set (`[0.0,0.0,0.74230628,0.0]`)

  - **Test Set Scores:**  Scores of "banana" across all test set (`[0.9526607,0.61666846,0.9526607]`)

- **Explanation (Data Leakage Effect):** When TF-IDF is computed with data leakage (on the whole dataset before splitting), the IDF value for "banana" is influenced by its presence in both the training and test sets. Since "banana" is more frequent in the test set, its importance (TF-IDF score) in the test set is not as high as it would be if the IDF was calculated on the training data alone. This is because the IDF is lower (indicating more common) when considering the whole dataset, thus reducing the overall TF-IDF score for "banana" in the test set.

- **Explanation (No Data Leakage Effect):** When TF-IDF is computed without data leakage (on the training set only), the IDF value is calculated based solely on the word's frequency in the training set. Because "banana" is less frequent in the training set than it is in the full dataset (or the test set), its IDF value is higher. This results in higher TF-IDF scores in both the training and test sets when the test set is transformed using the training-based TF-IDF vectorizer. The higher IDF value (due to "banana" being less common in training data alone) emphasizes "banana" more in the test set when calculated correctly without data leakage.

*In this case data leakage is caused lower accuracy than the correct one.*

:::

:::{.cell .markdown}

#### Case 2 : Word is More Frequent in the Training Set and Less Frequent in the Test Set

:::

:::{.cell .code}
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Define the training and test documents
train_docs = ['apple apple orange',
 'apple banana banana',
 'banana banana apple',
 'banana apple apple'
]
train_labels = [0, 1, 1, 1]
test_docs = ['apple apple orange orange', 'banana apple orange', 'orange orange apple']
test_labels = [0, 1, 0]
# Initialize the TF-IDF vectorizer
vectorizer_with_leakage = TfidfVectorizer()
vectorizer_without_leakage = TfidfVectorizer()

# Fit the vectorizer with data leakage (fit on the whole dataset)
X_combined = vectorizer_with_leakage.fit_transform(train_docs + test_docs)

# Transform the training and test sets separately
X_train_leakage = X_combined[:len(train_docs)]
X_test_leakage = X_combined[len(train_docs):]

# Fit the vectorizer without data leakage (fit on training set only)
vectorizer_without_leakage.fit(train_docs)
X_train_without_leakage = vectorizer_without_leakage.transform(train_docs)
X_test_without_leakage = vectorizer_without_leakage.transform(test_docs)


# Compare the TF-IDF scores for the word 'banana' with and without data leakage
print("With Data Leakage:")
print_tfidf_scores('banana', vectorizer_with_leakage, X_train_leakage, X_test_leakage)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_leakage, train_labels)
preds = clf.predict(X_test_leakage)
dist, idx = clf.kneighbors(X_test_leakage)
print("predictions: ", preds)
print("True labels: ", np.array(test_labels))
print(f"Accuracy:  {np.mean(preds == np.array(test_labels))*100}%")

print("\nWithout Data Leakage:")
print_tfidf_scores('banana', vectorizer_without_leakage, X_train_without_leakage, X_test_without_leakage)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train_without_leakage, train_labels)
preds = clf.predict(X_test_without_leakage)
dist, idx = clf.kneighbors(X_test_without_leakage)
print("predictions: ", preds)
print("True labels: ", np.array(test_labels))
print(f"Accuracy:  {np.mean(preds == np.array(test_labels))*100}%")
```
:::

:::{.cell .markdown}

##### Analysis of the Results
1. With Data Leakage:

  - **Training Set Scores:** Scores of "banana" across all train set (`[0.0, 0.94673372, 0.94673372, 0.59223757]`)

  - **Test Set Scores:**  Scores of "banana" across all test set (`[0.0, 0.63721833, 0.0]`)
  
2. Without Data Leakage:
  - **Training Set Scores:** Scores of "banana" across all train set (`[0.0, 0.92564688, 0.92564688, 0.52173612]`)

  - **Test Set Scores:**  Scores of "banana" across all test set (`[0.0, 0.49248889, 0.0]`)

- **Explanation (Data Leakage Effect):** Since "banana" is less frequent in the test set, its importance (TF-IDF score) in the test set is higher as it would be if the IDF was calculated on the training data alone. This is because the IDF is greater (indicating more common) when considering the whole dataset, thus increasing the overall TF-IDF score for "banana" in the test set.

- **Explanation (No Data Leakage Effect):** Because "banana" is more frequent in the training set than it is in the full dataset (or the test set), its IDF value is lower. This results in lower TF-IDF scores in both the training and test sets when the test set is transformed using the training-based TF-IDF vectorizer. The lower IDF value (due to "banana" being more common in training data alone) emphasizes "banana" less in the test set when calculated correctly without data leakage.

*In this case data leakage is caused higher accuracy than the correct one.*

:::

:::{.cell .markdown}

### Tweakable Toy Example

:::

:::{.cell .code}
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import random

# Set the seed for reproducibility
random.seed(50)
np.random.seed(50)

# Function to generate random sentences
def generate_sentences(vocab, num_sentences, sentence_length, target_word, target_word_frequency):
    sentences = []
    for _ in range(num_sentences):
        sentence = random.choices(vocab, k=sentence_length)
        # Add the target word to the sentence based on the specified frequency
        if random.random() < target_word_frequency:
            sentence[random.randint(0, sentence_length - 1)] = target_word
        sentences.append(' '.join(sentence))
    return sentences

# Define vocabulary and parameters
vocab = [
    "banana",  "calculator", "albatross", "dolphin", "envelope", "forest", 
    "galaxy", "hammock", "iceberg", "jigsaw", "kaleidoscope", "labyrinth", 
    "magnolia", "nebula", "oasis", "parrot", "quasar", "raspberry", 
    "sunflower", "trapeze", "umbrella", "volcano", "waterfall", "xylophone", 
    "yodel", "zephyr", "apricot", "bumblebee", "candelabra", "dragonfly", 
    "eucalyptus", "feather", "giraffe", "hyacinth", "iguana", "jasmine", 
    "kiwi", "lantern", "mermaid", "nutmeg"
]

# Function to print TF-IDF scores for a given word
def print_tfidf_scores(word, vectorizer, X_train, X_test):
    word_index = vectorizer.vocabulary_.get(word)
    if word_index is not None:
        train_scores = X_train[:, word_index].toarray().flatten()
        test_scores = X_test[:, word_index].toarray().flatten()
        print(f"Training Set Scores for '{word}':", train_scores)
        print(f"Test Set Scores for '{word}':", test_scores)
    else:
        print(f"Word '{word}' not found in vocabulary.")

# Train and evaluate KNN classifier
def evaluate_classifier(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy = np.mean(preds == y_test) * 100
    print("Predictions:", preds)
    print("True labels:", np.array(y_test))
    print(f"Accuracy: {accuracy}%")
    return accuracy

target_word = 'banana'  # The word that affects the labeling
train_sentence_length = 5
test_sentence_length = 5
```
:::

:::{.cell .markdown}

#### Case 1 : Word is Less Frequent in the Training Set and More Frequent in the Test Set

:::

:::{.cell .code}
```python
# Parameters for training and test set generation
num_train_sentences = 20
num_test_sentences = 10
train_target_word_freq = 0.3  # Frequency of the target word in the training set
test_target_word_freq = 0.7  # Frequency of the target word in the test set

# Generate training and test sentences
train_sentences = generate_sentences(vocab, num_train_sentences, train_sentence_length, target_word, train_target_word_freq)
test_sentences = generate_sentences(vocab, num_test_sentences, test_sentence_length, target_word, test_target_word_freq)

# Generate labels (1 if target_word exists, 0 otherwise)
train_labels = [1 if target_word in sentence else 0 for sentence in train_sentences]
test_labels = [1 if target_word in sentence else 0 for sentence in test_sentences]

# Initialize TF-IDF vectorizers
vectorizer_with_leakage = TfidfVectorizer()
vectorizer_without_leakage = TfidfVectorizer()

# With data leakage (fitting on both train and test set)
X_combined = vectorizer_with_leakage.fit_transform(train_sentences + test_sentences)
X_train_leakage = X_combined[:num_train_sentences]
X_test_leakage = X_combined[num_train_sentences:]

# Without data leakage (fitting on training set only)
vectorizer_without_leakage.fit(train_sentences)
X_train_without_leakage = vectorizer_without_leakage.transform(train_sentences)
X_test_without_leakage = vectorizer_without_leakage.transform(test_sentences)

# Check TF-IDF scores for target_word with and without data leakage
print("With Data Leakage:")
print_tfidf_scores(target_word, vectorizer_with_leakage, X_train_leakage, X_test_leakage)

print("\nWithout Data Leakage:")
print_tfidf_scores(target_word, vectorizer_without_leakage, X_train_without_leakage, X_test_without_leakage)


print("\nEvaluation with Data Leakage:")
evaluate_classifier(X_train_leakage, train_labels, X_test_leakage, test_labels)

print("\nEvaluation without Data Leakage:")
evaluate_classifier(X_train_without_leakage, train_labels, X_test_without_leakage, test_labels)
print()
```
:::

:::{.cell .markdown}

#### Case 2 : Word is More Frequent in the Training Set and Less Frequent in the Test Set

:::

:::{.cell .code}
```python
# Set the seed for reproducibility
random.seed(100)
np.random.seed(100)

# Parameters for training and test set generation
num_train_sentences = 50
num_test_sentences = 10
train_target_word_freq = 0.7  # Frequency of the target word in the training set
test_target_word_freq = 0.3  # Frequency of the target word in the test set

# Generate training and test sentences
train_sentences = generate_sentences(vocab, num_train_sentences, train_sentence_length, target_word, train_target_word_freq)
test_sentences = generate_sentences(vocab, num_test_sentences, test_sentence_length, target_word, test_target_word_freq)

# Generate labels (1 if target_word exists, 0 otherwise)
train_labels = [1 if target_word in sentence else 0 for sentence in train_sentences]
test_labels = [1 if target_word in sentence else 0 for sentence in test_sentences]

# Initialize TF-IDF vectorizers
vectorizer_with_leakage = TfidfVectorizer()
vectorizer_without_leakage = TfidfVectorizer()

# With data leakage (fitting on both train and test set)
X_combined = vectorizer_with_leakage.fit_transform(train_sentences + test_sentences)
X_train_leakage = X_combined[:num_train_sentences]
X_test_leakage = X_combined[num_train_sentences:]

# Without data leakage (fitting on training set only)
vectorizer_without_leakage.fit(train_sentences)
X_train_without_leakage = vectorizer_without_leakage.transform(train_sentences)
X_test_without_leakage = vectorizer_without_leakage.transform(test_sentences)

# Check TF-IDF scores for target_word with and without data leakage
print("With Data Leakage:")
print_tfidf_scores(target_word, vectorizer_with_leakage, X_train_leakage, X_test_leakage)

print("\nWithout Data Leakage:")
print_tfidf_scores(target_word, vectorizer_without_leakage, X_train_without_leakage, X_test_without_leakage)


print("\nEvaluation with Data Leakage:")
evaluate_classifier(X_train_leakage, train_labels, X_test_leakage, test_labels)

print("\nEvaluation without Data Leakage:")
evaluate_classifier(X_train_without_leakage, train_labels, X_test_without_leakage, test_labels)
print()
```
:::

