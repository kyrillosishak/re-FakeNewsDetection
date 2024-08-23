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

### Why that caused Data Leakage?

:::

::: {.cell .markdown}

* The IDF part of TF-IDF takes into account the frequency of terms across all documents in the dataset. If you compute IDF using the entire dataset, the resulting IDF values will be influenced by the documents that end up in the test set. This means that information from the test set leaks into the training process, giving the model a peek into the test data distribution. The model might perform well in cross-validation or on the training set but fail to generalize to truly unseen data.

* When you split the dataset after computing TF-IDF, both the training and test sets will share the same IDF values, which are based on the entire dataset. However, in a real-world scenario, the test data is unknown during the training phase, and its distribution might be different from the training data. By using IDF values that include test data, you're effectively giving your model information it shouldn't have during training.

* Example of the Problem:

    * Imagine a scenario where a word appears in the test set but not in the training set. If TF-IDF is calculated using the entire dataset (both training and test sets), the vectorizer will include this word in the vocabulary, and its TF-IDF representation will be based on its presence in the entire dataset. As a result, during the testing phase, although the model will not recognize this word nor use its TF-IDF value for classification but its presence will affect the TF-IDF vaues of other words that can affect the classification.

    * In the correct approach, where TF-IDF is computed only on the training set, this word will not be included in the vocabulary used to transform the test set the other words TF-IDF values will not depend on this word. Consequently, during testing, while the model will not have any information about this word in both cases, the prediction will be based only on the words TF-IDF values without been affected with the other words. This prevents the model from making biased predictions based on information it should not have had access to during training.

:::

::: {.cell .code}
```python
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import brown
import re

# Function to clean the vocabulary, removing words with non-alphanumeric characters
def clean_vocab(vocab):
    return {word for word in vocab if re.match(r'^\w+$', word)}

# Function to randomly assign coefficients to vocabulary
def assign_random_coefficients(vocab):
    return {word: round(random.uniform(0.01, 1.0), 4) for word in vocab}

# Function to split a vocabulary into train and test with overlap
def split_vocab_with_overlap(vocab, train_size, overlap=0.2):
    vocab = list(vocab)
    random.shuffle(vocab)

    overlap_size = int(train_size * overlap)
    test_size = int(train_size / 2)
    unique_test_size = test_size - overlap_size

    train_vocab = vocab[:train_size - overlap_size]
    overlap_vocab = random.sample(train_vocab, overlap_size)

    remaining_vocab = vocab[train_size - overlap_size:]
    test_vocab = set(overlap_vocab + remaining_vocab[:unique_test_size])

    train_vocab = set(train_vocab + overlap_vocab)

    # Assign coefficients
    coefficients = assign_random_coefficients(train_vocab)

    # Ensure overlap words have the same coefficients in both vocabularies
    test_vocab_with_coefficients = {word: coefficients[word] if word in coefficients else round(random.uniform(0.01, 1.0), 4) for word in test_vocab}

    return coefficients, test_vocab_with_coefficients

import nltk
nltk.download('brown')

# Load a vocabulary collection (example using NLTK words)
full_vocab = set(brown.words())

# Clean the vocabulary to remove words with non-alphanumeric characters
full_vocab = clean_vocab(full_vocab)

# Split the vocabulary into train and test with a 20% overlap
train_vocab, test_vocab = split_vocab_with_overlap(full_vocab, 10, overlap=0.2)
```
:::

::: {.cell .code}
```python
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

random.seed(60)
def generate_sentences(vocab, num_sentences=5, sentence_length=7):
    sentences = []
    for _ in range(num_sentences):
        sentence = random.choices(list(vocab.keys()), k=sentence_length)
        sentences.append(" ".join(sentence))
    return sentences

# Generate sentences for the train and test sets
train_sentences = generate_sentences(train_vocab)
test_sentences = generate_sentences(test_vocab)

print("Train Sentences:")
for sentence in train_sentences:
    print(f"  - {sentence}")

print("\nTest Sentences:")
for sentence in test_sentences:
    print(f"  - {sentence}")

# Function to calculate and display row * coefficient calculations
def simulate_labels(tfidf_matrix_no_leakage, vectorizer_no_leakage, tfidf_matrix_leakage, vectorizer_leakage, coefficients, sentences):
    labels_no_leakage = []
    labels_leakage = []

    Classifier1 = np.mean(list(coefficients.values()))

    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i}: {sentence}")

        # No Data Leakage
        print("\nTF-IDF without Data Leakage:")
        row_no_leakage = tfidf_matrix_no_leakage[i].toarray().flatten()
        score_no_leakage = 0
        for j, word in enumerate(vectorizer_no_leakage.get_feature_names_out()):
            if word in coefficients:
                tfidf_value = row_no_leakage[vectorizer_no_leakage.vocabulary_.get(word)]
                coef = coefficients[word]
                product = tfidf_value * coef
                score_no_leakage += product
                print(f"{word:10}: TF-IDF = {tfidf_value:5.4f}, Coefficient = {coef:5.4f}, Product = {product:5.4f}")
        print(f"Total Score: {score_no_leakage:5.4f}")
        label_no_leakage = 1 if score_no_leakage > Classifier1 else 0
        labels_no_leakage.append(label_no_leakage)

        # Data Leakage
        print("\nTF-IDF with Data Leakage:")
        row_leakage = tfidf_matrix_leakage[i].toarray().flatten()
        score_leakage = 0
        for j, word in enumerate(vectorizer_leakage.get_feature_names_out()):
            if word in coefficients:
                tfidf_value = row_leakage[vectorizer_leakage.vocabulary_.get(word)]
                coef = coefficients[word]
                product = tfidf_value * coef
                score_leakage += product
                print(f"{word:10}: TF-IDF = {tfidf_value:5.4f}, Coefficient = {coef:5.4f}, Product = {product:5.4f}")
        print(f"Total Score: {score_leakage:5.4f}")
        label_leakage = 1 if score_leakage > Classifier1 else 0
        labels_leakage.append(label_leakage)

        print(f"Boundary (Median Coefficient): {Classifier1:5.4f}")
        print("-" * 40)

    return np.array(labels_no_leakage), np.array(labels_leakage)

# Apply TF-IDF without Data Leakage
vectorizer_no_leakage = TfidfVectorizer(vocabulary=train_vocab.keys(), lowercase=False)
tfidf_train_no_leakage = vectorizer_no_leakage.fit_transform(train_sentences)
tfidf_test_no_leakage = vectorizer_no_leakage.transform(test_sentences)

# TF-IDF with Data Leakage (before splitting)
combined_sentences = train_sentences + test_sentences
combined_vocab = set(train_vocab.keys()).union(set(test_vocab.keys()))
vectorizer_leakage = TfidfVectorizer(vocabulary=list(combined_vocab),lowercase=False)
tfidf_leakage = vectorizer_leakage.fit_transform(combined_sentences)

# Generate labels based on coefficients for train and test sets
print("\nTrain Sentences Analysis:")
train_labels_no_leakage, train_labels_leakage = simulate_labels(
    tfidf_train_no_leakage,
    vectorizer_no_leakage,
    tfidf_leakage[:len(train_sentences)],
    vectorizer_leakage,
    train_vocab,
    train_sentences
)

print("\nTest Sentences Analysis:")
test_labels_no_leakage, test_labels_leakage = simulate_labels(
    tfidf_test_no_leakage,
    vectorizer_no_leakage,
    tfidf_leakage[len(train_sentences):],
    vectorizer_leakage,
    train_vocab,
    test_sentences
)

def compare_labels(train_labels, test_labels, train_leakage_labels, test_leakage_labels):
    print("\nComparison of Labels without and with Data Leakage:")
    print(f"{'Train Sentences':<20}{'Without Leakage':<20}{'With Leakage':<20}")
    for i, (orig, leak) in enumerate(zip(train_labels, train_leakage_labels)):
        print(f"{'Train '+str(i):<20}{orig:<20}{leak:<20}")

    print("\nTest Labels:")
    print(f"{'Test Sentences':<20}{'Without Leakage':<20}{'With Leakage':<20}")
    for i, (orig, leak) in enumerate(zip(test_labels, test_leakage_labels)):
        print(f"{'Test '+str(i):<20}{orig:<20}{leak:<20}")
# Compare labels to show the impact of TF-IDF changes due to leakage
compare_labels(train_labels_no_leakage, test_labels_no_leakage, train_labels_leakage, test_labels_leakage)
```
:::

::: {.cell .markdown}

**Effects of Data Leakage on TF-IDF-based Classification**

This illustrates the effects of data leakage on TF-IDF-based classification by generating and analyzing sentences from a split vocabulary with overlap. It first prepares the vocabulary by cleaning and assigning random coefficients, then splits the vocabulary into training and test sets with a specified overlap. Sentences are generated from these vocabularies, and TF-IDF values are computed for both scenarios: without and with data leakage. By comparing classification results, it demonstrates how leakage can influence TF-IDF values and ultimately affect model performance. This approach helps visualize the impact of data leakage on feature extraction and classification.

**Example Sentence Analysis:**

*Sentence 3: stag Porch lessened lessened buckshot buckshot tunes*
```
TF-IDF without Data Leakage:
Lovejoy      : TF-IDF = 0.0000, Coefficient = 0.7488, Product = 0.0000
stag         : TF-IDF = 0.3833, Coefficient = 0.7915, Product = 0.3034
Desolation   : TF-IDF = 0.0000, Coefficient = 0.0246, Product = 0.0000
arithmetical : TF-IDF = 0.0000, Coefficient = 0.6371, Product = 0.0000
brilliance   : TF-IDF = 0.0000, Coefficient = 0.9147, Product = 0.0000
buckshot     : TF-IDF = 0.9236, Coefficient = 0.5965, Product = 0.5509
crags        : TF-IDF = 0.0000, Coefficient = 0.5030, Product = 0.0000
cops         : TF-IDF = 0.0000, Coefficient = 0.4496, Product = 0.0000
Total Score  : 0.8543

TF-IDF with Data Leakage:
Lovejoy      : TF-IDF = 0.0000, Coefficient = 0.7488, Product = 0.0000
stag         : TF-IDF = 0.2648, Coefficient = 0.7915, Product = 0.2096
Desolation   : TF-IDF = 0.0000, Coefficient = 0.0246, Product = 0.0000
arithmetical : TF-IDF = 0.0000, Coefficient = 0.6371, Product = 0.0000
brilliance   : TF-IDF = 0.0000, Coefficient = 0.9147, Product = 0.0000
buckshot     : TF-IDF = 0.5296, Coefficient = 0.5965, Product = 0.3159
crags        : TF-IDF = 0.0000, Coefficient = 0.5030, Product = 0.0000
cops         : TF-IDF = 0.0000, Coefficient = 0.4496, Product = 0.0000
Total Score  : 0.5255
Boundary (Median Coefficient): 0.5832
```
**Analysis:**

For this sentence, we have five words: stag, Porch, lessened, buckshot, tunes. Among these, only two words (stag and buckshot) are recognized by the model (i.e., they exist in the training set). The remaining three words are not present in the training set but would be recognized by the vectorizer if we incorrectly used it on the whole train and test sets. Although these words do not directly affect classification, they influence the TF-IDF values of other words (e.g., stag and buckshot), leading to biased classification.

**Approach Comparison:**

Incorrect Approach: Data Leakage

1. Fit TF-IDF on the entire dataset.
2. Split the data.
3. Train & evaluate the model.

Correct Approach: No Data Leakage

1. Split the data.
2. Fit TF-IDF on the training set.
3. Train & evaluate the model.
:::