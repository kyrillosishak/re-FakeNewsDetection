# re-FakeNewsDetection
In this series of notebooks, we will explore the effects of data leakage when using TF-IDF for feature extraction, specifically when applying it to the entire dataset instead of only the training set.

The first notebook introduces a toy example to demonstrate the impact of data leakage with TF-IDF:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/TFIDF_ToyExample.ipynb) [Exploring TF-IDF Data Leakage with a Toy Example](https://github.com/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/TFIDF_ToyExample.ipynb)

In this example, we simulate a scenario where TF-IDF features are extracted using the entire dataset. This can lead to data leakage, as information from the test set can unintentionally influence the training process, resulting in an overly optimistic evaluation of model performance.

The second notebook focuses on another type of data leakage known as temporal leakage. Here, we examine cases where future information leaks into the past when creating features for time series data:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/TemporalLeakage_ToyExample.ipynb) [Exploring Temporal Data Leakage](https://github.com/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/TemporalLeakage_ToyExample.ipynb)

Finally, the third notebook addresses the issue of illegitimate features that can arise when using data that should not be available at training time:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/IllegitimateFeatures_ToyExample.ipynb) [Investigating Illegitimate Features](https://github.com/kyrillosishak/re-FakeNewsDetection/blob/main/notebooks/IllegitimateFeatures_ToyExample.ipynb)

---

These notebooks can be executed on Google Colab using the buttons above. This project is part of the 2024 Summer of Reproducibility organized by the [UC Santa Cruz Open Source Program Office](https://ucsc-ospo.github.io/).

* Contributor: [Kyrillos Ishak](https://github.com/kyrillosishak)
* Mentors: [Fraida Fund](https://github.com/ffund), [Mohamed Saeed](https://github.com/mohammed183)
