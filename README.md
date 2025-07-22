# Fake vs True News Detection using NLP
## I. Introduction
### Domain Context
The COVID-19 pandemic in 2020 was accompanied by a surge in fake news and misinformation, especially around vaccines. This misinformation fueled public fear and resistance, disrupting vaccination campaigns globally. Accurate detection of fake versus true news is critical to support public health efforts and restore trust.

### Objectives
This project aims to develop machine learning classifiers to detect fake versus true COVID-19 related news. The goal is to provide a reliable tool that helps users discern trustworthy information, promoting informed decisions about vaccinations and pandemic measures.

### Dataset
The dataset corona_fake.csv contains 1,164 COVID-19 related news articles labeled as FAKE or TRUE with nearly balanced classes. It includes columns: Title, Text, Source, and Label. Fake news labels were unified into a single “FAKE” category for binary classification.

### Evaluation Methodology
Binary classification models are evaluated using accuracy, recall, confusion matrices, and nested cross-validation. Recall is emphasized to measure the model’s ability to correctly identify true positive cases in each class. Nested cross-validation mitigates overfitting and ensures robustness across data splits.

## II. Implementation
### Preprocessing
- Missing values filled or dropped where necessary.

- Text fields cleaned of HTML, URLs, special characters, and stopwords.

- Tokenization and normalization (lowercasing) applied.

- Exploratory data analysis performed, including frequency distributions and word clouds for fake vs true news.

- Text converted into TF-IDF vectors for modeling.

- Dataset split into 70% train and 30% test subsets.

### Baseline Models
Benchmarked models include Decision Tree, Logistic Regression, Passive-Aggressive Classifier, and a custom K-Nearest Neighbour implementation. These provide performance references against the chosen approach.

### Classification Approach
The primary model is a Multinomial Naïve Bayes classifier, suitable for categorical text data and effective in text classification due to its probabilistic modeling of word occurrences based on Bayes’ theorem. This model assumes word independence and calculates class probabilities to predict labels.

## III. Conclusions
### Evaluation Results

Multinomial Naïve Bayes:

Test Accuracy: 94%

Train Accuracy: 96%

Test Recall: 94% (FAKE), 93% (TRUE)

Train Recall: 96% (both classes)

Confusion Matrix: TP=170, TN=156, FP=10, FN=12

Nested Cross-validation Accuracy: 90% ± 5%

### Baselines Summary:

Logistic Regression matches NB test accuracy (94%)

Decision Tree and Passive-Aggressive Classifier achieve perfect train accuracy (100%)

Passive-Aggressive Classifier achieves best test accuracy and recall

K-Nearest Neighbour performs worst, especially on TRUE recall

## Summary
The Multinomial Naïve Bayes classifier achieves strong balanced performance in detecting fake vs true COVID-19 news, meeting project objectives effectively. The Passive-Aggressive Classifier performs best overall, particularly on test data.

## Future Work

- Expand dataset size and include multilingual data

- Explore deep learning approaches for potentially improved performance

- Evaluate alternative classifiers such as Support Vector Machines, which are effective in high-dimensional spaces but less suited for noisy or large datasets

- Address the naïve independence assumption limitations inherent to Naïve Bayes

## References

Dataset source: https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/corona_fake.csv

[Patwa et al., 2020]

[Meel, 2020]

[Engineer, 2019]
