# Fake vs True COVID Related News Detection using NLP

<img width="994" height="556" alt="image" src="https://github.com/user-attachments/assets/112ae815-6ebb-49e9-bee3-fc117485af41" />

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

  <img width="429" height="308" alt="image" src="https://github.com/user-attachments/assets/293630dd-8ba3-41db-838f-7356b38ca3d6" />

- Tokenization and normalization (lowercasing) applied.

- Exploratory data analysis performed, including frequency distributions and word clouds for fake vs true news.

  <img width="596" height="216" alt="image" src="https://github.com/user-attachments/assets/d9e13c92-6c23-4d37-8076-c856e9f676b7" />

- Text converted into TF-IDF vectors for modeling.

- Dataset split into 70% train and 30% test subsets.

### Baseline Models
Benchmarked models include Decision Tree, Logistic Regression, Passive-Aggressive Classifier, and a custom K-Nearest Neighbour implementation. These provide performance references against the chosen approach.

### Classification Approach
The primary model is a Multinomial Naïve Bayes classifier, suitable for categorical text data and effective in text classification due to its probabilistic modeling of word occurrences based on Bayes’ theorem. This model assumes word independence and calculates class probabilities to predict labels.

## III. Conclusions
### Evaluation Results

Multinomial Naïve Bayes:

<img width="473" height="257" alt="image" src="https://github.com/user-attachments/assets/c745fdf0-4ffb-47d1-b9c5-525d72e3ce12" />

- Test Accuracy: 94%

- Train Accuracy: 96%

- Test Recall: 94% (FAKE), 93% (TRUE)

- Train Recall: 96% (both classes)

- Confusion Matrix: TP=170, TN=156, FP=10, FN=12

- Nested Cross-validation Accuracy: 90% ± 5%

<img width="299" height="302" alt="image" src="https://github.com/user-attachments/assets/de762af5-6037-4ffa-843a-081088d41d9f" />

### Baselines Summary:

- Logistic Regression matches NB test accuracy (94%)

- Decision Tree and Passive-Aggressive Classifier achieve perfect train accuracy (100%)

- Passive-Aggressive Classifier achieves best test accuracy and recall

- K-Nearest Neighbour performs worst, especially on TRUE recall

## Summary
The Multinomial Naïve Bayes classifier achieves strong balanced performance in detecting fake vs true COVID-19 news, meeting project objectives effectively. The Passive-Aggressive Classifier performs best overall, particularly on test data.

<img width="907" height="163" alt="image" src="https://github.com/user-attachments/assets/0ec69751-8b27-4bf3-99a1-9c080aec7c9a" />

## Future Work

- Expand dataset size and include multilingual data

- Explore deep learning approaches for potentially improved performance

- Evaluate alternative classifiers such as Support Vector Machines, which are effective in high-dimensional spaces but less suited for noisy or large datasets

- Address the naïve independence assumption limitations inherent to Naïve Bayes

## THE END

For more information, don't hesitate to contact me www.linkedin.com/in/patience-buxton-msc . Thank you!
