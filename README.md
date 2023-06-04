# Fake vs True News, using NLP
# I. Introduction
# 1. Domain-specific area
The year 2020 has been very challenging for everyone. Together with the pandemic a spike in fake news,
rumours and unfounded information around Covid-19 has risen on the Web and on social media, creating
doubts, panic and fear among many individuals. In particular, the rise of the so called “anti-vax” movements all
around the world has created additional challenges to governments for the implementation of the vaccination
campaign against the Covid-19 virus. Despite the scientific-proven and reliable information on the security and
efficacy of the life-saving Covid-19 vaccination, shared by the World Health Organization and other reliable
bodies, some of these anti vaccination movements are using fake or unfounded information to boost people’s
fear and anger against the vaccine and against governments rules to tackle the virus. Due to the fake news,
many countries have experienced violent and disruptive manifestations with some individuals even physically
attempting to prevent others receiving the vaccination. Tackling the fake information around Covid-19 is now,
more than ever, very important because these rumours can be dangerous and, as experienced by many
countries, can actually cause significant harm to people and disruption to every day’s life. To tackle fake
information, I will be creating text classifiers for fake news detection with the aim of trying and detect if a piece
of Covid-19 related information is true or false.

# 2. Objectives
The objectives of this project are to create a machine learning model which is able to detect whether any Covid19 related piece of information is fake or real; this project wants to contribute in creating a tool to spot and filter
out fake and misleading news while highlighting real, reliable news, taken from official sources and
organizations so that people can trust the government campaigns against the virus. Due to the rise in fake
news, there is an urgent need of clarity of information: this project will give people a tool they can use to confirm
scientific proven information on Covid-19. The project will help achieve this goal of clarity and reliability, with the
implementation of text classifiers to spot fake versus real information. The results may help people to be more
self-aware and critical when reading online or social media articles about the pandemic or the vaccination. It
may also help people to choose reliable sources only, when making personal decisions on getting vaccinated:
looking for true news from reliable sources and organization can help people trust the vaccination campaign
and its efficacy; it may convince people on getting the lifesaving vaccine to protect themselves and others, all
while feeling more confident and comfortable in their decision.

# 3. Dataset
The dataset chosen for this project is the “corona_fake.csv” dataset which comprises of 1164 rows and 4
columns which are: Title, Text, Source and Label. The dataset is made up of a series of Coronavirus-related
articles and piece of information in English language. The title column shows the title of the article, the column
text reports the body of the article, column source shows the source from where the article has been taken from
(this can be a website or a social media page) and finally column label explains if the article is either Fake or
True. Within these 1164 rows, the lines with false news are 575, while the lines with true news are 584. As
shown in the code below, of the news labelled as fake news, some are labelled as “Fake” (345 lines) and others
as “fake” (230 lines). To unify the Fake news and create more homogeneity as we will be implementing binary
classifiers, these 2 labels have been unified into a unique “FAKE” label.

The dataset is a Csv file, 4.29 MG big, acquired from Githubusercontent.com. (2020), more specifically, from
the following website: https://raw.githubusercontent.com/susanli2016/NLP-withPython/master/data/corona_fake.csv (https://raw.githubusercontent.com/susanli2016/NLP-withPython/master/data/corona_fake.csv) (please refer to the reference section for the full reference). From the
provided link, I have saved the dataset as a Comma Separated Values file and then uploaded it into the
Notebook using pd.read_csv as a data frame, as shown in the code below.

# 4. Evaluation methodology
This project is a machine learning binary classification problem: I’ll classify the news between two mutually
exclusive classes, Fake or Real. The models will be trained to classify between these two labels; to assess the
outputs of my investigations I will use accuracy, recall and confusion matrix as evaluation metrics. These will be
applied to the classifiers, after implementation, using the sklearn.metrics package and the accuracy_score,
confusion_matrix, classification_report metrics. The classification_report metric will show the recall but also the
precision and f1-score. Our focus will be on recall mainly because it tells us how many true positives were
actually found by our model, which means: out of all the articles that are true, how many articles are correctly
identified? I will also be looking at the accuracy (which is the ratio of number of correct predictions to the total
number of predictions) because it is a good metric for evaluating balanced data: our dataset, with 50,39% real
news and 49,61% false news, is indeed quite balanced. Accuracy and recall will be calculated for both train and
test data, for most models.
Furthermore, I will use a confusion matrix to show more insights into my model as it is also a common way of
presenting true positive, true negative, false positive and false negative predictions. The values will be
presented in the form of a matrix where the Y-axis shows the true labels while the X-axis shows the predicted
labels.
Lastly, in order to additionally evaluate the models, I will also use nested cross-validation which will perform
multiple evaluations on different test sets to then combine the scores from those evaluations. Applying nested
cross-validation will allow to examine how widely the performance varies across different training sets: if scores
are similar for all N training sets, then I can be fairly confident that the score is accurate. On the other hand, if
scores vary greatly across the N training sets, then I should probably be dubious about the accuracy of the
evaluation score. The procedure for nested cross-validation will see the original dataset divided into N subsets
called folds. For each of these folds, the model is trained using all of the data except the data in that fold; then,
the model is tested on the fold. Even though the individual folds might be too small to give accurate evaluation
scores on their own, the combined evaluation score is based on a large amount of data, and is therefore much
more reliable. Another advantage of nested cross-validation is that it can help with model overfit which occurs
when a machine learning algorithm captures the noise of the data and it fits the data too well.

# II. Implementation
# 5. Preprocessing
The uploaded dataset is firstly analysed through some data exploration and data cleaning steps. There appear
to be some empty cells with missing values, which I am replacing with an empty string, as shown in the code
below, with df.fillna(' '). The column of more interest is the label column, those 5 empty cells are dropped from
the initial dataset in order to obtain a cleaner and more comprehensive dataset (with only false or true articles)
which is then saved in the New_df dataframe: as this is a binary classification problem, it is not of use having
missing labels as the models won’t be able to classify the relative text. To pre-process the data and make it
algorithm ready, I will be removing from columns Text, Title and source all of the unnecessary special
characters, using regex library. In more detail, any html tag and URLs will be removed. Additionally, to add
homogeneity to these columns and avoid duplication of the same words, all the letters will be in lowered
characters. I will then proceed with removing all of the stop words from these 3 columns, using an implemented
cleaning function. Stop words are high frequency words with little lexical content, when using our machine
learning models later on, these can add noise to the results, hence why I am eliminating them. After getting all
of the English stop words list, the cleaning function will basically tokenize the text (done using word_tokenize
from the nltk library), removing any stop words from it and then return each cleaned tokenized sentence, joined
together to form the full text again (as shown in the printed results for each column).
Tokenizing the text will be needed for exploratory data analysis, where I will be calculating the words frequency
distribution and implementing two word clouds to include only the fake and the new bag of words. To proceed
with the exploratory data analysis, I’ll first concatenate the 3, special characters and stop word cleaned
columns, title, text and source and name the newly formed column: title_text_source and, for simplicity, rename
it as txt. To calculate the frequency distribution of txt, I first transform the dataset into string types with
txt.to_string(). The dataset counts 67221 items. I then split the text in individual strings using blank spaces “ “
and apply FreqDist() from the nltk library. As shown in the outcome, the punctuation is preventing the most
frequent words to show. So I'll remove the punctuation using again regex library. After removing the punctuation
as well, to further explore this bag of words through the use of simple visualizations, a Frequency distribution is
performed. The 10 most common words appear to be in order: coronavirus, covid19, new, virus, china, us,
gated, bill and people as shown in the Frequency Distribution bar graph with coronavirus being them most
frequent word with 413 counts. The last step of the data analysis will be implementing two word clouds to
include only the fake and the new bag of words. To create the fake news word cloud, I first filter column label of
the dataframe with only the FAKE news and place them into a separate dataframe, together with the
title_text_source column (which we remember from above, is special characters and stop word cleaned but not
punctuation cleaned). I then create a text list of all the words in this column, setting it as a string type so that I
can perform WordCloud() on it. The mask, which is a .jpg picture used for the creation of the shape “Fake”, is
uploaded and used for the implementation of the word cloud, as shown in the script below. Exactly the same
steps are then applied to create the “Real” word cloud.
The last and final step of this pre-processing phase is to transform the words in the column title_text_source
(special characters and stop word cleaned) into TFIDF Vectorized items so I can start applying the text
classifiers. Machine learning algorithms are not able to work with raw text directly, the text must be converted
into numbers, specifically, into vectors of numbers. Scikit-learn’s TfidfVectorizer() from the
sklearn.feature_extraction.text library is therefore used to convert the collection of text in title_text_source into a
vector of term/token counts. In more detail, I switch the lowercase off as we have already converted the text to
lowercase earlier, I switch use_idf on to enable inverse-document-frequency reweighting, with norm, each
output row will have unit norm and smooth_idf is switched on to avoid zero divisions. While simple count
vectorizers just count the word frequencies, in the TFIDF Vectorizer the value increases proportionally to the
count, but is offset by the frequency of the word in the corpus (some words basically appear more frequently):
this is the inverse document frequency (IDF) effect. While fit_transform learns the vocabulary dictionary and
returns a vector-term matrix, the output is then stored into item X. I then define the y variable, in which I will
store the target (or labels). Before applying the models, I am going to choose a subset of the data to split in
between train and test: 30% of the entire dataset is going to be used to test the overall performance of the
model. The split can be easily done using the train_test_split function from Sklearn package which output will
be 4 new variables X_train, X_test, y_train and y_test, ready to be trained and tested in the models.

# 6. Baseline performance
The chosen approach to evaluate this project will be to perform a binary classification task (real vs fake) with a
Multinomial Naïve Bayes machine learning algorithm. The project dataset will be then benchmarked with four
machine learning baselines.
The baselines against which I am going to compare the performance of my chosen approach will be: Decision
Tree, Logistic Regression machine learning models, referenced in the already published baseline from (Patwa
et al., 2020); a Passive - Aggressive Classifier work of (Meel, 2020) and a basic K-Nearest Neighbour algorithm
I have implemented myself from scratch.
The Decision Tree Classifier has been chosen as a baseline because it is a powerful algorithm that can’t be
affected by outliers; it can be used in any type of classification task and requires very little data preparation. A
decision tree classifier is a tree-structured classifier, where internal nodes represent the features of a dataset,
while the branches represent the decision rules and each leaf node represents the outcome. In this project we
are performing the task of binary classification, using the DecisionTreeClassifier() from the sklearn.tree library.

Entropy is the criteria used in the classifier and it measures the impurity in a given attribute by specifying
randomness in the data. The model is first applied to the training data and predicted on the test data. The
results are then evaluated and finally nested cross-validation is applied.
The second baseline is the logistic regression which is a simple classification algorithm and one of the most
efficient machine learning classifiers for binary classification problems. The target of this algorithm (or the
dependent variable) has only two possible classes, having data coded as either 1 or 0 (so it is already binary):
this feature makes the Logistic regression classifier a very good machine learning algorithm for our binary
classification task. Mathematically, a logistic regression model predicts P(Y=1) as a function of X; it is easier to
implement and very efficient to train, however it is quite sensible to outliers. In this project, we are using the
LogisticRegression() method from the sklearn.linear_model library. Again, the model is first applied to the
training data and predicted on the test data. The results are then evaluated and, finally, nested cross-validation
is applied.
The third baseline against which I am going to compare the performance of the chosen approach, is the
PassiveAggressiveClassifier() method which is a fairly new classifier as is an online-learning algorithm: it is
usually used when there is an enormous amount of data as input, from a live source (like a social media
website). This algorithm is chosen because it is very good to detect fake news on a social media website like
Twitter, where new data is being added every second. The system is trained incrementally by feeding it
instances sequentially, in small groups called mini-batches. The algorithm is defined passive because when the
prediction is correct, it keeps the model and does not make any changes; however, it is aggressive because
when the prediction is incorrect, it makes aggressive changes to the model. Python’s scikit-learn library
implementation of Passive-Aggressive classifier is the PassiveAggressiveClassifier() method, from the
sklearn.linear_model library. Again, the model is first applied to the training data and predicted on the test data.
The results are then evaluated and, finally, nested cross-validation is applied.
Finally, the fourth and last baseline is a basic K-Nearest Neighbour algorithm I have implemented myself from
scratch, following (Engineer, 2019) video. The algorithm KNN_Classifier creates a K-Nearest neighbour model
(as outlined in the code below). A KNN algorithm essentially forms a majority vote between the K most similar
instances to a given “unseen” observation. The similarity is defined according to a distance metric between two
data points (in this case the Euclidean distance). The entire training dataset is stored and when a prediction is
required, the k-most similar records to a new record from the training dataset are located. Once these
neighbours are discovered, the summary prediction can be made by returning the most common outcome. The
function I have implemented, first computes the Euclidean distance function between 2 points x1 and x2 and
stores it. Then it implements the KNN classifier by first storing k (or the neighbours) and also by storing the
training dataset and labels. The function then predicts the label for each new samples in our test dataset,
converting the outcome into a numpy array. Everything is then put together with def _predict(self,x), where one
sample x in the dataset will be chosen, the Euclidean distance calculated and the nearest neighbours selected
and the labels of the nearest neighbours chosen as well. The model is first applied to the training data and
predicted on the test data. The results are then evaluated (in this instance, no nested cross-validation is applied
as the results do not require it).

# 7. Classification approach
The chosen approach to evaluate this project will be building a Multinomial Naïve Bayes model to classify the
labels. As outlined in the pre-processing phase, our TF-IDF vectorised feature text is stored into item X, while
our target (or labels) is stored into item y. Our labels (true, fake) are the outputs of the model that we want to
predict, while the feature text is the dataset the models will use to train and test its performance. The
Multinomial Naïve Bayes model has been selected as the chosen approach, because is one of the most
popular supervised learning classifications that is used for the analysis of the categorical text data and is the
best model to be implemented in this text classification task, due to its ability to generate features (or words)
from a multinomial distribution and getting the probability of observing counts (even across various categories if
present). The algorithm used in the Multinomial Naïve Bayes model is based on the Bayes theorem and
predicts the tag of a text by calculating the probability of each tag for a given sample and then retrieving the tag
with the highest probability as the output. Briefly, the Bayes theorem estimates the likelihood of occurrence
based on prior knowledge of the event's conditions. It calculates the probability of an event occurring based on
the prior knowledge of conditions related to an event and is based on the following formula: P(A|B) = P(A) *

P(B|A)/P(B) Where we are calculating the probability of class A when predictor B is already provided. P(B) =
prior probability of B P(A) = prior probability of class A P(B|A) = occurrence of predictor B given class A
probability This formula helps in calculating the probability of the tags in the text and it is used in the Multinomial
Naïve Bayes model. The Naive assumption in the Multinomial Naive Bayes theorem is about the correlation
between the words (or features) in the document: it is assumed that every word in a sentence is independent of
the other ones, which is true in text classification problems. This means that we are not looking at entire
sentences, but rather at individual words. If the events are independent (or mutually exclusive), the joint
probability is equal to their individual probabilities, probability of words (events) occurring together in a sentence
is their individual probability. Sklearn provides an easy-to-implement object called MultinomialNB() which I run
by first fitting it to the training data and then predicting on the test data. The results are evaluated with accuracy
and recall on both train and test dataset, with a confusion matrix and with a nested cross-validation approach.

# III Conclusions
# 8. Evaluation
The results of the Multinomial Naïve Bayes classifier show accuracy on the Test and on the Train dataset at
0.94 for the first and 0.96 for the latter. Recall on the test dataset is 0.94 for FAKE and 0.93 for TRUE: which as
we recall, it means that out of all the articles that are FAKE/TRUE, these many articles were correctly identified.
On the other hand, recall on the train dataset is 0.96 for both labels. The confusion matrix is self-explanatory
and shows the following results: TP = True Positives = 170 TN = True Negatives = 156 FP = False Positives =
10 FN = False Negatives = 12 Lastly, the nested cross-validation check on our Multinomial model shows that
the accuracy is 0.90, with a standard deviation of 0.05.
The baselines result’s together with the Multinomial Naïve Bayes approach are summarized in the table below:

<img width="545" alt="image" src="https://github.com/Pat5c/Natural-Language-Processing---NLP/assets/124057584/72deb401-97d9-4a62-ba1f-da0b2a62586f">

<img width="538" alt="image" src="https://github.com/Pat5c/Natural-Language-Processing---NLP/assets/124057584/1f107ea9-c27d-4260-b5c4-aad3cfb645e6">

As we can see, the multinomial naïve bayes classifier performs pretty well compared to the other models; the
test accuracy performs as well as the logistic regression classifier with a score of 0.94. While the train accuracy
also performs well, it is however surpassed by the Decision Tree and the Passive Aggressive Classifier with a
perfect score of 1.00 for both. While the recall results on the train dataset for our Multinomial Naïve Bayes
model score a pretty good 0.96, these receive, again, a perfect score of 1.00 for both Fake and True for
Decision Tree and Passive Aggressive Classifier. The nested cross-validation accuracy with the standard
deviations performs well and aligns with the calculated model accuracies. Finally, the classifier performing the
worst is the K-Nearest Neighbour Classifier with a train Accuracy of only 0.79 and while the Train Recall on
Fake label scores a good 1, the True label scores only 0.58.
The confusion matrices are summarized below:

<img width="473" alt="image" src="https://github.com/Pat5c/Natural-Language-Processing---NLP/assets/124057584/d9acf673-c9ea-4398-a726-ad12f9b58244">

# 9. Summary and conclusions
In this project, I describe a 1164 rows length, real versus fake news detection dataset containing articles and
news related to COVID-19. The data is balanced and is used to develop binary classification algorithms for the
detection of fake or real news. The Multinomial Naïve Bayes model approach achieves very impressive results
with 94% of accuracy on the test dataset, and a 96% recall for both labels. I also benchmark the dataset using
other machine learning algorithms and project them as the potential baselines. Among these machine learning
models, the Passive-Aggressive classifier performs the best with 100% accuracy on the test dataset and perfect
recall on the train dataset of 100% and with only 6 False Positives and 8 False Negatives in the confusion
matrix. I can securely say that the chosen approach is a fairly good classifier which can successfully help us to
detect if an article provides true or fake information on Covid-19, making the objectives of this project fully
achieved. Both models can be easily transferred to any binary classification task where the text data needs to
be trained to select the correct label. It is fairly easily replicable on a different dataset, where the pre-processing
can be tailored made to ensure the text is as cleaned as possible. The approach can be replicated by others
also using deep learning instead of machine learning to implement the algorithm. Future work could be targeted
towards collecting more data to make the dataset larger or collecting multilingual data to assess if any changes
to the languages results in better or worst performance.
Alternative approaches could be using Support Vector machines or Naïve Bayes models as classifiers instead.
Potential benefits are that both models can be used with multiple class prediction problems. For Naïve Bayes,
when the assumption of independent predictors holds true, it can perform better than other models and requires
a small amount of training data to estimate the test data. Drawbacks of this alternative approach could be that
the model is naïve, which means that the assumption of independent predictors is almost impossible to have in
real life The Support Vector Machines algorithm, on the other hand, works relatively well when there is a clear
margin of separation between classes and is very effective in high dimensional spaces. However, this algorithm
is not suitable for large data sets and it does not perform very well when the data set has more noise or for
example when the target classes are overlapping.



