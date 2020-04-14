# Project Exodiux

## Summary of the idea
The idea is to build a model that is able to discern if a post in a blog belongs
to a specific topic (for example CSS, AWS, Machine Learning, ...). The second
part of the classification is about learning if it's an Beginner, Medium or
Advanced post.

Therefore, we want to detect:
* Topics to tag the article
* Proficiency of the article

### What's the value?
By building these two models, we want to be able to automatize the process to
tag an article and the proficiency to later build a search feature that will
allow to find blogs by proficiency and topics.

## Data
* We could use Stack overflow text and use categories as tag information
* Reddit posts are also categorized, it's another source of information
* Are there DataSets available?
* For non technical posts, we could use news or papers

* Minimum number of data points per category depends on the number of categories
at least would be necessary to have from 25 data points for non close related
topics, to thousands of samples; in the case of Java vs C++ it would be about
thousands of samples

## How to proceed
**Supervised learning:** we have articles and the expected categorization
**Unsupervised learning:** would help to figure out the categories wihtout having a predefined list of them

Supervised learning is easier to accomplish. You have a finite number of categories.

The learning process takes more time depending on the number of categorizations, but maybe it's not the most impactful element. The number of samples and length of the texts (number of different words).

So normalization of documents, removing stop words, ... to prepare the samples for the training.

You can create a model that gives you a probability of belong to a specfici category. For this, you would need to define the thresholds depending on the risk you want to take depending on the tags.


1. Search for data related with the data we want to classify
2. Specify the categories and make sure that the data we will use have the data necessary to understand if an article belongs to one category or another
3. Define a metric => it will allow to measure the quality of the resulting model <accuracy, precision>; we look the number of successfully classified documents
4. Format/prepare the data
5. Prepare the model
6. Use it in production
7. Keep training and improving data and models (get feedback)

We can do first an approximation with Stack Overflow entries and then adjust the models to use it for blog entries for example. This way we can get a first approximation and keep improving our models with data closer to what we want to predict.

### Data format
* CSV files can be used, but have some counterparts, because texts have commas, quotes, ... A good CSV reader library should do part of this work
* Parquet files are column documents

### Prepare data
* Text normalization:
  * lowercase
  * remove commas, dots, colons, ...
  * remove stop words
  * stemming (remove suffixes) => tries to figure out the root of the word (beautifully => beautiful, friendly => friend, cantava => cant)
  * lematization => returns the root directly (was => be, cantava => cantar, ...)
* Tokanization:
  * Break text into words (we go from an string to an array of strings keeping the order of the words)
* Other modifications like: TF-IDF, ...
* Convert text to numbers (Featuring): each lines is a word and each column if appears or not

**NOTE:** if you do steamming you don't do lematization (they are mutually exclusive)

The library (scikit learn) has functions to store the words in a sparse matrix.

We can use stack overflow datasets to train our model: https://archive.org/download/stackexchange

In order to check by language, you should check what is the language of the text and then use the specific model trained for the selected language.

Facebook created words vectors to understand the relation from english to other languages (157 languages), but this would use Deep Learning (https://fasttext.cc/docs/en/crawl-vectors.html).

### Train the model
For visualization and testing we can start with a Jupyter notebook.
After that, to add it in a pipeline is better to move the code to a format more suitable.

We want to get a function that prepares the data in the same way we did.
This will be the input of another function that will return a vector with the probabilities of each category.

We want to measure the quality of the results (of the categorization guess).

Create a vector wich for each text if to which categories do the text has. Scikit learn already has multiple functions that helps to create this vectors.

The enxt step is to select a model (for example [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), which creates linear regressions), and it will create lines to check the slope and the offset (bias). It basically trains a linear regression for each category to classify to get a line slope and offset for each. It normalizes the number to know the probability to be part of the specific category.

This model can predict the best match (only predicts a single output with the category with the higher probability). It also outputs the probablity of being a member of that category.

With predict_proba we can get an array of probabilities for each category.

In order to know the quality of the predictions we are doing, we could split 80% to training and 20% for validation process (standard). To test the quality we need metrics (accuracy_score). For a single category output this would be easy, but for multiple category probabilities, we could use a different metric.

Then we compare teh results of different models.

For a recomender multi tag problem like this, it could be interesting to use fast text. Still we will need a metric to check the accuracy of the model.

To randomize data to use for the training, we can use **shuffle** and **split** for selecting training and verification data.

For the data that we will use for training (80%), we will perform a cross validation (it's an out-of-the-box method to partitionate the training data into different splits)

If you want to make sure the data is well balanced between the different splits, we could use stratified cross validation.

If there are great number of entries with certain categories and little with others, maybe the best would be to discard those categories. This is called downsampling.


#### Tools
* **Jupyiter:** open source notebook tool that allows to document and execute code https://jupyter.org/install
* **Scikit learn:** libraries with models already implemented https://scikit-learn.org/stable/
* **Keras:** wrapper over tensorflow (deep learning at higher level)
* **Deep learning:** tensorflow


## Resources
* [Kaggle](https://www.kaggle.com/)
* [Stack overflow dataset](https://archive.org/download/stackexchange)
* [Fast text vectors](https://fasttext.cc/docs/en/crawl-vectors.html)


### Notes
* User python 3.7 if possible
