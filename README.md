# Twitter-Data-Analytics

Computed the sentiment of Twitter tweets to get insights for Canadian Elections 2019. Data cleaning and EDA has been done to identify tweets relevant to specific parties. BOW, TF-IDF and N-grams used for model preparation. Logistic regression, KNN, SVM, Random forest classifiers were implemented for model results and predicting election outcomes.


The purpose of this repo is to compute the sentiment of tweets posted recently on Canadian Elections, get insight into the Canadian Elections and answer the Research question： What can public opinion on Twitter tell us about the Canadian political landscape in 2019?

# BACKGROUND : 

Sentiment Analysis is a branch of Natural Language Processing (NLP) that allows us to determine algorithmically whether a statement or document is “positive” or “negative”. It's a technology of increasing importance in the modern society as it allows individuals and organizations to detect trends in public opinion by analyzing social media content. Keeping abreast of socio-political developments is especially important during periods of policy shifts such as election years, when both electoral candidates and companies can benefit from sentiment analysis by making appropriate changes to their campaigning and business strategies respectively. 

# REQUIREMENT :

Numpy, Scipy, Scikit, Matplotlib, Pandas, NLTK. 

# APPROACH :

Data cleaning: Design a procedure that prepares the Twitter data for analysis

Remove all html tags and attributes (i.e., /<[^>]+>/)
Replace Html character codes (i.e., &...;) with an ASCII equivalent
Remove all URLs
Remove all characters in the text are in lowercase
Remove all stop words are removed
Preserve empty tweet after pre-processing 

Exploratory data analysis: determine political party of given tweet using Bag of words, TF-IDF, N-grams

Model preparation: multiple classification algorithms for generic tweets (logistic regression, k-NN, Naive Bayes, SVM, decision trees)

Train classification model to predict the sentiment value (positive or negative)
Train multi-class classification models to predict the reason for the negative tweets. 
