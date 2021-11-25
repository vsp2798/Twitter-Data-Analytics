#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing required libraries
import pandas as pd
import tweepy
import json
import csv
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import re
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
# Download resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[11]:


import pandas as pd
df = pd.read_csv('generic_tweets.txt')


# In[12]:


df.head(10)


# In[13]:


df = df.rename(columns = {"class": "Emotions_class"})


df.head(2)


# In[16]:


df['text'][:5]


# In[17]:


for tweet_no,tweet in enumerate(df['text'][:5]):
    print(tweet_no,tweet)
    print('\n')
    
    


# In[18]:


df['length'] = df['text'].apply(len)
df


# Visualising the length of the text data. 

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


df['length'].plot.hist(bins = 100)


# In[22]:


df['length'].describe()


# #### STEP 1) DATA CLEANING

# In[23]:


df['text'].iloc[0]


# Not including other features and including only text data 

# In[24]:


df1 = df[['text']]
df1


# It can be seen that classes are equally divided between them.

# In[15]:


#df['Emotions_class'].unique()


#  All html tags and attributes (i.e., /<[^>]+>/) are removed.
#  
#  Html character codes (i.e., &...;) are replaced with an ASCII equivalent.
#  
#  All URLs are removed.
#  
#  All characters in the text are in lowercase.
#  
#  All stop words are removed. Be clear in what you consider as a stop word.
#  If a tweet is empty after pre-processing, it should be preserved as suh

# In[25]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# text_values = df1.text.values


# In[ ]:


# list_text = text_values.tolist()
# list_text
# #type(list_text)


# In[26]:


# Using Python builtin split() and len() function
# Split returns a list of words delimited by sequences of whitespace (including tabs, newlines, etc, like re's \s) 
df1.loc[:,'word_count']  = df['text'].apply(lambda x: len(str(x).split())).copy()
# Using NLTK word tokenizer and len function. Splits all punctuation excluding periods 
df1.loc[:,'word_count_nltk'] =  df['text'].apply(lambda x: len(nltk.word_tokenize(str(x)))).copy()
# Using NLTK twitter tokenizer and len function. Smarter with handles, urls and punctuation
from nltk.tokenize import TweetTokenizer
Tokenizer = TweetTokenizer()
df1.loc[:,'word_count_twitter_nltk'] =  df['text'].apply(lambda x: len(Tokenizer.tokenize(str(x)))).copy()


# In[27]:


df1.head()


# In[28]:


" ".join(Tokenizer.tokenize(str(df1['text'][0])))


# #### PREPROCESSING OF TEXTS 

# #### Converting all "texts" to lower case by using lambda function

# In[29]:


df_pp=df[["text"]]
df_pp.head()


# In[30]:


df_pp.loc[:,'lower_case_text'] = df_pp["text"].apply(lambda x: x.lower()).copy()
df_pp[['text','lower_case_text']].head()


# #### Now we have converted all upper case text data to lower cases

# #### Removing all URLS and Hashtags by using lambda functions

# In[31]:


df_pp.loc[:,'without any urls and hashtags'] = df_pp["lower_case_text"].apply(lambda x:re.sub('(@[\w]+)|(^rt\s+)|(http[s]:\/\/[\w\.-\/]+[\s]*)|(#)',' ',x))

df_pp[['lower_case_text','without any urls and hashtags']].head()


# In[32]:


df_pp[['without any urls and hashtags']]


# #### Removing Punctuations and special characters

# In[33]:


#using the re.sub to remove all special,characters,tags,numbers
df_pp.loc[:,'without any urls and hashtags and puncs']=df_pp['without any urls and hashtags'].apply(lambda x:re.sub('([^\w]+)',' ',x))
df_pp[['without any urls and hashtags','without any urls and hashtags and puncs']].head()


# #### Removing all stopwords

# In[34]:


stop_words=stopwords.words("english")

df_pp.loc[:,'without any urls,hashtags,puncs and stopwords'] = df_pp['without any urls and hashtags and puncs'].apply(lambda x: str(" ".join(x for x in x.split() if x not in stop_words)))
df_pp[['without any urls and hashtags and puncs','without any urls,hashtags,puncs and stopwords']].head()


# #### Correction of all spelling mistakes

# In[30]:


# # This takes less than 5 minutes to complete, may take longer 
# correct_spelling = []
# for i in range(200000):
#   correct_spelling.append(str(TextBlob(df_pp["without any urls,hashtags,puncs and stopwords"].iloc[i]).correct()))

# #spell_corrected=pd.DataFrame(l)  


# df_pp.loc[:,"without any urls,hashtags,puncs and stopwords with correct spelling "]=correct_spelling
# df_pp[['without any urls,hashtags,puncs and stopwords','without any urls,hashtags,puncs and stopwords with correct spelling']].head()


# #### Removing Rare words

# In[35]:


## Words are considered rare if count = 1 in all the texts

# Tokenizing each texts using split()


list_tokenized_texts = [ texts.split() for texts in df_pp['without any urls,hashtags,puncs and stopwords']]

# Concatenating all tokens in texts

all_tokenized_texts = []

for tokenized_texts in list_tokenized_texts:
    all_tokenized_texts = all_tokenized_texts + tokenized_texts
    
# Getting token frequency in descending order

    
df_all_tokenized_texts = pd.DataFrame(Counter(all_tokenized_texts).most_common(),
                             columns=['words', 'count'])
rare_words = df_all_tokenized_texts[df_all_tokenized_texts['count'] == 1]

#print(rare_words.words.tolist())

def rare_words_removal(words):
    words_without_rare = words
    for word in words.split():
        if word in rare_words.words.tolist():
            words_without_rare = words_without_rare.replace(word, '')
    return words_without_rare


df_pp.loc[:,'without any urls,hashtags,puncs,stopwords and rarewords'] =  df_pp['without any urls,hashtags,puncs and stopwords'].apply(lambda x: rare_words_removal(x)).copy()

            
    
df_pp[['without any urls,hashtags,puncs and stopwords','without any urls,hashtags,puncs,stopwords and rarewords']].head()






# #### NLTK  tokenization

# In[43]:


# # Using NLTK word tokenizer
df_pp.loc[:,'NLTK  tokenization'] =  df_pp['without any urls,hashtags,puncs,stopwords and rarewords'].apply(lambda x: nltk.word_tokenize(str(x))).copy()
df_pp[['without any urls,hashtags,puncs,stopwords and rarewords','NLTK  tokenization']].head()


# #### STEMMING AND LEMMITIZATION

# In[44]:


#snow ball stemming
from nltk.stem.snowball import SnowballStemmer
def stemmed(word_list):
    stemmed_list = []
    stem=SnowballStemmer('english')
    for word in word_list:
        stemmed_list.append(stem.stem(word))
    return stemmed_list



df_pp.loc[:,'Stemmed'] =  df_pp['NLTK  tokenization'].apply(lambda x: stemmed(x)).copy()
df_pp[['NLTK  tokenization','Stemmed']].head()


# In[45]:


#lemmatization
from nltk.stem import WordNetLemmatizer


def lemmatized(word_list):
    lemm=WordNetLemmatizer()
    lemmatized_list = []
    for word in word_list:
        lemmatized_list.append(lemm.lemmatize(word))
    return lemmatized_list



df_pp.loc[:,'Lemmatized'] =  df_pp['NLTK  tokenization'].apply(lambda x: lemmatized(x)).copy()
df_pp[['NLTK  tokenization','Lemmatized']].head()


# ### EXPLORATORY DATA ANALYSIS

# #### VISUALISING THE LENGTH OF THE LEMMATIZED TEXTS
# 

# In[54]:


df_pp['length of lemmatized'] = df_pp['Lemmatized'].apply(len)


# In[55]:


df_pp.head()


# In[57]:


df_pp['length of lemmatized'].plot.hist(bins = 100)


# In[115]:


Class_df = df[['Emotions_class','text']]

Class_df


# In[ ]:



#Class_df.hist(column = 'text', by = 'Emotions_class', bins = 60, figsize = (12,4))


# ### MODEL PREPARATION

# In[153]:


df_pp[['Lemmatized']]


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


modall = df_pp['Lemmatized'].apply(lambda x:  ' '.join(x))

modall


# #### Bringing down the number of features by requiring atleast five appearances of each token.

# In[7]:


bow = CountVectorizer(min_df = 5).fit(modall)
print(len(bow.vocabulary_))

text_bow = bow.transform(modall)

print("Shape of the Sparse Matrix:", text_bow.shape)



#print("text_bow:\n{}".format(repr(text_bow)))


#feature = vect.get_feature_names()

#print(feature)


# In[268]:


print("text_bow:\n{}".format(repr(text_bow)))


# In[271]:


feature = bow.get_feature_names()
feature


# In[ ]:





# In[233]:


#print(train.shape)


# In[262]:


#feature[:50]


# ### Finding sparsity

# In[272]:


text_bow.nnz  # Checking the amount of non zero occurences


# In[273]:


##Checking the sparsity 

sparsity = (100.0 * text_bow.nnz / (text_bow.shape[0] * text_bow.shape[1]))

sparsity  

# This basically compares the number of non zero texts and the total number of texts


# ### TERM FREQUENCY

# In[247]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[285]:


tfidf_vect = TfidfVectorizer().fit(modall) 


# In[328]:


text_tfidf = tfidf_vect.transform(modall)

print(text_tfidf)


# In[261]:


#train1.shape


# ### BAG OF WORDS MODEL PREPARATION

# #### SPLITTING THE DATASET

# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


text_train,text_test,class_train,class_test = train_test_split(text_bow, df[['Emotions_class']],test_size = 0.3)


# In[307]:


print(text_train.shape, class_train.shape)

print(text_test.shape, class_test.shape)


# In[309]:


#text_bow.shape,df[['Emotions_class']].shape


# ### First model implementation 
# 
# 
# ### (1) Logistic Regression

# In[314]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[311]:


logmodel = LogisticRegression()


# In[312]:


logmodel.fit(text_train,class_train)


# In[313]:


pred = logmodel.predict(text_test)


# In[326]:


ACC = accuracy_score(pred,class_test)

print('Accuracy using logistic regression:', ACC*100,'%')


# ###  (2) Naive Bayes

# In[320]:


from sklearn.naive_bayes import MultinomialNB


# In[322]:


NB = MultinomialNB()

NB.fit(text_train,class_train)


# In[323]:


predNB = NB.predict(text_test)


# In[327]:


ACCNB = accuracy_score(predNB,class_test)

print('Accuracy using naive_bayes :', ACCNB*100,'%')


# ### (3) KNN algorithm

# In[2]:


from sklearn.neighbors import KNeighborsClassifier


# In[3]:


knn = KNeighborsClassifier()


# In[4]:


knn.fit(text_train,class_train)


# In[ ]:


predknn = knn.fit(text_test)



ACCknn = accuracy_score(predknn,class_test)

print('Accuracy using knn :', ACCknn*100,'%')













