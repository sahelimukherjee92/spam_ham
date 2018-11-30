#Importing Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import chardet
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import  CountVectorizer
from nltk.util import ngrams
from nltk import word_tokenize
from itertools import chain
from scipy.stats import binom_test
from wordcloud import WordCloud, STOPWORDS





#Setting Working Directory
os.chdir('/home/saheli/Desktop/NLP/Spam assignment')
os.listdir()

#Importing the dataset
with open('spam.csv', 'rb') as f:
    result = chardet.detect(f.read())
data = pd.read_csv('spam.csv', encoding = result['encoding'])

data.head()
data.tail()

data = data.drop(data.columns[2:5], axis = 1)
data = data.dropna(axis = 1)

data.columns = ['Label', 'Text']

data['Level']= data['Label'].map({'spam' : 1, 'ham' :0})



#Visualisation


def create_wc(text):
    wc = WordCloud(width= 800,
                   height= 600,
                   background_color= 'black',
                   max_words= 200,
                   stopwords = set(STOPWORDS),
                   ).generate(text)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wc)
    plt.axis("off")


spam_words = ''.join(list(data[data['Level'] == 1]['Text']))

ham_words = ''.join(list(data[data['Level'] == 0]['Text']))

create_wc(spam_words)
create_wc(ham_words)

#Data Preprocessing



##Creating the Bag of Words model

corpus = []

for i in range(0, data.shape[0]):
    text = re.sub('[^a-zA-Z]', ' ', data['Text'][i]).lower().split()
    lemmatizer = WordNetLemmatizer()
    unique_words = sorted(set([lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]))
    filtered_words = [word for word in unique_words if len(word) > 2]
    # using set() for unique items
    clean_text = ' '.join(filtered_words)
    corpus.append(clean_text)


Y = data.iloc[:, 2].values

Y_transpose = pd.DataFrame(Y.T)


def freq_pval_finder(X, vectorizer, gram, alternative):
    X_df = pd.DataFrame(X)
    zeros =[]
    ones = []

    for i in X_df:
        counter = 0
        # print(X_df[i].shape)
        # break
        zero_filter = []
        ones_filter = []
        while counter < X_df[i].shape[0]:
            # print(counter)
            if X_df[i][counter] == 1:
                ones_filter.append(counter)
            if X_df[i][counter] == 0:
                zero_filter.append(counter)

        #     print(counter)
        # counter = 0
        # while counter < i[0]:
            counter += 1
        zeros.append(zero_filter)
        ones.append(ones_filter)

    success = [] # word is present
    failure = [] # word is absent

    for i, j in zip(ones, zeros):
        each_success = []
        each_failure = []
        for k in i:
            each_success.append(Y_transpose[0][k])
        for l in j:
            each_failure.append(Y_transpose[0][l])

        success.append(each_success)
        failure.append(each_failure)


    success_prop = []
    failure_prop = []

    for i,j in zip(success, failure):
        success_prop.append(sum(i)/ len(i)) #gives the proportion of spams when the word is present
        failure_prop.append(sum(j)/len(j)) #gives the proportion of spams when the word is not present

    P_VAL = []
    for i, j in zip(success, failure_prop):
        p_val = binom_test(sum(i), len(i), j, alternative = alternative)
        P_VAL.append(p_val)


    word_keys = sorted(vectorizer.vocabulary_.items(), key = lambda x : x[1])


    def word_to_ngrams(words, n):
        return [' '.join(words[i: i + n]) for i in range(len(words) - n + 1)]

    tokens = [nltk.word_tokenize(i) for i in corpus]

    ngrams = []
    for i in tokens:
        ngrams.append(word_to_ngrams(i, gram))

    word_to_grams = list(chain(*ngrams))


    fdist = nltk.FreqDist(word_to_grams)

    pval = []
    c = 0
    while c < len(P_VAL):
        pval.append((word_keys[c][0], P_VAL[c]))
        c += 1

    sorted_pval = sorted(pval, key= lambda x: x[1])

    word_freq_pval = []
    excluded_words = []
    for i in sorted_pval:
        flag = 0
        for j in fdist.most_common(1000):
            if i[0] == j[0]:
                word_freq_pval.append((i[0], i[1], j[1]))
                flag = 1
        if flag == 0:
            excluded_words.append((i[0], i[1]))

    for i in excluded_words:
        word_freq_pval.append((i[0], i[1],fdist[i[0]]))

    word_freq_pval = sorted(word_freq_pval, key = lambda x: x[1])

    filtered_freq_pval = []
    for i in word_freq_pval:
        if i[1] <= 0.02 or i[1] >= 0.8:
            filtered_freq_pval.append(i)

    df_filtered_freq_pval = pd.DataFrame(filtered_freq_pval, columns= ['Words', 'P-Value', 'Frequency'])
    return df_filtered_freq_pval

# Two-sided test

## Unigrams

vectorizer_unigram = CountVectorizer(max_features = 1000)
X_unigram = vectorizer_unigram.fit_transform(corpus).toarray()
df_unigram = freq_pval_finder(X_unigram, vectorizer_unigram, 1, 'two-sided')
df_unigram.to_csv("Unigram_PValue_Frequency.csv")


## Bigrams

vectorizer_bigram = CountVectorizer(max_features = 1000, ngram_range = (2,2))
X_bigram = vectorizer_bigram.fit_transform(corpus).toarray()
df_bigram = freq_pval_finder(X_bigram, vectorizer_bigram ,2, 'two-sided')
df_bigram.to_csv("Bigram_PValue_Frequency.csv")

## Trigrams
vectorizer_trigram = CountVectorizer(max_features = 1000, ngram_range = (3,3))
X_trigram = vectorizer_trigram.fit_transform(corpus).toarray()
df_trigram = freq_pval_finder(X_trigram, vectorizer_trigram ,3, 'two-sided')
df_trigram.to_csv("Trigram_PValue_Frequency.csv")


## One_tailed test

df_onetailed_unigram = freq_pval_finder(X_unigram, vectorizer_unigram, 1, 'greater')
df_onetailed_unigram.to_csv("OneTailedUnigram_PValue_Frequency.csv")

df_onetailed_bigram = freq_pval_finder(X_bigram, vectorizer_bigram ,2, 'greater')
df_onetailed_bigram.to_csv("OneTailedBigram_PValue_Frequency.csv")

df_onetailed_trigram = freq_pval_finder(X_trigram, vectorizer_trigram ,3, 'greater')
df_onetailed_trigram.to_csv("OneTailedTrigram_PValue_Frequency.csv")







# #Splitting into Training & Test datasets
#
# from sklearn.model_selection import train_test_split as split
# X_train, X_test, Y_train, Y_test = split(X, Y, test_size = 0.2, random_state = 0)
#
#
# #Fitting the MultinomialNB model
# from sklearn.naive_bayes import MultinomialNB
# model = MultinomialNB()
# model.fit(X_train, Y_train)
#
# prediction_MNB = model.predict(X_test)
#
# #Model Evaluation
# from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score
#
# def metrics(df, pred):
#     print('Confusion Matrix :','\n', confusion_matrix(df, pred))
#     print('Accuracy:', accuracy_score(df, pred), sep = '\t')
#     print('F1 score:', f1_score(df, pred), sep = '\t')
#     print('Recall:', recall_score(df, pred), sep = '\t')
#     print('Precision:', precision_score(df, pred), sep = '\t')
#
# metrics(Y_test, prediction_MNB)
#

