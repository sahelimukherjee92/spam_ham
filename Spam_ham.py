#Importing Libraries
import os
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import chardet
import sklearn
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# from nltk.stem.porter import PorterStemmer as PS
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# import csv
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize



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
# data['Level'] = np.where(data['Label'] == 'spam', 1, 0)



#Visualisation

from wordcloud import WordCloud, STOPWORDS

'''cdir = os.path.dirname(os.path.realpath('__file__'))'''

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



# corpus = []
# for i in range(0,data.shape[0]):
#     text = re.sub('[^a-zA-Z]', ' ', data['Text'][i]).lower().split()
#     ps = PS()
#     text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
#     text = ' '.join(text)
#     corpus.append(text)



#Creating the Bag of Words model


#
# clean_text = ' '.join(corpus)
# tokens = nltk.word_tokenize(clean_text)
# words = sorted([lemmatizer.lemmatize(word) for word in tokens if word not in set(stopwords.words('english'))])

corpus = []

for i in range(0, data.shape[0]):
    text = re.sub('[^a-zA-Z]', ' ', data['Text'][i]).lower().split()
    lemmatizer = WordNetLemmatizer()
    unique_words = sorted(set([lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]))
    filtered_words = [word for word in unique_words if len(word) > 2]
    # using set() for unique items
    clean_text = ' '.join(filtered_words)
    corpus.append(clean_text)

# corpus = []
# for i in range(0, data.shape[0]):
#     text = re.sub('[^a-zA-Z]', ' ', data['Text'][i]).lower().split()
#     ps = PS()
#     unique_words = sorted(set([ps.stem(word) for word in text if word not in set(stopwords.words('english'))]))
#     filtered_words = [word for word in unique_words if len(word) > 2]
#     # using set() for unique items
#     clean_text = ' '.join(filtered_words)
#     corpus.append(clean_text)

# toks = nltk.word_tokenize(''.join(corpus))


# def BagOfWords(text_str):
#     words = nltk.word_tokenize(''.join(text_str))
#     bags = np.zeros(len(words))
#     for sent in text_str:
#         for i, word in enumerate(words):
#             if word == sent:
#                 bags[i] = 1
#
#     return np.array(bags)
#
#
# BOW_matrix = BagOfWords(corpus)



from sklearn.feature_extraction.text import  CountVectorizer
vectorizer = CountVectorizer(max_features = 1000)
X_unigram = vectorizer.fit_transform(corpus).toarray()
Y = data.iloc[:, 2].values

Y_transpose = pd.DataFrame(Y.T)
# X_df_unigram = pd.DataFrame(X_unigram)
# print(Y_transpose.shape[0])
# print(X_df.shape)


# (2 in X)
'''countr = 0
drop_index_arr = []
while countr < Y_transpose.shape[0]:
    if Y_transpose[countr] == 1:
        print(Y_transpose[countr], type(Y_transpose[countr]))
        drop_index_arr.append(countr)
    countr += 1'''

'''print(len(drop_index_arr))
X_spam = X_df.drop(X_df.index[drop_index_arr])
print(X_spam.shape)

from scipy.stats import binom_test
#prop_X =

count = 0
success = []
while count < X_spam.shape[1]:
    counter_1 = 0
    for i in X_spam[count]:
        if i == 1:
            counter_1 +=1
    success.append(counter_1)

    #if X_spam[count] == 1:
    count+=1

print(len(success))

P_VAL = []
for i in success:
    p_val = binom_test(i, X_spam.shape[0])
    P_VAL.append(p_val)

print(P_VAL)'''

def freq_pval_finder(X, gram):
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
        # break

    success = [] # word is present
    failure = [] # word is absent
    # for i in range(0, Y_transpose.shape[0]):
    #     # print(i)
    for i, j in zip(ones, zeros):
        each_success = []
        each_failure = []
        for k in i:
            each_success.append(Y_transpose[0][k])
        for l in j:
            each_failure.append(Y_transpose[0][l])
        # break
        success.append(each_success)
        failure.append(each_failure)
            # print(Y_transpose)
        # if Y_transpose ==


    success_prop = []
    failure_prop = []
    # success_count = 0

    for i,j in zip(success, failure):
        success_prop.append(sum(i)/ len(i)) #gives the proportion of spams when the word is present
        failure_prop.append(sum(j)/len(j)) #gives the proportion of spams when the word is not present
        # while success_count < len(i):
        # each_success_count = 0
        # for j in i:
        #     if j == 1:
        #         each_success_count += 1

            # success_count += 1


    from scipy.stats import binom_test

    P_VAL = []
    for i, j in zip(success, failure_prop):
        p_val = binom_test(sum(i), len(i), j)
        P_VAL.append(p_val)

    # word_keys = []
    # for key in vectorizer.vocabulary_.keys():
    #     word_keys.append(key)

    word_keys = sorted(vectorizer.vocabulary_.items(), key = lambda x : x[1])

    # word_index = list(range(0, 100))
    # word_list = zip(word_index, vectorizer.vocabulary_)

    # wordwise_pval = zip(word_keys, P_VAL)

    # import operator
    # sorted_pval = sorted(list(wordwise_pval), key=operator.itemgetter(1))


    tokens = nltk.word_tokenize(' . '.join(corpus))
    fdist = nltk.FreqDist(ngrams(tokens, gram))

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

# df_word_freq_pval.to_csv("Wordwise_PValue_Frequency.csv")

freq_pval_finder(X_unigram)
df_unigram = freq_pval_finder(X_unigram)

df_unigram.to_csv("Unigram_PValue_Frequency.csv")



# Bigrams
# bigrams = []
# for i in corpus:
#     tokens = nltk.word_tokenize(i)
#     bigram = ngrams(tokens, 2)
#     print(Counter(bigram))

vectorizer_bigram = CountVectorizer(max_features = 1000, ngram_range = (2,2))
X_bigram = vectorizer_bigram.fit_transform(corpus).toarray()

df_bigram = freq_pval_finder(X_bigram)
df_bigram.to_csv("Bigram_PValue_Frequency.csv")






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

