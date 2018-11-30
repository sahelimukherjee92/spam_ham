# data['Level'] = np.where(data['Label'] == 'spam', 1, 0)


# corpus = []
# for i in range(0,data.shape[0]):
#     text = re.sub('[^a-zA-Z]', ' ', data['Text'][i]).lower().split()
#     ps = PS()
#     text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
#     text = ' '.join(text)
#     corpus.append(text)





#
# clean_text = ' '.join(corpus)
# tokens = nltk.word_tokenize(clean_text)
# words = sorted([lemmatizer.lemmatize(word) for word in tokens if word not in set(stopwords.words('english'))])


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

# while success_count < len(i):
# each_success_count = 0
# for j in i:
#     if j == 1:
#         each_success_count += 1

# success_count += 1


# word_keys = []
    # for key in vectorizer.vocabulary_.keys():
    #     word_keys.append(key)


# word_index = list(range(0, 100))
# word_list = zip(word_index, vectorizer.vocabulary_)

# wordwise_pval = zip(word_keys, P_VAL)

# import operator
# sorted_pval = sorted(list(wordwise_pval), key=operator.itemgetter(1))
