from __future__ import division
import collections
import random
import numpy as np
import sys
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import svm, cross_validation
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import codecs
from nltk.stem.porter import PorterStemmer
import os
import warnings
warnings.filterwarnings('ignore')

def number_of_chars(s):
    return len(s)


def unique_chars(s):
    s2 = ''.join(set(s))
    return len(s2)


def weighted_unique_chars(s):
    return unique_chars(s) / number_of_chars(s)


def words_count(s):
    return collections.Counter(s)


def words_counter_object(s):
    cnt = collections.Counter()
    words = s.split()
    for w in words:
        cnt[w] += 1
    return cnt


def total_words(cnt):
    sum = 0
    for k in dict(cnt).keys():
        sum += int(cnt[k])
    return sum


def most_common(cnt, n):
    for k, v in cnt.most_common(n):
        # print "most common  k = %s : v = %s" %(k,v)
        pass


def is_repeated(cnt):
    for k, v in cnt.most_common(1):
        freq = v / total_words(cnt)
        # print 'freq=',freq
        if freq > 0.5:
            return 1
    return 0


def make_feature_vector(critique, labels):
    " construct feature vector"
    feature_vector = []
    for i in range(len(critique)):
        s = critique[i]
        feature = []
        counter_obj = words_counter_object(s)

        feature.append(number_of_chars(s))
        feature.append(unique_chars(s))
        feature.append(weighted_unique_chars(s))
        feature.append(total_words(counter_obj))
        feature.append(is_repeated(counter_obj))

        feature.append(labels[i])
        feature_vector.append(feature)

    return feature_vector


def read_data():
    ''' reads data files and returns lists of comments and labels'''
    f = open(os.path.join(os.path.dirname(sys.argv[0]), r'./badCritiques.txt'), 'r')
    # f = open('bad.txt', 'r')
    bad = f.read().split('|')
    bad = [sentence.replace('\n', '') for sentence in bad]
    f.close()

    f = open(os.path.join(os.path.dirname(sys.argv[0]), r'.\goodCritiques.txt'), 'r')
    # f = open('good.txt', 'r')
    good = f.read().split('|')
    good = [sentence.replace('\n', '') for sentence in good]
    f.close()

    return bad + good, [0] * len(bad) + [1] * len(good)


def make_np_array_XY(xy):
    """ takes XY (feature + lable) lists, then makes np array for X, Y """
    a = np.array(xy)
    x = a[:, 0:-1]
    y = a[:, -1]
    return x, y


def get_f1_score(Y_test, Y_predict):
    test_size = len(Y_test)
    score = 0
    for i in range(test_size):
        if Y_predict[i] == Y_test[i]:
            score += 1
    print('Got %s out of %s' % (score, test_size))
    print('f1 macro = %.2f' % (f1_score(Y_test, Y_predict, average='macro')))
    print('f1 micro = %.2f' % (f1_score(Y_test, Y_predict, average='micro')))
    print('f1 weighted = %.2f' % (f1_score(Y_test, Y_predict, average='weighted')))


def preprocessing(critiques, labels):
    """Normalizing Text Steeming, Lematization, remove stopwords"""
    critiques_tokens = list()
    to_remove = list()
    for i in range(len(critiques)):
        try:
            critiques_tokens.append(word_tokenize(critiques[i]))
        except Exception as e:
            to_remove.append(i)

    for i in reversed(to_remove):
        labels.pop(i)
    filtered_sentences = list()
    i = 0
    to_remove = list()
    for word_token in critiques_tokens:
        new_sentence = list()
        for w in word_token:
            new_word = w.lower()
            if new_word not in stop_words:
                try:
                    new_word = porterStemmer.stem(new_word)
                    new_word = lemmatizer.lemmatize(new_word)
                    new_sentence.append(new_word)
                except Exception as e:
                    pass
        if new_sentence:
            filtered_sentences.append(new_sentence)
        else:
            to_remove.append(i)
        i += 1
    for i in reversed(to_remove):
        labels.pop(i)
    critiques = [' '.join(sentence) for sentence in filtered_sentences]
    return critiques, labels


if __name__ == '__main__':
    stop_words = set(stopwords.words('english'))
    porterStemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    critiques, labels = read_data()
    critiques, labels = preprocessing(critiques, labels)

    features_and_labels = make_feature_vector(critiques, labels)
    number_of_features = len(features_and_labels[0]) - 1

    # shuffle to mix good and bad ones
    random.seed(0)
    random.shuffle(features_and_labels)

    xy = np.array(features_and_labels)
    X = xy[:, 0:-1]
    y = xy[:, -1]

    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    classifiers = {
        'lda': LinearDiscriminantAnalysis().fit(X_train, Y_train),
        'qda': QuadraticDiscriminantAnalysis().fit(X_train, Y_train),
        'lr': LogisticRegression().fit(X_train, Y_train),
        'svc': svm.SVC(kernel='linear').fit(X_train, Y_train),
    }
    # svc = LinearDiscriminantAnalysis().fit(X_train, Y_train)

    for key, svc in classifiers.items():
        Y_predict = svc.predict(X_test)
        print(key, ':')
        # print('Y_predict:\n', Y_predict)
        # print('Y_test:   \n', Y_test)
        get_f1_score(Y_test, Y_predict)
