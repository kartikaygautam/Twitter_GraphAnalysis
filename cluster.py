import numpy as np
import os
import re
import string
import tarfile
from TwitterAPI import TwitterAPI
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from collections import Counter, defaultdict
from itertools import chain, combinations
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import pickle


def tokenize_string(my_string):

    tokens = []

    my_string = my_string.lower()

    my_string = re.sub('@\S+', ' ', my_string)  # Remove mentions.

    my_string = re.sub('http\S+', ' ', my_string)  # Remove urls.

    tokens = re.findall('[A-Za-z]+', my_string) # Retain words.

    return tokens


def read_afinn():

    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    return afinn

afinn = read_afinn()

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO

    for token in tokens:
        token = 'token=' + str(token)
        feats[token] += 1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1),
    ('token_pair=c__d', 1)]
    """
    ###TODO

    # obtain the windows from the input list

    windows = []

    for i in range(0,len(tokens)):

        if (k <= len(tokens)):

            append_list = tokens[i:k]
            windows.append(append_list)
            k = k + 1

        else:
            break

    # obtain the combination tuples and perform operation

    for window in windows:

        combos = combinations(window, 2)

        for combo in combos:
            featsDict_keyEntry = 'token_pair=' + str(combo[0]) + "__" + str(combo[1])
            feats[featsDict_keyEntry] += 1

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO

    positiveTokens = 0
    positiveScore = 0

    negativeTokens = 0
    negativeScore = 0

    for token in tokens:
        if token in afinn:

            token_score = afinn[token]

            if token_score > 0:
                positiveTokens += 1
                positiveScore += token_score
            else:
                negativeTokens += 1
                negativeScore += -1 * token_score

    feats['pos_count'] = positiveTokens
    feats['pos_score'] = positiveScore

    feats['neg_count'] = negativeTokens
    feats['neg_score'] = negativeScore

    feats['objective_score'] = positiveScore - negativeScore

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO

    feats = defaultdict(lambda: 0)

    for function in feature_fns:
        function(tokens,feats)

    return sorted(feats.items(), key = lambda x: x[0])

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO

    result = []

    counter = Counter()

    for docTokens in tokens_list:
        feats = featurize(np.array(docTokens),feature_fns)
        result += feats


    for featEntry in result:
        if featEntry[1] != 0:
            freq = featEntry[0]
            counter[freq] += 1


    prunedFeatures = []

    for feature, freq in counter.items():
        if freq >= min_freq:
            prunedFeatures.append(feature)

    prunedFeatures = sorted(prunedFeatures)

    if vocab == None:
        vocab = defaultdict(int)
        for index,value in enumerate(prunedFeatures):
            vocab[value] = index


    csr_data = []
    csr_row = []
    csr_col = []


    for index, docToken in enumerate(tokens_list):

        feats = featurize(docToken,feature_fns)

        for feat in feats:

            if feat[0] in vocab :

                csr_data.append(feat[1])
                csr_row.append(index)
                csr_col.append(vocab[feat[0]])

    csr_data = np.array(csr_data)
    csr_row = np.array(csr_row)
    csr_col = np.array(csr_col)

    csr_matrixx = csr_matrix((csr_data, (csr_row, csr_col)), shape = (len(tokens_list),len(vocab)), dtype=np.int64)

    return csr_matrixx, vocab

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO

    result = []

    counter = Counter()

    for docTokens in tokens_list:
        feats = featurize(np.array(docTokens),feature_fns)
        result += feats


    for featEntry in result:
        if featEntry[1] != 0:
            freq = featEntry[0]
            counter[freq] += 1


    prunedFeatures = []

    for feature, freq in counter.items():
        if freq >= min_freq:
            prunedFeatures.append(feature)

    prunedFeatures = sorted(prunedFeatures)

    if vocab == None:
        vocab = defaultdict(int)
        for index,value in enumerate(prunedFeatures):
            vocab[value] = index


    csr_data = []
    csr_row = []
    csr_col = []


    for index, docToken in enumerate(tokens_list):

        feats = featurize(docToken,feature_fns)

        for feat in feats:

            if feat[0] in vocab :

                csr_data.append(feat[1])
                csr_row.append(index)
                csr_col.append(vocab[feat[0]])

    csr_data = np.array(csr_data)
    csr_row = np.array(csr_row)
    csr_col = np.array(csr_col)

    csr_matrixx = csr_matrix((csr_data, (csr_row, csr_col)), shape = (len(tokens_list),len(vocab)), dtype=np.int64)

    return csr_matrixx, vocab


def main():

    print("***************Commencing Clustering.***************")
    print(" ")

    collect_data_file = open('collect.pkl','rb')
    collect_data = pickle.load(collect_data_file)
    collect_data_file.close()

    tweets_tokenized = collect_data['collect_tweets_tokenized']

    feature_fns = [token_features, token_pair_features, lexicon_features]
    X, vocab = vectorize(tweets_tokenized, feature_fns, min_freq=1)

    featuresPerTweet = []
    for i in range(0, X.shape[0]):
        featuresPerTweet.append(X[i])
    featuresPerTweet = np.array(featuresPerTweet)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    #label_zero_indices = np.where(kmeans.labels_==0)
    #label_one_indices = np.where(kmeans.labels_==1)

    cluster_data = { "cluster_train_X": X, "cluster_train_vocab": vocab, "cluster_kmeans": kmeans }

    cluster_file = open('cluster.pkl','wb')
    pickle.dump(cluster_data, cluster_file)
    cluster_file.close()

    print("***************Clustering Completed.***************")

if __name__ == main():
    main()
