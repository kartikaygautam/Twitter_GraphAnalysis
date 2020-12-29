import re
import string
import pickle
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict
from itertools import chain, combinations


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

def get_twitter():

    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def search_tweets(twitter, params, count):

    tweets = []

    users = set()

    while True:

        request = twitter.request('statuses/filter', params)

        if request.status_code != 200:
            break

        else:

            for result in request.get_iterator():

                if 'text' in result and result['user']['name'] not in users:
                    tweets.append(result)
                    users.add(result['user']['name'])

                if len(tweets) >= count:
                    break
                #elif len(tweets) % 100 == 0:
                #    print(len(tweets))

    #tweets = [result for result in request if result['lang']=='en'] # each tweet is a dict

    #print(users)

    return tweets, users

def tokenize_string(my_string):

    tokens = []

    my_string = my_string.lower()

    my_string = re.sub('@\S+', ' ', my_string)  # Remove mentions.

    my_string = re.sub('http\S+', ' ', my_string)  # Remove urls.

    tokens = re.findall('[A-Za-z]+', my_string) # Retain words.

    return tokens


def remove_stopWords(tweets_tokenized, search_string):

    stopWords = set()
    stopWords.update(['rt','http', 'https', 'htt', 't', 's', search_string])

    for tweet_tokens_list in tweets_tokenized:
        for token in tweet_tokens_list:
            if token in stopWords:
                tweet_tokens_list.remove(token)

def main():

    twitter = get_twitter()

    search_string = 'trump' # Fetch Tweets with this keyword
    count = 2000 # Number of tweets that we want to fetch

    print("***************Commencing Data Collection.***************")
    print(" ")
    print("***************This may taka a few minutes.***************")
    print(" ")

    tweets, train_users = search_tweets(twitter, {'track': search_string, 'language':'en'}, count)
    tweets_tokenized = [tokenize_string(tweet['text']) for tweet in tweets]
    remove_stopWords(tweets_tokenized, search_string)

    collect_data = { "collect_tweets": tweets, "collect_tweets_tokenized": tweets_tokenized,
                    "collect_search_string": search_string, "collect_count": count, "collect_train_users": train_users}

    collect_file = open('collect.pkl','wb')
    pickle.dump(collect_data, collect_file)
    collect_file.close()

    print("***************Data Collection Finished.***************")

if __name__ == main():
    main()
