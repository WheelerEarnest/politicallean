import twitter

import numpy as np
from twit_handler import get_twitter_api, get_tweets
from readfile import read_twitter_handles, sort_politicians


def make_tweet_files(count):
  api = get_twitter_api()

  # Empty list for where the twitter text will be put later
  dem_text = []
  repub_text = []

  handles = read_twitter_handles()
  dems, repubs = sort_politicians(handles)

  for handle in dems:
    tweets = get_tweets(handle, api, count=count)
    for tweet in tweets:
      asDict = tweet.AsDict()
      tweet_text = asDict['text']
      # do the split and join to remove the url
      split_text = tweet_text.split()
      del split_text[-1]
      dem_text.append(' '.join(split_text))

  for handle in repubs:
    tweets = get_tweets(handle, api, count=count)
    for tweet in tweets:
      asDict = tweet.AsDict()
      tweet_text = asDict['text']
      split_text = tweet_text.split()
      del split_text[-1]
      repub_text.append(' '.join(split_text))

  np.save('demTweets', dem_text)
  np.save('repubTweets', repub_text)

make_tweet_files(3600)
# dT = np.load('demTweets.npy')
# tweet = dT[0]
# print(tweet)