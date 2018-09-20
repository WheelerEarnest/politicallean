import tweepy as tw
import numpy as np
import twitter
import json

def get_twitter_api():

  api = twitter.Api(consumer_key='redacted',
                    consumer_secret='redacted',
                    access_token_key='redacted',
                    access_token_secret='redacted',
                    sleep_on_rate_limit=True)

  return api

# print(api.rate_limit_status())

def get_tweets(screen_name, api, count=200):
  """
  Gets the last 200 tweets of the given user
  :param screen_name: screen name of the user in question
  :param api: instance of the twitter api
  :return: returns a list of tweets of that person
  """

  tweets = []
  if count <= 200:
    tweets = api.GetUserTimeline(screen_name=screen_name, count=count)
  if count > 200:
    tweets = api.GetUserTimeline(screen_name=screen_name, count=200)
    n = int(count / 200)
    for i in range(n):
      print(len(tweets))
      last_tweet = tweets[-1]
      id = last_tweet.id
      if i == (n-1):
        if count%200 == 0:
          break
        tweets.extend(api.GetUserTimeline(screen_name=screen_name, max_id=id, count=(count%200)))
      else:
        tweets.extend(api.GetUserTimeline(screen_name=screen_name, max_id=id, count=200))



  return tweets


#
api = get_twitter_api()
print(api.CheckRateLimit('https://api.twitter.com/1.1/statuses/user_timeline.json'))
# fun = get_tweets('JeffFlake', api, 800)
# print(len(fun))
# print(fun[380:400])
# print(fun[400:420])
# print(api.CheckRateLimit('https://api.twitter.com/1.1/statuses/user_timeline.json'))
# diction = fun[0].AsDict()
# print(diction['text'])
# print(api.CheckRateLimit('https://api.twitter.com/1.1/statuses/user_timeline.json'))