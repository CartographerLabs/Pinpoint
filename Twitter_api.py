import datetime
import re
import sys
import time
import tweepy
import Serializer
from ConfigManager import ConfigManager

class Twitter:
    '''
    Twitter aggregator class
    '''
    tweepy_api = None

    def __init__(self):
        '''
        Constrcutor
        '''

        twitter_config = ConfigManager.getTwitterConfig()
        consumer_key = twitter_config["consumer_key"]
        consumer_secret = twitter_config["consumer_secret"]
        access_token = twitter_config["access_token"]
        access_token_secret = twitter_config["access_token_secret"]

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.tweepy_api = tweepy.API(auth)

    def get_tweet(self, tweet_info, attempts = 1):
        '''
        returns a list of up to two tweets. This is because the provided tweet could be a quoted tweet. If this is the case
        we take that as two seperate tweets. Otherwise one tweet is returned with the necessary extracted data.
        :param tweet_info:
        :return: a list of up to two tweets with the necessary data extracted as defined in the serilizer.
        '''

        # If we've received several errors in a row then it's probably not going to fix itself.
        if attempts > 5:
            return []

        list_of_tweets = []
        tweet = None

        try:

            retweets = tweet_info.retweet_count
            likes = tweet_info.favorite_count
            date = tweet_info.created_at.timestamp()

            # Gets full tweet if normal tweet or re-tweet
            if tweet_info.retweeted:
                try:
                    tweet = tweet_info.retweeted_status.full_text
                    retweets = tweet_info.retweeted_status.retweet_count
                    likes = tweet_info.retweeted_status.favorite_count
                    tweet_info = self.tweepy_api.get_status(id=tweet_info.id, tweet_mode='extended')

                    # Gets author of tweet
                    source = tweet_info.full_text.split(":", 1)[0]
                    regex = r"RT @(.+)"
                    matchObj = re.match(regex, source)

                    if matchObj:
                        source = matchObj.group(1)
                    else:
                        source = "self"
                except AttributeError as e:
                    print(e)
                    pass

            else:
                # Gets full tweet and sets author to self
                tweet = tweet_info.full_text
                source = "self"

            # For quotes retweets we take the quoted tweet and the parent tweet as two seperate tweets.

            if tweet_info.is_quote_status:
                try:
                    quoted_id = tweet_info.quoted_status_id
                    quoted_tweet_info = self.tweepy_api.get_status(id=quoted_id, tweet_mode='extended')

                    quoted_tweet_text = quoted_tweet_info.full_text
                    quoted_source = quoted_tweet_info.user.name
                    quoted_retweets = quoted_tweet_info.retweet_count
                    quoted_likes = quoted_tweet_info.favorite_count
                    quoted_date = quoted_tweet_info.created_at.timestamp()

                    # As this function can return two tweets (i.e. a quoted tweet and normal tweet) the tweets are added to a list
                    list_of_tweets.append(Serializer.createPostDict(date=quoted_date, post_text=quoted_tweet_text, likes=quoted_likes, comments='', shares=quoted_retweets, source=quoted_source))
                except AttributeError as e:
                    print("Tweepy Twitter api error. On attempt {} \n {}".format(attempts, e))
                    pass

            # As this function can return two tweets (i.e. a quoted tweet and normal tweet) the tweets are added to a list

            if tweet is not None:
                list_of_tweets.append(Serializer.createPostDict(date=date, post_text=tweet, likes=likes, comments='', shares=retweets, source=source))

        except tweepy.RateLimitError as e:
            print("Tweepy Twitter api rate limit reached. On attempt {} \n {}".format(attempts, e))
            time.sleep(300)
            return self.get_tweet(tweet_info, attempts + 1) # if error, try again.

        except tweepy.TweepError as e:
            print("Tweepy Twitter api error. On attempt {} \n {}".format(attempts, e))
            pass

        return list_of_tweets

    def get_posts(self, username, attempts = 1):
        '''
        Loops through all tweets for the provided user
        :param username:
        :return: a list of serilised tweets
        '''

        # If a participant has enteres their username with spaces in error this will format it.
        username = username.replace(" ", "")

        # Checks attempts. If exceeded return empty list.
        if attempts > 3:
            return []

        list_of_tweets = []

        # If an @ symbol has been added to the string then it's removed.
        if str(username).startswith("@"):
            username = username[1:]

        try:
            for tweet_info in tweepy.Cursor(self.tweepy_api.user_timeline, id=username, tweet_mode='extended').items():
                # As this function can return two tweets (i.e. a quoted tweet and normal tweet) the tweets are added to a list
                list_of_tweets = list_of_tweets + self.get_tweet(tweet_info)

        except tweepy.error.TweepError as e:
            print("Tweepy Twitter api error on user {}. On Attempt {} .\n {}".format(username, attempts, e))
            time.sleep(300)
            return self.get_posts(username, sys.maxsize) #Unlinkely to be an error that can be fixed by waiting

        return list_of_tweets

    def get_user(self, user_name):
        """
        Gets a Twepy user object for a given user name
        :param user_name: a string representation of a Twitter username
        :return: a Tweepy user object, None if no user found
        """

        user = None

        try:
            user = self.tweepy_api.get_user(user_name)
        except:
            pass

        return user

    def is_valid_user(self, user_name):

        """
        Gets a Twepy user object for a given user name
        :param user_name: a string representation of a Twitter username
        :return: None if doesn't exist or suspended, user object if valid.
        """

        user = None

        try:
            user = self.tweepy_api.get_user(user_name)
            if user.suspended:
                user = None
        except:
            pass

        return user

    def get_user_post_frequency(self, user_name):
        """
        A utility function used to retrieve a users post frequency
        :param user_name:
        :return:
        """
        user = self.tweepy_api.get_user(user_name)

        created_at_time = user.created_at
        number_of_posts = user.statuses_count

        current_date = datetime.datetime.now()
        elapse_time = current_date - created_at_time

        frequency = number_of_posts/elapse_time.days

        return frequency

    def get_follower_following_frequency(self, user_name):
        """
        A utility function used to retrieve a users follower/ following frequency
        :param user_name:
        :return:
        """
        user = self.tweepy_api.get_user(user_name)
        followers_count = user.followers_count
        following_count = user.friends_count

        ration = following_count/followers_count

        return ration