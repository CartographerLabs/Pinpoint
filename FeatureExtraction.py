import csv
import json
import re
import uuid

import pandas as pd
import uuid

from Twitter_api import Twitter
from Grapher import grapher
from Logger import logger
from Aggregator_NGram import n_gram_aggregator
from Sanitizer import sanitization
from Aggregator_TfIdf import tf_idf_aggregator
from Aggregator_Word2Vec import word_2_vec_aggregator
from Aggregator_WordingChoice import wording_choice_aggregator

class feature_extraction():
    """
    This class is used to wrap the functionality of aggregating tweets from CSV files and extracting features pertinent
    to building a random forest extremist classifier.
    """

    # A graph used to store connections between aggregated users
    graph = grapher()
    # A list storing dictionaries of user ids and their features.
    user_features = []
    # the global TF IDF model used for the Word 2 Vec model
    saved_tf_idf_model = None
    # A dictionary used for the translation of actual Twitter username to UUID
    dict_of_users = {}

    def _get_unique_id_from_username(self, username):
        """
        A function used to retrieve a UUID based on a twitter username. If a username has been used before the same UUID
        will be returned as it is stored in a dictionary.
        :param username:
        :return: a string representation of a UUID relating to a Twitter username
        """

        if username in self.dict_of_users:
            # username already in dictionary
            unique_id = self.dict_of_users[username]
        else:
            # make new UUID
            unique_id = uuid.uuid4().hex
            while unique_id in self.dict_of_users.values():
                # stops uuid collisions
                unique_id = uuid.uuid4().hex

            # Add new user id to dictionary
            self.dict_of_users[username] = unique_id

        # todo it's less efficient writing the whole file every run
        with open('users.json', 'w') as outfile:
            json.dump(self.dict_of_users, outfile)

        return unique_id

    def _add_to_graph(self, originating_user_name, message):
        """
        A wrapper function used for adding a node/ connection to the graph.
        :param originating_user_name: the Twitter username
        :param message: The Tweet
        """

        # Process mentions
        mentions = re.findall("(@.+) ", message)

        # For all mentions in the tweet add them to the graph as a node
        for mention in mentions:
            recipient_user_name = mention.replace("@", "")
            self.graph.add_edge_wrapper(originating_user_name, recipient_user_name, 1, "mention")

        # process hashtags
        hashtags = re.findall("(#.+) ", message)

        # For all hashtags in the tweet add them to the graph as a node
        for hashtag in hashtags:
            hashtag = hashtag.replace("#", "")
            self.graph.add_edge_wrapper(originating_user_name, hashtag, 1, "hashtag")

    def _get_post_frequency(self, user_name):
        """
        A wrapper function used to return a given twitter users post frequency.
        :param user_name:
        :return: A given users post frequency
        """

        return Twitter().get_user_post_frequency(user_name)

    def _get_follower_following_frequency(self, user_name):
        """
        A wrapper function used to retrieve the follower/ following frequency of a twitter user
        :param user_name:
        :return: returns the follower/ following frequency for a iven twitter user
        """
        return Twitter().get_follower_following_frequency(user_name)

    def _get_capitalised_word_frequency(self, message):
        """
        A wrapper function for returning the frequency of capitalised words in a message.
        :param message:
        :return: the frequency of capitalised words in a message.
        """
        return wording_choice_aggregator().get_frequency_of_capatalised_words(
            message)  # NEEDS TO BE DONE before lower case

    def _get_violent_word_frequency(self, message):
        """
        A wrapper function used to retrieve the frequency of violent words in a message.
        :param message: a string representation of a social media message
        :return: The frequency of violent words in the message
        """
        return wording_choice_aggregator().get_frequency_of_violent_or_curse_words(message)

    def _get_tweet_vector(self, message):
        """
        A wrapper function used retrieve the 200 size vector representation (Average and Max vector concatenated)
        of that message.
        :param message: a string representation of a message
        :param tf_idf_model:
        :return: a 200 size vector of the tweet
        """
        vectors = []
        tf_idf_model = self._get_tf_idf_model()

        for word in message.split(" "):
            word = sanitization().sanitize(word, force_new_data_and_dont_persisit=True)
            try:
                vectors.append(tf_idf_model.wv[word])
                logger().print_message("Word '{}' in vocabulary...".format(word))
            except KeyError as e:
                pass
                logger().print_message(e)
                logger().print_message("Word '{}' not in vocabulary...".format(word))

        # Lists of the values used to store the max and average vector values
        max_value_list = []
        average_value_list = []

        # Check for if at least one word in the message is in the vocabulary of the model
        final_array_of_vectors = pd.np.zeros(100)
        if len(vectors) > 0:

            # Loop through the elements in the vectors
            for iterator in range(vectors[0].size):

                list_of_all_values = []

                # Loop through each vector
                for vector in vectors:
                    value = vector[iterator]
                    list_of_all_values.append(value)

                average_value = sum(list_of_all_values) / len(list_of_all_values)
                max_value = max(list_of_all_values)
                max_value_list.append(max_value)
                average_value_list.append(average_value)

            final_array_of_vectors = pd.np.append(pd.np.array([max_value_list]), pd.np.array([average_value_list]))
        return final_array_of_vectors

    def _process_tweet(self, user_name, message):
        """
        Wrapper function for taking a username and tweet and extracting the features.
        :param user_name:
        :param message:
        :return: a dictionary of all features from the message
        """
        self._add_to_graph(user_name, message)
        # Psychological Signals
        return {"post_freq": 0,
                # self.set_post_frequency(user_name), # todo post frequency data is not available in the dataset
                "follower_freq": 0,
                # self.set_follower_following_frequency(user_name), # todo post follow data is not available in the dataset
                "cap_freq": self._get_capitalised_word_frequency(message),
                "violent_freq": self._get_violent_word_frequency(message),
                "message_vector": self._get_tweet_vector(message)}

    def _get_tf_idf_model(self):
        """
        A function used to retrieve the TFIDF model trained on the extremist dataset. If the model has already been
        created then the previously created model will be used.
        :return: a TF-IDF model
        """

        # if already made model, reuse
        if self.saved_tf_idf_model is None:

            tweet_data_set_name = "data-sets/religious_texts.csv"

            data_set = ""
            with open(tweet_data_set_name, 'r', encoding='cp1252') as file:
                reader = csv.reader(file)

                is_header = True
                for row in reader:

                    if is_header:
                        is_header = False
                        continue

                    # take quote from dataset and add it to dataset
                    message = row[5]
                    data_set = data_set + message + "/n"

            # clean data set
            clean_data = sanitization().sanitize(data_set)

            # get ngrams
            uni_grams, bi_grams, tri_grams = ngram_aggregator = n_gram_aggregator().get_ngrams(clean_data)
            ngrams = uni_grams + bi_grams + tri_grams

            # todo The TF_IDF most important ngrams arn't being used. Should these be used instead of the other ngrams
            tf_idf_scores = tf_idf_aggregator().get_tf_idf_scores(ngrams, data_set)
            list_of_most_important_ngrams = sorted(tf_idf_scores, key=tf_idf_scores.get, reverse=True)[:50]

            # create a word 2 vec model
            model = word_2_vec_aggregator().get_model(list_of_sentences=ngrams)
        else:
            model = self.saved_tf_idf_model

        return model

    def _get_extremist_data(self):
        """
        This function is responsible for aggregating tweets from the extremist dataset, extracting the features, and
        saving them to a file for a model to be created.
        """
        tweet_data_set_name = "data-sets/tweets.csv"

        # Counts the total rows in the CSV. Used for progress reporting.
        with open(tweet_data_set_name, 'r', encoding="utf-8") as file:
            row_count = sum(1 for row in file)

        # Loops through all rows in the dataset CSV file.
        with open(tweet_data_set_name, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)

            current_processed_rows = 0
            is_header = True
            for row in reader:
                current_processed_rows = current_processed_rows + 1

                # Skips the first entry, as it's the CSV header
                if is_header:
                    is_header = False
                    continue

                username = row[1]
                user_unique_id = self._get_unique_id_from_username(username)

                tweet = row[7]
                # todo followers = row[4]
                # todo number_of_posts = row[5]

                # clean/ remove markup in dataset
                tweet = tweet.replace("ENGLISH TRANSLATION:", "")
                sanitised_message = sanitization().sanitize(tweet, force_new_data_and_dont_persisit=True)

                # If no message skip entry
                if not len(tweet) > 0 and not len(sanitised_message) > 0:
                    continue

                tweet_dict = self._process_tweet(user_unique_id, tweet)
                tweet_dict["is_extremist"] = True
                self.user_features.append({user_unique_id: tweet_dict})

                print("user {} completed | of extremist - row {} of {}".format(user_unique_id, current_processed_rows,
                                                                               row_count))

    def _get_counterpoise_data(self):
        """
        This function is responsible for aggregating tweets from the counterpoise (related to the topic but from
        legitimate sources, e.g. news outlets) dataset, extracting the features, and saving them to a file for a
        model to be created.
        """
        tweet_data_set_name = "data-sets/AboutIsis.csv"

        # Counts the total rows in the CSV. Used for progress reporting.
        with open(tweet_data_set_name, 'r', encoding="utf-8") as file:
            row_count = sum(1 for row in file)

        # Loops through all rows/ entries in the counterpoise dataset
        with open(tweet_data_set_name, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)

            current_processed_rows = 0
            is_header = True
            for row in reader:
                current_processed_rows = current_processed_rows + 1

                if is_header:
                    is_header = False
                    continue

                username = row[0]
                user_unique_id = self._get_unique_id_from_username(username)
                tweet = row[4]

                sanitised_message = sanitization().sanitize(tweet, force_new_data_and_dont_persisit=True)

                # If no message skip entry
                if not len(tweet) > 0 and not len(sanitised_message) > 0:
                    continue

                tweet_dict = self._process_tweet(user_unique_id, tweet)
                tweet_dict["is_extremist"] = False
                self.user_features.append({user_unique_id: tweet_dict})

                print(
                    "user {} completed | of counterpoise - row {} of {}".format(user_unique_id, current_processed_rows,
                                                                                row_count))

    def _get_standard_tweets(self):
        """
        This function is responsible for aggregating tweets from the baseline (random sample of twitter posts)
        dataset, extracting the features, and saving them to a file for a model to be created.
        """
        tweet_data_set_name = "data-sets/normal_tweets.csv"

        # Counts the total rows in the CSV. Used for progress reporting.
        with open(tweet_data_set_name, 'r', encoding='latin-1') as file:
            row_count = sum(1 for row in file)

        with open(tweet_data_set_name, 'r', encoding='latin-1') as file:
            reader = csv.reader(file)

            current_processed_rows = 0
            for row in reader:
                current_processed_rows = current_processed_rows + 1

                username = row[4]
                user_unique_id = self._get_unique_id_from_username(username)
                tweet = row[5]
                sanitised_message = sanitization().sanitize(tweet, force_new_data_and_dont_persisit=True)

                # If no message skip entry
                if not len(tweet) > 0 and not len(sanitised_message) > 0:
                    continue

                tweet_dict = self._process_tweet(user_unique_id, tweet)
                tweet_dict["is_extremist"] = False
                self.user_features.append({user_unique_id: tweet_dict})

                print("user {} completed | of standard - row {} of {}".format(user_unique_id, current_processed_rows,
                                                                              row_count))

    def dump_json_of_features(self, feature_file_path):
        """
        The entrypoint function, used to dump all features, for all users in the extreamist, counterpoise, and baseline
        datsets to a json file.
        :param feature_file_path: The filepath to save the dasets to
        """
        self._get_extremist_data()
        self._get_counterpoise_data()
        self._get_standard_tweets()

        # Add the centrality (has to be done after all users are added to graph)
        for entry in self.user_features:
            # should be only one user id per entry in list
            for user_id in entry:
                self.user_features[user_id]["centrality"] = self.graph.get_degree_centrality_for_user(user_id)

        try:
            with open(feature_file_path, 'a') as outfile:
                json.dump(self.user_features, outfile)

        except:
            with open(feature_file_path, 'w') as outfile:
                json.dump(self.user_features, outfile)
