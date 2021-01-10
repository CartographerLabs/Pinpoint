import csv
import json
import os
import re
import uuid

import pandas as pd

from Pinpoint.Aggregator_NGram import n_gram_aggregator
from Pinpoint.Aggregator_TfIdf import tf_idf_aggregator
from Pinpoint.Aggregator_Word2Vec import word_2_vec_aggregator
from Pinpoint.Aggregator_WordingChoice import wording_choice_aggregator
from Pinpoint.Grapher import grapher
from Pinpoint.Logger import logger
from Pinpoint.Sanitizer import sanitization
from Pinpoint.Twitter_api import Twitter


class feature_extraction():
    """
    This class is used to wrap the functionality of aggregating tweets from CSV files and extracting features pertinent
    to building a random forest extremist classifier.
    """

    # A graph used to store connections between aggregated users
    graph = grapher()
    archived_graphs = [] # an archive of the previous graphs
    # A list storing dictionaries of user ids and their features.
    tweet_user_features = []
    completed_tweet_user_features = [] # has centrality added
    # the global TF IDF model used for the Word 2 Vec model
    saved_tf_idf_model = None
    # A dictionary used for the translation of actual Twitter username to UUID
    dict_of_users = {}

    MAX_RECORD_SIZE = 1000 #todo sys.maxsize
    SHOULD_USE_LIWC = True

    # Datasets for training
    violent_words_dataset_location = None
    tf_idf_training_dataset_location = None
    outputs_location = None

    # Used for knowing which columns to access data from
    DEFAULT_USERNAME_COLUMN_ID = 0
    DEFAULT_MESSAGE_COLUMN_ID = 1
    DEFAULT_ANALYTIC_COLUMN_ID = 2
    DEFAULT_CLOUT_COLUMN_ID = 3
    DEFAULT_AUTHENTIC_COLUMN_ID = 4
    DEFAULT_TONE_COLUMN_ID = 5

    def __init__(self, violent_words_dataset_location = r"Pinpoint/violent_or_curse_word_datasets"
                 , tf_idf_training_dataset_location= r"Pinpoint/data-sets/religious_texts.csv",
                 outputs_location =r"Pinpoint/outputs"):
        """
        Constructor
        """

        self.violent_words_dataset_location = violent_words_dataset_location
        self.tf_idf_training_dataset_location = tf_idf_training_dataset_location
        self.outputs_location = outputs_location

    def _reset_stored_feature_data(self):
        """
        Resets memeber variables from a previous run. Importantly does not reset to TF IDF model.
        :return:
        """

        # A graph used to store connections between aggregated users
        self.graph = grapher()
        archived_graphs = []  # an archive of the previous graphs
        # A list storing dictionaries of user ids and their features.
        self.tweet_user_features = []
        self.completed_tweet_user_features = []  # has centrality added
        # the global TF IDF model used for the Word 2 Vec model
        self.dict_of_users = {}

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
        path = os.path.join(self.outputs_location, "users.json")
        with open(path, 'w') as outfile:
            json.dump(self.dict_of_users, outfile)

        return unique_id

    def _add_to_graph(self, originating_user_name, message):
        """
        A wrapper function used for adding a node/ connection to the graph.
        :param originating_user_name: the Twitter username
        :param message: The Tweet
        """

        # Adds node to graph so that if they don't interact with anyone they still have a centrality
        self.graph.add_node(originating_user_name)

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
        return wording_choice_aggregator().get_frequency_of_violent_or_curse_words(message, self.violent_words_dataset_location)

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
            word = sanitization().sanitize(word, self.outputs_location, force_new_data_and_dont_persisit=True)
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

        # Convert array to list
        list_of_vectors = []
        for vector in final_array_of_vectors:
            list_of_vectors.append(vector)

        return list_of_vectors

    def _process_tweet(self, user_name, message):
        """
        Wrapper function for taking a username and tweet and extracting the features.
        :param user_name:
        :param message:
        :return: a dictionary of all features from the message
        """
        self._add_to_graph(user_name, message)

        features_dict = {"post_freq": 0,
                # self.set_post_frequency(user_name), # todo post frequency data is not available in the dataset
                "follower_freq": 0,
                # self.set_follower_following_frequency(user_name), # todo post follow data is not available in the dataset
                "cap_freq": self._get_capitalised_word_frequency(message),
                "violent_freq": self._get_violent_word_frequency(message),
                "message_vector": self._get_tweet_vector(message)}

        return features_dict

    def _get_tf_idf_model(self):
        """
        A function used to retrieve the TFIDF model trained on the extremist dataset. If the model has already been
        created then the previously created model will be used.
        :return: a TF-IDF model
        """

        # if already made model, reuse
        if self.saved_tf_idf_model is None:

            tweet_data_set_name = self.tf_idf_training_dataset_location

            data_set = ""
            with open(tweet_data_set_name, 'r', encoding='cp1252') as file:
                reader = csv.reader(file)

                is_header = True
                for row in reader:

                    if is_header:
                        is_header = False
                        continue

                    # take quote from dataset and add it to dataset
                    message = row[5] # data column
                    data_set = data_set + message + "/n"

            # clean data set
            clean_data = sanitization().sanitize(data_set, self.outputs_location)

            # get ngrams
            uni_grams, bi_grams, tri_grams = ngram_aggregator = n_gram_aggregator().get_ngrams(clean_data)
            ngrams = uni_grams + bi_grams + tri_grams

            # todo The TF_IDF most important ngrams arn't being used. Should these be used instead of the other ngrams
            tf_idf_scores = tf_idf_aggregator().get_tf_idf_scores(ngrams, data_set)
            list_of_most_important_ngrams = sorted(tf_idf_scores, key=tf_idf_scores.get, reverse=True)[:50]

            # create a word 2 vec model
            model = word_2_vec_aggregator().get_model(list_of_sentences=ngrams)
            self.saved_tf_idf_model = model
        else:
            model = self.saved_tf_idf_model

        return model

    def open_wrapper(self, location, access_type, list_of_encodings = ["utf-8",'latin-1']):
        """
        A wrapper around the open built in function that has fallbacks for different encodings.
        :return:
        """
        dir_path = os.getcwd()

        for encoding in list_of_encodings:
            try:
                file = open(location, access_type, encoding=encoding)
                # Attempt to read file, if fails try other encoding
                file.readlines()
                file.seek(0)
                file.close()
                file = open(location, access_type, encoding=encoding)
                return file
            except LookupError as e:
                continue

        raise Exception("No valid encoding provided for file: '{}'. Encodings provided: '{}'".format(location, list_of_encodings))

    def _get_type_of_message_data(self, data_set_location, username_column_number= DEFAULT_USERNAME_COLUMN_ID,
                                  message_column_number = DEFAULT_MESSAGE_COLUMN_ID, has_header = True, is_extremist = None,
                                  clout_column_number = DEFAULT_CLOUT_COLUMN_ID, analytic_column_number = DEFAULT_ANALYTIC_COLUMN_ID,
                                  tone_column_number = DEFAULT_TONE_COLUMN_ID, authentic_column_number = DEFAULT_AUTHENTIC_COLUMN_ID):

        # Counts the total rows in the CSV. Used for progress reporting.
        with self.open_wrapper(data_set_location, 'r') as file:
            row_count = sum(1 for row in file)

        if row_count > self.MAX_RECORD_SIZE:
            row_count = self.MAX_RECORD_SIZE

        # Loops through all rows in the dataset CSV file.
        with self.open_wrapper(data_set_location, 'r') as file:
            reader = csv.reader(file)

            current_processed_rows = 0
            is_header = True
            for row in reader:
                current_processed_rows = current_processed_rows + 1

                # Makes sure same number for each dataset
                if current_processed_rows > self.MAX_RECORD_SIZE:
                    break

                # Skips the first entry, as it's the CSV header
                if has_header and is_header:
                    is_header = False
                    continue

                # Retrieve username
                username = row[username_column_number]
                user_unique_id = self._get_unique_id_from_username(username)

                # Attempt to get LIWC scores from csv, if not present return 0's
                try:
                    clout = row[clout_column_number]
                    analytic = row[analytic_column_number]
                    tone = row[tone_column_number]
                    authentic = row[authentic_column_number]
                except:
                    clout = 0
                    analytic = 0
                    tone = 0
                    authentic = 0

                liwc_dict = {
                    "clout":clout,
                    "analytic": analytic,
                    "tone": tone,
                    "authentic":authentic
                }

                # Retrieve Tweet for message
                tweet = row[message_column_number]
                # todo current datasets don't support this feature
                # todo followers = row[4]
                # todo number_of_posts = row[5]

                # clean/ remove markup in dataset
                tweet = tweet.replace("ENGLISH TRANSLATION:", "")
                sanitised_message = sanitization().sanitize(tweet, self.outputs_location, force_new_data_and_dont_persisit=True)

                # If no message skip entry
                if not len(tweet) > 0 or not len(sanitised_message) > 0 or sanitised_message == '' or not len(sanitised_message.split(" ")) > 0:
                    continue

                # Process Tweet and save as dict
                tweet_dict = self._process_tweet(user_unique_id, tweet)

                # If the message vector is not 200 skip (meaning that a blank message was processed)
                if not len(tweet_dict["message_vector"]) == 200:
                    continue

                if is_extremist is not None:
                    tweet_dict["is_extremist"] = is_extremist

                # Merge liwc dict with tweet dict
                tweet_dict = {**tweet_dict, **liwc_dict}

                self.tweet_user_features.append({user_unique_id: tweet_dict})

                logger().print_message("Added message from user: '{}', from dataset: '{}'. {} rows of {} completed."
                      .format(user_unique_id, data_set_location, current_processed_rows,row_count), 1)

        # Add the centrality (has to be done after all users are added to graph)
        completed_tweet_user_features = []
        # Loops through each item in the list which represents each message/ tweet
        for entry in self.tweet_user_features:
            # Loops through the data in that tweet (Should only be one entry per tweet).
            for user_id in entry:
                updated_entry = {}
                updated_entry[user_id] = entry[user_id]
                # Adds centrality
                updated_entry[user_id]["centrality"] = self.graph.get_degree_centrality_for_user(user_id)
                logger().print_message("Added '{}' Centrality for user '{}'".format(updated_entry[user_id]["centrality"], user_id),1)
                completed_tweet_user_features.append(updated_entry)
                break # Only one entry per list

        self.completed_tweet_user_features = self.completed_tweet_user_features + completed_tweet_user_features
        self.tweet_user_features = []
        self.archived_graphs.append(self.graph)
        self.graph = grapher()


    def _get_extremist_data(self, dataset_location):
        """
        This function is responsible for aggregating tweets from the extremist dataset, extracting the features, and
        saving them to a file for a model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location,is_extremist=True)

    def _get_counterpoise_data(self,dataset_location):
        """
        This function is responsible for aggregating tweets from the counterpoise (related to the topic but from
        legitimate sources, e.g. news outlets) dataset, extracting the features, and saving them to a file for a
        model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location,is_extremist=False)

    def _get_standard_tweets(self,dataset_location):
        """
        This function is responsible for aggregating tweets from the baseline (random sample of twitter posts)
        dataset, extracting the features, and saving them to a file for a model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location, is_extremist=False)


    def dump_features_for_list_of_datasets(self, feature_file_path_to_save_to, list_of_dataset_locations,
                                           force_new_dataset = True):
        """
        Saves features representing a provided dataset to a json file. Designed to be used for testing after a
        model has been created.
        :param feature_file_path_to_save_to:
        :param dataset_location:
        :return:
        """

        self._reset_stored_feature_data()

        if force_new_dataset or not os.path.isfile(feature_file_path_to_save_to):
            for dataset in list_of_dataset_locations:

                self._get_type_of_message_data(data_set_location=dataset, is_extremist=None)

            with open(feature_file_path_to_save_to, 'w') as outfile:
                json.dump(self.completed_tweet_user_features, outfile, indent=4)

        else:
            with open(feature_file_path_to_save_to, 'r') as file:
                data = file.read()

            # parse file
            self.completed_tweet_user_features = json.loads(data)


    def dump_training_data_features(self, feature_file_path_to_save_to, extremist_data_location, counterpoise_data_location,
                                    baseline_data_location, force_new_dataset = True):
        """
        The entrypoint function, used to dump all features, for all users in the extreamist, counterpoise, and baseline
        datsets to a json file.
        :param feature_file_path_to_save_to: The filepath to save the dasets to
        """

        self._reset_stored_feature_data()

        if force_new_dataset or not os.path.isfile(feature_file_path_to_save_to):

            self._get_extremist_data(extremist_data_location)
            self._get_counterpoise_data(counterpoise_data_location)
            self._get_standard_tweets(baseline_data_location)

            with open(feature_file_path_to_save_to, 'w') as outfile:
                json.dump(self.completed_tweet_user_features, outfile, indent=4)

