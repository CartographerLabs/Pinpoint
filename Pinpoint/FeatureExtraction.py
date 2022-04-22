import ast
import base64
import codecs
import csv
import gc
import json
import os
import pickle
import re
import shutil
import time

import numpy
import pandas as pd
import uuid
from scipy.spatial import distance

from Pinpoint.Aggregator_NGram import n_gram_aggregator
from Pinpoint.Aggregator_TfIdf import tf_idf_aggregator
from Pinpoint.Aggregator_Word2Vec import word_2_vec_aggregator
from Pinpoint.Aggregator_WordingChoice import wording_choice_aggregator
from Pinpoint.Grapher import grapher
from Pinpoint.Logger import logger
from Pinpoint.Sanitizer import sanitization, sys


class feature_extraction():
    """
    This class is used to wrap the functionality of aggregating tweets from CSV files and extracting features pertinent
    to building a random forest extremist classifier.
    """

    # A graph used to store connections between aggregated users
    graph = grapher()
    archived_graphs = []  # an archive of the previous graphs
    # A list storing dictionaries of user ids and their features.
    tweet_user_features = []
    completed_tweet_user_features = []  # has centrality added
    # the global TF IDF model used for the Word 2 Vec model
    saved_tf_idf_model = None
    # A dictionary used for the translation of actual Twitter username to UUID
    dict_of_users = {}

    # The max size for all data entries  (i.e. baseline tweets)
    MAX_RECORD_SIZE = sys.maxsize  # 3050

    # Datasets for training
    violent_words_dataset_location = None
    tf_idf_training_dataset_location = None
    outputs_location = None

    # Used for knowing which columns to access data from. For Twitter data.
    # Summary variables
    DEFAULT_USERNAME_COLUMN_ID = 0
    DEFAULT_DATE_COLUMN_ID = 1
    DEFAULT_MESSAGE_COLUMN_ID = 2
    DEFAULT_ANALYTIC_COLUMN_ID = 4
    DEFAULT_CLOUT_COLUMN_ID = 5
    DEFAULT_AUTHENTIC_COLUMN_ID = 6
    DEFAULT_TONE_COLUMN_ID = 7
    # Emotional Analysis
    DEFAULT_ANGER_COLUMN_ID = 36
    DEFAULT_SADNESS_COLUMN_ID = 37
    DEFAULT_ANXIETY_COLUMN_ID = 35
    # Personal Drives:
    DEFAULT_POWER_COLUMN_ID = 62
    DEFAULT_REWARD_COLUMN_ID = 63
    DEFAULT_RISK_COLUMN_ID = 64
    DEFAULT_ACHIEVEMENT_COLUMN_ID = 61
    DEFAULT_AFFILIATION_COLUMN_ID = 60
    # Personal pronouns
    DEFAULT_P_PRONOUN_COLUMN_ID = 13
    DEFAULT_I_PRONOUN_COLUMN_ID = 19

    # Constants for the fields in the baseline data set (i.e. ISIS magazine/ Stormfront, etc)
    DEFAULT_BASELINE_MESSAGE_COLUMN_ID = 5
    # Summary variables
    DEFAULT_BASELINE_CLOUT_COLUMN_ID = 10
    DEFAULT_BASELINE_ANALYTIC_COLUMN_ID = 9
    DEFAULT_BASELINE_TONE_COLUMN_ID = 12
    DEFAULT_BASELINE_AUTHENTIC_COLUMN_ID = 11
    # Emotional Analysis
    DEFAULT_BASELINE_ANGER_COLUMN_ID = 41
    DEFAULT_BASELINE_SADNESS_COLUMN_ID = 42
    DEFAULT_BASELINE_ANXIETY_COLUMN_ID = 40
    # Personal Drives
    DEFAULT_BASELINE_POWER_COLUMN_ID = 67
    DEFAULT_BASELINE_REWARD_COLUMN_ID = 68
    DEFAULT_BASELINE_RISK_COLUMN_ID = 69
    DEFAULT_BASELINE_ACHIEVEMENT_COLUMN_ID = 66
    DEFAULT_BASELINE_AFFILIATION_COLUMN_ID = 65
    # Personal pronouns
    DEFAULT_BASELINE_P_PRONOUN_COLUMN_ID = 18
    DEFAULT_BASELINE_I_PRONOUN_COLUMN_ID = 24

    # Used for Minkowski distance
    _average_clout = 0
    _average_analytic = 0
    _average_tone = 0
    _average_authentic = 0
    _average_anger = 0
    _average_sadness = 0
    average_anxiety = 0
    average_power = 0
    average_reward = 0
    average_risk = 0
    average_achievement = 0
    average_affiliation = 0
    average_p_pronoun = 0
    average_i_pronoun = 0

    # Used to chache messages to free memory
    MESSAGE_TMP_CACHE_LOCATION = "message_cache"

    def __init__(self, violent_words_dataset_location=None
                 , baseline_training_dataset_location=None,
                 outputs_location=r"outputs"):
        """
        Constructor

        The feature_extraction() class can be initialised with violent_words_dataset_location,
        tf_idf_training_dataset_location, and outputs_location locations. All files in the violent_words_dataset_location
        will be read (one line at a time) and added to the corpus of violent and swear words. The csv file at
        baseline_training_dataset_location is used to train the TFIDF model and a Minkowski distance score is calculated based on the LIWC scores present.

        If the constant variable need to be changed, do this by setting the member variables.
        """

        # Error if datasets not provided
        if violent_words_dataset_location is None:
            raise Exception("No Violent Words dir provided. Provide a directory that contains new line seperated "
                            "files where each line is a violent, extremist, etc word")

        if baseline_training_dataset_location is None:
            raise Exception("No baseline (TF-IDF/ Minkowski) dataset provided. Thus should be a csv file containing "
                            "extremist content and LIWC scores.")

        # Set datasets to member variables
        self.violent_words_dataset_location = violent_words_dataset_location
        self.tf_idf_training_dataset_location = baseline_training_dataset_location
        self.outputs_location = outputs_location

        # Attempt to make the outputs folder if it doesn't exist
        try:
            os.makedirs(outputs_location)
        except:
            pass

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

        # Used for Minkowski distance
        self._average_clout = 0
        self._average_analytic = 0
        self._average_tone = 0
        self._average_authentic = 0
        self._average_anger = 0
        self._average_sadness = 0
        self.average_anxiety = 0
        self.average_power = 0
        self.average_reward = 0
        self.average_risk = 0
        self.average_achievement = 0
        self.average_affiliation = 0
        self.average_p_pronoun = 0
        self.average_i_pronoun = 0

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
            # stops uuid collisions
            while unique_id in self.dict_of_users.values():
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
        mentions = re.findall("\@([a-zA-Z\-\_]+)", message)

        # For all mentions in the tweet add them to the graph as a node
        for mention in mentions:
            self.graph.add_edge_wrapper(originating_user_name, mention, 1, "mention")

        # process hashtags
        hashtags = re.findall("\#([a-zA-Z\-\_]+)", message)

        # For all hashtags in the tweet add them to the graph as a node
        for hashtag in hashtags:
            self.graph.add_edge_wrapper(originating_user_name, hashtag, 1, "hashtag")

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
        return wording_choice_aggregator().get_frequency_of_violent_or_curse_words(message,
                                                                                   self.violent_words_dataset_location)

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
            # todo add  back word = sanitization().sanitize(word, self.outputs_location, force_new_data_and_dont_persisit=True)
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

    def _process_tweet(self, user_name, message, row):
        """
        Wrapper function for taking a username and tweet and extracting the features.
        :param user_name:
        :param message:
        :return: a dictionary of all features from the message
        """
        self._add_to_graph(user_name, message)

        features_dict = {"cap_freq": self._get_capitalised_word_frequency(message),
                         "violent_freq": self._get_violent_word_frequency(message),
                         "message_vector": self._get_tweet_vector(message)}


        return features_dict

    def _get_average_liwc_scores_for_baseline_data(self):
        """
        Calculate the LIWC scores for the baseline dataset and the minkowski dataset.
        """

        # Checks if the values have already been set this run, if so don't calculate again
        # TODO what of the edge case where average clout is 0?
        if self._average_clout == 0:
            logger.print_message("Opening dataset {} for LIWC feature extraction and Minkowski distance".format(
                self.tf_idf_training_dataset_location))
            baseline_data_set_name = self.tf_idf_training_dataset_location

            clout_list = []
            analytic_list = []
            tone_list = []
            authentic_list = []
            anger_list = []
            sadness_list = []
            anxiety_list = []
            power_list = []
            reward_list = []
            risk_list = []
            achievement_list = []
            affiliation_list = []
            p_pronoun_list = []
            i_pronoun_list = []

            with open(baseline_data_set_name, 'r', encoding='cp1252') as file:
                reader = csv.reader(file)

                is_header = True
                for row in reader:

                    if is_header:
                        is_header = False
                        continue

                    # Try and access columns, if can't then LIWC fields haven't been set and should be set to 0
                    try:
                        clout = row[self.DEFAULT_BASELINE_CLOUT_COLUMN_ID]
                        analytic = row[self.DEFAULT_BASELINE_ANALYTIC_COLUMN_ID]
                        tone = row[self.DEFAULT_BASELINE_TONE_COLUMN_ID]
                        authentic = row[self.DEFAULT_BASELINE_AUTHENTIC_COLUMN_ID]
                        anger = row[self.DEFAULT_BASELINE_ANGER_COLUMN_ID]
                        sadness = row[self.DEFAULT_BASELINE_SADNESS_COLUMN_ID]
                        anxiety = row[self.DEFAULT_BASELINE_ANXIETY_COLUMN_ID]
                        power = row[self.DEFAULT_BASELINE_POWER_COLUMN_ID]
                        reward = row[self.DEFAULT_BASELINE_REWARD_COLUMN_ID]
                        risk = row[self.DEFAULT_BASELINE_RISK_COLUMN_ID]
                        achievement = row[self.DEFAULT_BASELINE_ACHIEVEMENT_COLUMN_ID]
                        affiliation = row[self.DEFAULT_BASELINE_AFFILIATION_COLUMN_ID]
                        p_pronoun = row[self.DEFAULT_BASELINE_P_PRONOUN_COLUMN_ID]
                        i_pronoun = row[self.DEFAULT_BASELINE_I_PRONOUN_COLUMN_ID]
                    except:
                        clout = 0
                        analytic = 0
                        tone = 0
                        authentic = 0
                        anger = 0
                        sadness = 0
                        anxiety = 0
                        power = 0
                        reward = 0
                        risk = 0
                        achievement = 0
                        affiliation = 0
                        p_pronoun = 0
                        i_pronoun = 0

                    clout_list.append(float(clout))
                    analytic_list.append(float(analytic))
                    tone_list.append(float(tone))
                    authentic_list.append(float(authentic))
                    anger_list.append(float(anger))
                    sadness_list.append(float(sadness))
                    anxiety_list.append(float(anxiety))
                    power_list.append(float(power))
                    reward_list.append(float(reward))
                    risk_list.append(float(risk))
                    achievement_list.append(float(achievement))
                    affiliation_list.append(float(affiliation))
                    p_pronoun_list.append(float(p_pronoun))
                    i_pronoun_list.append(float(i_pronoun))

            #  Get average for variables, used for distance score. These are member variables so that they don't
            #  have to be re-calculated on later runs
            self._average_clout = sum(clout_list) / len(clout_list)
            self._average_analytic = sum(analytic_list) / len(analytic_list)
            self._average_tone = sum(tone_list) / len(tone_list)
            self._average_authentic = sum(authentic_list) / len(authentic_list)
            self._average_anger = sum(anger_list) / len(anger_list)
            self._average_sadness = sum(sadness_list) / len(sadness_list)
            self.average_anxiety = sum(anxiety_list) / len(anxiety_list)
            self.average_power = sum(power_list) / len(power_list)
            self.average_reward = sum(reward_list) / len(reward_list)
            self.average_risk = sum(risk_list) / len(risk_list)
            self.average_achievement = sum(achievement_list) / len(achievement_list)
            self.average_affiliation = sum(affiliation_list) / len(affiliation_list)
            self.average_p_pronoun = sum(p_pronoun_list) / len(p_pronoun_list)
            self.average_i_pronoun = sum(i_pronoun_list) / len(i_pronoun_list)

        return [self._average_clout, self._average_analytic, self._average_tone, self._average_authentic,
                self._average_anger, self._average_sadness, self.average_anxiety,
                self.average_power, self.average_reward, self.average_risk, self.average_achievement,
                self.average_affiliation,
                self.average_p_pronoun, self.average_i_pronoun]

    def _get_tf_idf_model(self):
        """
        A function used to retrieve the TFIDF model trained on the extremist dataset. If the model has already been
        created then the previously created model will be used.
        :return: a TF-IDF model
        """

        # if already made model, reuse
        if self.saved_tf_idf_model is None:
            logger.print_message("Opening dataset {} for TF-IDF".format(self.tf_idf_training_dataset_location))
            baseline_data_set_name = self.tf_idf_training_dataset_location

            data_set = ""

            with open(baseline_data_set_name, 'r', encoding='cp1252') as file:
                reader = csv.reader(file)

                is_header = True
                for row in reader:

                    if is_header:
                        is_header = False
                        continue

                    # take quote from dataset and add it to dataset
                    message = row[self.DEFAULT_BASELINE_MESSAGE_COLUMN_ID]  # data column
                    data_set = data_set + message + "/n"

            # clean data set
            # todo should we be doing sanitization clean_data = sanitization().sanitize(data_set, self.outputs_location) # if so remove line below
            clean_data = data_set

            # get ngrams
            uni_grams, bi_grams, tri_grams = n_gram_aggregator().get_ngrams(clean_data)
            ngrams = uni_grams + bi_grams + tri_grams

            # todo The TF_IDF most important ngrams arn't being used. Should these be used instead of the other ngrams
            tf_idf_scores = tf_idf_aggregator().get_tf_idf_scores(ngrams, data_set)
            number_of_most_important_ngrams = int(len(ngrams) / 2)  # number is half all ngrams
            list_of_most_important_ngrams = sorted(tf_idf_scores, key=tf_idf_scores.get, reverse=True)[
                                            :number_of_most_important_ngrams]

            # create a word 2 vec model
            model = word_2_vec_aggregator().get_model(list_of_sentences=list_of_most_important_ngrams)
            self.saved_tf_idf_model = model
        else:
            model = self.saved_tf_idf_model

        return model

    def open_wrapper(self, location, access_type, list_of_encodings=["utf-8", 'latin-1', 'cp1252']):
        """
        A wrapper around the open built in function that has fallbacks for different encodings.
        :return:
        """

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
            except UnicodeDecodeError as e:
                continue

        raise Exception(
            "No valid encoding provided for file: '{}'. Encodings provided: '{}'".format(location, list_of_encodings))

    def _add_user_post_db_cache(self, user_id, dict_to_add):
        """
        Used to add data to the post message db cache used to free up memory.
        """

        if not os.path.isdir(self.MESSAGE_TMP_CACHE_LOCATION):
            os.mkdir(self.MESSAGE_TMP_CACHE_LOCATION)

        # Save file as pickle
        file_name = "{}-{}.pickle".format(user_id,int(time.time()))
        file_name = os.path.join(self.MESSAGE_TMP_CACHE_LOCATION, file_name)
        with open(file_name, 'wb') as pickle_handle:
            pickle.dump({"description":"a temporery file used for saving memory",
                         "data":dict_to_add}, pickle_handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_user_post_db_cache(self, file_name):
        """
        Retrieves data from the cache database used to free up memory.
        """
        if not os.path.isdir(self.MESSAGE_TMP_CACHE_LOCATION):
            raise Exception("Attempted to access temporery cache files before files are created")

        if not os.path.isfile(file_name):
            raise Exception("Attempted to access cache file {}, however, it does not exist".format(file_name))

        with (open(file_name, "rb")) as openfile:
            cache_data = pickle.load(openfile)

        return cache_data["data"]

    def _delete_user_post_db_cache(self):
        if os.path.isdir(self.MESSAGE_TMP_CACHE_LOCATION):
            shutil.rmtree(self.MESSAGE_TMP_CACHE_LOCATION)

    def _get_type_of_message_data(self, data_set_location, has_header=True, is_extremist=None):
        # Ensure all temp files are deleted
        self._delete_user_post_db_cache()

        # Counts the total rows in the CSV. Used for progress reporting.
        print("Starting entity count. Will count '{}'".format(self.MAX_RECORD_SIZE))

        # Read one entry at a time
        max_chunksize = 1

        header = "infer"
        if has_header:
            row_count = 0
        else:
            header = None
            row_count = 0

        for chunk in pd.read_csv(data_set_location, header=header, chunksize=max_chunksize, iterator=True,encoding='latin-1'):

            for row in chunk.iterrows():
                row_count = row_count + 1

                if row_count >= self.MAX_RECORD_SIZE:
                    break

            if row_count >= self.MAX_RECORD_SIZE:
                break

        print("Finished entity count. Number of rows is: '{}'".format(row_count))

        # Loops through all rows in the dataset CSV file.
        current_processed_rows = 0

        for chunk in pd.read_csv(data_set_location, header=header, chunksize=max_chunksize, iterator=True,encoding='latin-1'):

            for row in chunk.values.tolist():

                # Makes sure same number for each dataset
                if current_processed_rows >= row_count:
                    break

                # Retrieve username
                try:
                    username = row[self.DEFAULT_USERNAME_COLUMN_ID]
                    date = row[self.DEFAULT_DATE_COLUMN_ID]
                    user_unique_id = self._get_unique_id_from_username(username)
                except:
                    # if empty entry
                    continue
                # Attempt to get LIWC scores from csv, if not present return 0's
                try:
                    # Summary variables
                    clout = float(row[self.DEFAULT_CLOUT_COLUMN_ID])
                    analytic = float(row[self.DEFAULT_ANALYTIC_COLUMN_ID])
                    tone = float(row[self.DEFAULT_TONE_COLUMN_ID])
                    authentic = float(row[self.DEFAULT_AUTHENTIC_COLUMN_ID])
                    # Emotional Analysis
                    anger = float(row[self.DEFAULT_ANGER_COLUMN_ID])
                    sadness = float(row[self.DEFAULT_SADNESS_COLUMN_ID])
                    anxiety = float(row[self.DEFAULT_ANXIETY_COLUMN_ID])
                    # Personal Drives:
                    power = float(row[self.DEFAULT_POWER_COLUMN_ID])
                    reward = float(row[self.DEFAULT_REWARD_COLUMN_ID])
                    risk = float(row[self.DEFAULT_RISK_COLUMN_ID])
                    achievement = float(row[self.DEFAULT_ACHIEVEMENT_COLUMN_ID])
                    affiliation = float(row[self.DEFAULT_AFFILIATION_COLUMN_ID])
                    # Personal pronouns
                    i_pronoun = float(row[self.DEFAULT_I_PRONOUN_COLUMN_ID])
                    p_pronoun = float(row[self.DEFAULT_P_PRONOUN_COLUMN_ID])

                except:
                    # Summary variables
                    clout = 0
                    analytic = 0
                    tone = 0
                    authentic = 0
                    # Emotional Analysis
                    anger = 0
                    sadness = 0
                    anxiety = 0
                    # Personal Drives:
                    power = 0
                    reward = 0
                    risk = 0
                    achievement = 0
                    affiliation = 0
                    # Personal pronouns
                    i_pronoun = 0
                    p_pronoun = 0

                liwc_dict = {
                    "clout": clout,
                    "analytic": analytic,
                    "tone": tone,
                    "authentic": authentic,
                    "anger": anger,
                    "sadness": sadness,
                    "anxiety": anxiety,
                    "power": power,
                    "reward": reward,
                    "risk": risk,
                    "achievement": achievement,
                    "affiliation": affiliation,
                    "i_pronoun": i_pronoun,
                    "p_pronoun": p_pronoun,
                }

                # Calculate minkowski distance
                average_row = self._get_average_liwc_scores_for_baseline_data()

                actual_row = [clout, analytic, tone, authentic,
                              anger, sadness, anxiety,
                              power, reward, risk, achievement, affiliation,
                              p_pronoun, i_pronoun
                              ]

                try:
                    liwc_dict["minkowski"] = distance.minkowski(actual_row, average_row, 1)
                except ValueError:
                    continue

                # Retrieve Tweet for message
                tweet = str(row[self.DEFAULT_MESSAGE_COLUMN_ID])

                # clean/ remove markup in dataset
                sanitised_message = sanitization().sanitize(tweet, self.outputs_location,
                                                            force_new_data_and_dont_persisit=True)

                # If no message skip entry
                if not len(tweet) > 0 or not len(sanitised_message) > 0 or sanitised_message == '' or not len(
                        sanitised_message.split(" ")) > 0:
                    continue

                # Process Tweet and save as dict
                tweet_dict = self._process_tweet(user_unique_id, tweet, row)

                # If the message vector is not 200 skip (meaning that a blank message was processed)
                if not len(tweet_dict["message_vector"]) == 200:
                    continue

                if is_extremist is not None:
                    tweet_dict["is_extremist"] = is_extremist

                tweet_dict["date"] = date

                # Merge liwc dict with tweet dict
                tweet_dict = {**tweet_dict, **liwc_dict}

                #tweet_dict["user_unique_id"]= user_unique_id

                self._add_user_post_db_cache(user_unique_id, {user_unique_id: tweet_dict})
                #self.tweet_user_features.append()
                # TODO here save to cache json instead of list and graph

                logger().print_message("Added message from user: '{}', from dataset: '{}'. {} rows of {} completed."
                                       .format(user_unique_id, data_set_location, current_processed_rows, row_count), 1)
                current_processed_rows = current_processed_rows + 1
                print("Finished reading row")

        # Add the centrality (has to be done after all users are added to graph)
        completed_tweet_user_features = []
        # Loops through each item in the list which represents each message/ tweet

        # Loop through all data in cache file
        for cached_message_file in os.listdir(self.MESSAGE_TMP_CACHE_LOCATION):
            cached_message_file = os.fsdecode(cached_message_file)
            cached_message_file = os.path.join(self.MESSAGE_TMP_CACHE_LOCATION,cached_message_file)

            # Only process pickle files
            if not cached_message_file.endswith(".pickle"):
                continue

            print("Reading cache file: '{}'".format(cached_message_file))
            cached_message_data = self._get_user_post_db_cache(cached_message_file)
            # Loops through the data in that tweet (Should only be one entry per tweet).
            for user_id in cached_message_data.keys():
                updated_entry = {}
                updated_entry[user_id] = cached_message_data[user_id]
                # Adds centrality
                updated_entry[user_id]["centrality"] = self.graph.get_degree_centrality_for_user(user_id)
                logger().print_message(
                    "Added '{}' Centrality for user '{}'".format(updated_entry[user_id]["centrality"], user_id), 1)
                completed_tweet_user_features.append(updated_entry)
                gc.collect()
                break  # Only one entry per list


        self._delete_user_post_db_cache()
        self.completed_tweet_user_features = self.completed_tweet_user_features + completed_tweet_user_features

        self.tweet_user_features = []
        #self.archived_graphs.append(self.graph)
        self.graph = grapher()
        print("Finished messages")

    def _get_extremist_data(self, dataset_location, has_header = True):
        """
        This function is responsible for aggregating tweets from the extremist dataset, extracting the features, and
        saving them to a file for a model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location, is_extremist=True, has_header=has_header)

    def _get_counterpoise_data(self, dataset_location):
        """
        This function is responsible for aggregating tweets from the counterpoise (related to the topic but from
        legitimate sources, e.g. news outlets) dataset, extracting the features, and saving them to a file for a
        model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location, is_extremist=False)

    def _get_standard_tweets(self, dataset_location, has_header = True):
        """
        This function is responsible for aggregating tweets from the baseline (random sample of twitter posts)
        dataset, extracting the features, and saving them to a file for a model to be created.
        """

        self._get_type_of_message_data(data_set_location=dataset_location, is_extremist=False, has_header=has_header)

    def dump_features_for_list_of_datasets(self, feature_file_path_to_save_to, list_of_dataset_locations,
                                           force_new_dataset=True):
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

    def dump_training_data_features(self, feature_file_path_to_save_to, extremist_data_location,
                                    baseline_data_location, force_new_dataset=True):
        """
        The entrypoint function, used to dump all features, for all users in the extreamist, counterpoise, and baseline
        datsets to a json file.
        :param feature_file_path_to_save_to: The filepath to save the datasets to
        """

        self._reset_stored_feature_data()

        if force_new_dataset or not os.path.isfile(feature_file_path_to_save_to):
            print("Starting baseline messages")
            self._get_standard_tweets(baseline_data_location)
            print("Starting extremist messages")
            self._get_extremist_data(extremist_data_location)


            with open(feature_file_path_to_save_to, 'w') as outfile:
                json.dump(self.completed_tweet_user_features, outfile, indent=4)
