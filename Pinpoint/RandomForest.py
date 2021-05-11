import csv
import json
import os
import pickle
from datetime import datetime

import pandas
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from Pinpoint import Logger


class random_forest():
    """
    A class used for creating a random forest binary classifier.
    """

    model = None
    accuracy = None
    precision = None
    recall = None
    f_measure = None

    # Model variables populated on creation or reading of file

    original_name = None
    creation_date = None

    _FRAMEWORK_VERSION = 0.2  # Used when creating a new model file
    # v0.1 - versioning added.
    # v0.2 - Added more LIWC scores and minkowski distance

    model_version = _FRAMEWORK_VERSION  # can be updated if reading and using a model file of a different version

    _outputs_folder = None
    _model_folder = None

    # Categories of features used in the model
    RADICAL_LANGUAGE_ENABLED = True  # RF-IDF Scores, Word Embeddings
    PSYCHOLOGICAL_SIGNALS_ENABLED = True  # LIWC Dictionaries, Minkowski distance
    BEHAVIOURAL_FEATURES_ENABLED = True  # frequency of tweets, followers / following ratio,  centrality

    def __init__(self, outputs_folder="outputs", model_folder=None):
        """
        Constructor

        The random_forest() class can be initialised with outputs_folder() and model_folder(). The outputs folder is
        where output files are stored and the model folder is where the model will be created if not overwritten.
        """

        if model_folder is None:
            model_folder = outputs_folder

        self._outputs_folder = outputs_folder
        self._model_folder = model_folder

    def get_features_as_df(self, features_file, force_new_dataset=True):
        """
        Reads a JSON file file and converts to a Pandas dataframe that can be used to train and test the classifier.
        :param features_file: the location of the JSON features file to convert to a dataframe
        :param force_new_dataset: if true a new CSV file will be created even if one already exists.
        :return: a Pandas dataframe with the features.
        """

        with open(features_file) as json_features_file:
            csv_file = "{}.csv".format(features_file)

            if force_new_dataset or not os.path.isfile(csv_file):
                features = json.load(json_features_file)

                # todo remove the data for the features not being used.
                filtered_list_after_filters_applied = []

                # If any of the filters are not true remove the features not requested
                column_names = []

                if self.PSYCHOLOGICAL_SIGNALS_ENABLED:
                    column_names = column_names + ["clout", "analytic", "tone", "authentic",
                                                   "anger", "sadness", "anxiety",
                                                   "power", "reward", "risk", "achievement", "affiliation",
                                                   "i_pronoun", "p_pronoun",
                                                   "minkowski"]
                if self.BEHAVIOURAL_FEATURES_ENABLED:
                    column_names = column_names + ['post_freq', 'follower_freq', 'centrality']

                if self.RADICAL_LANGUAGE_ENABLED:
                    # Add column names
                    column_names = column_names + ["cap_freq", "violent_freq"]
                    # Add the two hundred vectors columns
                    for iterator in range(1, 201):
                        column_names.append("message_vector_{}".format(iterator))

                column_names = column_names + ['is_extremist']

                if not self.BEHAVIOURAL_FEATURES_ENABLED or not self.PSYCHOLOGICAL_SIGNALS_ENABLED or self.RADICAL_LANGUAGE_ENABLED:

                    # Loops through list of dicts (messages)
                    number_of_processed_messages = 0
                    for message in features:
                        number_of_processed_messages = number_of_processed_messages + 1
                        Logger.logger.print_message(
                            "Extracting information from message {} of {} in file {}".format(
                                number_of_processed_messages,
                                len(features),
                                features_file),
                            logging_level=1)

                        # Loops through dict keys (usernames)
                        for user in message.keys():

                            message_features = message[user]

                            feature_dict = {}

                            if self.PSYCHOLOGICAL_SIGNALS_ENABLED:
                                # Summary variables
                                feature_dict["clout"] = message_features["clout"]
                                feature_dict["analytic"] = message_features["analytic"]
                                feature_dict["tone"] = message_features["tone"]
                                feature_dict["authentic"] = message_features["authentic"]

                                # Emotional Analysis
                                feature_dict["anger"] = message_features["anger"]
                                feature_dict["sadness"] = message_features["sadness"]
                                feature_dict["anxiety"] = message_features["anxiety"]

                                # Personal Drives
                                feature_dict["power"] = message_features["power"]
                                feature_dict["reward"] = message_features["reward"]
                                feature_dict["risk"] = message_features["risk"]
                                feature_dict["achievement"] = message_features["achievement"]
                                feature_dict["affiliation"] = message_features["affiliation"]

                                # Personal Pronouns
                                feature_dict["i_pronoun"] = message_features["i_pronoun"]
                                feature_dict["p_pronoun"] = message_features["p_pronoun"]

                                # Minkowski distance
                                feature_dict["minkowski"] = message_features["minkowski"]

                            if self.BEHAVIOURAL_FEATURES_ENABLED:
                                feature_dict['post_freq'] = message_features['post_freq']
                                feature_dict['follower_freq'] = message_features['follower_freq']
                                feature_dict['centrality'] = message_features['centrality']

                            if self.RADICAL_LANGUAGE_ENABLED:
                                feature_dict["message_vector"] = message_features["message_vector"]
                                feature_dict["violent_freq"] = message_features["violent_freq"]
                                feature_dict["cap_freq"] = message_features["cap_freq"]

                            feature_dict['is_extremist'] = message_features['is_extremist']

                            user = {user: feature_dict}
                            filtered_list_after_filters_applied.append(user)

                number_of_features = len(filtered_list_after_filters_applied)

                # Creates the columns for the data frame
                df = pd.DataFrame(
                    columns=column_names)

                completed_features = 0
                iterator = 0
                error_count = 0
                for message in features:
                    # should only be one user per entry
                    for user_id in message:
                        feature_data = message[user_id]
                        # ID is not included as it's hexidecimal and not float

                        row = []

                        if self.PSYCHOLOGICAL_SIGNALS_ENABLED:
                            clout = feature_data['clout']
                            analytic = feature_data['analytic']
                            tone = feature_data['tone']
                            authentic = feature_data['authentic']

                            anger = feature_data["anger"]
                            sadness = feature_data["sadness"]
                            anxiety = feature_data["anxiety"]
                            power = feature_data["power"]
                            reward = feature_data["reward"]
                            risk = feature_data["risk"]
                            achievement = feature_data["achievement"]
                            affiliation = feature_data["affiliation"]
                            i_pronoun = feature_data["i_pronoun"]
                            p_pronoun = feature_data["p_pronoun"]
                            minkowski = feature_data["minkowski"]

                            row = row + [clout, analytic, tone, authentic, anger, sadness, anxiety, power,
                                         reward, risk, achievement, affiliation, i_pronoun, p_pronoun, minkowski]

                        if self.BEHAVIOURAL_FEATURES_ENABLED:
                            post_freq = feature_data['post_freq']
                            follower_freq = feature_data['follower_freq']
                            centrality = feature_data['centrality']

                            row = row + [post_freq, follower_freq, centrality]

                        if self.RADICAL_LANGUAGE_ENABLED:
                            cap_freq = feature_data['cap_freq']
                            violent_freq = feature_data['violent_freq']
                            message_vector = feature_data['message_vector']

                            row = row + [cap_freq, violent_freq] + message_vector

                        is_extremist = feature_data['is_extremist']

                        row = row + [is_extremist]
                        try:
                            df.loc[iterator] = row
                        except ValueError as e:
                            print(e)
                            error_count = error_count + 1
                            pass  # if error with value probably column mismatch which is down to taking a mesage with no data

                        iterator = iterator + 1
                    completed_features = completed_features + 1
                    user_name = list(message.keys())[0]
                    Logger.logger.print_message(
                        "Added a message from user {} to data frame - {} messages of {} completed".format(user_name,
                                                                                                          completed_features,
                                                                                                          number_of_features),
                        logging_level=1)

                Logger.logger.print_message("Total errors when creating data frame: {}".format(error_count),
                                            logging_level=1)

                # Replace boolean with float
                df.replace({False: 0, True: 1}, inplace=True)

                # Sets ID field
                df.index.name = "ID"
                df.to_csv("{}.csv".format(features_file))

            else:
                df = pandas.read_csv(csv_file)

        return df

    def create_model_info_output_file(self, location_of_output_file = None, training_data_csv_location = None):
        """
        If the model has been loaded or trained this function will create a summary text file with information relating to
        the model.
        :param location_of_output_file: The location to save the output file to.
        :param training_data_csv_location: The location of the training data csv. This is used to retrieve the name of the
        feature columns.
        """

        # Check if model has been created
        if not  self.creation_date:
            Logger.logger.print_message("Model has not been trained, created, or loaded. Cannot output model data in this state.",logging_level=1)
        else:
            Logger.logger.print_message("Creating model info text file")
            output_text = ""

            # Add summary information
            output_text += "Model {}, version {}, created at {} \n".format(self.original_name, self.model_version, self.creation_date)
            output_text += "\nAccuracy: {}\nRecall: {} \nPrecision: {}\nF-Measure: {}\n".format(self.accuracy, self.recall,
                                                                                   self.precision, self.f_measure)

            # Retrieve the header names if available
            if training_data_csv_location:
                with open(training_data_csv_location, "r") as csv_file:
                    reader = csv.reader(csv_file)
                    headers = next(reader)

            # Loop through all feature importance scores
            for iterator in range(len(self.model.feature_importances_)):
                if training_data_csv_location:
                    # Plus one to ignore ID field
                    output_text += "\n{}: {}".format(headers[iterator+1], self.model.feature_importances_[iterator])
                else:
                    output_text += "\nFeature {}: {}".format(iterator,self.model.feature_importances_[iterator])

        # If no name has been set write to outputs folder
        if location_of_output_file:
            file_name = location_of_output_file
        else:
            file_name = os.path.join(self._outputs_folder,"model-output-{}.txt".format(datetime.today().strftime('%Y-%m-%d-%H%M%S')))

        # Write to file
        with open(file_name, "w") as output_file:
            output_file.write(output_text)

    def train_model(self, features_file, force_new_dataset=True, model_location=None):
        """
        Trains the model of the proveded data unless the model file already exists or if the force new dataset flag is True.
        :param features_file: the location of the feature file to be used to train the model
        :param force_new_dataset: If True a new dataset will be created and new model created even if a model already exists.
        :param model_location: the location to save the model file to
        """

        # Sets model location based on default folder location and placeholder name if none was given
        if model_location is None:
            model_location = os.path.join(self._model_folder, "predictor.model")

        # if told to force the creation of a new dataset to train off or the model location does not exist then make a new model
        if force_new_dataset or not os.path.isfile(model_location):

            # Import train_test_split function
            feature_data = self.get_features_as_df(features_file, force_new_dataset)

            # Removes index column
            if "ID" in feature_data.keys():
                feature_data.drop(feature_data.columns[0], axis=1, inplace=True)
            feature_data.reset_index(drop=True, inplace=True)

            y = feature_data[['is_extremist']]  # Labels
            X = feature_data.drop(axis=1, labels=['is_extremist'])  # Features

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 80% training and 20% test

            # Create a Gaussian Classifier
            random_forest = RandomForestClassifier(n_estimators=100, max_depth=50, oob_score=True,
                                         class_weight={0:5,1:1})  # A higher weight for the minority class (is_extreamist)
                                        #TODO change back to 0:1, 1:5
            # Train the model using the training sets y_pred=random_forest.predict(X_test)
            random_forest.fit(X_train, y_train.values.ravel())

            y_pred = random_forest.predict(X_test)

            # Model Accuracy, how often is the classifier correct?
            self.accuracy = metrics.accuracy_score(y_test, y_pred)
            self.recall = metrics.recall_score(y_test, y_pred)
            self.precision = metrics.precision_score(y_test, y_pred)
            self.f_measure = metrics.f1_score(y_test, y_pred)

            Logger.logger.print_message("Accuracy: {}".format(self.accuracy), logging_level=1)
            Logger.logger.print_message("Recall: {}".format(self.recall), logging_level=1)
            Logger.logger.print_message("Precision: {}".format(self.precision), logging_level=1)
            Logger.logger.print_message("F-Measure: {}".format(self.f_measure), logging_level=1)

            self.model = random_forest
            self.original_name = model_location
            self.creation_date = datetime.today().strftime('%Y-%m-%d')

            # write model and accuracy to file to file
            model_data = {"model": self.model,
                          "original_name": self.original_name,
                          "creation_date": self.creation_date,
                          "accuracy": self.accuracy,
                          "recall": self.recall,
                          "precision": self.precision,
                          "f1": self.f_measure,
                          "version": self._FRAMEWORK_VERSION
                          }

            pickle.dump(model_data, open(model_location, "wb"))

        else:
            # Read model and accuracy from file
            saved_file = pickle.load(open(model_location, "rb"))

            self.accuracy = saved_file["accuracy"]
            self.recall = saved_file["recall"]
            self.precision = saved_file["precision"]
            self.f_measure = saved_file["f1"]
            self.model = saved_file["model"]
            self.model_version = saved_file["version"]
            self.original_name = saved_file["original_name"]
            self.creation_date = saved_file["creation_date"]

            # A check to identify if the loaded model is of the same version as the tooling
            if self.model_version is not self._FRAMEWORK_VERSION:
                Logger.logger.print_message("Model provided is of version {}, tooling is of "
                                            "version {}. Using the model may not work as expected."
                                            .format(self.model_version, self._FRAMEWORK_VERSION))