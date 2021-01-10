import json
import os
import pickle

import pandas
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class random_forest():
    """
    A class used for creating a random forest binary classifier.
    """

    model = None
    accuracy = None

    outputs_folder = None
    model_folder = None

    def __init__(self, outputs_folder="Pinpoint/outputs", model_folder="Pinpoint/model"):
        """
        Constructor

        The random_forest() class can be initialised with outputs_folder() and model_folder(). The outputs folder is
        where output files are stored and the model folder is where the model will be created if not overwritten.
        """
        self.outputs_folder = outputs_folder
        self.model_folder = model_folder

    def get_features_as_df(self, features_file, force_new_dataset=True):
        """
        Reads a JSON file file and converts to a Pandas dataframe that can be used to train and test the classifier.
        :param features_file: the location of the JSON features file to convert to a dataframe
        :param force_new_dataset: if true a new CSV file will be created even if one already exists.
        :return: a Pandas dataframe with the features.
        """

        column_names = ["post_freq", "follower_freq", "cap_freq", "violent_freq",
                        "centrality", 'is_extremist', "clout", "analytic", "tone", "authentic"]

        # Add the two hundred vectors columns
        for iterator in range(1, 201):
            column_names.append("message_vector_{}".format(iterator))

        # Creates the columns for the data frame
        df = pd.DataFrame(
            columns=column_names)

        with open(features_file) as json_features_file:
            csv_file = "{}.csv".format(features_file)

            if force_new_dataset or not os.path.isfile(csv_file):
                features = json.load(json_features_file)

                number_of_features = len(features)

                completed_features = 0
                iterator = 0
                error_count = 0
                for message in features:
                    # should only be one user per entry
                    for user_id in message:
                        feature_data = message[user_id]
                        # ID is not included as it's hexidecimal and not float
                        post_freq = feature_data['post_freq']
                        follower_freq = feature_data['follower_freq']
                        cap_freq = feature_data['cap_freq']
                        violent_freq = feature_data['violent_freq']
                        message_vector = feature_data['message_vector']
                        centrality = feature_data['centrality']
                        is_extremist = feature_data['is_extremist']
                        clout = feature_data['clout']
                        analytic = feature_data['analytic']
                        tone = feature_data['tone']
                        authentic = feature_data['authentic']

                        row = [post_freq, follower_freq, cap_freq, violent_freq,
                               centrality, is_extremist, clout, analytic, tone, authentic] + message_vector
                        try:
                            df.loc[iterator] = row
                        except ValueError as e:
                            error_count = error_count + 1
                            pass  # if error with value probably column mismatch which is down to taking a mesage with no data

                        iterator = iterator + 1
                    completed_features = completed_features + 1
                    user_name = list(message.keys())[0]
                    print("Added a message from user {} to data frame - {} messages of {} completed".format(user_name,
                                                                                                            completed_features,
                                                                                                            number_of_features))

                print("Total errors when creating data frame: {}".format(error_count))

                # Replace boolean with float
                df.replace({False: 0, True: 1}, inplace=True)

                # Sets ID field
                df.index.name = "ID"
                df.to_csv("{}.csv".format(features_file))

            else:
                df = pandas.read_csv(csv_file)

        return df

    def train_model(self, features_file, force_new_dataset=True, model_location=None):
        """
        Trains the model of the proveded data unless the model file already exists or if the force new dataset flag is True.
        :param features_file: the location of the feature file to be used to train the model
        :param force_new_dataset: If True a new dataset will be created and new model created even if a model already exists.
        :param model_location: the location to save the model file to
        """

        # Sets model location based on default folder location and placeholder name if none was given
        if model_location is None:
            model_location = os.path.join(self.model_folder, "predictor.model")

        # if told to force the creation of a new dataset to train off or the model location does not exist then make a new model
        if force_new_dataset or not os.path.isfile(model_location):

            # Import train_test_split function
            feature_data = self.get_features_as_df(features_file, force_new_dataset)

            # Removes index column
            feature_data.drop(feature_data.columns[0], axis=1, inplace=True)
            feature_data.reset_index(drop=True, inplace=True)

            y = feature_data[['is_extremist']]  # Labels
            X = feature_data.drop(axis=1, labels=['is_extremist'])  # Features

            # Split dataset into training set and test set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

            # Create a Gaussian Classifier
            clf = RandomForestClassifier(n_estimators=100)

            # Train the model using the training sets y_pred=clf.predict(X_test)
            clf.fit(X_train, y_train.values.ravel())

            y_pred = clf.predict(X_test)

            # Import scikit-learn metrics module for accuracy calculation

            # Model Accuracy, how often is the classifier correct?
            self.accuracy = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy:", self.accuracy)

            self.model = clf

            # write model and accuracy to file to file
            pickle.dump({"model": self.model, "accuracy": self.accuracy}, open(model_location, "wb"))

        else:
            # Read model and accuracy from file
            saved_file = pickle.load(open(model_location, "rb"))
            self.accuracy = saved_file["accuracy"]
            self.model = saved_file["model"]
