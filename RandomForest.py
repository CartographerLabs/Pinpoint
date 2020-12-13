import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

class random_forest():

    def get_features_as_df(self, features_file):
        df = pd.DataFrame(
            columns=['user_id', "post_freq", "follower_freq", "cap_freq", "violent_freq", "message_vector",
                     "centrality"])

        with open(features_file) as json_features_file:
            features = json.load(json_features_file)

            for entry in features:
                # should only be one user per entry
                for user in entry:
                    user_id = user
                    user_features = features[user_id]
                    post_freq = user_features['post_freq']
                    follower_freq = user_features['follower_freq']
                    cap_freq = user_features['cap_freq']
                    violent_freq = user_features['violent_freq']
                    message_vector = user_features['message_vector']
                    centrality = user_features['centrality']

                    df.append([user_id, post_freq, follower_freq, cap_freq, violent_freq, message_vector, centrality])

        return df

    def train_model(self):
        # Import train_test_split function
        feature_data = self.get_features_as_df("features.json")

        X = feature_data[['post_freq', 'follower_freq', 'cap_freq', 'violent_freq']]  # Features
        y = feature_data['is_extremist']  # Labels

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

        # Create a Gaussian Classifier
        clf = RandomForestClassifier(n_estimators=100)

        # Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        #clf.predict([[3, 5, 4, 2]])