"""
Example of training a model using this package.
"""
import random
from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction()
extractor.dump_training_data_features(r"Pinpoint\outputs\training_features.json",
                                                 extremist_data_location= r"Pinpoint/data-sets/liwc-datasets/EXTREEME-LIWC.csv",
                                                 counterpoise_data_location=r"Pinpoint/data-sets/liwc-datasets/COUNTER-LIWC.csv",
                                                 baseline_data_location=r"Pinpoint/data-sets/liwc-datasets/BASELINE-2-LIWC.csv",
                                                 force_new_dataset= False)

# Trains a model off the features file created in the previous stage
model = random_forest()
model.train_model(r"Pinpoint\outputs\training_features.json", False)#, model_location=r"Pinpoint/model/my.model"

# Additional testing stage (Not necessary)

# Creates a new feature set off another dataset known to be extremist
extractor.dump_features_for_list_of_datasets(r"Pinpoint\outputs\testing_features.json",
                                             list_of_dataset_locations=
                                             [r"Pinpoint\data-sets\liwc-datasets\SYMMPATHISER-LIWC.csv"],
                                             force_new_dataset=False)

dict_of_results = {}

# Get list of features and randomise
list_of_features = extractor.completed_tweet_user_features
random.shuffle(list_of_features)

for entry in list_of_features:
    # Loops through the data in that tweet (Should only be one entry per tweet).
    for user_id in entry:
        post_freq = entry[user_id]['post_freq']
        follower_freq = entry[user_id]['follower_freq']
        cap_freq = entry[user_id]['cap_freq']
        violent_freq = entry[user_id]['violent_freq']
        message_vector = entry[user_id]['message_vector']
        centrality = entry[user_id]['centrality']
        clout = entry[user_id]['clout']
        analytic = entry[user_id]['analytic']
        tone = entry[user_id]['tone']
        authentic = entry[user_id]['authentic']

        list_to_predict = [post_freq, follower_freq, cap_freq, violent_freq,centrality,clout,analytic,tone,authentic]+message_vector

        prediction = model.model.predict([list_to_predict])[0]

        if prediction not in dict_of_results:
            dict_of_results[prediction] = 1
        else:
            dict_of_results[prediction] = dict_of_results[prediction] + 1

        print("Prediction on message from {}. Result: {}".format(user_id, prediction))

# Use values to identify the percentage of tweets identified as extremist.
print("Prediction results: {}".format(dict_of_results))