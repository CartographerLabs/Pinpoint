"""
Example of training a model using this package.
"""

from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction(violent_words_dataset_location=r"\Pinpoint\datasets\swears",
                               baseline_training_dataset_location=r"\Pinpoint\datasets\far-right\LIWC2015 Results (Storm_Front_Posts).csv")

extractor.dump_training_data_features(
    feature_file_path_to_save_to=r"\Pinpoint\outputs\training_features.json",
    extremist_data_location=r"\Pinpoint\datasets\far-right\LIWC2015 Results (extreamist-messages.csv).csv",
    baseline_data_location=r"\Pinpoint\datasets\baseline tweets\tweet_dataset.csv",
    force_new_dataset=True)

# Trains a model off the features file created in the previous stage
model = random_forest()

model.RADICAL_LANGUAGE_ENABLED = True
model.BEHAVIOURAL_FEATURES_ENABLED = True
model.PSYCHOLOGICAL_SIGNALS_ENABLED = True

model.train_model(features_file= r"\Pinpoint\outputs\training_features-tweeter-baseline.json",
                  force_new_dataset=False, model_location=r"outputs\far-right-baseline-twitter-baseline.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-baseline-output-twitter-baseline.txt",
                                    training_data_csv_location=r"\Pinpoint\outputs\training_features-tweeter-baseline.json.csv")


print("Finished")