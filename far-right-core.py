"""
Example of training a model using this package.
"""

from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction(violent_words_dataset_location=r"datasets/swears",
                               baseline_training_dataset_location=r"datasets/far-right/LIWC2015 Results (Storm_Front_Posts).csv")

extractor.dump_training_data_features(
    feature_file_path_to_save_to=r"outputs/training_features.json",
    extremist_data_location=r"datasets/far-right/LIWC2015 Results (extreamist-messages.csv).csv",
    baseline_data_location=r"datasets/far-right/LIWC2015 Results (non-extreamist-messages.csv).csv",
    force_new_dataset=False)

# Trains a model off the features file created in the previous stage
model = random_forest()

model.RADICAL_LANGUAGE_ENABLED = True
model.BEHAVIOURAL_FEATURES_ENABLED = False
model.PSYCHOLOGICAL_SIGNALS_ENABLED = False

model.train_model(features_file= r"outputs/training_features.json",
                  force_new_dataset=True, model_location=r"outputs/far-right-radical-language.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-radical-language-output.txt",
                                    training_data_csv_location=r"outputs/training_features.json.csv")


model.RADICAL_LANGUAGE_ENABLED = False
model.BEHAVIOURAL_FEATURES_ENABLED = True
model.PSYCHOLOGICAL_SIGNALS_ENABLED = False

model.train_model(features_file= r"outputs/training_features.json",
                  force_new_dataset=True, model_location=r"outputs/far-right-behavioural.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-behavioural-output.txt",
                                    training_data_csv_location=r"outputs/training_features.json.csv")


model.RADICAL_LANGUAGE_ENABLED = False
model.BEHAVIOURAL_FEATURES_ENABLED = False
model.PSYCHOLOGICAL_SIGNALS_ENABLED = True

model.train_model(features_file= r"outputs/training_features.json",
                  force_new_dataset=True, model_location=r"outputs/far-right-psychological.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-psychological-output.txt",
                                    training_data_csv_location=r"outputs/training_features.json.csv")


model.RADICAL_LANGUAGE_ENABLED = True
model.BEHAVIOURAL_FEATURES_ENABLED = True
model.PSYCHOLOGICAL_SIGNALS_ENABLED = True

model.train_model(features_file= r"outputs/training_features.json",
                  force_new_dataset=True, model_location=r"outputs/far-right-baseline.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-baseline-output.txt",
                                    training_data_csv_location=r"outputs/training_features.json.csv")

print("Finished")