"""
Example of training a model using this package.
"""
from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction(violent_words_dataset_location=r"C:\Projects\Pinpoint\datasets\swears",
                               baseline_training_dataset_location=r"C:\Projects\Pinpoint\datasets\religious-texts\ISIS Religious Texts v1.csv")

extractor.dump_training_data_features(
    feature_file_path_to_save_to=r"C:\Projects\Pinpoint\outputs\training_features.json",
    extremist_data_location=r"C:\Projects\Pinpoint\datasets\extreamist\tweets.csv",
    baseline_data_location=r"C:\Projects\Pinpoint\datasets\baseline tweets\tweet_dataset.csv",
    force_new_dataset=False)

# Trains a model off the features file created in the previous stage
model = random_forest(outputs_folder="outputs", model_folder="outputs")
model.train_model(r"C:\Projects\Pinpoint\outputs\training_features.json",
                  False)  # , model_location=r"Pinpoint/model/my.model"
