"""
Entry point to aggregating data
"""

from FeatureExtraction import *

# Create a json file with all of the tweets and features, including: extremist, counterpoise, and baseline
feature_extraction().dump_json_of_features("features.json")