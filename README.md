# Pinpoint
Pinpoint is a suite of functionality for building a Gaussian classifier for the identification of extremist content. This tooling builds off the methodology in the paper [Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith](https://arxiv.org/pdf/1905.08067.pdf)'
.

## Installation

```shell
python -m pip install git+https://github.com/user1342/Pinpoint.git
```

## Datasets

### Baseline dataset (i.e. radical magazine or forum)
A CSV file should be provided as the ```baseline_training_dataset_location``` paramiter. The following datasets should be marked up using [LIWC](http://liwc.wpengine.com/).

- Extremist magazine https://www.kaggle.com/fifthtribe/isis-religious-texts

### Message data sets (i.e. radical tweets)
CSV files should be proivded as the ```extremist_data_location``` and ```baseline_data_location``` paramiters respectively. The following datasets should be marked up using [LIWC](http://liwc.wpengine.com/).

- Counterpoise Tweets https://www.kaggle.com/activegalaxy/isis-related-tweets/home?select=AboutIsis.csv
- Extremist tweets https://www.kaggle.com/fifthtribe/how-isis-uses-twitter/data
- Baseline Tweets https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data

### Violent, extreamist, or curse word datasets
Directory provided as the ```violent_words_dataset_location``` paramiter.

- Violent / Swear words https://www.kaggle.com/highflyingbird/swear-words
- Violent / Swear words https://www.kaggle.com/cfiszter/swear-word-list

## Example Usage

```python 
from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction(violent_words_dataset_location=r"datasets/swears",
                               baseline_training_dataset_location=r"datasets/far-right/LIWC2015 Results (Storm_Front_Posts).csv")

extractor.MAX_RECORD_SIZE = 250000

extractor.dump_training_data_features(
    feature_file_path_to_save_to=r"outputs/training_features.json",
    extremist_data_location=r"datasets/far-right/LIWC2015 Results (extreamist-messages.csv).csv",
    baseline_data_location=r"datasets/far-right/LIWC2015 Results (non-extreamist-messages.csv).csv")

# Trains a model off the features file created in the previous stage
model = random_forest()

model.RADICAL_LANGUAGE_ENABLED = True
model.BEHAVIOURAL_FEATURES_ENABLED = True
model.PSYCHOLOGICAL_SIGNALS_ENABLED = True

model.train_model(features_file= r"outputs/training_features.json",
                  force_new_dataset=True, model_location=r"outputs/far-right-baseline.model")  # , model_location=r"Pinpoint/model/my.model"

model.create_model_info_output_file(location_of_output_file="outputs/far-right-baseline-output.txt",
                                    training_data_csv_location=r"outputs/training_features.json.csv")
  ```
