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
extractor = feature_extraction(violent_words_dataset_location=r"C:\Projects\Pinpoint\datasets\swears",
                               baseline_training_dataset_location=r"C:\Projects\Pinpoint\datasets\religious-texts\ISIS Religious Texts v1.csv")

extractor.dump_training_data_features(
    feature_file_path_to_save_to=r"C:\Projects\Pinpoint\outputs\training_features.json",
    extremist_data_location=r"C:\Projects\Pinpoint\datasets\extreamist\tweets.csv",
    baseline_data_location=r"C:\Projects\Pinpoint\datasets\baseline tweets\tweet_dataset.csv",
    force_new_dataset=False)

# Trains a model off the features file created in the previous stage
random_forest().train_model(features_file= r"C:\Projects\Pinpoint\outputs\training_features.json",
                  force_new_dataset=False)  
  ```
