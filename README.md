# Pinpoint
Pinpoint is a suite of functionality for building a Gaussian classifier for the identification of extremist content. This tooling builds off the methodology in the paper [Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith](https://arxiv.org/pdf/1905.08067.pdf)'
.

## Installation

```shell
python -m pip install git+https://github.com/user1342/Pinpoint.git
```

## Setup

The ```feature_extraction()``` class can be initialised with ```violent_words_dataset_location```
, ```tf_idf_training_dataset_location```, and ```outputs_location``` locations. All files in
the ```violent_words_dataset_location``` will be read (one line at a time) and added to the corpus of violent and swear
words. The csv file at ```baseline_training_dataset_location``` should be marked up with LIWC scores. The ```outputs_location``` folder is where files created by the tooling are stored.

- Extremist magazine https://www.kaggle.com/fifthtribe/isis-religious-texts

The ```random_forest()``` class can be initialised with ```outputs_folder()``` and ```model_folder()```. The outputs
folder is where output files are stored and the model folder is where the model will be created if not overwritten.

### Message data sets (i.e. radical tweets)

The following datasets should be marked up using [LIWC](http://liwc.wpengine.com/).

- Counterpoise Tweets https://www.kaggle.com/activegalaxy/isis-related-tweets/home?select=AboutIsis.csv
- Extremist tweets https://www.kaggle.com/fifthtribe/how-isis-uses-twitter/data
- Baseline Tweets https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data

### Violent or curse word datasets

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
model = random_forest()
model.train_model(features_file= r"C:\Projects\Pinpoint\outputs\training_features.json",
                  force_new_dataset=False)  
  ```
