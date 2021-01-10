# Pinpoint 
Pinpoint is a suite of Python tooling created to replicate the extremism classification model created for research paper 'Understanding the [Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith](https://arxiv.org/pdf/1905.08067.pdf)'.

## Installation 
```shell
python -m pip install git+https://github.com/user1342/Pinpoint.git
```
## Setup 
The ```feature_extraction()``` class can be initialised with ```violent_words_dataset_location```, ```tf_idf_training_dataset_location```, and ```outputs_location``` locations. All files in the ```violent_words_dataset_location``` will be read (one line at a time) and added to the corpus of violent and swear words. The csv file at ```tf_idf_training_dataset_location``` should have a column at position 5 which contains data to train the TF IDF model against. The ```outputs_location``` folder is where files created by the tooling are stored. 

The ```random_forest()``` class can be initialised with ```outputs_folder()``` and ```model_folder()```. The outputs folder is where output files are stored and the model folder is where the model will be created if not overwritten. 

### /data-sets 
The following datasets should be marked up using [LIWC](http://liwc.wpengine.com/) and be in the following column format: username(0), message(1), analytic LIWC score(2), clout LIWC score(3), authentic LIWC score(4), tone LIWC score(5). 
- Extremist magazine https://www.kaggle.com/fifthtribe/isis-religious-texts
- Counterpoise Tweets https://www.kaggle.com/activegalaxy/isis-related-tweets/home?select=AboutIsis.csv
- Extremist tweets https://www.kaggle.com/fifthtribe/how-isis-uses-twitter/data
- Baseline Tweets https://www.kaggle.com/maxjon/complete-tweet-sentiment-extraction-data

### /violent_or_curse_word_datasets
- Violent / Swear words https://www.kaggle.com/highflyingbird/swear-words
- Violent / Swear words https://www.kaggle.com/cfiszter/swear-word-list

## Example Usage
```python 
import random
from Pinpoint.FeatureExtraction import *
from Pinpoint.RandomForest import *

# Performs feature extraction from the provided Extremist, Counterpoise, and Baseline datasets.
extractor = feature_extraction()
extractor.dump_training_data_features(r"Pinpoint\outputs\training_features.json",
                                                 extremist_data_location= r"Pinpoint\data-sets\liwc-datasets\EXTREEME-LIWC.csv",
                                                 counterpoise_data_location=r"Pinpoint\data-sets\liwc-datasets\COUNTER-LIWC.csv",
                                                 baseline_data_location=r"Pinpoint\data-sets\liwc-datasets\BASELINE-2-LIWC.csv",
                                                 force_new_dataset= False)

# Trains a model off the features file created in the previous stage
model = random_forest()
model.train_model(r"Pinpoint\outputs\training_features.json", False)
print(model.accuracy)
```
