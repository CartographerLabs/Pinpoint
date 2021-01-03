# Pinpoint 
Pinpoint is a suite of Python tooling created to replicate the extremism classification model created for research paper 'Understanding the [Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith](https://arxiv.org/pdf/1905.08067.pdf)'.

## Installation 
```shell
python -m pip install git+https://github.com/user1342/Pinpoint.git
```
## Setup 
Create the following folders in the root of the Pinpoint module: ```model```, ```outputs```, ```data-sets```, and ```violent_or_curse_word_datasets```.
Inside of the datasets folder add an extremist, counterpoise, and baseline dataset. 

### /data-sets 
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
