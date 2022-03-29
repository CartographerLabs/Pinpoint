| :warning: This repository is based on PhD research that seeks to identify radicalisation on online platforms. Due to this; text, themes, and content relating to far-right extremism are present in this repository. Please continue with care. :warning:   <br />  <br />  [Samaritans](http://www.samaritans.org) - Call 116 123 \| [ACT Early](https://www.act.campaign.gov.uk) \| [actearly.uk](https://www.actearly.uk) \| Prevent advice line 0800 011 3764|
| --- |

<p align="center">
    <img width=100% src="Pinpoint.png">
  </a>
</p>
<p align="center"> 📍 Pinpoint is a suite of functionality for building and using a binary classifier for the identification of extremist content. 💻</p>

# Pinpoint
Pinpoint is a suite of functionality for building a Gaussian classifier for the identification of extremist content. This tooling builds off the methodology in the paper [Radical Mind: Identifying Signals to Detect Extremist Content on Twitter by Mariam Nouh, Jason R.C. Nurse, and Michael Goldsmith](https://arxiv.org/pdf/1905.08067.pdf)'
.

## Installation

```shell
python -m pip install git+https://github.com/user1342/Pinpoint.git
```

## Datasets

### Parler dataset
A dataset was acquired from [A Large Open Dataset from the Parler Social Network](https://zenodo.org/record/4442460). This dataset was further broken into two separate datasets using the [Log-Likelihood tooling](https://github.com/CartographerLabs/Parler-Toolbox) from the Parler Toolbox repository. For this, 100 posts in the dataset were manually marked as either violent extremist or non-extremist, and using the tooling a list of the top 30 keywords relating to violent-far-right extremism were identified. A subsection of these can be seen below:
- genocidal
- fire
- destroyers
- democraticnazi
- fucker
- tribunals
- invoke
- squad
- punch
- tyrannical

After these violent-extremist words were aggregated the dataset was split with text posts containing the keywords being marked as violent-far-right-extremist and those without marked as a baseline. After this text posts were converted to CSV and marked up with the [LIWC Text Analysis Engine](https://www.liwc.app/). 

### Stormfront dataset
The second dataset, used for developing a known radical corpus, was extracted from [Hate speech dataset from a white supremacist forum](https://github.com/Vicomtech/hate-speech-dataset) and converted to CSV format.

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
## Outputs
Once trained and a model created it will be pickled and saved as a re-loadable file in the tooling’s ```output``` directory for future use. In addition to this a text file is also created detailing the specifications and related accuracy scores of the created model - examples of these have been provided in the provided folder. 
