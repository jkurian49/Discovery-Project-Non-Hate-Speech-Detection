# Discovery-Project-Non-Hate-Speech-Detection

The goal of this project is to supplement a hate speech dataset with edge cases, text that could be misclassified as hateful or offensive when taken out context, in order to alleviate biases in datasets used to train neural hate speech classifers. Chess and the Civilization franchise were chosen as the source material since these games have aspects that could be misunderstood as racist and offensive, even though within the context of the game they are perfectly reasonable. 

## Dataset Creation

The majority of data is gathered from YouTube transcripts. [Get_Youtube_Transcripts.ipynb](Get_Youtube_Transcripts.ipynb) accepts a file of YouTube video IDs and aggregates all the transcripts into a readable format. Phrases and sentences containing words that might elicit a strong response from the classifer (e.g. black,white,attack,hate) were manually pulled from these transcripts and added to the dataset.

See [chess_dataset.csv](chess_dataset.csv) and [civ_dataset.csv](civ_dataset.csv) for the final datasets.

## Model Evaluation

We adapted an existing testing framework from https://github.com/aman-saha/hate-speech-detection to extract features from our text and input them into the XGBoost classification algorithm. Please see [hate-speech-detection](hate-speech-detection) for further details on the model running process. The dataset of edge cases was appended onto the Hate Speech and Offensive Language (HSOL) dataset of ~25k tweets to train the model. This model correctly classified all of the edge cases in the test set. 

