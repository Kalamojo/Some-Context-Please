# Some Context, Please?

Sarcasm Detection Model

> The Demo of the 2D Basic and LSTM models can be found at https://some-context-please.streamlit.app/

## Background

### Why Detect Sarcasm?

- Sentiment analysis is an effective method for turning text into meaningful feedback that can be addressed without manual review.
- While accurate sentiment analysis tools exist, they are often rendered useless when predicting on sarcastic text.

### Difficulty

Sarcasm detection is a difficult problem that requires 3 key elements:

- The words that make up a statement
- The tone of voice in which the statement was made
- The context prompting the statement

## Purpose and Hypothesis

As a result of the nature of sarcastic text, identical sentences made in different situations can have wildly different meanings, and determining if a sentence is sarcastic is a challenging task for even humans at times.

> Word clouds from different subbreddits

r/politics Sarcastic         |  r/politics Sincere
:---------------------------:|:---------------------------:
![r/politics sarcastic comments word cloud](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/politics_sarcastic.png?raw=true) | ![r/politics sincer comments word cloud](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/politics_sincere.png?raw=true)

r/ProgrammerHumor Sarcastic     |  r/ProgrammerHumor Sincere
:------------------------------:|:------------------------------:
![r/ProgrammerHumor sarcastic comments word cloud](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/ProgrammerHumor_sarcastic.png?raw=true) | ![r/ProgrammerHumor sincere comments word cloud](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/ProgrammerHumor_sincere.png?raw=true)

To help combat this reality, we provide our model with that much needed context in the form of the title a comment is replying to. Furthermore, we limit predictions to one subject area (politics) to ensure that subjects and phrases specific to that area are known by the model.
With this setup, our hope is that our model will be able to vary its confidence in sarcasm depending on the title a comment is made on. Serious titles will likely have more serious comments, and laughable titles will most likely be replied to with more sarcastic comments. This makes our data much closer to real-world situations, as opposed to expecting a model to determine sarcasm on a sentence alone.

## Materials and Methods

### Data Collection

To limit our text to one context area, and to provide context in the form of a title, we decided on using text from Reddit. We made use of the Python Reddit API Wrapper (PRAW) and restricted our search to r/politics. Sarcastic text was determined by whether a comment contained the ‘/s’ tag or not. We made the assumption that the majority of text on the subreddit was not sarcastic, and that each comment containing ‘/s’ was.
Because of the limitations of the PRAW, only 100 posts were able to be scraped at a time, containing about 2,000 sarcastic comments. To maximize training data (as well as isolate in-the-wild testing data), we made use of the
ChatGPT API to generate sarcastic titles and comments with this prompt: 

" Please generate X examples of sarcastic text, along with the title of the post they commented on, like that found on Reddit in the politics subreddit (r/politics). Do this in the format:
Title: ...
Comment: … "

By looping these requests, we extracted another 2,000 artificial sarcastic comments for training.

### Model and Implementation

We tried 3 different inputs for training our Sarcasm detection models:

1. 1D concatenation of Title + comment
2. 2D input of Title, comment
3. Comment alone

We also compared the results of a basic Embedding vs. LSTM approach to see if the sequence recognizing
capabilities of the LSTM truly provided an advantage in detecting sarcasm.

## Results

Configuration Model | Type | Train Accuracy | Validation Accuracy | Test Accuracy | In-the-wild Test Accuracy
:------------------:|:-----:|:-------------:|:-------------------:|:-------------:|:------------------------:
1D Title + Comment | Basic | 1 | 0.98 | 0.977 | 0.669
1D Title + Comment | LSTM | 0.988 | 0.997 | 0.962 | 0.766
2D Title, Comment | Basic | 1 | 0.98 | 0.967 | 0.613
2D Title, Comment | LSTM | 0.99 | 0.932 | 0.903 | 0.57
Just Comment | Basic | 0.999 | 0.909 | 0.995 | 0.561
Just Comment | LSTM | 0.994 | 0.926 | 0.921 | 0.565

2D Basic Model | 2D LSTM Model
:-------------:|:------------:
![Training and validation loss/accuracy for basic model](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/basic_model_training.png?raw=true) | ![Training and validation loss/accuracy for LSTM model](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/LSTM_model_training.png?raw=true)

## Areas of Improvement

### Lack of Data

With only 2k sentences from each class trained on, great improvements could arise from just an increase in data.

### Overfitting

Modifications need to be made to the model to prevent overfitting while producing consistent results.

## Acknowledgments

- [PRAW](https://praw.readthedocs.io/en/stable/)
- [Sarcasm Detection : RNN-LSTM | Kaggle](https://www.kaggle.com/code/tanumoynandy/sarcasm-detection-rnn-lstm)
- [Datasets used for iSarcasmEval shared-task (Task 6 at SemEval 2022)](https://github.com/iabufarha/iSarcasmEval)