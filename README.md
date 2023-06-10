# Some Context, Please?

Sarcasm Detection Model

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

> Word cloud from r/politics comments


Politics Sarcastic         |  Politics Sincere
:-------------------------:|:-------------------------:
![alt text](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/politics_sarcastic.png?raw=true) | ![alt text](https://github.com/Kalamojo/Some-Context-Please/blob/main/images/politics_sincere.png?raw=true)

To help combat this reality, we provide our model with that much needed context in the form of the title a comment is replying to. Furthermore, we limit predictions to one subject area (politics) to ensure that subjects and phrases specific to that area are known by the model.
With this setup, our hope is that our model will be able to vary its confidence in sarcasm depending on the title a comment is made on. Serious titles will likely have more serious comments, and laughable titles will most likely be replied to with more sarcastic comments. This makes our data much closer to real-world situations, as opposed to expecting a model to determine sarcasm on a sentence alone.

