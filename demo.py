import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

model1 = load_model("models/Basic_model.h5")
with open('tokenizers/Basic_tokenizer.pickle', 'rb') as handle:
    token1 = pickle.load(handle)

model2 = load_model("models/LSTM_model.h5")
with open('tokenizers/LSTM_tokenizer.pickle', 'rb') as handle:
    token2 = pickle.load(handle)

st.title("Sarcasm Detection Model")
st.markdown("> From https://github.com/Kalamojo/Some-Context-Please")
st.write("Enter a title with a reply and I'll tell you something interesting:")

input_1 = st.text_input("Title of Post")
input_2 = st.text_input("Comment Text")

padd_max = 100

def predict1(input_1, input_2):
    #print(input_1, input_2)
    to_pred = [[input_1, input_2]]
    to_pred = np.array(to_pred)
    print(to_pred)
    to_pred_temp1 = pad_sequences(token1.texts_to_sequences(to_pred[:, 0]), maxlen=padd_max)
    to_pred_temp2 = pad_sequences(token1.texts_to_sequences(to_pred[:, 1]), maxlen=padd_max)
    to_pred_num = np.column_stack((to_pred_temp1, to_pred_temp2)).reshape(1, 2, padd_max)

    prediction = model1.predict(to_pred_num)
    if prediction[0][0] < 0.5:
        return "### Basic model is ***:blue[" + str(round(prediction[0][0]*100, 2)) + "\%]*** sure that was sarcastic"
    return "### Basic model is ***:red[" + str(round(prediction[0][0]*100, 2)) + "\%]*** sure that was sarcastic"

def predict2(input_1, input_2):
    #print(input_1, input_2)
    to_pred = [[input_1, input_2]]
    to_pred = np.array(to_pred)
    print(to_pred)
    to_pred_temp1 = pad_sequences(token2.texts_to_sequences(to_pred[:, 0]), maxlen=padd_max)
    to_pred_temp2 = pad_sequences(token2.texts_to_sequences(to_pred[:, 1]), maxlen=padd_max)
    to_pred_num = np.column_stack((to_pred_temp1, to_pred_temp2)).reshape(1, 2, padd_max)

    prediction = model2.predict(to_pred_num)
    if prediction[0][0] < 0.5:
        return "### LSTM model is ***:blue[" + str(round(prediction[0][0]*100, 2)) + "\%]*** sure that was sarcastic"
    return "### LSTM model is ***:red[" + str(round(prediction[0][0]*100, 2)) + "\%]*** sure that was sarcastic"

if st.button("Get prediction"):
    st.markdown(predict1(input_1, input_2))
    st.markdown(predict2(input_1, input_2))

st.write("Examples:")
st.markdown("> High crime in Republican cities fueled by guns, inequality")
st.markdown("- Who would have thought that making it easy to access firearms would result in more people shooting each other? Wild!")
st.markdown("- The high crime rate in Dem controlled cities is also fueled by guns. Most of their crime is within demographic segments, so I’m not sure inequality is the issue there.")
st.markdown("- Woah, you mean crime is perpetrated by people who are desperate, poor, hungry, and carrying deadly weapons? What a novel concept!")
st.markdown("- What a grossly misleading title. There is nothing in the article about guns. It says social safety nets reduce poverty, which in turn reduces violent crimes.")
