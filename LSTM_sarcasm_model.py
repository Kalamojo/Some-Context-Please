import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Reshape, Bidirectional, Reshape

padd_max = 100

def filt(line, q):
    ind = line.find(q)
    if ind != -1:
        return line[:line.find(q)]
    return line

sarcastic = []
ind = 0
with open('reddit_generated_sarcastic.txt', 'r', encoding="utf8") as f:
    for line in f:
        line = bytes(line, 'utf-8').decode('utf-8', 'ignore')[:-1]
        if line[:7] == "Title: ":
            sarcastic.append([line[7:]])
        elif line[:9] == "Comment: ":
            line = line[9:]
            line = filt(line, "/s")
            line = filt(line, "#s")
            line = filt(line, "Sarcas")
            line = filt(line, "sarcas")
            sarcastic[ind].append(line)
            ind += 1

sar_len = len(sarcastic)

with open('reddit_wild_sincere.txt', 'r', encoding="utf8") as f:
    serious_org = [name.rstrip() for name in f.read().split('--------------------')][:-1]

serious_list1 = random.sample(serious_org, sar_len)
serious_list = serious_list1.copy()

for i in range(len(serious_list)):
    if serious_list[i][:1] == '\n':
        serious_list[i] = serious_list[i][1:]
    serious_list[i] = serious_list[i].split('\n---\n')
ser_len = len(serious_list)

sarcastic = np.array(sarcastic)
serious = np.array(serious_list)

docs = sarcastic.flatten().tolist() + serious.flatten().tolist()

labels = [1] * sar_len + [0] * ser_len
labels = np.array(labels)

token = Tokenizer()
token.fit_on_texts(docs)

# saving
with open('LSTM_tokenizer.pickle', 'wb') as handle:
    pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

sar_temp1 = pad_sequences(token.texts_to_sequences(sarcastic[:, 0]), maxlen=padd_max)
sar_temp2 = pad_sequences(token.texts_to_sequences(sarcastic[:, 1]), maxlen=padd_max)
sarcastic_num = np.column_stack((sar_temp1, sar_temp2)).reshape(sar_len, 2, padd_max)

ser_temp1 = pad_sequences(token.texts_to_sequences(serious[:, 0]), maxlen=padd_max)
ser_temp2 = pad_sequences(token.texts_to_sequences(serious[:, 1]), maxlen=padd_max)
serious_num = np.column_stack((ser_temp1, ser_temp2)).reshape(ser_len, 2, padd_max)

docs_x = np.concatenate((sarcastic_num, serious_num), axis=0)
#print(docs_x)


x_train, x_test, y_train, y_test = train_test_split(docs_x, labels, test_size=0.1, random_state=42)
#print(x_train)
#print(x_train.shape)

word_size = len(token.word_index) + 1
print("Vocabulary Size:", word_size)

model = Sequential()
model.add(Embedding(word_size, 100, input_shape=(2, padd_max)))
model.add(Reshape((padd_max, 200)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

print("\nSame-distribution test accuracy:", model.evaluate(x_test, y_test)[1])
from sklearn.metrics import classification_report

predict_x=model.predict(x_test)
classes_x=np.round(predict_x)
print(classification_report(y_test, classes_x))

model.save("LSTM_model.h5")

#############
# Testing Set
#############

with open('reddit_wild_sarcastic.txt', 'r', encoding="utf8") as f:
    sarcastic = [filt(name, "/s").rstrip() for name in f.read().split('--------------------')][:-1]
    sarcastic = sarcastic[:2000]

for i in range(len(sarcastic)):
    if sarcastic[i][:1] == '\n':
        sarcastic[i] = sarcastic[i][1:]
    sarcastic[i] = sarcastic[i].split('\n---\n')
sar_len = len(sarcastic)

serious_list2 = set(random.sample(serious_org, sar_len*2))
serious_list2 = serious_list2 - set(serious_list1)
serious_list2 = random.sample(list(serious_list2), sar_len)

for i in range(len(serious_list2)):
    if serious_list2[i][:1] == '\n':
        serious_list2[i] = serious_list2[i][1:]
    serious_list2[i] = serious_list2[i].split('\n---\n')
ser_len = len(serious_list2)


sarcastic = np.array(sarcastic)
serious = np.array(serious_list2)

docs2 = sarcastic.flatten().tolist() + serious.flatten().tolist()

labels = [1] * sar_len + [0] * ser_len
labels = np.array(labels)


sar_temp1 = pad_sequences(token.texts_to_sequences(sarcastic[:, 0]), maxlen=padd_max)
sar_temp2 = pad_sequences(token.texts_to_sequences(sarcastic[:, 1]), maxlen=padd_max)
sarcastic_num = np.column_stack((sar_temp1, sar_temp2)).reshape(sar_len, 2, padd_max)

ser_temp1 = pad_sequences(token.texts_to_sequences(serious[:, 0]), maxlen=padd_max)
ser_temp2 = pad_sequences(token.texts_to_sequences(serious[:, 1]), maxlen=padd_max)
serious_num = np.column_stack((ser_temp1, ser_temp2)).reshape(ser_len, 2, padd_max)

docs_x2 = np.concatenate((sarcastic_num, serious_num), axis=0)
print("In-the-wild test accuracy:", model.evaluate(docs_x2, labels)[1])

predict_x=model.predict(docs_x2)
classes_x=np.round(predict_x)
print(classification_report(labels, classes_x))


import matplotlib.pyplot as plt

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_acc = history.history['accuracy']
y_vacc = history.history['val_accuracy']

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.arange(len(y_vloss)), y_vloss, marker='.', c='red')
ax1.plot(np.arange(len(y_loss)), y_loss, marker='.', c='blue')
ax1.grid()
plt.setp(ax1, xlabel='epoch', ylabel='loss')

ax2.plot(np.arange(len(y_vacc)), y_vacc, marker='.', c='red')
ax2.plot(np.arange(len(y_acc)), y_acc, marker='.', c='blue')
ax2.grid()

plt.setp(ax2, xlabel='epoch', ylabel='accuracy')

plt.show()
