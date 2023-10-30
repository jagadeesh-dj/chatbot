import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.legacy import SGD
import random

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json', encoding="utf-8").read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))#set used to removing the repeated words from the tokenized words list
classes = sorted(list(set(classes)))#there also remove the repeated words from the classes list

pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)#classes length is 11
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]#0 represent the words and 1 represent the tages in the document list
    # print(pattern_words) #the first doc value is hi word
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]#here hi convert to lowercase and lemmatize(plural to singular) 
    print(pattern_words)

    # create our bag of words array with 1, if word match found in current pattern
    for w in words: #here the word represent the sorted list of words-> if the pattern_words(hi) is present in the list of sorted words append bags=1 else 0; 
        bag.append(1) if w in pattern_words else bag.append(0)
    # print(words)
    print(bag)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    # print(output_row)

    training.append([bag, output_row])
    # print(training[0:5])
# shuffle our features and turn into np.array
random.shuffle(training) #here combain the all the training datas into single random list 
# print(training)
training = np.array(training,dtype=object) #here we conver the combained training datas into rows and columns with the help of numpy array->shape ==(50,2)
# print(training.shape)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])# here represent the patterns
train_y = list(training[:, 1])# here y represent the classes(tags)
print("Training data created")
print(len(train_x[0]),len(train_y[0]))
from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test=train_test_split(train_x,train_y, train_size=0.2 ,random_state=1) 
# X_train, X_val, Y_train, Y_val=train_test_split(X_train,Y_train,test_size=0.2 ,random_state=1)
# print(X_train, X_test, Y_train, Y_test)

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.summary()

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


#fitting and saving the mode
hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
# import matplotlib.pyplot as plt
# training_acc=hist.history['accuracy']
# epochs=range(200)
# plt.plot(epochs,training_acc,'r',label='training_accuracy')
# plt.title('training accuracy vs epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Training Accuracy')
# plt.legend()
# plt.show()
# training_loss=hist.history['loss']
# epochs=range(200)
# plt.plot(epochs,training_loss,'r',label='training_loss')
# plt.title('training loss vs epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Training loss')
# plt.legend()
# plt.show()
# training_loss=hist.history['loss']
# training_accuracy=hist.history['accuracy']
# epochs=range(200)
# plt.plot(epochs,training_loss,'r',label='training_loss')
# plt.plot(epochs,training_accuracy,'b',label='training_accuracy')
# plt.title('training loss vs training acc')
# plt.xlabel('Epochs')
# plt.ylabel('Training loss')
# plt.legend()
# plt.show()
# print(np.array(train_y))
# print(model.predict(np.array(train_x)))

print(model.evaluate(np.array(train_x),np.array(train_y)))
model.save('model.h5', hist)