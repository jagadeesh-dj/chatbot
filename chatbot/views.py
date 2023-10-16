from django.shortcuts import render
from django.http import JsonResponse
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import tensorflow as tf
import speech_recognition as sr


model = load_model('model.h5')#deeplearning model
intents = json.loads(open('data.json').read())#dataset
words = pickle.load(open('texts.pkl', 'rb'))#patterns
classes = pickle.load(open('labels.pkl', 'rb'))#tags

# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence): #this function user to preprocess the user input as word-tokenize and lemmatize
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence) #pass argument userinput to clear_up_function
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False) #passing the argument userinput and patterns
    res = model.predict(np.array([p]))[0] #predict the deeplearning model
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list



def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    response = getResponse(ints, intents)
    print(response)
    return response

def res(request,msg):
    response=chatbot_response(msg)
    return JsonResponse({"res":response})

def bot(request):
    if request.method=='POST':
        message=request.POST.get('message')
    return render(request,'index.html')


def speech(request):

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Record audio from a microphone
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

    # Use a speech-to-text API to convert the audio to text
    try:
        text = recognizer.recognize_google(audio)  # You can also use other APIs like recognize_bing or recognize_wit
        print("You said: " + text)
        if text!=None:
            response=chatbot_response(text)
            return JsonResponse({"res":response,"text":text})
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Error making the request; {0}".format(e))

















































