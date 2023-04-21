import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
from flask import Flask, render_template, request, jsonify, session
import secrets

model = load_model('./data/model.h5')
plantmodel = joblib.load('./data/plantmodel.joblib')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('./data/data.json').read())
words = pickle.load(open('./data/texts.pkl','rb'))
classes = pickle.load(open('./data/labels.pkl','rb'))
plantpklmodel = pickle.load(open('./data/plant.pkl', 'rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = secrets.token_hex(16)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if(userText == '1'):
        session["plantrealtedtopic"] = True
        #plantrealtedtopic = 1
        return "Please specify the symptoms of the plant"
    if(session.get("plantrealtedtopic")):

        vectorizer = joblib.load('./data/vectorizer1.joblib')

        vocab = vectorizer.vocabulary_
        vectorizer = CountVectorizer(stop_words='english',vocabulary=vocab)

        new_symptom_vec = vectorizer.transform([userText])
        prediction = plantmodel.predict(new_symptom_vec)[0]
        session["plantrealtedtopic"] = False
    # return chatbot_response(userText)
        return "Your plant must be infected with disease: " + prediction
    else:
        return chatbot_response(userText)

if __name__ == "__main__":
    app.run()