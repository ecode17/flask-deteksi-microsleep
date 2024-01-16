from flask import Flask, Response, jsonify, redirect, url_for,render_template,request
import datetime
from flask_cors import CORS
import random
import numpy as np
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pymysql
from drowsiness import generate_frames

lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)

# Database configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}

@app.route("/",methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def submit():
    return render_template('new.html')

@app.route("/driver", methods=['GET', 'POST'])
def driver():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res =res1.replace("{n}",name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

#SENTIMEN

# Function to insert data into MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Validate and convert date
            try:
                tanggal = datetime.datetime.strptime(data['tanggal'], '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Invalid date format. Use YYYY-MM-DD.")

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], tanggal, data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)

@app.route('/sentimen', methods=['GET', 'POST'])
def sentimen():
    return render_template('sentimen.html')

@app.route('/submitsentimen', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})
    
    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})

@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':

        return redirect(url_for('index'))

    return render_template('contact.html')

if __name__ == "__main__":
    app.run()