# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_fr = QA("french_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    french_para = "Google LLC est une entreprise américaine de services technologiques fondée en 1998 dans la Silicon Valley, en Californie, par Larry Page et Sergey Brin, créateurs du moteur de recherche Google. C'est une filiale de la société Alphabet depuis août 2015. L'entreprise s'est principalement fait connaître à travers la situation monopolistique de son moteur de recherche, concurrencé historiquement par AltaVista puis par Yahoo! et Bing. Elle a ensuite procédé à de nombreuses acquisitions et développements et détient aujourd'hui de nombreux logiciels et sites web notables parmi lesquels YouTube, le système d'exploitation pour téléphones mobiles Android, ainsi que d'autres services tels que Gmail, Google Earth, Google Maps ou Google Play. Google s'est donné comme mission « d'organiser l'information à l'échelle mondiale et de la rendre universellement accessible et utile ». Après Larry Page et Eric Schmidt, son DG est, depuis 2015, Pichai Sundararajan."

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a' , encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()

    # This block calls the prediction function and return the response
    try:        
        out = model_fr.predict(french_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 10:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
