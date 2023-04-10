import random
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from telegram import Update
from telegram.ext import MessageHandler, filters, Application
from dotenv import load_dotenv
import os

load_dotenv()

config_file = open("big_bot_config.json")
bot_config = json.load(config_file)
vectorizer = CountVectorizer()
X = []
y = []
for name, date in bot_config["intents"].items():
    for example in date["examples"]:
        X.append(example)
        y.append(name)
vectorizer.fit(X)
vecX = vectorizer.transform(X)
model = RandomForestClassifier()
model.fit(vecX, y)


def filter(text):
    text = text.lower()
    alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz -"
    result = [c for c in text if c in alphabet]
    return "".join(result)


def matching(text, example):
    text = filter(text)
    example = example.lower()
    distance = nltk.edit_distance(text, example) / len(example)  # от 0 до 1
    return distance < 0.2


def get_intent(text):
    for intent in bot_config["intents"]:
        for examples in bot_config["intents"][intent]["examples"]:
            if matching(text, examples):
                return intent


def bot(text):
    intent = get_intent(text)
    if not intent:
        transform_text = vectorizer.transform([text])
        intent = model.predict(transform_text)[0]
    if intent:
        return random.choice(bot_config["intents"][intent]["responses"])
    return random.choice(bot_config["failure_phrases"])


my_token = os.getenv("TOKEN")


async def botReactOnMsg(update: Update, context):
    upd_msg = update.message
    text = upd_msg.text  # написал пользователь
    print(f"[user] : {text}")
    reply = bot(text)
    await upd_msg.reply_text(reply)


app = Application.builder().token(my_token).build()
handler = MessageHandler(filters.TEXT, botReactOnMsg)
app.add_handler(handler)
app.run_polling()
