import os
import dotenv
import time
from threading import Thread
import schedule
import telebot
from repository.user_repo import UserRepo

dotenv.load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
REMINDER_TIME = os.environ.get('REMINDER_TIME')
bot = telebot.TeleBot(BOT_TOKEN)
chat_ids = []
user_db = UserRepo()

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    print(f"Received message: {message.text}")
    bot.reply_to(message, 
        "Welcome to ShowerTracker! 🚿\n\n"
        "Commands:\n"
        "/track - Record your shower\n"
        "/status - Check your streak\n"
    )

def echo_all(message):
    bot.reply_to(message, "Sorry, I don't understand that command. Try /help for available commands.")

def send_daily_reminder():
    for chat_id in chat_ids:
        bot.send_message(chat_id, "Good morning! Don't forget to shower!")

schedule.every().day.at(REMINDER_TIME).do(send_daily_reminder)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

scheduler_thread = Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

# Start polling for bot commands
bot.infinity_polling()

