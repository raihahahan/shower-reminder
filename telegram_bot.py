import os
import telebot
import dotenv
dotenv.load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    print(f"Received message: {message.text}")
    bot.reply_to(message, 
        "Welcome to ShowerTracker! 🚿\n\n"
        "Commands:\n"
        "/track - Record your shower\n"
        "/status - Check your streak\n"
    )

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Sorry, I don't understand that command. Try /help for available commands.")

if __name__ == "__main__":
    print("Bot started...")
    bot.infinity_polling()

