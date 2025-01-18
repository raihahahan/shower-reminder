def initialise(bot):
    @bot.message_handler(commands=['start', 'hello'])
    def send_welcome(message):
        print(f"Received message: {message.text}")
        bot.reply_to(message, 
            "Welcome to ShowerTracker! 🚿\n\n"
            "Commands:\n"
            "/track - Record your shower\n"
            "/status - Check your streak\n"
    )

    @bot.message_handler(func=lambda message: True)  # Matches all messages
    def echo_all(message):
        bot.reply_to(message, "Sorry, I don't understand that command. Try /help for available commands.")