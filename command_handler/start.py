def initialise(bot):
    @bot.message_handler(commands=['start', 'hello'])
    def send_welcome(message):
        print(f"Received message: {message.text}")
        bot.reply_to(message, 
            "Welcome to ShowerTracker! 🚿\n\n"
            "Commands:\n"
            "/status - Are you showering?\n"
    )