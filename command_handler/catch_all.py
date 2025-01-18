def initialise(bot):
    @bot.message_handler(func=lambda message: True)  # Matches all messages
    def echo_all(message):
        bot.reply_to(message, "Sorry, I don't understand that command. Try /help for available commands.")