def initialise(bot):
    @bot.message_handler(func=lambda message: True)  # Matches all messages
    def echo_all(message):
        bot.send_message(message.chat.id, "Sorry, I don't understand that command. Try /help for available commands.")