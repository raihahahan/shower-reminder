def initialise(bot):
    @bot.message_handler(commands=['end'])
    def send_end(message):
        bot.reply_to(message, "Please shower")