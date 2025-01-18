def initialise(bot):
    @bot.message_handler(commands=['shower'])
    def send_shower(message):
        bot.reply_to(message, "Okay got take a shower u smelly dog")
        user_id = message.from_user.id
        username = message.from_user.username
        