from service import user_service

def initialise(bot):
    @bot.message_handler(commands=['shower'])
    def send_shower(message):
        bot.reply_to(message, "Okay go take a shower u smelly dog")
        user_id = message.from_user.id
        username = message.from_user.username
        user_service.handle_shower_request(username, user_id)