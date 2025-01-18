from service import user_service

def initialise(bot):
    @bot.message_handler(commands=['leaderboard'])
    def send_leaderboard(message):
        leaderboard = user_service.handle_leaderboard_request(bot)
        bot.send_message(message.chat.id, leaderboard)