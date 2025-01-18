from service import user_service

def initialise(bot):
    @bot.message_handler(commands=['help'])
    def send_help(message):
        help_response = (
            "Here are the available commands for ShowerTracker Bot 🚿:\n\n"
            "/start - Start interacting with the bot\n"
            "/shower - Start a shower session 🛁\n"
            "/end - End your current shower session 🚿\n"
            "/check - Check your current showering status 🤔\n"
            "/leaderboard - View the shower leaderboard \n"
            "/help - Display this help message ℹ️\n"
        )
        bot.send_message(message.chat.id, help_response)