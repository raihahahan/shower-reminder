from service import user_service
from utils import logger
from command_handler.shower import send_shower_helper

def initialise(bot):
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        logger.log(f"Received message: {message.text}", "START COMMAND HANDLER")
        if message.text.startswith('/start '):
            parameter = message.text.split(' ')[1]
            if parameter == 'shower':
                send_shower_helper(message, bot)
                return

        username = message.from_user.username
        chat_id = message.chat.id
        user_service.handle_start_user(username, chat_id)

        bot.send_message(chat_id, 
            "Welcome to ShowerTracker! 🚿\n\n"
            "/shower - Start a shower 🛀\n"
            "/check - Have you showered? 🤔\n"
            "/leaderboard - Check the leaderboard 🔝\n"
        )
