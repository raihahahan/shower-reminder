from service import user_service
from utils import logger

def initialise(bot):
    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        logger.log(f"Received message: {message.text}", "START COMMAND HANDLER")
        username = message.from_user.username
        chat_id = message.chat.id
        user_service.handle_start_user(username, chat_id)

        bot.reply_to(message, 
            "Welcome to ShowerTracker! 🚿\n\n"
            "/check - Have you showered?\n"
    )
