from service import user_service
from utils import logger

def initialise(bot):
    @bot.message_handler(commands=['end'])
    def send_end(message):
        chat_id = message.chat.id
				
        logger.log(f"Received message: {message.text}", "END COMMAND HANDLER")

        try:
            result = user_service.handle_end_shower(chat_id)
            bot.reply_to(message, result["message"])

        except Exception as e:
            bot.reply_to(message, f"An error occurred: {e}")
