from service import user_service
from service import logging_service

def initialise(bot):
    @bot.message_handler(commands=['end'])
    def send_end(message):
        chat_id = message.chat.id
				
        logging_service.log(f"Received message: {message.text}", "END COMMAND HANDLER")

        result = user_service.end_shower(chat_id)

        if result["status"] == "success":
            bot.reply_to(message, result["message"])
        else:
            bot.reply_to(message, result["message"])
