from service import user_service
from service import logging_service

def initialise(bot):
    @bot.message_handler(commands=['start', 'hello'])
    def send_welcome(message):
        logging_service.log(f"Received message: {message.text}", "START COMMAND HANDLER")
        user_id = message.from_user.id
        username = message.from_user.username
        user_service.handle_start_user(username, user_id)
        bot.reply_to(message, 
            "Welcome to ShowerTracker! 🚿\n\n"
            "Commands:\n"
            "/track - Record your shower\n"
            "/status - Check your streak\n"
    )
        