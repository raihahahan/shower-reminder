from service import user_service
from utils import messages

def initialise(bot):
    @bot.message_handler(commands=['help'])
    def send_help(message):
        help_response = (
            "Here are the available commands for ShowerTracker Bot 🚿:\n\n" +
            messages.HELP
        )
        bot.send_message(message.chat.id, help_response)