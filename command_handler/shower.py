from service import user_service
from service import state_service
import api.photo_validator

def send_shower_helper(message, bot):
    user_id = message.from_user.id
    username = message.from_user.username
    chat_id = message.chat.id

    status = user_service.handle_status_user(username, user_id)
    
    if status['shower_status']:
        bot.send_message(chat_id, "You are currently taking a shower.")
    else:
        bot.send_message(chat_id, "Please send a picture of your shower head for proof.")
        user_id = message.from_user.id
        state_service.handle_shower_request(user_id)

def initialise(bot):
    @bot.message_handler(commands=['shower'])
    def send_shower(message):
        send_shower_helper(message, bot)

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        user_id = message.from_user.id
        username = message.from_user.username
        chat_id = message.chat.id

        if state_service.handle_photo_request(user_id):
            photo = message.photo[-1].file_id
            file_info = bot.get_file(photo)
            downloaded_photo = bot.download_file(file_info.file_path)

            is_showerhead, confidence, predicted_class, is_confident = api.photo_validator.validate_photo(downloaded_photo)

            if is_showerhead:
                response_message = (
                    f"Nice! You're good to go. 🛁\n\n"
                    f"**Prediction Details:**\n"
                    f"- Predicted Class: {predicted_class}\n"
                    f"- Confidence: {confidence:.2f}\n"
                    f"- Confident Prediction: {'Yes' if is_confident else 'No'}"
                )
                bot.send_message(chat_id, response_message, parse_mode="Markdown")
                user_service.handle_shower_request(username, user_id)
            else:
                response_message = (
                    f"Please send a picture of a shower head instead 😾.\n\n"
                    f"**Prediction Details:**\n"
                    f"- Predicted Class: {predicted_class}\n"
                    f"- Confidence: {confidence:.2f}\n"
                    f"- Confident Prediction: {'Yes' if is_confident else 'No'}"
                )
                bot.reply_to(message, response_message, parse_mode="Markdown")
                state_service.finish_photo_request(user_id)
        
        else:
            bot.reply_to(message, "I wasn't expecting a picture from you right now.")
