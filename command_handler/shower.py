from service import user_service
from service import state_service
import api.photo_validator

def initialise(bot):
    @bot.message_handler(commands=['shower'])
    def send_shower(message):
        user_id = message.from_user.id
        username = message.from_user.username
        chat_id = message.chat.id

        is_showering = user_service.handle_status_user(username, user_id)
        
        if is_showering: 
            bot.send_message(chat_id, "You are currently taking a shower.")
        else:
            bot.send_message(chat_id, "Okay go take a shower u smelly dog. Send me a picture of the shower head for proof.")
            user_id = message.from_user.id
            state_service.handle_shower_request(user_id)

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        user_id = message.from_user.id
        username = message.from_user.username

        if state_service.handle_photo_request(user_id):
            photo = message.photo[-1].file_id
            file_info = bot.get_file(photo)
            downloaded_photo = bot.download_file(file_info.file_path)

            is_valid = api.photo_validator.validate_photo(downloaded_photo)

            if is_valid:
                bot.reply_to(message, "Nice! You're good to go.")
                state_service.finish_photo_request(user_id)
                user_service.handle_shower_request(username, user_id)
            else:
                bot.reply_to(message, "Send a picture first.")
        
        else:
            bot.reply_to(message, "I wasn't expecting a picture from you right now.")
