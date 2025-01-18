from service import user_service

def initialise(bot):
    @bot.message_handler(commands=['check'])
    def send_status(message):
        user_id = message.from_user.id
        username = message.from_user.username
        status = user_service.handle_status_user(username, user_id)
        has_showered = status['has_showered_today']
        shower_status = status['shower_status']

        if has_showered:
            bot.send_message(
                user_id, "You have showered today."
                )
        else:
            if shower_status:
                bot.send_message(user_id, "You haven't showered but are currently showering. You may end with /end")
            else:
                bot.send_message(
                    user_id, "You haven't showered...\nDo you wish to start? \n\n"
                    "/shower - Start showering\n"
                    "/end - No later, hopefully\n"
                    )

