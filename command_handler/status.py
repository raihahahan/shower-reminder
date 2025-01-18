from service import user_service

def initialise(bot):
    @bot.message_handler(commands=['status'])
    def send_status(message):
        bot.reply_to(message, "Status command called")
        user_id = message.from_user.id
        username = message.from_user.username
        is_showering = user_service.handle_status_user(username, user_id)

        if is_showering:
            bot.send_message(
                user_id, "You are currently showering.\nDo you wish to stop? \n\n"
                "/end - Stop showering\n"
                )
        else:
            bot.send_message(
                user_id, "You are not showering...\nDo you wish to start? \n\n"
                "/shower - Start showering\n"
                "/end - No later, hopefully\n"
                )

        #calling status will return the status of the user 
        #calls a function from service folder -> to
        # 1. fetch data from databse 
        # 2. use the results to determine if the user is showering or not

