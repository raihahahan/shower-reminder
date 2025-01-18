def initialise(bot):
    @bot.message_handler(commands=['status'])
    def send_status(message):
        bot.reply_to(message, "Status command called")
        
        #calling status will return the status of the user 
        #calls a function from service folder -> to
        # 1. fetch data from databse 
        # 2. use the results to determine if the user is showering or not

        