import command_handler.catch_all
import command_handler.end
import command_handler.start
import telebot
import command_handler
from service import scheduler, logging_service
from globals import BOT_TOKEN

# Initialise bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialise command handlers
command_handler.start.initialise(bot)
command_handler.end.initialise(bot)
command_handler.catch_all.initialise(bot)
command_handler.start.initialise(bot)

# Initialise scheduler
scheduler.initialise(bot)

# Start bot
logging_service.log("Bot started.", "BOT")
bot.infinity_polling()
