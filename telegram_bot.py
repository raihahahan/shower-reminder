
import command_handler.catch_all
import command_handler.end
import command_handler.leaderboard
import command_handler.start
import command_handler.check
import command_handler.shower
import command_handler.status
from globals import BOT_TOKEN
import scheduler.daily_reminder
import scheduler.day_end
import telebot
from utils import logger

# Initialise bot
bot = telebot.TeleBot(BOT_TOKEN)

# Initialise command handlers
command_handler.start.initialise(bot)
command_handler.end.initialise(bot)
command_handler.check.initialise(bot)
command_handler.shower.initialise(bot)
command_handler.leaderboard.initialise(bot)

command_handler.catch_all.initialise(bot) # This should be the last command handler to be initialised

# Initialise schedulers
scheduler.daily_reminder.initialise(bot)
scheduler.day_end.initialise(bot)

# Start bot
logger.log("Bot started.", "BOT")
bot.infinity_polling()
