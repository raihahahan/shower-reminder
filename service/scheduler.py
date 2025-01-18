import time
import datetime
from threading import Thread
import schedule
from globals import REMINDER_TIME
from utils import logger


def initialise(bot):
    def send_daily_reminder():
        for chat_id in []:
            bot.send_message(chat_id, "Good morning! Don't forget to shower!")

    schedule.every().day.at(str(REMINDER_TIME)).do(send_daily_reminder)

    def run_scheduler():
        logger.log("Scheduler started.", "SCHEDULER")
        while True:
            schedule.run_pending()
            time.sleep(1)

    scheduler_thread = Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()