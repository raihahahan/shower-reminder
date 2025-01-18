import time
from threading import Thread
from schedule import Scheduler
from globals import REMINDER_TIME, BOT_TOKEN    
from utils import logger
from service import user_service

scheduler = Scheduler()

CONTEXT = "DAILY REMINDER SCHEDULER"

def initialise(bot):
    def send_daily_reminder():
        chat_ids = user_service.get_all_chat_ids()
        for chat_id in chat_ids:
            try:
                bot.send_message(chat_id, "Good morning! Don't forget to shower!")
            except:
                logger.log(f"Failed to send reminder to {chat_id}", CONTEXT)

    scheduler.every().day.at(REMINDER_TIME).do(send_daily_reminder)

    def run_scheduler():
        logger.log("Scheduler started.", CONTEXT)
        while True:
            scheduler.run_pending()
            time.sleep(1)

    scheduler_thread = Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()