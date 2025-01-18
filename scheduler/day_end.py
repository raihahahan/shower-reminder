import time
from threading import Thread
from schedule import Scheduler
from globals import END_TIME  
from utils import logger
from service import user_service

CONTEXT = "DAY END SCHEDULER"

scheduler = Scheduler()

def initialise(bot):
    def send_day_end_reminder():
        data = user_service.get_users_status()
        print(len(data))
        for item in data:
            chat_id = item['chat_id']
            has_showered_today = item['has_showered_today']
            try:
                if has_showered_today:
                    bot.send_message(chat_id, "You have showered today! Good job! ✅")
                else:
                    user_service.handle_not_showered(chat_id)
                    user_service.reset_for_day(chat_id, has_showered=has_showered_today)
                    bot.send_message(chat_id, "You have not showered today :( 🤮")
            except Exception as e:
                logger.log(f"Failed to send reminder to {chat_id}: {e}", CONTEXT)

    scheduler.every().day.at(END_TIME).do(send_day_end_reminder)

    def run_day_end_scheduler():
        logger.log("Day end scheduler started.", CONTEXT)
        while True:
            scheduler.run_pending()
            time.sleep(1)

    scheduler_thread = Thread(target=run_day_end_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()