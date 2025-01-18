import os
import dotenv

dotenv.load_dotenv()

REMINDER_TIME = os.environ.get('REMINDER_TIME')
BOT_TOKEN = os.environ.get('BOT_TOKEN')