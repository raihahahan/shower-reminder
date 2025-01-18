import os
import dotenv

env_file = os.getenv("ENV_FILE", ".env")
dotenv.load_dotenv(dotenv_path=env_file)

REMINDER_TIME = os.environ.get('REMINDER_TIME')
END_TIME = os.environ.get('END_TIME')
BOT_TOKEN = os.environ.get('BOT_TOKEN')
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
WEB_BLOCKER_TRIAL_URL = os.getenv("WEB_BLOCKER_TRIAL_URL")
TELEGRAM_URL = os.getenv("TELEGRAM_URL")