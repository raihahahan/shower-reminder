import os
import dotenv

dotenv.load_dotenv()

REMINDER_TIME = os.environ.get('REMINDER_TIME')
END_TIME = os.environ.get('END_TIME')
BOT_TOKEN = os.environ.get('BOT_TOKEN')
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")