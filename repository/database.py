from supabase import create_client, Client
from globals import SUPABASE_KEY, SUPABASE_URL

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)