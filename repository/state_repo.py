from supabase import create_client
from globals import SUPABASE_KEY, SUPABASE_URL
from repository.database import supabase_client

class StateRepo:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not set in the environment variables.")

        self.supabase = supabase_client

    def set_user_state(self, user_id, state):
        """Save or update user state in Supabase."""
        response = self.supabase.table("user_states").upsert({
            "user_id": user_id,
            "state": state,
        }).execute()
        return response

    def get_user_state(self, user_id):
        """Retrieve user state from Supabase."""
        response = self.supabase.table("user_states").select("state").eq("user_id", user_id).execute()
        if response.data:
            return response.data[0]["state"]
        return None

    def delete_user_state(self,user_id):
        """Delete user state from Supabase."""
        self.supabase.table("user_states").delete().eq("user_id", user_id).execute()

state_db = StateRepo()