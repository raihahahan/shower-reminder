from globals import SUPABASE_KEY, SUPABASE_URL
from repository.database import supabase_client

class UserRepo:
    def __init__(self):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase credentials not set in the environment variables.")

        self.supabase = supabase_client
    
    def get_user_by_chat_id(self, chat_id):
        """Fetch a user by their Telegram chat ID."""
        response = self.supabase.table("users").select("*").eq("chat_id", chat_id).execute()
        if response.error:
            raise Exception(f"Error fetching user by chat_id: {response.error}")
        return response.data

    def create_user(self, username, chat_id):
        """Insert a new user into the database."""
        try:
            response = self.supabase.table("users").upsert({"username": username, "chat_id": chat_id}).execute()
            if not response:
                raise Exception(f"Error creating user: {response.error}")
            return response.data
        except:
            return

    def update_user_chat_id(self, username, new_data):
        """Update a user's chat ID."""
        response = self.supabase.table("users").update(new_data).eq("username", username).execute()
        if response.error:
            raise Exception(f"Error updating user: {response.error}")
        return response.data

    def delete_user_by_chat_id(self, chat_id):
        """Delete a user from the database by chat ID."""
        response = self.supabase.table("users").delete().eq("chat_id", chat_id).execute()
        if response.error:
            raise Exception(f"Error deleting user by chat_id: {response.error}")
        return response.data

user_db = UserRepo()