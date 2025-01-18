from repository.user_repo import user_db

def handle_start_user(username: str, chat_id: str):
    user_db.create_user(username, chat_id)