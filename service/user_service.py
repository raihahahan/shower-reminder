from repository.user_repo import user_db
from utils import logger

CONTEXT = "USER SERVICE"

def handle_start_user(username: str, chat_id: str):
    logger.log("Start command called", CONTEXT)
    user_db.create_user(username, chat_id)

def handle_status_user(username: str, chat_id: str):
    logger.log("Status command called", CONTEXT)
    data = user_db.get_user_by_chat_id(chat_id)

    # get the data from the database and check if the user is showering or not

    if data[0]['has_showered_today']:
        logger.log("You have showered", CONTEXT)
        return True
    else:            
        logger.log("You have not showered", CONTEXT)
        return False

def handle_shower_request(username: str, chat_id: str):
    logger.log("Shower command called", CONTEXT)
    handle_start_user(username, chat_id)
    user_db.update_user(chat_id, {"shower_status": True})

def get_all_chat_ids():
    data = user_db.get_users(columns="chat_id")
    return list(map(lambda i : ['chat_id'], data))

def get_users_status():
    data = user_db.get_users(columns="chat_id, has_showered_today")
    return data

def handle_not_showered(chat_id):
    new_data = { 'shower_count': 0 }
    user_db.update_user(chat_id, new_data)


def handle_leaderboard_request():
    data = user_db.get_users()
    print(data)
    leaderboard = "Shower Leaderboard🚿🏆\n\n"
    for i, user in enumerate(data, start=1):
        leaderboard += (
            f"{i}️⃣ {user['username']}\n"
            f"   - Shower Status: {'✅' if user['shower_status'] else '❌'}\n"
            f"   - Shower Count: {user['shower_count']}\n"
            f"   - Showered Today: {'✅' if user['has_showered_today'] else '❌'}\n\n"
        )
    return leaderboard