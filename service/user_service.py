from repository.user_repo import user_db
from utils import logger
from datetime import datetime

CONTEXT = "USER SERVICE"

from datetime import datetime

def handle_start_user(username: str, chat_id: str):
    logger.log("Start command called", CONTEXT)
    user_db.create_user(username, chat_id)

def reset_for_day(chat_id, has_showered):
    user_db.update_user(chat_id, { "shower_status": False, "start_time": None, "end_time": None, "has_showered_today": False })
    if not has_showered:
        user_db.update_user(chat_id, { "shower_count": 0 })

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
    try:
       res = user_db.update_user(chat_id, { "shower_status": True, "start_time": datetime.now().isoformat() })
    except Exception as e:
        logger.log("Error handling shower request", CONTEXT)

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
def end_shower(chat_id: str):
    user = user_db.get_user_by_chat_id(chat_id)
    if not user:
        raise ValueError("User not found!!")

    start_time = user.get("start_time")
    if start_time is None:
        raise ValueError("You did not start a shower session!! Go scan the QR code.")
    if not isinstance(start_time, str):
        raise ValueError(f"Invalid start_time format: {start_time}")

    start_time = datetime.fromisoformat(start_time)
    end_time = datetime.now()
    duration = end_time - start_time
    minutes = duration.total_seconds() / 60

    if minutes > 5:
        user_db.update_user(
            chat_id,
            {
                "shower_status": False,
                "end_time": end_time.isoformat(),
                "shower_count": user.get("shower_count", 0) + 1,
                "has_showered_today": True
            }
        )
        return {
            "status": "success",
            "message": "Shower ended! You've spent {int(minutes)} showering!"
        }
    else:
        return {
            "status": "failed",
            "message": "You did not shower at all."
        }
