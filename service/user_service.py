from repository.user_repo import user_db
from utils import logger
from datetime import datetime, timezone

CONTEXT = "USER SERVICE"

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
    has_showered = data['has_showered_today']
    shower_status = data['shower_status']

    return { 'has_showered_today': has_showered, 'shower_status': shower_status }

def handle_shower_request(username: str, chat_id: str):
    logger.log("Shower command called", CONTEXT)
    handle_start_user(username, chat_id)
    try:
       res = user_db.update_user(chat_id, { "shower_status": True, "start_time": datetime.now(timezone.utc).isoformat() })
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
    data = sorted(data, key=lambda user: user['shower_count'], reverse=True)
    
    leaderboard = "Shower Leaderboard🚿🏆\n\n"

    for i, user in enumerate(data, start=1):
        total_duration = user.get('total_duration', 0.0)
        duration = f"{total_duration:.2f} minutes" if total_duration > 0.0 else "hasn't showered 🗿️️" 

        leaderboard += (
            f"{i}️⃣ {user['username']}\n"
            f" {'👑 Shower King 👑' if user['shower_count'] > 5 else '🚿 Rookie 🚿'}\n\n"
            f"   • *Showered Today:* {'✅' if user['has_showered_today'] else '❌'}\n"
            f"   • *Showering Now:* {'✅' if user['shower_status'] else '❌'}\n"
            f"   • *Shower Count:* {user['shower_count']}\n"
            f"   • *Total Shower Time:* {duration}\n"
            "━━━\n" 
        )

    return leaderboard

def handle_end_shower(chat_id: str):
    user = user_db.get_user_by_chat_id(chat_id)
    if not user:
        return {
            "status": "failed",
            "message": "User not found!! Start a new chat now using the /start command."
        }

    start_time = user.get("start_time")
    if not user.get("shower_status"):
        return {
            "status": "failed",
            "message": "You did not start a shower session!! Start your shower now using the /shower command."
        }
    
    start_time = datetime.fromisoformat(start_time)
    end_time = datetime.now(timezone.utc)
    duration = end_time - start_time
    minutes = duration.total_seconds() / 60

    logger.log(f"Start Time: {start_time}, End Time: {end_time}, Duration: {minutes} minutes", CONTEXT)

    if minutes > 0.1:
        user_db.update_user(
            chat_id,
            {
                "shower_status": False,
                "end_time": end_time.isoformat(),
                "shower_count": user.get("shower_count", 0) + 1,
                "has_showered_today": True,
                "total_duration": user.get("shower_duration", 0.0) + minutes
            }
        )
        return {
            "status": "success",
            "message": f"Shower ended! You've spent {minutes:.2f} minutes showering! "
        }
    else:
        return {
            "status": "failed",
            "message": "Shower longer!"
        }


def handle_check_email(username):
    #find your email through username
    print(username)
    data = user_db.get_user_by_username(username)
    return False if 'email' not in data else True