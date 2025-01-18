from repository.user_repo import user_db

def handle_start_user(username: str, chat_id: str):
    print("Start command called")
    user_db.create_user(username, chat_id)

def handle_status_user(username: str, chat_id: str):
    print("Status command called")
    data = user_db.get_user_by_chat_id(chat_id)
    print(data)

    #get the data from the database and check if the user is showering or not
    
    if data[0]['shower_status']:
        print("User is showering")
        return True
    else:            
        print("User is not showering")
        return False
    