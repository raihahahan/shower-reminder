from repository.state_repo import state_db

STATES = {
    'AWAIT': 'awaiting_picture'
}

def handle_shower_request(user_id):
    state_db.set_user_state(user_id, STATES['AWAIT'])

def handle_photo_request(user_id):
    state = state_db.get_user_state(user_id)
    return state == STATES['AWAIT']

def finish_photo_request(user_id):
    state_db.delete_user_state(user_id)
        