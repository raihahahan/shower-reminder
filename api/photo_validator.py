import base64

def validate_photo(photo):
    base64_photo = base64.b64encode(photo).decode('utf-8')    
    # TODO: make request to check if photo is valid
    print(f"Photo received: {base64_photo}")
    return True