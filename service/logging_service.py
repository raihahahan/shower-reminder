import datetime

def log(message, context):
    print(f"[{context}][{datetime.datetime.now()}] {message}")