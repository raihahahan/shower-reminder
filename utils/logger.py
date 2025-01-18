import datetime

def log(message, context):
    content = f"[{context}][{datetime.datetime.now()}] {message}"
    with open("logs.txt", "a") as file:
        file.write(content + "\n")
    print(content)