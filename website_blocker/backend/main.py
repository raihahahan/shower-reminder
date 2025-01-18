from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
import os
import dotenv
import uvicorn
dotenv.load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = FastAPI()

@app.get("/check_shower/{username}")
async def check_shower(username: str):
    try:
        response = supabase.table("users").select("has_showered_today").eq("username", username).single().execute()
        
        if not response:
            raise HTTPException(status_code=400, detail=f"Error querying Supabase: {response.error}")
        
        if response.data:
            return {"has_showered_today": response.data["has_showered_today"]}
        else:
            raise HTTPException(status_code=404, detail="User not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

