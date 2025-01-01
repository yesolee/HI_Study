from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

count = 0

class Action(BaseModel):
    action: str

@app.get("/get-count")
def get_count():
    return {"count":count}

@app.post("/update-count")
def update_count(action: str):
    global count
    if action == "increase":
        count += 1 
    elif action == "decrease":
        count -= 1
    return {"count":count}


