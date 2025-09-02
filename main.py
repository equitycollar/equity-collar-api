# main.py  (diagnostic to prove wiring)
from fastapi import FastAPI
API_VERSION = "v1.7 DIAGNOSTIC"
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION}
