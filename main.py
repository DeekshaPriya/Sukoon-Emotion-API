from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sukoon Emotion ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Sukoon Emotion ML API is running!",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict-emotion")
async def predict_emotion(video: UploadFile = File(...)):

    print("Received video:", video.filename)

    return {
        "status": "success",
        "filename": video.filename,
        "emotion": "Happy",
        "confidence": 0.85,
        "message": "Dummy prediction - model not connected yet"
    }