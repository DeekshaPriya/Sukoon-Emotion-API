import os
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_video


app = FastAPI(
    title="Sukoon Emotion ML API"
)


# Allow requests from the Sukoon frontend
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
async def predict_emotion(
    video: UploadFile = File(...)
):

    # Create temporary file
    suffix = os.path.splitext(
        video.filename
    )[1] or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        contents = await video.read()

        temp_file.write(contents)

        temp_video_path = temp_file.name

    try:

        # Run the REAL ML model
        result = predict_video(
            temp_video_path
        )

        return {
            "status": "success",
            "filename": video.filename,
            **result
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }

    finally:

        # Delete temporary video
        if os.path.exists(
            temp_video_path
        ):
            os.remove(
                temp_video_path
            )