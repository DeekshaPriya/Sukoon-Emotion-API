
import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from transformers import AutoModel, AutoImageProcessor

import mediapipe as mp
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


with open(
    os.path.join(BASE_DIR, "config.json"),
    "r"
) as f:
    CONFIG = json.load(f)


VIT_NAME = CONFIG["vit_name"]
NUM_FRAMES = CONFIG["num_frames"]
IMAGE_SIZE = CONFIG["image_size"]
FACE_MARGIN = CONFIG["face_margin"]
HIDDEN_SIZE = CONFIG["hidden_size"]
NUM_LSTM_LAYERS = CONFIG["num_lstm_layers"]
DROPOUT = CONFIG["dropout"]
NUM_CLASSES = CONFIG["num_classes"]

EMOTION_NAMES = CONFIG["emotion_names"]


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class VideoEmotionAttentionModel(nn.Module):

    def __init__(
        self,
        vit_model,
        input_size=768,
        hidden_size=256,
        num_layers=2,
        num_classes=6,
        dropout=0.3
    ):
        super().__init__()

        self.vit = vit_model

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):

        batch_size, num_frames, C, H, W = pixel_values.shape

        x = pixel_values.view(
            batch_size * num_frames,
            C,
            H,
            W
        )

        outputs = self.vit(
            pixel_values=x
        )

        features = (
            outputs.last_hidden_state[:, 0, :]
        )

        features = features.view(
            batch_size,
            num_frames,
            -1
        )

        lstm_out, _ = self.lstm(
            features
        )

        scores = self.attention(
            lstm_out
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            lstm_out * weights,
            dim=1
        )

        return self.classifier(context)


image_processor = AutoImageProcessor.from_pretrained(
    VIT_NAME
)

vit_model = AutoModel.from_pretrained(
    VIT_NAME
)

for param in vit_model.parameters():
    param.requires_grad = False

for param in vit_model.encoder.layer[-1].parameters():
    param.requires_grad = True

for param in vit_model.layernorm.parameters():
    param.requires_grad = True


model = VideoEmotionAttentionModel(
    vit_model=vit_model,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)


checkpoint = torch.load(
    os.path.join(
        BASE_DIR,
        "video_emotion_model.pt"
    ),
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# ------------------------------------------------------------
# MEDIAPIPE
# ------------------------------------------------------------

face_model_path = os.path.join(
    BASE_DIR,
    "blaze_face_short_range.tflite"
)

base_options = mp.tasks.BaseOptions(
    model_asset_path=face_model_path
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(
    options
)


# ------------------------------------------------------------
# FACE EXTRACTION
# ------------------------------------------------------------

def extract_video_faces(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            "Could not open video."
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise ValueError(
            "Video contains no frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    faces = []
    last_face = None

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:

            if last_face is not None:
                faces.append(last_face)

            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = face_detector.detect(
            mp_image
        )

        if result.detections:

            detection = max(
                result.detections,
                key=lambda d:
                d.bounding_box.width *
                d.bounding_box.height
            )

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            mx = int(w * FACE_MARGIN)
            my = int(h * FACE_MARGIN)

            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(
                rgb.shape[1],
                x + w + mx
            )
            y2 = min(
                rgb.shape[0],
                y + h + my
            )

            crop = rgb[y1:y2, x1:x2]

            if crop.size > 0:

                last_face = Image.fromarray(
                    crop
                )

                faces.append(last_face)

                continue

        if last_face is not None:
            faces.append(last_face)

    cap.release()

    if len(faces) == 0:
        raise ValueError(
            "No face detected."
        )

    while len(faces) < NUM_FRAMES:
        faces.append(faces[-1])

    return faces[:NUM_FRAMES]


# ------------------------------------------------------------
# PUBLIC API
# ------------------------------------------------------------

def predict_video(video_path):

    faces = extract_video_faces(
        video_path
    )

    frames = []

    for face in faces:

        face = face.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        inputs = image_processor(
            images=face,
            return_tensors="pt"
        )

        frames.append(
            inputs["pixel_values"].squeeze(0)
        )

    x = torch.stack(frames)
    x = x.unsqueeze(0)
    x = x.to(DEVICE)

    with torch.no_grad():

        if DEVICE.type == "cuda":

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(x)

        else:
            logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    index = torch.argmax(
        probabilities
    ).item()

    return {
        "emotion": EMOTION_NAMES[index],
        "confidence": round(
            float(probabilities[index]),
            4
        ),
        "probabilities": {
            EMOTION_NAMES[i]:
            round(
                float(probabilities[i]),
                4
            )
            for i in range(NUM_CLASSES)
        }
    }
