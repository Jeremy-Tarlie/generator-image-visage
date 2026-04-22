from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Final

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from .model import GeneratorService

CHECKPOINT_FILENAME: Final[str] = "gan_faces.pt"
CHECKPOINT_PATH: Final[Path] = Path(__file__).resolve().parents[1] / "generation_model" / CHECKPOINT_FILENAME


class GenerateFaceResponse(BaseModel):
    image_base64: str
    format: str
    width: int
    height: int


app = FastAPI(title="GAN Face Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


generator_service: GeneratorService | None = None


@app.on_event("startup")
def startup_event() -> None:
    global generator_service
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint introuvable: {CHECKPOINT_PATH}")
    generator_service = GeneratorService(CHECKPOINT_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateFaceResponse)
def generate_face() -> GenerateFaceResponse:
    if generator_service is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    generated_tensor = generator_service.generate()
    image_array = (generated_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
    image = Image.fromarray(image_array)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return GenerateFaceResponse(
        image_base64=encoded,
        format="png",
        width=image.width,
        height=image.height,
    )
