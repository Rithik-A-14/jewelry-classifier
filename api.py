# api.py
import io
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict
import os
import logging

# --- Configuration ---
LOCAL_DINO_PATH = "/root/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
FULL_MODEL_PATH = "full_jewelry_model.pth" # Ensure this matches the filename

# *********************************************************************************
# *** IMPORTANT: REPLACE THIS LIST WITH YOUR EXACT 50 CLASS NAMES IN CORRECT ORDER ***
# *********************************************************************************
CLASS_NAMES = [
    'Antique Bangles', 'Bib Necklaces', 'Bridal Necklaces', 'Button Earrings', 'Bypass Earring',
    'Byzantine Chains', 'Chandelier Earrings', 'Charm Pendants', 'Choker Necklaces', 'Chur Bangles',
    'Churi Bangles', 'Classic Rings', 'Cluster Earrings', 'Cluster Rings', 'Contemporary Bangles',
    'Contemporary Necklaces', 'Cuff Bracelets', 'Dangle Earrings', 'Drop Earrings', 'Ear Cuffs Earrings',
    'Engraved Bangles', 'Fashion Rings', 'Flower Pendants', 'Geometric Earrings', 'Geometric Pendants',
    'Heart Pendants', 'Heart_Initial Pendants', 'Initial Pendants', 'Jhumka Earrings', 'Lariat Necklaces',
    'Link Bracelets', 'Matinee Necklaces', 'Moden Pendants', 'Multi-Strand Chains', 'Pendant Chain',
    'Plain Bangles', 'Princess Necklaces', 'Religious Pendants', 'Rope Necklaces', 'Signet Rings',
    'Statement Rings', 'Stud Earrings', 'Thin Pricess Necklaces', 'Traditional Choker Necklaces',
    'Traditional Matinee Necklaces', 'Traditional Princess Necklaces', 'Traditional Rope Necklaces',
    'Vintage Rings', 'Wedding Bands', 'Wheat Chains'
]
# *********************************************************************************

# --- Set up logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI app ---
app = FastAPI(title="Jewelry Classifier API", description="API for classifying jewelry images using DINOv3.")

# --- Pydantic model for API response ---
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    top_5_predictions: Dict[str, float]

# --- Recreate the Full Model Architecture ---
class DINOv3Classifier(nn.Module):
    def __init__(self, dinov3_model, num_classes, dropout_rate=0.3):
        super(DINOv3Classifier, self).__init__()
        self.dinov3 = dinov3_model
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.dinov3.config.hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.dinov3(x)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

# --- Global variables for model and processor ---
model = None
processor = None
device = None

# --- Startup event to load model ---
@app.on_event("startup")
async def load_model():
    global model, processor, device
    logger.info("Starting up: Loading model components...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")

    try:
        # Load DINOv3 processor and model from local path
        processor = AutoImageProcessor.from_pretrained(LOCAL_DINO_PATH)
        base_model = AutoModel.from_pretrained(LOCAL_DINO_PATH)
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("DINOv3 components loaded from local cache.")

        # Create and load the full model
        num_classes = len(CLASS_NAMES)
        model = DINOv3Classifier(base_model, num_classes)
        model.load_state_dict(torch.load(FULL_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Full model weights loaded and model set to evaluation mode.")
    except Exception as e:
        logger.error(f"Error loading model components: {e}")
        raise # Re-raise to prevent app from starting in a broken state

# --- API Endpoint ---
@app.post("/predict/", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")

    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")

    try:
        contents = await file.read()
        logger.debug("File content read.")

        image = Image.open(io.BytesIO(contents)).convert('RGB')
        logger.debug("Image converted to PIL.")

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        logger.debug("Image preprocessed.")

        with torch.no_grad():
            outputs = model(pixel_values)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        logger.info(f"Prediction made: {predicted_class} (Confidence: {confidence_score:.4f})")

        top_k = min(5, len(CLASS_NAMES))
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs_np = top_probs.cpu().numpy().flatten()
        top_indices_np = top_indices.cpu().numpy().flatten()

        top_5_dict = {CLASS_NAMES[idx]: float(prob) for idx, prob in zip(top_indices_np, top_probs_np)}
        logger.debug(f"Top 5 predictions: {top_5_dict}")

        return PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence_score,
            top_5_predictions=top_5_dict
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(device)}

# --- Root endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Jewelry Classifier API!", "docs": "/docs"}