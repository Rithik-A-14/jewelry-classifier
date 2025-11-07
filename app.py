# app.py
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import gradio as gr
import os
import logging  # Import logging module

# --- Set up logging ---
# Configure the root logger to show INFO level messages and above
logging.basicConfig(
    level=logging.INFO,  # Adjust to DEBUG for more verbose logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Create a logger object for this module/script
logger = logging.getLogger(__name__)
# --- End logging setup ---

# --- Configuration ---
# Path to the local snapshot of the DINOv3 model inside the container's HF cache
LOCAL_DINO_PATH = "/root/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51"
FULL_MODEL_PATH = "full_jewelry_model_dinov3.pth" # Ensure this matches the filename
INPUT_SIZE = 518

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {DEVICE}") # Log device info
if DEVICE.type == 'cuda':
    logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")

# --- Recreate the Full Model Architecture EXACTLY as during training ---
class DINOv3Classifier(nn.Module):
    def __init__(self, dinov3_model, num_classes, dropout_rate=0.3):
        super(DINOv3Classifier, self).__init__()
        self.dinov3 = dinov3_model
        hidden_size = self.dinov3.config.hidden_size
        logger.debug(f"DINOv3 hidden size detected: {hidden_size}") # Log debug info
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 512),
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

# --- Load Components ---
logger.info("Loading DINOv3 processor and base model from local cache...")
try:
    processor = AutoImageProcessor.from_pretrained(LOCAL_DINO_PATH)
    base_model = AutoModel.from_pretrained(LOCAL_DINO_PATH)
    for param in base_model.parameters():
        param.requires_grad = False
    logger.info("DINOv3 components loaded from local cache.")
except Exception as e:
    logger.error(f"Error loading DINOv3 components from local cache: {e}")
    exit(1)

# --- Create and Load the Full Model ---
NUM_CLASSES = len(CLASS_NAMES)
logger.info(f"Number of classes configured: {NUM_CLASSES}")

full_model = DINOv3Classifier(base_model, NUM_CLASSES)

# --- Load the COMPLETE Trained Model Weights ---
logger.info(f"Loading FULL trained model weights from {FULL_MODEL_PATH}...")
try:
    state_dict = torch.load(FULL_MODEL_PATH, map_location=DEVICE)
    full_model.load_state_dict(state_dict)
    logger.info("FULL model weights loaded successfully.")
except FileNotFoundError:
    logger.error(f"Error: Could not find the full model file at {FULL_MODEL_PATH}.")
    exit(1)
except Exception as e:
    logger.error(f"Error loading full model weights: {e}")
    exit(1)

full_model.to(DEVICE)
full_model.eval()
logger.info("Full model is ready for prediction.")

# --- Prediction Function for Gradio (WITH LOGGING) ---
def predict_jewelry(image):
    logger.info("Received prediction request.") # Log when prediction starts
    if image is None:
        logger.warning("No image uploaded.") # Log warning
        return "No image uploaded.", {}

    try:
        if image.mode != 'RGB':
            logger.debug("Converting image to RGB.")
            image = image.convert('RGB')

        logger.debug("Preprocessing image...")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(DEVICE)
        logger.debug("Image preprocessed.")

        logger.debug("Running model inference...")
        with torch.no_grad():
            outputs = full_model(pixel_values)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        logger.debug("Model inference completed.")

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

        logger.info(f"Prediction: {predicted_class} (Confidence: {confidence_score:.4f})") # Log main result

        top_k = min(5, NUM_CLASSES)
        logger.debug("Calculating top 5 predictions...")
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        top_probs_np = top_probs.cpu().numpy().flatten()
        top_indices_np = top_indices.cpu().numpy().flatten()

        results_dict = {CLASS_NAMES[idx]: float(prob) for idx, prob in zip(top_indices_np, top_probs_np)}
        logger.debug(f"Top 5 predictions calculated: {results_dict}") # Log top 5

        return predicted_class, results_dict

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True) # Log error with traceback
        return f"Error during prediction: {str(e)}", {}

# --- Create Gradio Interface ---
description_text = """
<h1 style='text-align: center; color: #4A90E2;'>💎 Jewelry Classifier 💎</h1>
<p style='text-align: center;'>Upload an image of jewelry, and this model (based on DINOv3) will classify it into one of the 50 categories.</p>
<p style='text-align: center;'><em>Powered by PyTorch & Hugging Face Transformers</em></p>
"""

interface = gr.Interface(
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Class"),
        gr.Label(num_top_classes=5, label="Top 5 Predictions & Confidences")
    ],
    title="",
    description=description_text,
    examples=[],
    flagging_mode="never", # Updated from allow_flagging
    theme=gr.themes.Soft()
)

# --- Launch the Interface ---
if __name__ == "__main__":
    logger.info("\nLaunching Gradio UI...")
    logger.info("Please wait for the interface to load in your browser...")
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)