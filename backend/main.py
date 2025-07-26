import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, auth

# --- 1. CONFIGURATION ---
print("--- Loading DeepReveal V2 Final Ensemble System ---")
DEVICE = torch.device("cpu")

# Paths for our two specialist models
# Ensure these files are in the same directory as this script.
AUGMENTED_MODEL_PATH = 'DeepReveal_V2_Augmented.pth'
SPECIALIST_MODEL_PATH = 'DeepReveal_CIPLAB_Specialist.pth'

if not os.path.exists(AUGMENTED_MODEL_PATH) or not os.path.exists(SPECIALIST_MODEL_PATH):
    raise FileNotFoundError("One or both model files are missing. Please place both .pth files in this directory.")

# Class mapping (0=fake, 1=real)
idx_to_class = {0: 'fake', 1: 'real'}
class_to_idx = {v: k for k, v in idx_to_class.items()}


# --- 2. MODEL DEFINITIONS ---
# We need both model architectures defined, as they are different.

# Architecture for the "Augmented" model (without attention)
class DeepRevealModel4Branch(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__(); self.rgb_branch = models.efficientnet_b0(weights=None); self.ela_branch = models.efficientnet_b0(weights=None); self.res_branch = models.efficientnet_b0(weights=None); num_features = self.rgb_branch.classifier[1].in_features; self.rgb_branch.classifier = nn.Identity(); self.ela_branch.classifier = nn.Identity(); self.res_branch.classifier = nn.Identity(); self.fft_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); self.classifier = nn.Sequential(nn.Linear(num_features * 3 + 32, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, rgb, ela, res, fft):
        feat_rgb = self.rgb_branch(rgb); feat_ela = self.ela_branch(ela); feat_res = self.res_branch(res); feat_fft = self.fft_cnn(fft).flatten(1); fused = torch.cat((feat_rgb, feat_ela, feat_res, feat_fft), dim=1); return self.classifier(fused)

# Squeeze-and-Excitation Block for the Attention Model
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__(); self.avg_pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size(); y = self.avg_pool(x).view(b, c); y = self.fc(y).view(b, c, 1); return x * y.expand_as(x)

# Architecture for the "CIPLAB Specialist" model (with attention)
class DeepRevealModelWithAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__(); self.rgb_branch = models.efficientnet_b0(weights=None); self.ela_branch = models.efficientnet_b0(weights=None); self.res_branch = models.efficientnet_b0(weights=None); num_features = self.rgb_branch.classifier[1].in_features; self.rgb_branch.classifier = nn.Identity(); self.ela_branch.classifier = nn.Identity(); self.res_branch.classifier = nn.Identity(); self.fft_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)); fused_features_dim = num_features * 3 + 32; self.attention = SEBlock(fused_features_dim); self.classifier = nn.Sequential(nn.Linear(fused_features_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
    def forward(self, rgb, ela, res, fft):
        feat_rgb = self.rgb_branch(rgb); feat_ela = self.ela_branch(ela); feat_res = self.res_branch(res); feat_fft = self.fft_cnn(fft).flatten(1); fused = torch.cat((feat_rgb, feat_ela, feat_res, feat_fft), dim=1); fused_reshaped = fused.unsqueeze(-1); attended_features = self.attention(fused_reshaped).squeeze(-1); return self.classifier(attended_features)

# Wrapper class for Grad-CAM (to be used with the Augmented model)
class DeepRevealWithCAM(nn.Module):
    def __init__(self, model):
        super(DeepRevealWithCAM, self).__init__(); self.model = model; self.rgb_features = self.model.rgb_branch.features
    def forward(self, rgb, ela, res, fft):
        feature_maps = self.rgb_features(rgb); pooled_features = self.model.rgb_branch.avgpool(feature_maps); flattened_features = torch.flatten(pooled_features, 1); feat_ela = self.model.ela_branch(ela); feat_res = self.model.res_branch(res); feat_fft = self.model.fft_cnn(fft).flatten(1); fused = torch.cat((flattened_features, feat_ela, feat_res, feat_fft), dim=1); output = self.model.classifier(fused); return output, feature_maps


# --- 3. LOAD ALL MODELS ---
print("Loading Augmented Model (Fake Detection Expert)...")
model_augmented_base = DeepRevealModel4Branch().to(DEVICE)
model_augmented_base.load_state_dict(torch.load(AUGMENTED_MODEL_PATH, map_location=DEVICE))
model_augmented_base.eval()
# Wrap the augmented model for Grad-CAM
model_augmented_cam = DeepRevealWithCAM(model_augmented_base).to(DEVICE)
model_augmented_cam.eval()

print("Loading Specialist Model (Real Identification Expert)...")
model_specialist = DeepRevealModelWithAttention().to(DEVICE)
model_specialist.load_state_dict(torch.load(SPECIALIST_MODEL_PATH, map_location=DEVICE))
model_specialist.eval()
print("✅ All ensemble models loaded successfully.")


# --- FastAPI, Firebase, and Preprocessing ---
SERVICE_ACCOUNT_KEY_PATH = 'firebase-service-account.json'
if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH): raise FileNotFoundError("Firebase service account key not found.")
cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
if not firebase_admin._apps: firebase_admin.initialize_app(cred)
app = FastAPI(title="DeepReveal API (Final Ensemble)", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try: return auth.verify_id_token(token)
    except Exception: raise HTTPException(status_code=401, detail="Invalid credentials")
def generate_ela(image, quality=90): output_io = io.BytesIO(); image.save(output_io, "JPEG", quality=quality); output_io.seek(0); temp_image = Image.open(output_io); ela_image = ImageChops.difference(image, temp_image); extrema = ela_image.getextrema(); max_diff = max([ex[1] for ex in extrema]) if extrema else 1; scale = 255.0 / max_diff if max_diff != 0 else 1; return Image.eval(ela_image, lambda p: p * scale)
def generate_residual(image): img_cv = np.array(image.convert('RGB')); denoised_img = cv2.medianBlur(img_cv, 3); residual = cv2.absdiff(img_cv, denoised_img); return Image.fromarray(residual)
def generate_fft(image): img_gray = image.convert('L'); img_gray_np = np.array(img_gray); f = np.fft.fft2(img_gray_np); fshift = np.fft.fftshift(f); magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1); magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); return Image.fromarray(magnitude_spectrum).convert('L')


# --- 4. THE FINAL ENSEMBLE PREDICTION FUNCTION ---

def get_ensemble_prediction(input_image: Image.Image):
    original_img_pil = input_image.convert('RGB')
    img_to_draw_on = np.array(original_img_pil.resize((224, 224)))

    # Define transformations
    rgb_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    fft_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    
    # Prepare all 4 data modalities
    ela_img = generate_ela(original_img_pil); res_img = generate_residual(original_img_pil); fft_img = generate_fft(original_img_pil)
    rgb_tensor = rgb_transform(original_img_pil).unsqueeze(0).to(DEVICE); ela_tensor = rgb_transform(ela_img).unsqueeze(0).to(DEVICE); res_tensor = rgb_transform(res_img).unsqueeze(0).to(DEVICE); fft_tensor = fft_transform(fft_img).unsqueeze(0).to(DEVICE)
    
    # Get predictions from both models
    with torch.no_grad():
        outputs_augmented, _ = model_augmented_cam(rgb_tensor, ela_tensor, res_tensor, fft_tensor)
        outputs_specialist = model_specialist(rgb_tensor, ela_tensor, res_tensor, fft_tensor)
    
    # Ensemble Logic: Average the probabilities
    probs_augmented = F.softmax(outputs_augmented, dim=1)[0]
    probs_specialist = F.softmax(outputs_specialist, dim=1)[0]
    avg_probs = (probs_augmented + probs_specialist) / 2.0
    
    prediction_idx = avg_probs.argmax().item()
    prediction_label = idx_to_class[prediction_idx].upper()
    confidences = {idx_to_class[i].upper(): f"{prob.item():.4f}" for i, prob in enumerate(avg_probs)}

    # --- Grad-CAM Heatmap Generation (if FAKE) ---
    if prediction_label == 'FAKE':
        try:
            # Re-run the CAM model with gradients enabled to get feature maps
            rgb_tensor.requires_grad = True
            output_cam, feature_maps = model_augmented_cam(rgb_tensor, ela_tensor, res_tensor, fft_tensor)
            
            score = output_cam[:, class_to_idx['fake']]
            score.backward() # This calculates gradients
            
            grads = torch.autograd.grad(score, feature_maps, retain_graph=True)[0]
            pooled_grads = torch.mean(grads, dim=[0, 2, 3], keepdim=True)
            
            feature_maps = feature_maps.detach()
            heatmap = torch.sum(feature_maps * pooled_grads, dim=1).squeeze()
            heatmap = np.maximum(heatmap.cpu().numpy(), 0)
            heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1.0
            
            heatmap = cv2.resize(heatmap, (224, 244))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            img_to_draw_on = cv2.addWeighted(img_to_draw_on, 0.5, heatmap, 0.5, 0)
        except Exception as e:
            print(f"ERROR: Could not generate Grad-CAM: {e}")

    # Add final prediction text
    color = (0, 0, 255) if "FAKE" in prediction_label else (0, 255, 0)
    text = f"Prediction: {prediction_label}"
    img_bgr = cv2.cvtColor(img_to_draw_on, cv2.COLOR_RGB2BGR)
    cv2.putText(img_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    final_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    result_pil = Image.fromarray(final_image_rgb)
    buff = io.BytesIO()
    result_pil.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return prediction_label, confidences, f"data:image/png;base64,{img_str}"


# --- 5. API ENDPOINT ---
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if not file.content_type.startswith("image/"): raise HTTPException(status_code=400, detail="File is not an image.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        prediction, confidence, result_image_str = get_ensemble_prediction(pil_image)
        return {"prediction": prediction, "confidence": confidence, "result_image": result_image_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
