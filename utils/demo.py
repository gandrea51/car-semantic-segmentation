from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dataset.transform import preprocess, postprocess
from dataset.test import get_model
#from utils.llm import build_report, generate_explanation
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import numpy as np
import torch, cv2, time, base64

PALETTE = np.array([[0,0,0], [34,139,34], [205,133,63], [255,215,0], [0,191,255]], dtype=np.uint8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIT_PATH = './network/mit.pth'
MIT_ENCODER = 'mit_b2'
DEEPLAB_PATH = './network/deeplab.pth'
DEEPLAB_ENCODER = "resnet18"
RESNEXT_PATH = "./network/resnext.pth"
RESNEXT_ENCODER = "resnext50_32x4d"

CLASSES = 5

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

models = {
    "unetxmit": get_model(smp.Unet, MIT_PATH, DEVICE, encoder_name=MIT_ENCODER, encoder_weights=None, in_channels=3, classes=CLASSES),
    "deeplab": get_model(smp.DeepLabV3Plus, DEEPLAB_PATH, DEVICE, encoder_name=DEEPLAB_ENCODER, encoder_weights=None, in_channels=3, classes=CLASSES),    
    "unetxresnext": get_model(smp.Unet, RESNEXT_PATH, DEVICE, encoder_name=RESNEXT_ENCODER, encoder_weights=None, in_channels=3, classes=CLASSES)
}
for m in models.values():
    m.eval()

def colorize(mask):
    return PALETTE[mask]

def overlay(image, mask, alpha=0.5):
    return np.clip(image * (1 - alpha) + mask / 255.0 * alpha, 0, 1)

def normalize_map(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)

def encode_png(image):
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return None
    return base64.b64encode(buffer.tobytes()).decode("utf-8")

@app.get("/")
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    if model_name not in models:
        return JSONResponse(content={"error": "Invalid model."}, status_code=400)

    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File must be an image."}, status_code=400)

    try:
        contents = await file.read()

        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(content={"error": "Invalid image."}, status_code=400)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = preprocess(image, DEVICE)
        model = models[model_name]

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            conf = torch.max(probs, dim=1).values
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        pred_mask = preds.squeeze(0).cpu().numpy()
        conf_map = conf.squeeze(0).cpu().numpy()
        entropy_map = entropy.squeeze(0).cpu().numpy()

        h, w = image.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)
        entropy_map = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_LINEAR)

        #report = build_report(pred_mask, conf_map, entropy_map)
        #explanation = generate_explanation(report, model_name)

        mask_color = colorize(pred_mask)
        image_float = image.astype(np.float32) / 255.0
        over = overlay(image_float, mask_color)
        over = (over * 255).astype(np.uint8)

        # Confidence
        conf_norm = normalize_map(conf_map)
        conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_INFERNO)
        conf_color = cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)

        # Entropy
        entropy_norm = normalize_map(entropy_map)
        entropy_color = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_MAGMA)
        entropy_color = cv2.cvtColor(entropy_color, cv2.COLOR_BGR2RGB)

        return {
            "overlay": encode_png(cv2.cvtColor(over, cv2.COLOR_RGB2BGR)),
            "confidence": encode_png(cv2.cvtColor(conf_color, cv2.COLOR_RGB2BGR)),
            "entropy": encode_png(cv2.cvtColor(entropy_color, cv2.COLOR_RGB2BGR)),
            "model_used": model_name
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/metrics")
def metrics():
    gpu_allocated = 0
    gpu_reserved = 0
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated()/1024**2
        gpu_reserved = torch.cuda.memory_reserved()/1024**2
    
    return {
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys()),
        "num_classes": CLASSES,
        "input_size": [512, 512],
        "gpu_memory_allocated_mb": round(gpu_allocated, 2),
        "gpu_memory_reserved_mb": round(gpu_reserved, 2),
    }