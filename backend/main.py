from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch
import io
import base64
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import AutoEncoder, add_rician_noise

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(in_channels=1, height=512, width=512)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.to(device)
model.eval()

model_post = AutoEncoder(in_channels=1, height=512, width=512)
model_post.load_state_dict(torch.load(
    "saved_model_post.pt", map_location=device))
model_post.to(device)
model_post.eval()


def extract_feature_maps(model, tensor, layer_index):
    """
    Extract feature maps from a specific layer in the model.

    Parameters:
    - model: The autoencoder model.
    - tensor: Input tensor (batch of images).
    - layer_index: Index of the layer to extract feature maps from.

    Returns:
    - feature_maps: Feature maps from the specified layer.
    """
    # Hook to extract feature maps
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu().numpy())

    # Register hook to the specified layer
    layer = list(model.encoder.children())[layer_index]
    hook = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        model(tensor)

    # Remove hook
    hook.remove()

    # Return the feature maps for the first image in the batch
    return feature_maps[0]


def normalize_and_convert_to_image(feature_maps, num_maps=1):
    """
    Normalize feature maps and convert them to an image format.

    Parameters:
    - feature_maps: Feature maps to normalize and convert.
    - num_maps: Number of feature maps to visualize (default: 1).

    Returns:
    - feature_maps_pil: PIL image of the feature maps.
    """
    # Select the first `num_maps` feature maps
    # Shape: (num_maps, height, width)
    feature_maps = feature_maps[0, :num_maps]

    # Normalize each feature map to [0, 1]
    feature_maps = (feature_maps - feature_maps.min()) / \
        (feature_maps.max() - feature_maps.min())

    # Convert to uint8 and create PIL image
    feature_maps = (feature_maps * 255).astype(np.uint8)

    # Create a grid of feature maps
    fig, axes = plt.subplots(1, num_maps, figsize=(10.24, 5.12))
    if num_maps == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(feature_maps[i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
    plt.close()
    buf.seek(0)

    # Convert buffer to PIL image
    feature_maps_pil = Image.open(buf)
    return feature_maps_pil


@app.post("/noisy-image")
async def add_noise_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = np.array(image, dtype=np.float32)
    image = image / 255

    # Add Rician noise with high and low SNR
    high_snr = add_rician_noise(image, 6)  # High SNR
    low_snr = add_rician_noise(image, 1)   # Low SNR

    # Convert noisy images back to PIL Images
    high_snr_pil = Image.fromarray((high_snr * 255).astype(np.uint8))
    low_snr_pil = Image.fromarray((low_snr * 255).astype(np.uint8))

    # Save the noisy images to buffers
    buf_high = io.BytesIO()
    high_snr_pil.save(buf_high, format="PNG")
    encoded_high_snr = base64.b64encode(buf_high.getvalue()).decode("utf-8")

    buf_low = io.BytesIO()
    low_snr_pil.save(buf_low, format="PNG")
    encoded_low_snr = base64.b64encode(buf_low.getvalue()).decode("utf-8")

    # Return the URLs of the noisy images
    return JSONResponse(content={
        "high_snr_url": f"data:image/png;base64,{encoded_high_snr}",
        "low_snr_url": f"data:image/png;base64,{encoded_low_snr}"
    })


@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert to grayscale and normalize
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = np.array(image, dtype=np.float32) / 255.0

    # Convert to PyTorch tensor and move to correct device
    tensor = torch.from_numpy(image).unsqueeze(
        0).unsqueeze(0).to(device, dtype=torch.float32)

    # Run model inference
    with torch.no_grad():
        output_model1 = model(tensor).squeeze().cpu().numpy()

    with torch.no_grad():
        output_model2 = model_post(tensor).squeeze().cpu().numpy()

    # Convert output to image format
    processed_image_model1 = (output_model1 * 255).astype(np.uint8)
    processed_image_model2 = (output_model2 * 255).astype(np.uint8)

    processed_pil_model1 = Image.fromarray(processed_image_model1)
    processed_pil_model2 = Image.fromarray(processed_image_model2)

    # Save reconstructed images to buffers
    buf_model1 = io.BytesIO()
    processed_pil_model1.save(buf_model1, format="PNG")
    encoded_img_model1 = base64.b64encode(
        buf_model1.getvalue()).decode("utf-8")

    buf_model2 = io.BytesIO()
    processed_pil_model2.save(buf_model2, format="PNG")
    encoded_img_model2 = base64.b64encode(
        buf_model2.getvalue()).decode("utf-8")

    # Extract feature maps from the first model
    layer_index = 0  # Change this to visualize different layers
    feature_maps = extract_feature_maps(model, tensor, layer_index)

    # Normalize and convert feature maps to image format
    feature_maps_pil = normalize_and_convert_to_image(
        feature_maps, num_maps=4)  # Visualize 4 feature maps

    # Save feature maps to buffer
    buf_feature = io.BytesIO()
    feature_maps_pil.save(buf_feature, format="PNG")
    encoded_feature = base64.b64encode(buf_feature.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "image_url_model1": f"data:image/png;base64,{encoded_img_model1}",
        "image_url_model2": f"data:image/png;base64,{encoded_img_model2}",
        "feature_maps_url": f"data:image/png;base64,{encoded_feature}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
