import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

def hook_fn(module, input, output):
    intermediate_features.append(output)

def extract_features(model, img, layer_index=20):
    global intermediate_features
    intermediate_features = []
    hook = model.model.model[layer_index].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img)
    hook.remove()
    return intermediate_features[0]

model = YOLO("yolov8_trained.pt")

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)
    
    return img

img = preprocess_image("test_data/2.jpg")
features = extract_features(model, img)
print(features.shape)
print(features)