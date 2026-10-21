"""Shared bits: class names, model builder, image transforms."""
import torch
import torchvision
from torchvision import transforms as T

CLASSES = ["clean", "messy", "no_surf"]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch="convnext_tiny", pretrained=True, num_classes=len(CLASSES)):
    """ImageNet-pretrained backbone with a new 3-class head."""
    if arch == "convnext_tiny":
        w = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = torchvision.models.convnext_tiny(weights=w)
        m.classifier[2] = torch.nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == "resnet50":
        w = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet18":
        w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        m = torchvision.models.resnet18(weights=w)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def train_transform():
    # Rotations/flips are safe: a clean peeling line is still clean at any
    # orientation, and it stops the model learning "waves only come from the west".
    # Colour jitter covers different light, haze and sun glint between days.
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.7),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.RandomGrayscale(0.1),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
