"""
common/clip_utils.py
====================
"""

import torch
import torch.nn.functional as F
import numpy as np
import clip  # pip install git+https://github.com/openai/CLIP.git

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _get_device()
print(f"[common.clip_utils] Using device: {DEVICE}")

# ---------------------------------------------
# The 80 official CLIP handcrafted prompt templates
# Source: openai/CLIP zero_shot_classifier
# ---------------------------------------------
IMAGENET_TEMPLATES = [
    'a photo of a {}.',
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def load_clip_model(model_name="ViT-B/16", device=DEVICE):
    """Load official CLIP model and preprocessing.

    CLIP's fp16 patching path only reliably targets CUDA. On MPS this can
    trigger dtype-mismatch errors or unsupported-op fallbacks, so we force
    fp32 on any non-CUDA device (MPS or CPU).
    """
    model, preprocess = clip.load(model_name, device=device, jit=False)
    if device != "cuda":
        model = model.float()
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_text_prompts(model, classnames, templates, device=DEVICE):
    """
    Build zero-shot text classifier weights.
    For each class, average embeddings across all templates.
    Returns: (n_classes, embed_dim) normalised tensor.
    """
    zeroshot_weights = []
    for classname in classnames:
        texts = [t.format(classname) for t in templates]
        tokens = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(tokens)          # (n_templates, D)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        class_embedding = class_embeddings.mean(dim=0)        # average
        class_embedding = F.normalize(class_embedding, dim=-1)
        zeroshot_weights.append(class_embedding)
    return torch.stack(zeroshot_weights, dim=1)               # (D, n_classes)


@torch.no_grad()
def encode_single_template(model, classnames, template, device=DEVICE):
    """
    Build text classifier weights for a single template string.
    Returns: (n_classes, embed_dim) normalised tensor.
    """
    texts = [template.format(c) for c in classnames]
    tokens = clip.tokenize(texts).to(device)
    embeddings = model.encode_text(tokens)                    # (n_classes, D)
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings.T                                       # (D, n_classes)


@torch.no_grad()
def encode_images(model, dataloader, device=DEVICE):
    """
    Extract image features from a DataLoader.
    Returns: (N, D) features, (N,) labels.
    """
    all_features, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        features = model.encode_image(images)
        features = F.normalize(features, dim=-1)
        all_features.append(features.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


def compute_accuracy(logits, labels, topk=(1, 5)):
    """Compute top-k accuracy from logits and labels."""
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = (correct_k * 100.0 / batch_size).item()
    return results


@torch.no_grad()
def zero_shot_eval(model, dataloader, text_weights, device=DEVICE):
    """
    Run zero-shot evaluation.
    text_weights: (D, n_classes) from encode_text_prompts.
    Returns top-1 and top-5 accuracy.
    """
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = 100.0 * image_features @ text_weights.to(device)
        all_preds.append(logits.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return compute_accuracy(logits, labels)
