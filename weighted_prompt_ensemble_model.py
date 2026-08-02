
# Weighted Prompt Ensemble for CLIP


#This method learns a weight for each handcrafted prompt template.
#CLIP remains frozen. Only the prompt weights are trainable.




import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class WeightedPromptEnsemble(nn.Module):
    def __init__(self, clip_model, classnames, templates, device):
        super().__init__()

        self.clip_model = clip_model
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.templates = templates
        self.device = device

        self.num_templates = len(templates)
        self.num_classes = len(classnames)

        # Learnable weight per prompt template
        self.prompt_logits = nn.Parameter(torch.zeros(self.num_templates))

        # Precompute all text embeddings:
        # shape = (num_templates, num_classes, feature_dim)
        self.register_buffer(
            "text_features",
            self._build_text_features()
        )

    @torch.no_grad()
    def _build_text_features(self):
        all_template_features = []

        for template in self.templates:
            prompts = [template.format(c) for c in self.classnames]
            tokens = clip.tokenize(prompts, truncate=True).to(self.device)

            text_features = self.clip_model.encode_text(tokens)
            text_features = F.normalize(text_features.float(), dim=-1)

            all_template_features.append(text_features)

        return torch.stack(all_template_features, dim=0)

    def get_weighted_text_features(self):
        
        # Returns:
        #    weighted text classifier of shape (feature_dim, num_classes)

        weights = F.softmax(self.prompt_logits, dim=0)

        weighted_features = (
            weights[:, None, None] * self.text_features
        ).sum(dim=0)

        weighted_features = F.normalize(weighted_features, dim=-1)

        return weighted_features.T

    def forward(self, images):

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features.float(), dim=-1)

        text_classifier = self.get_weighted_text_features()

        logits = 100.0 * image_features @ text_classifier

        return logits
