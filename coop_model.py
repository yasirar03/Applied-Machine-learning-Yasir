# CoOp: Context Optimization for CLIP (Zhou et al., 2022)

# Instead of a handcrafted template ("a photo of a {}.") or a learned WEIGHT over several handcrafted templates (your exam model), CoOp replaces the template's words entirely with n_ctx continuous, learnable vectors that live directly in CLIP's text-embedding space:

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class PromptLearner(nn.Module):
    """Builds the [prefix][learnable ctx][class tokens][suffix] embedding
    sequence fed into CLIP's text transformer, and owns the trainable
    context vectors."""

    def __init__(self, clip_model, classnames, n_ctx=16, ctx_init=None,
                 class_specific_context=False, device="cpu"):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # Initialise the learnable context 
            ctx_init = ctx_init.replace("_", " ").strip()
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :].clone()
        else:
            if class_specific_context:
                # CSC: a separate set of context vectors per class                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)  # the only trainable parameters

        classnames = [c.replace("_", " ") for c in classnames]
        prompts = [f"{'X ' * n_ctx}{name}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Fixed (non-trainable) parts of the sequence
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.class_specific_context = class_specific_context
        self.tokenized_prompts = tokenized_prompts  # needed to find each EOS position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class TextEncoder(nn.Module):
    """Thin wrapper around CLIP's frozen text transformer so it can consume
    embeddings directly (built by PromptLearner) instead of token ids."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.token_embedding.weight.dtype

    def forward(self, prompt_embeddings, tokenized_prompts):
        x = prompt_embeddings + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)          # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)          # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # Take the feature at each sequence's EOS token position
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CoOpCLIP(nn.Module):
    """Full CoOp model: frozen image encoder + frozen text transformer +
    learnable context vectors."""

    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init=None,
                 class_specific_context=False, device="cpu"):
        super().__init__()
        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx, ctx_init, class_specific_context, device
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.token_embedding.weight.dtype

    def get_text_classifier(self):
        """(D, n_cls) text classifier for the CURRENT context vectors.
        Must be recomputed after every optimiser step since ctx changes -
        unlike the exam model, there's no fixed buffer to reuse here."""
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = F.normalize(text_features.float(), dim=-1)
        return text_features.T

    def forward(self, images=None, image_features=None):
        """Accepts EITHER raw images (encodes them, frozen/no_grad) OR
        pre-cached image_features (skips encoding entirely) - the latter is
        what the training script uses for speed, since the image encoder
        never changes during CoOp training."""
        if image_features is None:
            with torch.no_grad():
                image_features = self.image_encoder(images.type(self.dtype))
                image_features = F.normalize(image_features.float(), dim=-1)

        text_classifier = self.get_text_classifier()
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_classifier
