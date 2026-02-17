# TimeSformer: Video Classification with Divided Space-Time Attention

A from-scratch PyTorch implementation of [TimeSformer](https://arxiv.org/abs/2102.05095) (Bertasius et al., ICML 2021) — a pure-transformer architecture that extends Vision Transformer (ViT) to video using **divided space-time attention**.

---

## The Core Idea

Traditional video models use 3D convolutions (C3D, SlowFast, I3D). TimeSformer asks: *can we use pure self-attention instead?*

The challenge: a video with T frames and N patches per frame has **T×N** tokens. Joint attention over all of them scales as **O((NF)²)** — impractical for real videos.

**The solution: Divided Space-Time Attention (T+S)**

Factorize attention into two sequential steps:

1. **Temporal attention**: each patch looks at the **same spatial position across all frames** → captures motion
2. **Spatial attention**: each patch looks at **all patches within the same frame** → captures appearance

This reduces complexity from O((NF)²) to **O(NF(N+F))** while achieving the best accuracy among 5 attention schemes.

---

## Architecture

```
Video (B, T, C, H, W)
    │
    ▼
Patch Embedding ──── Conv2d(3, 768, k=16, s=16)
    │
    ▼
(B, T, 196, 768)  ── + time_embed + space_embed
    │
    ▼
Prepend CLS token → (B, T+1, N, D)
    │
    ▼
╔═══════════════════════════════════════╗
║       ×12 TimeSformerBlock            ║
║                                       ║
║  1. Temporal Attn  (B*N, T, D)        ║
║       └─ + residual + LayerNorm       ║
║  2. Spatial Attn   (B*T, N, D)        ║
║       └─ + residual + LayerNorm       ║
║  3. MLP (dim→4*dim→dim)              ║
║       └─ + residual + LayerNorm       ║
╚═══════════════════════════════════════╝
    │
    ▼
CLS output x[:, 0, 0] → LayerNorm → Linear → logits
```

---

## The `einops.rearrange` Trick

The entire divided attention hinges on reshaping a 4D tensor into 3D for standard `nn.MultiheadAttention`:

### Temporal Attention
```python
xt = rearrange(x, 'b t n d -> (b n) t d')   # (B*N, T, D)
xt = self.temporal_attn(xt, xt, xt)[0]        # attention over T
xt = rearrange(xt, '(b n) t d -> b t n d', b=B, n=N)
x = x + self.norm1(xt)
```

Each of B×N patch positions creates a sequence of T tokens across time — captures **motion**.

### Spatial Attention
```python
xs = rearrange(x, 'b t n d -> (b t) n d')   # (B*T, N, D)
xs = self.spatial_attn(xs, xs, xs)[0]         # attention over N
xs = rearrange(xs, '(b t) n d -> b t n d', b=B, t=T)
x = x + self.norm2(xs)
```

Each of B×T frame instances creates a sequence of N tokens — captures **appearance**.

---

## CLS Token & Positional Embeddings

```python
self.cls_token   = nn.Parameter(torch.zeros(1, 1, 1, D))       # learnable summary token
self.time_embed  = nn.Parameter(torch.randn(1, T+1, D))        # +1 for CLS position
self.space_embed = nn.Parameter(torch.randn(1, N+1, D))        # +1 for CLS position
```

Each token receives a unique identity: `patch_content + time_embed[t] + space_embed[n]`

---

## Complexity Comparison

| Scheme | Per-Token Cost | Total Complexity | K400 Acc |
|--------|---------------|-----------------|----------|
| Space Only (S) | O(N) | O(TN²) | 75.2% |
| Joint (ST) | O(NF) | O((NF)²) | 77.9% |
| **Divided (T+S)** | **O(N+F)** | **O(NF(N+F))** | **78.0%** |
| Sparse L+G | varies | ~approx full | 75.7% |
| Axial (T+W+H) | O(T+W+H) | 3 × 1D | 76.7% |

With N=196, F=8: Divided does ~204 comparisons/token vs ~1,568 for Joint (7.7× cheaper).

---

## Transfer Learning

Spatial attention + MLP weights initialize from **pretrained ViT-Base** (ImageNet). Temporal attention learns from scratch. The model starts with strong spatial understanding and gradually acquires temporal reasoning.

---

## Usage

```python
model = TimeSformer(
    num_classes=2,
    num_frames=8,
    img_size=224,
    patch_size=16,
    embed_dim=768,
    depth=12,
    heads=12,
).to(device)

vit = timm.create_model('vit_base_patch16_224', pretrained=True)
load_vit_weights_into_timesformer(model, vit)

video = torch.randn(2, 8, 3, 224, 224)  # (batch, frames, C, H, W)
logits = model(video)  # (2, num_classes)
```

---

## Requirements

```
torch
torchvision
einops
timm
```

---

## References

- [Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095) (ICML 2021)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (ViT)
- [Facebook Research TimeSformer](https://github.com/facebookresearch/TimeSformer)
