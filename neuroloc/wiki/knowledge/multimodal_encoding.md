# Multimodal Encoding for Unified Architectures

This document covers techniques for encoding different modalities (images,
audio, 3D data) into shared token sequences for processing by a unified
transformer backbone.


## 1. ViT-Style Patch Embeddings for Images

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
ArXiv ID: 2010.11929
Authors: Alexey Dosovitskiy et al. (Google Brain)

Mechanism:
1. Split image (H x W x 3) into non-overlapping patches of size P x P
   - Standard: P = 16, so a 224x224 image yields 14x14 = 196 patches
2. Flatten each patch: P * P * 3 = 768 values (for P=16, RGB)
3. Linear projection: flatten -> d_model via a learned linear layer
   - This is the "patch embedding" layer
   - Equivalent to a 2D convolution with kernel_size=P, stride=P
4. Prepend a learnable [CLS] token embedding
5. Add learnable 1D positional embeddings (one per patch position + CLS)

Key design decisions:
- 1D positional embeddings work as well as 2D-aware ones (validated in paper)
- The model learns spatial structure from data, not from architectural inductive bias
- Patch size P controls the sequence length: L = (H/P) * (W/P)
- Standard configurations: ViT-B (d=768), ViT-L (d=1024), ViT-H (d=1280)

For variable resolution (MSPE, arxiv 2405.18240):
- Multi-Scale Patch Embedding uses multiple patch sizes simultaneously
- Patches at different scales are concatenated into a single sequence
- Enables resolution-agnostic processing without retraining


## 2. Spectrogram Patch Embeddings for Audio

Paper: "AST: Audio Spectrogram Transformer"
ArXiv ID: 2104.01778
Authors: Yuan Gong, Yu-An Chung, James Glass (MIT)

Mechanism:
1. Convert audio waveform to log Mel spectrogram:
   - Standard: 128 Mel bins, 10ms hop size
   - Result: 2D time-frequency representation (T_frames x 128)
2. Split spectrogram into overlapping patches:
   - Patch size: 16 x 16 (time x frequency)
   - Stride: fstride=10, tstride=10 (overlap of 6 in both dimensions)
   - Overlap is critical: unlike images, spectrograms have continuous
     frequency and time structure that benefits from patch overlap
3. Flatten and linearly project each patch to d_model (768)
4. Add learnable positional embeddings (one per patch position)
5. Prepend [CLS] token

Key differences from image patches:
- Patches overlap (stride < patch_size) due to spectrogram structure
- 2D structure encodes different physical quantities (time vs frequency)
- ImageNet-pretrained ViT weights transfer well to audio spectrograms
  (cross-modal transfer from images to spectrograms)
- For variable-length audio: adjust number of time-dimension patches

Results: 0.485 mAP on AudioSet, 95.6% on ESC-50, 98.1% on Speech Commands V2


## 3. Meta-Transformer: Unified Modality Encoding

Paper: "Meta-Transformer: A Unified Framework for Multimodal Learning"
ArXiv ID: 2307.10802
Authors: Yiyuan Zhang et al. (CUHK / OpenGVLab)

Framework:
1. Modality-specialist tokenizers (different per modality):
   - Images: ViT-style patch embedding
   - Audio: log Mel filterbank -> spectrogram patches (with overlap)
   - Text: standard subword tokenization
   - Point clouds: Farthest Point Sampling -> local patch grouping
   - Other modalities: appropriate domain-specific preprocessing

2. Modality-shared encoder:
   - Single frozen ViT backbone (pretrained on LAION-2B via contrastive learning)
   - Processes token sequences from ALL modalities with the same parameters
   - Key insight: the same transformer weights can extract meaningful
     features from diverse modalities once properly tokenized

3. Task-specific heads:
   - Classification, detection, segmentation, etc.
   - Lightweight per-task decoders

Audio-specific processing:
- Pre-process with log Mel filterbank for fixed duration
- Hamming window with stride on frequency dimension
- Overlapping patch splitting on the spectrogram
- Flatten patches into token sequence

Point cloud processing:
- Farthest Point Sampling to select N center points
- k-NN grouping around each center (local patches)
- Each local patch is encoded into a token embedding


## 4. 4M: Massively Multimodal Masked Modeling

Paper: "4M: Massively Multimodal Masked Modeling"
ArXiv ID: 2312.06647
Authors: EPFL (David Mizrahi et al.)
Published: NeurIPS 2023

Mechanism:
1. Tokenize all modalities into discrete tokens:
   - Each modality has its own tokenizer (VQ-VAE or modality-specific)
   - The tokenizer converts continuous signals to sequences of discrete indices
   - All modality tokens share the same vocabulary structure

2. Unified transformer encoder-decoder:
   - Operates on a small randomized subset of tokens from all modalities
   - Masked modeling objective: predict masked tokens from visible ones
   - Any-to-any generation: can predict any modality from any other

3. Supported modalities include:
   - RGB images, depth, surface normals, semantic segmentation
   - CLIP features, DINOv2 features, text
   - Geometric modalities (depth, normals) and semantic modalities

Key insight: By tokenizing everything into discrete sequences, the model
treats all modalities uniformly. The only modality-specific component is
the tokenizer.

4M-21 extension (2024) scales to 21 modalities with the same approach.


## 5. Multivector Projection for 3D Data

From the Echoloc project and GATr framework (arxiv 2305.18415):

For 3D geometric data, standard patch embeddings lose geometric structure.
The Geometric Algebra approach encodes 3D data as multivectors in G(3,0,1):

1. Points: encoded as grade-1 vectors (x*e1 + y*e2 + z*e3 + e0)
2. Directions: encoded as grade-1 vectors without e0 component
3. Planes: encoded via dual representation (grade-3 trivectors)
4. Lines: encoded as grade-2 bivectors

Multivector projection:
- Each 3D geometric primitive maps to a 16-component multivector
- The mapping preserves geometric structure and E(3) equivariance
- A learned equivariant linear map (9 parameters per channel pair)
  projects multivectors to the model's hidden dimension
- Grade-wise RMS normalization maintains grade structure

This approach is superior to naive flattening of 3D coordinates because:
- Rotations, translations, and reflections act naturally on multivectors
- The model inherits geometric symmetries without data augmentation
- Different geometric types (points, lines, planes) coexist in one representation


## 6. Sharing Token Sequences Across Modalities

The general pattern for unified multimodal processing:

    +-----------+     +-----------+     +-----------+
    | Image     |     | Audio     |     | 3D Points |
    +-----------+     +-----------+     +-----------+
         |                 |                 |
    [patch embed]    [spec patches]    [MV project]
         |                 |                 |
    [d_model tokens] [d_model tokens] [d_model tokens]
         |                 |                 |
         +--------+--------+--------+--------+
                  |                  |
           [modality tokens]   [special tokens]
                  |                  |
                  v                  v
         [  Shared Transformer Backbone  ]
                       |
                  [task heads]

Key design decisions:
1. Modality tokens/tags: prepend modality-identifying tokens to distinguish
   which modality each token came from
2. Positional embeddings: can be shared or separate per modality
   - Shared: simpler, works if all modalities use similar sequence lengths
   - Separate: more flexible, required if modalities have different structures
3. Cross-modal attention: all tokens attend to all other tokens
4. Sequence packing: concatenate tokens from multiple modalities into one sequence

Sequence length management:
- Images (224x224, P=16): 196 tokens
- Audio (10s at 16kHz, 128 Mel): ~60 tokens (with overlap)
- 3D point clouds (1024 points, k=32): 32-64 tokens
- Text: variable (tokenizer-dependent)
- Total sequence: sum of all modality tokens + special tokens


## References

- ViT: arxiv 2010.11929
- AST: arxiv 2104.01778
- Meta-Transformer: arxiv 2307.10802
- 4M: arxiv 2312.06647
- GATr: arxiv 2305.18415
- MSPE: arxiv 2405.18240
- Echoloc project: eptesicuslabs/echoloc
