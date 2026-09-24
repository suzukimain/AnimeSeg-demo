# AnimeSeg Demo Models

Models used by:
https://github.com/suzukimain/AnimeSeg-demo

Supported architectures:

| Folder | Architecture | Checkpoint | Classes | Input Size |
|---|---|---|:---:|:-----:|
| dinov2_unetpp/ | DINOv2 + U-Net++ | dinov2_unetpp.safetensors | 13 | 768 × 768 |
| oneformer/ | OneFormer | oneformer_dev.safetensors | 12 | 768 × 768 |
| thinhrnet_ocr/ | ThinHRNet + OCR | thinhrnet_ocr_dev.safetensors | 13 | 512 × 512 |
| afs/ | AFS U-Net | afs_unet_dev.safetensors | 12 | 1024 × 1024 |
| mask2former/ | Mask2Former | mask2former.safetensors | 12 | 768 × 768 |

config.json at the repository root lists the checkpoints. Each folder contains one safetensors file.
The _dev suffix marks development checkpoints.

## Example

![Animated five-model comparison](examples/test/comparison.gif)

[Color-mask comparison](examples/test/comparison_masks.png) | [Individual results](examples/test/README.md)
