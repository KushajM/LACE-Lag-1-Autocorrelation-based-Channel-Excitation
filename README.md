# LACE-Lag-1-Autocorrelation-based-Channel-Excitation

##Method Abstract
###Channel attention modules in Convolutional Neural Networks (CNNs) reweight each channel by a scalar read off some descriptor of the activation map. The descriptors used in practice: mean, standard deviation, maximum, are all marginal statistics, which means a coherent feature map and a noisy feature map with the same value distribution receive identical weights. Two of these statistics also collapse to learned constants under Batch Normalization (BN), leaving the gate measuring BN parameters instead of the input. Our proposed methodology **LACE (Lag-1 Autocorrelation-based Channel Excitation)** uses a different scalar: the lag-1 spatial Pearson autocorrelation of the channel, which is Moran's I specialized to a regular grid. This quantity is sensitive to where activations sit on the grid, has a frequency-domain reading via the Wiener-Khinchin theorem, and is preserved by the affine transform BN applies. Variance and spatial coherence move in opposite directions under noise and blur, hence we combine them multiplicatively into an indicator that fires only when both are unfavorable simultaneously, and use it alongside the channel mean and variance in a per-channel linear gate. LACE is a drop-in replacement for popularly used channel attention modules in the literature such as SE, CBAM, SRM, and ECA at the same cost as global average pooling. Tested on three CNN backbones (ResNet-18, DenseNet-BC-100-12, MobileNetV2) and three datasets (CIFAR-10, GTSRB MiniImageNet), it improves mean-corruption accuracy without reducing clean accuracy.

##Method Architecture
<img width="1151" height="460" alt="ChannelExcitation drawio (1) (1)" src="https://github.com/user-attachments/assets/096fa705-760e-4fbe-930f-5af8d53e8c9c" />

## Robustness Evaluation (Accuracy) on CIFAR-10

### ResNet18

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 93.03 | 50.25 | 73.60 | 86.34 | 79.15 | 74.46 |
| SE | 94.09 | 50.61 | 72.94 | 85.65 | 78.82 | 74.08 |
| CBAM | 94.11 | 52.96 | 74.82 | 86.32 | 78.72 | 72.21 |
| SRM | 94.36 | 57.67 | 78.89 | 87.32 | 78.89 | 77.25 |
| ECA-Net | 94.00 | 55.89 | 74.71 | 84.23 | 77.74 | 74.67 |
| FCA-Net | 93.29 | 53.68 | 73.56 | 81.45 | 73.23 | 72.53 |
| SCSA | 93.37 | 57.45 | 76.71 | 87.56 | 80.56 | 76.06 |
| ELA | 93.33 | 54.68 | 72.69 | 80.24 | 72.45 | 72.86 |
| **LACE** | **95.32** | **65.56** | **80.44** | **88.44** | **81.87** | **79.87** |

### DenseNet-BC-100-12

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 93.45 | 52.92 | 63.69 | 80.16 | 74.71 | 68.87 |
| SE | 93.85 | 54.91 | 66.09 | 81.95 | 76.00 | 70.73 |
| CBAM | 93.87 | 56.36 | 69.33 | 82.16 | 75.20 | 71.72 |
| SRM | 93.91 | 56.91 | 68.77 | 81.82 | 74.81 | 71.49 |
| ECA-Net | 94.19 | 56.38 | 71.19 | 82.50 | 77.02 | 72.80 |
| FCA-Net | 93.56 | 55.98 | 70.25 | 80.56 | 75.06 | 71.29 |
| SCSA | 93.22 | 55.28 | 67.56 | 80.26 | 74.24 | 70.07 |
| ELA | 94.15 | 54.23 | 68.78 | 80.12 | 74.56 | 70.75 |
| **LACE** | **94.98** | **59.47** | **69.72** | **83.72** | **75.54** | **72.96** |

### MobileNetV2

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 92.49 | 42.24 | 67.40 | 80.76 | 74.24 | 69.48 |
| SE | 92.44 | 38.55 | 68.25 | 82.42 | 75.12 | 70.71 |
| CBAM | 93.40 | 42.45 | 70.46 | 81.34 | 75.59 | 69.90 |
| SRM | 94.34 | 53.46 | 72.74 | 85.93 | 77.81 | 76.19 |
| ECA-Net | 94.03 | 53.58 | 72.47 | 83.65 | 76.84 | 73.72 |
| FCA-Net | 93.38 | 52.69 | 70.48 | 80.45 | 75.80 | 72.32 |
| SCSA | 93.18 | 52.58 | 69.67 | 80.12 | 70.82 | 69.67 |
| ELA | 93.18 | 51.45 | 70.47 | 79.65 | 70.84 | 70.79 |
| **LACE (Ours)** | **95.15** | **62.72** | **78.06** | **85.10** | **80.56** | **78.33** |




---

## Robustness Evaluation (Accuracy) on GTSRB

### ResNet18

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 99.30 | 71.56 | 79.10 | 71.04 | 83.19 | 76.22 |
| SE | 99.20 | 73.88 | 78.57 | 74.38 | 84.41 | 77.81 |
| CBAM | 99.11 | 64.35 | 79.41 | 71.90 | 82.46 | 74.53 |
| SRM | 98.52 | 68.99 | 79.81 | 67.53 | 82.16 | 74.62 |
| ECA-Net | 99.49 | 74.57 | 79.41 | 74.12 | 85.15 | 78.31 |
| FCA-Net | 99.38 | 68.78 | 79.37 | 76.19 | 84.66 | 77.25 |
| SCSA | 99.26 | 65.27 | 75.31 | 72.30 | 80.33 | 73.30 |
| ELA | 99.24 | 68.92 | 79.52 | 76.34 | 84.82 | 77.40 |
| **LACE** | **99.89** | **77.44** | **79.81** | **76.99** | **88.96** | **80.80** |

### DenseNet-BC-100-12

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 98.73 | 58.87 | 71.04 | 67.85 | 75.33 | 68.27 |
| SE | 98.82 | 62.94 | 75.31 | 70.41 | 78.47 | 71.78 |
| CBAM | 98.33 | 52.80 | 76.38 | 67.43 | 78.54 | 68.79 |
| SRM | 98.35 | 65.54 | 78.51 | 68.28 | 82.54 | 73.72 |
| ECA-Net | 98.13 | 63.77 | 73.18 | 71.39 | 78.90 | 71.81 |
| FCA-Net | 98.39 | 64.86 | 74.84 | 71.84 | 79.82 | 72.84 |
| SCSA | 98.96 | 66.18 | 76.37 | 73.31 | 81.46 | 74.33 |
| ELA | 98.84 | 63.33 | 73.08 | 70.16 | 77.95 | 71.13 |
| **LACE (Ours)** | **99.18** | **71.57** | **79.98** | **73.28** | **85.33** | **77.54** |

### MobileNetV2

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 98.78 | 71.51 | 73.02 | 70.11 | 79.04 | 73.42 |
| SE | 99.03 | 68.03 | 75.82 | 73.48 | 81.76 | 74.77 |
| CBAM | 98.94 | 67.52 | 73.58 | 68.61 | 79.25 | 72.24 |
| SRM | 97.85 | 67.18 | 71.37 | 66.56 | 80.61 | 71.43 |
| ECA-Net | 99.18 | 64.98 | 75.29 | 70.19 | 81.21 | 72.92 |
| FCA-Net | 99.04 | 66.62 | 76.87 | 73.80 | 81.99 | 74.82 |
| SCSA | 98.87 | 67.03 | 77.34 | 74.25 | 82.50 | 75.28 |
| ELA | 99.01 | 68.41 | 78.93 | 75.78 | 84.20 | 76.83 |
| **LACE (Ours)** | **99.30** | **75.01** | **78.64** | **76.10** | **86.33** | **79.02** |


## Robustness Evaluation (Accuracy) on MiniImageNet

### ResNet18

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 52.85% | 15.48% | 18.26% | 24.73% | 28.81% | 21.82% |
| SE | 52.75% | 15.97% | 19.63% | 26.32% | 31.56% | 23.37% |
| CBAM | 52.27% | 16.02% | 19.02% | 25.03% | 31.06% | 22.78% |
| SRM | 52.48% | 19.78% | 19.20% | 25.22% | 30.24% | 23.61% |
| ECA-Net | 52.02% | 15.85% | 19.50% | 26.10% | 29.83% | 22.82% |
| FCA | 52.53% | 15.11% | 18.89% | 25.98% | 30.42% | 22.60% |
| SCSA | 52.06% | 17.19% | 19.50% | 25.25% | 30.67% | 23.15% |
| ELA | 51.86% | 14.46% | 19.29% | 25.05% | 30.80% | 22.40% |
| **LACE (Ours)** | **53.78%** | 18.32% | 20.48% | 26.87% | 32.08% | **24.44%** |

### DenseNet-BC-100-12

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 55.24% | 14.92% | 19.57% | 26.44% | 30.43% | 22.84% |
| SE | 55.25% | 14.47% | 20.11% | 26.83% | 31.23% | 23.16% |
| CBAM | 56.07% | 14.40% | 20.14% | 27.17% | 31.45% | 23.29% |
| SRM | 50.13% | 15.89% | 19.31% | 25.24% | 29.32% | 22.44% |
| ECA-Net | 54.59% | 15.14% | 21.09% | 27.80% | 31.57% | 23.90% |
| FCA | 56.10% | 14.35% | 19.99% | 27.22% | 31.04% | 23.15% |
| SCSA | 55.69% | 14.93% | 19.97% | 26.31% | 31.03% | 23.06% |
| ELA | 53.24% | 13.39% | 20.01% | 25.93% | 30.71% | 22.51% |
| **LACE (Ours)** | **57.56%** | 16.51% | 21.09% | 28.04% | 32.28% | **24.48%** |

### MobileNetV2

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|:-----:|:-----:|:----:|:-------:|:-------:|:------:|
| Baseline | 49.75% | 11.53% | 18.47% | 24.86% | 28.02% | 20.72% |
| SE | 50.53% | 10.81% | 18.69% | 24.76% | 27.51% | 20.44% |
| CBAM | 50.27% | 9.79% | 18.74% | 26.20% | 27.24% | 20.49% |
| SRM | 50.28% | 10.22% | 19.90% | 25.82% | 28.55% | 21.12% |
| ECA-Net | 49.62% | 11.23% | 19.60% | 25.22% | 28.35% | 21.10% |
| FCA | 50.32% | 11.06% | 18.71% | 25.07% | 27.36% | 20.55% |
| SCSA | 50.92% | 10.43% | 19.38% | 26.24% | 29.55% | 21.40% |
| ELA | 49.68% | 10.76% | 19.67% | 25.41% | 28.33% | 21.04% |
| **LACE (Ours)** | **52.26%** | 12.88% | 21.15% | 28.69% | 31.20% | **23.48%** |

