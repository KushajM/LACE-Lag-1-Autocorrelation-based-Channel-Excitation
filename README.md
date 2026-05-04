# LACE-Lag-1-Autocorrelation-based-Channel-Excitation

##Method Abstract
###Channel attention modules in Convolutional Neural Networks (CNNs) reweight each channel by a scalar read off some descriptor of the activation map. The descriptors used in practice: mean, standard deviation, maximum, are all marginal statistics, which means a coherent feature map and a noisy feature map with the same value distribution receive identical weights. Two of these statistics also collapse to learned constants under Batch Normalization (BN), leaving the gate measuring BN parameters instead of the input. Our proposed methodology **LACE (Lag-1 Autocorrelation-based Channel Excitation)** uses a different scalar: the lag-1 spatial Pearson autocorrelation of the channel, which is Moran's I specialized to a regular grid. This quantity is sensitive to where activations sit on the grid, has a frequency-domain reading via the Wiener-Khinchin theorem, and is preserved by the affine transform BN applies. Variance and spatial coherence move in opposite directions under noise and blur, hence we combine them multiplicatively into an indicator that fires only when both are unfavorable simultaneously, and use it alongside the channel mean and variance in a per-channel linear gate. LACE is a drop-in replacement for popularly used channel attention modules in the literature such as SE, CBAM, SRM, and ECA at the same cost as global average pooling. Tested on three CNN backbones (ResNet-18, DenseNet-BC-100-12, MobileNetV2) and three datasets (CIFAR-10, GTSRB MiniImageNet), it improves mean-corruption accuracy without reducing clean accuracy.

##Method Architecture
<img width="1151" height="460" alt="ChannelExcitation drawio (1) (1)" src="https://github.com/user-attachments/assets/096fa705-760e-4fbe-930f-5af8d53e8c9c" />

## Robustness Evaluation (Accuracy) on CIFAR-10 and GTSRB
### MobileNetV2

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|-------|-------|------|---------|---------|--------|-------|-------|------|---------|---------|--------|
| | **CIFAR-10** | | | | | | **GTSRB** | | | | | |
| Baseline | 0.9249 | 0.4224 | 0.6740 | 0.8076 | 0.7424 | 0.6775 | 0.9935 | 0.7375 | 0.7531 | 0.7231 | 0.8152 | 0.7342 |
| SE | 0.9244 | 0.3855 | 0.6825 | 0.8242 | 0.7512 | 0.6792 | 0.9903 | 0.6741 | 0.7513 | 0.7281 | 0.8102 | 0.7477 |
| CBAM | 0.9340 | 0.4695 | 0.7147 | 0.8334 | 0.7659 | 0.7110 | 0.9894 | 0.6972 | 0.7598 | 0.7084 | 0.8183 | 0.7224 |
| SRM | 0.9352 | 0.5346 | 0.7274 | 0.8593 | 0.7781 | 0.7375 | 0.9785 | 0.6764 | 0.7186 | 0.6702 | 0.8117 | 0.7143 |
| ECA-Net | 0.9339 | 0.4158 | 0.7047 | 0.8365 | 0.7684 | 0.6991 | 0.9918 | 0.6561 | 0.7602 | 0.7087 | 0.8199 | 0.7292 |
| **Proposed** | 0.9342 | 0.5672 | 0.7206 | 0.8510 | 0.7743 | **0.7390** | 0.9930 | 0.7182 | 0.7529 | 0.7286 | 0.8266 | **0.7603** |

### DenseNet-BC-100-12

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|-------|-------|------|---------|---------|--------|-------|-------|------|---------|---------|--------|
| | **CIFAR-10** | | | | | | **GTSRB** | | | | | |
| Baseline | 0.9345 | 0.5292 | 0.6369 | 0.8016 | 0.7471 | 0.6887 | 0.9916 | 0.6110 | 0.7373 | 0.7042 | 0.7818 | 0.7159 |
| SE | 0.9385 | 0.5491 | 0.6609 | 0.8195 | 0.7600 | 0.7073 | 0.9882 | 0.6231 | 0.7455 | 0.6970 | 0.7768 | 0.7178 |
| CBAM | 0.9387 | 0.5636 | 0.6933 | 0.8216 | 0.7520 | 0.7172 | 0.9833 | 0.5187 | 0.7503 | 0.6624 | 0.7715 | 0.6879 |
| SRM | 0.9391 | 0.5691 | 0.6877 | 0.8182 | 0.7481 | 0.7149 | 0.9835 | 0.6469 | 0.7749 | 0.6739 | 0.8147 | 0.7372 |
| ECA-Net | 0.9419 | 0.5638 | 0.7119 | 0.8250 | 0.7702 | 0.7280 | 0.9813 | 0.6323 | 0.7256 | 0.7079 | 0.7824 | 0.7181 |
| **Proposed** | 0.9408 | 0.5947 | 0.6972 | 0.8372 | 0.7554 | **0.7296** | 0.9918 | 0.7090 | 0.7923 | 0.7259 | 0.8453 | **0.7754** |

### ResNet18

| Method | Clean | Noise | Blur | Weather | Digital | Mean-C | Clean | Noise | Blur | Weather | Digital | Mean-C |
|--------|-------|-------|------|---------|---------|--------|-------|-------|------|---------|---------|--------|
| | **CIFAR-10** | | | | | | **GTSRB** | | | | | |
| Baseline | 0.9491 | 0.4925 | 0.7360 | 0.8634 | 0.7915 | 0.7361 | 0.9930 | 0.7091 | 0.7838 | 0.7039 | 0.8243 | 0.7622 |
| SE | 0.9508 | 0.5061 | 0.7294 | 0.8565 | 0.7882 | 0.7343 | 0.9920 | 0.7339 | 0.7804 | 0.7388 | 0.8385 | 0.7781 |
| CBAM | 0.9491 | 0.5595 | 0.7482 | 0.8632 | 0.7872 | 0.7515 | 0.9911 | 0.6357 | 0.7845 | 0.7103 | 0.8146 | 0.7453 |
| SRM | 0.9561 | 0.5967 | 0.7889 | 0.8832 | 0.8089 | 0.7809 | 0.9852 | 0.6816 | 0.7885 | 0.6672 | 0.8117 | 0.7462 |
| ECA-Net | 0.9495 | 0.5589 | 0.7571 | 0.8644 | 0.7974 | 0.7568 | 0.9949 | 0.7404 | 0.7884 | 0.7359 | 0.8454 | 0.7831 |
| **Proposed** | 0.9532 | 0.6466 | 0.7944 | 0.8844 | 0.8087 | **0.7926** | 0.9941 | 0.7614 | 0.8147 | 0.7569 | 0.8746 | **0.8080** |


