# DECS-Net: with Haar wavelet-based attention
We have made improvements based on [previous work](https://doi.org/10.1016/j.autcon.2023.105217):

<div align=center>
<img src="net.png" width="800px">
</div>

1. Treating the original input as a high-frequency component is arbitrary. Therefore, we first use **Haar wavelet decomposition** to obtain low-frequency components. Then upsample the low-frequency component and subtract it from the original input to obtain the high-frequency component.

<div align=center>
<img src="HLA.png" width="800px">
</div>

2. We have further improved the feature fusion module. Considering the issue of feature misalignment and mismatch, we first use a convolution to adjust. Then design cross-domain fusion module to perform deep fusion between features.

<div align=center>
<img src="FFM.png" width="600px">
</div>

For more details, please refer to our [paper](). If you find our work helpful, please cite us:

```
```
