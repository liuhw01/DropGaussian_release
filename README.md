<h1 align="center">DropGaussian:Structural Regularization<br>for Sparse-view Gaussian Splatting</h1>


Official Pytorch implementation [**"DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting"**](https://arxiv.org/abs/2304.09502v1) <br>
[Hyunwoo Park](https://github.com/HWP97?tab=repositories), [Gun Ryu](https://github.com/jerry-ryu), and [Wonjun Kim](https://sites.google.com/view/dcvl/team/professor) (Corresponding Author) <br>
🎸***IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)***, Jun. 2025.🎸

<p align="center"><img src='figures/Fig1.jpg'></p>
<p align="center">[ Training pipeline ]</p>

## :eyes: Overview 
We propose a simple yet powerful regularization technique, **DropGaussian**, for neural rendering with sparse input views.

By randomly eliminating Gaussians during training, DropGaussian allows the remaining Gaussians to receive stronger gradients, encouraging them to contribute more meaningfully to the optimization process of 3D Gaussian Splatting (3DGS).  
This approach effectively mitigates overfitting, which is a common issue under sparse-view settings.

We provide:

- ✅ **Full implementation** of DropGaussian
- ⚡ **Minimal plug-and-play code snippet** for quick integration

## 🚀 Quick Snippet

Here's a minimal example of how to use `DropGaussian` in your training loop:

```python
import torch
# (Assume the rest of the 3DGS pipeline is already set up)
# Create initial compensation factor (1 for each Gaussian)
compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

# Apply DropGaussian with compensation
drop_rate = 0.1
d = torch.nn.Dropout(p=drop_rate)
compensation = d(compensation)

# Apply to opacity
opacity = opacity * compensation[:, None]
```

### Installation

<!--

_This section will be released soon!_

 -->

### Download

<!--

_This section will be released soon!_

 -->

### Demo

We provide demo codes to run end-to-end inference on the test images. </br>

Please check [Demo.md](documents/Demo.md) for more information.

### Experiments

We provide guidelines to train and evaluate our model on Human3.6M and 3DPW. </br>

Please check [Experiments.md](documents/Experiments.md) for more information.


## :page_with_curl: Results 

### Quantitative result
| Model                        | Dataset   | MPJPE | PA-MPJPE | Checkpoint            |
| ---------------------------- | --------- | ----- | -------- | --------------- |
| PointHMR-HR32                | Human3.6M |48.3   | 32.9     | [Download](https://drive.google.com/file/d/1Np8SAEFEou2HcfDYH7b1a4rjLI1GnwVQ/view?usp=sharing)|
| PointHMR-HR32                | 3DPW      |73.9   | 44.9     | [Download]()|

### Qualitative results

Results on **3DPW** dataset:

<p align="center"><img src='documents/fig3.jpg'></p>

Results on **COCO** dataset:

<p align="center"><img src='documents/fig4.jpg'></p>

## License

This research code is released under the MIT license. Please see [LICENSE](LICENSE) for more information.

SMPL and MANO models are subject to **Software Copyright License for non-commercial scientific research purposes**. Please see [SMPL-Model License](https://smpl.is.tue.mpg.de/modellicense.html) and [MANO License](https://mano.is.tue.mpg.de/license.html) for more information.

We use submodules from third party ([hassony2/manopth](https://github.com/hassony2/manopth)). Please see [NOTICE](documents/NOTICE.md) for more information.


## Acknowledgments
This work was supported by Institute of Information \& communications Technology Planning \& Evaluation(IITP) grant funded by the Korea government(MSIT) (2021-0-02084, eXtended Reality and Volumetric media generation and transmission technology for immersive experience sharing in noncontact environment with a Korea-EU international cooperative research).

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[microsoft/MeshTransformer](https://github.com/microsoft/MeshTransformer)  </br>
[microsoft/MeshGraphormer](https://github.com/microsoft/MeshGraphormer)  </br>
[postech-ami/FastMETRO](https://github.com/postech-ami/FastMETRO)  </br>
[Arthur151/ROMP](https://github.com/Arthur151/ROMP)  </br>



## Citation
```bibtex
@InProceedings{PointHMR,
author = {Kim, Jeonghwan and Gwon, Mi-Gyeong and Park, Hyunwoo and Kwon, Hyukmin and Um, Gi-Mun and Kim, Wonjun},
title = {{Sampling is Matter}: Point-guided 3D Human Mesh Reconstruction},
booktitle = {CVPR},
month = {June},
year = {2023}
}
```
<!--
## License
 -->
