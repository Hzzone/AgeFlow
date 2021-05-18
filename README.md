### AgeFlow
This repository contains the PyTorch implementation and the dataset of the paper: **AgeFlow: Conditional Age Progression and Regression with Normalizing Flows (IJCAI 2021)**

**Code to be public in about 2 weeks at the early June.**

> AgeFlow: Conditional Age Progression and Regression with Normalizing Flows<br>
> https://arxiv.org/abs/2105.07239<br>
> Abstract: Age progression and regression aim to synthesize photorealistic appearance of a given face image with aging and rejuvenation effects, respectively. Existing generative adversarial networks (GANs) based methods suffer from the following three major issues: 1) unstable training introducing strong ghost artifacts in the generated faces, 2) unpaired training leading to unexpected changes in facial attributes such as genders and races, and 3) non-bijective age mappings increasing the uncertainty in the face transformation. To overcome these issues, this paper proposes a novel framework, termed AgeFlow, to integrate the advantages of both flow-based models and GANs. The proposed AgeFlow contains three parts: an encoder that maps a given face to a latent space through an invertible neural network, a novel invertible conditional translation module (ICTM) that translates the source latent vector to target one, and a decoder that reconstructs the generated face from the target latent vector using the same encoder network; all parts are invertible achieving bijective age mappings. The novelties of ICTM are two-fold. First, we propose an attribute-aware knowledge distillation to learn the manipulation direction of age progression while keeping other unrelated attributes unchanged, alleviating unexpected changes in facial attributes. Second, we propose to use GANs in the latent space to ensure the learned latent vector indistinguishable from the real ones, which is much easier than traditional use of GANs in the image domain. Experimental results demonstrate superior performance over existing GANs-based methods on two benchmarked datasets.

### Illustration

* learned face latent space and decoded average latent variables.
![](fig/example.png)

* Generator
![](fig/generator.png)

* Training
![](fig/training.png)

#### Citation

If you found this code, pre-trained model, or our work useful please cite us:

```
@article{huang2021ageflow,
  title={AgeFlow: Conditional Age Progression and Regression with Normalizing Flows},
  author={Huang, Zhizhong and Chen, Shouzhen and Zhang, Junping and Shan, Hongming},
  journal={IJCAI},
  year={2021},
}
```