---
license: apache-2.0
---



# SkyReels-A2: Compose Anything in Video Diffusion Transformers 
<p align="center">
  <img src="logo.png" alt="Skyreels Logo" width="60%">
</p>

<p align="center">
<a href="https://github.com/SkyworkAI/SkyReels-A2" target="_blank">🌐 Github</a> · <a href="https://www.skyreels.ai/home?utm_campaign=huggingface_A1" target="_blank">👋 Playground</a> · <a href="https://discord.gg/PwM6NYtccQ" target="_blank">Discord</a>· <a href="https://huggingface.co/spaces/Skywork/SkyReels_A2_Bench" target="_blank">🔥 A2-Bench Leaderboard</a>
</p>


This repo contains Diffusers style model weights for Skyreels-A2 models. 
You can find the inference code on [SkyReels-A2](https://github.com/SkyworkAI/SkyReels-A2) repository.


## 🪄 Models 


| Models                   | Download Link                                                                                                                                                                           | Video Size         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| A2-Wan2.1-14B-Preview | [Huggingface](https://huggingface.co/Skywork/SkyReels-A2) 🤗                                                                                                                                                              | ~ 81 x 480 x 832    | 
| A2-Wan2.1-14B         | [To be released](https://github.com/SkyworkAI/SkyReels-A2)  | ~ 81 x 480 x 832    | 
| A2-Wan2.1-14B-Infinity         | [To be released](https://github.com/SkyworkAI/SkyReels-A2)  | ~ Inf x 720 x 1080   | 




---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/62e34a12c9bece303d146af8/cx7ZBef8xjNF0g9Ip915G.png)

Overview of SkyReels-A2 framework. Our approach initiates by encoding all reference images using two distinct branches. The first, termed the spatial feature branch (represented in red, top row), leverages a fine-grained VAE encoder to process per-composition images. The second, identified as the semantic feature branch (represented in red, bottom row), utilizes a CLIP vision encoder followed by an MLP projection to encode semantic references. Subsequently, the spatial features are concatenated with the noised video tokens along the channel dimension before being passed through the diffusion transformer blocks. Meanwhile, the semantic features extracted from the reference images are incorporated into the diffusion transformers via supplementary cross-attention layers, ensuring that the semantic context is effectively integrated during diffusion.

---

Some generated results:


<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/62e34a12c9bece303d146af8/G-_g4xPAvAxvqmInfkylW.mp4"></video>




## Citation

If you find SkyReels-A2 useful for your research, welcome to cite our work using the following BibTeX:
```bibtex
@article{fei2025skyreels,
  title={SkyReels-A2: Compose Anything in Video Diffusion Transformers},
  author={Fei, Zhengcong and Li, Debang and Qiu, Di and Wang, Jiahua and Dou, Yikun and Wang, Rui and Xu, Jingtao and Fan, Mingyuan and Chen, Guibin and Li, Yang and others},
  journal={arXiv preprint arXiv:2504.02436},
  year={2025}
}
```








