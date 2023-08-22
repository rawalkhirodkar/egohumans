<div align="center">

<h2><a href="https://arxiv.org/pdf/2305.16487.pdf">EgoHumans: An Egocentric 3D Multi-Human Benchmark</a></h2>

[Rawal Khirodkar](https://github.com/rawalkhirodkar)<sup>1</sup>, [Aayush Bansal](https://www.aayushbansal.xyz/)<sup>2</sup>, [Lingni Ma](https://scholar.google.nl/citations?user=eUAgpwkAAAAJ&hl=en/)<sup>2</sup>, [Richard Newcombe](https://scholar.google.co.uk/citations?user=MhowvPkAAAAJ&hl=en)<sup>2</sup>, [Minh Voh](https://minhpvo.github.io/)<sup>2</sup>, [Kris Kitani](https://kriskitani.github.io/)<sup>1</sup>
 
<sup>1</sup>[CMU](https://www.cmu.edu/), <sup>2</sup>[Meta](https://about.meta.com/)

ICCV 2023 (Oral)

[Project Page](https://rawalkhirodkar.github.io/egohumans/)

</div>


We present EgoHumans, a new multi-view multi-human
video benchmark to advance the state-of-the-art of egocentric human 3D pose estimation and tracking. Existing egocentric benchmarks either capture single subject or indooronly scenarios, which limit the generalization of computer
vision algorithms for real-world applications. We propose
a novel 3D capture setup to construct a comprehensive egocentric multi-human benchmark in the wild with annotations
to support diverse tasks such as human detection, tracking,
2D/3D pose estimation, and mesh recovery. We leverage
consumer-grade wearable camera-equipped glasses for the
egocentric view, which enables us to capture dynamic activities like playing tennis, fencing, volleyball, etc. Furthermore,
our multi-view setup generates accurate 3D ground truth
even under severe or complete occlusion. The dataset consists of more than 125k egocentric images, spanning diverse
scenes with a particular focus on challenging and unchoreographed multi-human activities and fast-moving egocentric
views. We rigorously evaluate existing state-of-the-art methods and highlight their limitations in the egocentric scenario,
specifically on multi-human tracking. To address such limitations, we propose EgoFormer, a novel approach with a
multi-stream transformer architecture and explicit 3D spatial
reasoning to estimate and track the human pose. EgoFormer
significantly outperforms prior art by 13.6% IDF1 on the
EgoHumans dataset


## Overview

![summary_tab](assets/overview.png)


## Get Started
- [🛠️Installation](assets/INSTALL.md)
- [📘Download Data](assets/DOWNLOAD.md)
- [👀Visualization](assets/VISUALIZE.md)

## Tracking Benchmark (Coming Soon)
- ETA September end.

## EgoFormer Training/Testing (Coming Soon)
- ETA September end.

## BibTeX & Citation

```
@article{khirodkar2023egohumans,
  title={EgoHumans: An Egocentric 3D Multi-Human Benchmark},
  author={Khirodkar, Rawal and Bansal, Aayush and Ma, Lingni and Newcombe, Richard and Vo, Minh and Kitani, Kris},
  journal={arXiv preprint arXiv:2305.16487},
  year={2023}
}
```

## Acknowledgement
[Aria Toolkit](https://github.com/facebookresearch/projectaria_tools), [COLMAP](https://github.com/colmap/colmap), [mmpose](https://github.com/open-mmlab/mmpose/tree/main), [mmhuman3D](https://github.com/open-mmlab/mmhuman3d), [CLIFF](https://github.com/haofanwang/CLIFF), [timm](https://github.com/rwightman/pytorch-image-models), [detectron2](https://github.com/facebookresearch/detectron2), [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection).




## Contact

- For help and issues associated with EgoHumans, or reporting a bug, please open a [GitHub Issue](https://github.com/rawalkhirodkar/egohumans).

- Please contact [Rawal Khirodkar](https://rawalkhirodkar.github.io/) (`rkhirodk@cs.cmu.edu`) for any queries.