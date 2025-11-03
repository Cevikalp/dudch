# dudch
This repo includes the mains source files for the method described in the paper titled "Learning Uniform Hyperspherical Centers for Open and Closed Set Recognition".
# Learning Uniform Hyperspherical Centers for Open and Closed Set Recognition
**Abstract:** Open-set recognition remains a challenging problem, particularly when traditional closed- set classifiers are unable to generalize to unseen classes. In this paper, we propose a unified framework 
that leverages hyperspherical embeddings with learnable class centers for both open-set and closed-set recognition. Each class is represented by a center point uniformly distributed on the surface of a hypersphere,
and training samples are encouraged to form compact clusters around their respective centers. Unlike previous methods that constrain features to the hypersphere boundary, we adopt a Euclidean distance-based
formulation to improve flexibility and generalization. Our approach jointly optimizes class centers and feature representations, eliminating the reliance on predefined center locations. Additionally, we introduce 
a mechanism to incorporate background or unknown samples during training to further enhance open-set robustness. Extensive experiments on multiple benchmarks demonstrate that our method outperforms 
existing approaches, achieving state-of-the-art accuracy in both open-set and closed-set settings.

<img width="555" height="321" alt="DeepHypersphere" src="https://github.com/user-attachments/assets/2458dfb7-ccc2-40fb-ba9a-8f41c40a0355" />
