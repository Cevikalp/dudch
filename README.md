# dudch
This repo includes the mains source files for the method described in the paper titled "Learning Uniform Hyperspherical Centers for Open and Closed Set Recognition".
# Learning Uniform Hyperspherical Centers for Open and Closed Set Recognition
**Abstract:** Open-set recognition remains a challenging problem, particularly when traditional closed- set classifiers are unable to generalize to unseen classes. In this paper, we propose a unified framework 
that leverages hyperspherical embeddings with learnable class centers for both open-set and closed-set recognition. Each class is represented by a center point uniformly distributed on the surface of a hypersphere,
and training samples are encouraged to form compact clusters around their respective centers. Unlike previous methods that constrain features to the hypersphere boundary, we adopt a Euclidean distance-based
formulation to improve flexibility and generalization. Our approach jointly optimizes class centers and feature representations, eliminating the reliance on predefined center locations. Additionally, we introduce 
a mechanism to incorporate background or unknown samples during training to further enhance open-set robustness. Extensive experiments on multiple benchmarks demonstrate that our method outperforms 
existing approaches, achieving state-of-the-art accuracy in both open-set and closed-set settings.

**Our main contributions are as follows:**

 -- We introduce a method that learns uniformly distributed class centers in an end-to-end manner from training data. This eliminates the need for manually predefining
centers—a laborious step in prior approaches. Moreover, our model captures semantically meaningful feature representations, as demonstrated in the experiments.

 -- As opposed to the class centers, the learned CNN features are not inherently normalized to reside on the boundary of the hypersphere. As a result, it becomes
relatively simpler to identify and reject the unknown samples by evaluating the Euclidean distances to the known class centers.

 -- The proposed framework is robust to class imbalance, as it minimizes the distance between each sample and its corresponding class center independently, without being
biased toward larger classes.

<img width="555" height="321" alt="DeepHypersphere" src="https://github.com/user-attachments/assets/2458dfb7-ccc2-40fb-ba9a-8f41c40a0355" />

**Fig 1.** Comparison of embedding spaces between the proposed method and state-of-the-art face recognition approaches
that maximize angular margins in open set recognition: Known class centers are illustrated with star symbols in different colors,
while their corresponding samples are depicted as circles of the same color. Unknown class samples are shown as gray circles.
In face recognition methods such as CosFace and ArcFace, where both features and classifier weights are normalized, known
and unknown samples tend to overlap, making it difficult to distinguish and reject the unknowns (left side of the figure). In
contrast, our method avoids feature normalization of class samples, which allows unknown samples to be more easily separated
based on their Euclidean distances to the learned class centers (right side of the figure).
