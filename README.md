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
# 1. Requirements
## Environments
Following packages are required for this repo.

    - python 3.8+
    - torch  2.4+
    - torchvision 0.19+ 
    - torch 1.9+
    - CUDA 12.1+
    - cython 3.1+
    - scikit-learn 1.3+
    - numpy 1.24+
    - tqdm 4.67.1
    - bcolz 1.2.1  
    - matplotlib 3.7.5   

# 2. Training & Evaluation
## Synthetic Experiments
- For synthetic experiments, simply run **'main_nirvana_hypersphere_3_class.py'** to create the embedding by using 3 classes and run **'main_nirvana_hypersphere_10class.py'** to create the embedding by using 10 classes. The classes will be chosen from the cifar10 dataset.
   Tey will produce the figures used in Fig. 2 given in the paper. To create embeddings for different number of classes, just change the line 173 in the code. For examle  dataset = CIFAR10_OSR(known=[0, 1, 2, 3], batch_size=args.batch_size, use_gpu=True)  will use 4 classes,
   dataset = CIFAR10_OSR(known=[0, 1, 2, 3, 4], batch_size=args.batch_size, use_gpu=True) will use 5 classes, so on.
## Open Set Recognition
- For open set recognition, simply run **'NirvanaOSR_Hypersphere.py'** for to run experiments add datasets to "data" folder and choose dataset from mnist | svhn | cifar10 | cifar100 | tiny_imagenet one of them. Use "classifier32" networks for all experiments except tiny_imagenet and for that use resnet50. Use "nirvana_hypersphere" in dchs as loss function. Noter that this loss function is different than the loss function used for closed ste recognition since background samples are also used.
## Closed Set Recognition
- Just use **'main_cifar_hypersphere.py'** to train the network for Cifar100 and Cifar10 datasets. The dataset must be under **data** folder, the code will downlaod the datasets. Use **'main_mnist_hypersphere.py'** to train the network for Mnist dataset.
## Experiments on Imbalanced Dataset
- Just use **'main_cifar_imbalaced_hypersphere.py'** to train the network on Cifar10-LT. You can change the ibalance ratio by changing the line  163 in the code, e.g., imbalance_ratio=0.01, sets the imbalance ratio to 0.01.
# 3. Results
### The learned feature embeddings:
<img width="1210" height="607" alt="figure_embeddings" src="https://github.com/user-attachments/assets/aef1efe3-99af-411b-91c4-ae8bd85ea5f4" />
**Fig 2.** The outputs of the deep neural network classifiers trained by using the proposed loss functions for 3, 5 and 10 classes.
The first row shows the embedding obtained by DUDCH-U1 classifier and the embeddings shown in the second row are obtained
by using DUDCH-U2. For both cases, the class centers are uniformly distributed on the boundary of the hypersphere, and the
class-specific samples compactly cluster in the vicinity of their class centers.
<img width="1491" height="431" alt="figure_uniformity_loss" src="https://github.com/user-attachments/assets/a239bb2f-d99c-4393-98c1-c16b52eb6547" />
**Fig 3.** Embeddings of the class samples for different κ values. Setting κ parameter to larger values returned more uniform
distributions.
