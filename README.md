# Attention Mechanism on BAGNETS
In Relation to AI Project at Friedrich Alexander University- Apply Visual Attention Mechanisms on BAGNETS model
With guidance and help from Dario Zanca, Thomas Robert Altstidl 

## Abstract

Deep Neural Networks(DNNs) have excelled in many perceptual tasks but
it is notoriously difficult to understand their decision making process. In
contrast, the variant Bag-Of-Local features or BagNets model classifies an
image based on the presence of small local image features, independent of
their spatial ordering. By doing so it achieves a Top5 accuracy of 80.5% on
ImageNet Dataset, which is similar to AlexNet. Another important feature
in perceptual tasks is attention. In this study we apply various Human
attention models to BagNets replacing the Simple Average model used by
the author. Experimental results on the ImageNet Dataset, show that the
various attention models performed worse than the Simple Average model.
This implies that decreasing the number of patches of an Image used for
Classification using Attention Mechanism doesn’t help in improving accuracy
in this case.


## Description of Files

All the python scripts are present in the Code directory.

- FeatureExtractor.py - Extract Features from the Images of Dataset
- SaliencyMap.py - Generate the Scanpath in case of CLE and Saliency maps in case of ITTI Saliency Map


## Report
![Detailed Project Report](https://github.com/jeet-ss/AttentionMechanism_BAGNETS/blob/main/Bagnets_Project_Report.pdf)
