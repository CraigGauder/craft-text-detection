# craft-text-detection

Implementation of CRAFT convolutional network architecture for text detection and localization in natural images.

This neural network architecture was provided by the following paper (please read for more info): https://arxiv.org/pdf/1904.01941.pdf

BibText Entry for this Paper:

@article{DBLP:journals/corr/abs-1904-01941,
  author    = {Youngmin Baek and
               Bado Lee and
               Dongyoon Han and
               Sangdoo Yun and
               Hwalsuk Lee},
  title     = {Character Region Awareness for Text Detection},
  journal   = {CoRR},
  volume    = {abs/1904.01941},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.01941},
  archivePrefix = {arXiv},
  eprint    = {1904.01941},
  timestamp = {Wed, 24 Apr 2019 12:21:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1904-01941.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

This is intended for training on SynthText dataset which can be found here -- https://www.robots.ox.ac.uk/~vgg/data/scenetext/

BibTex Entry for this Dataset:

@InProceedings{Gupta16,
  author       = "Ankush Gupta and Andrea Vedaldi and Andrew Zisserman",
  title        = "Synthetic Data for Text Localisation in Natural Images",
  booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2016",
}

## Current Status

- Implementation of CRAFTNet architecture
- Implementation of a preprocesser for converting label data in SynthText dataset to the appropriate format the model is expected for calculating loss (i.e. a gaussian heatmap for character locations, as well as character affinities).

## Todo

- Implementing CRAFTLoss
- Implementing further training mechanisms for subsequent datasets on top of SynthText (see paper for more info)
