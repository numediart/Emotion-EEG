# Emotion-EEG

Set of models for emotion estimation from EEG. Composed by the combination of two deep-learing models learning together (RNN and CNN) with the help of a saliency analysis. 

![alt text](Model.png "EEMotion - DL model estimating emotion from multi-approach")

## Instruction

The three proposed models are direcly available here:

* Saliency based combination of the hierarchical RNN and CNN. [Sal_Model.py](Sal_Model.py)
* Feature fusion of the hierarchical RNN and CNN. [Feat_Model.py](Feat_Model.py
)[Sal_Model](Sal_Model.py)* Ouput fusion of the hierarchical RNN and CNN. [Loss_Model.py](Loss_Model.py)

## Installation and Dependencies

[Pytorch 1.5](https://pytorch.org/get-started/locally)

[MNE](https://mne.tools/stable/install/mne_python.html#install-python-and-mne-python)

[Cuda 10.1](https://developer.nvidia.com/cuda-toolkit)

Installation with pip: `pip install -r req.txt`

Import of the environment with conda: `conda env create -f env.yml`

## Remarks

Due to the EULA for each dataset, some example signals have been proposed to test the models, however, **they are not corresponding to signals from one of tested dataset**.

If you are interested in our work, don't hesitate to contact us. 

Best! :smile: 

ps: if you use this repo in other research project, please cite the original paper:

@article{delvigne2022emotion,
	title={Emotion Estimation from EEG--A Dual Deep Learning Approach Combined with Saliency},
	author={Delvigne, Victor and Facchini, Antoine and Wannous, Hazem and Dutoit, Thierry and Ris, Laurence and Vandeborre, Jean-Philippe},
	journal={arXiv preprint arXiv:2201.03891},
	year={2022}
}