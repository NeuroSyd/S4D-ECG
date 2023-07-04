# S4D-ECG: A shallow state-of-the-art model for cardiac arrhythmia classification ([Paper](https://www.medrxiv.org/content/10.1101/2023.06.30.23292069v1))

This work consists of three main code files. The ECG.py file includes the model definition and training process. ECG_predict.py evaluates the model's performance on a test set. Finally, ECG_generalization assesses the model's generalization and robustness using different datasets. 

Together, these files provide a comprehensive pipeline for developing, training, evaluating, and testing the S4D-ECG model.

The code underwent enhancements based on the work by Hasani et al. in 2022, titled "Liquid Structural State-Space Models." 

@article{hasani2022liquid,
  title={Liquid Structural State-Space Models},
  author={Hasani, Ramin and Lechner, Mathias and Wang, Tsun-Huang and Chahine, Makram and Amini, Alexander and Rus, Daniela},
  journal={arXiv preprint arXiv:2209.12951},
  year={2022}
}

## Setup

### Requirements
This repository requires Python 3.8+ and Pytorch 1.9+.  
Other packages are listed in `requirements.txt`.

`pip3 install -r requirement.txt`

## Citation

```
@article {Huang2023.06.30.23292069,
	author = {Zhaojing Huang and Luis Fernando Herbozo Contrera and Leping Yu and Nhan Duy Truong and Armin Nikpour and Omid Kavehei},
	title = {S4D-ECG: A shallow state-of-the-art model for cardiac arrhythmia classification},
	elocation-id = {2023.06.30.23292069},
	year = {2023},
	doi = {10.1101/2023.06.30.23292069},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2023/07/01/2023.06.30.23292069},
	eprint = {https://www.medrxiv.org/content/early/2023/07/01/2023.06.30.23292069.full.pdf},
	journal = {medRxiv}
}

```
