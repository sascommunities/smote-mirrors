# SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling

## Overview

This repository contains the source code for the paper: SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling by Georgi Ganev, Reza Nazari, Rees Davison, Amir Dizche, Xinmin Wu, Ralph Abbey, Jorge Silva, Emiliano De Cristofaro, [ICLR 2026](https://openreview.net/forum?id=ZQSZMpsQKj)

The code implements distinguishing and reconstruction attacks against the SMOTE algorithm to reproduce the experiments described in the paper.
Our intent in publishing this code is to support research and education by helping the community understand privacy risks in data augmentation and synthetic data generation and to encourage the development of more privacy-preserving methods.
You must ensure you use these materials responsibly and ensure that your use complies with applicable laws and data use agreements.


## Installation

The experiments require Python 3.11. All necessary dependencies are listed in requirements.txt.


## Source Code Structure

In this paper, we expose the privacy vulnerabilities of SMOTE.

Releveant code/notebooks with results from the paper:

* code/attacks/smote_detection_attack.py (DistinSMOTE attack)
* code/attacks/smote_reconstruction_attack.py (ReconSMOTE attack)
* code/nb_00_toy_example.ipynb (toy examples/gifs used in our [blogpost](https://desfontain.es/blog/smote-and-mirrors.html))
* code/nb_10_aug_naive.ipynb (Table 1 top left and Table 3 left)
* code/nb_11_aug_mia.ipynb (Table 1 top middle and Table 3 middle)
* code/nb_12_aug_distin.ipynb (Table 1 top right and Table 3 right)
* code/nb_20_synth_naive.ipynb (Table 1 bottom left and Table 4 left)
* code/nb_21_synth_mia.ipynb (Table 1 bottom middle and Table 4 middle)
* code/nb_app_0_k2.ipynb (Appendix)
* code/nb_app_1_big.ipynb (Appendix)
* code/nb_app_2_mix.ipynb (Appendix)
* code/nb_app_3_noise.ipynb (Appendix)


## Contributing

Maintainers are not currently accepting patches and contributions to this project.


## License

This project is licensed under the [Apache 2.0 License](LICENSE).


## Citing

If you use this code, please cite the associated paper:
```
@inproceedings{ganev2026smote,
  title={{SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling}},
  author={Ganev, Georgi and Nazari, Reza and Davison, Rees and Dizche, Amir and Wu, Xinmin and Abbey, Ralph and Silva, Jorge and De Cristofaro, Emiliano},
  booktitle={ICLR},
  year={2026}
}
```
