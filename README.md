# `DeepInv` - A python library of deep learning methods for inverse problems

## Overview

Inverse problems are ubiquitous in scientific imaging and signal processing. Although deep learning has achieved significant success in many imaging problems, a comprehensive and timely library is still missing. `DeepInv` will implement the most mainstream deep learning approaches for imaging, from supervised, unsupervised, and self-supervised methods. We hope people could effectively solve their inverse problems by simply calling DeepInv's APIs which should be very easy to setup and friendly to use.  

## Outline

## Log
* [2022-10-13] upload v0.1 (core module), only implemented ei) [dongdong]
* [2022-11-02] upload v0.2 (core module), added 'ei', 'rei', 'sure', 'mc', 'sup', 'measplit', 'noise mode', new interface and allow flexible losses combination) [dongdong]

## To Do Julian
* add Measurement splitting

## To Do Dongdong
* [Done] add **REI**, **SURE**, **MC**, **Sup** and * by next meeting (2022-10-20)
* [Done] add Measurement splitting (MeaSplit) -- just made a first go, need to double-check.

## Reference

The below papers are related to `DeepInv`.

```
@inproceedings{chen2021equivariant,
    title     = {Equivariant Imaging: Learning Beyond the Range Space},
    author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4379-4388}}
    
@inproceedings{chen2022robust,
    title     = {Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements},
    author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}}

@article{chen2022imaging,
    title={Imaging with Equivariant Deep Learning},
    author={Chen, Dongdong and Davies, Mike and Ehrhardt, Matthias J and Sch{\"o}nlieb, Carola-Bibiane and Sherry, Ferdia and Tachella, Juli{\'a}n},
    journal={IEEE Signal Processing Magazine},
    year={2022}}

@article{tachella2022sampling,
    title={Unsupervised Learning From Incomplete Measurements for Inverse Problems},
    author={Tachella, Juli{\'a}n and Chen, Dongdong and Davies, Mike},
    journal={To appear in Proceedings of the Thirty-sixth Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}}

@article{tachella2022sampling,
    title={Sampling Theorems for Unsupervised Learning in Linear Inverse Problems},
    author={Tachella, Juli{\'a}n and Chen, Dongdong and Davies, Mike},
    journal={arXiv preprint arXiv:2203.12513},
    year={2022}}
```