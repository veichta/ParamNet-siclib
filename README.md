# Perspective Fields Files for GeoCalib Evaluation

This repo hosts files related to the Perspective Fields models for single-image calibration to be used in [GeoCalib](https://github.com/cvg/GeoCalib).

## Setup

To setup the Perspective Fields models for GeoCalib evaluation, clone this repository under `third_party` in the GeoCalib repository. Afterward, copy the `siclib` directory from this repository to the root of the GeoCalib repository. This can be done with the following commands (assuming you are in the root of the GeoCalib repository):

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/veichta/ParamNet-siclib.git && cd ParamNet-siclib
cp -r siclib ../..
cd ../..
```

Afterwards you can follow the instructions in the GeoCalib repository to evaluate the Perspective Fields models.

## 🚨 Disclaimer 🚨

As this code is closely based on the original code from the [Perspective Fields](https://github.com/jinlinyi/PerspectiveFields) repository, using these files will require you to comply with the original license. Please refer to the original repository for more information.
