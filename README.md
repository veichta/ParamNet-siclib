# Perspective Fields Files for GeoCalib Evaluation

This repo hosts files related to the Perspective Fields models for single-image calibration to be used in GeoCalib.

## Setup

First clone this repository under `third_party` in the GeoCalib repository:

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/veichta/ParamNet-siclib.git
```

Running the `setup.sh` script will copy the necessary files to the correct location in the GeoCalib 
repository and download all necessary models:

```bash
source setup.sh
```

Afterwards you can follow the instructions in the GeoCalib repository to evaluate the Perspective Fields models.
