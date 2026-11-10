#!/bin/bash
AlgoType=$(basename "$PWD")
weight_version=V1.0.0

echo "AlgoType:${AlgoType}"
rm -rf ./weights
mkdir -p weights
cd weights && wget http://10.222.20.48:20000/AGI/AGI_modelweights/ASR/dev_env/V1.0.0/ASR_weights_V1.0.0.zip&& unzip *.zip && rm *.zip
