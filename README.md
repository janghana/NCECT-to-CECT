# Contrast Agent-Free Approach for Enhancing Hepatocellular Carcinoma (HCC) Visualization in CECT Derived from NCECT

## Overview
This repository is dedicated to the study of an innovative technique for enhancing the visualization of Hepatocellular Carcinoma (HCC) in Contrast-Enhanced Computed Tomography (CECT) images which are derived from Non-Contrast Enhanced Computed Tomography (NCECT) scans. Our research aims to eliminate the need for multiple contrast agent (CA) injections, minimizing patient exposure to potential risks and streamlining the diagnostic process for HCC.

## Introduction
Contrast agents are commonly used in medical imaging to improve the clarity and detail of the images. However, the use of CAs can lead to complications and discomfort for patients. This project explores a new approach that simulates the effects of CA without actual injection, utilizing the principles behind CA injections to enhance NCECT images.

## Methodology
The proposed method relies on advanced imaging algorithms that adjust the contrast levels of NCECT images to mimic CECT scans. By doing so, it's possible to achieve similar diagnostic effectiveness as traditional CECT, reducing the need for additional scans and CA administration.

## Benefits
- **Patient Safety**: Significantly reduces the risks associated with the use of contrast agents.
- **Efficiency**: Streamlines the diagnostic process by eliminating the need for multiple scans and CA injections.
- **Cost-Effectiveness**: Decreases the costs related to CAs and the overall diagnostic procedure.

## Results
Preliminary results indicate that the contrast agent-free approach could become a promising alternative to conventional methods, providing enhanced visualization of HCC with potentially higher safety and efficiency.

## Usage
Detailed information about the implementation of this method and the usage guidelines are provided within this repository.

```
python3 train_pre2art.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --contrast1 PRE --contrast2 ART --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 4 --save_content --local_rank 0 --input_path ../../seo
```


## Contribution
We welcome contributions from the community. Whether you are a radiologist, medical imaging expert, or someone interested in the application of AI in healthcare, your insights and input could be invaluable.
