# Title
This repo contains the code of the paper:

*A Machine Learning Framework for Automated Planning of Hybrid Energy Systems Using Generalizable Deep Reinforcement Learning* 
(under review) 

by Manuel Sage & Bi Cheng Zhao & Yaoyao Fiona Zhao (2026).

------
### Preprint
https://dx.doi.org/10.2139/ssrn.6685512

-----
### Dataset

All data with hourly resolution is part of this repository (see /data/).

Data with finer temporal resolutions (30min, 15min, 5min, 1.25min) is shared here:
https://doi.org/10.5281/zenodo.19736898.

The 1.25-minute data was reconstructed using 5-minute data following the temporal downscaling approach described 
by [Serifi et al.](https://doi.org/10.3389/fclim.2021.656479), with code available 
[here](https://github.com/aserifi/convolutional-downscaling).

-----
### Installation/Usage
Preferred installation:

```pip install -r requirements.txt```

Note: Trained DRL agents are saved in /models/saved_dispatch_models/ for 1h and 1.25min resolution.



----
For comments and questions reach out to [manuel.sage@mail.mcgill.ca](manuel.sage@mail.mcgill.ca).