# PV_active_learning

Accompanying code for the paper 'Toward global rooftop PV detection with Deep Active Learning': https://www.sciencedirect.com/science/article/pii/S2666792424000295?via%3Dihub 

This code is used to make the simulations (snakemake) and for the analysis of the results (jupyter notebook).

In deepal_PV, higher level abstractions (in object-orientated design) are saved for reusability. The python files in this folder the main rules in the snakemake setting used for the simulations. 

## Data download

* CA-F and CA-S: California, USA
    * Article: https://www.nature.com/articles/sdata2016106
    * Data: https://figshare.com/articles/dataset/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780/4 

* FR-G and FR-I: France
    * Article: https://www.nature.com/articles/s41597-023-01951-4 
    * Data: https://zenodo.org/records/7059985
* DE-G - Oldenburg, Germany: By request from corresponding author
* NY-Q - Queens, New York, USA: By request from corresponding author
    * Data Landing Page: https://data.gis.ny.gov/ 
    * Orthodata Direct Link: https://orthos.dhses.ny.gov/

## Installation

The whole project is implemented in python (mainly pytorch). To install the needed libraries to be able run the experiments and to make the plotting, the libraries as stated in the requirements.txt are needed.

Note that SLURM is used to run the code on a HPC. In principle, the code should also run on a local computer, but the requirements (number of GPUs) are usually too large for any PC.

In config.py, the relevant paths are set.


# How to run the experiments

snakemake all_run_all_simulations --profile . --resources load=100

Note that you may need to adapt the GPU architecture (e.g. volta) on your local system.
