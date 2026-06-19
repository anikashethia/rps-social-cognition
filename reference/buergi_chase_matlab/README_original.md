# A neural signature of mentalization

This repository contains analysis code, behavioral data, and the multivariate pattern to recreate the main results and figures from the paper "A neural signature of adaptive mentalization" by Buergi, Aydogan, Konovalov, & Ruff (Nature Neuroscience, 2026).

## Instructions

To run the statistical tests and recreate the main figures, make sure the working directory is set to the location of this repository and run `BAKR_2024_results_and_figures` (or individual parts therein). This file contains the final steps of the analysis pipeline, relying on computational model fits and statistical parameteric maps of fMRI data produced in previous steps (see below). Note that some parts of this script that are based on fMRI data require that SPM outputs are saved in the results folder (which are not part of the repository due to space constraints); these parts will be skipped if the corresponding outputs cannot be found in their respective subfolders.

To re-run the preceding analysis steps and recreate the whole analysis pipeline, two additional files are provided:
- `BAKR_2024_run_model_fitting` contains all analyses steps pertaining to the computational models (i.e., model fitting, parameter recovery, model recovery, and model simulations).
- `BAKR_2024_run_fmri_analyses` contains all analyses steps pertaining to neuroimaging analyses (i.e., univariate and multivariate/decoding analyses).

See the comments within those files for details. For fMRI analysis, the preprocessed neuroimaging data (available upon request) needs to be added to the data folder. Note that due to the complexity of the underlying analyses, these files will take substantially longer to run than the final results file above (i.e., a few days on a modern machine with parallel processing).

Before running any of these scripts, make sure that all required external toolboxes (see below) are downloaded and correctly installed.

## Requirements

- MERLIN toolbox (pre-release version) for model fitting (included in `/source/MERLIN_toolbox` and automatically addded to the path when running the files above)
- VBA toolbox (5899497) for random effects model comparison (https://mbb-team.github.io/VBA-toolbox/)
- SPM12 (7771) and SnPM13 (13.1.09) for univariate neuroimaging analysis (https://github.com/spm/ and https://warwick.ac.uk/snpm)
- The Decoding Toolbox (3.999G) for level decoding (https://sites.google.com/site/tdtdecodingtoolbox/)
- CanlabCore (d0122bc) for belief update decoding (https://github.com/canlab/CanlabCore)
- MRIcroGL (1.2.20220720) for visualization of neural results (https://www.nitrc.org/projects/mricrogl/)
- AAL (v4) for automated labelling of anatomical areas of the signature (https://github.com/Neurita/std_brains/tree/master/atlases/aal_SPM12/aal)

## Contents

- `/data` contains behavioral data, and a subfolder for fmri data (available upon request)
- `/masks` contains binary masks for ROI analyses, and subfolders containing masks of the significant activation clusters from the paper (one for BU within the different ROIs, and one for all pmods across ROIs)
- `/pattern` contains the neural signature of adaptive mentalization (see BAKR_2024_run_fmri_analyses for usage examples)
- `/results` contains outputs from intermediate analysis steps, including model fits and simulations, significant voxels, decoding outputs, and statical parametric maps from SPM
- `/source` contains the source files that are called from the main scripts above, and a subfolder containing the MERLIN toolbox used for model fitting

## Further information

Developed and tested in Matlab R2018b and R2023b.

If any of the code from this repository is reused, please cite the associated paper: Buergi, N., Aydogan, G., Konovalov, A., & Ruff, C. C. (forthcoming). A neural signature of adaptive mentalization. Nature Neuroscience.

For questions, please contact [Niklas Buergi](mailto:niklas.g.buergi@gmail.com) or [Gökhan Aydogan](mailto:goekhan.aydogan@econ.uzh.ch).
