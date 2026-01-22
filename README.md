# RAIL Notebooks
<!--Note that this readme is automatically updated weekly. To make changes to the text here, modify the template in update-readme.py--otherwise, whatever you write here will just be overwritten!-->

This repository hosts **rendered RAIL notebooks** used in the RAIL documentation. It is built as a standalone **Read the Docs subproject** and is linked from the main RAIL docs.

The rendered notebooks provide a stable, browsable reference without requiring users to execute notebooks locally.

## Automation and workflow

Notebook rendering is fully automated and runs weekly via GitHub Actions.

The workflow:

- Converts notebooks in the main RAIL repository to `.rst` using `nbconvert`
- Writes rendered outputs to `docs/` and saves logs to `logs/`
- Updates the status tables below

Rendering logic lives in `render_notebooks.py` and is invoked by the `render-notebooks.yml` workflow.

## Adding notebooks

No changes to this repository are required.

Any notebook added to one of the following RAIL repositories is included automatically:

- core
- creation
- estimation
- evaluation
- goldenspike

### Formatting requirements

- Use **one** top-level markdown heading (`# Title`) per notebook.
- Additional top-level headings may be interpreted as separate documents in Read the Docs.

## Notebook Status

The tables below show the current rendering status by notebook category and are updated automatically by a weekly workflow.

<!--auto update below-->

**Tables last updated:** January 22, 2026

### Core Notebooks

| Notebook | Return Code |
|----------|-------------|
| 00_Useful_Utilities.ipynb | 1 |
| 01_FileIO_DataStore.ipynb | 0 |
| 02_FluxtoMag_and_Deredden.ipynb | 0 |
| 03_Hyperbolic_Magnitude.ipynb | 1 |
| 04_Iterate_Tabular_Data.ipynb | 0 |
| 05_Build_Save_Load_Run_Pipeline.ipynb | 1 |
| 06_Rail_Interfaces.ipynb | 0 |

### Creation Notebooks

| Notebook | Return Code |
|----------|-------------|
| 00_Quick_Start_in_Creation.ipynb | 1 |
| 01_Photometric_Realization.ipynb | 1 |
| 02_Photometric_Realization_with_Other_Surveys.ipynb | 1 |
| 03_GridSelection_for_HSC.ipynb | 1 |
| 04_Plotting_interface_skysim_cosmoDC2_COSMOS2020.ipynb | 1 |
| 05_True_Posterior.ipynb | 1 |
| 06_Blending_Degrader.ipynb | 1 |
| 07_DSPS_SED.ipynb | 1 |
| 08_FSPS_SED.ipynb | 1 |
| 09_Spatial_Variability.ipynb | 1 |
| 10_SOM_Spectroscopic_Selector.ipynb | 1 |
| 11_Spectroscopic_Selection_for_zCOSMOS.ipynb | 1 |

### Estimation Notebooks

| Notebook | Return Code |
|----------|-------------|
| 00_Quick_Start_in_Estimation.ipynb | 0 |
| 01_FlexZBoost_PDF_Representation_Comparison.ipynb | 0 |
| 02_BPZ_lite.ipynb | 0 |
| 03_BPZ_lite_Custom_SEDs.ipynb | 0 |
| 04_CMNN.ipynb | 0 |
| 05_DNF.ipynb | 0 |
| 06_GPz.ipynb | 0 |
| 07_NZDir.ipynb | 1 |
| 08_NZDir_pipeline.ipynb | 1 |
| 09_PZFlow.ipynb | 0 |
| 10_YAW.ipynb | 0 |
| 11_SomocluSOM.ipynb | 1 |
| 12_SomocluSOM_Quality_Control.ipynb | 1 |
| 13_Sampled_Summarizers.ipynb | 0 |
| 14_LePhare_LSST.ipynb | 0 |
| 15_LePhare_COSMOS.ipynb | 0 |
| 16_Running_with_different_data.ipynb | 0 |

### Evaluation Notebooks

| Notebook | Return Code |
|----------|-------------|
| 00_Single_Evaluation.ipynb | 0 |
| 01_Evaluation_by_Type.ipynb | 0 |

### Goldenspike Notebooks

| Notebook | Return Code |
|----------|-------------|
| Goldenspike.ipynb | 1 |


