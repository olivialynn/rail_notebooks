# RAIL Notebooks
<!--Note that this readme is automatically updated weekly. To make changes to the text here, modify the template in update-readme.py--otherwise, whatever you write here will just be overwritten!-->

![Static Badge](https://img.shields.io/badge/Read_the_Docs-RAIL_Notebooks-blue)

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

Tables are updated automatically by a weekly workflow.

See more info in each group's log file, in `logs/`.

<!--auto update below-->

**Tables last updated:** January 22, 2026

### Core Notebooks

|   | Notebook |
|---|----------|
| :x: | 00_Useful_Utilities.ipynb |
| :heavy_check_mark: | 01_FileIO_DataStore.ipynb |
| :heavy_check_mark: | 02_FluxtoMag_and_Deredden.ipynb |
| :x: | 03_Hyperbolic_Magnitude.ipynb |
| :heavy_check_mark: | 04_Iterate_Tabular_Data.ipynb |
| :x: | 05_Build_Save_Load_Run_Pipeline.ipynb |
| :heavy_check_mark: | 06_Rail_Interfaces.ipynb |

### Creation Notebooks

|   | Notebook |
|---|----------|
| :x: | 00_Quick_Start_in_Creation.ipynb |
| :x: | 01_Photometric_Realization.ipynb |
| :x: | 02_Photometric_Realization_with_Other_Surveys.ipynb |
| :x: | 03_GridSelection_for_HSC.ipynb |
| :x: | 04_Plotting_interface_skysim_cosmoDC2_COSMOS2020.ipynb |
| :x: | 05_True_Posterior.ipynb |
| :x: | 06_Blending_Degrader.ipynb |
| :x: | 07_DSPS_SED.ipynb |
| :x: | 08_FSPS_SED.ipynb |
| :x: | 09_Spatial_Variability.ipynb |
| :x: | 10_SOM_Spectroscopic_Selector.ipynb |
| :x: | 11_Spectroscopic_Selection_for_zCOSMOS.ipynb |

### Estimation Notebooks

|   | Notebook |
|---|----------|
| :heavy_check_mark: | 00_Quick_Start_in_Estimation.ipynb |
| :heavy_check_mark: | 01_FlexZBoost_PDF_Representation_Comparison.ipynb |
| :heavy_check_mark: | 02_BPZ_lite.ipynb |
| :heavy_check_mark: | 03_BPZ_lite_Custom_SEDs.ipynb |
| :heavy_check_mark: | 04_CMNN.ipynb |
| :heavy_check_mark: | 05_DNF.ipynb |
| :heavy_check_mark: | 06_GPz.ipynb |
| :x: | 07_NZDir.ipynb |
| :x: | 08_NZDir_pipeline.ipynb |
| :heavy_check_mark: | 09_PZFlow.ipynb |
| :heavy_check_mark: | 10_YAW.ipynb |
| :x: | 11_SomocluSOM.ipynb |
| :x: | 12_SomocluSOM_Quality_Control.ipynb |
| :heavy_check_mark: | 13_Sampled_Summarizers.ipynb |
| :heavy_check_mark: | 14_LePhare_LSST.ipynb |
| :heavy_check_mark: | 15_LePhare_COSMOS.ipynb |
| :heavy_check_mark: | 16_Running_with_different_data.ipynb |

### Evaluation Notebooks

|   | Notebook |
|---|----------|
| :heavy_check_mark: | 00_Single_Evaluation.ipynb |
| :heavy_check_mark: | 01_Evaluation_by_Type.ipynb |

### Goldenspike Notebooks

|   | Notebook |
|---|----------|
| :x: | Goldenspike.ipynb |


