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

### Core Notebooks
| Rendered?          | Notebook                |
| ------------------ | ----------------------- |
| :heavy_check_mark: | 15_LePhare_COSMOS.ipynb |
| :heavy_check_mark: | 14_LePhare_LSST.ipynb   |
| :x:                | 08_NZDir_pipeline.ipynb |
| :heavy_check_mark: | 02_BPZ_lite.ipynb       |

### Creation Notebooks
| Rendered?          | Notebook                |
| ------------------ | ----------------------- |
| :heavy_check_mark: | 15_LePhare_COSMOS.ipynb |
| :heavy_check_mark: | 14_LePhare_LSST.ipynb   |
| :x:                | 08_NZDir_pipeline.ipynb |
| :heavy_check_mark: | 02_BPZ_lite.ipynb       |

### Estimation Notebooks
| Rendered?          | Notebook                |
| ------------------ | ----------------------- |
| :heavy_check_mark: | 15_LePhare_COSMOS.ipynb |
| :heavy_check_mark: | 14_LePhare_LSST.ipynb   |
| :x:                | 08_NZDir_pipeline.ipynb |
| :heavy_check_mark: | 02_BPZ_lite.ipynb       |


### Evaluation Notebooks
| Rendered?          | Notebook                |
| ------------------ | ----------------------- |
| :heavy_check_mark: | 15_LePhare_COSMOS.ipynb |
| :heavy_check_mark: | 14_LePhare_LSST.ipynb   |
| :x:                | 08_NZDir_pipeline.ipynb |

### Goldenspike Notebook
| Rendered?          | Notebook                |
| ------------------ | ----------------------- |
| :x:                | 08_NZDir_pipeline.ipynb |

