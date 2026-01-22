# RAIL Notebooks
*Note that this readme is automatically updated weekly. To make changes to the text here, modify the template in update-readme.py.*

## Overview

RAIL Notebooks is part of the [RAIL ecosystem](https://github.com/lsstdesc/rail), where it maintains the rendered notebooks seen in the RAIL documentation. This repo is built as its own [Read the Docs subproject](https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/index.html), which is linked to the main [RAIL Read the Docs project](https://rail-hub.readthedocs.io/en/latest/).

### How does it work?

The main functionality is in [render_notebooks.py](https://github.com/olivialynn/rail_notebooks/blob/main/src/rail_notebooks/render_notebooks.py), which is run weekly by the [render-notebooks.yml](https://github.com/olivialynn/rail_notebooks/blob/main/.github/workflows/render-notebooks.yml) workflow, and will:
- gather all notebooks in RAIL
- render them to .rst files using nbconvert
- save the rendered versions in this repository, in the [docs/](https://github.com/olivialynn/rail_notebooks/tree/main/docs) directory
- output logs in the [logs/](https://github.com/olivialynn/rail_notebooks/tree/main/logs) directory
- finally, update the tables below for at-a-glance monitoring

### How can I add my RAIL notebook?

All notebooks placed in the main repository's core, creation, estimation, evaluation, or goldenspike notebook repositories are automatically included.

We strongly recommend that you only use one title in your notebook, at the top (ie, a single # in markdown as in `# My Notebook`)--otherwise the second title will be interpreted as a new notebook in the navigation on Read the Docs.

## Notebook Status (last updated yyyy-mm-dd)
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

