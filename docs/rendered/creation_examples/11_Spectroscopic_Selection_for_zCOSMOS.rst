Spectroscopic Selection Degrader to Emulate zCOSMOS Training Samples
====================================================================

last run successfully: April 26, 2023

The spectroscopic_selection degrader can be used to model the
spectroscopic success rates in training sets based on real data. Given a
2-dimensional grid of spec-z success ratio as a function of two
variables (often magnitude, color, or redshift), the degrader will draw
the appropriate fraction of samples from the input data and return a
sample with incompleteness modeled.

The degrader takes the following arguments:

-  ``N_tot``: number of selected sources
-  ``nondetect_val``: non detected magnitude value to be excluded
   (usually 99.0, -99.0 or NaN).
-  ``downsample``: If true, downsample the selected sources into a total
   number of N_tot.
-  ``success_rate_dir``: The path to the directory containing success
   rate files.
-  ``colnames``: a dictionary that includes necessary columns
   (magnitudes, colors and redshift) for selection. For magnitudes, the
   keys are ugrizy; for colors, the keys are, for example, gr standing
   for g-r; for redshift, the key is ‘redshift’. In this demo, zCOSMOS
   takes {‘i’:‘i’, ‘redshift’:‘redshift’} as minimum necessary input

In this quick notebook we’ll select galaxies based on zCOSMOS selection
function.

.. code:: ipython3

    import rail
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import tables_io
    import pandas as pd
    #from rail.core.data import TableHandle
    from rail.utils.path_utils import find_rail_file
    from rail.core.stage import RailStage
    %matplotlib inline 

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

Let’s make fake data for zCOSMOS selection.

.. code:: ipython3

    i = np.random.uniform(low=18, high=25.9675, size=(2000000,))
    gz = np.random.uniform(low=-1.98, high=5.98, size=(2000000,))
    u = np.full_like(i, 20.0, dtype=np.double)
    g = np.full_like(i, 20.0, dtype=np.double)
    r = np.full_like(i, 20.0, dtype=np.double)
    y = np.full_like(i, 20.0, dtype=np.double)
    z = g - gz
    redshift = np.random.uniform(size=len(i)) * 2

Standardize the column names:

.. code:: ipython3

    mockdict = {}
    for label, item in zip(['u', 'g','r','i', 'z','y', 'redshift'], [u,g,r,i,z,y, redshift]):
        mockdict[f'{label}'] = item

np.repeat(item, 100).flatten()

.. code:: ipython3

    df = pd.DataFrame(mockdict)

.. code:: ipython3

    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>redshift</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>25.053536</td>
          <td>20.272371</td>
          <td>20.0</td>
          <td>0.037690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>23.337925</td>
          <td>14.583073</td>
          <td>20.0</td>
          <td>0.115295</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>24.772935</td>
          <td>21.473370</td>
          <td>20.0</td>
          <td>1.910440</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>23.114512</td>
          <td>14.247621</td>
          <td>20.0</td>
          <td>0.255211</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>19.385995</td>
          <td>17.694107</td>
          <td>20.0</td>
          <td>0.482723</td>
        </tr>
      </tbody>
    </table>
    </div>



Now, let’s import the spectroscopic_selections degrader for zCOSMOS.

The ratio file for zCOSMOS is located in the
``RAIL/src/rail/examples/creation/data/success_rate_data/`` directory,
as we are in ``RAIL/examples/creation`` folder named
``zCOSMOS_success.txt``; the binning in i band and redshift are given in
``zCOSMOS_I_sampling.txt`` and ``zCOSMOS_z_sampling.txt``.

We will set a random seed for reproducibility, and set the output file
to write our incomplete catalog to “test_hsc.pq”.

.. code:: ipython3

    import sys
    from rail.creation.degraders import spectroscopic_selections
    from importlib import reload
    from rail.creation.degraders.spectroscopic_selections import SpecSelection_zCOSMOS

.. code:: ipython3

    zcosmos_selecter = SpecSelection_zCOSMOS.make_stage(downsample=False, 
                                                        colnames={'i':'i','redshift':'redshift'})

Let’s run the code and see how long it takes:

.. code:: ipython3

    %%time
    trim_data = zcosmos_selecter(df)


.. parsed-literal::

    Inserting handle into data store.  input: None, SpecSelection_zCOSMOS


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, SpecSelection_zCOSMOS
    CPU times: user 1.42 s, sys: 38.9 ms, total: 1.46 s
    Wall time: 1.44 s


.. code:: ipython3

    trim_data.data.info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    Index: 502722 entries, 4 to 1999997
    Data columns (total 7 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   u         502722 non-null  float64
     1   g         502722 non-null  float64
     2   r         502722 non-null  float64
     3   i         502722 non-null  float64
     4   z         502722 non-null  float64
     5   y         502722 non-null  float64
     6   redshift  502722 non-null  float64
    dtypes: float64(7)
    memory usage: 30.7 MB


And we see that we’ve kept 503967 out of the 2,000,000 galaxies in the
initial sample, so about 25% of the initial sample. To visualize our
cuts, let’s read in the success ratios file and plot our sample overlaid
with an alpha of 0.05, that way the strength of the black dot will give
a visual indication of how many galaxies in each cell we’ve kept.

.. code:: ipython3

    # compare to sum of ratios * 100
    ratio_file=find_rail_file('examples_data/creation_data/data/success_rate_data/zCOSMOS_success.txt')

.. code:: ipython3

    ratios = np.loadtxt(ratio_file)

.. code:: ipython3

    ibin_ = np.arange(18, 22.4, 0.01464226, dtype=np.float64)
    zbin_ = np.arange(0, 1.4, 0.00587002, dtype=np.float64)
    
    ibin, zbin = np.meshgrid(ibin_, zbin_)

.. code:: ipython3

    trim_data.data




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>u</th>
          <th>g</th>
          <th>r</th>
          <th>i</th>
          <th>z</th>
          <th>y</th>
          <th>redshift</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>4</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>19.385995</td>
          <td>17.694107</td>
          <td>20.0</td>
          <td>0.482723</td>
        </tr>
        <tr>
          <th>21</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>18.676592</td>
          <td>14.199407</td>
          <td>20.0</td>
          <td>0.079694</td>
        </tr>
        <tr>
          <th>25</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>19.918538</td>
          <td>20.454380</td>
          <td>20.0</td>
          <td>0.264222</td>
        </tr>
        <tr>
          <th>26</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.325481</td>
          <td>17.342709</td>
          <td>20.0</td>
          <td>0.636450</td>
        </tr>
        <tr>
          <th>28</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>22.003687</td>
          <td>14.538969</td>
          <td>20.0</td>
          <td>0.802549</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>1999979</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.684257</td>
          <td>15.037078</td>
          <td>20.0</td>
          <td>0.414346</td>
        </tr>
        <tr>
          <th>1999980</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>19.477692</td>
          <td>19.275321</td>
          <td>20.0</td>
          <td>0.142882</td>
        </tr>
        <tr>
          <th>1999992</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.644344</td>
          <td>17.803062</td>
          <td>20.0</td>
          <td>1.269723</td>
        </tr>
        <tr>
          <th>1999993</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>19.714894</td>
          <td>21.766142</td>
          <td>20.0</td>
          <td>0.083337</td>
        </tr>
        <tr>
          <th>1999997</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.327899</td>
          <td>20.472590</td>
          <td>20.0</td>
          <td>0.512355</td>
        </tr>
      </tbody>
    </table>
    <p>502722 rows × 7 columns</p>
    </div>



.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.title('zCOSMOS', fontsize=20)
    
    c = plt.pcolormesh(zbin, ibin, ratios.T, cmap='turbo',vmin=0, vmax=1, alpha=0.8)
    plt.scatter(trim_data.data['redshift'], trim_data.data['i'], s=2, c='k',alpha =.05)
    plt.xlabel("redshift", fontsize=15)
    plt.ylabel("i band Magnitude", fontsize=18)
    cb = plt.colorbar(c, label='success rate',orientation='horizontal', pad=0.1)
    cb.set_label(label='success rate', size=15)



.. image:: ../../../docs/rendered/creation_examples/11_Spectroscopic_Selection_for_zCOSMOS_files/../../../docs/rendered/creation_examples/11_Spectroscopic_Selection_for_zCOSMOS_22_0.png


The colormap shows the zCOSMOS success ratios and the strenth of the
black dots shows how many galaxies were actually kept. We see perfect
agreement between our predicted ratios and the actual number of galaxies
kept, the degrader is functioning properly, and we see a nice visual
representation of the resulting spectroscopic sample incompleteness.
