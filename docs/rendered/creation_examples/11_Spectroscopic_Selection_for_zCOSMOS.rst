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
          <td>19.424408</td>
          <td>16.979426</td>
          <td>20.0</td>
          <td>1.757157</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>25.043681</td>
          <td>21.252637</td>
          <td>20.0</td>
          <td>1.826549</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>24.929123</td>
          <td>17.594647</td>
          <td>20.0</td>
          <td>1.934078</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>25.823039</td>
          <td>14.369084</td>
          <td>20.0</td>
          <td>1.414721</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>25.508232</td>
          <td>15.017949</td>
          <td>20.0</td>
          <td>0.632912</td>
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
    CPU times: user 1.42 s, sys: 37.7 ms, total: 1.46 s
    Wall time: 1.44 s


.. code:: ipython3

    trim_data.data.info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    Index: 503810 entries, 10 to 1999998
    Data columns (total 7 columns):
     #   Column    Non-Null Count   Dtype  
    ---  ------    --------------   -----  
     0   u         503810 non-null  float64
     1   g         503810 non-null  float64
     2   r         503810 non-null  float64
     3   i         503810 non-null  float64
     4   z         503810 non-null  float64
     5   y         503810 non-null  float64
     6   redshift  503810 non-null  float64
    dtypes: float64(7)
    memory usage: 30.8 MB


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
          <th>10</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>22.020710</td>
          <td>18.762508</td>
          <td>20.0</td>
          <td>0.326955</td>
        </tr>
        <tr>
          <th>15</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.102050</td>
          <td>17.514079</td>
          <td>20.0</td>
          <td>0.739134</td>
        </tr>
        <tr>
          <th>17</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.256583</td>
          <td>20.277638</td>
          <td>20.0</td>
          <td>1.132784</td>
        </tr>
        <tr>
          <th>19</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>22.336706</td>
          <td>17.824993</td>
          <td>20.0</td>
          <td>0.867200</td>
        </tr>
        <tr>
          <th>28</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.812262</td>
          <td>21.261706</td>
          <td>20.0</td>
          <td>0.433181</td>
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
          <th>1999985</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>18.600941</td>
          <td>20.963294</td>
          <td>20.0</td>
          <td>0.151162</td>
        </tr>
        <tr>
          <th>1999987</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.149530</td>
          <td>15.156604</td>
          <td>20.0</td>
          <td>0.644130</td>
        </tr>
        <tr>
          <th>1999990</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>21.096556</td>
          <td>21.520341</td>
          <td>20.0</td>
          <td>1.117666</td>
        </tr>
        <tr>
          <th>1999992</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>22.053252</td>
          <td>16.034017</td>
          <td>20.0</td>
          <td>0.799143</td>
        </tr>
        <tr>
          <th>1999998</th>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.0</td>
          <td>20.852133</td>
          <td>14.588141</td>
          <td>20.0</td>
          <td>0.958358</td>
        </tr>
      </tbody>
    </table>
    <p>503810 rows × 7 columns</p>
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
