Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


Create a random catalog with ugrizy+YJHF bands as the the true input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    data = np.random.normal(23, 3, size = (1000,9))
    
    data_df = pd.DataFrame(data=data,    # values
                columns=['u', 'g', 'r', 'i', 'z', 'y', 'Y', 'J', 'H'])
    data_truth = PqHandle('input')
    data_truth.set_data(data_df)

.. code:: ipython3

    data_df




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
          <th>Y</th>
          <th>J</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>26.097889</td>
          <td>22.809557</td>
          <td>19.340145</td>
          <td>19.816400</td>
          <td>25.105193</td>
          <td>24.038282</td>
          <td>27.787983</td>
          <td>25.282348</td>
          <td>23.585392</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.600242</td>
          <td>25.928355</td>
          <td>19.749998</td>
          <td>27.282952</td>
          <td>23.709925</td>
          <td>23.754237</td>
          <td>25.157380</td>
          <td>24.293430</td>
          <td>24.215793</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.475897</td>
          <td>25.034205</td>
          <td>27.638884</td>
          <td>21.980932</td>
          <td>22.512525</td>
          <td>20.609389</td>
          <td>22.137299</td>
          <td>23.632958</td>
          <td>24.481071</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.569347</td>
          <td>24.629401</td>
          <td>23.716032</td>
          <td>23.147947</td>
          <td>28.201018</td>
          <td>25.107603</td>
          <td>22.737749</td>
          <td>26.274988</td>
          <td>23.433612</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.120035</td>
          <td>27.257671</td>
          <td>21.733552</td>
          <td>28.199637</td>
          <td>25.476253</td>
          <td>23.199289</td>
          <td>22.301003</td>
          <td>22.785329</td>
          <td>17.274228</td>
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
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>21.231326</td>
          <td>22.720336</td>
          <td>24.901532</td>
          <td>24.730920</td>
          <td>19.905693</td>
          <td>28.899283</td>
          <td>25.890342</td>
          <td>26.765848</td>
          <td>26.058673</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.341661</td>
          <td>28.860230</td>
          <td>22.277072</td>
          <td>23.236047</td>
          <td>21.198325</td>
          <td>23.235796</td>
          <td>23.836381</td>
          <td>19.026787</td>
          <td>26.553223</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.390987</td>
          <td>21.697548</td>
          <td>26.616844</td>
          <td>21.303277</td>
          <td>22.281338</td>
          <td>19.340486</td>
          <td>19.916961</td>
          <td>18.578169</td>
          <td>26.624340</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.299467</td>
          <td>18.631123</td>
          <td>20.168051</td>
          <td>24.278020</td>
          <td>23.409023</td>
          <td>21.560586</td>
          <td>21.428924</td>
          <td>22.896255</td>
          <td>22.538009</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.361411</td>
          <td>23.408198</td>
          <td>17.239709</td>
          <td>24.496001</td>
          <td>25.569110</td>
          <td>20.934031</td>
          <td>19.465338</td>
          <td>29.822789</td>
          <td>20.890812</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 9 columns</p>
    </div>



The LSST error model adds noise to the optical bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_lsst = LSSTErrorModel.make_stage(name="error_model")
    
    samples_w_errs = errorModel_lsst(data_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>Y</th>
          <th>J</th>
          <th>H</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>26.333337</td>
          <td>0.330807</td>
          <td>22.798740</td>
          <td>0.007194</td>
          <td>19.338659</td>
          <td>0.005010</td>
          <td>19.823217</td>
          <td>0.005039</td>
          <td>25.138672</td>
          <td>0.114507</td>
          <td>23.958195</td>
          <td>0.091104</td>
          <td>27.787983</td>
          <td>25.282348</td>
          <td>23.585392</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.031619</td>
          <td>1.078247</td>
          <td>25.945405</td>
          <td>0.085404</td>
          <td>19.751324</td>
          <td>0.005017</td>
          <td>28.060242</td>
          <td>0.657924</td>
          <td>23.701153</td>
          <td>0.032165</td>
          <td>23.684114</td>
          <td>0.071539</td>
          <td>25.157380</td>
          <td>24.293430</td>
          <td>24.215793</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.475208</td>
          <td>0.005521</td>
          <td>25.006008</td>
          <td>0.037223</td>
          <td>27.603072</td>
          <td>0.308407</td>
          <td>21.978902</td>
          <td>0.006115</td>
          <td>22.504026</td>
          <td>0.011887</td>
          <td>20.615304</td>
          <td>0.006759</td>
          <td>22.137299</td>
          <td>23.632958</td>
          <td>24.481071</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.590131</td>
          <td>0.007450</td>
          <td>24.636070</td>
          <td>0.026921</td>
          <td>23.694389</td>
          <td>0.011076</td>
          <td>23.161840</td>
          <td>0.011242</td>
          <td>26.736889</td>
          <td>0.428344</td>
          <td>24.815342</td>
          <td>0.190974</td>
          <td>22.737749</td>
          <td>26.274988</td>
          <td>23.433612</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.109867</td>
          <td>0.009879</td>
          <td>27.240985</td>
          <td>0.258475</td>
          <td>21.733319</td>
          <td>0.005309</td>
          <td>30.142648</td>
          <td>2.060211</td>
          <td>25.649998</td>
          <td>0.177789</td>
          <td>23.247836</td>
          <td>0.048586</td>
          <td>22.301003</td>
          <td>22.785329</td>
          <td>17.274228</td>
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
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>21.232858</td>
          <td>0.006495</td>
          <td>22.715952</td>
          <td>0.006942</td>
          <td>24.874831</td>
          <td>0.029123</td>
          <td>24.720889</td>
          <td>0.041436</td>
          <td>19.901678</td>
          <td>0.005132</td>
          <td>27.633392</td>
          <td>1.383026</td>
          <td>25.890342</td>
          <td>26.765848</td>
          <td>26.058673</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.324127</td>
          <td>0.011387</td>
          <td>27.653762</td>
          <td>0.359837</td>
          <td>22.269803</td>
          <td>0.005735</td>
          <td>23.207084</td>
          <td>0.011618</td>
          <td>21.196387</td>
          <td>0.006038</td>
          <td>23.209598</td>
          <td>0.046965</td>
          <td>23.836381</td>
          <td>19.026787</td>
          <td>26.553223</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.398168</td>
          <td>0.011996</td>
          <td>21.700792</td>
          <td>0.005412</td>
          <td>26.779370</td>
          <td>0.155505</td>
          <td>21.301992</td>
          <td>0.005375</td>
          <td>22.284560</td>
          <td>0.010169</td>
          <td>19.337475</td>
          <td>0.005231</td>
          <td>19.916961</td>
          <td>18.578169</td>
          <td>26.624340</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.285594</td>
          <td>0.024244</td>
          <td>18.626180</td>
          <td>0.005008</td>
          <td>20.169246</td>
          <td>0.005030</td>
          <td>24.353246</td>
          <td>0.029949</td>
          <td>23.408270</td>
          <td>0.024898</td>
          <td>21.566304</td>
          <td>0.011728</td>
          <td>21.428924</td>
          <td>22.896255</td>
          <td>22.538009</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.593714</td>
          <td>0.405322</td>
          <td>23.397010</td>
          <td>0.010062</td>
          <td>17.239688</td>
          <td>0.005001</td>
          <td>24.507689</td>
          <td>0.034311</td>
          <td>25.936719</td>
          <td>0.226175</td>
          <td>20.937985</td>
          <td>0.007848</td>
          <td>19.465338</td>
          <td>29.822789</td>
          <td>20.890812</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "ugrizy":
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


The Roman error model adds noise to the infrared bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Roman = RomanErrorModel.make_stage(name="error_model", )
    


.. code:: ipython3

    errorModel_Roman.config['m5']['Y'] = 27.0

.. code:: ipython3

    errorModel_Roman.config['theta']['Y'] = 27.0

.. code:: ipython3

    samples_w_errs_roman = errorModel_Roman(data_truth)
    samples_w_errs_roman()


.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>Y</th>
          <th>Y_err</th>
          <th>J</th>
          <th>J_err</th>
          <th>H</th>
          <th>H_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>26.097889</td>
          <td>22.809557</td>
          <td>19.340145</td>
          <td>19.816400</td>
          <td>25.105193</td>
          <td>24.038282</td>
          <td>27.776610</td>
          <td>0.372262</td>
          <td>25.350152</td>
          <td>0.079721</td>
          <td>23.576089</td>
          <td>0.016745</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.600242</td>
          <td>25.928355</td>
          <td>19.749998</td>
          <td>27.282952</td>
          <td>23.709925</td>
          <td>23.754237</td>
          <td>25.144742</td>
          <td>0.038935</td>
          <td>24.239650</td>
          <td>0.029700</td>
          <td>24.192667</td>
          <td>0.028494</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.475897</td>
          <td>25.034205</td>
          <td>27.638884</td>
          <td>21.980932</td>
          <td>22.512525</td>
          <td>20.609389</td>
          <td>22.132136</td>
          <td>0.005565</td>
          <td>23.624874</td>
          <td>0.017442</td>
          <td>24.478641</td>
          <td>0.036711</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.569347</td>
          <td>24.629401</td>
          <td>23.716032</td>
          <td>23.147947</td>
          <td>28.201018</td>
          <td>25.107603</td>
          <td>22.727738</td>
          <td>0.006548</td>
          <td>26.234465</td>
          <td>0.172115</td>
          <td>23.424956</td>
          <td>0.014787</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.120035</td>
          <td>27.257671</td>
          <td>21.733552</td>
          <td>28.199637</td>
          <td>25.476253</td>
          <td>23.199289</td>
          <td>22.295547</td>
          <td>0.005750</td>
          <td>22.797641</td>
          <td>0.009288</td>
          <td>17.278054</td>
          <td>0.005000</td>
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
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>21.231326</td>
          <td>22.720336</td>
          <td>24.901532</td>
          <td>24.730920</td>
          <td>19.905693</td>
          <td>28.899283</td>
          <td>25.846944</td>
          <td>0.072755</td>
          <td>27.499999</td>
          <td>0.476435</td>
          <td>26.376325</td>
          <td>0.194092</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.341661</td>
          <td>28.860230</td>
          <td>22.277072</td>
          <td>23.236047</td>
          <td>21.198325</td>
          <td>23.235796</td>
          <td>23.821538</td>
          <td>0.012585</td>
          <td>19.012682</td>
          <td>0.005006</td>
          <td>26.352528</td>
          <td>0.190235</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.390987</td>
          <td>21.697548</td>
          <td>26.616844</td>
          <td>21.303277</td>
          <td>22.281338</td>
          <td>19.340486</td>
          <td>19.927217</td>
          <td>0.005010</td>
          <td>18.579151</td>
          <td>0.005003</td>
          <td>26.527665</td>
          <td>0.220343</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.299467</td>
          <td>18.631123</td>
          <td>20.168051</td>
          <td>24.278020</td>
          <td>23.409023</td>
          <td>21.560586</td>
          <td>21.431480</td>
          <td>0.005162</td>
          <td>22.887036</td>
          <td>0.009859</td>
          <td>22.522301</td>
          <td>0.007870</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.361411</td>
          <td>23.408198</td>
          <td>17.239709</td>
          <td>24.496001</td>
          <td>25.569110</td>
          <td>20.934031</td>
          <td>19.466169</td>
          <td>0.005004</td>
          <td>29.195588</td>
          <td>1.398517</td>
          <td>20.886336</td>
          <td>0.005179</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 12 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "YJH":
        # pull out the magnitudes and errors
        mags = samples_w_errs_roman.data[band].to_numpy()
        errs = samples_w_errs_roman.data[band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


.. parsed-literal::

    Inserting handle into data store.  input: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>Y</th>
          <th>Y_err</th>
          <th>J</th>
          <th>J_err</th>
          <th>H</th>
          <th>H_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>26.097889</td>
          <td>22.809557</td>
          <td>19.340145</td>
          <td>19.816400</td>
          <td>25.105193</td>
          <td>24.038282</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.416132</td>
          <td>0.414635</td>
          <td>23.777272</td>
          <td>0.116041</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.600242</td>
          <td>25.928355</td>
          <td>19.749998</td>
          <td>27.282952</td>
          <td>23.709925</td>
          <td>23.754237</td>
          <td>25.822378</td>
          <td>0.645880</td>
          <td>24.407198</td>
          <td>0.183080</td>
          <td>24.169690</td>
          <td>0.162865</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.475897</td>
          <td>25.034205</td>
          <td>27.638884</td>
          <td>21.980932</td>
          <td>22.512525</td>
          <td>20.609389</td>
          <td>22.179239</td>
          <td>0.030758</td>
          <td>23.691607</td>
          <td>0.098635</td>
          <td>24.303549</td>
          <td>0.182515</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.569347</td>
          <td>24.629401</td>
          <td>23.716032</td>
          <td>23.147947</td>
          <td>28.201018</td>
          <td>25.107603</td>
          <td>22.705463</td>
          <td>0.049119</td>
          <td>26.057352</td>
          <td>0.661700</td>
          <td>23.426825</td>
          <td>0.085311</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.120035</td>
          <td>27.257671</td>
          <td>21.733552</td>
          <td>28.199637</td>
          <td>25.476253</td>
          <td>23.199289</td>
          <td>22.220343</td>
          <td>0.031898</td>
          <td>22.785137</td>
          <td>0.044123</td>
          <td>17.268763</td>
          <td>0.005009</td>
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
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>21.231326</td>
          <td>22.720336</td>
          <td>24.901532</td>
          <td>24.730920</td>
          <td>19.905693</td>
          <td>28.899283</td>
          <td>25.335961</td>
          <td>0.454120</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.764670</td>
          <td>1.104485</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.341661</td>
          <td>28.860230</td>
          <td>22.277072</td>
          <td>23.236047</td>
          <td>21.198325</td>
          <td>23.235796</td>
          <td>23.968135</td>
          <td>0.149287</td>
          <td>19.026233</td>
          <td>0.005192</td>
          <td>25.126427</td>
          <td>0.357935</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.390987</td>
          <td>21.697548</td>
          <td>26.616844</td>
          <td>21.303277</td>
          <td>22.281338</td>
          <td>19.340486</td>
          <td>19.913934</td>
          <td>0.006285</td>
          <td>18.588234</td>
          <td>0.005087</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.299467</td>
          <td>18.631123</td>
          <td>20.168051</td>
          <td>24.278020</td>
          <td>23.409023</td>
          <td>21.560586</td>
          <td>21.450171</td>
          <td>0.016388</td>
          <td>23.036441</td>
          <td>0.055202</td>
          <td>22.558533</td>
          <td>0.039416</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.361411</td>
          <td>23.408198</td>
          <td>17.239709</td>
          <td>24.496001</td>
          <td>25.569110</td>
          <td>20.934031</td>
          <td>19.460201</td>
          <td>0.005594</td>
          <td>26.011739</td>
          <td>0.641123</td>
          <td>20.907547</td>
          <td>0.009998</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 12 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "YJH":
        # pull out the magnitudes and errors
        mags = samples_w_errs_Euclid.data[band].to_numpy()
        errs = samples_w_errs_Euclid.data[band + "_err"].to_numpy()
    
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
    
        # plot errs vs mags
        ax.plot(mags, errs, label=band)
    
    ax.legend()
    ax.set(xlabel="Magnitude (AB)", ylabel="Error (mags)")
    plt.show()




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


