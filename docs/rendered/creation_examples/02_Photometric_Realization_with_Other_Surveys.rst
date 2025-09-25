Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


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
          <td>20.178353</td>
          <td>24.278878</td>
          <td>27.186704</td>
          <td>25.722156</td>
          <td>24.587403</td>
          <td>18.540004</td>
          <td>18.494662</td>
          <td>23.376029</td>
          <td>25.413457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.157900</td>
          <td>20.593556</td>
          <td>27.710649</td>
          <td>22.500748</td>
          <td>30.625135</td>
          <td>23.011896</td>
          <td>25.558675</td>
          <td>24.322773</td>
          <td>23.241556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.312643</td>
          <td>23.149472</td>
          <td>19.236355</td>
          <td>22.208744</td>
          <td>25.459880</td>
          <td>25.493794</td>
          <td>26.797727</td>
          <td>20.758113</td>
          <td>20.824502</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.924909</td>
          <td>22.766273</td>
          <td>24.416959</td>
          <td>18.333824</td>
          <td>18.605340</td>
          <td>22.430375</td>
          <td>25.677504</td>
          <td>24.747959</td>
          <td>22.181863</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.441493</td>
          <td>22.704527</td>
          <td>21.509532</td>
          <td>23.855177</td>
          <td>24.239148</td>
          <td>23.890891</td>
          <td>21.235060</td>
          <td>20.922612</td>
          <td>27.599162</td>
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
          <td>20.780542</td>
          <td>24.267780</td>
          <td>27.909265</td>
          <td>23.873024</td>
          <td>22.891641</td>
          <td>17.319760</td>
          <td>26.083516</td>
          <td>15.000875</td>
          <td>27.728593</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.176104</td>
          <td>21.182045</td>
          <td>17.705632</td>
          <td>25.176165</td>
          <td>26.683970</td>
          <td>29.887807</td>
          <td>23.555201</td>
          <td>24.161891</td>
          <td>24.515285</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.906126</td>
          <td>20.414348</td>
          <td>23.681746</td>
          <td>27.446978</td>
          <td>20.746593</td>
          <td>29.103998</td>
          <td>22.105545</td>
          <td>22.159201</td>
          <td>19.683905</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.137794</td>
          <td>19.494874</td>
          <td>23.290302</td>
          <td>21.648983</td>
          <td>23.558420</td>
          <td>19.798744</td>
          <td>28.587762</td>
          <td>20.919217</td>
          <td>21.037244</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.313000</td>
          <td>23.487462</td>
          <td>29.037316</td>
          <td>28.070776</td>
          <td>23.689449</td>
          <td>25.233930</td>
          <td>23.266824</td>
          <td>25.305866</td>
          <td>22.722889</td>
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
          <td>20.178358</td>
          <td>0.005350</td>
          <td>24.265789</td>
          <td>0.019612</td>
          <td>26.987271</td>
          <td>0.185599</td>
          <td>25.745041</td>
          <td>0.102522</td>
          <td>24.539226</td>
          <td>0.067590</td>
          <td>18.542032</td>
          <td>0.005068</td>
          <td>18.494662</td>
          <td>23.376029</td>
          <td>25.413457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.163600</td>
          <td>0.010224</td>
          <td>20.598661</td>
          <td>0.005083</td>
          <td>27.471056</td>
          <td>0.277250</td>
          <td>22.497431</td>
          <td>0.007475</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.085193</td>
          <td>0.042057</td>
          <td>25.558675</td>
          <td>24.322773</td>
          <td>23.241556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.333576</td>
          <td>0.011462</td>
          <td>23.146453</td>
          <td>0.008604</td>
          <td>19.241750</td>
          <td>0.005009</td>
          <td>22.198595</td>
          <td>0.006574</td>
          <td>25.328737</td>
          <td>0.135033</td>
          <td>25.030896</td>
          <td>0.228710</td>
          <td>26.797727</td>
          <td>20.758113</td>
          <td>20.824502</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.924688</td>
          <td>0.005971</td>
          <td>22.778912</td>
          <td>0.007131</td>
          <td>24.382387</td>
          <td>0.019053</td>
          <td>18.328725</td>
          <td>0.005006</td>
          <td>18.604989</td>
          <td>0.005021</td>
          <td>22.418484</td>
          <td>0.023427</td>
          <td>25.677504</td>
          <td>24.747959</td>
          <td>22.181863</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.449580</td>
          <td>0.007020</td>
          <td>22.696571</td>
          <td>0.006887</td>
          <td>21.516404</td>
          <td>0.005219</td>
          <td>23.843297</td>
          <td>0.019283</td>
          <td>24.176846</td>
          <td>0.049007</td>
          <td>23.845331</td>
          <td>0.082487</td>
          <td>21.235060</td>
          <td>20.922612</td>
          <td>27.599162</td>
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
          <td>20.776625</td>
          <td>0.005790</td>
          <td>24.271328</td>
          <td>0.019704</td>
          <td>28.045564</td>
          <td>0.435630</td>
          <td>23.875384</td>
          <td>0.019813</td>
          <td>22.906110</td>
          <td>0.016282</td>
          <td>17.319057</td>
          <td>0.005013</td>
          <td>26.083516</td>
          <td>15.000875</td>
          <td>27.728593</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.076817</td>
          <td>0.269206</td>
          <td>21.185609</td>
          <td>0.005190</td>
          <td>17.701024</td>
          <td>0.005002</td>
          <td>25.145098</td>
          <td>0.060383</td>
          <td>27.382827</td>
          <td>0.683465</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.555201</td>
          <td>24.161891</td>
          <td>24.515285</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.926087</td>
          <td>0.017982</td>
          <td>20.421725</td>
          <td>0.005065</td>
          <td>23.684127</td>
          <td>0.010995</td>
          <td>27.690532</td>
          <td>0.505323</td>
          <td>20.748778</td>
          <td>0.005507</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.105545</td>
          <td>22.159201</td>
          <td>19.683905</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.142909</td>
          <td>0.005005</td>
          <td>19.492818</td>
          <td>0.005021</td>
          <td>23.289402</td>
          <td>0.008501</td>
          <td>21.650491</td>
          <td>0.005659</td>
          <td>23.559637</td>
          <td>0.028408</td>
          <td>19.790400</td>
          <td>0.005477</td>
          <td>28.587762</td>
          <td>20.919217</td>
          <td>21.037244</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.310700</td>
          <td>0.005118</td>
          <td>23.504004</td>
          <td>0.010825</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.599225</td>
          <td>1.617714</td>
          <td>23.692400</td>
          <td>0.031918</td>
          <td>25.425914</td>
          <td>0.315535</td>
          <td>23.266824</td>
          <td>25.305866</td>
          <td>22.722889</td>
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




.. image:: ../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_files/../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_8_0.png


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
          <td>20.178353</td>
          <td>24.278878</td>
          <td>27.186704</td>
          <td>25.722156</td>
          <td>24.587403</td>
          <td>18.540004</td>
          <td>18.492503</td>
          <td>0.005001</td>
          <td>23.396556</td>
          <td>0.014452</td>
          <td>25.448686</td>
          <td>0.086973</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.157900</td>
          <td>20.593556</td>
          <td>27.710649</td>
          <td>22.500748</td>
          <td>30.625135</td>
          <td>23.011896</td>
          <td>25.569228</td>
          <td>0.056838</td>
          <td>24.310545</td>
          <td>0.031622</td>
          <td>23.202802</td>
          <td>0.012405</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.312643</td>
          <td>23.149472</td>
          <td>19.236355</td>
          <td>22.208744</td>
          <td>25.459880</td>
          <td>25.493794</td>
          <td>27.346069</td>
          <td>0.263876</td>
          <td>20.753608</td>
          <td>0.005140</td>
          <td>20.823492</td>
          <td>0.005159</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.924909</td>
          <td>22.766273</td>
          <td>24.416959</td>
          <td>18.333824</td>
          <td>18.605340</td>
          <td>22.430375</td>
          <td>25.662232</td>
          <td>0.061744</td>
          <td>24.738961</td>
          <td>0.046291</td>
          <td>22.160957</td>
          <td>0.006633</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.441493</td>
          <td>22.704527</td>
          <td>21.509532</td>
          <td>23.855177</td>
          <td>24.239148</td>
          <td>23.890891</td>
          <td>21.235578</td>
          <td>0.005113</td>
          <td>20.920568</td>
          <td>0.005190</td>
          <td>27.630002</td>
          <td>0.524411</td>
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
          <td>20.780542</td>
          <td>24.267780</td>
          <td>27.909265</td>
          <td>23.873024</td>
          <td>22.891641</td>
          <td>17.319760</td>
          <td>26.110754</td>
          <td>0.091865</td>
          <td>15.000693</td>
          <td>0.005000</td>
          <td>29.728118</td>
          <td>1.808275</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.176104</td>
          <td>21.182045</td>
          <td>17.705632</td>
          <td>25.176165</td>
          <td>26.683970</td>
          <td>29.887807</td>
          <td>23.562350</td>
          <td>0.010388</td>
          <td>24.150322</td>
          <td>0.027451</td>
          <td>24.532049</td>
          <td>0.038497</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.906126</td>
          <td>20.414348</td>
          <td>23.681746</td>
          <td>27.446978</td>
          <td>20.746593</td>
          <td>29.103998</td>
          <td>22.100657</td>
          <td>0.005535</td>
          <td>22.151101</td>
          <td>0.006607</td>
          <td>19.676832</td>
          <td>0.005020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.137794</td>
          <td>19.494874</td>
          <td>23.290302</td>
          <td>21.648983</td>
          <td>23.558420</td>
          <td>19.798744</td>
          <td>27.839500</td>
          <td>0.390892</td>
          <td>20.921381</td>
          <td>0.005190</td>
          <td>21.039032</td>
          <td>0.005235</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.313000</td>
          <td>23.487462</td>
          <td>29.037316</td>
          <td>28.070776</td>
          <td>23.689449</td>
          <td>25.233930</td>
          <td>23.273631</td>
          <td>0.008590</td>
          <td>25.233960</td>
          <td>0.071922</td>
          <td>22.724958</td>
          <td>0.008866</td>
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




.. image:: ../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_files/../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_14_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


.. parsed-literal::

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
          <td>20.178353</td>
          <td>24.278878</td>
          <td>27.186704</td>
          <td>25.722156</td>
          <td>24.587403</td>
          <td>18.540004</td>
          <td>18.499150</td>
          <td>0.005106</td>
          <td>23.517897</td>
          <td>0.084641</td>
          <td>25.488623</td>
          <td>0.472408</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.157900</td>
          <td>20.593556</td>
          <td>27.710649</td>
          <td>22.500748</td>
          <td>30.625135</td>
          <td>23.011896</td>
          <td>26.575039</td>
          <td>1.048124</td>
          <td>24.673945</td>
          <td>0.228989</td>
          <td>23.114889</td>
          <td>0.064705</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.312643</td>
          <td>23.149472</td>
          <td>19.236355</td>
          <td>22.208744</td>
          <td>25.459880</td>
          <td>25.493794</td>
          <td>27.385978</td>
          <td>1.616256</td>
          <td>20.753661</td>
          <td>0.008487</td>
          <td>20.814031</td>
          <td>0.009388</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.924909</td>
          <td>22.766273</td>
          <td>24.416959</td>
          <td>18.333824</td>
          <td>18.605340</td>
          <td>22.430375</td>
          <td>26.404183</td>
          <td>0.945545</td>
          <td>25.562795</td>
          <td>0.463366</td>
          <td>22.133376</td>
          <td>0.027045</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.441493</td>
          <td>22.704527</td>
          <td>21.509532</td>
          <td>23.855177</td>
          <td>24.239148</td>
          <td>23.890891</td>
          <td>21.240871</td>
          <td>0.013821</td>
          <td>20.919418</td>
          <td>0.009421</td>
          <td>26.002518</td>
          <td>0.682544</td>
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
          <td>20.780542</td>
          <td>24.267780</td>
          <td>27.909265</td>
          <td>23.873024</td>
          <td>22.891641</td>
          <td>17.319760</td>
          <td>25.133101</td>
          <td>0.388962</td>
          <td>14.999979</td>
          <td>0.005000</td>
          <td>27.415978</td>
          <td>1.562453</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.176104</td>
          <td>21.182045</td>
          <td>17.705632</td>
          <td>25.176165</td>
          <td>26.683970</td>
          <td>29.887807</td>
          <td>23.503944</td>
          <td>0.099710</td>
          <td>24.001106</td>
          <td>0.129236</td>
          <td>24.469524</td>
          <td>0.209899</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.906126</td>
          <td>20.414348</td>
          <td>23.681746</td>
          <td>27.446978</td>
          <td>20.746593</td>
          <td>29.103998</td>
          <td>22.069999</td>
          <td>0.027930</td>
          <td>22.162867</td>
          <td>0.025422</td>
          <td>19.683056</td>
          <td>0.005734</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.137794</td>
          <td>19.494874</td>
          <td>23.290302</td>
          <td>21.648983</td>
          <td>23.558420</td>
          <td>19.798744</td>
          <td>26.161105</td>
          <td>0.810907</td>
          <td>20.924859</td>
          <td>0.009455</td>
          <td>21.056820</td>
          <td>0.011118</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.313000</td>
          <td>23.487462</td>
          <td>29.037316</td>
          <td>28.070776</td>
          <td>23.689449</td>
          <td>25.233930</td>
          <td>23.227844</td>
          <td>0.078162</td>
          <td>25.192858</td>
          <td>0.348613</td>
          <td>22.645409</td>
          <td>0.042588</td>
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




.. image:: ../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_files/../../../docs/rendered/creation_examples/02_Photometric_Realization_with_Other_Surveys_17_0.png


