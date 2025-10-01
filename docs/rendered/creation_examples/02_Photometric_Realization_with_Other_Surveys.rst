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
          <td>23.796917</td>
          <td>25.832503</td>
          <td>24.914439</td>
          <td>22.381812</td>
          <td>22.829376</td>
          <td>26.513849</td>
          <td>28.781935</td>
          <td>23.489851</td>
          <td>27.327062</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.384965</td>
          <td>23.963183</td>
          <td>24.294339</td>
          <td>24.367030</td>
          <td>22.062772</td>
          <td>20.883266</td>
          <td>30.555944</td>
          <td>19.508533</td>
          <td>22.682374</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.608672</td>
          <td>27.583491</td>
          <td>23.919450</td>
          <td>23.661894</td>
          <td>18.995745</td>
          <td>24.934324</td>
          <td>30.519315</td>
          <td>27.284659</td>
          <td>20.568214</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.233605</td>
          <td>20.162156</td>
          <td>21.505549</td>
          <td>21.356001</td>
          <td>20.151134</td>
          <td>24.280401</td>
          <td>27.757980</td>
          <td>25.567400</td>
          <td>18.146511</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.368908</td>
          <td>25.530333</td>
          <td>22.293496</td>
          <td>24.148751</td>
          <td>17.594371</td>
          <td>26.153465</td>
          <td>22.101980</td>
          <td>25.638409</td>
          <td>21.801502</td>
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
          <td>25.554337</td>
          <td>22.581778</td>
          <td>17.122218</td>
          <td>20.703980</td>
          <td>26.195326</td>
          <td>19.995959</td>
          <td>23.147056</td>
          <td>24.236378</td>
          <td>25.788301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.653548</td>
          <td>22.857158</td>
          <td>21.826983</td>
          <td>29.299029</td>
          <td>23.362433</td>
          <td>21.525108</td>
          <td>18.956998</td>
          <td>20.672026</td>
          <td>26.398879</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.855895</td>
          <td>22.707482</td>
          <td>25.812426</td>
          <td>17.590613</td>
          <td>23.426172</td>
          <td>21.706109</td>
          <td>24.214937</td>
          <td>21.190651</td>
          <td>18.095140</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.061478</td>
          <td>20.662241</td>
          <td>21.538327</td>
          <td>24.526135</td>
          <td>23.269445</td>
          <td>22.708114</td>
          <td>24.963427</td>
          <td>24.582445</td>
          <td>19.930134</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.178855</td>
          <td>26.162930</td>
          <td>19.699495</td>
          <td>19.587800</td>
          <td>22.824572</td>
          <td>21.634438</td>
          <td>23.982146</td>
          <td>19.518565</td>
          <td>19.878189</td>
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
          <td>23.853826</td>
          <td>0.039612</td>
          <td>25.824738</td>
          <td>0.076792</td>
          <td>24.887353</td>
          <td>0.029445</td>
          <td>22.377877</td>
          <td>0.007070</td>
          <td>22.826853</td>
          <td>0.015268</td>
          <td>26.112736</td>
          <td>0.533564</td>
          <td>28.781935</td>
          <td>23.489851</td>
          <td>27.327062</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.094878</td>
          <td>0.273188</td>
          <td>23.963914</td>
          <td>0.015306</td>
          <td>24.266480</td>
          <td>0.017295</td>
          <td>24.399820</td>
          <td>0.031200</td>
          <td>22.069641</td>
          <td>0.008853</td>
          <td>20.889485</td>
          <td>0.007653</td>
          <td>30.555944</td>
          <td>19.508533</td>
          <td>22.682374</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.552260</td>
          <td>0.030467</td>
          <td>27.307925</td>
          <td>0.272984</td>
          <td>23.884192</td>
          <td>0.012746</td>
          <td>23.650568</td>
          <td>0.016426</td>
          <td>18.995516</td>
          <td>0.005035</td>
          <td>24.975383</td>
          <td>0.218395</td>
          <td>30.519315</td>
          <td>27.284659</td>
          <td>20.568214</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.237761</td>
          <td>0.006505</td>
          <td>20.162441</td>
          <td>0.005047</td>
          <td>21.511597</td>
          <td>0.005217</td>
          <td>21.361772</td>
          <td>0.005414</td>
          <td>20.151561</td>
          <td>0.005195</td>
          <td>24.175202</td>
          <td>0.110177</td>
          <td>27.757980</td>
          <td>25.567400</td>
          <td>18.146511</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.295677</td>
          <td>0.058400</td>
          <td>25.538435</td>
          <td>0.059622</td>
          <td>22.300837</td>
          <td>0.005772</td>
          <td>24.122397</td>
          <td>0.024485</td>
          <td>17.590283</td>
          <td>0.005006</td>
          <td>25.644923</td>
          <td>0.375027</td>
          <td>22.101980</td>
          <td>25.638409</td>
          <td>21.801502</td>
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
          <td>25.204532</td>
          <td>0.129262</td>
          <td>22.587395</td>
          <td>0.006603</td>
          <td>17.127497</td>
          <td>0.005001</td>
          <td>20.710549</td>
          <td>0.005146</td>
          <td>26.456201</td>
          <td>0.344605</td>
          <td>19.992648</td>
          <td>0.005660</td>
          <td>23.147056</td>
          <td>24.236378</td>
          <td>25.788301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.317749</td>
          <td>0.326743</td>
          <td>22.847946</td>
          <td>0.007357</td>
          <td>21.827804</td>
          <td>0.005360</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.382803</td>
          <td>0.024355</td>
          <td>21.512929</td>
          <td>0.011280</td>
          <td>18.956998</td>
          <td>20.672026</td>
          <td>26.398879</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.858773</td>
          <td>0.005070</td>
          <td>22.699998</td>
          <td>0.006897</td>
          <td>25.847463</td>
          <td>0.068898</td>
          <td>17.583320</td>
          <td>0.005003</td>
          <td>23.378376</td>
          <td>0.024262</td>
          <td>21.687862</td>
          <td>0.012848</td>
          <td>24.214937</td>
          <td>21.190651</td>
          <td>18.095140</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.297497</td>
          <td>0.321527</td>
          <td>20.662232</td>
          <td>0.005090</td>
          <td>21.538346</td>
          <td>0.005227</td>
          <td>24.495527</td>
          <td>0.033945</td>
          <td>23.282101</td>
          <td>0.022330</td>
          <td>22.684748</td>
          <td>0.029535</td>
          <td>24.963427</td>
          <td>24.582445</td>
          <td>19.930134</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.163755</td>
          <td>0.051999</td>
          <td>26.156094</td>
          <td>0.102737</td>
          <td>19.694871</td>
          <td>0.005016</td>
          <td>19.589125</td>
          <td>0.005028</td>
          <td>22.824210</td>
          <td>0.015235</td>
          <td>21.642918</td>
          <td>0.012417</td>
          <td>23.982146</td>
          <td>19.518565</td>
          <td>19.878189</td>
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
          <td>23.796917</td>
          <td>25.832503</td>
          <td>24.914439</td>
          <td>22.381812</td>
          <td>22.829376</td>
          <td>26.513849</td>
          <td>27.359885</td>
          <td>0.266871</td>
          <td>23.494496</td>
          <td>0.015651</td>
          <td>26.950352</td>
          <td>0.311307</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.384965</td>
          <td>23.963183</td>
          <td>24.294339</td>
          <td>24.367030</td>
          <td>22.062772</td>
          <td>20.883266</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.511350</td>
          <td>0.005014</td>
          <td>22.684516</td>
          <td>0.008647</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.608672</td>
          <td>27.583491</td>
          <td>23.919450</td>
          <td>23.661894</td>
          <td>18.995745</td>
          <td>24.934324</td>
          <td>28.379504</td>
          <td>0.584135</td>
          <td>27.449158</td>
          <td>0.458649</td>
          <td>20.569789</td>
          <td>0.005100</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.233605</td>
          <td>20.162156</td>
          <td>21.505549</td>
          <td>21.356001</td>
          <td>20.151134</td>
          <td>24.280401</td>
          <td>27.715029</td>
          <td>0.354746</td>
          <td>25.601650</td>
          <td>0.099509</td>
          <td>18.147042</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.368908</td>
          <td>25.530333</td>
          <td>22.293496</td>
          <td>24.148751</td>
          <td>17.594371</td>
          <td>26.153465</td>
          <td>22.104601</td>
          <td>0.005539</td>
          <td>25.639195</td>
          <td>0.102842</td>
          <td>21.803043</td>
          <td>0.005902</td>
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
          <td>25.554337</td>
          <td>22.581778</td>
          <td>17.122218</td>
          <td>20.703980</td>
          <td>26.195326</td>
          <td>19.995959</td>
          <td>23.145451</td>
          <td>0.007971</td>
          <td>24.234304</td>
          <td>0.029560</td>
          <td>25.733799</td>
          <td>0.111720</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.653548</td>
          <td>22.857158</td>
          <td>21.826983</td>
          <td>29.299029</td>
          <td>23.362433</td>
          <td>21.525108</td>
          <td>18.966736</td>
          <td>0.005002</td>
          <td>20.669757</td>
          <td>0.005121</td>
          <td>26.262659</td>
          <td>0.176292</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.855895</td>
          <td>22.707482</td>
          <td>25.812426</td>
          <td>17.590613</td>
          <td>23.426172</td>
          <td>21.706109</td>
          <td>24.202755</td>
          <td>0.017122</td>
          <td>21.198778</td>
          <td>0.005313</td>
          <td>18.092085</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.061478</td>
          <td>20.662241</td>
          <td>21.538327</td>
          <td>24.526135</td>
          <td>23.269445</td>
          <td>22.708114</td>
          <td>24.963257</td>
          <td>0.033135</td>
          <td>24.563577</td>
          <td>0.039593</td>
          <td>19.932477</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.178855</td>
          <td>26.162930</td>
          <td>19.699495</td>
          <td>19.587800</td>
          <td>22.824572</td>
          <td>21.634438</td>
          <td>23.989145</td>
          <td>0.014365</td>
          <td>19.527105</td>
          <td>0.005015</td>
          <td>19.882363</td>
          <td>0.005029</td>
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
          <td>23.796917</td>
          <td>25.832503</td>
          <td>24.914439</td>
          <td>22.381812</td>
          <td>22.829376</td>
          <td>26.513849</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.634009</td>
          <td>0.093765</td>
          <td>28.245809</td>
          <td>2.245170</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.384965</td>
          <td>23.963183</td>
          <td>24.294339</td>
          <td>24.367030</td>
          <td>22.062772</td>
          <td>20.883266</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.511011</td>
          <td>0.005457</td>
          <td>22.611654</td>
          <td>0.041326</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.608672</td>
          <td>27.583491</td>
          <td>23.919450</td>
          <td>23.661894</td>
          <td>18.995745</td>
          <td>24.934324</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.569695</td>
          <td>0.008081</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.233605</td>
          <td>20.162156</td>
          <td>21.505549</td>
          <td>21.356001</td>
          <td>20.151134</td>
          <td>24.280401</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.754767</td>
          <td>0.533964</td>
          <td>18.146543</td>
          <td>0.005046</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.368908</td>
          <td>25.530333</td>
          <td>22.293496</td>
          <td>24.148751</td>
          <td>17.594371</td>
          <td>26.153465</td>
          <td>22.095820</td>
          <td>0.028573</td>
          <td>24.798751</td>
          <td>0.253843</td>
          <td>21.823112</td>
          <td>0.020644</td>
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
          <td>25.554337</td>
          <td>22.581778</td>
          <td>17.122218</td>
          <td>20.703980</td>
          <td>26.195326</td>
          <td>19.995959</td>
          <td>23.035891</td>
          <td>0.065924</td>
          <td>24.144369</td>
          <td>0.146266</td>
          <td>25.360846</td>
          <td>0.429027</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.653548</td>
          <td>22.857158</td>
          <td>21.826983</td>
          <td>29.299029</td>
          <td>23.362433</td>
          <td>21.525108</td>
          <td>18.948183</td>
          <td>0.005239</td>
          <td>20.682284</td>
          <td>0.008139</td>
          <td>29.257260</td>
          <td>3.170048</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.855895</td>
          <td>22.707482</td>
          <td>25.812426</td>
          <td>17.590613</td>
          <td>23.426172</td>
          <td>21.706109</td>
          <td>24.361099</td>
          <td>0.208423</td>
          <td>21.171851</td>
          <td>0.011241</td>
          <td>18.089111</td>
          <td>0.005042</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.061478</td>
          <td>20.662241</td>
          <td>21.538327</td>
          <td>24.526135</td>
          <td>23.269445</td>
          <td>22.708114</td>
          <td>24.465715</td>
          <td>0.227429</td>
          <td>25.042357</td>
          <td>0.309320</td>
          <td>19.931540</td>
          <td>0.006120</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.178855</td>
          <td>26.162930</td>
          <td>19.699495</td>
          <td>19.587800</td>
          <td>22.824572</td>
          <td>21.634438</td>
          <td>24.058378</td>
          <td>0.161297</td>
          <td>19.519053</td>
          <td>0.005463</td>
          <td>19.870997</td>
          <td>0.006012</td>
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


