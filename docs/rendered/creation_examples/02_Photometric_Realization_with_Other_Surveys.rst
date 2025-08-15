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
          <td>23.502950</td>
          <td>26.459357</td>
          <td>27.240213</td>
          <td>27.039947</td>
          <td>22.788214</td>
          <td>25.295602</td>
          <td>20.107831</td>
          <td>22.948181</td>
          <td>26.861375</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.873457</td>
          <td>17.685494</td>
          <td>20.339847</td>
          <td>21.301636</td>
          <td>18.084960</td>
          <td>20.775624</td>
          <td>23.627680</td>
          <td>22.090723</td>
          <td>23.187336</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.620305</td>
          <td>22.544264</td>
          <td>25.741914</td>
          <td>21.687124</td>
          <td>20.156011</td>
          <td>17.635615</td>
          <td>20.857974</td>
          <td>21.116199</td>
          <td>21.304333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.996961</td>
          <td>26.658135</td>
          <td>21.862419</td>
          <td>24.459888</td>
          <td>21.737323</td>
          <td>20.598460</td>
          <td>25.449199</td>
          <td>25.380071</td>
          <td>21.836085</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.899406</td>
          <td>19.956528</td>
          <td>23.020750</td>
          <td>16.385081</td>
          <td>24.039810</td>
          <td>22.197564</td>
          <td>15.658224</td>
          <td>27.050426</td>
          <td>20.493046</td>
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
          <td>21.363912</td>
          <td>25.346212</td>
          <td>22.040602</td>
          <td>21.601316</td>
          <td>24.288050</td>
          <td>22.897516</td>
          <td>22.566047</td>
          <td>23.209700</td>
          <td>26.519075</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.616598</td>
          <td>21.923674</td>
          <td>23.619002</td>
          <td>23.283611</td>
          <td>28.224596</td>
          <td>21.112721</td>
          <td>27.734206</td>
          <td>19.898536</td>
          <td>27.367678</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.571503</td>
          <td>22.002477</td>
          <td>24.160253</td>
          <td>25.972750</td>
          <td>21.136998</td>
          <td>26.635737</td>
          <td>24.014189</td>
          <td>29.452182</td>
          <td>28.473639</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.534108</td>
          <td>26.492987</td>
          <td>22.345804</td>
          <td>23.860724</td>
          <td>19.051417</td>
          <td>23.886442</td>
          <td>25.753904</td>
          <td>30.914663</td>
          <td>25.698355</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.153916</td>
          <td>24.920552</td>
          <td>25.602284</td>
          <td>21.517192</td>
          <td>22.836775</td>
          <td>24.505152</td>
          <td>27.102272</td>
          <td>26.648769</td>
          <td>25.799738</td>
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
          <td>23.535737</td>
          <td>0.030035</td>
          <td>26.718691</td>
          <td>0.167031</td>
          <td>27.122690</td>
          <td>0.207993</td>
          <td>27.587207</td>
          <td>0.468026</td>
          <td>22.791982</td>
          <td>0.014846</td>
          <td>25.011057</td>
          <td>0.224974</td>
          <td>20.107831</td>
          <td>22.948181</td>
          <td>26.861375</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.852119</td>
          <td>0.039553</td>
          <td>17.681548</td>
          <td>0.005003</td>
          <td>20.344504</td>
          <td>0.005038</td>
          <td>21.295922</td>
          <td>0.005372</td>
          <td>18.091920</td>
          <td>0.005011</td>
          <td>20.787074</td>
          <td>0.007280</td>
          <td>23.627680</td>
          <td>22.090723</td>
          <td>23.187336</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.619624</td>
          <td>0.005635</td>
          <td>22.535072</td>
          <td>0.006482</td>
          <td>25.752027</td>
          <td>0.063311</td>
          <td>21.680189</td>
          <td>0.005692</td>
          <td>20.160091</td>
          <td>0.005198</td>
          <td>17.640347</td>
          <td>0.005020</td>
          <td>20.857974</td>
          <td>21.116199</td>
          <td>21.304333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.990387</td>
          <td>0.006065</td>
          <td>26.638633</td>
          <td>0.155999</td>
          <td>21.860391</td>
          <td>0.005380</td>
          <td>24.480876</td>
          <td>0.033509</td>
          <td>21.737676</td>
          <td>0.007388</td>
          <td>20.599989</td>
          <td>0.006718</td>
          <td>25.449199</td>
          <td>25.380071</td>
          <td>21.836085</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.877997</td>
          <td>0.040459</td>
          <td>19.947611</td>
          <td>0.005036</td>
          <td>23.011091</td>
          <td>0.007335</td>
          <td>16.375303</td>
          <td>0.005001</td>
          <td>23.975626</td>
          <td>0.040995</td>
          <td>22.180983</td>
          <td>0.019132</td>
          <td>15.658224</td>
          <td>27.050426</td>
          <td>20.493046</td>
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
          <td>21.363978</td>
          <td>0.006794</td>
          <td>25.251712</td>
          <td>0.046251</td>
          <td>22.044489</td>
          <td>0.005511</td>
          <td>21.609782</td>
          <td>0.005618</td>
          <td>24.219435</td>
          <td>0.050896</td>
          <td>22.896512</td>
          <td>0.035588</td>
          <td>22.566047</td>
          <td>23.209700</td>
          <td>26.519075</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.618650</td>
          <td>0.007548</td>
          <td>21.925056</td>
          <td>0.005582</td>
          <td>23.614529</td>
          <td>0.010470</td>
          <td>23.275249</td>
          <td>0.012221</td>
          <td>26.715767</td>
          <td>0.421507</td>
          <td>21.111417</td>
          <td>0.008649</td>
          <td>27.734206</td>
          <td>19.898536</td>
          <td>27.367678</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.721032</td>
          <td>0.893374</td>
          <td>22.006594</td>
          <td>0.005660</td>
          <td>24.147375</td>
          <td>0.015686</td>
          <td>26.293237</td>
          <td>0.164732</td>
          <td>21.139987</td>
          <td>0.005949</td>
          <td>28.172140</td>
          <td>1.795913</td>
          <td>24.014189</td>
          <td>29.452182</td>
          <td>28.473639</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>26.311225</td>
          <td>0.117614</td>
          <td>22.345855</td>
          <td>0.005830</td>
          <td>23.881837</td>
          <td>0.019922</td>
          <td>19.046375</td>
          <td>0.005038</td>
          <td>23.897853</td>
          <td>0.086394</td>
          <td>25.753904</td>
          <td>30.914663</td>
          <td>25.698355</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.096724</td>
          <td>0.049022</td>
          <td>24.920740</td>
          <td>0.034530</td>
          <td>25.595544</td>
          <td>0.055104</td>
          <td>21.518639</td>
          <td>0.005533</td>
          <td>22.840373</td>
          <td>0.015435</td>
          <td>24.628593</td>
          <td>0.162989</td>
          <td>27.102272</td>
          <td>26.648769</td>
          <td>25.799738</td>
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
          <td>23.502950</td>
          <td>26.459357</td>
          <td>27.240213</td>
          <td>27.039947</td>
          <td>22.788214</td>
          <td>25.295602</td>
          <td>20.109945</td>
          <td>0.005014</td>
          <td>22.960058</td>
          <td>0.010371</td>
          <td>27.187353</td>
          <td>0.375391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.873457</td>
          <td>17.685494</td>
          <td>20.339847</td>
          <td>21.301636</td>
          <td>18.084960</td>
          <td>20.775624</td>
          <td>23.614080</td>
          <td>0.010778</td>
          <td>22.095937</td>
          <td>0.006470</td>
          <td>23.204693</td>
          <td>0.012423</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.620305</td>
          <td>22.544264</td>
          <td>25.741914</td>
          <td>21.687124</td>
          <td>20.156011</td>
          <td>17.635615</td>
          <td>20.848968</td>
          <td>0.005056</td>
          <td>21.118409</td>
          <td>0.005271</td>
          <td>21.287108</td>
          <td>0.005367</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.996961</td>
          <td>26.658135</td>
          <td>21.862419</td>
          <td>24.459888</td>
          <td>21.737323</td>
          <td>20.598460</td>
          <td>25.391100</td>
          <td>0.048494</td>
          <td>25.467283</td>
          <td>0.088412</td>
          <td>21.849186</td>
          <td>0.005975</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.899406</td>
          <td>19.956528</td>
          <td>23.020750</td>
          <td>16.385081</td>
          <td>24.039810</td>
          <td>22.197564</td>
          <td>15.660882</td>
          <td>0.005000</td>
          <td>26.949410</td>
          <td>0.311072</td>
          <td>20.485326</td>
          <td>0.005086</td>
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
          <td>21.363912</td>
          <td>25.346212</td>
          <td>22.040602</td>
          <td>21.601316</td>
          <td>24.288050</td>
          <td>22.897516</td>
          <td>22.571363</td>
          <td>0.006197</td>
          <td>23.218748</td>
          <td>0.012558</td>
          <td>27.363974</td>
          <td>0.430049</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.616598</td>
          <td>21.923674</td>
          <td>23.619002</td>
          <td>23.283611</td>
          <td>28.224596</td>
          <td>21.112721</td>
          <td>27.066534</td>
          <td>0.209374</td>
          <td>19.909726</td>
          <td>0.005030</td>
          <td>27.648886</td>
          <td>0.531683</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.571503</td>
          <td>22.002477</td>
          <td>24.160253</td>
          <td>25.972750</td>
          <td>21.136998</td>
          <td>26.635737</td>
          <td>24.005042</td>
          <td>0.014551</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.534108</td>
          <td>26.492987</td>
          <td>22.345804</td>
          <td>23.860724</td>
          <td>19.051417</td>
          <td>23.886442</td>
          <td>25.559266</td>
          <td>0.056336</td>
          <td>28.523730</td>
          <td>0.956953</td>
          <td>25.735509</td>
          <td>0.111887</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.153916</td>
          <td>24.920552</td>
          <td>25.602284</td>
          <td>21.517192</td>
          <td>22.836775</td>
          <td>24.505152</td>
          <td>27.040403</td>
          <td>0.204838</td>
          <td>27.190129</td>
          <td>0.376203</td>
          <td>26.021765</td>
          <td>0.143446</td>
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
          <td>23.502950</td>
          <td>26.459357</td>
          <td>27.240213</td>
          <td>27.039947</td>
          <td>22.788214</td>
          <td>25.295602</td>
          <td>20.110372</td>
          <td>0.006768</td>
          <td>22.972436</td>
          <td>0.052141</td>
          <td>26.193409</td>
          <td>0.775816</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.873457</td>
          <td>17.685494</td>
          <td>20.339847</td>
          <td>21.301636</td>
          <td>18.084960</td>
          <td>20.775624</td>
          <td>23.659023</td>
          <td>0.114208</td>
          <td>22.101956</td>
          <td>0.024104</td>
          <td>23.228358</td>
          <td>0.071566</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.620305</td>
          <td>22.544264</td>
          <td>25.741914</td>
          <td>21.687124</td>
          <td>20.156011</td>
          <td>17.635615</td>
          <td>20.855939</td>
          <td>0.010341</td>
          <td>21.120021</td>
          <td>0.010824</td>
          <td>21.302115</td>
          <td>0.013403</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.996961</td>
          <td>26.658135</td>
          <td>21.862419</td>
          <td>24.459888</td>
          <td>21.737323</td>
          <td>20.598460</td>
          <td>24.709103</td>
          <td>0.277785</td>
          <td>25.584301</td>
          <td>0.470885</td>
          <td>21.859833</td>
          <td>0.021307</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.899406</td>
          <td>19.956528</td>
          <td>23.020750</td>
          <td>16.385081</td>
          <td>24.039810</td>
          <td>22.197564</td>
          <td>15.662898</td>
          <td>0.005001</td>
          <td>26.023675</td>
          <td>0.646462</td>
          <td>20.486797</td>
          <td>0.007720</td>
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
          <td>21.363912</td>
          <td>25.346212</td>
          <td>22.040602</td>
          <td>21.601316</td>
          <td>24.288050</td>
          <td>22.897516</td>
          <td>22.589562</td>
          <td>0.044297</td>
          <td>23.167241</td>
          <td>0.062020</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.616598</td>
          <td>21.923674</td>
          <td>23.619002</td>
          <td>23.283611</td>
          <td>28.224596</td>
          <td>21.112721</td>
          <td>25.500196</td>
          <td>0.513090</td>
          <td>19.898663</td>
          <td>0.005895</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.571503</td>
          <td>22.002477</td>
          <td>24.160253</td>
          <td>25.972750</td>
          <td>21.136998</td>
          <td>26.635737</td>
          <td>23.874587</td>
          <td>0.137723</td>
          <td>27.179658</td>
          <td>1.315816</td>
          <td>26.511707</td>
          <td>0.949926</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.534108</td>
          <td>26.492987</td>
          <td>22.345804</td>
          <td>23.860724</td>
          <td>19.051417</td>
          <td>23.886442</td>
          <td>25.990274</td>
          <td>0.724382</td>
          <td>28.525494</td>
          <td>2.403709</td>
          <td>26.112181</td>
          <td>0.735102</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.153916</td>
          <td>24.920552</td>
          <td>25.602284</td>
          <td>21.517192</td>
          <td>22.836775</td>
          <td>24.505152</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.633869</td>
          <td>0.240636</td>
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


