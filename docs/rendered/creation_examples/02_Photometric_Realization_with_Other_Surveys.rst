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
          <td>19.013634</td>
          <td>23.093681</td>
          <td>17.468793</td>
          <td>21.169179</td>
          <td>23.368409</td>
          <td>30.314467</td>
          <td>27.486158</td>
          <td>21.851140</td>
          <td>24.917578</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.368805</td>
          <td>21.801343</td>
          <td>23.170189</td>
          <td>25.530931</td>
          <td>22.879669</td>
          <td>23.416856</td>
          <td>29.749672</td>
          <td>17.976523</td>
          <td>24.808678</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.448667</td>
          <td>21.589797</td>
          <td>25.977712</td>
          <td>16.418719</td>
          <td>15.505970</td>
          <td>29.906863</td>
          <td>20.908317</td>
          <td>22.663408</td>
          <td>24.818505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.457491</td>
          <td>21.477862</td>
          <td>28.126940</td>
          <td>20.370853</td>
          <td>15.688850</td>
          <td>29.425169</td>
          <td>20.413508</td>
          <td>27.245355</td>
          <td>17.839053</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.075649</td>
          <td>23.818317</td>
          <td>29.199671</td>
          <td>21.969948</td>
          <td>18.033806</td>
          <td>25.481676</td>
          <td>24.870236</td>
          <td>23.854506</td>
          <td>17.843800</td>
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
          <td>22.540699</td>
          <td>24.664834</td>
          <td>19.264528</td>
          <td>21.477584</td>
          <td>19.313298</td>
          <td>20.536912</td>
          <td>20.113709</td>
          <td>23.327368</td>
          <td>24.920538</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.983283</td>
          <td>26.367913</td>
          <td>20.270488</td>
          <td>26.232456</td>
          <td>23.344400</td>
          <td>19.852955</td>
          <td>22.980826</td>
          <td>23.450292</td>
          <td>21.821247</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.035933</td>
          <td>25.250831</td>
          <td>24.453501</td>
          <td>12.704286</td>
          <td>26.644401</td>
          <td>22.962988</td>
          <td>28.797432</td>
          <td>24.103374</td>
          <td>24.232537</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.902210</td>
          <td>23.715798</td>
          <td>20.823723</td>
          <td>22.929376</td>
          <td>27.745788</td>
          <td>26.164335</td>
          <td>21.348048</td>
          <td>23.449163</td>
          <td>23.411387</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.766744</td>
          <td>28.084197</td>
          <td>21.214587</td>
          <td>21.017919</td>
          <td>27.814115</td>
          <td>25.912937</td>
          <td>23.165588</td>
          <td>22.979744</td>
          <td>18.992221</td>
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
          <td>19.021884</td>
          <td>0.005084</td>
          <td>23.104774</td>
          <td>0.008401</td>
          <td>17.463164</td>
          <td>0.005001</td>
          <td>21.160567</td>
          <td>0.005299</td>
          <td>23.395511</td>
          <td>0.024624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.486158</td>
          <td>21.851140</td>
          <td>24.917578</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.371827</td>
          <td>0.006814</td>
          <td>21.814571</td>
          <td>0.005491</td>
          <td>23.165130</td>
          <td>0.007929</td>
          <td>25.448531</td>
          <td>0.078992</td>
          <td>22.885567</td>
          <td>0.016011</td>
          <td>23.412180</td>
          <td>0.056217</td>
          <td>29.749672</td>
          <td>17.976523</td>
          <td>24.808678</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.612556</td>
          <td>0.183147</td>
          <td>21.594565</td>
          <td>0.005350</td>
          <td>25.910616</td>
          <td>0.072858</td>
          <td>16.414903</td>
          <td>0.005001</td>
          <td>15.501615</td>
          <td>0.005001</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.908317</td>
          <td>22.663408</td>
          <td>24.818505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.453827</td>
          <td>0.007032</td>
          <td>21.475898</td>
          <td>0.005293</td>
          <td>28.068438</td>
          <td>0.443241</td>
          <td>20.369914</td>
          <td>0.005087</td>
          <td>15.688572</td>
          <td>0.005001</td>
          <td>27.315873</td>
          <td>1.164097</td>
          <td>20.413508</td>
          <td>27.245355</td>
          <td>17.839053</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.090381</td>
          <td>0.048749</td>
          <td>23.822628</td>
          <td>0.013695</td>
          <td>31.620633</td>
          <td>2.887007</td>
          <td>21.970017</td>
          <td>0.006100</td>
          <td>18.030083</td>
          <td>0.005010</td>
          <td>25.408546</td>
          <td>0.311184</td>
          <td>24.870236</td>
          <td>23.854506</td>
          <td>17.843800</td>
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
          <td>22.570619</td>
          <td>0.013611</td>
          <td>24.658935</td>
          <td>0.027461</td>
          <td>19.271812</td>
          <td>0.005009</td>
          <td>21.474228</td>
          <td>0.005496</td>
          <td>19.306375</td>
          <td>0.005055</td>
          <td>20.532808</td>
          <td>0.006550</td>
          <td>20.113709</td>
          <td>23.327368</td>
          <td>24.920538</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.989386</td>
          <td>0.005273</td>
          <td>26.606909</td>
          <td>0.151819</td>
          <td>20.262259</td>
          <td>0.005034</td>
          <td>26.534414</td>
          <td>0.202032</td>
          <td>23.368821</td>
          <td>0.024062</td>
          <td>19.851629</td>
          <td>0.005526</td>
          <td>22.980826</td>
          <td>23.450292</td>
          <td>21.821247</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.996636</td>
          <td>0.019047</td>
          <td>25.267654</td>
          <td>0.046909</td>
          <td>24.481178</td>
          <td>0.020715</td>
          <td>12.705704</td>
          <td>0.005000</td>
          <td>26.763464</td>
          <td>0.437075</td>
          <td>22.947089</td>
          <td>0.037215</td>
          <td>28.797432</td>
          <td>24.103374</td>
          <td>24.232537</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.725900</td>
          <td>0.012719</td>
          <td>20.830455</td>
          <td>0.005076</td>
          <td>22.924977</td>
          <td>0.009552</td>
          <td>27.851391</td>
          <td>0.927507</td>
          <td>26.847518</td>
          <td>0.879713</td>
          <td>21.348048</td>
          <td>23.449163</td>
          <td>23.411387</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.664195</td>
          <td>0.033572</td>
          <td>28.167487</td>
          <td>0.530796</td>
          <td>21.211565</td>
          <td>0.005136</td>
          <td>21.026373</td>
          <td>0.005241</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.099311</td>
          <td>0.528373</td>
          <td>23.165588</td>
          <td>22.979744</td>
          <td>18.992221</td>
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
          <td>19.013634</td>
          <td>23.093681</td>
          <td>17.468793</td>
          <td>21.169179</td>
          <td>23.368409</td>
          <td>30.314467</td>
          <td>27.235289</td>
          <td>0.240919</td>
          <td>21.844287</td>
          <td>0.005967</td>
          <td>24.842230</td>
          <td>0.050756</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.368805</td>
          <td>21.801343</td>
          <td>23.170189</td>
          <td>25.530931</td>
          <td>22.879669</td>
          <td>23.416856</td>
          <td>28.485094</td>
          <td>0.629322</td>
          <td>17.976004</td>
          <td>0.005001</td>
          <td>24.879792</td>
          <td>0.052484</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.448667</td>
          <td>21.589797</td>
          <td>25.977712</td>
          <td>16.418719</td>
          <td>15.505970</td>
          <td>29.906863</td>
          <td>20.903380</td>
          <td>0.005062</td>
          <td>22.642675</td>
          <td>0.008431</td>
          <td>24.937688</td>
          <td>0.055263</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.457491</td>
          <td>21.477862</td>
          <td>28.126940</td>
          <td>20.370853</td>
          <td>15.688850</td>
          <td>29.425169</td>
          <td>20.403291</td>
          <td>0.005025</td>
          <td>27.052066</td>
          <td>0.337563</td>
          <td>17.842092</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.075649</td>
          <td>23.818317</td>
          <td>29.199671</td>
          <td>21.969948</td>
          <td>18.033806</td>
          <td>25.481676</td>
          <td>24.845262</td>
          <td>0.029848</td>
          <td>23.845521</td>
          <td>0.021046</td>
          <td>17.847957</td>
          <td>0.005001</td>
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
          <td>22.540699</td>
          <td>24.664834</td>
          <td>19.264528</td>
          <td>21.477584</td>
          <td>19.313298</td>
          <td>20.536912</td>
          <td>20.122527</td>
          <td>0.005015</td>
          <td>23.317822</td>
          <td>0.013570</td>
          <td>24.907424</td>
          <td>0.053792</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.983283</td>
          <td>26.367913</td>
          <td>20.270488</td>
          <td>26.232456</td>
          <td>23.344400</td>
          <td>19.852955</td>
          <td>22.979596</td>
          <td>0.007308</td>
          <td>23.433865</td>
          <td>0.014895</td>
          <td>21.816960</td>
          <td>0.005923</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.035933</td>
          <td>25.250831</td>
          <td>24.453501</td>
          <td>12.704286</td>
          <td>26.644401</td>
          <td>22.962988</td>
          <td>28.442300</td>
          <td>0.610705</td>
          <td>24.109564</td>
          <td>0.026485</td>
          <td>24.190968</td>
          <td>0.028451</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.902210</td>
          <td>23.715798</td>
          <td>20.823723</td>
          <td>22.929376</td>
          <td>27.745788</td>
          <td>26.164335</td>
          <td>21.342319</td>
          <td>0.005138</td>
          <td>23.441305</td>
          <td>0.014985</td>
          <td>23.390649</td>
          <td>0.014383</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.766744</td>
          <td>28.084197</td>
          <td>21.214587</td>
          <td>21.017919</td>
          <td>27.814115</td>
          <td>25.912937</td>
          <td>23.150538</td>
          <td>0.007994</td>
          <td>22.992725</td>
          <td>0.010614</td>
          <td>18.992238</td>
          <td>0.005006</td>
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
          <td>19.013634</td>
          <td>23.093681</td>
          <td>17.468793</td>
          <td>21.169179</td>
          <td>23.368409</td>
          <td>30.314467</td>
          <td>26.750726</td>
          <td>1.160205</td>
          <td>21.848077</td>
          <td>0.019359</td>
          <td>25.249573</td>
          <td>0.393948</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.368805</td>
          <td>21.801343</td>
          <td>23.170189</td>
          <td>25.530931</td>
          <td>22.879669</td>
          <td>23.416856</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.979396</td>
          <td>0.005028</td>
          <td>24.724698</td>
          <td>0.259301</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.448667</td>
          <td>21.589797</td>
          <td>25.977712</td>
          <td>16.418719</td>
          <td>15.505970</td>
          <td>29.906863</td>
          <td>20.907622</td>
          <td>0.010728</td>
          <td>22.709022</td>
          <td>0.041229</td>
          <td>25.064014</td>
          <td>0.340768</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.457491</td>
          <td>21.477862</td>
          <td>28.126940</td>
          <td>20.370853</td>
          <td>15.688850</td>
          <td>29.425169</td>
          <td>20.413332</td>
          <td>0.007831</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.841261</td>
          <td>0.005026</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.075649</td>
          <td>23.818317</td>
          <td>29.199671</td>
          <td>21.969948</td>
          <td>18.033806</td>
          <td>25.481676</td>
          <td>26.650774</td>
          <td>1.095634</td>
          <td>23.776411</td>
          <td>0.106251</td>
          <td>17.847960</td>
          <td>0.005027</td>
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
          <td>22.540699</td>
          <td>24.664834</td>
          <td>19.264528</td>
          <td>21.477584</td>
          <td>19.313298</td>
          <td>20.536912</td>
          <td>20.106835</td>
          <td>0.006758</td>
          <td>23.374044</td>
          <td>0.074525</td>
          <td>24.375604</td>
          <td>0.193974</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.983283</td>
          <td>26.367913</td>
          <td>20.270488</td>
          <td>26.232456</td>
          <td>23.344400</td>
          <td>19.852955</td>
          <td>22.997585</td>
          <td>0.063717</td>
          <td>23.403116</td>
          <td>0.076469</td>
          <td>21.852332</td>
          <td>0.021170</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.035933</td>
          <td>25.250831</td>
          <td>24.453501</td>
          <td>12.704286</td>
          <td>26.644401</td>
          <td>22.962988</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.000981</td>
          <td>0.129222</td>
          <td>24.211008</td>
          <td>0.168711</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.902210</td>
          <td>23.715798</td>
          <td>20.823723</td>
          <td>22.929376</td>
          <td>27.745788</td>
          <td>26.164335</td>
          <td>21.344093</td>
          <td>0.015019</td>
          <td>23.460214</td>
          <td>0.080434</td>
          <td>23.269596</td>
          <td>0.074231</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.766744</td>
          <td>28.084197</td>
          <td>21.214587</td>
          <td>21.017919</td>
          <td>27.814115</td>
          <td>25.912937</td>
          <td>23.277209</td>
          <td>0.081652</td>
          <td>22.967644</td>
          <td>0.051919</td>
          <td>18.996868</td>
          <td>0.005218</td>
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


