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
          <td>19.902455</td>
          <td>24.276503</td>
          <td>25.766012</td>
          <td>23.666239</td>
          <td>20.171638</td>
          <td>27.984709</td>
          <td>20.036303</td>
          <td>28.017520</td>
          <td>23.014058</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.099085</td>
          <td>21.802566</td>
          <td>28.568507</td>
          <td>16.748483</td>
          <td>26.300211</td>
          <td>20.779690</td>
          <td>23.197630</td>
          <td>26.332260</td>
          <td>20.747318</td>
        </tr>
        <tr>
          <th>2</th>
          <td>14.380205</td>
          <td>24.277863</td>
          <td>18.539757</td>
          <td>22.471995</td>
          <td>26.143805</td>
          <td>18.739484</td>
          <td>26.797536</td>
          <td>26.073631</td>
          <td>19.455983</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.911279</td>
          <td>22.338681</td>
          <td>21.002176</td>
          <td>19.167062</td>
          <td>24.615670</td>
          <td>23.049216</td>
          <td>19.901875</td>
          <td>27.375195</td>
          <td>25.628677</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.354241</td>
          <td>28.229397</td>
          <td>25.459968</td>
          <td>20.439088</td>
          <td>24.615008</td>
          <td>23.342480</td>
          <td>24.846065</td>
          <td>21.868126</td>
          <td>22.072345</td>
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
          <td>25.882811</td>
          <td>23.474084</td>
          <td>22.472427</td>
          <td>24.194784</td>
          <td>25.182133</td>
          <td>21.926511</td>
          <td>17.324176</td>
          <td>21.499869</td>
          <td>28.091550</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.108291</td>
          <td>25.905684</td>
          <td>21.833241</td>
          <td>18.888148</td>
          <td>22.645047</td>
          <td>28.516122</td>
          <td>21.030316</td>
          <td>21.853672</td>
          <td>23.778268</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.378295</td>
          <td>26.931794</td>
          <td>22.151542</td>
          <td>27.069978</td>
          <td>20.494871</td>
          <td>24.011471</td>
          <td>23.342719</td>
          <td>17.829891</td>
          <td>24.947023</td>
        </tr>
        <tr>
          <th>998</th>
          <td>15.232964</td>
          <td>32.879313</td>
          <td>22.525521</td>
          <td>17.361552</td>
          <td>24.558206</td>
          <td>19.209522</td>
          <td>21.474919</td>
          <td>24.026892</td>
          <td>27.221964</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.672724</td>
          <td>25.651010</td>
          <td>14.108120</td>
          <td>20.442882</td>
          <td>24.605656</td>
          <td>23.091215</td>
          <td>20.207867</td>
          <td>28.507878</td>
          <td>20.901521</td>
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
          <td>19.904600</td>
          <td>0.005245</td>
          <td>24.275068</td>
          <td>0.019766</td>
          <td>25.823618</td>
          <td>0.067458</td>
          <td>23.647548</td>
          <td>0.016385</td>
          <td>20.171355</td>
          <td>0.005202</td>
          <td>25.920118</td>
          <td>0.462808</td>
          <td>20.036303</td>
          <td>28.017520</td>
          <td>23.014058</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.998292</td>
          <td>0.252486</td>
          <td>21.802951</td>
          <td>0.005482</td>
          <td>27.739989</td>
          <td>0.343877</td>
          <td>16.752469</td>
          <td>0.005001</td>
          <td>25.910013</td>
          <td>0.221209</td>
          <td>20.775291</td>
          <td>0.007240</td>
          <td>23.197630</td>
          <td>26.332260</td>
          <td>20.747318</td>
        </tr>
        <tr>
          <th>2</th>
          <td>14.374566</td>
          <td>0.005001</td>
          <td>24.254855</td>
          <td>0.019433</td>
          <td>18.539811</td>
          <td>0.005004</td>
          <td>22.478450</td>
          <td>0.007407</td>
          <td>26.516853</td>
          <td>0.361427</td>
          <td>18.739213</td>
          <td>0.005092</td>
          <td>26.797536</td>
          <td>26.073631</td>
          <td>19.455983</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.905705</td>
          <td>0.005245</td>
          <td>22.330123</td>
          <td>0.006085</td>
          <td>21.000605</td>
          <td>0.005098</td>
          <td>19.169587</td>
          <td>0.005016</td>
          <td>24.546820</td>
          <td>0.068046</td>
          <td>23.005978</td>
          <td>0.039206</td>
          <td>19.901875</td>
          <td>27.375195</td>
          <td>25.628677</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.359142</td>
          <td>0.025809</td>
          <td>27.865132</td>
          <td>0.423689</td>
          <td>25.532231</td>
          <td>0.052092</td>
          <td>20.453756</td>
          <td>0.005098</td>
          <td>24.477786</td>
          <td>0.064009</td>
          <td>23.335363</td>
          <td>0.052512</td>
          <td>24.846065</td>
          <td>21.868126</td>
          <td>22.072345</td>
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
          <td>26.265675</td>
          <td>0.313475</td>
          <td>23.481277</td>
          <td>0.010655</td>
          <td>22.476608</td>
          <td>0.006023</td>
          <td>24.205301</td>
          <td>0.026314</td>
          <td>25.116249</td>
          <td>0.112291</td>
          <td>21.948390</td>
          <td>0.015781</td>
          <td>17.324176</td>
          <td>21.499869</td>
          <td>28.091550</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.131594</td>
          <td>0.021297</td>
          <td>25.856767</td>
          <td>0.078992</td>
          <td>21.826199</td>
          <td>0.005359</td>
          <td>18.886694</td>
          <td>0.005011</td>
          <td>22.634330</td>
          <td>0.013117</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.030316</td>
          <td>21.853672</td>
          <td>23.778268</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.377016</td>
          <td>0.005456</td>
          <td>26.604732</td>
          <td>0.151536</td>
          <td>22.152041</td>
          <td>0.005608</td>
          <td>26.841060</td>
          <td>0.260510</td>
          <td>20.491384</td>
          <td>0.005335</td>
          <td>24.063616</td>
          <td>0.099935</td>
          <td>23.342719</td>
          <td>17.829891</td>
          <td>24.947023</td>
        </tr>
        <tr>
          <th>998</th>
          <td>15.233416</td>
          <td>0.005002</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.522307</td>
          <td>0.006100</td>
          <td>17.364466</td>
          <td>0.005002</td>
          <td>24.580047</td>
          <td>0.070077</td>
          <td>19.211367</td>
          <td>0.005189</td>
          <td>21.474919</td>
          <td>24.026892</td>
          <td>27.221964</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.669178</td>
          <td>0.005057</td>
          <td>25.549214</td>
          <td>0.060194</td>
          <td>14.103142</td>
          <td>0.005000</td>
          <td>20.438621</td>
          <td>0.005096</td>
          <td>24.583008</td>
          <td>0.070261</td>
          <td>23.103393</td>
          <td>0.042741</td>
          <td>20.207867</td>
          <td>28.507878</td>
          <td>20.901521</td>
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
          <td>19.902455</td>
          <td>24.276503</td>
          <td>25.766012</td>
          <td>23.666239</td>
          <td>20.171638</td>
          <td>27.984709</td>
          <td>20.035759</td>
          <td>0.005013</td>
          <td>27.350220</td>
          <td>0.425571</td>
          <td>23.006818</td>
          <td>0.010722</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.099085</td>
          <td>21.802566</td>
          <td>28.568507</td>
          <td>16.748483</td>
          <td>26.300211</td>
          <td>20.779690</td>
          <td>23.190793</td>
          <td>0.008179</td>
          <td>26.970017</td>
          <td>0.316242</td>
          <td>20.743602</td>
          <td>0.005138</td>
        </tr>
        <tr>
          <th>2</th>
          <td>14.380205</td>
          <td>24.277863</td>
          <td>18.539757</td>
          <td>22.471995</td>
          <td>26.143805</td>
          <td>18.739484</td>
          <td>27.224194</td>
          <td>0.238721</td>
          <td>25.990762</td>
          <td>0.139660</td>
          <td>19.465589</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.911279</td>
          <td>22.338681</td>
          <td>21.002176</td>
          <td>19.167062</td>
          <td>24.615670</td>
          <td>23.049216</td>
          <td>19.903342</td>
          <td>0.005010</td>
          <td>28.037354</td>
          <td>0.698940</td>
          <td>25.511867</td>
          <td>0.091955</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.354241</td>
          <td>28.229397</td>
          <td>25.459968</td>
          <td>20.439088</td>
          <td>24.615008</td>
          <td>23.342480</td>
          <td>24.867100</td>
          <td>0.030430</td>
          <td>21.880574</td>
          <td>0.006028</td>
          <td>22.066194</td>
          <td>0.006400</td>
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
          <td>25.882811</td>
          <td>23.474084</td>
          <td>22.472427</td>
          <td>24.194784</td>
          <td>25.182133</td>
          <td>21.926511</td>
          <td>17.325495</td>
          <td>0.005000</td>
          <td>21.494659</td>
          <td>0.005529</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.108291</td>
          <td>25.905684</td>
          <td>21.833241</td>
          <td>18.888148</td>
          <td>22.645047</td>
          <td>28.516122</td>
          <td>21.032027</td>
          <td>0.005078</td>
          <td>21.850964</td>
          <td>0.005978</td>
          <td>23.825111</td>
          <td>0.020680</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.378295</td>
          <td>26.931794</td>
          <td>22.151542</td>
          <td>27.069978</td>
          <td>20.494871</td>
          <td>24.011471</td>
          <td>23.345371</td>
          <td>0.008981</td>
          <td>17.832650</td>
          <td>0.005001</td>
          <td>25.016827</td>
          <td>0.059299</td>
        </tr>
        <tr>
          <th>998</th>
          <td>15.232964</td>
          <td>32.879313</td>
          <td>22.525521</td>
          <td>17.361552</td>
          <td>24.558206</td>
          <td>19.209522</td>
          <td>21.480295</td>
          <td>0.005177</td>
          <td>24.012212</td>
          <td>0.024321</td>
          <td>26.925472</td>
          <td>0.305159</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.672724</td>
          <td>25.651010</td>
          <td>14.108120</td>
          <td>20.442882</td>
          <td>24.605656</td>
          <td>23.091215</td>
          <td>20.212246</td>
          <td>0.005017</td>
          <td>28.692599</td>
          <td>1.059029</td>
          <td>20.908338</td>
          <td>0.005186</td>
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
          <td>19.902455</td>
          <td>24.276503</td>
          <td>25.766012</td>
          <td>23.666239</td>
          <td>20.171638</td>
          <td>27.984709</td>
          <td>20.034877</td>
          <td>0.006566</td>
          <td>26.786500</td>
          <td>1.055234</td>
          <td>22.973979</td>
          <td>0.057079</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.099085</td>
          <td>21.802566</td>
          <td>28.568507</td>
          <td>16.748483</td>
          <td>26.300211</td>
          <td>20.779690</td>
          <td>23.284230</td>
          <td>0.082160</td>
          <td>25.443786</td>
          <td>0.423489</td>
          <td>20.749362</td>
          <td>0.009004</td>
        </tr>
        <tr>
          <th>2</th>
          <td>14.380205</td>
          <td>24.277863</td>
          <td>18.539757</td>
          <td>22.471995</td>
          <td>26.143805</td>
          <td>18.739484</td>
          <td>29.228703</td>
          <td>3.237755</td>
          <td>25.745478</td>
          <td>0.530365</td>
          <td>19.462270</td>
          <td>0.005500</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.911279</td>
          <td>22.338681</td>
          <td>21.002176</td>
          <td>19.167062</td>
          <td>24.615670</td>
          <td>23.049216</td>
          <td>19.895510</td>
          <td>0.006246</td>
          <td>26.809372</td>
          <td>1.069509</td>
          <td>25.248824</td>
          <td>0.393720</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.354241</td>
          <td>28.229397</td>
          <td>25.459968</td>
          <td>20.439088</td>
          <td>24.615008</td>
          <td>23.342480</td>
          <td>25.093079</td>
          <td>0.377067</td>
          <td>21.901636</td>
          <td>0.020267</td>
          <td>22.054451</td>
          <td>0.025235</td>
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
          <td>25.882811</td>
          <td>23.474084</td>
          <td>22.472427</td>
          <td>24.194784</td>
          <td>25.182133</td>
          <td>21.926511</td>
          <td>17.323904</td>
          <td>0.005012</td>
          <td>21.515457</td>
          <td>0.014674</td>
          <td>26.535348</td>
          <td>0.963774</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.108291</td>
          <td>25.905684</td>
          <td>21.833241</td>
          <td>18.888148</td>
          <td>22.645047</td>
          <td>28.516122</td>
          <td>21.028612</td>
          <td>0.011726</td>
          <td>21.831676</td>
          <td>0.019091</td>
          <td>23.567933</td>
          <td>0.096605</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.378295</td>
          <td>26.931794</td>
          <td>22.151542</td>
          <td>27.069978</td>
          <td>20.494871</td>
          <td>24.011471</td>
          <td>23.352735</td>
          <td>0.087284</td>
          <td>17.827794</td>
          <td>0.005021</td>
          <td>24.989934</td>
          <td>0.321308</td>
        </tr>
        <tr>
          <th>998</th>
          <td>15.232964</td>
          <td>32.879313</td>
          <td>22.525521</td>
          <td>17.361552</td>
          <td>24.558206</td>
          <td>19.209522</td>
          <td>21.480668</td>
          <td>0.016809</td>
          <td>23.936481</td>
          <td>0.122182</td>
          <td>26.684213</td>
          <td>1.053813</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.672724</td>
          <td>25.651010</td>
          <td>14.108120</td>
          <td>20.442882</td>
          <td>24.605656</td>
          <td>23.091215</td>
          <td>20.200212</td>
          <td>0.007039</td>
          <td>26.852309</td>
          <td>1.096610</td>
          <td>20.886854</td>
          <td>0.009858</td>
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


