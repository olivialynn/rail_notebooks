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
          <td>23.006546</td>
          <td>24.435145</td>
          <td>25.980596</td>
          <td>21.561344</td>
          <td>20.428570</td>
          <td>25.339232</td>
          <td>23.906196</td>
          <td>18.083278</td>
          <td>22.290505</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.225410</td>
          <td>22.275192</td>
          <td>19.854958</td>
          <td>18.202166</td>
          <td>17.968408</td>
          <td>24.837990</td>
          <td>22.870655</td>
          <td>26.001790</td>
          <td>22.669862</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.166320</td>
          <td>17.713075</td>
          <td>24.544026</td>
          <td>20.745018</td>
          <td>25.478347</td>
          <td>23.035827</td>
          <td>21.798623</td>
          <td>21.219059</td>
          <td>21.512496</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.425765</td>
          <td>27.827786</td>
          <td>22.600566</td>
          <td>19.288495</td>
          <td>17.022336</td>
          <td>29.012562</td>
          <td>25.794248</td>
          <td>22.395351</td>
          <td>18.618039</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.603379</td>
          <td>19.455728</td>
          <td>24.188216</td>
          <td>27.508920</td>
          <td>23.647413</td>
          <td>19.146316</td>
          <td>18.620359</td>
          <td>27.029529</td>
          <td>25.624028</td>
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
          <td>24.498081</td>
          <td>23.015011</td>
          <td>25.537342</td>
          <td>23.618058</td>
          <td>23.063937</td>
          <td>22.408231</td>
          <td>26.306980</td>
          <td>23.485044</td>
          <td>23.077876</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.501877</td>
          <td>23.260542</td>
          <td>17.659123</td>
          <td>21.936748</td>
          <td>24.707084</td>
          <td>26.275399</td>
          <td>26.175914</td>
          <td>19.564346</td>
          <td>22.444160</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.851397</td>
          <td>18.161578</td>
          <td>20.680827</td>
          <td>20.815199</td>
          <td>28.662354</td>
          <td>21.202449</td>
          <td>20.715046</td>
          <td>24.664038</td>
          <td>21.839262</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.700784</td>
          <td>21.425232</td>
          <td>23.373511</td>
          <td>24.033280</td>
          <td>24.376909</td>
          <td>24.289181</td>
          <td>19.343508</td>
          <td>26.276176</td>
          <td>25.151760</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.635269</td>
          <td>19.978991</td>
          <td>18.923007</td>
          <td>28.012940</td>
          <td>20.558743</td>
          <td>20.846735</td>
          <td>17.532820</td>
          <td>23.445233</td>
          <td>21.720991</td>
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
          <td>22.992980</td>
          <td>0.018990</td>
          <td>24.435171</td>
          <td>0.022640</td>
          <td>26.015637</td>
          <td>0.079942</td>
          <td>21.569579</td>
          <td>0.005579</td>
          <td>20.429687</td>
          <td>0.005304</td>
          <td>25.465452</td>
          <td>0.325635</td>
          <td>23.906196</td>
          <td>18.083278</td>
          <td>22.290505</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.241682</td>
          <td>0.023359</td>
          <td>22.281247</td>
          <td>0.006007</td>
          <td>19.858036</td>
          <td>0.005020</td>
          <td>18.197734</td>
          <td>0.005005</td>
          <td>17.968372</td>
          <td>0.005010</td>
          <td>24.752198</td>
          <td>0.181051</td>
          <td>22.870655</td>
          <td>26.001790</td>
          <td>22.669862</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.172851</td>
          <td>0.006375</td>
          <td>17.715976</td>
          <td>0.005003</td>
          <td>24.527509</td>
          <td>0.021550</td>
          <td>20.746496</td>
          <td>0.005155</td>
          <td>25.449492</td>
          <td>0.149829</td>
          <td>23.028933</td>
          <td>0.040012</td>
          <td>21.798623</td>
          <td>21.219059</td>
          <td>21.512496</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.400355</td>
          <td>0.026734</td>
          <td>27.630467</td>
          <td>0.353322</td>
          <td>22.601908</td>
          <td>0.006247</td>
          <td>19.286288</td>
          <td>0.005019</td>
          <td>17.018473</td>
          <td>0.005003</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.794248</td>
          <td>22.395351</td>
          <td>18.618039</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.617840</td>
          <td>0.014106</td>
          <td>19.458089</td>
          <td>0.005020</td>
          <td>24.187275</td>
          <td>0.016204</td>
          <td>27.011345</td>
          <td>0.299108</td>
          <td>23.658586</td>
          <td>0.030983</td>
          <td>19.140202</td>
          <td>0.005170</td>
          <td>18.620359</td>
          <td>27.029529</td>
          <td>25.624028</td>
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
          <td>24.476191</td>
          <td>0.068452</td>
          <td>23.020408</td>
          <td>0.008020</td>
          <td>25.586492</td>
          <td>0.054663</td>
          <td>23.617629</td>
          <td>0.015989</td>
          <td>23.057499</td>
          <td>0.018456</td>
          <td>22.373456</td>
          <td>0.022536</td>
          <td>26.306980</td>
          <td>23.485044</td>
          <td>23.077876</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.451611</td>
          <td>0.066989</td>
          <td>23.273307</td>
          <td>0.009289</td>
          <td>17.658669</td>
          <td>0.005002</td>
          <td>21.932599</td>
          <td>0.006036</td>
          <td>24.677084</td>
          <td>0.076356</td>
          <td>27.077453</td>
          <td>1.013318</td>
          <td>26.175914</td>
          <td>19.564346</td>
          <td>22.444160</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.266646</td>
          <td>0.662331</td>
          <td>18.159995</td>
          <td>0.005005</td>
          <td>20.678015</td>
          <td>0.005061</td>
          <td>20.811822</td>
          <td>0.005172</td>
          <td>27.655319</td>
          <td>0.819261</td>
          <td>21.199660</td>
          <td>0.009128</td>
          <td>20.715046</td>
          <td>24.664038</td>
          <td>21.839262</td>
        </tr>
        <tr>
          <th>998</th>
          <td>31.242780</td>
          <td>3.819327</td>
          <td>21.434675</td>
          <td>0.005275</td>
          <td>23.389610</td>
          <td>0.009030</td>
          <td>24.086825</td>
          <td>0.023743</td>
          <td>24.414919</td>
          <td>0.060538</td>
          <td>24.308046</td>
          <td>0.123680</td>
          <td>19.343508</td>
          <td>26.276176</td>
          <td>25.151760</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.639456</td>
          <td>0.005653</td>
          <td>19.982195</td>
          <td>0.005037</td>
          <td>18.924600</td>
          <td>0.005006</td>
          <td>27.731390</td>
          <td>0.520702</td>
          <td>20.562655</td>
          <td>0.005376</td>
          <td>20.845147</td>
          <td>0.007485</td>
          <td>17.532820</td>
          <td>23.445233</td>
          <td>21.720991</td>
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
          <td>23.006546</td>
          <td>24.435145</td>
          <td>25.980596</td>
          <td>21.561344</td>
          <td>20.428570</td>
          <td>25.339232</td>
          <td>23.885640</td>
          <td>0.013230</td>
          <td>18.081748</td>
          <td>0.005001</td>
          <td>22.296005</td>
          <td>0.007026</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.225410</td>
          <td>22.275192</td>
          <td>19.854958</td>
          <td>18.202166</td>
          <td>17.968408</td>
          <td>24.837990</td>
          <td>22.868604</td>
          <td>0.006940</td>
          <td>26.071428</td>
          <td>0.149710</td>
          <td>22.674619</td>
          <td>0.008595</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.166320</td>
          <td>17.713075</td>
          <td>24.544026</td>
          <td>20.745018</td>
          <td>25.478347</td>
          <td>23.035827</td>
          <td>21.794759</td>
          <td>0.005311</td>
          <td>21.219178</td>
          <td>0.005325</td>
          <td>21.510693</td>
          <td>0.005544</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.425765</td>
          <td>27.827786</td>
          <td>22.600566</td>
          <td>19.288495</td>
          <td>17.022336</td>
          <td>29.012562</td>
          <td>25.874023</td>
          <td>0.074523</td>
          <td>22.398874</td>
          <td>0.007378</td>
          <td>18.617556</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.603379</td>
          <td>19.455728</td>
          <td>24.188216</td>
          <td>27.508920</td>
          <td>23.647413</td>
          <td>19.146316</td>
          <td>18.618296</td>
          <td>0.005001</td>
          <td>27.527039</td>
          <td>0.486116</td>
          <td>25.731299</td>
          <td>0.111477</td>
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
          <td>24.498081</td>
          <td>23.015011</td>
          <td>25.537342</td>
          <td>23.618058</td>
          <td>23.063937</td>
          <td>22.408231</td>
          <td>26.487835</td>
          <td>0.127757</td>
          <td>23.484699</td>
          <td>0.015526</td>
          <td>23.098658</td>
          <td>0.011466</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.501877</td>
          <td>23.260542</td>
          <td>17.659123</td>
          <td>21.936748</td>
          <td>24.707084</td>
          <td>26.275399</td>
          <td>26.110380</td>
          <td>0.091834</td>
          <td>19.570119</td>
          <td>0.005016</td>
          <td>22.455051</td>
          <td>0.007592</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.851397</td>
          <td>18.161578</td>
          <td>20.680827</td>
          <td>20.815199</td>
          <td>28.662354</td>
          <td>21.202449</td>
          <td>20.712372</td>
          <td>0.005044</td>
          <td>24.605784</td>
          <td>0.041110</td>
          <td>21.828382</td>
          <td>0.005942</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.700784</td>
          <td>21.425232</td>
          <td>23.373511</td>
          <td>24.033280</td>
          <td>24.376909</td>
          <td>24.289181</td>
          <td>19.347839</td>
          <td>0.005004</td>
          <td>26.238295</td>
          <td>0.172677</td>
          <td>25.238313</td>
          <td>0.072200</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.635269</td>
          <td>19.978991</td>
          <td>18.923007</td>
          <td>28.012940</td>
          <td>20.558743</td>
          <td>20.846735</td>
          <td>17.530322</td>
          <td>0.005000</td>
          <td>23.452934</td>
          <td>0.015128</td>
          <td>21.727328</td>
          <td>0.005793</td>
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
          <td>23.006546</td>
          <td>24.435145</td>
          <td>25.980596</td>
          <td>21.561344</td>
          <td>20.428570</td>
          <td>25.339232</td>
          <td>24.180690</td>
          <td>0.179011</td>
          <td>18.085668</td>
          <td>0.005034</td>
          <td>22.293982</td>
          <td>0.031162</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.225410</td>
          <td>22.275192</td>
          <td>19.854958</td>
          <td>18.202166</td>
          <td>17.968408</td>
          <td>24.837990</td>
          <td>22.890291</td>
          <td>0.057914</td>
          <td>27.862050</td>
          <td>1.835871</td>
          <td>22.633315</td>
          <td>0.042131</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.166320</td>
          <td>17.713075</td>
          <td>24.544026</td>
          <td>20.745018</td>
          <td>25.478347</td>
          <td>23.035827</td>
          <td>21.809957</td>
          <td>0.022251</td>
          <td>21.228335</td>
          <td>0.011723</td>
          <td>21.511765</td>
          <td>0.015875</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.425765</td>
          <td>27.827786</td>
          <td>22.600566</td>
          <td>19.288495</td>
          <td>17.022336</td>
          <td>29.012562</td>
          <td>25.454870</td>
          <td>0.496243</td>
          <td>22.418172</td>
          <td>0.031837</td>
          <td>18.615516</td>
          <td>0.005109</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.603379</td>
          <td>19.455728</td>
          <td>24.188216</td>
          <td>27.508920</td>
          <td>23.647413</td>
          <td>19.146316</td>
          <td>18.618430</td>
          <td>0.005132</td>
          <td>26.475929</td>
          <td>0.872831</td>
          <td>26.052480</td>
          <td>0.706147</td>
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
          <td>24.498081</td>
          <td>23.015011</td>
          <td>25.537342</td>
          <td>23.618058</td>
          <td>23.063937</td>
          <td>22.408231</td>
          <td>26.205704</td>
          <td>0.834601</td>
          <td>23.549039</td>
          <td>0.087000</td>
          <td>23.041040</td>
          <td>0.060591</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.501877</td>
          <td>23.260542</td>
          <td>17.659123</td>
          <td>21.936748</td>
          <td>24.707084</td>
          <td>26.275399</td>
          <td>28.573100</td>
          <td>2.626848</td>
          <td>19.559100</td>
          <td>0.005497</td>
          <td>22.535116</td>
          <td>0.038603</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.851397</td>
          <td>18.161578</td>
          <td>20.680827</td>
          <td>20.815199</td>
          <td>28.662354</td>
          <td>21.202449</td>
          <td>20.713912</td>
          <td>0.009387</td>
          <td>24.324052</td>
          <td>0.170596</td>
          <td>21.845324</td>
          <td>0.021043</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.700784</td>
          <td>21.425232</td>
          <td>23.373511</td>
          <td>24.033280</td>
          <td>24.376909</td>
          <td>24.289181</td>
          <td>19.354802</td>
          <td>0.005493</td>
          <td>25.833169</td>
          <td>0.565096</td>
          <td>24.812285</td>
          <td>0.278504</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.635269</td>
          <td>19.978991</td>
          <td>18.923007</td>
          <td>28.012940</td>
          <td>20.558743</td>
          <td>20.846735</td>
          <td>17.530824</td>
          <td>0.005018</td>
          <td>23.410977</td>
          <td>0.077004</td>
          <td>21.730267</td>
          <td>0.019068</td>
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


