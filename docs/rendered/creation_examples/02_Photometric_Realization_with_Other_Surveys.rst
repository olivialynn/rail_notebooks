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
          <td>29.038595</td>
          <td>20.308441</td>
          <td>20.597148</td>
          <td>28.179099</td>
          <td>31.812685</td>
          <td>24.162910</td>
          <td>30.251929</td>
          <td>23.164178</td>
          <td>22.350703</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.201502</td>
          <td>23.937282</td>
          <td>30.080040</td>
          <td>21.446809</td>
          <td>18.378288</td>
          <td>19.801030</td>
          <td>25.040524</td>
          <td>20.575688</td>
          <td>22.727182</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.877232</td>
          <td>23.021218</td>
          <td>17.491714</td>
          <td>22.993833</td>
          <td>22.089700</td>
          <td>25.805671</td>
          <td>23.418613</td>
          <td>20.230926</td>
          <td>20.450836</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.302866</td>
          <td>25.625164</td>
          <td>28.528699</td>
          <td>16.517343</td>
          <td>16.313146</td>
          <td>21.763790</td>
          <td>28.125880</td>
          <td>23.414822</td>
          <td>23.609189</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.743590</td>
          <td>24.310503</td>
          <td>22.694278</td>
          <td>25.813444</td>
          <td>22.180547</td>
          <td>24.083937</td>
          <td>20.034121</td>
          <td>22.547063</td>
          <td>24.031482</td>
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
          <td>18.524023</td>
          <td>27.866537</td>
          <td>20.083999</td>
          <td>26.187547</td>
          <td>32.213351</td>
          <td>26.309141</td>
          <td>20.943618</td>
          <td>21.192117</td>
          <td>21.725943</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.166278</td>
          <td>21.160477</td>
          <td>27.242981</td>
          <td>25.724092</td>
          <td>24.125209</td>
          <td>20.711039</td>
          <td>21.147776</td>
          <td>22.467908</td>
          <td>20.140631</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.366486</td>
          <td>17.074490</td>
          <td>22.919641</td>
          <td>18.958684</td>
          <td>22.675284</td>
          <td>22.458847</td>
          <td>21.172269</td>
          <td>18.941262</td>
          <td>18.695609</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.277727</td>
          <td>22.986550</td>
          <td>22.373395</td>
          <td>24.733625</td>
          <td>24.092018</td>
          <td>22.068006</td>
          <td>18.189812</td>
          <td>28.214324</td>
          <td>25.711553</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.307886</td>
          <td>22.067876</td>
          <td>22.176979</td>
          <td>24.889471</td>
          <td>23.057752</td>
          <td>20.935526</td>
          <td>21.627362</td>
          <td>26.149240</td>
          <td>22.464571</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>20.297638</td>
          <td>0.005055</td>
          <td>20.595060</td>
          <td>0.005054</td>
          <td>28.615786</td>
          <td>0.945734</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.145506</td>
          <td>0.107357</td>
          <td>30.251929</td>
          <td>23.164178</td>
          <td>22.350703</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.194915</td>
          <td>0.005357</td>
          <td>23.919476</td>
          <td>0.014775</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.444595</td>
          <td>0.005473</td>
          <td>18.375295</td>
          <td>0.005016</td>
          <td>19.808764</td>
          <td>0.005491</td>
          <td>25.040524</td>
          <td>20.575688</td>
          <td>22.727182</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.878436</td>
          <td>0.005009</td>
          <td>23.011518</td>
          <td>0.007982</td>
          <td>17.500806</td>
          <td>0.005001</td>
          <td>22.996553</td>
          <td>0.010016</td>
          <td>22.097601</td>
          <td>0.009006</td>
          <td>25.360206</td>
          <td>0.299347</td>
          <td>23.418613</td>
          <td>20.230926</td>
          <td>20.450836</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.144831</td>
          <td>0.608436</td>
          <td>25.709239</td>
          <td>0.069347</td>
          <td>28.309976</td>
          <td>0.530286</td>
          <td>16.518414</td>
          <td>0.005001</td>
          <td>16.309983</td>
          <td>0.005002</td>
          <td>21.745533</td>
          <td>0.013431</td>
          <td>28.125880</td>
          <td>23.414822</td>
          <td>23.609189</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.739110</td>
          <td>0.005750</td>
          <td>24.333550</td>
          <td>0.020765</td>
          <td>22.696652</td>
          <td>0.006446</td>
          <td>25.716952</td>
          <td>0.100031</td>
          <td>22.174199</td>
          <td>0.009452</td>
          <td>24.142772</td>
          <td>0.107101</td>
          <td>20.034121</td>
          <td>22.547063</td>
          <td>24.031482</td>
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
          <td>18.514382</td>
          <td>0.005048</td>
          <td>28.151119</td>
          <td>0.524499</td>
          <td>20.086421</td>
          <td>0.005026</td>
          <td>26.327618</td>
          <td>0.169629</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.842124</td>
          <td>0.436381</td>
          <td>20.943618</td>
          <td>21.192117</td>
          <td>21.725943</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.131964</td>
          <td>0.050565</td>
          <td>21.157775</td>
          <td>0.005183</td>
          <td>27.580996</td>
          <td>0.302994</td>
          <td>25.827364</td>
          <td>0.110171</td>
          <td>24.208884</td>
          <td>0.050421</td>
          <td>20.706403</td>
          <td>0.007020</td>
          <td>21.147776</td>
          <td>22.467908</td>
          <td>20.140631</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.358584</td>
          <td>0.025797</td>
          <td>17.081356</td>
          <td>0.005002</td>
          <td>22.923801</td>
          <td>0.007049</td>
          <td>18.953848</td>
          <td>0.005012</td>
          <td>22.664326</td>
          <td>0.013425</td>
          <td>22.467582</td>
          <td>0.024442</td>
          <td>21.172269</td>
          <td>18.941262</td>
          <td>18.695609</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.299479</td>
          <td>0.024531</td>
          <td>22.992601</td>
          <td>0.007903</td>
          <td>22.367168</td>
          <td>0.005859</td>
          <td>24.767671</td>
          <td>0.043192</td>
          <td>24.115133</td>
          <td>0.046395</td>
          <td>22.083851</td>
          <td>0.017639</td>
          <td>18.189812</td>
          <td>28.214324</td>
          <td>25.711553</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.302661</td>
          <td>0.005413</td>
          <td>22.068284</td>
          <td>0.005725</td>
          <td>22.173297</td>
          <td>0.005629</td>
          <td>24.934837</td>
          <td>0.050102</td>
          <td>23.071990</td>
          <td>0.018681</td>
          <td>20.932054</td>
          <td>0.007823</td>
          <td>21.627362</td>
          <td>26.149240</td>
          <td>22.464571</td>
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
          <td>29.038595</td>
          <td>20.308441</td>
          <td>20.597148</td>
          <td>28.179099</td>
          <td>31.812685</td>
          <td>24.162910</td>
          <td>28.834292</td>
          <td>0.796882</td>
          <td>23.169821</td>
          <td>0.012096</td>
          <td>22.349809</td>
          <td>0.007204</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.201502</td>
          <td>23.937282</td>
          <td>30.080040</td>
          <td>21.446809</td>
          <td>18.378288</td>
          <td>19.801030</td>
          <td>24.986996</td>
          <td>0.033840</td>
          <td>20.575825</td>
          <td>0.005102</td>
          <td>22.732602</td>
          <td>0.008909</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.877232</td>
          <td>23.021218</td>
          <td>17.491714</td>
          <td>22.993833</td>
          <td>22.089700</td>
          <td>25.805671</td>
          <td>23.414573</td>
          <td>0.009391</td>
          <td>20.235013</td>
          <td>0.005054</td>
          <td>20.451367</td>
          <td>0.005081</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.302866</td>
          <td>25.625164</td>
          <td>28.528699</td>
          <td>16.517343</td>
          <td>16.313146</td>
          <td>21.763790</td>
          <td>28.117396</td>
          <td>0.482646</td>
          <td>23.406069</td>
          <td>0.014563</td>
          <td>23.612337</td>
          <td>0.017260</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.743590</td>
          <td>24.310503</td>
          <td>22.694278</td>
          <td>25.813444</td>
          <td>22.180547</td>
          <td>24.083937</td>
          <td>20.026463</td>
          <td>0.005012</td>
          <td>22.547662</td>
          <td>0.007981</td>
          <td>24.023401</td>
          <td>0.024559</td>
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
          <td>18.524023</td>
          <td>27.866537</td>
          <td>20.083999</td>
          <td>26.187547</td>
          <td>32.213351</td>
          <td>26.309141</td>
          <td>20.945721</td>
          <td>0.005067</td>
          <td>21.192242</td>
          <td>0.005310</td>
          <td>21.725961</td>
          <td>0.005791</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.166278</td>
          <td>21.160477</td>
          <td>27.242981</td>
          <td>25.724092</td>
          <td>24.125209</td>
          <td>20.711039</td>
          <td>21.146337</td>
          <td>0.005096</td>
          <td>22.473399</td>
          <td>0.007665</td>
          <td>20.132993</td>
          <td>0.005045</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.366486</td>
          <td>17.074490</td>
          <td>22.919641</td>
          <td>18.958684</td>
          <td>22.675284</td>
          <td>22.458847</td>
          <td>21.160980</td>
          <td>0.005099</td>
          <td>18.940790</td>
          <td>0.005005</td>
          <td>18.690068</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.277727</td>
          <td>22.986550</td>
          <td>22.373395</td>
          <td>24.733625</td>
          <td>24.092018</td>
          <td>22.068006</td>
          <td>18.183803</td>
          <td>0.005000</td>
          <td>27.088800</td>
          <td>0.347500</td>
          <td>25.732350</td>
          <td>0.111579</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.307886</td>
          <td>22.067876</td>
          <td>22.176979</td>
          <td>24.889471</td>
          <td>23.057752</td>
          <td>20.935526</td>
          <td>21.628592</td>
          <td>0.005231</td>
          <td>26.321927</td>
          <td>0.185377</td>
          <td>22.474639</td>
          <td>0.007670</td>
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
          <td>29.038595</td>
          <td>20.308441</td>
          <td>20.597148</td>
          <td>28.179099</td>
          <td>31.812685</td>
          <td>24.162910</td>
          <td>26.946483</td>
          <td>1.292622</td>
          <td>23.175840</td>
          <td>0.062496</td>
          <td>22.356333</td>
          <td>0.032932</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.201502</td>
          <td>23.937282</td>
          <td>30.080040</td>
          <td>21.446809</td>
          <td>18.378288</td>
          <td>19.801030</td>
          <td>25.133693</td>
          <td>0.389140</td>
          <td>20.588636</td>
          <td>0.007728</td>
          <td>22.739563</td>
          <td>0.046316</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.877232</td>
          <td>23.021218</td>
          <td>17.491714</td>
          <td>22.993833</td>
          <td>22.089700</td>
          <td>25.805671</td>
          <td>23.502332</td>
          <td>0.099569</td>
          <td>20.224762</td>
          <td>0.006540</td>
          <td>20.444230</td>
          <td>0.007549</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.302866</td>
          <td>25.625164</td>
          <td>28.528699</td>
          <td>16.517343</td>
          <td>16.313146</td>
          <td>21.763790</td>
          <td>26.486399</td>
          <td>0.994101</td>
          <td>23.304826</td>
          <td>0.070087</td>
          <td>23.875692</td>
          <td>0.126417</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.743590</td>
          <td>24.310503</td>
          <td>22.694278</td>
          <td>25.813444</td>
          <td>22.180547</td>
          <td>24.083937</td>
          <td>20.053833</td>
          <td>0.006615</td>
          <td>22.507308</td>
          <td>0.034456</td>
          <td>24.080435</td>
          <td>0.150873</td>
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
          <td>18.524023</td>
          <td>27.866537</td>
          <td>20.083999</td>
          <td>26.187547</td>
          <td>32.213351</td>
          <td>26.309141</td>
          <td>20.942496</td>
          <td>0.011002</td>
          <td>21.187353</td>
          <td>0.011370</td>
          <td>21.741284</td>
          <td>0.019248</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.166278</td>
          <td>21.160477</td>
          <td>27.242981</td>
          <td>25.724092</td>
          <td>24.125209</td>
          <td>20.711039</td>
          <td>21.159272</td>
          <td>0.012960</td>
          <td>22.428583</td>
          <td>0.032132</td>
          <td>20.137455</td>
          <td>0.006572</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.366486</td>
          <td>17.074490</td>
          <td>22.919641</td>
          <td>18.958684</td>
          <td>22.675284</td>
          <td>22.458847</td>
          <td>21.181714</td>
          <td>0.013189</td>
          <td>18.932418</td>
          <td>0.005162</td>
          <td>18.688880</td>
          <td>0.005125</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.277727</td>
          <td>22.986550</td>
          <td>22.373395</td>
          <td>24.733625</td>
          <td>24.092018</td>
          <td>22.068006</td>
          <td>18.192871</td>
          <td>0.005061</td>
          <td>26.341337</td>
          <td>0.800551</td>
          <td>24.956910</td>
          <td>0.312945</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.307886</td>
          <td>22.067876</td>
          <td>22.176979</td>
          <td>24.889471</td>
          <td>23.057752</td>
          <td>20.935526</td>
          <td>21.639840</td>
          <td>0.019224</td>
          <td>25.483166</td>
          <td>0.436363</td>
          <td>22.446502</td>
          <td>0.035677</td>
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


