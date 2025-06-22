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
          <td>29.548860</td>
          <td>26.719509</td>
          <td>22.606562</td>
          <td>20.931297</td>
          <td>25.591667</td>
          <td>25.095942</td>
          <td>19.947248</td>
          <td>22.230408</td>
          <td>21.801882</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.620623</td>
          <td>27.637371</td>
          <td>24.522380</td>
          <td>24.815648</td>
          <td>17.880666</td>
          <td>18.044053</td>
          <td>23.172841</td>
          <td>22.032893</td>
          <td>26.294624</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.562932</td>
          <td>22.379813</td>
          <td>23.372699</td>
          <td>26.359945</td>
          <td>19.546935</td>
          <td>24.669629</td>
          <td>23.005237</td>
          <td>23.786671</td>
          <td>16.567120</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.860792</td>
          <td>23.560748</td>
          <td>20.741514</td>
          <td>24.160964</td>
          <td>21.740607</td>
          <td>23.933797</td>
          <td>18.371604</td>
          <td>21.653959</td>
          <td>24.853832</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.295457</td>
          <td>24.725390</td>
          <td>26.214391</td>
          <td>19.991069</td>
          <td>19.537239</td>
          <td>29.406522</td>
          <td>22.151366</td>
          <td>24.206203</td>
          <td>20.669567</td>
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
          <td>17.204525</td>
          <td>22.860794</td>
          <td>27.645450</td>
          <td>24.649266</td>
          <td>23.461022</td>
          <td>19.708659</td>
          <td>19.044300</td>
          <td>25.839163</td>
          <td>23.030408</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.927163</td>
          <td>19.599696</td>
          <td>24.928454</td>
          <td>22.943064</td>
          <td>19.969158</td>
          <td>21.395259</td>
          <td>26.251615</td>
          <td>25.191619</td>
          <td>24.426854</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.309880</td>
          <td>22.179323</td>
          <td>24.907598</td>
          <td>27.369095</td>
          <td>33.103091</td>
          <td>20.469669</td>
          <td>23.167880</td>
          <td>24.169020</td>
          <td>28.379891</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.874520</td>
          <td>18.530991</td>
          <td>24.686016</td>
          <td>26.800393</td>
          <td>25.292340</td>
          <td>20.283857</td>
          <td>24.284365</td>
          <td>27.750954</td>
          <td>24.721060</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.597559</td>
          <td>25.382406</td>
          <td>29.488811</td>
          <td>22.748768</td>
          <td>26.098367</td>
          <td>25.328146</td>
          <td>19.790857</td>
          <td>24.044865</td>
          <td>25.464486</td>
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
          <td>28.903398</td>
          <td>1.701483</td>
          <td>26.827654</td>
          <td>0.183211</td>
          <td>22.605605</td>
          <td>0.006254</td>
          <td>20.931958</td>
          <td>0.005207</td>
          <td>25.636072</td>
          <td>0.175701</td>
          <td>25.329701</td>
          <td>0.292080</td>
          <td>19.947248</td>
          <td>22.230408</td>
          <td>21.801882</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.744772</td>
          <td>0.204668</td>
          <td>27.425988</td>
          <td>0.300328</td>
          <td>24.525686</td>
          <td>0.021516</td>
          <td>24.818739</td>
          <td>0.045194</td>
          <td>17.884403</td>
          <td>0.005009</td>
          <td>18.050634</td>
          <td>0.005034</td>
          <td>23.172841</td>
          <td>22.032893</td>
          <td>26.294624</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.837212</td>
          <td>0.487079</td>
          <td>22.374005</td>
          <td>0.006160</td>
          <td>23.375132</td>
          <td>0.008949</td>
          <td>26.296858</td>
          <td>0.165242</td>
          <td>19.545135</td>
          <td>0.005077</td>
          <td>24.747921</td>
          <td>0.180396</td>
          <td>23.005237</td>
          <td>23.786671</td>
          <td>16.567120</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.862273</td>
          <td>0.005232</td>
          <td>23.564795</td>
          <td>0.011299</td>
          <td>20.739358</td>
          <td>0.005066</td>
          <td>24.213535</td>
          <td>0.026503</td>
          <td>21.739709</td>
          <td>0.007395</td>
          <td>23.957722</td>
          <td>0.091066</td>
          <td>18.371604</td>
          <td>21.653959</td>
          <td>24.853832</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.177892</td>
          <td>0.622730</td>
          <td>24.692008</td>
          <td>0.028263</td>
          <td>26.398512</td>
          <td>0.111874</td>
          <td>19.988933</td>
          <td>0.005049</td>
          <td>19.544016</td>
          <td>0.005077</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.151366</td>
          <td>24.206203</td>
          <td>20.669567</td>
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
          <td>17.204652</td>
          <td>0.005013</td>
          <td>22.853020</td>
          <td>0.007375</td>
          <td>27.395093</td>
          <td>0.260605</td>
          <td>24.663552</td>
          <td>0.039383</td>
          <td>23.480724</td>
          <td>0.026516</td>
          <td>19.712277</td>
          <td>0.005420</td>
          <td>19.044300</td>
          <td>25.839163</td>
          <td>23.030408</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.925409</td>
          <td>0.017972</td>
          <td>19.598290</td>
          <td>0.005023</td>
          <td>24.911230</td>
          <td>0.030068</td>
          <td>22.934811</td>
          <td>0.009613</td>
          <td>19.968680</td>
          <td>0.005147</td>
          <td>21.398500</td>
          <td>0.010405</td>
          <td>26.251615</td>
          <td>25.191619</td>
          <td>24.426854</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.321221</td>
          <td>0.005423</td>
          <td>22.183646</td>
          <td>0.005866</td>
          <td>24.922484</td>
          <td>0.030367</td>
          <td>27.502282</td>
          <td>0.439049</td>
          <td>27.854766</td>
          <td>0.929446</td>
          <td>20.467955</td>
          <td>0.006401</td>
          <td>23.167880</td>
          <td>24.169020</td>
          <td>28.379891</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.824288</td>
          <td>0.038602</td>
          <td>18.529540</td>
          <td>0.005007</td>
          <td>24.672998</td>
          <td>0.024426</td>
          <td>26.993567</td>
          <td>0.294857</td>
          <td>25.237185</td>
          <td>0.124745</td>
          <td>20.276582</td>
          <td>0.006038</td>
          <td>24.284365</td>
          <td>27.750954</td>
          <td>24.721060</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.608403</td>
          <td>0.182506</td>
          <td>25.384975</td>
          <td>0.052045</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.757315</td>
          <td>0.008606</td>
          <td>25.729005</td>
          <td>0.190076</td>
          <td>25.298173</td>
          <td>0.284733</td>
          <td>19.790857</td>
          <td>24.044865</td>
          <td>25.464486</td>
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
          <td>29.548860</td>
          <td>26.719509</td>
          <td>22.606562</td>
          <td>20.931297</td>
          <td>25.591667</td>
          <td>25.095942</td>
          <td>19.940992</td>
          <td>0.005011</td>
          <td>22.246234</td>
          <td>0.006872</td>
          <td>21.798053</td>
          <td>0.005894</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.620623</td>
          <td>27.637371</td>
          <td>24.522380</td>
          <td>24.815648</td>
          <td>17.880666</td>
          <td>18.044053</td>
          <td>23.166680</td>
          <td>0.008067</td>
          <td>22.029547</td>
          <td>0.006318</td>
          <td>25.975345</td>
          <td>0.137813</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.562932</td>
          <td>22.379813</td>
          <td>23.372699</td>
          <td>26.359945</td>
          <td>19.546935</td>
          <td>24.669629</td>
          <td>22.999615</td>
          <td>0.007381</td>
          <td>23.790656</td>
          <td>0.020077</td>
          <td>16.565339</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.860792</td>
          <td>23.560748</td>
          <td>20.741514</td>
          <td>24.160964</td>
          <td>21.740607</td>
          <td>23.933797</td>
          <td>18.378422</td>
          <td>0.005001</td>
          <td>21.651456</td>
          <td>0.005695</td>
          <td>24.874316</td>
          <td>0.052228</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.295457</td>
          <td>24.725390</td>
          <td>26.214391</td>
          <td>19.991069</td>
          <td>19.537239</td>
          <td>29.406522</td>
          <td>22.159491</td>
          <td>0.005593</td>
          <td>24.194332</td>
          <td>0.028536</td>
          <td>20.656511</td>
          <td>0.005118</td>
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
          <td>17.204525</td>
          <td>22.860794</td>
          <td>27.645450</td>
          <td>24.649266</td>
          <td>23.461022</td>
          <td>19.708659</td>
          <td>19.034637</td>
          <td>0.005002</td>
          <td>26.006714</td>
          <td>0.141596</td>
          <td>23.013091</td>
          <td>0.010770</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.927163</td>
          <td>19.599696</td>
          <td>24.928454</td>
          <td>22.943064</td>
          <td>19.969158</td>
          <td>21.395259</td>
          <td>26.414336</td>
          <td>0.119850</td>
          <td>25.115056</td>
          <td>0.064714</td>
          <td>24.421121</td>
          <td>0.034881</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.309880</td>
          <td>22.179323</td>
          <td>24.907598</td>
          <td>27.369095</td>
          <td>33.103091</td>
          <td>20.469669</td>
          <td>23.169271</td>
          <td>0.008079</td>
          <td>24.185097</td>
          <td>0.028304</td>
          <td>27.947645</td>
          <td>0.657281</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.874520</td>
          <td>18.530991</td>
          <td>24.686016</td>
          <td>26.800393</td>
          <td>25.292340</td>
          <td>20.283857</td>
          <td>24.289765</td>
          <td>0.018423</td>
          <td>27.582052</td>
          <td>0.506293</td>
          <td>24.670975</td>
          <td>0.043569</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.597559</td>
          <td>25.382406</td>
          <td>29.488811</td>
          <td>22.748768</td>
          <td>26.098367</td>
          <td>25.328146</td>
          <td>19.795929</td>
          <td>0.005008</td>
          <td>24.086185</td>
          <td>0.025947</td>
          <td>25.549944</td>
          <td>0.095089</td>
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
          <td>29.548860</td>
          <td>26.719509</td>
          <td>22.606562</td>
          <td>20.931297</td>
          <td>25.591667</td>
          <td>25.095942</td>
          <td>19.951738</td>
          <td>0.006367</td>
          <td>22.253391</td>
          <td>0.027525</td>
          <td>21.815583</td>
          <td>0.020511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.620623</td>
          <td>27.637371</td>
          <td>24.522380</td>
          <td>24.815648</td>
          <td>17.880666</td>
          <td>18.044053</td>
          <td>23.036794</td>
          <td>0.065977</td>
          <td>22.037112</td>
          <td>0.022781</td>
          <td>25.143404</td>
          <td>0.362729</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.562932</td>
          <td>22.379813</td>
          <td>23.372699</td>
          <td>26.359945</td>
          <td>19.546935</td>
          <td>24.669629</td>
          <td>22.990395</td>
          <td>0.063311</td>
          <td>23.541859</td>
          <td>0.086451</td>
          <td>16.563523</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.860792</td>
          <td>23.560748</td>
          <td>20.741514</td>
          <td>24.160964</td>
          <td>21.740607</td>
          <td>23.933797</td>
          <td>18.371399</td>
          <td>0.005084</td>
          <td>21.642541</td>
          <td>0.016285</td>
          <td>25.228508</td>
          <td>0.387581</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.295457</td>
          <td>24.725390</td>
          <td>26.214391</td>
          <td>19.991069</td>
          <td>19.537239</td>
          <td>29.406522</td>
          <td>22.202033</td>
          <td>0.031385</td>
          <td>24.036677</td>
          <td>0.133281</td>
          <td>20.672285</td>
          <td>0.008583</td>
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
          <td>17.204525</td>
          <td>22.860794</td>
          <td>27.645450</td>
          <td>24.649266</td>
          <td>23.461022</td>
          <td>19.708659</td>
          <td>19.048270</td>
          <td>0.005286</td>
          <td>25.593671</td>
          <td>0.474191</td>
          <td>23.031668</td>
          <td>0.060087</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.927163</td>
          <td>19.599696</td>
          <td>24.928454</td>
          <td>22.943064</td>
          <td>19.969158</td>
          <td>21.395259</td>
          <td>30.001305</td>
          <td>3.981969</td>
          <td>25.177238</td>
          <td>0.344346</td>
          <td>24.220240</td>
          <td>0.170043</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.309880</td>
          <td>22.179323</td>
          <td>24.907598</td>
          <td>27.369095</td>
          <td>33.103091</td>
          <td>20.469669</td>
          <td>23.134469</td>
          <td>0.071955</td>
          <td>24.353724</td>
          <td>0.174958</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.874520</td>
          <td>18.530991</td>
          <td>24.686016</td>
          <td>26.800393</td>
          <td>25.292340</td>
          <td>20.283857</td>
          <td>24.533562</td>
          <td>0.240575</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.747928</td>
          <td>0.264277</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.597559</td>
          <td>25.382406</td>
          <td>29.488811</td>
          <td>22.748768</td>
          <td>26.098367</td>
          <td>25.328146</td>
          <td>19.776198</td>
          <td>0.006021</td>
          <td>23.910782</td>
          <td>0.119480</td>
          <td>27.007696</td>
          <td>1.265776</td>
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


