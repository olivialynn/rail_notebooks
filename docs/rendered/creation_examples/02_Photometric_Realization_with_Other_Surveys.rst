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
          <td>23.317955</td>
          <td>25.942378</td>
          <td>23.858966</td>
          <td>24.499299</td>
          <td>23.070298</td>
          <td>27.065480</td>
          <td>26.772266</td>
          <td>18.749219</td>
          <td>23.362126</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.297609</td>
          <td>17.452350</td>
          <td>29.595065</td>
          <td>21.788295</td>
          <td>15.621707</td>
          <td>22.915699</td>
          <td>24.859748</td>
          <td>27.698085</td>
          <td>23.280897</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.177033</td>
          <td>22.449841</td>
          <td>20.415068</td>
          <td>26.096678</td>
          <td>25.472080</td>
          <td>26.126820</td>
          <td>14.463243</td>
          <td>23.777784</td>
          <td>25.760018</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.967657</td>
          <td>21.787404</td>
          <td>23.131983</td>
          <td>25.090147</td>
          <td>25.267191</td>
          <td>31.703026</td>
          <td>20.706940</td>
          <td>21.433175</td>
          <td>24.796290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.674324</td>
          <td>26.469084</td>
          <td>27.278404</td>
          <td>25.788069</td>
          <td>17.852552</td>
          <td>27.364293</td>
          <td>24.917670</td>
          <td>26.936622</td>
          <td>22.378946</td>
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
          <td>24.050213</td>
          <td>23.088306</td>
          <td>21.523832</td>
          <td>24.135307</td>
          <td>23.084862</td>
          <td>27.216046</td>
          <td>24.965984</td>
          <td>17.636767</td>
          <td>22.356505</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.026350</td>
          <td>23.038780</td>
          <td>20.525421</td>
          <td>20.626490</td>
          <td>23.875218</td>
          <td>20.640754</td>
          <td>29.471451</td>
          <td>22.770962</td>
          <td>26.142810</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.908418</td>
          <td>20.481654</td>
          <td>22.649533</td>
          <td>21.362214</td>
          <td>19.280655</td>
          <td>17.896567</td>
          <td>19.612981</td>
          <td>25.896861</td>
          <td>22.983616</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.346097</td>
          <td>20.448021</td>
          <td>16.451349</td>
          <td>20.909701</td>
          <td>24.825560</td>
          <td>26.485689</td>
          <td>24.107845</td>
          <td>20.076147</td>
          <td>24.752411</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.389460</td>
          <td>22.927266</td>
          <td>23.976983</td>
          <td>22.152273</td>
          <td>22.709724</td>
          <td>24.105500</td>
          <td>26.366163</td>
          <td>21.979979</td>
          <td>20.675473</td>
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
          <td>23.296443</td>
          <td>0.024468</td>
          <td>25.914627</td>
          <td>0.083122</td>
          <td>23.846484</td>
          <td>0.012387</td>
          <td>24.499813</td>
          <td>0.034074</td>
          <td>23.042702</td>
          <td>0.018229</td>
          <td>29.097168</td>
          <td>2.594783</td>
          <td>26.772266</td>
          <td>18.749219</td>
          <td>23.362126</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.344300</td>
          <td>0.145772</td>
          <td>17.457496</td>
          <td>0.005002</td>
          <td>30.386187</td>
          <td>1.802675</td>
          <td>21.792225</td>
          <td>0.005828</td>
          <td>15.618099</td>
          <td>0.005001</td>
          <td>22.894143</td>
          <td>0.035514</td>
          <td>24.859748</td>
          <td>27.698085</td>
          <td>23.280897</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.267206</td>
          <td>0.136434</td>
          <td>22.448454</td>
          <td>0.006300</td>
          <td>20.410077</td>
          <td>0.005041</td>
          <td>25.999109</td>
          <td>0.127918</td>
          <td>25.371010</td>
          <td>0.140049</td>
          <td>26.010210</td>
          <td>0.494914</td>
          <td>14.463243</td>
          <td>23.777784</td>
          <td>25.760018</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.026626</td>
          <td>0.046093</td>
          <td>21.775064</td>
          <td>0.005462</td>
          <td>23.122391</td>
          <td>0.007753</td>
          <td>25.083529</td>
          <td>0.057173</td>
          <td>25.437403</td>
          <td>0.148282</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.706940</td>
          <td>21.433175</td>
          <td>24.796290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.679480</td>
          <td>0.005184</td>
          <td>26.350540</td>
          <td>0.121699</td>
          <td>26.786601</td>
          <td>0.156471</td>
          <td>25.662050</td>
          <td>0.095329</td>
          <td>17.852407</td>
          <td>0.005008</td>
          <td>25.894139</td>
          <td>0.453866</td>
          <td>24.917670</td>
          <td>26.936622</td>
          <td>22.378946</td>
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
          <td>24.180740</td>
          <td>0.052782</td>
          <td>23.102342</td>
          <td>0.008390</td>
          <td>21.523451</td>
          <td>0.005221</td>
          <td>24.083366</td>
          <td>0.023673</td>
          <td>23.065834</td>
          <td>0.018585</td>
          <td>28.279108</td>
          <td>1.883217</td>
          <td>24.965984</td>
          <td>17.636767</td>
          <td>22.356505</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.288601</td>
          <td>0.138967</td>
          <td>23.023089</td>
          <td>0.008032</td>
          <td>20.528103</td>
          <td>0.005049</td>
          <td>20.623183</td>
          <td>0.005128</td>
          <td>23.956520</td>
          <td>0.040307</td>
          <td>20.632703</td>
          <td>0.006806</td>
          <td>29.471451</td>
          <td>22.770962</td>
          <td>26.142810</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.915369</td>
          <td>0.017826</td>
          <td>20.481288</td>
          <td>0.005071</td>
          <td>22.651308</td>
          <td>0.006347</td>
          <td>21.364409</td>
          <td>0.005415</td>
          <td>19.276674</td>
          <td>0.005052</td>
          <td>17.894421</td>
          <td>0.005028</td>
          <td>19.612981</td>
          <td>25.896861</td>
          <td>22.983616</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.555236</td>
          <td>0.174481</td>
          <td>20.439125</td>
          <td>0.005067</td>
          <td>16.449567</td>
          <td>0.005001</td>
          <td>20.912821</td>
          <td>0.005201</td>
          <td>24.752470</td>
          <td>0.081610</td>
          <td>29.601433</td>
          <td>3.061432</td>
          <td>24.107845</td>
          <td>20.076147</td>
          <td>24.752411</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.391928</td>
          <td>0.005130</td>
          <td>22.926954</td>
          <td>0.007643</td>
          <td>23.968696</td>
          <td>0.013605</td>
          <td>22.158287</td>
          <td>0.006478</td>
          <td>22.713243</td>
          <td>0.013948</td>
          <td>23.986933</td>
          <td>0.093434</td>
          <td>26.366163</td>
          <td>21.979979</td>
          <td>20.675473</td>
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
          <td>23.317955</td>
          <td>25.942378</td>
          <td>23.858966</td>
          <td>24.499299</td>
          <td>23.070298</td>
          <td>27.065480</td>
          <td>26.480768</td>
          <td>0.126975</td>
          <td>18.746676</td>
          <td>0.005004</td>
          <td>23.340668</td>
          <td>0.013819</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.297609</td>
          <td>17.452350</td>
          <td>29.595065</td>
          <td>21.788295</td>
          <td>15.621707</td>
          <td>22.915699</td>
          <td>24.883198</td>
          <td>0.030866</td>
          <td>28.130146</td>
          <td>0.743976</td>
          <td>23.273078</td>
          <td>0.013100</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.177033</td>
          <td>22.449841</td>
          <td>20.415068</td>
          <td>26.096678</td>
          <td>25.472080</td>
          <td>26.126820</td>
          <td>14.463504</td>
          <td>0.005000</td>
          <td>23.777964</td>
          <td>0.019860</td>
          <td>25.835633</td>
          <td>0.122092</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.967657</td>
          <td>21.787404</td>
          <td>23.131983</td>
          <td>25.090147</td>
          <td>25.267191</td>
          <td>31.703026</td>
          <td>20.704500</td>
          <td>0.005043</td>
          <td>21.439138</td>
          <td>0.005480</td>
          <td>24.768116</td>
          <td>0.047510</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.674324</td>
          <td>26.469084</td>
          <td>27.278404</td>
          <td>25.788069</td>
          <td>17.852552</td>
          <td>27.364293</td>
          <td>24.871120</td>
          <td>0.030538</td>
          <td>27.052673</td>
          <td>0.337725</td>
          <td>22.379711</td>
          <td>0.007309</td>
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
          <td>24.050213</td>
          <td>23.088306</td>
          <td>21.523832</td>
          <td>24.135307</td>
          <td>23.084862</td>
          <td>27.216046</td>
          <td>25.026018</td>
          <td>0.035033</td>
          <td>17.638334</td>
          <td>0.005000</td>
          <td>22.354595</td>
          <td>0.007220</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.026350</td>
          <td>23.038780</td>
          <td>20.525421</td>
          <td>20.626490</td>
          <td>23.875218</td>
          <td>20.640754</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.780790</td>
          <td>0.009187</td>
          <td>26.346697</td>
          <td>0.189300</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.908418</td>
          <td>20.481654</td>
          <td>22.649533</td>
          <td>21.362214</td>
          <td>19.280655</td>
          <td>17.896567</td>
          <td>19.619593</td>
          <td>0.005006</td>
          <td>25.924764</td>
          <td>0.131913</td>
          <td>22.994540</td>
          <td>0.010628</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.346097</td>
          <td>20.448021</td>
          <td>16.451349</td>
          <td>20.909701</td>
          <td>24.825560</td>
          <td>26.485689</td>
          <td>24.091873</td>
          <td>0.015618</td>
          <td>20.074682</td>
          <td>0.005041</td>
          <td>24.681146</td>
          <td>0.043966</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.389460</td>
          <td>22.927266</td>
          <td>23.976983</td>
          <td>22.152273</td>
          <td>22.709724</td>
          <td>24.105500</td>
          <td>26.622393</td>
          <td>0.143523</td>
          <td>21.976654</td>
          <td>0.006208</td>
          <td>20.660811</td>
          <td>0.005119</td>
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
          <td>23.317955</td>
          <td>25.942378</td>
          <td>23.858966</td>
          <td>24.499299</td>
          <td>23.070298</td>
          <td>27.065480</td>
          <td>26.774234</td>
          <td>1.175696</td>
          <td>18.755275</td>
          <td>0.005117</td>
          <td>23.422025</td>
          <td>0.084950</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.297609</td>
          <td>17.452350</td>
          <td>29.595065</td>
          <td>21.788295</td>
          <td>15.621707</td>
          <td>22.915699</td>
          <td>25.512186</td>
          <td>0.517621</td>
          <td>28.048174</td>
          <td>1.989995</td>
          <td>23.241782</td>
          <td>0.072423</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.177033</td>
          <td>22.449841</td>
          <td>20.415068</td>
          <td>26.096678</td>
          <td>25.472080</td>
          <td>26.126820</td>
          <td>14.464740</td>
          <td>0.005000</td>
          <td>23.595237</td>
          <td>0.090617</td>
          <td>25.359091</td>
          <td>0.428455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.967657</td>
          <td>21.787404</td>
          <td>23.131983</td>
          <td>25.090147</td>
          <td>25.267191</td>
          <td>31.703026</td>
          <td>20.704697</td>
          <td>0.009331</td>
          <td>21.430661</td>
          <td>0.013709</td>
          <td>24.617665</td>
          <td>0.237435</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.674324</td>
          <td>26.469084</td>
          <td>27.278404</td>
          <td>25.788069</td>
          <td>17.852552</td>
          <td>27.364293</td>
          <td>25.646832</td>
          <td>0.570660</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.411631</td>
          <td>0.034588</td>
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
          <td>24.050213</td>
          <td>23.088306</td>
          <td>21.523832</td>
          <td>24.135307</td>
          <td>23.084862</td>
          <td>27.216046</td>
          <td>24.631554</td>
          <td>0.260761</td>
          <td>17.634735</td>
          <td>0.005015</td>
          <td>22.324368</td>
          <td>0.032012</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.026350</td>
          <td>23.038780</td>
          <td>20.525421</td>
          <td>20.626490</td>
          <td>23.875218</td>
          <td>20.640754</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.735403</td>
          <td>0.042210</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.908418</td>
          <td>20.481654</td>
          <td>22.649533</td>
          <td>21.362214</td>
          <td>19.280655</td>
          <td>17.896567</td>
          <td>19.614356</td>
          <td>0.005775</td>
          <td>25.974041</td>
          <td>0.624473</td>
          <td>22.961302</td>
          <td>0.056438</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.346097</td>
          <td>20.448021</td>
          <td>16.451349</td>
          <td>20.909701</td>
          <td>24.825560</td>
          <td>26.485689</td>
          <td>24.049802</td>
          <td>0.160118</td>
          <td>20.073822</td>
          <td>0.006202</td>
          <td>25.585214</td>
          <td>0.507472</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.389460</td>
          <td>22.927266</td>
          <td>23.976983</td>
          <td>22.152273</td>
          <td>22.709724</td>
          <td>24.105500</td>
          <td>25.310007</td>
          <td>0.445319</td>
          <td>22.005011</td>
          <td>0.022156</td>
          <td>20.689347</td>
          <td>0.008672</td>
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


