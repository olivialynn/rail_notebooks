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
          <td>21.133691</td>
          <td>24.696847</td>
          <td>18.803787</td>
          <td>24.208933</td>
          <td>25.355160</td>
          <td>22.477420</td>
          <td>21.204675</td>
          <td>26.113302</td>
          <td>23.497935</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.376050</td>
          <td>23.129630</td>
          <td>20.378342</td>
          <td>26.532655</td>
          <td>28.418073</td>
          <td>22.589849</td>
          <td>22.329705</td>
          <td>24.274507</td>
          <td>19.007520</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.644011</td>
          <td>20.024369</td>
          <td>24.266819</td>
          <td>19.015655</td>
          <td>24.814871</td>
          <td>17.773707</td>
          <td>30.122845</td>
          <td>27.795567</td>
          <td>21.471978</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.492739</td>
          <td>27.031614</td>
          <td>23.828011</td>
          <td>19.879316</td>
          <td>23.110211</td>
          <td>22.398903</td>
          <td>22.604158</td>
          <td>22.585424</td>
          <td>20.236546</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.564934</td>
          <td>21.169285</td>
          <td>18.272494</td>
          <td>19.550293</td>
          <td>24.744358</td>
          <td>21.452161</td>
          <td>21.852094</td>
          <td>20.199381</td>
          <td>19.854754</td>
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
          <td>21.953339</td>
          <td>22.444120</td>
          <td>18.267066</td>
          <td>22.971857</td>
          <td>19.924904</td>
          <td>20.713737</td>
          <td>21.410337</td>
          <td>17.264740</td>
          <td>22.054411</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.526750</td>
          <td>23.891466</td>
          <td>21.969442</td>
          <td>23.600969</td>
          <td>29.401781</td>
          <td>23.509560</td>
          <td>27.592569</td>
          <td>28.235782</td>
          <td>23.364771</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.105239</td>
          <td>19.610761</td>
          <td>22.339538</td>
          <td>24.026420</td>
          <td>19.344400</td>
          <td>26.336814</td>
          <td>20.679390</td>
          <td>25.564457</td>
          <td>25.112593</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.627059</td>
          <td>27.005658</td>
          <td>19.487840</td>
          <td>20.590701</td>
          <td>19.619627</td>
          <td>22.808079</td>
          <td>22.473209</td>
          <td>27.169437</td>
          <td>19.335309</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.224898</td>
          <td>26.709276</td>
          <td>25.580814</td>
          <td>23.590241</td>
          <td>23.829205</td>
          <td>23.483758</td>
          <td>18.259923</td>
          <td>22.417699</td>
          <td>21.071425</td>
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
          <td>21.140524</td>
          <td>0.006314</td>
          <td>24.708202</td>
          <td>0.028664</td>
          <td>18.811321</td>
          <td>0.005006</td>
          <td>24.224050</td>
          <td>0.026748</td>
          <td>25.194435</td>
          <td>0.120199</td>
          <td>22.462940</td>
          <td>0.024344</td>
          <td>21.204675</td>
          <td>26.113302</td>
          <td>23.497935</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.261491</td>
          <td>0.135764</td>
          <td>23.115262</td>
          <td>0.008451</td>
          <td>20.371380</td>
          <td>0.005039</td>
          <td>26.513471</td>
          <td>0.198510</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.615213</td>
          <td>0.027791</td>
          <td>22.329705</td>
          <td>24.274507</td>
          <td>19.007520</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.637511</td>
          <td>0.187042</td>
          <td>20.032451</td>
          <td>0.005040</td>
          <td>24.269593</td>
          <td>0.017339</td>
          <td>19.011796</td>
          <td>0.005013</td>
          <td>24.840046</td>
          <td>0.088155</td>
          <td>17.768777</td>
          <td>0.005023</td>
          <td>30.122845</td>
          <td>27.795567</td>
          <td>21.471978</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.487393</td>
          <td>0.005530</td>
          <td>26.920421</td>
          <td>0.198112</td>
          <td>23.833628</td>
          <td>0.012268</td>
          <td>19.880116</td>
          <td>0.005042</td>
          <td>23.116207</td>
          <td>0.019389</td>
          <td>22.390028</td>
          <td>0.022859</td>
          <td>22.604158</td>
          <td>22.585424</td>
          <td>20.236546</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.565005</td>
          <td>0.005160</td>
          <td>21.174292</td>
          <td>0.005187</td>
          <td>18.272709</td>
          <td>0.005003</td>
          <td>19.542170</td>
          <td>0.005026</td>
          <td>24.659036</td>
          <td>0.075148</td>
          <td>21.465085</td>
          <td>0.010901</td>
          <td>21.852094</td>
          <td>20.199381</td>
          <td>19.854754</td>
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
          <td>21.954124</td>
          <td>0.008988</td>
          <td>22.441241</td>
          <td>0.006285</td>
          <td>18.264789</td>
          <td>0.005003</td>
          <td>22.980935</td>
          <td>0.009911</td>
          <td>19.932480</td>
          <td>0.005139</td>
          <td>20.702219</td>
          <td>0.007007</td>
          <td>21.410337</td>
          <td>17.264740</td>
          <td>22.054411</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.532004</td>
          <td>0.013224</td>
          <td>23.870159</td>
          <td>0.014212</td>
          <td>21.967852</td>
          <td>0.005452</td>
          <td>23.585603</td>
          <td>0.015579</td>
          <td>27.245399</td>
          <td>0.621441</td>
          <td>23.509841</td>
          <td>0.061306</td>
          <td>27.592569</td>
          <td>28.235782</td>
          <td>23.364771</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.086643</td>
          <td>0.116732</td>
          <td>19.611500</td>
          <td>0.005024</td>
          <td>22.341326</td>
          <td>0.005824</td>
          <td>24.023925</td>
          <td>0.022491</td>
          <td>19.342822</td>
          <td>0.005057</td>
          <td>26.260040</td>
          <td>0.593130</td>
          <td>20.679390</td>
          <td>25.564457</td>
          <td>25.112593</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.735947</td>
          <td>0.451597</td>
          <td>26.889583</td>
          <td>0.193039</td>
          <td>19.480492</td>
          <td>0.005012</td>
          <td>20.594712</td>
          <td>0.005122</td>
          <td>19.615569</td>
          <td>0.005086</td>
          <td>22.819060</td>
          <td>0.033237</td>
          <td>22.473209</td>
          <td>27.169437</td>
          <td>19.335309</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.216457</td>
          <td>0.006461</td>
          <td>26.562437</td>
          <td>0.146135</td>
          <td>25.523864</td>
          <td>0.051706</td>
          <td>23.593175</td>
          <td>0.015674</td>
          <td>23.745118</td>
          <td>0.033435</td>
          <td>23.423695</td>
          <td>0.056795</td>
          <td>18.259923</td>
          <td>22.417699</td>
          <td>21.071425</td>
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
          <td>21.133691</td>
          <td>24.696847</td>
          <td>18.803787</td>
          <td>24.208933</td>
          <td>25.355160</td>
          <td>22.477420</td>
          <td>21.214278</td>
          <td>0.005109</td>
          <td>25.970344</td>
          <td>0.137219</td>
          <td>23.475511</td>
          <td>0.015409</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.376050</td>
          <td>23.129630</td>
          <td>20.378342</td>
          <td>26.532655</td>
          <td>28.418073</td>
          <td>22.589849</td>
          <td>22.338894</td>
          <td>0.005808</td>
          <td>24.248292</td>
          <td>0.029928</td>
          <td>19.006708</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.644011</td>
          <td>20.024369</td>
          <td>24.266819</td>
          <td>19.015655</td>
          <td>24.814871</td>
          <td>17.773707</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.009123</td>
          <td>0.685632</td>
          <td>21.469415</td>
          <td>0.005506</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.492739</td>
          <td>27.031614</td>
          <td>23.828011</td>
          <td>19.879316</td>
          <td>23.110211</td>
          <td>22.398903</td>
          <td>22.613202</td>
          <td>0.006283</td>
          <td>22.579325</td>
          <td>0.008125</td>
          <td>20.230498</td>
          <td>0.005054</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.564934</td>
          <td>21.169285</td>
          <td>18.272494</td>
          <td>19.550293</td>
          <td>24.744358</td>
          <td>21.452161</td>
          <td>21.846552</td>
          <td>0.005341</td>
          <td>20.191091</td>
          <td>0.005050</td>
          <td>19.851130</td>
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
          <td>21.953339</td>
          <td>22.444120</td>
          <td>18.267066</td>
          <td>22.971857</td>
          <td>19.924904</td>
          <td>20.713737</td>
          <td>21.407575</td>
          <td>0.005155</td>
          <td>17.270666</td>
          <td>0.005000</td>
          <td>22.055120</td>
          <td>0.006375</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.526750</td>
          <td>23.891466</td>
          <td>21.969442</td>
          <td>23.600969</td>
          <td>29.401781</td>
          <td>23.509560</td>
          <td>28.125697</td>
          <td>0.485632</td>
          <td>28.346280</td>
          <td>0.856553</td>
          <td>23.360469</td>
          <td>0.014039</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.105239</td>
          <td>19.610761</td>
          <td>22.339538</td>
          <td>24.026420</td>
          <td>19.344400</td>
          <td>26.336814</td>
          <td>20.676348</td>
          <td>0.005041</td>
          <td>25.372934</td>
          <td>0.081344</td>
          <td>25.115070</td>
          <td>0.064715</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.627059</td>
          <td>27.005658</td>
          <td>19.487840</td>
          <td>20.590701</td>
          <td>19.619627</td>
          <td>22.808079</td>
          <td>22.463910</td>
          <td>0.006000</td>
          <td>27.628289</td>
          <td>0.523755</td>
          <td>19.333053</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.224898</td>
          <td>26.709276</td>
          <td>25.580814</td>
          <td>23.590241</td>
          <td>23.829205</td>
          <td>23.483758</td>
          <td>18.261953</td>
          <td>0.005000</td>
          <td>22.422115</td>
          <td>0.007465</td>
          <td>21.080942</td>
          <td>0.005254</td>
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
          <td>21.133691</td>
          <td>24.696847</td>
          <td>18.803787</td>
          <td>24.208933</td>
          <td>25.355160</td>
          <td>22.477420</td>
          <td>21.206139</td>
          <td>0.013445</td>
          <td>25.636071</td>
          <td>0.489384</td>
          <td>23.489900</td>
          <td>0.090192</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.376050</td>
          <td>23.129630</td>
          <td>20.378342</td>
          <td>26.532655</td>
          <td>28.418073</td>
          <td>22.589849</td>
          <td>22.371255</td>
          <td>0.036471</td>
          <td>24.225275</td>
          <td>0.156791</td>
          <td>19.015331</td>
          <td>0.005225</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.644011</td>
          <td>20.024369</td>
          <td>24.266819</td>
          <td>19.015655</td>
          <td>24.814871</td>
          <td>17.773707</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.465857</td>
          <td>0.015288</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.492739</td>
          <td>27.031614</td>
          <td>23.828011</td>
          <td>19.879316</td>
          <td>23.110211</td>
          <td>22.398903</td>
          <td>22.676006</td>
          <td>0.047846</td>
          <td>22.513901</td>
          <td>0.034658</td>
          <td>20.238291</td>
          <td>0.006849</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.564934</td>
          <td>21.169285</td>
          <td>18.272494</td>
          <td>19.550293</td>
          <td>24.744358</td>
          <td>21.452161</td>
          <td>21.853761</td>
          <td>0.023113</td>
          <td>20.194583</td>
          <td>0.006467</td>
          <td>19.844219</td>
          <td>0.005967</td>
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
          <td>21.953339</td>
          <td>22.444120</td>
          <td>18.267066</td>
          <td>22.971857</td>
          <td>19.924904</td>
          <td>20.713737</td>
          <td>21.379995</td>
          <td>0.015466</td>
          <td>17.270624</td>
          <td>0.005008</td>
          <td>22.017116</td>
          <td>0.024425</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.526750</td>
          <td>23.891466</td>
          <td>21.969442</td>
          <td>23.600969</td>
          <td>29.401781</td>
          <td>23.509560</td>
          <td>26.636655</td>
          <td>1.086683</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.395854</td>
          <td>0.083008</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.105239</td>
          <td>19.610761</td>
          <td>22.339538</td>
          <td>24.026420</td>
          <td>19.344400</td>
          <td>26.336814</td>
          <td>20.666029</td>
          <td>0.009100</td>
          <td>25.109320</td>
          <td>0.326306</td>
          <td>25.564187</td>
          <td>0.499670</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.627059</td>
          <td>27.005658</td>
          <td>19.487840</td>
          <td>20.590701</td>
          <td>19.619627</td>
          <td>22.808079</td>
          <td>22.557678</td>
          <td>0.043056</td>
          <td>25.326607</td>
          <td>0.387011</td>
          <td>19.340160</td>
          <td>0.005403</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.224898</td>
          <td>26.709276</td>
          <td>25.580814</td>
          <td>23.590241</td>
          <td>23.829205</td>
          <td>23.483758</td>
          <td>18.266999</td>
          <td>0.005069</td>
          <td>22.392107</td>
          <td>0.031110</td>
          <td>21.061667</td>
          <td>0.011157</td>
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


