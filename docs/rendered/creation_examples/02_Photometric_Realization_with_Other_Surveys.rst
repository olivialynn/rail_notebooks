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
          <td>22.845720</td>
          <td>29.145662</td>
          <td>22.851675</td>
          <td>20.744179</td>
          <td>19.995484</td>
          <td>20.009340</td>
          <td>24.267134</td>
          <td>17.729743</td>
          <td>25.017184</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.017500</td>
          <td>22.439251</td>
          <td>19.731817</td>
          <td>21.471084</td>
          <td>20.403585</td>
          <td>18.135296</td>
          <td>25.291469</td>
          <td>18.576994</td>
          <td>21.562868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.008084</td>
          <td>23.878160</td>
          <td>24.703509</td>
          <td>18.768148</td>
          <td>21.810593</td>
          <td>21.805933</td>
          <td>20.500819</td>
          <td>23.478488</td>
          <td>17.362339</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.292864</td>
          <td>19.628071</td>
          <td>23.357528</td>
          <td>25.141466</td>
          <td>24.430247</td>
          <td>20.066796</td>
          <td>23.503279</td>
          <td>21.256129</td>
          <td>25.455252</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.593539</td>
          <td>20.511619</td>
          <td>30.715599</td>
          <td>18.555102</td>
          <td>14.404488</td>
          <td>23.002493</td>
          <td>24.844005</td>
          <td>24.921077</td>
          <td>29.553163</td>
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
          <td>18.173262</td>
          <td>22.205592</td>
          <td>19.110751</td>
          <td>24.359751</td>
          <td>21.047572</td>
          <td>19.658297</td>
          <td>22.692282</td>
          <td>22.458514</td>
          <td>27.429448</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.511989</td>
          <td>17.496112</td>
          <td>20.241014</td>
          <td>20.858577</td>
          <td>22.170322</td>
          <td>22.997643</td>
          <td>19.467587</td>
          <td>23.053492</td>
          <td>22.175098</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.416220</td>
          <td>19.416493</td>
          <td>22.012360</td>
          <td>15.656204</td>
          <td>26.590841</td>
          <td>25.573880</td>
          <td>20.482598</td>
          <td>18.997461</td>
          <td>24.603747</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.514808</td>
          <td>23.549583</td>
          <td>27.337856</td>
          <td>28.695736</td>
          <td>25.871082</td>
          <td>25.495871</td>
          <td>27.403671</td>
          <td>27.562983</td>
          <td>25.764914</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.510559</td>
          <td>25.380133</td>
          <td>24.895647</td>
          <td>26.095975</td>
          <td>25.019397</td>
          <td>24.866441</td>
          <td>23.807946</td>
          <td>21.715919</td>
          <td>22.937164</td>
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
          <td>22.837682</td>
          <td>0.016747</td>
          <td>29.552669</td>
          <td>1.282143</td>
          <td>22.856978</td>
          <td>0.006851</td>
          <td>20.744188</td>
          <td>0.005154</td>
          <td>20.004608</td>
          <td>0.005155</td>
          <td>20.000792</td>
          <td>0.005668</td>
          <td>24.267134</td>
          <td>17.729743</td>
          <td>25.017184</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.018712</td>
          <td>0.006108</td>
          <td>22.441277</td>
          <td>0.006286</td>
          <td>19.728691</td>
          <td>0.005017</td>
          <td>21.483212</td>
          <td>0.005503</td>
          <td>20.409374</td>
          <td>0.005294</td>
          <td>18.127747</td>
          <td>0.005038</td>
          <td>25.291469</td>
          <td>18.576994</td>
          <td>21.562868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.664806</td>
          <td>0.427940</td>
          <td>23.888945</td>
          <td>0.014423</td>
          <td>24.681753</td>
          <td>0.024611</td>
          <td>18.762015</td>
          <td>0.005010</td>
          <td>21.821530</td>
          <td>0.007702</td>
          <td>21.815101</td>
          <td>0.014183</td>
          <td>20.500819</td>
          <td>23.478488</td>
          <td>17.362339</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.295941</td>
          <td>0.005116</td>
          <td>19.637935</td>
          <td>0.005024</td>
          <td>23.353491</td>
          <td>0.008832</td>
          <td>25.077515</td>
          <td>0.056868</td>
          <td>24.375891</td>
          <td>0.058478</td>
          <td>20.066196</td>
          <td>0.005742</td>
          <td>23.503279</td>
          <td>21.256129</td>
          <td>25.455252</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.591215</td>
          <td>0.005165</td>
          <td>20.511177</td>
          <td>0.005074</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.562108</td>
          <td>0.005008</td>
          <td>14.399392</td>
          <td>0.005000</td>
          <td>22.995196</td>
          <td>0.038834</td>
          <td>24.844005</td>
          <td>24.921077</td>
          <td>29.553163</td>
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
          <td>18.173906</td>
          <td>0.005034</td>
          <td>22.195381</td>
          <td>0.005882</td>
          <td>19.110154</td>
          <td>0.005008</td>
          <td>24.377716</td>
          <td>0.030600</td>
          <td>21.054492</td>
          <td>0.005828</td>
          <td>19.650273</td>
          <td>0.005381</td>
          <td>22.692282</td>
          <td>22.458514</td>
          <td>27.429448</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.515309</td>
          <td>0.005151</td>
          <td>17.505761</td>
          <td>0.005003</td>
          <td>20.240607</td>
          <td>0.005033</td>
          <td>20.861255</td>
          <td>0.005185</td>
          <td>22.188867</td>
          <td>0.009542</td>
          <td>22.938648</td>
          <td>0.036938</td>
          <td>19.467587</td>
          <td>23.053492</td>
          <td>22.175098</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.432097</td>
          <td>0.012291</td>
          <td>19.418589</td>
          <td>0.005019</td>
          <td>22.010756</td>
          <td>0.005484</td>
          <td>15.654357</td>
          <td>0.005000</td>
          <td>26.793640</td>
          <td>0.447163</td>
          <td>25.649947</td>
          <td>0.376496</td>
          <td>20.482598</td>
          <td>18.997461</td>
          <td>24.603747</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.942755</td>
          <td>0.526379</td>
          <td>23.533318</td>
          <td>0.011050</td>
          <td>27.494951</td>
          <td>0.282677</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.029950</td>
          <td>0.244307</td>
          <td>25.231078</td>
          <td>0.269633</td>
          <td>27.403671</td>
          <td>27.562983</td>
          <td>25.764914</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.533744</td>
          <td>0.029984</td>
          <td>25.341848</td>
          <td>0.050094</td>
          <td>24.862095</td>
          <td>0.028800</td>
          <td>26.305528</td>
          <td>0.166468</td>
          <td>24.951311</td>
          <td>0.097208</td>
          <td>24.812792</td>
          <td>0.190564</td>
          <td>23.807946</td>
          <td>21.715919</td>
          <td>22.937164</td>
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
          <td>22.845720</td>
          <td>29.145662</td>
          <td>22.851675</td>
          <td>20.744179</td>
          <td>19.995484</td>
          <td>20.009340</td>
          <td>24.276832</td>
          <td>0.018223</td>
          <td>17.734619</td>
          <td>0.005001</td>
          <td>25.091975</td>
          <td>0.063399</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.017500</td>
          <td>22.439251</td>
          <td>19.731817</td>
          <td>21.471084</td>
          <td>20.403585</td>
          <td>18.135296</td>
          <td>25.261961</td>
          <td>0.043220</td>
          <td>18.580158</td>
          <td>0.005003</td>
          <td>21.561181</td>
          <td>0.005595</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.008084</td>
          <td>23.878160</td>
          <td>24.703509</td>
          <td>18.768148</td>
          <td>21.810593</td>
          <td>21.805933</td>
          <td>20.494222</td>
          <td>0.005029</td>
          <td>23.478310</td>
          <td>0.015445</td>
          <td>17.353845</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.292864</td>
          <td>19.628071</td>
          <td>23.357528</td>
          <td>25.141466</td>
          <td>24.430247</td>
          <td>20.066796</td>
          <td>23.505919</td>
          <td>0.009987</td>
          <td>21.249468</td>
          <td>0.005343</td>
          <td>25.566111</td>
          <td>0.096450</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.593539</td>
          <td>20.511619</td>
          <td>30.715599</td>
          <td>18.555102</td>
          <td>14.404488</td>
          <td>23.002493</td>
          <td>24.821597</td>
          <td>0.029231</td>
          <td>24.899809</td>
          <td>0.053429</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>18.173262</td>
          <td>22.205592</td>
          <td>19.110751</td>
          <td>24.359751</td>
          <td>21.047572</td>
          <td>19.658297</td>
          <td>22.690990</td>
          <td>0.006458</td>
          <td>22.447477</td>
          <td>0.007562</td>
          <td>27.401241</td>
          <td>0.442378</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.511989</td>
          <td>17.496112</td>
          <td>20.241014</td>
          <td>20.858577</td>
          <td>22.170322</td>
          <td>22.997643</td>
          <td>19.476498</td>
          <td>0.005004</td>
          <td>23.057986</td>
          <td>0.011127</td>
          <td>22.172242</td>
          <td>0.006663</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.416220</td>
          <td>19.416493</td>
          <td>22.012360</td>
          <td>15.656204</td>
          <td>26.590841</td>
          <td>25.573880</td>
          <td>20.481312</td>
          <td>0.005028</td>
          <td>18.998978</td>
          <td>0.005006</td>
          <td>24.605457</td>
          <td>0.041098</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.514808</td>
          <td>23.549583</td>
          <td>27.337856</td>
          <td>28.695736</td>
          <td>25.871082</td>
          <td>25.495871</td>
          <td>27.900961</td>
          <td>0.409842</td>
          <td>26.950697</td>
          <td>0.311393</td>
          <td>25.660271</td>
          <td>0.104760</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.510559</td>
          <td>25.380133</td>
          <td>24.895647</td>
          <td>26.095975</td>
          <td>25.019397</td>
          <td>24.866441</td>
          <td>23.789285</td>
          <td>0.012277</td>
          <td>21.722358</td>
          <td>0.005786</td>
          <td>22.948590</td>
          <td>0.010288</td>
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
          <td>22.845720</td>
          <td>29.145662</td>
          <td>22.851675</td>
          <td>20.744179</td>
          <td>19.995484</td>
          <td>20.009340</td>
          <td>23.933986</td>
          <td>0.144964</td>
          <td>17.739742</td>
          <td>0.005018</td>
          <td>25.356133</td>
          <td>0.427491</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.017500</td>
          <td>22.439251</td>
          <td>19.731817</td>
          <td>21.471084</td>
          <td>20.403585</td>
          <td>18.135296</td>
          <td>25.193280</td>
          <td>0.407434</td>
          <td>18.575339</td>
          <td>0.005085</td>
          <td>21.558241</td>
          <td>0.016498</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.008084</td>
          <td>23.878160</td>
          <td>24.703509</td>
          <td>18.768148</td>
          <td>21.810593</td>
          <td>21.805933</td>
          <td>20.501161</td>
          <td>0.008228</td>
          <td>23.405860</td>
          <td>0.076655</td>
          <td>17.366266</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.292864</td>
          <td>19.628071</td>
          <td>23.357528</td>
          <td>25.141466</td>
          <td>24.430247</td>
          <td>20.066796</td>
          <td>23.376624</td>
          <td>0.089143</td>
          <td>21.244808</td>
          <td>0.011869</td>
          <td>26.742922</td>
          <td>1.090651</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.593539</td>
          <td>20.511619</td>
          <td>30.715599</td>
          <td>18.555102</td>
          <td>14.404488</td>
          <td>23.002493</td>
          <td>25.365602</td>
          <td>0.464342</td>
          <td>24.853711</td>
          <td>0.265529</td>
          <td>25.746662</td>
          <td>0.570590</td>
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
          <td>18.173262</td>
          <td>22.205592</td>
          <td>19.110751</td>
          <td>24.359751</td>
          <td>21.047572</td>
          <td>19.658297</td>
          <td>22.647995</td>
          <td>0.046666</td>
          <td>22.401037</td>
          <td>0.031357</td>
          <td>27.929910</td>
          <td>1.974673</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.511989</td>
          <td>17.496112</td>
          <td>20.241014</td>
          <td>20.858577</td>
          <td>22.170322</td>
          <td>22.997643</td>
          <td>19.454415</td>
          <td>0.005588</td>
          <td>23.036988</td>
          <td>0.055228</td>
          <td>22.172217</td>
          <td>0.027985</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.416220</td>
          <td>19.416493</td>
          <td>22.012360</td>
          <td>15.656204</td>
          <td>26.590841</td>
          <td>25.573880</td>
          <td>20.476346</td>
          <td>0.008111</td>
          <td>18.998295</td>
          <td>0.005182</td>
          <td>24.486859</td>
          <td>0.212965</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.514808</td>
          <td>23.549583</td>
          <td>27.337856</td>
          <td>28.695736</td>
          <td>25.871082</td>
          <td>25.495871</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.159254</td>
          <td>1.301526</td>
          <td>25.148863</td>
          <td>0.364282</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.510559</td>
          <td>25.380133</td>
          <td>24.895647</td>
          <td>26.095975</td>
          <td>25.019397</td>
          <td>24.866441</td>
          <td>23.844335</td>
          <td>0.134167</td>
          <td>21.703528</td>
          <td>0.017133</td>
          <td>23.008389</td>
          <td>0.058855</td>
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


