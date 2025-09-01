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
          <td>18.944861</td>
          <td>23.905353</td>
          <td>27.633128</td>
          <td>26.353999</td>
          <td>25.687859</td>
          <td>20.189687</td>
          <td>27.550403</td>
          <td>19.559521</td>
          <td>21.997423</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.122244</td>
          <td>21.031580</td>
          <td>21.383751</td>
          <td>25.337766</td>
          <td>21.432578</td>
          <td>22.527463</td>
          <td>21.561437</td>
          <td>16.542580</td>
          <td>23.448297</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.423921</td>
          <td>21.902085</td>
          <td>21.469972</td>
          <td>25.838667</td>
          <td>22.652023</td>
          <td>21.747781</td>
          <td>21.011915</td>
          <td>27.948252</td>
          <td>19.499755</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.306085</td>
          <td>28.251054</td>
          <td>26.952805</td>
          <td>26.724106</td>
          <td>22.046564</td>
          <td>23.577766</td>
          <td>18.574676</td>
          <td>25.472165</td>
          <td>21.623335</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.499402</td>
          <td>26.725716</td>
          <td>18.271355</td>
          <td>21.005216</td>
          <td>22.835608</td>
          <td>19.217113</td>
          <td>25.582600</td>
          <td>23.147021</td>
          <td>27.402731</td>
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
          <td>23.046195</td>
          <td>22.150686</td>
          <td>22.054807</td>
          <td>24.860011</td>
          <td>21.533700</td>
          <td>17.300740</td>
          <td>24.854345</td>
          <td>20.585742</td>
          <td>27.345879</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.310341</td>
          <td>19.669696</td>
          <td>23.692779</td>
          <td>22.691391</td>
          <td>29.572221</td>
          <td>25.364730</td>
          <td>18.609802</td>
          <td>25.266809</td>
          <td>19.615300</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.560227</td>
          <td>17.095414</td>
          <td>20.199428</td>
          <td>29.794074</td>
          <td>20.667329</td>
          <td>19.563760</td>
          <td>22.223473</td>
          <td>19.175653</td>
          <td>19.443801</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.267388</td>
          <td>21.206180</td>
          <td>24.016269</td>
          <td>21.812715</td>
          <td>21.905253</td>
          <td>25.556597</td>
          <td>20.248785</td>
          <td>23.275830</td>
          <td>24.612567</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.045103</td>
          <td>22.728752</td>
          <td>25.819220</td>
          <td>18.686288</td>
          <td>25.798853</td>
          <td>24.738958</td>
          <td>19.535302</td>
          <td>18.725580</td>
          <td>17.628707</td>
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
          <td>18.941020</td>
          <td>0.005077</td>
          <td>23.869293</td>
          <td>0.014202</td>
          <td>27.354573</td>
          <td>0.252092</td>
          <td>26.346897</td>
          <td>0.172434</td>
          <td>25.650442</td>
          <td>0.177856</td>
          <td>20.209190</td>
          <td>0.005933</td>
          <td>27.550403</td>
          <td>19.559521</td>
          <td>21.997423</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.089514</td>
          <td>0.020564</td>
          <td>21.032073</td>
          <td>0.005152</td>
          <td>21.381475</td>
          <td>0.005177</td>
          <td>25.327021</td>
          <td>0.070947</td>
          <td>21.426565</td>
          <td>0.006489</td>
          <td>22.485198</td>
          <td>0.024818</td>
          <td>21.561437</td>
          <td>16.542580</td>
          <td>23.448297</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.166683</td>
          <td>0.289537</td>
          <td>21.909461</td>
          <td>0.005568</td>
          <td>21.469023</td>
          <td>0.005203</td>
          <td>26.045247</td>
          <td>0.133129</td>
          <td>22.657969</td>
          <td>0.013359</td>
          <td>21.757429</td>
          <td>0.013556</td>
          <td>21.011915</td>
          <td>27.948252</td>
          <td>19.499755</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.344136</td>
          <td>0.060944</td>
          <td>27.599743</td>
          <td>0.344882</td>
          <td>26.910396</td>
          <td>0.173893</td>
          <td>26.661830</td>
          <td>0.224717</td>
          <td>22.041977</td>
          <td>0.008706</td>
          <td>23.640131</td>
          <td>0.068808</td>
          <td>18.574676</td>
          <td>25.472165</td>
          <td>21.623335</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.689548</td>
          <td>0.195411</td>
          <td>26.632285</td>
          <td>0.155154</td>
          <td>18.272453</td>
          <td>0.005003</td>
          <td>21.005113</td>
          <td>0.005233</td>
          <td>22.843471</td>
          <td>0.015474</td>
          <td>19.223810</td>
          <td>0.005193</td>
          <td>25.582600</td>
          <td>23.147021</td>
          <td>27.402731</td>
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
          <td>23.029109</td>
          <td>0.019562</td>
          <td>22.155310</td>
          <td>0.005829</td>
          <td>22.049845</td>
          <td>0.005516</td>
          <td>24.862160</td>
          <td>0.046970</td>
          <td>21.519389</td>
          <td>0.006718</td>
          <td>17.297225</td>
          <td>0.005013</td>
          <td>24.854345</td>
          <td>20.585742</td>
          <td>27.345879</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.325378</td>
          <td>0.011397</td>
          <td>19.671986</td>
          <td>0.005025</td>
          <td>23.706931</td>
          <td>0.011176</td>
          <td>22.700517</td>
          <td>0.008327</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.779763</td>
          <td>0.416146</td>
          <td>18.609802</td>
          <td>25.266809</td>
          <td>19.615300</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.556759</td>
          <td>0.013470</td>
          <td>17.098331</td>
          <td>0.005002</td>
          <td>20.197765</td>
          <td>0.005031</td>
          <td>29.574501</td>
          <td>1.598613</td>
          <td>20.666960</td>
          <td>0.005444</td>
          <td>19.561108</td>
          <td>0.005330</td>
          <td>22.223473</td>
          <td>19.175653</td>
          <td>19.443801</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.290350</td>
          <td>0.024342</td>
          <td>21.204651</td>
          <td>0.005196</td>
          <td>24.037780</td>
          <td>0.014365</td>
          <td>21.813090</td>
          <td>0.005857</td>
          <td>21.892907</td>
          <td>0.007997</td>
          <td>26.124429</td>
          <td>0.538117</td>
          <td>20.248785</td>
          <td>23.275830</td>
          <td>24.612567</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.035384</td>
          <td>0.019663</td>
          <td>22.732171</td>
          <td>0.006989</td>
          <td>25.961275</td>
          <td>0.076195</td>
          <td>18.679263</td>
          <td>0.005009</td>
          <td>25.860848</td>
          <td>0.212324</td>
          <td>24.600124</td>
          <td>0.159074</td>
          <td>19.535302</td>
          <td>18.725580</td>
          <td>17.628707</td>
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
          <td>18.944861</td>
          <td>23.905353</td>
          <td>27.633128</td>
          <td>26.353999</td>
          <td>25.687859</td>
          <td>20.189687</td>
          <td>28.038152</td>
          <td>0.454870</td>
          <td>19.555263</td>
          <td>0.005016</td>
          <td>21.996521</td>
          <td>0.006248</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.122244</td>
          <td>21.031580</td>
          <td>21.383751</td>
          <td>25.337766</td>
          <td>21.432578</td>
          <td>22.527463</td>
          <td>21.560275</td>
          <td>0.005204</td>
          <td>16.539597</td>
          <td>0.005000</td>
          <td>23.412345</td>
          <td>0.014637</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.423921</td>
          <td>21.902085</td>
          <td>21.469972</td>
          <td>25.838667</td>
          <td>22.652023</td>
          <td>21.747781</td>
          <td>21.013416</td>
          <td>0.005076</td>
          <td>27.184459</td>
          <td>0.374546</td>
          <td>19.503927</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.306085</td>
          <td>28.251054</td>
          <td>26.952805</td>
          <td>26.724106</td>
          <td>22.046564</td>
          <td>23.577766</td>
          <td>18.583047</td>
          <td>0.005001</td>
          <td>25.679444</td>
          <td>0.106534</td>
          <td>21.630125</td>
          <td>0.005670</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.499402</td>
          <td>26.725716</td>
          <td>18.271355</td>
          <td>21.005216</td>
          <td>22.835608</td>
          <td>19.217113</td>
          <td>25.594032</td>
          <td>0.058107</td>
          <td>23.149778</td>
          <td>0.011914</td>
          <td>27.521417</td>
          <td>0.484090</td>
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
          <td>23.046195</td>
          <td>22.150686</td>
          <td>22.054807</td>
          <td>24.860011</td>
          <td>21.533700</td>
          <td>17.300740</td>
          <td>24.820316</td>
          <td>0.029198</td>
          <td>20.587455</td>
          <td>0.005104</td>
          <td>26.829251</td>
          <td>0.282365</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.310341</td>
          <td>19.669696</td>
          <td>23.692779</td>
          <td>22.691391</td>
          <td>29.572221</td>
          <td>25.364730</td>
          <td>18.607216</td>
          <td>0.005001</td>
          <td>25.229585</td>
          <td>0.071644</td>
          <td>19.611847</td>
          <td>0.005017</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.560227</td>
          <td>17.095414</td>
          <td>20.199428</td>
          <td>29.794074</td>
          <td>20.667329</td>
          <td>19.563760</td>
          <td>22.234313</td>
          <td>0.005675</td>
          <td>19.170034</td>
          <td>0.005008</td>
          <td>19.449485</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.267388</td>
          <td>21.206180</td>
          <td>24.016269</td>
          <td>21.812715</td>
          <td>21.905253</td>
          <td>25.556597</td>
          <td>20.240837</td>
          <td>0.005018</td>
          <td>23.278891</td>
          <td>0.013160</td>
          <td>24.559138</td>
          <td>0.039437</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.045103</td>
          <td>22.728752</td>
          <td>25.819220</td>
          <td>18.686288</td>
          <td>25.798853</td>
          <td>24.738958</td>
          <td>19.530784</td>
          <td>0.005005</td>
          <td>18.722893</td>
          <td>0.005003</td>
          <td>17.624244</td>
          <td>0.005000</td>
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
          <td>18.944861</td>
          <td>23.905353</td>
          <td>27.633128</td>
          <td>26.353999</td>
          <td>25.687859</td>
          <td>20.189687</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.557431</td>
          <td>0.005496</td>
          <td>21.940917</td>
          <td>0.022857</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.122244</td>
          <td>21.031580</td>
          <td>21.383751</td>
          <td>25.337766</td>
          <td>21.432578</td>
          <td>22.527463</td>
          <td>21.560095</td>
          <td>0.017967</td>
          <td>16.538482</td>
          <td>0.005002</td>
          <td>23.541033</td>
          <td>0.094346</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.423921</td>
          <td>21.902085</td>
          <td>21.469972</td>
          <td>25.838667</td>
          <td>22.652023</td>
          <td>21.747781</td>
          <td>21.021689</td>
          <td>0.011665</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.490247</td>
          <td>0.005525</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.306085</td>
          <td>28.251054</td>
          <td>26.952805</td>
          <td>26.724106</td>
          <td>22.046564</td>
          <td>23.577766</td>
          <td>18.582002</td>
          <td>0.005123</td>
          <td>26.622983</td>
          <td>0.956516</td>
          <td>21.635105</td>
          <td>0.017593</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.499402</td>
          <td>26.725716</td>
          <td>18.271355</td>
          <td>21.005216</td>
          <td>22.835608</td>
          <td>19.217113</td>
          <td>26.251245</td>
          <td>0.859265</td>
          <td>23.173215</td>
          <td>0.062350</td>
          <td>25.338604</td>
          <td>0.421818</td>
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
          <td>23.046195</td>
          <td>22.150686</td>
          <td>22.054807</td>
          <td>24.860011</td>
          <td>21.533700</td>
          <td>17.300740</td>
          <td>24.680023</td>
          <td>0.271290</td>
          <td>20.590321</td>
          <td>0.007735</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.310341</td>
          <td>19.669696</td>
          <td>23.692779</td>
          <td>22.691391</td>
          <td>29.572221</td>
          <td>25.364730</td>
          <td>18.611698</td>
          <td>0.005130</td>
          <td>25.250089</td>
          <td>0.364631</td>
          <td>19.615748</td>
          <td>0.005654</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.560227</td>
          <td>17.095414</td>
          <td>20.199428</td>
          <td>29.794074</td>
          <td>20.667329</td>
          <td>19.563760</td>
          <td>22.184673</td>
          <td>0.030906</td>
          <td>19.172707</td>
          <td>0.005250</td>
          <td>19.445611</td>
          <td>0.005486</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.267388</td>
          <td>21.206180</td>
          <td>24.016269</td>
          <td>21.812715</td>
          <td>21.905253</td>
          <td>25.556597</td>
          <td>20.254425</td>
          <td>0.007220</td>
          <td>23.402393</td>
          <td>0.076420</td>
          <td>24.463775</td>
          <td>0.208891</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.045103</td>
          <td>22.728752</td>
          <td>25.819220</td>
          <td>18.686288</td>
          <td>25.798853</td>
          <td>24.738958</td>
          <td>19.533737</td>
          <td>0.005674</td>
          <td>18.717971</td>
          <td>0.005110</td>
          <td>17.634146</td>
          <td>0.005018</td>
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


