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
          <td>16.957155</td>
          <td>20.449162</td>
          <td>24.490551</td>
          <td>24.326024</td>
          <td>23.636808</td>
          <td>24.224038</td>
          <td>23.524714</td>
          <td>23.233678</td>
          <td>28.464552</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.814532</td>
          <td>19.996148</td>
          <td>25.237678</td>
          <td>26.473320</td>
          <td>25.691341</td>
          <td>21.588687</td>
          <td>23.567850</td>
          <td>21.765042</td>
          <td>31.644475</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.070162</td>
          <td>23.630314</td>
          <td>22.349150</td>
          <td>22.017158</td>
          <td>23.181075</td>
          <td>18.299252</td>
          <td>22.600164</td>
          <td>22.045156</td>
          <td>23.513812</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.688312</td>
          <td>22.868756</td>
          <td>22.585485</td>
          <td>19.997192</td>
          <td>27.577378</td>
          <td>23.173233</td>
          <td>21.563009</td>
          <td>23.549466</td>
          <td>21.263522</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.882951</td>
          <td>26.447143</td>
          <td>26.670712</td>
          <td>25.758942</td>
          <td>26.562569</td>
          <td>26.011253</td>
          <td>27.061604</td>
          <td>25.049129</td>
          <td>19.042013</td>
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
          <td>27.488547</td>
          <td>22.631655</td>
          <td>21.165346</td>
          <td>21.120921</td>
          <td>26.887267</td>
          <td>23.109250</td>
          <td>23.051888</td>
          <td>17.476483</td>
          <td>22.157353</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.709431</td>
          <td>21.724414</td>
          <td>21.727664</td>
          <td>27.558420</td>
          <td>19.971493</td>
          <td>22.952474</td>
          <td>26.662702</td>
          <td>28.016393</td>
          <td>23.546421</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.625763</td>
          <td>21.032894</td>
          <td>27.815813</td>
          <td>24.247503</td>
          <td>21.345806</td>
          <td>24.189072</td>
          <td>20.371905</td>
          <td>27.939186</td>
          <td>20.047846</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.378059</td>
          <td>26.471287</td>
          <td>23.205571</td>
          <td>20.047233</td>
          <td>25.060985</td>
          <td>24.438920</td>
          <td>21.317288</td>
          <td>24.088197</td>
          <td>25.796152</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.793784</td>
          <td>21.927870</td>
          <td>25.611626</td>
          <td>25.242086</td>
          <td>29.794462</td>
          <td>27.582915</td>
          <td>22.258975</td>
          <td>17.555488</td>
          <td>25.423870</td>
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
          <td>16.948065</td>
          <td>0.005010</td>
          <td>20.442662</td>
          <td>0.005067</td>
          <td>24.526000</td>
          <td>0.021522</td>
          <td>24.350300</td>
          <td>0.029872</td>
          <td>23.686232</td>
          <td>0.031746</td>
          <td>24.432814</td>
          <td>0.137780</td>
          <td>23.524714</td>
          <td>23.233678</td>
          <td>28.464552</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.696329</td>
          <td>0.196527</td>
          <td>20.003034</td>
          <td>0.005038</td>
          <td>25.346745</td>
          <td>0.044183</td>
          <td>26.461168</td>
          <td>0.189956</td>
          <td>26.056600</td>
          <td>0.249724</td>
          <td>21.588930</td>
          <td>0.011925</td>
          <td>23.567850</td>
          <td>21.765042</td>
          <td>31.644475</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.073430</td>
          <td>0.005011</td>
          <td>23.621197</td>
          <td>0.011769</td>
          <td>22.348590</td>
          <td>0.005834</td>
          <td>22.020779</td>
          <td>0.006192</td>
          <td>23.171064</td>
          <td>0.020310</td>
          <td>18.308355</td>
          <td>0.005049</td>
          <td>22.600164</td>
          <td>22.045156</td>
          <td>23.513812</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.688022</td>
          <td>0.007799</td>
          <td>22.870065</td>
          <td>0.007434</td>
          <td>22.576854</td>
          <td>0.006199</td>
          <td>19.984424</td>
          <td>0.005049</td>
          <td>26.715398</td>
          <td>0.421388</td>
          <td>23.223665</td>
          <td>0.047555</td>
          <td>21.563009</td>
          <td>23.549466</td>
          <td>21.263522</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.811004</td>
          <td>0.038156</td>
          <td>26.512410</td>
          <td>0.139980</td>
          <td>26.702084</td>
          <td>0.145526</td>
          <td>25.770076</td>
          <td>0.104793</td>
          <td>26.913288</td>
          <td>0.489026</td>
          <td>25.094451</td>
          <td>0.241056</td>
          <td>27.061604</td>
          <td>25.049129</td>
          <td>19.042013</td>
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
          <td>27.311703</td>
          <td>0.683123</td>
          <td>22.625326</td>
          <td>0.006697</td>
          <td>21.165848</td>
          <td>0.005126</td>
          <td>21.119955</td>
          <td>0.005280</td>
          <td>26.994824</td>
          <td>0.519293</td>
          <td>23.104890</td>
          <td>0.042798</td>
          <td>23.051888</td>
          <td>17.476483</td>
          <td>22.157353</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.723821</td>
          <td>0.015306</td>
          <td>21.728532</td>
          <td>0.005430</td>
          <td>21.729602</td>
          <td>0.005308</td>
          <td>27.157531</td>
          <td>0.336122</td>
          <td>19.974127</td>
          <td>0.005148</td>
          <td>22.972484</td>
          <td>0.038061</td>
          <td>26.662702</td>
          <td>28.016393</td>
          <td>23.546421</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.620692</td>
          <td>0.005636</td>
          <td>21.032094</td>
          <td>0.005152</td>
          <td>27.856658</td>
          <td>0.376787</td>
          <td>24.291631</td>
          <td>0.028374</td>
          <td>21.349227</td>
          <td>0.006320</td>
          <td>24.332797</td>
          <td>0.126364</td>
          <td>20.371905</td>
          <td>27.939186</td>
          <td>20.047846</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.375315</td>
          <td>0.005127</td>
          <td>26.709354</td>
          <td>0.165708</td>
          <td>23.196546</td>
          <td>0.008066</td>
          <td>20.042556</td>
          <td>0.005053</td>
          <td>25.013583</td>
          <td>0.102658</td>
          <td>24.546978</td>
          <td>0.151997</td>
          <td>21.317288</td>
          <td>24.088197</td>
          <td>25.796152</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.790742</td>
          <td>0.005065</td>
          <td>21.930722</td>
          <td>0.005587</td>
          <td>25.652093</td>
          <td>0.057941</td>
          <td>25.251697</td>
          <td>0.066369</td>
          <td>29.315038</td>
          <td>1.980196</td>
          <td>25.862760</td>
          <td>0.443252</td>
          <td>22.258975</td>
          <td>17.555488</td>
          <td>25.423870</td>
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
          <td>16.957155</td>
          <td>20.449162</td>
          <td>24.490551</td>
          <td>24.326024</td>
          <td>23.636808</td>
          <td>24.224038</td>
          <td>23.537677</td>
          <td>0.010209</td>
          <td>23.234052</td>
          <td>0.012708</td>
          <td>29.817744</td>
          <td>1.881510</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.814532</td>
          <td>19.996148</td>
          <td>25.237678</td>
          <td>26.473320</td>
          <td>25.691341</td>
          <td>21.588687</td>
          <td>23.562350</td>
          <td>0.010388</td>
          <td>21.760199</td>
          <td>0.005838</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.070162</td>
          <td>23.630314</td>
          <td>22.349150</td>
          <td>22.017158</td>
          <td>23.181075</td>
          <td>18.299252</td>
          <td>22.593560</td>
          <td>0.006242</td>
          <td>22.042024</td>
          <td>0.006346</td>
          <td>23.519465</td>
          <td>0.015977</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.688312</td>
          <td>22.868756</td>
          <td>22.585485</td>
          <td>19.997192</td>
          <td>27.577378</td>
          <td>23.173233</td>
          <td>21.568836</td>
          <td>0.005207</td>
          <td>23.554633</td>
          <td>0.016449</td>
          <td>21.270247</td>
          <td>0.005356</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.882951</td>
          <td>26.447143</td>
          <td>26.670712</td>
          <td>25.758942</td>
          <td>26.562569</td>
          <td>26.011253</td>
          <td>27.227609</td>
          <td>0.239395</td>
          <td>25.192805</td>
          <td>0.069343</td>
          <td>19.037059</td>
          <td>0.005006</td>
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
          <td>27.488547</td>
          <td>22.631655</td>
          <td>21.165346</td>
          <td>21.120921</td>
          <td>26.887267</td>
          <td>23.109250</td>
          <td>23.044502</td>
          <td>0.007551</td>
          <td>17.481086</td>
          <td>0.005000</td>
          <td>22.148769</td>
          <td>0.006601</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.709431</td>
          <td>21.724414</td>
          <td>21.727664</td>
          <td>27.558420</td>
          <td>19.971493</td>
          <td>22.952474</td>
          <td>26.792506</td>
          <td>0.166069</td>
          <td>27.527742</td>
          <td>0.486370</td>
          <td>23.539203</td>
          <td>0.016240</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.625763</td>
          <td>21.032894</td>
          <td>27.815813</td>
          <td>24.247503</td>
          <td>21.345806</td>
          <td>24.189072</td>
          <td>20.375302</td>
          <td>0.005023</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.059662</td>
          <td>0.005039</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.378059</td>
          <td>26.471287</td>
          <td>23.205571</td>
          <td>20.047233</td>
          <td>25.060985</td>
          <td>24.438920</td>
          <td>21.311192</td>
          <td>0.005130</td>
          <td>24.096702</td>
          <td>0.026187</td>
          <td>25.554057</td>
          <td>0.095433</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.793784</td>
          <td>21.927870</td>
          <td>25.611626</td>
          <td>25.242086</td>
          <td>29.794462</td>
          <td>27.582915</td>
          <td>22.251397</td>
          <td>0.005695</td>
          <td>17.550054</td>
          <td>0.005000</td>
          <td>25.483602</td>
          <td>0.089693</td>
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
          <td>16.957155</td>
          <td>20.449162</td>
          <td>24.490551</td>
          <td>24.326024</td>
          <td>23.636808</td>
          <td>24.224038</td>
          <td>23.671409</td>
          <td>0.115449</td>
          <td>23.240357</td>
          <td>0.066186</td>
          <td>26.320155</td>
          <td>0.842376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.814532</td>
          <td>19.996148</td>
          <td>25.237678</td>
          <td>26.473320</td>
          <td>25.691341</td>
          <td>21.588687</td>
          <td>23.771918</td>
          <td>0.126004</td>
          <td>21.789839</td>
          <td>0.018425</td>
          <td>27.338846</td>
          <td>1.504114</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.070162</td>
          <td>23.630314</td>
          <td>22.349150</td>
          <td>22.017158</td>
          <td>23.181075</td>
          <td>18.299252</td>
          <td>22.637225</td>
          <td>0.046220</td>
          <td>22.052291</td>
          <td>0.023084</td>
          <td>23.549410</td>
          <td>0.095044</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.688312</td>
          <td>22.868756</td>
          <td>22.585485</td>
          <td>19.997192</td>
          <td>27.577378</td>
          <td>23.173233</td>
          <td>21.562170</td>
          <td>0.017999</td>
          <td>23.483694</td>
          <td>0.082121</td>
          <td>21.278045</td>
          <td>0.013151</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.882951</td>
          <td>26.447143</td>
          <td>26.670712</td>
          <td>25.758942</td>
          <td>26.562569</td>
          <td>26.011253</td>
          <td>25.049017</td>
          <td>0.364325</td>
          <td>24.903421</td>
          <td>0.276505</td>
          <td>19.048360</td>
          <td>0.005239</td>
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
          <td>27.488547</td>
          <td>22.631655</td>
          <td>21.165346</td>
          <td>21.120921</td>
          <td>26.887267</td>
          <td>23.109250</td>
          <td>23.046659</td>
          <td>0.066558</td>
          <td>17.471475</td>
          <td>0.005011</td>
          <td>22.161426</td>
          <td>0.027720</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.709431</td>
          <td>21.724414</td>
          <td>21.727664</td>
          <td>27.558420</td>
          <td>19.971493</td>
          <td>22.952474</td>
          <td>24.753901</td>
          <td>0.288058</td>
          <td>26.821317</td>
          <td>1.077008</td>
          <td>23.693726</td>
          <td>0.107874</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.625763</td>
          <td>21.032894</td>
          <td>27.815813</td>
          <td>24.247503</td>
          <td>21.345806</td>
          <td>24.189072</td>
          <td>20.356433</td>
          <td>0.007597</td>
          <td>26.296110</td>
          <td>0.777196</td>
          <td>20.051569</td>
          <td>0.006367</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.378059</td>
          <td>26.471287</td>
          <td>23.205571</td>
          <td>20.047233</td>
          <td>25.060985</td>
          <td>24.438920</td>
          <td>21.313828</td>
          <td>0.014655</td>
          <td>24.010810</td>
          <td>0.130328</td>
          <td>26.451016</td>
          <td>0.914951</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.793784</td>
          <td>21.927870</td>
          <td>25.611626</td>
          <td>25.242086</td>
          <td>29.794462</td>
          <td>27.582915</td>
          <td>22.261799</td>
          <td>0.033092</td>
          <td>17.549514</td>
          <td>0.005013</td>
          <td>25.506577</td>
          <td>0.478776</td>
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


