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
          <td>22.652804</td>
          <td>15.978936</td>
          <td>24.897587</td>
          <td>22.527881</td>
          <td>23.459124</td>
          <td>21.064835</td>
          <td>24.855869</td>
          <td>24.665202</td>
          <td>27.847642</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.699986</td>
          <td>26.001909</td>
          <td>26.711880</td>
          <td>22.367768</td>
          <td>21.979290</td>
          <td>23.256323</td>
          <td>18.820745</td>
          <td>26.224916</td>
          <td>24.835341</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.529983</td>
          <td>24.455335</td>
          <td>22.801492</td>
          <td>18.404744</td>
          <td>19.271438</td>
          <td>22.834151</td>
          <td>23.037598</td>
          <td>28.287663</td>
          <td>25.568117</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.922707</td>
          <td>24.584409</td>
          <td>24.998215</td>
          <td>29.882955</td>
          <td>22.340000</td>
          <td>20.397293</td>
          <td>28.220653</td>
          <td>27.960606</td>
          <td>25.984797</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.271115</td>
          <td>21.602067</td>
          <td>25.546484</td>
          <td>25.378740</td>
          <td>24.397972</td>
          <td>23.730590</td>
          <td>23.431200</td>
          <td>22.278356</td>
          <td>28.106869</td>
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
          <td>23.152323</td>
          <td>16.465165</td>
          <td>23.680751</td>
          <td>22.480280</td>
          <td>18.793522</td>
          <td>22.404892</td>
          <td>17.748096</td>
          <td>19.447260</td>
          <td>26.945523</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.677069</td>
          <td>24.337511</td>
          <td>20.073274</td>
          <td>23.076856</td>
          <td>21.845344</td>
          <td>23.824616</td>
          <td>24.356989</td>
          <td>17.369307</td>
          <td>24.379072</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.082123</td>
          <td>23.746632</td>
          <td>25.812101</td>
          <td>20.360859</td>
          <td>15.961755</td>
          <td>25.736891</td>
          <td>21.242894</td>
          <td>22.356667</td>
          <td>21.645796</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.498114</td>
          <td>18.194508</td>
          <td>27.066411</td>
          <td>25.108945</td>
          <td>25.957710</td>
          <td>19.152111</td>
          <td>21.881737</td>
          <td>22.804034</td>
          <td>24.019944</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.139776</td>
          <td>23.221664</td>
          <td>24.307104</td>
          <td>22.662773</td>
          <td>24.567384</td>
          <td>22.279528</td>
          <td>24.435790</td>
          <td>22.140182</td>
          <td>27.134541</td>
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
          <td>22.667424</td>
          <td>0.014652</td>
          <td>15.985792</td>
          <td>0.005001</td>
          <td>24.902147</td>
          <td>0.029829</td>
          <td>22.527100</td>
          <td>0.007586</td>
          <td>23.431952</td>
          <td>0.025415</td>
          <td>21.066347</td>
          <td>0.008424</td>
          <td>24.855869</td>
          <td>24.665202</td>
          <td>27.847642</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.698927</td>
          <td>0.007841</td>
          <td>26.271065</td>
          <td>0.113577</td>
          <td>26.774782</td>
          <td>0.154896</td>
          <td>22.369018</td>
          <td>0.007043</td>
          <td>21.988283</td>
          <td>0.008436</td>
          <td>23.307261</td>
          <td>0.051218</td>
          <td>18.820745</td>
          <td>26.224916</td>
          <td>24.835341</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.516052</td>
          <td>0.013068</td>
          <td>24.472111</td>
          <td>0.023368</td>
          <td>22.801397</td>
          <td>0.006700</td>
          <td>18.399072</td>
          <td>0.005006</td>
          <td>19.270937</td>
          <td>0.005052</td>
          <td>22.822745</td>
          <td>0.033345</td>
          <td>23.037598</td>
          <td>28.287663</td>
          <td>25.568117</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.919658</td>
          <td>0.005250</td>
          <td>24.610458</td>
          <td>0.026329</td>
          <td>24.968544</td>
          <td>0.031622</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.341691</td>
          <td>0.010577</td>
          <td>20.390452</td>
          <td>0.006242</td>
          <td>28.220653</td>
          <td>27.960606</td>
          <td>25.984797</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.272079</td>
          <td>0.005113</td>
          <td>21.602850</td>
          <td>0.005355</td>
          <td>25.638942</td>
          <td>0.057268</td>
          <td>25.386883</td>
          <td>0.074805</td>
          <td>24.345579</td>
          <td>0.056926</td>
          <td>23.656239</td>
          <td>0.069796</td>
          <td>23.431200</td>
          <td>22.278356</td>
          <td>28.106869</td>
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
          <td>23.147463</td>
          <td>0.021581</td>
          <td>16.463347</td>
          <td>0.005001</td>
          <td>23.692101</td>
          <td>0.011058</td>
          <td>22.474332</td>
          <td>0.007392</td>
          <td>18.791167</td>
          <td>0.005027</td>
          <td>22.410490</td>
          <td>0.023266</td>
          <td>17.748096</td>
          <td>19.447260</td>
          <td>26.945523</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.669327</td>
          <td>0.007729</td>
          <td>24.325911</td>
          <td>0.020631</td>
          <td>20.073564</td>
          <td>0.005026</td>
          <td>23.081251</td>
          <td>0.010616</td>
          <td>21.838827</td>
          <td>0.007771</td>
          <td>23.763273</td>
          <td>0.076725</td>
          <td>24.356989</td>
          <td>17.369307</td>
          <td>24.379072</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.224634</td>
          <td>0.643362</td>
          <td>23.746340</td>
          <td>0.012917</td>
          <td>25.849696</td>
          <td>0.069035</td>
          <td>20.367022</td>
          <td>0.005086</td>
          <td>15.959829</td>
          <td>0.005001</td>
          <td>25.745560</td>
          <td>0.405379</td>
          <td>21.242894</td>
          <td>22.356667</td>
          <td>21.645796</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.264275</td>
          <td>0.661250</td>
          <td>18.190699</td>
          <td>0.005005</td>
          <td>27.121798</td>
          <td>0.207838</td>
          <td>25.082262</td>
          <td>0.057108</td>
          <td>26.025413</td>
          <td>0.243395</td>
          <td>19.152659</td>
          <td>0.005173</td>
          <td>21.881737</td>
          <td>22.804034</td>
          <td>24.019944</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.119596</td>
          <td>0.021085</td>
          <td>23.236281</td>
          <td>0.009079</td>
          <td>24.314346</td>
          <td>0.017997</td>
          <td>22.676412</td>
          <td>0.008214</td>
          <td>24.517773</td>
          <td>0.066318</td>
          <td>22.315292</td>
          <td>0.021440</td>
          <td>24.435790</td>
          <td>22.140182</td>
          <td>27.134541</td>
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
          <td>22.652804</td>
          <td>15.978936</td>
          <td>24.897587</td>
          <td>22.527881</td>
          <td>23.459124</td>
          <td>21.064835</td>
          <td>24.811404</td>
          <td>0.028969</td>
          <td>24.647655</td>
          <td>0.042673</td>
          <td>30.832418</td>
          <td>2.772901</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.699986</td>
          <td>26.001909</td>
          <td>26.711880</td>
          <td>22.367768</td>
          <td>21.979290</td>
          <td>23.256323</td>
          <td>18.818203</td>
          <td>0.005001</td>
          <td>26.123346</td>
          <td>0.156532</td>
          <td>24.785920</td>
          <td>0.048271</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.529983</td>
          <td>24.455335</td>
          <td>22.801492</td>
          <td>18.404744</td>
          <td>19.271438</td>
          <td>22.834151</td>
          <td>23.045159</td>
          <td>0.007553</td>
          <td>28.449471</td>
          <td>0.914071</td>
          <td>25.617684</td>
          <td>0.100920</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.922707</td>
          <td>24.584409</td>
          <td>24.998215</td>
          <td>29.882955</td>
          <td>22.340000</td>
          <td>20.397293</td>
          <td>28.336171</td>
          <td>0.566314</td>
          <td>27.774088</td>
          <td>0.581884</td>
          <td>26.066470</td>
          <td>0.149073</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.271115</td>
          <td>21.602067</td>
          <td>25.546484</td>
          <td>25.378740</td>
          <td>24.397972</td>
          <td>23.730590</td>
          <td>23.418738</td>
          <td>0.009417</td>
          <td>22.288425</td>
          <td>0.007002</td>
          <td>28.606214</td>
          <td>1.006028</td>
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
          <td>23.152323</td>
          <td>16.465165</td>
          <td>23.680751</td>
          <td>22.480280</td>
          <td>18.793522</td>
          <td>22.404892</td>
          <td>17.754099</td>
          <td>0.005000</td>
          <td>19.441893</td>
          <td>0.005013</td>
          <td>26.628085</td>
          <td>0.239489</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.677069</td>
          <td>24.337511</td>
          <td>20.073274</td>
          <td>23.076856</td>
          <td>21.845344</td>
          <td>23.824616</td>
          <td>24.357227</td>
          <td>0.019511</td>
          <td>17.366040</td>
          <td>0.005000</td>
          <td>24.397998</td>
          <td>0.034172</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.082123</td>
          <td>23.746632</td>
          <td>25.812101</td>
          <td>20.360859</td>
          <td>15.961755</td>
          <td>25.736891</td>
          <td>21.239083</td>
          <td>0.005114</td>
          <td>22.366242</td>
          <td>0.007261</td>
          <td>21.650616</td>
          <td>0.005694</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.498114</td>
          <td>18.194508</td>
          <td>27.066411</td>
          <td>25.108945</td>
          <td>25.957710</td>
          <td>19.152111</td>
          <td>21.898044</td>
          <td>0.005374</td>
          <td>22.813783</td>
          <td>0.009387</td>
          <td>23.980259</td>
          <td>0.023652</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.139776</td>
          <td>23.221664</td>
          <td>24.307104</td>
          <td>22.662773</td>
          <td>24.567384</td>
          <td>22.279528</td>
          <td>24.415453</td>
          <td>0.020509</td>
          <td>22.142512</td>
          <td>0.006585</td>
          <td>27.198836</td>
          <td>0.378759</td>
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
          <td>22.652804</td>
          <td>15.978936</td>
          <td>24.897587</td>
          <td>22.527881</td>
          <td>23.459124</td>
          <td>21.064835</td>
          <td>24.478529</td>
          <td>0.229861</td>
          <td>24.546493</td>
          <td>0.205887</td>
          <td>26.509411</td>
          <td>0.948588</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.699986</td>
          <td>26.001909</td>
          <td>26.711880</td>
          <td>22.367768</td>
          <td>21.979290</td>
          <td>23.256323</td>
          <td>18.819957</td>
          <td>0.005190</td>
          <td>25.533272</td>
          <td>0.453202</td>
          <td>24.623653</td>
          <td>0.238614</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.529983</td>
          <td>24.455335</td>
          <td>22.801492</td>
          <td>18.404744</td>
          <td>19.271438</td>
          <td>22.834151</td>
          <td>23.010779</td>
          <td>0.064469</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.695660</td>
          <td>0.550032</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.922707</td>
          <td>24.584409</td>
          <td>24.998215</td>
          <td>29.882955</td>
          <td>22.340000</td>
          <td>20.397293</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.996879</td>
          <td>0.323091</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.271115</td>
          <td>21.602067</td>
          <td>25.546484</td>
          <td>25.378740</td>
          <td>24.397972</td>
          <td>23.730590</td>
          <td>23.283783</td>
          <td>0.082128</td>
          <td>22.231047</td>
          <td>0.026989</td>
          <td>25.259074</td>
          <td>0.396847</td>
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
          <td>23.152323</td>
          <td>16.465165</td>
          <td>23.680751</td>
          <td>22.480280</td>
          <td>18.793522</td>
          <td>22.404892</td>
          <td>17.751960</td>
          <td>0.005027</td>
          <td>19.455209</td>
          <td>0.005414</td>
          <td>26.528010</td>
          <td>0.959462</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.677069</td>
          <td>24.337511</td>
          <td>20.073274</td>
          <td>23.076856</td>
          <td>21.845344</td>
          <td>23.824616</td>
          <td>24.321151</td>
          <td>0.201554</td>
          <td>17.365475</td>
          <td>0.005009</td>
          <td>24.410580</td>
          <td>0.199771</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.082123</td>
          <td>23.746632</td>
          <td>25.812101</td>
          <td>20.360859</td>
          <td>15.961755</td>
          <td>25.736891</td>
          <td>21.224384</td>
          <td>0.013641</td>
          <td>22.358068</td>
          <td>0.030188</td>
          <td>21.655768</td>
          <td>0.017902</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.498114</td>
          <td>18.194508</td>
          <td>27.066411</td>
          <td>25.108945</td>
          <td>25.957710</td>
          <td>19.152111</td>
          <td>21.903701</td>
          <td>0.024141</td>
          <td>22.918250</td>
          <td>0.049682</td>
          <td>24.025798</td>
          <td>0.143945</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.139776</td>
          <td>23.221664</td>
          <td>24.307104</td>
          <td>22.662773</td>
          <td>24.567384</td>
          <td>22.279528</td>
          <td>24.866375</td>
          <td>0.315323</td>
          <td>22.124471</td>
          <td>0.024582</td>
          <td>inf</td>
          <td>inf</td>
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


