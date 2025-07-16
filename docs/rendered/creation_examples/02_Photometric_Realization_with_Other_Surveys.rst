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
          <td>23.039399</td>
          <td>22.282844</td>
          <td>29.004554</td>
          <td>22.437742</td>
          <td>26.239733</td>
          <td>28.064591</td>
          <td>22.129600</td>
          <td>23.800804</td>
          <td>22.430989</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.848043</td>
          <td>19.753470</td>
          <td>20.261427</td>
          <td>25.086982</td>
          <td>23.944290</td>
          <td>20.007187</td>
          <td>24.921280</td>
          <td>25.930219</td>
          <td>19.780373</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.677628</td>
          <td>21.751430</td>
          <td>19.580286</td>
          <td>24.559036</td>
          <td>25.858309</td>
          <td>24.452930</td>
          <td>26.352358</td>
          <td>20.876621</td>
          <td>24.710465</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.872586</td>
          <td>18.865556</td>
          <td>21.741943</td>
          <td>29.640059</td>
          <td>21.525880</td>
          <td>21.486907</td>
          <td>23.656106</td>
          <td>20.959283</td>
          <td>21.817248</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.694518</td>
          <td>22.752371</td>
          <td>18.794109</td>
          <td>16.609691</td>
          <td>27.398767</td>
          <td>25.298088</td>
          <td>25.083587</td>
          <td>23.762788</td>
          <td>27.161079</td>
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
          <td>23.281746</td>
          <td>26.785542</td>
          <td>20.855449</td>
          <td>18.987856</td>
          <td>24.074314</td>
          <td>25.741835</td>
          <td>24.772089</td>
          <td>23.994357</td>
          <td>30.481229</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.944976</td>
          <td>26.142015</td>
          <td>21.378418</td>
          <td>30.126461</td>
          <td>18.201323</td>
          <td>26.379353</td>
          <td>24.711936</td>
          <td>19.079708</td>
          <td>24.133916</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.401887</td>
          <td>20.327261</td>
          <td>22.816355</td>
          <td>26.126267</td>
          <td>21.841689</td>
          <td>22.815699</td>
          <td>21.743486</td>
          <td>18.592466</td>
          <td>27.136845</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.519749</td>
          <td>21.497662</td>
          <td>20.209871</td>
          <td>18.440590</td>
          <td>21.473983</td>
          <td>25.419900</td>
          <td>21.436348</td>
          <td>23.993626</td>
          <td>19.966272</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.726930</td>
          <td>21.433906</td>
          <td>25.007403</td>
          <td>18.156475</td>
          <td>17.902434</td>
          <td>19.791072</td>
          <td>21.093126</td>
          <td>23.325591</td>
          <td>24.402100</td>
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
          <td>23.041693</td>
          <td>0.019766</td>
          <td>22.282495</td>
          <td>0.006009</td>
          <td>30.226624</td>
          <td>1.675307</td>
          <td>22.438528</td>
          <td>0.007268</td>
          <td>26.319017</td>
          <td>0.309005</td>
          <td>26.795733</td>
          <td>0.851269</td>
          <td>22.129600</td>
          <td>23.800804</td>
          <td>22.430989</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.849280</td>
          <td>0.005228</td>
          <td>19.752766</td>
          <td>0.005028</td>
          <td>20.261315</td>
          <td>0.005034</td>
          <td>25.133511</td>
          <td>0.059766</td>
          <td>23.935673</td>
          <td>0.039569</td>
          <td>20.015634</td>
          <td>0.005685</td>
          <td>24.921280</td>
          <td>25.930219</td>
          <td>19.780373</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.687858</td>
          <td>0.005698</td>
          <td>21.748946</td>
          <td>0.005444</td>
          <td>19.588936</td>
          <td>0.005014</td>
          <td>24.566643</td>
          <td>0.036146</td>
          <td>25.725726</td>
          <td>0.189551</td>
          <td>24.328931</td>
          <td>0.125941</td>
          <td>26.352358</td>
          <td>20.876621</td>
          <td>24.710465</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.924524</td>
          <td>0.101387</td>
          <td>18.868386</td>
          <td>0.005010</td>
          <td>21.739376</td>
          <td>0.005312</td>
          <td>29.566108</td>
          <td>1.592151</td>
          <td>21.520471</td>
          <td>0.006720</td>
          <td>21.490357</td>
          <td>0.011099</td>
          <td>23.656106</td>
          <td>20.959283</td>
          <td>21.817248</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.926614</td>
          <td>0.238038</td>
          <td>22.763513</td>
          <td>0.007083</td>
          <td>18.793395</td>
          <td>0.005005</td>
          <td>16.598071</td>
          <td>0.005001</td>
          <td>32.349605</td>
          <td>4.836301</td>
          <td>25.692195</td>
          <td>0.389039</td>
          <td>25.083587</td>
          <td>23.762788</td>
          <td>27.161079</td>
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
          <td>23.273208</td>
          <td>0.023991</td>
          <td>26.675582</td>
          <td>0.161003</td>
          <td>20.852463</td>
          <td>0.005078</td>
          <td>18.988636</td>
          <td>0.005013</td>
          <td>24.080794</td>
          <td>0.045002</td>
          <td>26.130901</td>
          <td>0.540650</td>
          <td>24.772089</td>
          <td>23.994357</td>
          <td>30.481229</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.071330</td>
          <td>0.115193</td>
          <td>26.039557</td>
          <td>0.092769</td>
          <td>21.370338</td>
          <td>0.005174</td>
          <td>31.037453</td>
          <td>2.859529</td>
          <td>18.197878</td>
          <td>0.005013</td>
          <td>26.622035</td>
          <td>0.760351</td>
          <td>24.711936</td>
          <td>19.079708</td>
          <td>24.133916</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.386864</td>
          <td>0.011900</td>
          <td>20.331731</td>
          <td>0.005058</td>
          <td>22.809756</td>
          <td>0.006722</td>
          <td>26.372012</td>
          <td>0.176152</td>
          <td>21.832844</td>
          <td>0.007747</td>
          <td>22.810290</td>
          <td>0.032981</td>
          <td>21.743486</td>
          <td>18.592466</td>
          <td>27.136845</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.493607</td>
          <td>0.375159</td>
          <td>21.496685</td>
          <td>0.005302</td>
          <td>20.207407</td>
          <td>0.005031</td>
          <td>18.436485</td>
          <td>0.005007</td>
          <td>21.479412</td>
          <td>0.006615</td>
          <td>26.284542</td>
          <td>0.603507</td>
          <td>21.436348</td>
          <td>23.993626</td>
          <td>19.966272</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708847</td>
          <td>0.007879</td>
          <td>21.427155</td>
          <td>0.005272</td>
          <td>24.960349</td>
          <td>0.031394</td>
          <td>18.155529</td>
          <td>0.005005</td>
          <td>17.899714</td>
          <td>0.005009</td>
          <td>19.789587</td>
          <td>0.005476</td>
          <td>21.093126</td>
          <td>23.325591</td>
          <td>24.402100</td>
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
          <td>23.039399</td>
          <td>22.282844</td>
          <td>29.004554</td>
          <td>22.437742</td>
          <td>26.239733</td>
          <td>28.064591</td>
          <td>22.130379</td>
          <td>0.005563</td>
          <td>23.791117</td>
          <td>0.020085</td>
          <td>22.432168</td>
          <td>0.007503</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.848043</td>
          <td>19.753470</td>
          <td>20.261427</td>
          <td>25.086982</td>
          <td>23.944290</td>
          <td>20.007187</td>
          <td>24.953401</td>
          <td>0.032847</td>
          <td>26.197035</td>
          <td>0.166712</td>
          <td>19.784313</td>
          <td>0.005024</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.677628</td>
          <td>21.751430</td>
          <td>19.580286</td>
          <td>24.559036</td>
          <td>25.858309</td>
          <td>24.452930</td>
          <td>26.287013</td>
          <td>0.107242</td>
          <td>20.878309</td>
          <td>0.005176</td>
          <td>24.727812</td>
          <td>0.045833</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.872586</td>
          <td>18.865556</td>
          <td>21.741943</td>
          <td>29.640059</td>
          <td>21.525880</td>
          <td>21.486907</td>
          <td>23.662189</td>
          <td>0.011162</td>
          <td>20.963202</td>
          <td>0.005205</td>
          <td>21.817096</td>
          <td>0.005924</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.694518</td>
          <td>22.752371</td>
          <td>18.794109</td>
          <td>16.609691</td>
          <td>27.398767</td>
          <td>25.298088</td>
          <td>25.050477</td>
          <td>0.035803</td>
          <td>23.788201</td>
          <td>0.020035</td>
          <td>27.269123</td>
          <td>0.399933</td>
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
          <td>23.281746</td>
          <td>26.785542</td>
          <td>20.855449</td>
          <td>18.987856</td>
          <td>24.074314</td>
          <td>25.741835</td>
          <td>24.807499</td>
          <td>0.028869</td>
          <td>24.037860</td>
          <td>0.024872</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.944976</td>
          <td>26.142015</td>
          <td>21.378418</td>
          <td>30.126461</td>
          <td>18.201323</td>
          <td>26.379353</td>
          <td>24.700187</td>
          <td>0.026268</td>
          <td>19.086828</td>
          <td>0.005007</td>
          <td>24.110112</td>
          <td>0.026497</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.401887</td>
          <td>20.327261</td>
          <td>22.816355</td>
          <td>26.126267</td>
          <td>21.841689</td>
          <td>22.815699</td>
          <td>21.751343</td>
          <td>0.005288</td>
          <td>18.597799</td>
          <td>0.005003</td>
          <td>26.849663</td>
          <td>0.287072</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.519749</td>
          <td>21.497662</td>
          <td>20.209871</td>
          <td>18.440590</td>
          <td>21.473983</td>
          <td>25.419900</td>
          <td>21.442616</td>
          <td>0.005165</td>
          <td>23.981700</td>
          <td>0.023682</td>
          <td>19.962691</td>
          <td>0.005033</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.726930</td>
          <td>21.433906</td>
          <td>25.007403</td>
          <td>18.156475</td>
          <td>17.902434</td>
          <td>19.791072</td>
          <td>21.096801</td>
          <td>0.005088</td>
          <td>23.316405</td>
          <td>0.013555</td>
          <td>24.425127</td>
          <td>0.035006</td>
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
          <td>23.039399</td>
          <td>22.282844</td>
          <td>29.004554</td>
          <td>22.437742</td>
          <td>26.239733</td>
          <td>28.064591</td>
          <td>22.140205</td>
          <td>0.029715</td>
          <td>23.764334</td>
          <td>0.105133</td>
          <td>22.413440</td>
          <td>0.034644</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.848043</td>
          <td>19.753470</td>
          <td>20.261427</td>
          <td>25.086982</td>
          <td>23.944290</td>
          <td>20.007187</td>
          <td>24.908543</td>
          <td>0.326105</td>
          <td>25.011680</td>
          <td>0.301797</td>
          <td>19.770821</td>
          <td>0.005854</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.677628</td>
          <td>21.751430</td>
          <td>19.580286</td>
          <td>24.559036</td>
          <td>25.858309</td>
          <td>24.452930</td>
          <td>25.482722</td>
          <td>0.506543</td>
          <td>20.879536</td>
          <td>0.009179</td>
          <td>25.492109</td>
          <td>0.473639</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.872586</td>
          <td>18.865556</td>
          <td>21.741943</td>
          <td>29.640059</td>
          <td>21.525880</td>
          <td>21.486907</td>
          <td>24.015295</td>
          <td>0.155455</td>
          <td>20.968042</td>
          <td>0.009733</td>
          <td>21.824209</td>
          <td>0.020664</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.694518</td>
          <td>22.752371</td>
          <td>18.794109</td>
          <td>16.609691</td>
          <td>27.398767</td>
          <td>25.298088</td>
          <td>24.869947</td>
          <td>0.316224</td>
          <td>23.879398</td>
          <td>0.116256</td>
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
          <td>23.281746</td>
          <td>26.785542</td>
          <td>20.855449</td>
          <td>18.987856</td>
          <td>24.074314</td>
          <td>25.741835</td>
          <td>24.433723</td>
          <td>0.221457</td>
          <td>24.191141</td>
          <td>0.152267</td>
          <td>26.588772</td>
          <td>0.995524</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.944976</td>
          <td>26.142015</td>
          <td>21.378418</td>
          <td>30.126461</td>
          <td>18.201323</td>
          <td>26.379353</td>
          <td>24.795376</td>
          <td>0.297864</td>
          <td>19.083302</td>
          <td>0.005213</td>
          <td>24.230407</td>
          <td>0.171522</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.401887</td>
          <td>20.327261</td>
          <td>22.816355</td>
          <td>26.126267</td>
          <td>21.841689</td>
          <td>22.815699</td>
          <td>21.694129</td>
          <td>0.020137</td>
          <td>18.596068</td>
          <td>0.005088</td>
          <td>26.251908</td>
          <td>0.806078</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.519749</td>
          <td>21.497662</td>
          <td>20.209871</td>
          <td>18.440590</td>
          <td>21.473983</td>
          <td>25.419900</td>
          <td>21.457613</td>
          <td>0.016490</td>
          <td>23.902085</td>
          <td>0.118578</td>
          <td>19.957848</td>
          <td>0.006170</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.726930</td>
          <td>21.433906</td>
          <td>25.007403</td>
          <td>18.156475</td>
          <td>17.902434</td>
          <td>19.791072</td>
          <td>21.084629</td>
          <td>0.012234</td>
          <td>23.258162</td>
          <td>0.067242</td>
          <td>24.863516</td>
          <td>0.290306</td>
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


