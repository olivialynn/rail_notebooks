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
          <td>23.624677</td>
          <td>16.133952</td>
          <td>22.652504</td>
          <td>24.079185</td>
          <td>22.965758</td>
          <td>22.659309</td>
          <td>15.099145</td>
          <td>22.254575</td>
          <td>25.979261</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.079679</td>
          <td>21.580561</td>
          <td>21.059929</td>
          <td>29.685830</td>
          <td>30.403290</td>
          <td>27.279870</td>
          <td>23.319196</td>
          <td>23.944423</td>
          <td>23.654438</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.911607</td>
          <td>23.145094</td>
          <td>25.195025</td>
          <td>25.212203</td>
          <td>27.145500</td>
          <td>25.794629</td>
          <td>19.763713</td>
          <td>22.479966</td>
          <td>21.573310</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.387528</td>
          <td>20.053158</td>
          <td>23.885376</td>
          <td>27.987149</td>
          <td>22.787954</td>
          <td>25.958139</td>
          <td>21.055719</td>
          <td>25.512467</td>
          <td>25.812031</td>
        </tr>
        <tr>
          <th>4</th>
          <td>31.370075</td>
          <td>24.401832</td>
          <td>22.763513</td>
          <td>27.063482</td>
          <td>29.159965</td>
          <td>21.203689</td>
          <td>24.764904</td>
          <td>25.356566</td>
          <td>22.181625</td>
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
          <td>18.182613</td>
          <td>25.171336</td>
          <td>24.441885</td>
          <td>26.987743</td>
          <td>19.361989</td>
          <td>24.448618</td>
          <td>26.495940</td>
          <td>21.734331</td>
          <td>19.984780</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.468202</td>
          <td>21.062277</td>
          <td>22.733572</td>
          <td>24.432678</td>
          <td>20.859443</td>
          <td>18.691585</td>
          <td>20.969672</td>
          <td>25.702408</td>
          <td>21.973323</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.086488</td>
          <td>26.436750</td>
          <td>21.885758</td>
          <td>22.055257</td>
          <td>23.717246</td>
          <td>21.558418</td>
          <td>20.630816</td>
          <td>24.824660</td>
          <td>18.659387</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.444458</td>
          <td>23.717199</td>
          <td>19.885105</td>
          <td>21.503516</td>
          <td>26.677117</td>
          <td>17.981282</td>
          <td>21.908199</td>
          <td>24.249844</td>
          <td>25.222477</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.438562</td>
          <td>23.578586</td>
          <td>21.853404</td>
          <td>18.461683</td>
          <td>19.959791</td>
          <td>23.492988</td>
          <td>21.062121</td>
          <td>19.784354</td>
          <td>28.315064</td>
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
          <td>23.638818</td>
          <td>0.032840</td>
          <td>16.137582</td>
          <td>0.005001</td>
          <td>22.655169</td>
          <td>0.006355</td>
          <td>24.090158</td>
          <td>0.023812</td>
          <td>22.951150</td>
          <td>0.016895</td>
          <td>22.681414</td>
          <td>0.029449</td>
          <td>15.099145</td>
          <td>22.254575</td>
          <td>25.979261</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.052309</td>
          <td>0.047145</td>
          <td>21.579272</td>
          <td>0.005342</td>
          <td>21.057941</td>
          <td>0.005107</td>
          <td>27.367850</td>
          <td>0.396177</td>
          <td>27.440517</td>
          <td>0.710789</td>
          <td>27.220540</td>
          <td>1.102351</td>
          <td>23.319196</td>
          <td>23.944423</td>
          <td>23.654438</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.904368</td>
          <td>0.005944</td>
          <td>23.143186</td>
          <td>0.008588</td>
          <td>25.143235</td>
          <td>0.036892</td>
          <td>25.189649</td>
          <td>0.062817</td>
          <td>27.460113</td>
          <td>0.720245</td>
          <td>24.903652</td>
          <td>0.205690</td>
          <td>19.763713</td>
          <td>22.479966</td>
          <td>21.573310</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.381630</td>
          <td>0.150507</td>
          <td>20.037620</td>
          <td>0.005040</td>
          <td>23.899382</td>
          <td>0.012895</td>
          <td>27.226559</td>
          <td>0.354921</td>
          <td>22.762936</td>
          <td>0.014506</td>
          <td>25.622943</td>
          <td>0.368657</td>
          <td>21.055719</td>
          <td>25.512467</td>
          <td>25.812031</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.213038</td>
          <td>0.638197</td>
          <td>24.419448</td>
          <td>0.022338</td>
          <td>22.767604</td>
          <td>0.006614</td>
          <td>27.142838</td>
          <td>0.332232</td>
          <td>27.194415</td>
          <td>0.599522</td>
          <td>21.200097</td>
          <td>0.009130</td>
          <td>24.764904</td>
          <td>25.356566</td>
          <td>22.181625</td>
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
          <td>18.180168</td>
          <td>0.005034</td>
          <td>25.195122</td>
          <td>0.043992</td>
          <td>24.441875</td>
          <td>0.020034</td>
          <td>27.342719</td>
          <td>0.388559</td>
          <td>19.360999</td>
          <td>0.005059</td>
          <td>24.204248</td>
          <td>0.113004</td>
          <td>26.495940</td>
          <td>21.734331</td>
          <td>19.984780</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.450959</td>
          <td>0.012460</td>
          <td>21.058075</td>
          <td>0.005158</td>
          <td>22.734784</td>
          <td>0.006534</td>
          <td>24.434735</td>
          <td>0.032174</td>
          <td>20.854392</td>
          <td>0.005601</td>
          <td>18.688362</td>
          <td>0.005085</td>
          <td>20.969672</td>
          <td>25.702408</td>
          <td>21.973323</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.220787</td>
          <td>1.201079</td>
          <td>26.732267</td>
          <td>0.168972</td>
          <td>21.882061</td>
          <td>0.005393</td>
          <td>22.060386</td>
          <td>0.006268</td>
          <td>23.728529</td>
          <td>0.032950</td>
          <td>21.576308</td>
          <td>0.011815</td>
          <td>20.630816</td>
          <td>24.824660</td>
          <td>18.659387</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.447567</td>
          <td>0.027839</td>
          <td>23.728344</td>
          <td>0.012743</td>
          <td>19.879392</td>
          <td>0.005020</td>
          <td>21.497336</td>
          <td>0.005515</td>
          <td>26.545319</td>
          <td>0.369558</td>
          <td>17.977867</td>
          <td>0.005031</td>
          <td>21.908199</td>
          <td>24.249844</td>
          <td>25.222477</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.437055</td>
          <td>0.006986</td>
          <td>23.542404</td>
          <td>0.011121</td>
          <td>21.843972</td>
          <td>0.005370</td>
          <td>18.468468</td>
          <td>0.005007</td>
          <td>19.968505</td>
          <td>0.005147</td>
          <td>23.447667</td>
          <td>0.058016</td>
          <td>21.062121</td>
          <td>19.784354</td>
          <td>28.315064</td>
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
          <td>23.624677</td>
          <td>16.133952</td>
          <td>22.652504</td>
          <td>24.079185</td>
          <td>22.965758</td>
          <td>22.659309</td>
          <td>15.091343</td>
          <td>0.005000</td>
          <td>22.261642</td>
          <td>0.006919</td>
          <td>25.897777</td>
          <td>0.128863</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.079679</td>
          <td>21.580561</td>
          <td>21.059929</td>
          <td>29.685830</td>
          <td>30.403290</td>
          <td>27.279870</td>
          <td>23.323736</td>
          <td>0.008859</td>
          <td>23.907676</td>
          <td>0.022207</td>
          <td>23.663005</td>
          <td>0.018011</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.911607</td>
          <td>23.145094</td>
          <td>25.195025</td>
          <td>25.212203</td>
          <td>27.145500</td>
          <td>25.794629</td>
          <td>19.761251</td>
          <td>0.005008</td>
          <td>22.473490</td>
          <td>0.007666</td>
          <td>21.574726</td>
          <td>0.005609</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.387528</td>
          <td>20.053158</td>
          <td>23.885376</td>
          <td>27.987149</td>
          <td>22.787954</td>
          <td>25.958139</td>
          <td>21.054199</td>
          <td>0.005081</td>
          <td>25.619034</td>
          <td>0.101039</td>
          <td>25.768981</td>
          <td>0.115205</td>
        </tr>
        <tr>
          <th>4</th>
          <td>31.370075</td>
          <td>24.401832</td>
          <td>22.763513</td>
          <td>27.063482</td>
          <td>29.159965</td>
          <td>21.203689</td>
          <td>24.791029</td>
          <td>0.028453</td>
          <td>25.360174</td>
          <td>0.080431</td>
          <td>22.172646</td>
          <td>0.006664</td>
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
          <td>18.182613</td>
          <td>25.171336</td>
          <td>24.441885</td>
          <td>26.987743</td>
          <td>19.361989</td>
          <td>24.448618</td>
          <td>26.305421</td>
          <td>0.108983</td>
          <td>21.743116</td>
          <td>0.005814</td>
          <td>19.983570</td>
          <td>0.005034</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.468202</td>
          <td>21.062277</td>
          <td>22.733572</td>
          <td>24.432678</td>
          <td>20.859443</td>
          <td>18.691585</td>
          <td>20.970595</td>
          <td>0.005070</td>
          <td>25.689035</td>
          <td>0.107432</td>
          <td>21.968607</td>
          <td>0.006192</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.086488</td>
          <td>26.436750</td>
          <td>21.885758</td>
          <td>22.055257</td>
          <td>23.717246</td>
          <td>21.558418</td>
          <td>20.630777</td>
          <td>0.005037</td>
          <td>24.840537</td>
          <td>0.050679</td>
          <td>18.654183</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.444458</td>
          <td>23.717199</td>
          <td>19.885105</td>
          <td>21.503516</td>
          <td>26.677117</td>
          <td>17.981282</td>
          <td>21.917169</td>
          <td>0.005387</td>
          <td>24.246694</td>
          <td>0.029886</td>
          <td>25.200116</td>
          <td>0.069794</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.438562</td>
          <td>23.578586</td>
          <td>21.853404</td>
          <td>18.461683</td>
          <td>19.959791</td>
          <td>23.492988</td>
          <td>21.059749</td>
          <td>0.005082</td>
          <td>19.777745</td>
          <td>0.005024</td>
          <td>28.338265</td>
          <td>0.852187</td>
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
          <td>23.624677</td>
          <td>16.133952</td>
          <td>22.652504</td>
          <td>24.079185</td>
          <td>22.965758</td>
          <td>22.659309</td>
          <td>15.108005</td>
          <td>0.005000</td>
          <td>22.214221</td>
          <td>0.026593</td>
          <td>25.487408</td>
          <td>0.471980</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.079679</td>
          <td>21.580561</td>
          <td>21.059929</td>
          <td>29.685830</td>
          <td>30.403290</td>
          <td>27.279870</td>
          <td>23.271220</td>
          <td>0.081220</td>
          <td>23.895248</td>
          <td>0.117874</td>
          <td>23.793787</td>
          <td>0.117724</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.911607</td>
          <td>23.145094</td>
          <td>25.195025</td>
          <td>25.212203</td>
          <td>27.145500</td>
          <td>25.794629</td>
          <td>19.768169</td>
          <td>0.006007</td>
          <td>22.469955</td>
          <td>0.033332</td>
          <td>21.617384</td>
          <td>0.017333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.387528</td>
          <td>20.053158</td>
          <td>23.885376</td>
          <td>27.987149</td>
          <td>22.787954</td>
          <td>25.958139</td>
          <td>21.060025</td>
          <td>0.012007</td>
          <td>24.837674</td>
          <td>0.262071</td>
          <td>25.425337</td>
          <td>0.450501</td>
        </tr>
        <tr>
          <th>4</th>
          <td>31.370075</td>
          <td>24.401832</td>
          <td>22.763513</td>
          <td>27.063482</td>
          <td>29.159965</td>
          <td>21.203689</td>
          <td>25.226027</td>
          <td>0.417785</td>
          <td>25.293882</td>
          <td>0.377303</td>
          <td>22.203691</td>
          <td>0.028772</td>
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
          <td>18.182613</td>
          <td>25.171336</td>
          <td>24.441885</td>
          <td>26.987743</td>
          <td>19.361989</td>
          <td>24.448618</td>
          <td>25.733961</td>
          <td>0.607125</td>
          <td>21.728527</td>
          <td>0.017496</td>
          <td>19.978439</td>
          <td>0.006211</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.468202</td>
          <td>21.062277</td>
          <td>22.733572</td>
          <td>24.432678</td>
          <td>20.859443</td>
          <td>18.691585</td>
          <td>20.979868</td>
          <td>0.011308</td>
          <td>26.232874</td>
          <td>0.745330</td>
          <td>21.983653</td>
          <td>0.023722</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.086488</td>
          <td>26.436750</td>
          <td>21.885758</td>
          <td>22.055257</td>
          <td>23.717246</td>
          <td>21.558418</td>
          <td>20.637711</td>
          <td>0.008937</td>
          <td>25.259514</td>
          <td>0.367328</td>
          <td>18.660643</td>
          <td>0.005119</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.444458</td>
          <td>23.717199</td>
          <td>19.885105</td>
          <td>21.503516</td>
          <td>26.677117</td>
          <td>17.981282</td>
          <td>21.915316</td>
          <td>0.024387</td>
          <td>24.021001</td>
          <td>0.131484</td>
          <td>25.068638</td>
          <td>0.342016</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.438562</td>
          <td>23.578586</td>
          <td>21.853404</td>
          <td>18.461683</td>
          <td>19.959791</td>
          <td>23.492988</td>
          <td>21.064762</td>
          <td>0.012050</td>
          <td>19.771538</td>
          <td>0.005720</td>
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


