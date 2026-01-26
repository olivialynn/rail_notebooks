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
          <td>28.139771</td>
          <td>32.148375</td>
          <td>22.478837</td>
          <td>21.339589</td>
          <td>21.757575</td>
          <td>20.983276</td>
          <td>21.515205</td>
          <td>24.247398</td>
          <td>24.135098</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.591746</td>
          <td>26.050021</td>
          <td>25.999354</td>
          <td>26.912770</td>
          <td>25.124822</td>
          <td>23.333065</td>
          <td>23.591986</td>
          <td>26.355100</td>
          <td>14.793996</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.288676</td>
          <td>21.457820</td>
          <td>18.071596</td>
          <td>26.710421</td>
          <td>16.250865</td>
          <td>19.506625</td>
          <td>20.028471</td>
          <td>25.039758</td>
          <td>21.091267</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.825414</td>
          <td>19.666019</td>
          <td>16.687810</td>
          <td>18.238687</td>
          <td>25.632725</td>
          <td>24.116998</td>
          <td>19.880068</td>
          <td>25.303514</td>
          <td>22.899253</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.703592</td>
          <td>20.461177</td>
          <td>20.639646</td>
          <td>28.776560</td>
          <td>24.201164</td>
          <td>22.497836</td>
          <td>16.189726</td>
          <td>26.447051</td>
          <td>21.588842</td>
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
          <td>17.885285</td>
          <td>17.747373</td>
          <td>25.574892</td>
          <td>24.382829</td>
          <td>20.603121</td>
          <td>18.687794</td>
          <td>25.982560</td>
          <td>22.695803</td>
          <td>18.499480</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.764712</td>
          <td>21.069966</td>
          <td>21.094874</td>
          <td>23.901255</td>
          <td>22.694941</td>
          <td>22.659367</td>
          <td>19.194704</td>
          <td>23.071878</td>
          <td>21.530608</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.849328</td>
          <td>24.719496</td>
          <td>20.261710</td>
          <td>25.718822</td>
          <td>27.139225</td>
          <td>20.621565</td>
          <td>22.465998</td>
          <td>17.324097</td>
          <td>21.931418</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.154748</td>
          <td>19.158508</td>
          <td>20.580623</td>
          <td>22.368237</td>
          <td>22.781533</td>
          <td>21.594454</td>
          <td>26.535690</td>
          <td>26.782900</td>
          <td>23.240670</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.250507</td>
          <td>20.274016</td>
          <td>19.602615</td>
          <td>18.257034</td>
          <td>22.885145</td>
          <td>27.695603</td>
          <td>20.577613</td>
          <td>19.747314</td>
          <td>24.532159</td>
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
          <td>27.054412</td>
          <td>0.570598</td>
          <td>30.294491</td>
          <td>1.844927</td>
          <td>22.478400</td>
          <td>0.006026</td>
          <td>21.331437</td>
          <td>0.005394</td>
          <td>21.758643</td>
          <td>0.007463</td>
          <td>20.982870</td>
          <td>0.008039</td>
          <td>21.515205</td>
          <td>24.247398</td>
          <td>24.135098</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.534287</td>
          <td>0.792678</td>
          <td>25.982511</td>
          <td>0.088236</td>
          <td>25.992191</td>
          <td>0.078305</td>
          <td>27.306675</td>
          <td>0.377847</td>
          <td>25.189465</td>
          <td>0.119681</td>
          <td>23.360762</td>
          <td>0.053709</td>
          <td>23.591986</td>
          <td>26.355100</td>
          <td>14.793996</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.016314</td>
          <td>0.256239</td>
          <td>21.454469</td>
          <td>0.005284</td>
          <td>18.077046</td>
          <td>0.005003</td>
          <td>26.681083</td>
          <td>0.228338</td>
          <td>16.256875</td>
          <td>0.005002</td>
          <td>19.504875</td>
          <td>0.005302</td>
          <td>20.028471</td>
          <td>25.039758</td>
          <td>21.091267</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.824829</td>
          <td>0.005221</td>
          <td>19.662262</td>
          <td>0.005025</td>
          <td>16.688675</td>
          <td>0.005001</td>
          <td>18.235025</td>
          <td>0.005005</td>
          <td>25.881616</td>
          <td>0.216037</td>
          <td>24.464088</td>
          <td>0.141545</td>
          <td>19.880068</td>
          <td>25.303514</td>
          <td>22.899253</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.835225</td>
          <td>1.647901</td>
          <td>20.456713</td>
          <td>0.005068</td>
          <td>20.639484</td>
          <td>0.005057</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.219003</td>
          <td>0.050876</td>
          <td>22.477635</td>
          <td>0.024655</td>
          <td>16.189726</td>
          <td>26.447051</td>
          <td>21.588842</td>
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
          <td>17.889155</td>
          <td>0.005025</td>
          <td>17.744729</td>
          <td>0.005003</td>
          <td>25.578161</td>
          <td>0.054260</td>
          <td>24.325121</td>
          <td>0.029219</td>
          <td>20.600270</td>
          <td>0.005399</td>
          <td>18.685399</td>
          <td>0.005085</td>
          <td>25.982560</td>
          <td>22.695803</td>
          <td>18.499480</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.753534</td>
          <td>0.015667</td>
          <td>21.067489</td>
          <td>0.005160</td>
          <td>21.098603</td>
          <td>0.005114</td>
          <td>23.895670</td>
          <td>0.020157</td>
          <td>22.684141</td>
          <td>0.013633</td>
          <td>22.620804</td>
          <td>0.027927</td>
          <td>19.194704</td>
          <td>23.071878</td>
          <td>21.530608</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.847244</td>
          <td>0.005227</td>
          <td>24.722287</td>
          <td>0.029019</td>
          <td>20.258250</td>
          <td>0.005033</td>
          <td>25.880117</td>
          <td>0.115356</td>
          <td>27.556813</td>
          <td>0.768199</td>
          <td>20.619082</td>
          <td>0.006769</td>
          <td>22.465998</td>
          <td>17.324097</td>
          <td>21.931418</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.151639</td>
          <td>0.005098</td>
          <td>19.160343</td>
          <td>0.005014</td>
          <td>20.578147</td>
          <td>0.005052</td>
          <td>22.367839</td>
          <td>0.007039</td>
          <td>22.776628</td>
          <td>0.014665</td>
          <td>21.624168</td>
          <td>0.012243</td>
          <td>26.535690</td>
          <td>26.782900</td>
          <td>23.240670</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.254272</td>
          <td>0.006540</td>
          <td>20.279142</td>
          <td>0.005054</td>
          <td>19.609216</td>
          <td>0.005014</td>
          <td>18.262946</td>
          <td>0.005006</td>
          <td>22.863785</td>
          <td>0.015730</td>
          <td>26.243853</td>
          <td>0.586348</td>
          <td>20.577613</td>
          <td>19.747314</td>
          <td>24.532159</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_8_0.png


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
          <td>28.139771</td>
          <td>32.148375</td>
          <td>22.478837</td>
          <td>21.339589</td>
          <td>21.757575</td>
          <td>20.983276</td>
          <td>21.517401</td>
          <td>0.005189</td>
          <td>24.269593</td>
          <td>0.030497</td>
          <td>24.166566</td>
          <td>0.027846</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.591746</td>
          <td>26.050021</td>
          <td>25.999354</td>
          <td>26.912770</td>
          <td>25.124822</td>
          <td>23.333065</td>
          <td>23.592406</td>
          <td>0.010612</td>
          <td>26.431967</td>
          <td>0.203393</td>
          <td>14.795377</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.288676</td>
          <td>21.457820</td>
          <td>18.071596</td>
          <td>26.710421</td>
          <td>16.250865</td>
          <td>19.506625</td>
          <td>20.037504</td>
          <td>0.005013</td>
          <td>25.065846</td>
          <td>0.061943</td>
          <td>21.094085</td>
          <td>0.005260</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.825414</td>
          <td>19.666019</td>
          <td>16.687810</td>
          <td>18.238687</td>
          <td>25.632725</td>
          <td>24.116998</td>
          <td>19.878115</td>
          <td>0.005009</td>
          <td>25.336596</td>
          <td>0.078770</td>
          <td>22.902471</td>
          <td>0.009964</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.703592</td>
          <td>20.461177</td>
          <td>20.639646</td>
          <td>28.776560</td>
          <td>24.201164</td>
          <td>22.497836</td>
          <td>16.189492</td>
          <td>0.005000</td>
          <td>26.249802</td>
          <td>0.174376</td>
          <td>21.584507</td>
          <td>0.005619</td>
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
          <td>17.885285</td>
          <td>17.747373</td>
          <td>25.574892</td>
          <td>24.382829</td>
          <td>20.603121</td>
          <td>18.687794</td>
          <td>25.999324</td>
          <td>0.083263</td>
          <td>22.707075</td>
          <td>0.008768</td>
          <td>18.502057</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.764712</td>
          <td>21.069966</td>
          <td>21.094874</td>
          <td>23.901255</td>
          <td>22.694941</td>
          <td>22.659367</td>
          <td>19.194618</td>
          <td>0.005003</td>
          <td>23.072180</td>
          <td>0.011244</td>
          <td>21.521201</td>
          <td>0.005554</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.849328</td>
          <td>24.719496</td>
          <td>20.261710</td>
          <td>25.718822</td>
          <td>27.139225</td>
          <td>20.621565</td>
          <td>22.467427</td>
          <td>0.006006</td>
          <td>17.323197</td>
          <td>0.005000</td>
          <td>21.923961</td>
          <td>0.006106</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.154748</td>
          <td>19.158508</td>
          <td>20.580623</td>
          <td>22.368237</td>
          <td>22.781533</td>
          <td>21.594454</td>
          <td>26.536920</td>
          <td>0.133309</td>
          <td>26.618433</td>
          <td>0.237586</td>
          <td>23.254126</td>
          <td>0.012908</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.250507</td>
          <td>20.274016</td>
          <td>19.602615</td>
          <td>18.257034</td>
          <td>22.885145</td>
          <td>27.695603</td>
          <td>20.577735</td>
          <td>0.005034</td>
          <td>19.740238</td>
          <td>0.005022</td>
          <td>24.491538</td>
          <td>0.037134</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_14_0.png


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
          <td>28.139771</td>
          <td>32.148375</td>
          <td>22.478837</td>
          <td>21.339589</td>
          <td>21.757575</td>
          <td>20.983276</td>
          <td>21.567160</td>
          <td>0.018075</td>
          <td>24.142664</td>
          <td>0.146051</td>
          <td>24.066476</td>
          <td>0.149074</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.591746</td>
          <td>26.050021</td>
          <td>25.999354</td>
          <td>26.912770</td>
          <td>25.124822</td>
          <td>23.333065</td>
          <td>23.626715</td>
          <td>0.111031</td>
          <td>26.383123</td>
          <td>0.822547</td>
          <td>14.792854</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.288676</td>
          <td>21.457820</td>
          <td>18.071596</td>
          <td>26.710421</td>
          <td>16.250865</td>
          <td>19.506625</td>
          <td>20.037860</td>
          <td>0.006573</td>
          <td>25.065230</td>
          <td>0.315034</td>
          <td>21.096744</td>
          <td>0.011450</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.825414</td>
          <td>19.666019</td>
          <td>16.687810</td>
          <td>18.238687</td>
          <td>25.632725</td>
          <td>24.116998</td>
          <td>19.873696</td>
          <td>0.006202</td>
          <td>25.153506</td>
          <td>0.337948</td>
          <td>22.899490</td>
          <td>0.053413</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.703592</td>
          <td>20.461177</td>
          <td>20.639646</td>
          <td>28.776560</td>
          <td>24.201164</td>
          <td>22.497836</td>
          <td>16.192130</td>
          <td>0.005002</td>
          <td>26.028354</td>
          <td>0.648563</td>
          <td>21.579860</td>
          <td>0.016798</td>
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
          <td>17.885285</td>
          <td>17.747373</td>
          <td>25.574892</td>
          <td>24.382829</td>
          <td>20.603121</td>
          <td>18.687794</td>
          <td>26.027233</td>
          <td>0.742532</td>
          <td>22.690452</td>
          <td>0.040552</td>
          <td>18.489721</td>
          <td>0.005087</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.764712</td>
          <td>21.069966</td>
          <td>21.094874</td>
          <td>23.901255</td>
          <td>22.694941</td>
          <td>22.659367</td>
          <td>19.199129</td>
          <td>0.005375</td>
          <td>23.153483</td>
          <td>0.061265</td>
          <td>21.546893</td>
          <td>0.016343</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.849328</td>
          <td>24.719496</td>
          <td>20.261710</td>
          <td>25.718822</td>
          <td>27.139225</td>
          <td>20.621565</td>
          <td>22.472143</td>
          <td>0.039897</td>
          <td>17.325028</td>
          <td>0.005009</td>
          <td>21.938197</td>
          <td>0.022803</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.154748</td>
          <td>19.158508</td>
          <td>20.580623</td>
          <td>22.368237</td>
          <td>22.781533</td>
          <td>21.594454</td>
          <td>25.777552</td>
          <td>0.626010</td>
          <td>26.152227</td>
          <td>0.706025</td>
          <td>23.302995</td>
          <td>0.076461</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.250507</td>
          <td>20.274016</td>
          <td>19.602615</td>
          <td>18.257034</td>
          <td>22.885145</td>
          <td>27.695603</td>
          <td>20.592360</td>
          <td>0.008688</td>
          <td>19.747549</td>
          <td>0.005691</td>
          <td>24.582643</td>
          <td>0.230647</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_17_0.png


