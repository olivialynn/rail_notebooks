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
          <td>23.928854</td>
          <td>18.302601</td>
          <td>23.581394</td>
          <td>28.673865</td>
          <td>21.524611</td>
          <td>20.275141</td>
          <td>25.571721</td>
          <td>19.451239</td>
          <td>24.015987</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.254605</td>
          <td>16.737465</td>
          <td>23.159487</td>
          <td>25.027472</td>
          <td>17.448685</td>
          <td>22.950438</td>
          <td>27.994537</td>
          <td>22.075250</td>
          <td>27.017231</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.901193</td>
          <td>25.914291</td>
          <td>20.612991</td>
          <td>29.315590</td>
          <td>22.835082</td>
          <td>24.607867</td>
          <td>25.648112</td>
          <td>26.816013</td>
          <td>22.484595</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.495132</td>
          <td>27.842856</td>
          <td>27.004804</td>
          <td>25.418901</td>
          <td>22.669849</td>
          <td>29.640662</td>
          <td>28.662012</td>
          <td>24.977414</td>
          <td>24.078884</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.868285</td>
          <td>22.208810</td>
          <td>25.845018</td>
          <td>22.207384</td>
          <td>20.884896</td>
          <td>24.651360</td>
          <td>24.827057</td>
          <td>22.231324</td>
          <td>26.429766</td>
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
          <td>24.508497</td>
          <td>21.620960</td>
          <td>22.813833</td>
          <td>19.870995</td>
          <td>21.723892</td>
          <td>20.439596</td>
          <td>23.314390</td>
          <td>24.506417</td>
          <td>25.498132</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.653471</td>
          <td>18.837050</td>
          <td>25.099665</td>
          <td>21.842317</td>
          <td>24.591332</td>
          <td>19.727935</td>
          <td>24.952863</td>
          <td>23.559908</td>
          <td>18.915452</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.094723</td>
          <td>21.555715</td>
          <td>23.254121</td>
          <td>21.274989</td>
          <td>22.849241</td>
          <td>24.461669</td>
          <td>27.260651</td>
          <td>21.622271</td>
          <td>21.467078</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.695575</td>
          <td>26.134259</td>
          <td>24.472370</td>
          <td>21.974297</td>
          <td>22.349754</td>
          <td>20.873008</td>
          <td>27.177863</td>
          <td>17.007793</td>
          <td>25.175151</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.780362</td>
          <td>21.497511</td>
          <td>20.460822</td>
          <td>20.749389</td>
          <td>20.876557</td>
          <td>26.351421</td>
          <td>20.032560</td>
          <td>21.848955</td>
          <td>18.173079</td>
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
          <td>23.898160</td>
          <td>0.041180</td>
          <td>18.294641</td>
          <td>0.005006</td>
          <td>23.588740</td>
          <td>0.010285</td>
          <td>28.151132</td>
          <td>0.700164</td>
          <td>21.521047</td>
          <td>0.006722</td>
          <td>20.281282</td>
          <td>0.006046</td>
          <td>25.571721</td>
          <td>19.451239</td>
          <td>24.015987</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.254853</td>
          <td>0.005110</td>
          <td>16.735019</td>
          <td>0.005001</td>
          <td>23.148240</td>
          <td>0.007858</td>
          <td>25.052027</td>
          <td>0.055596</td>
          <td>17.449125</td>
          <td>0.005005</td>
          <td>22.995557</td>
          <td>0.038846</td>
          <td>27.994537</td>
          <td>22.075250</td>
          <td>27.017231</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.882259</td>
          <td>0.017356</td>
          <td>25.864037</td>
          <td>0.079500</td>
          <td>20.608795</td>
          <td>0.005055</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.837645</td>
          <td>0.015401</td>
          <td>24.798847</td>
          <td>0.188335</td>
          <td>25.648112</td>
          <td>26.816013</td>
          <td>22.484595</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.521919</td>
          <td>0.071260</td>
          <td>27.825119</td>
          <td>0.410931</td>
          <td>27.145512</td>
          <td>0.212001</td>
          <td>25.463457</td>
          <td>0.080039</td>
          <td>22.671592</td>
          <td>0.013501</td>
          <td>26.065946</td>
          <td>0.515643</td>
          <td>28.662012</td>
          <td>24.977414</td>
          <td>24.078884</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.855866</td>
          <td>0.016992</td>
          <td>22.210284</td>
          <td>0.005903</td>
          <td>25.863974</td>
          <td>0.069913</td>
          <td>22.216078</td>
          <td>0.006617</td>
          <td>20.889445</td>
          <td>0.005636</td>
          <td>24.677557</td>
          <td>0.169935</td>
          <td>24.827057</td>
          <td>22.231324</td>
          <td>26.429766</td>
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
          <td>24.504350</td>
          <td>0.070168</td>
          <td>21.622920</td>
          <td>0.005366</td>
          <td>22.822102</td>
          <td>0.006755</td>
          <td>19.869354</td>
          <td>0.005042</td>
          <td>21.720458</td>
          <td>0.007327</td>
          <td>20.441436</td>
          <td>0.006345</td>
          <td>23.314390</td>
          <td>24.506417</td>
          <td>25.498132</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.660435</td>
          <td>0.007696</td>
          <td>18.829999</td>
          <td>0.005010</td>
          <td>25.065087</td>
          <td>0.034430</td>
          <td>21.850107</td>
          <td>0.005909</td>
          <td>24.487656</td>
          <td>0.064571</td>
          <td>19.731817</td>
          <td>0.005434</td>
          <td>24.952863</td>
          <td>23.559908</td>
          <td>18.915452</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.088408</td>
          <td>0.005311</td>
          <td>21.554130</td>
          <td>0.005330</td>
          <td>23.278239</td>
          <td>0.008446</td>
          <td>21.279559</td>
          <td>0.005362</td>
          <td>22.843266</td>
          <td>0.015471</td>
          <td>24.640242</td>
          <td>0.164617</td>
          <td>27.260651</td>
          <td>21.622271</td>
          <td>21.467078</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.556985</td>
          <td>0.804491</td>
          <td>26.001101</td>
          <td>0.089689</td>
          <td>24.466538</td>
          <td>0.020458</td>
          <td>21.968349</td>
          <td>0.006097</td>
          <td>22.345062</td>
          <td>0.010602</td>
          <td>20.871552</td>
          <td>0.007584</td>
          <td>27.177863</td>
          <td>17.007793</td>
          <td>25.175151</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.782503</td>
          <td>0.005210</td>
          <td>21.495266</td>
          <td>0.005302</td>
          <td>20.458448</td>
          <td>0.005044</td>
          <td>20.756442</td>
          <td>0.005157</td>
          <td>20.871444</td>
          <td>0.005617</td>
          <td>26.097863</td>
          <td>0.527816</td>
          <td>20.032560</td>
          <td>21.848955</td>
          <td>18.173079</td>
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
          <td>23.928854</td>
          <td>18.302601</td>
          <td>23.581394</td>
          <td>28.673865</td>
          <td>21.524611</td>
          <td>20.275141</td>
          <td>25.634155</td>
          <td>0.060220</td>
          <td>19.455229</td>
          <td>0.005013</td>
          <td>24.039627</td>
          <td>0.024910</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.254605</td>
          <td>16.737465</td>
          <td>23.159487</td>
          <td>25.027472</td>
          <td>17.448685</td>
          <td>22.950438</td>
          <td>28.009775</td>
          <td>0.445241</td>
          <td>22.073742</td>
          <td>0.006417</td>
          <td>26.982567</td>
          <td>0.319426</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.901193</td>
          <td>25.914291</td>
          <td>20.612991</td>
          <td>29.315590</td>
          <td>22.835082</td>
          <td>24.607867</td>
          <td>25.632711</td>
          <td>0.060143</td>
          <td>26.566954</td>
          <td>0.227663</td>
          <td>22.488211</td>
          <td>0.007726</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.495132</td>
          <td>27.842856</td>
          <td>27.004804</td>
          <td>25.418901</td>
          <td>22.669849</td>
          <td>29.640662</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.910932</td>
          <td>0.053961</td>
          <td>24.064218</td>
          <td>0.025452</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.868285</td>
          <td>22.208810</td>
          <td>25.845018</td>
          <td>22.207384</td>
          <td>20.884896</td>
          <td>24.651360</td>
          <td>24.860492</td>
          <td>0.030252</td>
          <td>22.222072</td>
          <td>0.006802</td>
          <td>26.515802</td>
          <td>0.218174</td>
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
          <td>24.508497</td>
          <td>21.620960</td>
          <td>22.813833</td>
          <td>19.870995</td>
          <td>21.723892</td>
          <td>20.439596</td>
          <td>23.295061</td>
          <td>0.008703</td>
          <td>24.479222</td>
          <td>0.036730</td>
          <td>25.570660</td>
          <td>0.096837</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.653471</td>
          <td>18.837050</td>
          <td>25.099665</td>
          <td>21.842317</td>
          <td>24.591332</td>
          <td>19.727935</td>
          <td>24.981895</td>
          <td>0.033687</td>
          <td>23.575770</td>
          <td>0.016741</td>
          <td>18.907102</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.094723</td>
          <td>21.555715</td>
          <td>23.254121</td>
          <td>21.274989</td>
          <td>22.849241</td>
          <td>24.461669</td>
          <td>26.943686</td>
          <td>0.188819</td>
          <td>21.631219</td>
          <td>0.005672</td>
          <td>21.473120</td>
          <td>0.005510</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.695575</td>
          <td>26.134259</td>
          <td>24.472370</td>
          <td>21.974297</td>
          <td>22.349754</td>
          <td>20.873008</td>
          <td>27.065569</td>
          <td>0.209205</td>
          <td>17.008820</td>
          <td>0.005000</td>
          <td>25.087065</td>
          <td>0.063123</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.780362</td>
          <td>21.497511</td>
          <td>20.460822</td>
          <td>20.749389</td>
          <td>20.876557</td>
          <td>26.351421</td>
          <td>20.039092</td>
          <td>0.005013</td>
          <td>21.837446</td>
          <td>0.005956</td>
          <td>18.179614</td>
          <td>0.005001</td>
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
          <td>23.928854</td>
          <td>18.302601</td>
          <td>23.581394</td>
          <td>28.673865</td>
          <td>21.524611</td>
          <td>20.275141</td>
          <td>25.075500</td>
          <td>0.371940</td>
          <td>19.454963</td>
          <td>0.005414</td>
          <td>24.190722</td>
          <td>0.165816</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.254605</td>
          <td>16.737465</td>
          <td>23.159487</td>
          <td>25.027472</td>
          <td>17.448685</td>
          <td>22.950438</td>
          <td>27.313500</td>
          <td>1.560563</td>
          <td>22.067056</td>
          <td>0.023382</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.901193</td>
          <td>25.914291</td>
          <td>20.612991</td>
          <td>29.315590</td>
          <td>22.835082</td>
          <td>24.607867</td>
          <td>26.022877</td>
          <td>0.740376</td>
          <td>25.244943</td>
          <td>0.363166</td>
          <td>22.504835</td>
          <td>0.037576</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.495132</td>
          <td>27.842856</td>
          <td>27.004804</td>
          <td>25.418901</td>
          <td>22.669849</td>
          <td>29.640662</td>
          <td>25.682706</td>
          <td>0.585468</td>
          <td>25.681231</td>
          <td>0.505987</td>
          <td>24.201208</td>
          <td>0.167307</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.868285</td>
          <td>22.208810</td>
          <td>25.845018</td>
          <td>22.207384</td>
          <td>20.884896</td>
          <td>24.651360</td>
          <td>24.930586</td>
          <td>0.331866</td>
          <td>22.179409</td>
          <td>0.025793</td>
          <td>27.126929</td>
          <td>1.349232</td>
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
          <td>24.508497</td>
          <td>21.620960</td>
          <td>22.813833</td>
          <td>19.870995</td>
          <td>21.723892</td>
          <td>20.439596</td>
          <td>23.265566</td>
          <td>0.080815</td>
          <td>24.264546</td>
          <td>0.162150</td>
          <td>25.612704</td>
          <td>0.517817</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.653471</td>
          <td>18.837050</td>
          <td>25.099665</td>
          <td>21.842317</td>
          <td>24.591332</td>
          <td>19.727935</td>
          <td>24.612241</td>
          <td>0.256668</td>
          <td>23.675421</td>
          <td>0.097243</td>
          <td>18.921693</td>
          <td>0.005190</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.094723</td>
          <td>21.555715</td>
          <td>23.254121</td>
          <td>21.274989</td>
          <td>22.849241</td>
          <td>24.461669</td>
          <td>26.809026</td>
          <td>1.198830</td>
          <td>21.621504</td>
          <td>0.016004</td>
          <td>21.432861</td>
          <td>0.014882</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.695575</td>
          <td>26.134259</td>
          <td>24.472370</td>
          <td>21.974297</td>
          <td>22.349754</td>
          <td>20.873008</td>
          <td>25.572692</td>
          <td>0.540962</td>
          <td>17.020414</td>
          <td>0.005005</td>
          <td>24.908872</td>
          <td>0.301116</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.780362</td>
          <td>21.497511</td>
          <td>20.460822</td>
          <td>20.749389</td>
          <td>20.876557</td>
          <td>26.351421</td>
          <td>20.028098</td>
          <td>0.006549</td>
          <td>21.836296</td>
          <td>0.019166</td>
          <td>18.172217</td>
          <td>0.005049</td>
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


