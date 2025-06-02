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
          <td>21.852390</td>
          <td>22.568724</td>
          <td>20.165960</td>
          <td>22.258080</td>
          <td>19.883781</td>
          <td>25.927259</td>
          <td>23.160585</td>
          <td>25.849257</td>
          <td>20.641254</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.211711</td>
          <td>21.240539</td>
          <td>24.797190</td>
          <td>21.208614</td>
          <td>21.494663</td>
          <td>25.832661</td>
          <td>17.997286</td>
          <td>22.231227</td>
          <td>25.431567</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.147226</td>
          <td>20.945687</td>
          <td>21.456302</td>
          <td>30.077091</td>
          <td>23.005297</td>
          <td>26.680907</td>
          <td>20.162372</td>
          <td>19.329393</td>
          <td>21.147230</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.965364</td>
          <td>25.695520</td>
          <td>21.942997</td>
          <td>22.719441</td>
          <td>21.657903</td>
          <td>24.453254</td>
          <td>24.344804</td>
          <td>20.398637</td>
          <td>23.198269</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.951797</td>
          <td>23.373937</td>
          <td>23.374592</td>
          <td>19.193640</td>
          <td>22.583452</td>
          <td>27.617527</td>
          <td>26.412406</td>
          <td>22.499827</td>
          <td>25.759728</td>
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
          <td>24.382924</td>
          <td>26.751784</td>
          <td>26.155515</td>
          <td>24.386426</td>
          <td>23.943369</td>
          <td>21.092625</td>
          <td>19.421553</td>
          <td>19.639094</td>
          <td>24.386084</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.555210</td>
          <td>20.886378</td>
          <td>24.710198</td>
          <td>20.728741</td>
          <td>20.980185</td>
          <td>26.709229</td>
          <td>29.064250</td>
          <td>19.338082</td>
          <td>27.322827</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.254278</td>
          <td>22.116254</td>
          <td>20.584522</td>
          <td>23.073818</td>
          <td>23.242212</td>
          <td>24.786406</td>
          <td>23.629476</td>
          <td>21.856882</td>
          <td>27.362309</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.455166</td>
          <td>24.675801</td>
          <td>19.122307</td>
          <td>25.667985</td>
          <td>20.006415</td>
          <td>23.461506</td>
          <td>29.358669</td>
          <td>18.327667</td>
          <td>26.221006</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.145448</td>
          <td>23.874438</td>
          <td>23.573279</td>
          <td>21.121600</td>
          <td>24.654003</td>
          <td>22.956950</td>
          <td>18.557989</td>
          <td>24.938635</td>
          <td>28.817269</td>
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
          <td>21.849706</td>
          <td>0.008476</td>
          <td>22.568855</td>
          <td>0.006559</td>
          <td>20.156660</td>
          <td>0.005029</td>
          <td>22.254792</td>
          <td>0.006716</td>
          <td>19.886412</td>
          <td>0.005129</td>
          <td>26.075691</td>
          <td>0.519337</td>
          <td>23.160585</td>
          <td>25.849257</td>
          <td>20.641254</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.218613</td>
          <td>0.022909</td>
          <td>21.238522</td>
          <td>0.005206</td>
          <td>24.791341</td>
          <td>0.027073</td>
          <td>21.210656</td>
          <td>0.005324</td>
          <td>21.497054</td>
          <td>0.006660</td>
          <td>26.158877</td>
          <td>0.551705</td>
          <td>17.997286</td>
          <td>22.231227</td>
          <td>25.431567</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.117730</td>
          <td>0.021052</td>
          <td>20.946749</td>
          <td>0.005135</td>
          <td>21.458282</td>
          <td>0.005200</td>
          <td>28.432590</td>
          <td>0.843015</td>
          <td>22.997724</td>
          <td>0.017559</td>
          <td>27.557891</td>
          <td>1.329186</td>
          <td>20.162372</td>
          <td>19.329393</td>
          <td>21.147230</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.805737</td>
          <td>0.215350</td>
          <td>25.831585</td>
          <td>0.077257</td>
          <td>21.944084</td>
          <td>0.005435</td>
          <td>22.723802</td>
          <td>0.008439</td>
          <td>21.661678</td>
          <td>0.007131</td>
          <td>24.441755</td>
          <td>0.138847</td>
          <td>24.344804</td>
          <td>20.398637</td>
          <td>23.198269</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.970984</td>
          <td>0.018651</td>
          <td>23.369130</td>
          <td>0.009879</td>
          <td>23.385245</td>
          <td>0.009005</td>
          <td>19.190716</td>
          <td>0.005017</td>
          <td>22.576407</td>
          <td>0.012549</td>
          <td>27.275011</td>
          <td>1.137397</td>
          <td>26.412406</td>
          <td>22.499827</td>
          <td>25.759728</td>
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
          <td>24.502844</td>
          <td>0.070076</td>
          <td>26.816544</td>
          <td>0.181497</td>
          <td>26.097320</td>
          <td>0.085912</td>
          <td>24.356331</td>
          <td>0.030030</td>
          <td>23.928921</td>
          <td>0.039333</td>
          <td>21.094106</td>
          <td>0.008561</td>
          <td>19.421553</td>
          <td>19.639094</td>
          <td>24.386084</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.550317</td>
          <td>0.005157</td>
          <td>20.887404</td>
          <td>0.005124</td>
          <td>24.727029</td>
          <td>0.025598</td>
          <td>20.724849</td>
          <td>0.005150</td>
          <td>20.986880</td>
          <td>0.005743</td>
          <td>27.540190</td>
          <td>1.316720</td>
          <td>29.064250</td>
          <td>19.338082</td>
          <td>27.322827</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.256599</td>
          <td>0.006545</td>
          <td>22.117094</td>
          <td>0.005782</td>
          <td>20.586699</td>
          <td>0.005053</td>
          <td>23.081992</td>
          <td>0.010621</td>
          <td>23.223727</td>
          <td>0.021242</td>
          <td>24.618080</td>
          <td>0.161533</td>
          <td>23.629476</td>
          <td>21.856882</td>
          <td>27.362309</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.452518</td>
          <td>0.027958</td>
          <td>24.707830</td>
          <td>0.028655</td>
          <td>19.130652</td>
          <td>0.005008</td>
          <td>25.740110</td>
          <td>0.102081</td>
          <td>20.011133</td>
          <td>0.005157</td>
          <td>23.567188</td>
          <td>0.064503</td>
          <td>29.358669</td>
          <td>18.327667</td>
          <td>26.221006</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.179680</td>
          <td>0.022171</td>
          <td>23.885892</td>
          <td>0.014388</td>
          <td>23.582876</td>
          <td>0.010244</td>
          <td>21.136967</td>
          <td>0.005288</td>
          <td>24.573395</td>
          <td>0.069666</td>
          <td>23.043072</td>
          <td>0.040516</td>
          <td>18.557989</td>
          <td>24.938635</td>
          <td>28.817269</td>
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
          <td>21.852390</td>
          <td>22.568724</td>
          <td>20.165960</td>
          <td>22.258080</td>
          <td>19.883781</td>
          <td>25.927259</td>
          <td>23.168314</td>
          <td>0.008074</td>
          <td>25.923159</td>
          <td>0.131730</td>
          <td>20.635037</td>
          <td>0.005113</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.211711</td>
          <td>21.240539</td>
          <td>24.797190</td>
          <td>21.208614</td>
          <td>21.494663</td>
          <td>25.832661</td>
          <td>17.989982</td>
          <td>0.005000</td>
          <td>22.222553</td>
          <td>0.006803</td>
          <td>25.595430</td>
          <td>0.098967</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.147226</td>
          <td>20.945687</td>
          <td>21.456302</td>
          <td>30.077091</td>
          <td>23.005297</td>
          <td>26.680907</td>
          <td>20.161933</td>
          <td>0.005016</td>
          <td>19.325406</td>
          <td>0.005010</td>
          <td>21.156544</td>
          <td>0.005291</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.965364</td>
          <td>25.695520</td>
          <td>21.942997</td>
          <td>22.719441</td>
          <td>21.657903</td>
          <td>24.453254</td>
          <td>24.371648</td>
          <td>0.019753</td>
          <td>20.396304</td>
          <td>0.005073</td>
          <td>23.186683</td>
          <td>0.012253</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.951797</td>
          <td>23.373937</td>
          <td>23.374592</td>
          <td>19.193640</td>
          <td>22.583452</td>
          <td>27.617527</td>
          <td>26.373588</td>
          <td>0.115669</td>
          <td>22.505400</td>
          <td>0.007798</td>
          <td>25.867094</td>
          <td>0.125477</td>
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
          <td>24.382924</td>
          <td>26.751784</td>
          <td>26.155515</td>
          <td>24.386426</td>
          <td>23.943369</td>
          <td>21.092625</td>
          <td>19.419201</td>
          <td>0.005004</td>
          <td>19.638365</td>
          <td>0.005018</td>
          <td>24.441744</td>
          <td>0.035526</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.555210</td>
          <td>20.886378</td>
          <td>24.710198</td>
          <td>20.728741</td>
          <td>20.980185</td>
          <td>26.709229</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.342083</td>
          <td>0.005011</td>
          <td>27.819412</td>
          <td>0.600918</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.254278</td>
          <td>22.116254</td>
          <td>20.584522</td>
          <td>23.073818</td>
          <td>23.242212</td>
          <td>24.786406</td>
          <td>23.646264</td>
          <td>0.011032</td>
          <td>21.857807</td>
          <td>0.005990</td>
          <td>27.055638</td>
          <td>0.338519</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.455166</td>
          <td>24.675801</td>
          <td>19.122307</td>
          <td>25.667985</td>
          <td>20.006415</td>
          <td>23.461506</td>
          <td>28.596078</td>
          <td>0.679543</td>
          <td>18.326032</td>
          <td>0.005002</td>
          <td>26.368787</td>
          <td>0.192863</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.145448</td>
          <td>23.874438</td>
          <td>23.573279</td>
          <td>21.121600</td>
          <td>24.654003</td>
          <td>22.956950</td>
          <td>18.566219</td>
          <td>0.005001</td>
          <td>25.019159</td>
          <td>0.059422</td>
          <td>28.479116</td>
          <td>0.931042</td>
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
          <td>21.852390</td>
          <td>22.568724</td>
          <td>20.165960</td>
          <td>22.258080</td>
          <td>19.883781</td>
          <td>25.927259</td>
          <td>23.078520</td>
          <td>0.068469</td>
          <td>25.389688</td>
          <td>0.406311</td>
          <td>20.637777</td>
          <td>0.008406</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.211711</td>
          <td>21.240539</td>
          <td>24.797190</td>
          <td>21.208614</td>
          <td>21.494663</td>
          <td>25.832661</td>
          <td>17.993972</td>
          <td>0.005042</td>
          <td>22.277812</td>
          <td>0.028123</td>
          <td>25.327282</td>
          <td>0.418186</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.147226</td>
          <td>20.945687</td>
          <td>21.456302</td>
          <td>30.077091</td>
          <td>23.005297</td>
          <td>26.680907</td>
          <td>20.172148</td>
          <td>0.006951</td>
          <td>19.326541</td>
          <td>0.005329</td>
          <td>21.117824</td>
          <td>0.011631</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.965364</td>
          <td>25.695520</td>
          <td>21.942997</td>
          <td>22.719441</td>
          <td>21.657903</td>
          <td>24.453254</td>
          <td>24.841616</td>
          <td>0.309136</td>
          <td>20.391777</td>
          <td>0.007012</td>
          <td>23.233811</td>
          <td>0.071913</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.951797</td>
          <td>23.373937</td>
          <td>23.374592</td>
          <td>19.193640</td>
          <td>22.583452</td>
          <td>27.617527</td>
          <td>25.324871</td>
          <td>0.450343</td>
          <td>22.501808</td>
          <td>0.034288</td>
          <td>25.479305</td>
          <td>0.469130</td>
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
          <td>24.382924</td>
          <td>26.751784</td>
          <td>26.155515</td>
          <td>24.386426</td>
          <td>23.943369</td>
          <td>21.092625</td>
          <td>19.413322</td>
          <td>0.005547</td>
          <td>19.654251</td>
          <td>0.005587</td>
          <td>24.271643</td>
          <td>0.177642</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.555210</td>
          <td>20.886378</td>
          <td>24.710198</td>
          <td>20.728741</td>
          <td>20.980185</td>
          <td>26.709229</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.337117</td>
          <td>0.005336</td>
          <td>26.237223</td>
          <td>0.798407</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.254278</td>
          <td>22.116254</td>
          <td>20.584522</td>
          <td>23.073818</td>
          <td>23.242212</td>
          <td>24.786406</td>
          <td>23.665383</td>
          <td>0.114844</td>
          <td>21.846494</td>
          <td>0.019333</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.455166</td>
          <td>24.675801</td>
          <td>19.122307</td>
          <td>25.667985</td>
          <td>20.006415</td>
          <td>23.461506</td>
          <td>25.812291</td>
          <td>0.641369</td>
          <td>18.329227</td>
          <td>0.005054</td>
          <td>25.268917</td>
          <td>0.399870</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.145448</td>
          <td>23.874438</td>
          <td>23.573279</td>
          <td>21.121600</td>
          <td>24.654003</td>
          <td>22.956950</td>
          <td>18.563568</td>
          <td>0.005119</td>
          <td>24.923039</td>
          <td>0.280946</td>
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


