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
          <td>22.314386</td>
          <td>23.918189</td>
          <td>25.334986</td>
          <td>23.095449</td>
          <td>23.590106</td>
          <td>27.726421</td>
          <td>26.832983</td>
          <td>16.849749</td>
          <td>25.460777</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.609931</td>
          <td>22.830471</td>
          <td>24.267241</td>
          <td>21.313660</td>
          <td>28.221857</td>
          <td>19.491990</td>
          <td>25.770190</td>
          <td>26.573579</td>
          <td>23.535075</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.362773</td>
          <td>23.368922</td>
          <td>18.950705</td>
          <td>24.152540</td>
          <td>28.244525</td>
          <td>21.686352</td>
          <td>26.309502</td>
          <td>26.008335</td>
          <td>27.372868</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.785893</td>
          <td>21.239032</td>
          <td>20.270558</td>
          <td>26.594319</td>
          <td>20.118460</td>
          <td>26.464066</td>
          <td>27.794061</td>
          <td>20.068139</td>
          <td>21.662634</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.056336</td>
          <td>24.226448</td>
          <td>25.467764</td>
          <td>20.986680</td>
          <td>22.442503</td>
          <td>22.885064</td>
          <td>28.206894</td>
          <td>24.611571</td>
          <td>22.361921</td>
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
          <td>22.889217</td>
          <td>27.655452</td>
          <td>21.635530</td>
          <td>21.553892</td>
          <td>25.829352</td>
          <td>28.113020</td>
          <td>25.549153</td>
          <td>16.285209</td>
          <td>25.990959</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.654618</td>
          <td>27.043359</td>
          <td>26.571251</td>
          <td>28.108379</td>
          <td>29.269918</td>
          <td>20.737710</td>
          <td>21.050849</td>
          <td>22.405963</td>
          <td>21.004807</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.684956</td>
          <td>19.508298</td>
          <td>25.575798</td>
          <td>23.887208</td>
          <td>21.249897</td>
          <td>23.322317</td>
          <td>25.153414</td>
          <td>24.646690</td>
          <td>23.121571</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.167783</td>
          <td>25.744580</td>
          <td>22.960786</td>
          <td>17.076286</td>
          <td>26.864794</td>
          <td>23.109582</td>
          <td>25.952241</td>
          <td>24.702897</td>
          <td>18.851763</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.329299</td>
          <td>25.030832</td>
          <td>24.387894</td>
          <td>23.502625</td>
          <td>17.687798</td>
          <td>20.820170</td>
          <td>22.282335</td>
          <td>25.074807</td>
          <td>29.572036</td>
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
          <td>22.321658</td>
          <td>0.011368</td>
          <td>23.899678</td>
          <td>0.014545</td>
          <td>25.296076</td>
          <td>0.042241</td>
          <td>23.097153</td>
          <td>0.010735</td>
          <td>23.594198</td>
          <td>0.029281</td>
          <td>28.518242</td>
          <td>2.083787</td>
          <td>26.832983</td>
          <td>16.849749</td>
          <td>25.460777</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.676129</td>
          <td>0.193221</td>
          <td>22.828348</td>
          <td>0.007291</td>
          <td>24.238754</td>
          <td>0.016903</td>
          <td>21.322599</td>
          <td>0.005388</td>
          <td>28.105648</td>
          <td>1.080709</td>
          <td>19.486112</td>
          <td>0.005293</td>
          <td>25.770190</td>
          <td>26.573579</td>
          <td>23.535075</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.497983</td>
          <td>0.166207</td>
          <td>23.362790</td>
          <td>0.009837</td>
          <td>18.951927</td>
          <td>0.005007</td>
          <td>24.192950</td>
          <td>0.026033</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.698827</td>
          <td>0.012956</td>
          <td>26.309502</td>
          <td>26.008335</td>
          <td>27.372868</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.795150</td>
          <td>0.008232</td>
          <td>21.237344</td>
          <td>0.005205</td>
          <td>20.270433</td>
          <td>0.005034</td>
          <td>26.745780</td>
          <td>0.240894</td>
          <td>20.109695</td>
          <td>0.005183</td>
          <td>26.468120</td>
          <td>0.685587</td>
          <td>27.794061</td>
          <td>20.068139</td>
          <td>21.662634</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.084224</td>
          <td>0.020474</td>
          <td>24.234127</td>
          <td>0.019099</td>
          <td>25.503609</td>
          <td>0.050785</td>
          <td>20.990542</td>
          <td>0.005228</td>
          <td>22.422443</td>
          <td>0.011200</td>
          <td>22.877684</td>
          <td>0.035001</td>
          <td>28.206894</td>
          <td>24.611571</td>
          <td>22.361921</td>
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
          <td>22.906290</td>
          <td>0.017696</td>
          <td>27.546867</td>
          <td>0.330760</td>
          <td>21.631870</td>
          <td>0.005263</td>
          <td>21.550310</td>
          <td>0.005561</td>
          <td>25.970989</td>
          <td>0.232695</td>
          <td>28.771924</td>
          <td>2.303752</td>
          <td>25.549153</td>
          <td>16.285209</td>
          <td>25.990959</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.683495</td>
          <td>0.014834</td>
          <td>27.523602</td>
          <td>0.324705</td>
          <td>26.472391</td>
          <td>0.119307</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.458230</td>
          <td>2.101512</td>
          <td>20.727372</td>
          <td>0.007085</td>
          <td>21.050849</td>
          <td>22.405963</td>
          <td>21.004807</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.705875</td>
          <td>0.015094</td>
          <td>19.505679</td>
          <td>0.005021</td>
          <td>25.596352</td>
          <td>0.055144</td>
          <td>23.929817</td>
          <td>0.020751</td>
          <td>21.256204</td>
          <td>0.006141</td>
          <td>23.348786</td>
          <td>0.053141</td>
          <td>25.153414</td>
          <td>24.646690</td>
          <td>23.121571</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.158873</td>
          <td>0.021788</td>
          <td>25.732152</td>
          <td>0.070766</td>
          <td>22.958727</td>
          <td>0.007159</td>
          <td>17.083038</td>
          <td>0.005002</td>
          <td>27.799937</td>
          <td>0.898255</td>
          <td>23.080272</td>
          <td>0.041874</td>
          <td>25.952241</td>
          <td>24.702897</td>
          <td>18.851763</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.319363</td>
          <td>0.005422</td>
          <td>25.043083</td>
          <td>0.038461</td>
          <td>24.384208</td>
          <td>0.019082</td>
          <td>23.492767</td>
          <td>0.014460</td>
          <td>17.685337</td>
          <td>0.005007</td>
          <td>20.811688</td>
          <td>0.007365</td>
          <td>22.282335</td>
          <td>25.074807</td>
          <td>29.572036</td>
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
          <td>22.314386</td>
          <td>23.918189</td>
          <td>25.334986</td>
          <td>23.095449</td>
          <td>23.590106</td>
          <td>27.726421</td>
          <td>26.723744</td>
          <td>0.156585</td>
          <td>16.857074</td>
          <td>0.005000</td>
          <td>25.592156</td>
          <td>0.098683</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.609931</td>
          <td>22.830471</td>
          <td>24.267241</td>
          <td>21.313660</td>
          <td>28.221857</td>
          <td>19.491990</td>
          <td>25.917143</td>
          <td>0.077425</td>
          <td>26.262171</td>
          <td>0.176218</td>
          <td>23.512462</td>
          <td>0.015885</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.362773</td>
          <td>23.368922</td>
          <td>18.950705</td>
          <td>24.152540</td>
          <td>28.244525</td>
          <td>21.686352</td>
          <td>26.381170</td>
          <td>0.116436</td>
          <td>26.078884</td>
          <td>0.150672</td>
          <td>27.084808</td>
          <td>0.346408</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.785893</td>
          <td>21.239032</td>
          <td>20.270558</td>
          <td>26.594319</td>
          <td>20.118460</td>
          <td>26.464066</td>
          <td>27.739969</td>
          <td>0.361754</td>
          <td>20.065665</td>
          <td>0.005040</td>
          <td>21.665117</td>
          <td>0.005712</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.056336</td>
          <td>24.226448</td>
          <td>25.467764</td>
          <td>20.986680</td>
          <td>22.442503</td>
          <td>22.885064</td>
          <td>27.464439</td>
          <td>0.290522</td>
          <td>24.572970</td>
          <td>0.039926</td>
          <td>22.362486</td>
          <td>0.007248</td>
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
          <td>22.889217</td>
          <td>27.655452</td>
          <td>21.635530</td>
          <td>21.553892</td>
          <td>25.829352</td>
          <td>28.113020</td>
          <td>25.496467</td>
          <td>0.053270</td>
          <td>16.290746</td>
          <td>0.005000</td>
          <td>25.807541</td>
          <td>0.119143</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.654618</td>
          <td>27.043359</td>
          <td>26.571251</td>
          <td>28.108379</td>
          <td>29.269918</td>
          <td>20.737710</td>
          <td>21.051592</td>
          <td>0.005081</td>
          <td>22.404043</td>
          <td>0.007397</td>
          <td>21.022954</td>
          <td>0.005229</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.684956</td>
          <td>19.508298</td>
          <td>25.575798</td>
          <td>23.887208</td>
          <td>21.249897</td>
          <td>23.322317</td>
          <td>25.089670</td>
          <td>0.037073</td>
          <td>24.684907</td>
          <td>0.044114</td>
          <td>23.148144</td>
          <td>0.011899</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.167783</td>
          <td>25.744580</td>
          <td>22.960786</td>
          <td>17.076286</td>
          <td>26.864794</td>
          <td>23.109582</td>
          <td>25.853632</td>
          <td>0.073188</td>
          <td>24.698568</td>
          <td>0.044654</td>
          <td>18.846082</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.329299</td>
          <td>25.030832</td>
          <td>24.387894</td>
          <td>23.502625</td>
          <td>17.687798</td>
          <td>20.820170</td>
          <td>22.280441</td>
          <td>0.005731</td>
          <td>25.058018</td>
          <td>0.061513</td>
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
          <td>22.314386</td>
          <td>23.918189</td>
          <td>25.334986</td>
          <td>23.095449</td>
          <td>23.590106</td>
          <td>27.726421</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.850565</td>
          <td>0.005004</td>
          <td>25.011530</td>
          <td>0.326880</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.609931</td>
          <td>22.830471</td>
          <td>24.267241</td>
          <td>21.313660</td>
          <td>28.221857</td>
          <td>19.491990</td>
          <td>24.907161</td>
          <td>0.325746</td>
          <td>26.184556</td>
          <td>0.721602</td>
          <td>23.528004</td>
          <td>0.093270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.362773</td>
          <td>23.368922</td>
          <td>18.950705</td>
          <td>24.152540</td>
          <td>28.244525</td>
          <td>21.686352</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.061299</td>
          <td>0.663503</td>
          <td>27.298260</td>
          <td>1.473828</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.785893</td>
          <td>21.239032</td>
          <td>20.270558</td>
          <td>26.594319</td>
          <td>20.118460</td>
          <td>26.464066</td>
          <td>26.760618</td>
          <td>1.166710</td>
          <td>20.063328</td>
          <td>0.006181</td>
          <td>21.667062</td>
          <td>0.018073</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.056336</td>
          <td>24.226448</td>
          <td>25.467764</td>
          <td>20.986680</td>
          <td>22.442503</td>
          <td>22.885064</td>
          <td>25.086806</td>
          <td>0.375231</td>
          <td>26.069400</td>
          <td>0.667214</td>
          <td>22.362768</td>
          <td>0.033120</td>
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
          <td>22.889217</td>
          <td>27.655452</td>
          <td>21.635530</td>
          <td>21.553892</td>
          <td>25.829352</td>
          <td>28.113020</td>
          <td>24.616181</td>
          <td>0.257498</td>
          <td>16.291837</td>
          <td>0.005001</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.654618</td>
          <td>27.043359</td>
          <td>26.571251</td>
          <td>28.108379</td>
          <td>29.269918</td>
          <td>20.737710</td>
          <td>21.052177</td>
          <td>0.011936</td>
          <td>22.417472</td>
          <td>0.031817</td>
          <td>20.971401</td>
          <td>0.010454</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.684956</td>
          <td>19.508298</td>
          <td>25.575798</td>
          <td>23.887208</td>
          <td>21.249897</td>
          <td>23.322317</td>
          <td>25.122976</td>
          <td>0.385924</td>
          <td>24.428008</td>
          <td>0.186333</td>
          <td>23.058011</td>
          <td>0.061513</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.167783</td>
          <td>25.744580</td>
          <td>22.960786</td>
          <td>17.076286</td>
          <td>26.864794</td>
          <td>23.109582</td>
          <td>25.996389</td>
          <td>0.727363</td>
          <td>24.943330</td>
          <td>0.285605</td>
          <td>18.848127</td>
          <td>0.005167</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.329299</td>
          <td>25.030832</td>
          <td>24.387894</td>
          <td>23.502625</td>
          <td>17.687798</td>
          <td>20.820170</td>
          <td>22.276340</td>
          <td>0.033522</td>
          <td>24.920702</td>
          <td>0.280414</td>
          <td>29.865978</td>
          <td>3.753333</td>
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


