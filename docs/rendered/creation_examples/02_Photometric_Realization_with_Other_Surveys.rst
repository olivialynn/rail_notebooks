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
          <td>25.975524</td>
          <td>22.609756</td>
          <td>24.934426</td>
          <td>20.964225</td>
          <td>22.576640</td>
          <td>22.832098</td>
          <td>26.859503</td>
          <td>21.805204</td>
          <td>24.671345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.898933</td>
          <td>23.861113</td>
          <td>26.101992</td>
          <td>25.525488</td>
          <td>21.353894</td>
          <td>20.961787</td>
          <td>23.092095</td>
          <td>22.226541</td>
          <td>21.165723</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.540568</td>
          <td>22.803572</td>
          <td>22.750604</td>
          <td>22.495098</td>
          <td>22.318806</td>
          <td>17.438220</td>
          <td>21.452139</td>
          <td>22.970291</td>
          <td>25.640954</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.306566</td>
          <td>18.814049</td>
          <td>26.948970</td>
          <td>29.055743</td>
          <td>21.850565</td>
          <td>23.736675</td>
          <td>16.564763</td>
          <td>19.884128</td>
          <td>21.475308</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.786321</td>
          <td>23.406416</td>
          <td>26.116506</td>
          <td>19.288875</td>
          <td>17.778086</td>
          <td>26.526585</td>
          <td>22.046267</td>
          <td>26.573012</td>
          <td>25.350673</td>
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
          <td>22.933028</td>
          <td>21.890460</td>
          <td>18.926011</td>
          <td>24.328620</td>
          <td>26.683826</td>
          <td>25.711280</td>
          <td>28.297633</td>
          <td>25.820913</td>
          <td>24.747830</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.392005</td>
          <td>24.561671</td>
          <td>23.172025</td>
          <td>25.184136</td>
          <td>23.614814</td>
          <td>21.082477</td>
          <td>24.074512</td>
          <td>20.410931</td>
          <td>26.614218</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.152154</td>
          <td>25.683583</td>
          <td>25.688723</td>
          <td>26.337886</td>
          <td>24.512198</td>
          <td>23.926864</td>
          <td>21.363921</td>
          <td>20.342514</td>
          <td>26.073124</td>
        </tr>
        <tr>
          <th>998</th>
          <td>30.114979</td>
          <td>21.191929</td>
          <td>25.296891</td>
          <td>23.037485</td>
          <td>26.754999</td>
          <td>22.019873</td>
          <td>25.337298</td>
          <td>17.895186</td>
          <td>31.156540</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.861157</td>
          <td>28.946503</td>
          <td>26.676757</td>
          <td>26.957499</td>
          <td>23.117194</td>
          <td>20.914804</td>
          <td>22.975625</td>
          <td>20.900295</td>
          <td>17.998301</td>
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
          <td>25.720999</td>
          <td>0.200635</td>
          <td>22.609318</td>
          <td>0.006657</td>
          <td>24.850045</td>
          <td>0.028498</td>
          <td>20.961851</td>
          <td>0.005218</td>
          <td>22.576453</td>
          <td>0.012550</td>
          <td>22.809684</td>
          <td>0.032964</td>
          <td>26.859503</td>
          <td>21.805204</td>
          <td>24.671345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.887541</td>
          <td>0.005922</td>
          <td>23.866851</td>
          <td>0.014175</td>
          <td>26.138541</td>
          <td>0.089086</td>
          <td>25.433751</td>
          <td>0.077968</td>
          <td>21.345860</td>
          <td>0.006313</td>
          <td>20.958134</td>
          <td>0.007932</td>
          <td>23.092095</td>
          <td>22.226541</td>
          <td>21.165723</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.587691</td>
          <td>0.075500</td>
          <td>22.802801</td>
          <td>0.007207</td>
          <td>22.748621</td>
          <td>0.006567</td>
          <td>22.503308</td>
          <td>0.007497</td>
          <td>22.307099</td>
          <td>0.010327</td>
          <td>17.431437</td>
          <td>0.005015</td>
          <td>21.452139</td>
          <td>22.970291</td>
          <td>25.640954</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.290438</td>
          <td>0.011126</td>
          <td>18.816066</td>
          <td>0.005010</td>
          <td>26.825564</td>
          <td>0.161771</td>
          <td>30.387768</td>
          <td>2.271916</td>
          <td>21.843531</td>
          <td>0.007790</td>
          <td>23.690148</td>
          <td>0.071922</td>
          <td>16.564763</td>
          <td>19.884128</td>
          <td>21.475308</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.780013</td>
          <td>0.005209</td>
          <td>23.411072</td>
          <td>0.010157</td>
          <td>26.026540</td>
          <td>0.080715</td>
          <td>19.279766</td>
          <td>0.005019</td>
          <td>17.772175</td>
          <td>0.005008</td>
          <td>25.937464</td>
          <td>0.468857</td>
          <td>22.046267</td>
          <td>26.573012</td>
          <td>25.350673</td>
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
          <td>22.960708</td>
          <td>0.018495</td>
          <td>21.898960</td>
          <td>0.005559</td>
          <td>18.933555</td>
          <td>0.005006</td>
          <td>24.311510</td>
          <td>0.028873</td>
          <td>26.391608</td>
          <td>0.327428</td>
          <td>24.986870</td>
          <td>0.220494</td>
          <td>28.297633</td>
          <td>25.820913</td>
          <td>24.747830</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.396822</td>
          <td>0.026654</td>
          <td>24.516253</td>
          <td>0.024271</td>
          <td>23.174084</td>
          <td>0.007968</td>
          <td>25.208413</td>
          <td>0.063871</td>
          <td>23.629412</td>
          <td>0.030199</td>
          <td>21.068916</td>
          <td>0.008437</td>
          <td>24.074512</td>
          <td>20.410931</td>
          <td>26.614218</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.150579</td>
          <td>0.005337</td>
          <td>25.773906</td>
          <td>0.073424</td>
          <td>25.788304</td>
          <td>0.065380</td>
          <td>26.136513</td>
          <td>0.144033</td>
          <td>24.549358</td>
          <td>0.068199</td>
          <td>23.805024</td>
          <td>0.079606</td>
          <td>21.363921</td>
          <td>20.342514</td>
          <td>26.073124</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>21.194613</td>
          <td>0.005193</td>
          <td>25.205510</td>
          <td>0.038983</td>
          <td>23.055161</td>
          <td>0.010425</td>
          <td>26.576378</td>
          <td>0.378605</td>
          <td>22.016669</td>
          <td>0.016687</td>
          <td>25.337298</td>
          <td>17.895186</td>
          <td>31.156540</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.855113</td>
          <td>0.005024</td>
          <td>28.920145</td>
          <td>0.885629</td>
          <td>26.648195</td>
          <td>0.138927</td>
          <td>26.915131</td>
          <td>0.276725</td>
          <td>23.110849</td>
          <td>0.019302</td>
          <td>20.910125</td>
          <td>0.007734</td>
          <td>22.975625</td>
          <td>20.900295</td>
          <td>17.998301</td>
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
          <td>25.975524</td>
          <td>22.609756</td>
          <td>24.934426</td>
          <td>20.964225</td>
          <td>22.576640</td>
          <td>22.832098</td>
          <td>27.069330</td>
          <td>0.209865</td>
          <td>21.801079</td>
          <td>0.005899</td>
          <td>24.701151</td>
          <td>0.044757</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.898933</td>
          <td>23.861113</td>
          <td>26.101992</td>
          <td>25.525488</td>
          <td>21.353894</td>
          <td>20.961787</td>
          <td>23.095734</td>
          <td>0.007757</td>
          <td>22.230462</td>
          <td>0.006826</td>
          <td>21.164148</td>
          <td>0.005295</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.540568</td>
          <td>22.803572</td>
          <td>22.750604</td>
          <td>22.495098</td>
          <td>22.318806</td>
          <td>17.438220</td>
          <td>21.453801</td>
          <td>0.005168</td>
          <td>22.957774</td>
          <td>0.010354</td>
          <td>25.620520</td>
          <td>0.101171</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.306566</td>
          <td>18.814049</td>
          <td>26.948970</td>
          <td>29.055743</td>
          <td>21.850565</td>
          <td>23.736675</td>
          <td>16.561792</td>
          <td>0.005000</td>
          <td>19.882430</td>
          <td>0.005029</td>
          <td>21.480113</td>
          <td>0.005516</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.786321</td>
          <td>23.406416</td>
          <td>26.116506</td>
          <td>19.288875</td>
          <td>17.778086</td>
          <td>26.526585</td>
          <td>22.042565</td>
          <td>0.005483</td>
          <td>26.309274</td>
          <td>0.183402</td>
          <td>25.412294</td>
          <td>0.084223</td>
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
          <td>22.933028</td>
          <td>21.890460</td>
          <td>18.926011</td>
          <td>24.328620</td>
          <td>26.683826</td>
          <td>25.711280</td>
          <td>28.231797</td>
          <td>0.525099</td>
          <td>25.860264</td>
          <td>0.124735</td>
          <td>24.652845</td>
          <td>0.042871</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.392005</td>
          <td>24.561671</td>
          <td>23.172025</td>
          <td>25.184136</td>
          <td>23.614814</td>
          <td>21.082477</td>
          <td>24.086100</td>
          <td>0.015544</td>
          <td>20.404922</td>
          <td>0.005074</td>
          <td>26.676950</td>
          <td>0.249335</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.152154</td>
          <td>25.683583</td>
          <td>25.688723</td>
          <td>26.337886</td>
          <td>24.512198</td>
          <td>23.926864</td>
          <td>21.352932</td>
          <td>0.005140</td>
          <td>20.346019</td>
          <td>0.005067</td>
          <td>26.011844</td>
          <td>0.142224</td>
        </tr>
        <tr>
          <th>998</th>
          <td>30.114979</td>
          <td>21.191929</td>
          <td>25.296891</td>
          <td>23.037485</td>
          <td>26.754999</td>
          <td>22.019873</td>
          <td>25.318371</td>
          <td>0.045449</td>
          <td>17.888884</td>
          <td>0.005001</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.861157</td>
          <td>28.946503</td>
          <td>26.676757</td>
          <td>26.957499</td>
          <td>23.117194</td>
          <td>20.914804</td>
          <td>22.980384</td>
          <td>0.007311</td>
          <td>20.901639</td>
          <td>0.005184</td>
          <td>18.002625</td>
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
          <td>25.975524</td>
          <td>22.609756</td>
          <td>24.934426</td>
          <td>20.964225</td>
          <td>22.576640</td>
          <td>22.832098</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.793945</td>
          <td>0.018489</td>
          <td>24.842515</td>
          <td>0.285416</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.898933</td>
          <td>23.861113</td>
          <td>26.101992</td>
          <td>25.525488</td>
          <td>21.353894</td>
          <td>20.961787</td>
          <td>23.175008</td>
          <td>0.074588</td>
          <td>22.233612</td>
          <td>0.027050</td>
          <td>21.160811</td>
          <td>0.012014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.540568</td>
          <td>22.803572</td>
          <td>22.750604</td>
          <td>22.495098</td>
          <td>22.318806</td>
          <td>17.438220</td>
          <td>21.420017</td>
          <td>0.015984</td>
          <td>22.939309</td>
          <td>0.050624</td>
          <td>25.244119</td>
          <td>0.392291</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.306566</td>
          <td>18.814049</td>
          <td>26.948970</td>
          <td>29.055743</td>
          <td>21.850565</td>
          <td>23.736675</td>
          <td>16.568812</td>
          <td>0.005003</td>
          <td>19.884468</td>
          <td>0.005874</td>
          <td>21.486990</td>
          <td>0.015555</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.786321</td>
          <td>23.406416</td>
          <td>26.116506</td>
          <td>19.288875</td>
          <td>17.778086</td>
          <td>26.526585</td>
          <td>22.074920</td>
          <td>0.028052</td>
          <td>26.568769</td>
          <td>0.925096</td>
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
          <td>22.933028</td>
          <td>21.890460</td>
          <td>18.926011</td>
          <td>24.328620</td>
          <td>26.683826</td>
          <td>25.711280</td>
          <td>25.813796</td>
          <td>0.642040</td>
          <td>26.046127</td>
          <td>0.656592</td>
          <td>24.324448</td>
          <td>0.185773</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.392005</td>
          <td>24.561671</td>
          <td>23.172025</td>
          <td>25.184136</td>
          <td>23.614814</td>
          <td>21.082477</td>
          <td>24.138895</td>
          <td>0.172765</td>
          <td>20.408511</td>
          <td>0.007066</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.152154</td>
          <td>25.683583</td>
          <td>25.688723</td>
          <td>26.337886</td>
          <td>24.512198</td>
          <td>23.926864</td>
          <td>21.353701</td>
          <td>0.015137</td>
          <td>20.328268</td>
          <td>0.006819</td>
          <td>26.655249</td>
          <td>1.035913</td>
        </tr>
        <tr>
          <th>998</th>
          <td>30.114979</td>
          <td>21.191929</td>
          <td>25.296891</td>
          <td>23.037485</td>
          <td>26.754999</td>
          <td>22.019873</td>
          <td>25.273407</td>
          <td>0.433143</td>
          <td>17.892397</td>
          <td>0.005024</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.861157</td>
          <td>28.946503</td>
          <td>26.676757</td>
          <td>26.957499</td>
          <td>23.117194</td>
          <td>20.914804</td>
          <td>23.027926</td>
          <td>0.065459</td>
          <td>20.907874</td>
          <td>0.009350</td>
          <td>17.992344</td>
          <td>0.005035</td>
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


