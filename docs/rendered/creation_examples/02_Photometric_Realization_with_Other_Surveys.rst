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
          <td>20.536547</td>
          <td>19.386917</td>
          <td>24.423301</td>
          <td>24.967896</td>
          <td>31.407635</td>
          <td>25.834151</td>
          <td>27.406322</td>
          <td>23.318261</td>
          <td>21.679702</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.500018</td>
          <td>24.194987</td>
          <td>22.958482</td>
          <td>25.454751</td>
          <td>20.670439</td>
          <td>22.445985</td>
          <td>21.779120</td>
          <td>19.877803</td>
          <td>23.576679</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.558897</td>
          <td>18.925609</td>
          <td>23.121934</td>
          <td>20.311302</td>
          <td>19.899787</td>
          <td>20.692847</td>
          <td>26.416701</td>
          <td>17.778556</td>
          <td>23.163218</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974673</td>
          <td>24.525313</td>
          <td>24.855303</td>
          <td>21.628462</td>
          <td>19.192434</td>
          <td>21.212933</td>
          <td>20.855145</td>
          <td>22.669745</td>
          <td>27.570657</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.674868</td>
          <td>21.184521</td>
          <td>21.526874</td>
          <td>27.032905</td>
          <td>20.441547</td>
          <td>26.587209</td>
          <td>27.504137</td>
          <td>25.416446</td>
          <td>22.366543</td>
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
          <td>20.992660</td>
          <td>27.901994</td>
          <td>27.819755</td>
          <td>20.061409</td>
          <td>24.524917</td>
          <td>23.938294</td>
          <td>26.605777</td>
          <td>19.433443</td>
          <td>20.869670</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.756448</td>
          <td>18.844976</td>
          <td>22.151015</td>
          <td>19.496354</td>
          <td>21.767294</td>
          <td>24.051182</td>
          <td>21.638757</td>
          <td>26.049972</td>
          <td>19.313266</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.838745</td>
          <td>21.797377</td>
          <td>16.133118</td>
          <td>24.596726</td>
          <td>22.383507</td>
          <td>21.010163</td>
          <td>22.588947</td>
          <td>14.279639</td>
          <td>23.256441</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.657342</td>
          <td>23.638850</td>
          <td>14.814851</td>
          <td>21.630776</td>
          <td>18.990758</td>
          <td>23.799610</td>
          <td>21.370704</td>
          <td>16.278525</td>
          <td>23.703335</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.890881</td>
          <td>21.705576</td>
          <td>25.113984</td>
          <td>32.886671</td>
          <td>22.162260</td>
          <td>24.429457</td>
          <td>19.523615</td>
          <td>20.317448</td>
          <td>19.569962</td>
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
          <td>20.536670</td>
          <td>0.005567</td>
          <td>19.397480</td>
          <td>0.005019</td>
          <td>24.407427</td>
          <td>0.019459</td>
          <td>25.022678</td>
          <td>0.054166</td>
          <td>26.716599</td>
          <td>0.421774</td>
          <td>25.813291</td>
          <td>0.426928</td>
          <td>27.406322</td>
          <td>23.318261</td>
          <td>21.679702</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.473590</td>
          <td>0.028469</td>
          <td>24.201032</td>
          <td>0.018578</td>
          <td>22.952198</td>
          <td>0.007138</td>
          <td>25.517748</td>
          <td>0.083965</td>
          <td>20.663733</td>
          <td>0.005442</td>
          <td>22.419771</td>
          <td>0.023453</td>
          <td>21.779120</td>
          <td>19.877803</td>
          <td>23.576679</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.637305</td>
          <td>0.078861</td>
          <td>18.920542</td>
          <td>0.005011</td>
          <td>23.123518</td>
          <td>0.007757</td>
          <td>20.317450</td>
          <td>0.005080</td>
          <td>19.899970</td>
          <td>0.005132</td>
          <td>20.702615</td>
          <td>0.007009</td>
          <td>26.416701</td>
          <td>17.778556</td>
          <td>23.163218</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.964478</td>
          <td>0.005264</td>
          <td>24.497730</td>
          <td>0.023888</td>
          <td>24.869161</td>
          <td>0.028979</td>
          <td>21.629892</td>
          <td>0.005638</td>
          <td>19.194562</td>
          <td>0.005046</td>
          <td>21.212207</td>
          <td>0.009200</td>
          <td>20.855145</td>
          <td>22.669745</td>
          <td>27.570657</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.615424</td>
          <td>0.183590</td>
          <td>21.182315</td>
          <td>0.005189</td>
          <td>21.525834</td>
          <td>0.005222</td>
          <td>27.016235</td>
          <td>0.300287</td>
          <td>20.439740</td>
          <td>0.005309</td>
          <td>27.458459</td>
          <td>1.259948</td>
          <td>27.504137</td>
          <td>25.416446</td>
          <td>22.366543</td>
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
          <td>20.996133</td>
          <td>0.006073</td>
          <td>28.073540</td>
          <td>0.495444</td>
          <td>28.252230</td>
          <td>0.508343</td>
          <td>20.062322</td>
          <td>0.005055</td>
          <td>24.632587</td>
          <td>0.073411</td>
          <td>24.005275</td>
          <td>0.094951</td>
          <td>26.605777</td>
          <td>19.433443</td>
          <td>20.869670</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.779579</td>
          <td>0.015991</td>
          <td>18.841346</td>
          <td>0.005010</td>
          <td>22.156655</td>
          <td>0.005612</td>
          <td>19.499788</td>
          <td>0.005025</td>
          <td>21.766277</td>
          <td>0.007491</td>
          <td>24.124160</td>
          <td>0.105373</td>
          <td>21.638757</td>
          <td>26.049972</td>
          <td>19.313266</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.847741</td>
          <td>0.005872</td>
          <td>21.801617</td>
          <td>0.005481</td>
          <td>16.118414</td>
          <td>0.005000</td>
          <td>24.619160</td>
          <td>0.037865</td>
          <td>22.392557</td>
          <td>0.010963</td>
          <td>21.007316</td>
          <td>0.008147</td>
          <td>22.588947</td>
          <td>14.279639</td>
          <td>23.256441</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.586051</td>
          <td>0.031371</td>
          <td>23.641839</td>
          <td>0.011948</td>
          <td>14.814023</td>
          <td>0.005000</td>
          <td>21.626525</td>
          <td>0.005634</td>
          <td>18.990433</td>
          <td>0.005035</td>
          <td>23.890280</td>
          <td>0.085820</td>
          <td>21.370704</td>
          <td>16.278525</td>
          <td>23.703335</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.573861</td>
          <td>0.813350</td>
          <td>21.713399</td>
          <td>0.005420</td>
          <td>25.115781</td>
          <td>0.036007</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.152880</td>
          <td>0.009324</td>
          <td>24.562307</td>
          <td>0.154007</td>
          <td>19.523615</td>
          <td>20.317448</td>
          <td>19.569962</td>
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
          <td>20.536547</td>
          <td>19.386917</td>
          <td>24.423301</td>
          <td>24.967896</td>
          <td>31.407635</td>
          <td>25.834151</td>
          <td>27.634470</td>
          <td>0.332890</td>
          <td>23.309832</td>
          <td>0.013485</td>
          <td>21.687676</td>
          <td>0.005740</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.500018</td>
          <td>24.194987</td>
          <td>22.958482</td>
          <td>25.454751</td>
          <td>20.670439</td>
          <td>22.445985</td>
          <td>21.772394</td>
          <td>0.005299</td>
          <td>19.880581</td>
          <td>0.005028</td>
          <td>23.566652</td>
          <td>0.016614</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.558897</td>
          <td>18.925609</td>
          <td>23.121934</td>
          <td>20.311302</td>
          <td>19.899787</td>
          <td>20.692847</td>
          <td>26.364390</td>
          <td>0.114745</td>
          <td>17.768623</td>
          <td>0.005001</td>
          <td>23.168737</td>
          <td>0.012086</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974673</td>
          <td>24.525313</td>
          <td>24.855303</td>
          <td>21.628462</td>
          <td>19.192434</td>
          <td>21.212933</td>
          <td>20.852635</td>
          <td>0.005056</td>
          <td>22.678994</td>
          <td>0.008618</td>
          <td>27.872102</td>
          <td>0.623626</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.674868</td>
          <td>21.184521</td>
          <td>21.526874</td>
          <td>27.032905</td>
          <td>20.441547</td>
          <td>26.587209</td>
          <td>27.613885</td>
          <td>0.327493</td>
          <td>25.373359</td>
          <td>0.081374</td>
          <td>22.370627</td>
          <td>0.007276</td>
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
          <td>20.992660</td>
          <td>27.901994</td>
          <td>27.819755</td>
          <td>20.061409</td>
          <td>24.524917</td>
          <td>23.938294</td>
          <td>26.636027</td>
          <td>0.145219</td>
          <td>19.442106</td>
          <td>0.005013</td>
          <td>20.874646</td>
          <td>0.005175</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.756448</td>
          <td>18.844976</td>
          <td>22.151015</td>
          <td>19.496354</td>
          <td>21.767294</td>
          <td>24.051182</td>
          <td>21.636686</td>
          <td>0.005234</td>
          <td>26.120537</td>
          <td>0.156155</td>
          <td>19.307711</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.838745</td>
          <td>21.797377</td>
          <td>16.133118</td>
          <td>24.596726</td>
          <td>22.383507</td>
          <td>21.010163</td>
          <td>22.582990</td>
          <td>0.006220</td>
          <td>14.284263</td>
          <td>0.005000</td>
          <td>23.258644</td>
          <td>0.012953</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.657342</td>
          <td>23.638850</td>
          <td>14.814851</td>
          <td>21.630776</td>
          <td>18.990758</td>
          <td>23.799610</td>
          <td>21.366784</td>
          <td>0.005144</td>
          <td>16.279871</td>
          <td>0.005000</td>
          <td>23.710116</td>
          <td>0.018744</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.890881</td>
          <td>21.705576</td>
          <td>25.113984</td>
          <td>32.886671</td>
          <td>22.162260</td>
          <td>24.429457</td>
          <td>19.520585</td>
          <td>0.005005</td>
          <td>20.322060</td>
          <td>0.005064</td>
          <td>19.578215</td>
          <td>0.005016</td>
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
          <td>20.536547</td>
          <td>19.386917</td>
          <td>24.423301</td>
          <td>24.967896</td>
          <td>31.407635</td>
          <td>25.834151</td>
          <td>25.794720</td>
          <td>0.633566</td>
          <td>23.428334</td>
          <td>0.078196</td>
          <td>21.690310</td>
          <td>0.018432</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.500018</td>
          <td>24.194987</td>
          <td>22.958482</td>
          <td>25.454751</td>
          <td>20.670439</td>
          <td>22.445985</td>
          <td>21.800178</td>
          <td>0.022063</td>
          <td>19.869701</td>
          <td>0.005852</td>
          <td>23.559132</td>
          <td>0.095860</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.558897</td>
          <td>18.925609</td>
          <td>23.121934</td>
          <td>20.311302</td>
          <td>19.899787</td>
          <td>20.692847</td>
          <td>25.576151</td>
          <td>0.542321</td>
          <td>17.783103</td>
          <td>0.005020</td>
          <td>23.176226</td>
          <td>0.068329</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974673</td>
          <td>24.525313</td>
          <td>24.855303</td>
          <td>21.628462</td>
          <td>19.192434</td>
          <td>21.212933</td>
          <td>20.851412</td>
          <td>0.010308</td>
          <td>22.674956</td>
          <td>0.039997</td>
          <td>28.291678</td>
          <td>2.285344</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.674868</td>
          <td>21.184521</td>
          <td>21.526874</td>
          <td>27.032905</td>
          <td>20.441547</td>
          <td>26.587209</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.414497</td>
          <td>0.034677</td>
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
          <td>20.992660</td>
          <td>27.901994</td>
          <td>27.819755</td>
          <td>20.061409</td>
          <td>24.524917</td>
          <td>23.938294</td>
          <td>26.273240</td>
          <td>0.871347</td>
          <td>19.436137</td>
          <td>0.005400</td>
          <td>20.861083</td>
          <td>0.009687</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.756448</td>
          <td>18.844976</td>
          <td>22.151015</td>
          <td>19.496354</td>
          <td>21.767294</td>
          <td>24.051182</td>
          <td>21.643086</td>
          <td>0.019277</td>
          <td>27.677946</td>
          <td>1.688138</td>
          <td>19.309908</td>
          <td>0.005382</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.838745</td>
          <td>21.797377</td>
          <td>16.133118</td>
          <td>24.596726</td>
          <td>22.383507</td>
          <td>21.010163</td>
          <td>22.507187</td>
          <td>0.041162</td>
          <td>14.280576</td>
          <td>0.005000</td>
          <td>23.383861</td>
          <td>0.082133</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.657342</td>
          <td>23.638850</td>
          <td>14.814851</td>
          <td>21.630776</td>
          <td>18.990758</td>
          <td>23.799610</td>
          <td>21.344591</td>
          <td>0.015025</td>
          <td>16.271547</td>
          <td>0.005001</td>
          <td>23.916113</td>
          <td>0.130928</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.890881</td>
          <td>21.705576</td>
          <td>25.113984</td>
          <td>32.886671</td>
          <td>22.162260</td>
          <td>24.429457</td>
          <td>19.527561</td>
          <td>0.005667</td>
          <td>20.321331</td>
          <td>0.006799</td>
          <td>19.572430</td>
          <td>0.005606</td>
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


