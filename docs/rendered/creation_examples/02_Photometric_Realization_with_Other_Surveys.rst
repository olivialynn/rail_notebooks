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
          <td>28.522625</td>
          <td>26.338024</td>
          <td>20.736221</td>
          <td>26.297769</td>
          <td>25.671805</td>
          <td>24.091897</td>
          <td>22.065767</td>
          <td>20.336647</td>
          <td>23.500018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.575895</td>
          <td>26.013860</td>
          <td>19.474280</td>
          <td>23.655452</td>
          <td>20.639281</td>
          <td>30.727814</td>
          <td>23.129973</td>
          <td>25.367319</td>
          <td>22.475324</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.391592</td>
          <td>15.431237</td>
          <td>24.452454</td>
          <td>18.705002</td>
          <td>21.349877</td>
          <td>19.915137</td>
          <td>23.467009</td>
          <td>21.333093</td>
          <td>20.685160</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.694545</td>
          <td>22.528010</td>
          <td>22.497405</td>
          <td>19.658562</td>
          <td>26.676082</td>
          <td>27.349325</td>
          <td>22.104115</td>
          <td>21.818271</td>
          <td>19.422047</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.069476</td>
          <td>24.164203</td>
          <td>21.820565</td>
          <td>31.565646</td>
          <td>21.611315</td>
          <td>24.124085</td>
          <td>22.614158</td>
          <td>21.570358</td>
          <td>25.999957</td>
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
          <td>27.021770</td>
          <td>24.521698</td>
          <td>21.962241</td>
          <td>18.161073</td>
          <td>24.524627</td>
          <td>21.582570</td>
          <td>25.728233</td>
          <td>25.901266</td>
          <td>22.714687</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.507392</td>
          <td>29.379004</td>
          <td>24.244758</td>
          <td>24.775103</td>
          <td>22.761251</td>
          <td>22.189591</td>
          <td>20.961813</td>
          <td>24.162619</td>
          <td>21.200264</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.647540</td>
          <td>23.060302</td>
          <td>24.798887</td>
          <td>24.290218</td>
          <td>25.309004</td>
          <td>24.322809</td>
          <td>26.742797</td>
          <td>21.610726</td>
          <td>25.181643</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.940306</td>
          <td>22.013269</td>
          <td>22.654803</td>
          <td>22.958196</td>
          <td>27.899190</td>
          <td>23.157549</td>
          <td>22.662444</td>
          <td>19.134427</td>
          <td>22.311583</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.820918</td>
          <td>22.145252</td>
          <td>21.085953</td>
          <td>21.101887</td>
          <td>23.185898</td>
          <td>25.428107</td>
          <td>23.247950</td>
          <td>22.155786</td>
          <td>25.693470</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.239926</td>
          <td>0.110537</td>
          <td>20.735210</td>
          <td>0.005066</td>
          <td>26.117106</td>
          <td>0.141646</td>
          <td>25.821729</td>
          <td>0.205487</td>
          <td>24.175361</td>
          <td>0.110192</td>
          <td>22.065767</td>
          <td>20.336647</td>
          <td>23.500018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.573995</td>
          <td>0.005051</td>
          <td>25.956184</td>
          <td>0.086217</td>
          <td>19.474831</td>
          <td>0.005012</td>
          <td>23.674202</td>
          <td>0.016748</td>
          <td>20.641344</td>
          <td>0.005426</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.129973</td>
          <td>25.367319</td>
          <td>22.475324</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.387699</td>
          <td>0.011907</td>
          <td>15.431289</td>
          <td>0.005000</td>
          <td>24.463076</td>
          <td>0.020398</td>
          <td>18.707414</td>
          <td>0.005009</td>
          <td>21.344904</td>
          <td>0.006311</td>
          <td>19.908269</td>
          <td>0.005576</td>
          <td>23.467009</td>
          <td>21.333093</td>
          <td>20.685160</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.678522</td>
          <td>0.005689</td>
          <td>22.531904</td>
          <td>0.006475</td>
          <td>22.498529</td>
          <td>0.006059</td>
          <td>19.660140</td>
          <td>0.005031</td>
          <td>26.574818</td>
          <td>0.378146</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.104115</td>
          <td>21.818271</td>
          <td>19.422047</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.067193</td>
          <td>0.005030</td>
          <td>24.205788</td>
          <td>0.018652</td>
          <td>21.809372</td>
          <td>0.005350</td>
          <td>28.319299</td>
          <td>0.783324</td>
          <td>21.622971</td>
          <td>0.007011</td>
          <td>24.283889</td>
          <td>0.121113</td>
          <td>22.614158</td>
          <td>21.570358</td>
          <td>25.999957</td>
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
          <td>29.614391</td>
          <td>2.297702</td>
          <td>24.502385</td>
          <td>0.023983</td>
          <td>21.965874</td>
          <td>0.005450</td>
          <td>18.154489</td>
          <td>0.005005</td>
          <td>24.578547</td>
          <td>0.069984</td>
          <td>21.574886</td>
          <td>0.011802</td>
          <td>25.728233</td>
          <td>25.901266</td>
          <td>22.714687</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.506800</td>
          <td>0.005149</td>
          <td>30.138137</td>
          <td>1.718900</td>
          <td>24.268051</td>
          <td>0.017317</td>
          <td>24.800604</td>
          <td>0.044473</td>
          <td>22.760964</td>
          <td>0.014484</td>
          <td>22.231095</td>
          <td>0.019958</td>
          <td>20.961813</td>
          <td>24.162619</td>
          <td>21.200264</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.645018</td>
          <td>0.014401</td>
          <td>23.055094</td>
          <td>0.008172</td>
          <td>24.815604</td>
          <td>0.027652</td>
          <td>24.289208</td>
          <td>0.028314</td>
          <td>25.234755</td>
          <td>0.124482</td>
          <td>24.207580</td>
          <td>0.113332</td>
          <td>26.742797</td>
          <td>21.610726</td>
          <td>25.181643</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.942068</td>
          <td>0.005257</td>
          <td>22.014761</td>
          <td>0.005668</td>
          <td>22.659812</td>
          <td>0.006365</td>
          <td>22.964445</td>
          <td>0.009803</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.139978</td>
          <td>0.044151</td>
          <td>22.662444</td>
          <td>19.134427</td>
          <td>22.311583</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.830743</td>
          <td>0.016654</td>
          <td>22.145467</td>
          <td>0.005817</td>
          <td>21.091884</td>
          <td>0.005113</td>
          <td>21.100564</td>
          <td>0.005271</td>
          <td>23.191356</td>
          <td>0.020664</td>
          <td>25.775811</td>
          <td>0.414890</td>
          <td>23.247950</td>
          <td>22.155786</td>
          <td>25.693470</td>
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
          <td>28.522625</td>
          <td>26.338024</td>
          <td>20.736221</td>
          <td>26.297769</td>
          <td>25.671805</td>
          <td>24.091897</td>
          <td>22.065456</td>
          <td>0.005503</td>
          <td>20.336209</td>
          <td>0.005066</td>
          <td>23.502355</td>
          <td>0.015753</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.575895</td>
          <td>26.013860</td>
          <td>19.474280</td>
          <td>23.655452</td>
          <td>20.639281</td>
          <td>30.727814</td>
          <td>23.122094</td>
          <td>0.007869</td>
          <td>25.322271</td>
          <td>0.077778</td>
          <td>22.467617</td>
          <td>0.007642</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.391592</td>
          <td>15.431237</td>
          <td>24.452454</td>
          <td>18.705002</td>
          <td>21.349877</td>
          <td>19.915137</td>
          <td>23.452075</td>
          <td>0.009628</td>
          <td>21.336170</td>
          <td>0.005400</td>
          <td>20.682084</td>
          <td>0.005123</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.694545</td>
          <td>22.528010</td>
          <td>22.497405</td>
          <td>19.658562</td>
          <td>26.676082</td>
          <td>27.349325</td>
          <td>22.107945</td>
          <td>0.005542</td>
          <td>21.818572</td>
          <td>0.005926</td>
          <td>19.409644</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.069476</td>
          <td>24.164203</td>
          <td>21.820565</td>
          <td>31.565646</td>
          <td>21.611315</td>
          <td>24.124085</td>
          <td>22.609523</td>
          <td>0.006275</td>
          <td>21.566301</td>
          <td>0.005600</td>
          <td>26.221115</td>
          <td>0.170170</td>
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
          <td>27.021770</td>
          <td>24.521698</td>
          <td>21.962241</td>
          <td>18.161073</td>
          <td>24.524627</td>
          <td>21.582570</td>
          <td>25.662037</td>
          <td>0.061733</td>
          <td>25.832388</td>
          <td>0.121748</td>
          <td>22.700182</td>
          <td>0.008730</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.507392</td>
          <td>29.379004</td>
          <td>24.244758</td>
          <td>24.775103</td>
          <td>22.761251</td>
          <td>22.189591</td>
          <td>20.957844</td>
          <td>0.005068</td>
          <td>24.124509</td>
          <td>0.026835</td>
          <td>21.197065</td>
          <td>0.005312</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.647540</td>
          <td>23.060302</td>
          <td>24.798887</td>
          <td>24.290218</td>
          <td>25.309004</td>
          <td>24.322809</td>
          <td>26.923073</td>
          <td>0.185557</td>
          <td>21.614460</td>
          <td>0.005652</td>
          <td>25.166756</td>
          <td>0.067757</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.940306</td>
          <td>22.013269</td>
          <td>22.654803</td>
          <td>22.958196</td>
          <td>27.899190</td>
          <td>23.157549</td>
          <td>22.654581</td>
          <td>0.006374</td>
          <td>19.135410</td>
          <td>0.005007</td>
          <td>22.318814</td>
          <td>0.007100</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.820918</td>
          <td>22.145252</td>
          <td>21.085953</td>
          <td>21.101887</td>
          <td>23.185898</td>
          <td>25.428107</td>
          <td>23.246830</td>
          <td>0.008452</td>
          <td>22.154051</td>
          <td>0.006615</td>
          <td>25.549443</td>
          <td>0.095047</td>
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
          <td>28.522625</td>
          <td>26.338024</td>
          <td>20.736221</td>
          <td>26.297769</td>
          <td>25.671805</td>
          <td>24.091897</td>
          <td>22.075422</td>
          <td>0.028064</td>
          <td>20.340521</td>
          <td>0.006855</td>
          <td>23.532043</td>
          <td>0.093603</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.575895</td>
          <td>26.013860</td>
          <td>19.474280</td>
          <td>23.655452</td>
          <td>20.639281</td>
          <td>30.727814</td>
          <td>23.023430</td>
          <td>0.065198</td>
          <td>25.182727</td>
          <td>0.345840</td>
          <td>22.508701</td>
          <td>0.037706</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.391592</td>
          <td>15.431237</td>
          <td>24.452454</td>
          <td>18.705002</td>
          <td>21.349877</td>
          <td>19.915137</td>
          <td>23.440390</td>
          <td>0.094293</td>
          <td>21.341569</td>
          <td>0.012782</td>
          <td>20.683898</td>
          <td>0.008643</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.694545</td>
          <td>22.528010</td>
          <td>22.497405</td>
          <td>19.658562</td>
          <td>26.676082</td>
          <td>27.349325</td>
          <td>22.112093</td>
          <td>0.028986</td>
          <td>21.796901</td>
          <td>0.018535</td>
          <td>19.417523</td>
          <td>0.005462</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.069476</td>
          <td>24.164203</td>
          <td>21.820565</td>
          <td>31.565646</td>
          <td>21.611315</td>
          <td>24.124085</td>
          <td>22.598965</td>
          <td>0.044670</td>
          <td>21.557269</td>
          <td>0.015181</td>
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
          <td>27.021770</td>
          <td>24.521698</td>
          <td>21.962241</td>
          <td>18.161073</td>
          <td>24.524627</td>
          <td>21.582570</td>
          <td>24.912650</td>
          <td>0.327172</td>
          <td>25.634402</td>
          <td>0.488779</td>
          <td>22.715835</td>
          <td>0.045347</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.507392</td>
          <td>29.379004</td>
          <td>24.244758</td>
          <td>24.775103</td>
          <td>22.761251</td>
          <td>22.189591</td>
          <td>20.937950</td>
          <td>0.010966</td>
          <td>24.069331</td>
          <td>0.137099</td>
          <td>21.201473</td>
          <td>0.012392</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.647540</td>
          <td>23.060302</td>
          <td>24.798887</td>
          <td>24.290218</td>
          <td>25.309004</td>
          <td>24.322809</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.606718</td>
          <td>0.015810</td>
          <td>24.890719</td>
          <td>0.296748</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.940306</td>
          <td>22.013269</td>
          <td>22.654803</td>
          <td>22.958196</td>
          <td>27.899190</td>
          <td>23.157549</td>
          <td>22.687357</td>
          <td>0.048332</td>
          <td>19.142412</td>
          <td>0.005237</td>
          <td>22.256181</td>
          <td>0.030137</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.820918</td>
          <td>22.145252</td>
          <td>21.085953</td>
          <td>21.101887</td>
          <td>23.185898</td>
          <td>25.428107</td>
          <td>23.271895</td>
          <td>0.081269</td>
          <td>22.134242</td>
          <td>0.024793</td>
          <td>25.335388</td>
          <td>0.420784</td>
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


