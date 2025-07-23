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
          <td>29.346576</td>
          <td>23.649215</td>
          <td>24.574980</td>
          <td>23.445548</td>
          <td>19.791261</td>
          <td>25.680658</td>
          <td>23.585339</td>
          <td>23.269835</td>
          <td>21.326247</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.906479</td>
          <td>29.347745</td>
          <td>25.016563</td>
          <td>23.050146</td>
          <td>25.619179</td>
          <td>21.925930</td>
          <td>25.496960</td>
          <td>23.705493</td>
          <td>23.277157</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.789419</td>
          <td>27.766131</td>
          <td>23.060602</td>
          <td>25.430301</td>
          <td>21.049031</td>
          <td>25.595927</td>
          <td>26.561720</td>
          <td>23.594009</td>
          <td>25.085786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.494070</td>
          <td>23.022909</td>
          <td>26.057029</td>
          <td>20.951936</td>
          <td>21.863378</td>
          <td>22.192699</td>
          <td>25.024226</td>
          <td>19.417722</td>
          <td>22.493983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.964165</td>
          <td>22.217297</td>
          <td>20.337681</td>
          <td>25.814386</td>
          <td>18.133076</td>
          <td>21.842137</td>
          <td>20.655375</td>
          <td>18.066991</td>
          <td>20.639955</td>
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
          <td>19.527282</td>
          <td>17.143296</td>
          <td>16.325678</td>
          <td>24.087456</td>
          <td>27.024126</td>
          <td>25.443227</td>
          <td>22.649929</td>
          <td>21.444017</td>
          <td>18.429813</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.973790</td>
          <td>22.651915</td>
          <td>17.641324</td>
          <td>19.160299</td>
          <td>19.948362</td>
          <td>19.983643</td>
          <td>29.365495</td>
          <td>23.721571</td>
          <td>23.676725</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.888084</td>
          <td>21.472701</td>
          <td>25.409300</td>
          <td>25.506502</td>
          <td>26.038964</td>
          <td>23.844242</td>
          <td>21.750600</td>
          <td>22.489829</td>
          <td>25.978701</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.636115</td>
          <td>18.956715</td>
          <td>22.324651</td>
          <td>22.897488</td>
          <td>25.617307</td>
          <td>25.827263</td>
          <td>24.845611</td>
          <td>24.425539</td>
          <td>24.637045</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.902736</td>
          <td>24.991641</td>
          <td>26.289874</td>
          <td>20.476752</td>
          <td>26.623450</td>
          <td>19.655433</td>
          <td>20.777116</td>
          <td>20.480291</td>
          <td>22.514159</td>
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
          <td>27.696252</td>
          <td>0.879554</td>
          <td>23.627133</td>
          <td>0.011820</td>
          <td>24.544488</td>
          <td>0.021866</td>
          <td>23.462025</td>
          <td>0.014112</td>
          <td>19.793207</td>
          <td>0.005112</td>
          <td>25.042223</td>
          <td>0.230868</td>
          <td>23.585339</td>
          <td>23.269835</td>
          <td>21.326247</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.906306</td>
          <td>0.005074</td>
          <td>29.517949</td>
          <td>1.258203</td>
          <td>25.017517</td>
          <td>0.033015</td>
          <td>23.043932</td>
          <td>0.010344</td>
          <td>25.559831</td>
          <td>0.164664</td>
          <td>21.934424</td>
          <td>0.015604</td>
          <td>25.496960</td>
          <td>23.705493</td>
          <td>23.277157</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.807640</td>
          <td>0.008287</td>
          <td>28.091837</td>
          <td>0.502180</td>
          <td>23.065237</td>
          <td>0.007531</td>
          <td>25.445892</td>
          <td>0.078808</td>
          <td>21.042263</td>
          <td>0.005812</td>
          <td>26.905434</td>
          <td>0.912246</td>
          <td>26.561720</td>
          <td>23.594009</td>
          <td>25.085786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.479343</td>
          <td>0.012720</td>
          <td>23.020373</td>
          <td>0.008020</td>
          <td>25.946649</td>
          <td>0.075217</td>
          <td>20.956254</td>
          <td>0.005216</td>
          <td>21.865474</td>
          <td>0.007880</td>
          <td>22.193940</td>
          <td>0.019342</td>
          <td>25.024226</td>
          <td>19.417722</td>
          <td>22.493983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.975625</td>
          <td>0.018722</td>
          <td>22.220328</td>
          <td>0.005917</td>
          <td>20.337070</td>
          <td>0.005037</td>
          <td>26.027082</td>
          <td>0.131054</td>
          <td>18.132842</td>
          <td>0.005012</td>
          <td>21.825440</td>
          <td>0.014300</td>
          <td>20.655375</td>
          <td>18.066991</td>
          <td>20.639955</td>
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
          <td>19.529978</td>
          <td>0.005153</td>
          <td>17.141006</td>
          <td>0.005002</td>
          <td>16.326029</td>
          <td>0.005000</td>
          <td>24.033635</td>
          <td>0.022679</td>
          <td>26.691099</td>
          <td>0.413635</td>
          <td>25.346898</td>
          <td>0.296158</td>
          <td>22.649929</td>
          <td>21.444017</td>
          <td>18.429813</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.956629</td>
          <td>0.009002</td>
          <td>22.669449</td>
          <td>0.006813</td>
          <td>17.641275</td>
          <td>0.005002</td>
          <td>19.160627</td>
          <td>0.005016</td>
          <td>19.944745</td>
          <td>0.005141</td>
          <td>19.985068</td>
          <td>0.005652</td>
          <td>29.365495</td>
          <td>23.721571</td>
          <td>23.676725</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.886352</td>
          <td>0.005072</td>
          <td>21.478831</td>
          <td>0.005294</td>
          <td>25.407854</td>
          <td>0.046645</td>
          <td>25.579780</td>
          <td>0.088680</td>
          <td>26.245420</td>
          <td>0.291250</td>
          <td>23.856558</td>
          <td>0.083308</td>
          <td>21.750600</td>
          <td>22.489829</td>
          <td>25.978701</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.636306</td>
          <td>0.005650</td>
          <td>18.951368</td>
          <td>0.005011</td>
          <td>22.318037</td>
          <td>0.005794</td>
          <td>22.907876</td>
          <td>0.009447</td>
          <td>25.695907</td>
          <td>0.184837</td>
          <td>25.426123</td>
          <td>0.315587</td>
          <td>24.845611</td>
          <td>24.425539</td>
          <td>24.637045</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.873306</td>
          <td>0.017231</td>
          <td>25.005809</td>
          <td>0.037217</td>
          <td>26.159145</td>
          <td>0.090716</td>
          <td>20.468098</td>
          <td>0.005101</td>
          <td>27.033206</td>
          <td>0.534040</td>
          <td>19.661554</td>
          <td>0.005388</td>
          <td>20.777116</td>
          <td>20.480291</td>
          <td>22.514159</td>
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
          <td>29.346576</td>
          <td>23.649215</td>
          <td>24.574980</td>
          <td>23.445548</td>
          <td>19.791261</td>
          <td>25.680658</td>
          <td>23.578564</td>
          <td>0.010508</td>
          <td>23.287566</td>
          <td>0.013250</td>
          <td>21.316964</td>
          <td>0.005387</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.906479</td>
          <td>29.347745</td>
          <td>25.016563</td>
          <td>23.050146</td>
          <td>25.619179</td>
          <td>21.925930</td>
          <td>25.487909</td>
          <td>0.052865</td>
          <td>23.701128</td>
          <td>0.018602</td>
          <td>23.278839</td>
          <td>0.013159</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.789419</td>
          <td>27.766131</td>
          <td>23.060602</td>
          <td>25.430301</td>
          <td>21.049031</td>
          <td>25.595927</td>
          <td>26.422728</td>
          <td>0.120729</td>
          <td>23.608572</td>
          <td>0.017206</td>
          <td>25.053332</td>
          <td>0.061257</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.494070</td>
          <td>23.022909</td>
          <td>26.057029</td>
          <td>20.951936</td>
          <td>21.863378</td>
          <td>22.192699</td>
          <td>25.066989</td>
          <td>0.036333</td>
          <td>19.428085</td>
          <td>0.005012</td>
          <td>22.485499</td>
          <td>0.007715</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.964165</td>
          <td>22.217297</td>
          <td>20.337681</td>
          <td>25.814386</td>
          <td>18.133076</td>
          <td>21.842137</td>
          <td>20.653288</td>
          <td>0.005039</td>
          <td>18.059000</td>
          <td>0.005001</td>
          <td>20.647115</td>
          <td>0.005116</td>
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
          <td>19.527282</td>
          <td>17.143296</td>
          <td>16.325678</td>
          <td>24.087456</td>
          <td>27.024126</td>
          <td>25.443227</td>
          <td>22.659167</td>
          <td>0.006384</td>
          <td>21.437408</td>
          <td>0.005479</td>
          <td>18.424790</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.973790</td>
          <td>22.651915</td>
          <td>17.641324</td>
          <td>19.160299</td>
          <td>19.948362</td>
          <td>19.983643</td>
          <td>28.205663</td>
          <td>0.515152</td>
          <td>23.714903</td>
          <td>0.018820</td>
          <td>23.674976</td>
          <td>0.018194</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.888084</td>
          <td>21.472701</td>
          <td>25.409300</td>
          <td>25.506502</td>
          <td>26.038964</td>
          <td>23.844242</td>
          <td>21.744916</td>
          <td>0.005285</td>
          <td>22.495906</td>
          <td>0.007758</td>
          <td>26.081608</td>
          <td>0.151026</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.636115</td>
          <td>18.956715</td>
          <td>22.324651</td>
          <td>22.897488</td>
          <td>25.617307</td>
          <td>25.827263</td>
          <td>24.839514</td>
          <td>0.029697</td>
          <td>24.451328</td>
          <td>0.035830</td>
          <td>24.580138</td>
          <td>0.040182</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.902736</td>
          <td>24.991641</td>
          <td>26.289874</td>
          <td>20.476752</td>
          <td>26.623450</td>
          <td>19.655433</td>
          <td>20.773724</td>
          <td>0.005049</td>
          <td>20.478704</td>
          <td>0.005085</td>
          <td>22.524603</td>
          <td>0.007880</td>
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
          <td>29.346576</td>
          <td>23.649215</td>
          <td>24.574980</td>
          <td>23.445548</td>
          <td>19.791261</td>
          <td>25.680658</td>
          <td>23.607465</td>
          <td>0.109178</td>
          <td>23.273320</td>
          <td>0.068153</td>
          <td>21.330358</td>
          <td>0.013706</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.906479</td>
          <td>29.347745</td>
          <td>25.016563</td>
          <td>23.050146</td>
          <td>25.619179</td>
          <td>21.925930</td>
          <td>25.420550</td>
          <td>0.483778</td>
          <td>23.689224</td>
          <td>0.098429</td>
          <td>23.294760</td>
          <td>0.075905</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.789419</td>
          <td>27.766131</td>
          <td>23.060602</td>
          <td>25.430301</td>
          <td>21.049031</td>
          <td>25.595927</td>
          <td>28.638105</td>
          <td>2.686224</td>
          <td>23.773089</td>
          <td>0.105942</td>
          <td>24.673394</td>
          <td>0.248606</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.494070</td>
          <td>23.022909</td>
          <td>26.057029</td>
          <td>20.951936</td>
          <td>21.863378</td>
          <td>22.192699</td>
          <td>25.690180</td>
          <td>0.588589</td>
          <td>19.423133</td>
          <td>0.005391</td>
          <td>22.459002</td>
          <td>0.036075</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.964165</td>
          <td>22.217297</td>
          <td>20.337681</td>
          <td>25.814386</td>
          <td>18.133076</td>
          <td>21.842137</td>
          <td>20.660984</td>
          <td>0.009070</td>
          <td>18.063767</td>
          <td>0.005033</td>
          <td>20.663569</td>
          <td>0.008537</td>
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
          <td>19.527282</td>
          <td>17.143296</td>
          <td>16.325678</td>
          <td>24.087456</td>
          <td>27.024126</td>
          <td>25.443227</td>
          <td>22.662484</td>
          <td>0.047272</td>
          <td>21.439794</td>
          <td>0.013809</td>
          <td>18.422001</td>
          <td>0.005077</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.973790</td>
          <td>22.651915</td>
          <td>17.641324</td>
          <td>19.160299</td>
          <td>19.948362</td>
          <td>19.983643</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.542795</td>
          <td>0.086522</td>
          <td>23.738785</td>
          <td>0.112208</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.888084</td>
          <td>21.472701</td>
          <td>25.409300</td>
          <td>25.506502</td>
          <td>26.038964</td>
          <td>23.844242</td>
          <td>21.758836</td>
          <td>0.021289</td>
          <td>22.538493</td>
          <td>0.035424</td>
          <td>26.095836</td>
          <td>0.727093</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.636115</td>
          <td>18.956715</td>
          <td>22.324651</td>
          <td>22.897488</td>
          <td>25.617307</td>
          <td>25.827263</td>
          <td>24.563707</td>
          <td>0.246632</td>
          <td>24.397247</td>
          <td>0.181543</td>
          <td>24.535261</td>
          <td>0.221741</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.902736</td>
          <td>24.991641</td>
          <td>26.289874</td>
          <td>20.476752</td>
          <td>26.623450</td>
          <td>19.655433</td>
          <td>20.780080</td>
          <td>0.009812</td>
          <td>20.480417</td>
          <td>0.007311</td>
          <td>22.569983</td>
          <td>0.039820</td>
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


