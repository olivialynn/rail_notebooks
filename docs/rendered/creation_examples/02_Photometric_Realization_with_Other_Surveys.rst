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
          <td>24.413188</td>
          <td>24.174503</td>
          <td>32.279405</td>
          <td>21.852157</td>
          <td>20.499062</td>
          <td>23.583026</td>
          <td>23.655377</td>
          <td>22.124095</td>
          <td>24.478375</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.931767</td>
          <td>21.988660</td>
          <td>27.698279</td>
          <td>18.207831</td>
          <td>22.868625</td>
          <td>18.140690</td>
          <td>22.373917</td>
          <td>23.467038</td>
          <td>24.648634</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.831112</td>
          <td>18.541029</td>
          <td>21.578730</td>
          <td>23.858964</td>
          <td>27.944579</td>
          <td>22.480956</td>
          <td>24.430583</td>
          <td>23.225147</td>
          <td>24.154327</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.038226</td>
          <td>23.641948</td>
          <td>18.764663</td>
          <td>23.314332</td>
          <td>19.316480</td>
          <td>24.406308</td>
          <td>21.300848</td>
          <td>26.135722</td>
          <td>20.660379</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.348815</td>
          <td>19.057164</td>
          <td>21.227668</td>
          <td>24.699837</td>
          <td>24.853193</td>
          <td>22.820080</td>
          <td>18.854626</td>
          <td>25.529852</td>
          <td>24.736115</td>
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
          <td>18.433756</td>
          <td>28.009292</td>
          <td>25.365636</td>
          <td>24.610315</td>
          <td>22.924910</td>
          <td>26.565702</td>
          <td>22.492003</td>
          <td>20.722253</td>
          <td>22.206121</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.464845</td>
          <td>23.460859</td>
          <td>18.937284</td>
          <td>16.869293</td>
          <td>24.779007</td>
          <td>23.199046</td>
          <td>22.452568</td>
          <td>23.987180</td>
          <td>19.293915</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.537356</td>
          <td>27.405837</td>
          <td>23.140194</td>
          <td>21.170317</td>
          <td>20.353647</td>
          <td>21.963509</td>
          <td>23.888065</td>
          <td>26.279398</td>
          <td>23.470926</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.924342</td>
          <td>26.941287</td>
          <td>18.497417</td>
          <td>27.911267</td>
          <td>25.271431</td>
          <td>18.603395</td>
          <td>26.147955</td>
          <td>20.599816</td>
          <td>19.975879</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.125070</td>
          <td>24.056116</td>
          <td>24.383504</td>
          <td>26.749972</td>
          <td>23.531430</td>
          <td>14.545339</td>
          <td>18.494723</td>
          <td>20.914828</td>
          <td>25.777005</td>
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
          <td>24.333851</td>
          <td>0.060395</td>
          <td>24.140919</td>
          <td>0.017675</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.847357</td>
          <td>0.005905</td>
          <td>20.507244</td>
          <td>0.005344</td>
          <td>23.662174</td>
          <td>0.070164</td>
          <td>23.655377</td>
          <td>22.124095</td>
          <td>24.478375</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.944525</td>
          <td>0.018253</td>
          <td>21.994585</td>
          <td>0.005647</td>
          <td>27.125735</td>
          <td>0.208523</td>
          <td>18.202907</td>
          <td>0.005005</td>
          <td>22.880558</td>
          <td>0.015946</td>
          <td>18.133882</td>
          <td>0.005038</td>
          <td>22.373917</td>
          <td>23.467038</td>
          <td>24.648634</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.827952</td>
          <td>0.005848</td>
          <td>18.545053</td>
          <td>0.005007</td>
          <td>21.573903</td>
          <td>0.005240</td>
          <td>23.884535</td>
          <td>0.019967</td>
          <td>27.186521</td>
          <td>0.596180</td>
          <td>22.492605</td>
          <td>0.024977</td>
          <td>24.430583</td>
          <td>23.225147</td>
          <td>24.154327</td>
        </tr>
        <tr>
          <th>3</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.641373</td>
          <td>0.011944</td>
          <td>18.760511</td>
          <td>0.005005</td>
          <td>23.325844</td>
          <td>0.012697</td>
          <td>19.311712</td>
          <td>0.005055</td>
          <td>24.252375</td>
          <td>0.117840</td>
          <td>21.300848</td>
          <td>26.135722</td>
          <td>20.660379</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.381008</td>
          <td>0.011850</td>
          <td>19.059613</td>
          <td>0.005013</td>
          <td>21.224371</td>
          <td>0.005138</td>
          <td>24.694297</td>
          <td>0.040471</td>
          <td>24.782238</td>
          <td>0.083780</td>
          <td>22.814134</td>
          <td>0.033093</td>
          <td>18.854626</td>
          <td>25.529852</td>
          <td>24.736115</td>
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
          <td>18.435100</td>
          <td>0.005044</td>
          <td>27.647324</td>
          <td>0.358027</td>
          <td>25.426855</td>
          <td>0.047439</td>
          <td>24.559389</td>
          <td>0.035915</td>
          <td>22.912652</td>
          <td>0.016370</td>
          <td>25.606579</td>
          <td>0.363974</td>
          <td>22.492003</td>
          <td>20.722253</td>
          <td>22.206121</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.470067</td>
          <td>0.005518</td>
          <td>23.458241</td>
          <td>0.010487</td>
          <td>18.930604</td>
          <td>0.005006</td>
          <td>16.866826</td>
          <td>0.005001</td>
          <td>24.756239</td>
          <td>0.081881</td>
          <td>23.138541</td>
          <td>0.044095</td>
          <td>22.452568</td>
          <td>23.987180</td>
          <td>19.293915</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.534037</td>
          <td>0.005049</td>
          <td>27.373909</td>
          <td>0.287984</td>
          <td>23.141658</td>
          <td>0.007831</td>
          <td>21.163625</td>
          <td>0.005300</td>
          <td>20.339212</td>
          <td>0.005263</td>
          <td>21.950243</td>
          <td>0.015805</td>
          <td>23.888065</td>
          <td>26.279398</td>
          <td>23.470926</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.871019</td>
          <td>0.096764</td>
          <td>26.406111</td>
          <td>0.127704</td>
          <td>18.495370</td>
          <td>0.005004</td>
          <td>29.799253</td>
          <td>1.775760</td>
          <td>25.274310</td>
          <td>0.128824</td>
          <td>18.595733</td>
          <td>0.005074</td>
          <td>26.147955</td>
          <td>20.599816</td>
          <td>19.975879</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.156005</td>
          <td>0.021736</td>
          <td>24.032505</td>
          <td>0.016175</td>
          <td>24.405671</td>
          <td>0.019430</td>
          <td>27.190570</td>
          <td>0.345011</td>
          <td>23.511828</td>
          <td>0.027245</td>
          <td>14.551474</td>
          <td>0.005001</td>
          <td>18.494723</td>
          <td>20.914828</td>
          <td>25.777005</td>
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
          <td>24.413188</td>
          <td>24.174503</td>
          <td>32.279405</td>
          <td>21.852157</td>
          <td>20.499062</td>
          <td>23.583026</td>
          <td>23.639141</td>
          <td>0.010975</td>
          <td>22.125238</td>
          <td>0.006542</td>
          <td>24.419859</td>
          <td>0.034842</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.931767</td>
          <td>21.988660</td>
          <td>27.698279</td>
          <td>18.207831</td>
          <td>22.868625</td>
          <td>18.140690</td>
          <td>22.375550</td>
          <td>0.005861</td>
          <td>23.457187</td>
          <td>0.015180</td>
          <td>24.678530</td>
          <td>0.043863</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.831112</td>
          <td>18.541029</td>
          <td>21.578730</td>
          <td>23.858964</td>
          <td>27.944579</td>
          <td>22.480956</td>
          <td>24.410998</td>
          <td>0.020430</td>
          <td>23.238314</td>
          <td>0.012750</td>
          <td>24.151269</td>
          <td>0.027474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.038226</td>
          <td>23.641948</td>
          <td>18.764663</td>
          <td>23.314332</td>
          <td>19.316480</td>
          <td>24.406308</td>
          <td>21.299729</td>
          <td>0.005127</td>
          <td>26.112670</td>
          <td>0.155106</td>
          <td>20.651294</td>
          <td>0.005117</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.348815</td>
          <td>19.057164</td>
          <td>21.227668</td>
          <td>24.699837</td>
          <td>24.853193</td>
          <td>22.820080</td>
          <td>18.856784</td>
          <td>0.005001</td>
          <td>25.502717</td>
          <td>0.091216</td>
          <td>24.695510</td>
          <td>0.044532</td>
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
          <td>18.433756</td>
          <td>28.009292</td>
          <td>25.365636</td>
          <td>24.610315</td>
          <td>22.924910</td>
          <td>26.565702</td>
          <td>22.493227</td>
          <td>0.006050</td>
          <td>20.718738</td>
          <td>0.005132</td>
          <td>22.201082</td>
          <td>0.006742</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.464845</td>
          <td>23.460859</td>
          <td>18.937284</td>
          <td>16.869293</td>
          <td>24.779007</td>
          <td>23.199046</td>
          <td>22.445100</td>
          <td>0.005969</td>
          <td>23.990174</td>
          <td>0.023857</td>
          <td>19.291507</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.537356</td>
          <td>27.405837</td>
          <td>23.140194</td>
          <td>21.170317</td>
          <td>20.353647</td>
          <td>21.963509</td>
          <td>23.887415</td>
          <td>0.013248</td>
          <td>26.331862</td>
          <td>0.186941</td>
          <td>23.453697</td>
          <td>0.015137</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.924342</td>
          <td>26.941287</td>
          <td>18.497417</td>
          <td>27.911267</td>
          <td>25.271431</td>
          <td>18.603395</td>
          <td>26.119245</td>
          <td>0.092554</td>
          <td>20.611422</td>
          <td>0.005108</td>
          <td>19.978309</td>
          <td>0.005034</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.125070</td>
          <td>24.056116</td>
          <td>24.383504</td>
          <td>26.749972</td>
          <td>23.531430</td>
          <td>14.545339</td>
          <td>18.493763</td>
          <td>0.005001</td>
          <td>20.921967</td>
          <td>0.005190</td>
          <td>25.941547</td>
          <td>0.133844</td>
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
          <td>24.413188</td>
          <td>24.174503</td>
          <td>32.279405</td>
          <td>21.852157</td>
          <td>20.499062</td>
          <td>23.583026</td>
          <td>23.506965</td>
          <td>0.099975</td>
          <td>22.080577</td>
          <td>0.023659</td>
          <td>24.254076</td>
          <td>0.175010</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.931767</td>
          <td>21.988660</td>
          <td>27.698279</td>
          <td>18.207831</td>
          <td>22.868625</td>
          <td>18.140690</td>
          <td>22.389368</td>
          <td>0.037063</td>
          <td>23.425129</td>
          <td>0.077975</td>
          <td>24.186680</td>
          <td>0.165245</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.831112</td>
          <td>18.541029</td>
          <td>21.578730</td>
          <td>23.858964</td>
          <td>27.944579</td>
          <td>22.480956</td>
          <td>24.320047</td>
          <td>0.201367</td>
          <td>23.293082</td>
          <td>0.069360</td>
          <td>24.344159</td>
          <td>0.188895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.038226</td>
          <td>23.641948</td>
          <td>18.764663</td>
          <td>23.314332</td>
          <td>19.316480</td>
          <td>24.406308</td>
          <td>21.288891</td>
          <td>0.014363</td>
          <td>26.192481</td>
          <td>0.725457</td>
          <td>20.661619</td>
          <td>0.008527</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.348815</td>
          <td>19.057164</td>
          <td>21.227668</td>
          <td>24.699837</td>
          <td>24.853193</td>
          <td>22.820080</td>
          <td>18.858903</td>
          <td>0.005204</td>
          <td>25.001364</td>
          <td>0.299303</td>
          <td>24.747216</td>
          <td>0.264123</td>
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
          <td>18.433756</td>
          <td>28.009292</td>
          <td>25.365636</td>
          <td>24.610315</td>
          <td>22.924910</td>
          <td>26.565702</td>
          <td>22.505225</td>
          <td>0.041090</td>
          <td>20.729464</td>
          <td>0.008365</td>
          <td>22.216165</td>
          <td>0.029091</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.464845</td>
          <td>23.460859</td>
          <td>18.937284</td>
          <td>16.869293</td>
          <td>24.779007</td>
          <td>23.199046</td>
          <td>22.394322</td>
          <td>0.037227</td>
          <td>23.837548</td>
          <td>0.112087</td>
          <td>19.291782</td>
          <td>0.005370</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.537356</td>
          <td>27.405837</td>
          <td>23.140194</td>
          <td>21.170317</td>
          <td>20.353647</td>
          <td>21.963509</td>
          <td>23.775709</td>
          <td>0.126419</td>
          <td>25.409564</td>
          <td>0.412554</td>
          <td>23.325340</td>
          <td>0.077989</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.924342</td>
          <td>26.941287</td>
          <td>18.497417</td>
          <td>27.911267</td>
          <td>25.271431</td>
          <td>18.603395</td>
          <td>25.150523</td>
          <td>0.394237</td>
          <td>20.611543</td>
          <td>0.007824</td>
          <td>19.975219</td>
          <td>0.006205</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.125070</td>
          <td>24.056116</td>
          <td>24.383504</td>
          <td>26.749972</td>
          <td>23.531430</td>
          <td>14.545339</td>
          <td>18.490282</td>
          <td>0.005104</td>
          <td>20.921558</td>
          <td>0.009435</td>
          <td>25.387879</td>
          <td>0.437925</td>
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


