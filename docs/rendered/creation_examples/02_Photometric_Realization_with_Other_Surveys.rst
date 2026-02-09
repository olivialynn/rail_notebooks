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
          <td>26.044764</td>
          <td>27.383412</td>
          <td>23.322843</td>
          <td>14.784527</td>
          <td>23.903733</td>
          <td>22.218464</td>
          <td>26.308349</td>
          <td>23.842458</td>
          <td>20.138138</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.329931</td>
          <td>22.837009</td>
          <td>23.677828</td>
          <td>20.427291</td>
          <td>22.261060</td>
          <td>24.608374</td>
          <td>25.673589</td>
          <td>21.625713</td>
          <td>15.859870</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.442496</td>
          <td>22.559822</td>
          <td>24.146213</td>
          <td>19.280767</td>
          <td>21.360552</td>
          <td>22.054094</td>
          <td>26.644466</td>
          <td>26.560035</td>
          <td>20.304162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.380816</td>
          <td>17.357390</td>
          <td>27.195320</td>
          <td>27.859933</td>
          <td>22.482327</td>
          <td>22.633213</td>
          <td>24.174916</td>
          <td>17.279879</td>
          <td>20.922547</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.510361</td>
          <td>22.076363</td>
          <td>25.661370</td>
          <td>24.765824</td>
          <td>23.561478</td>
          <td>24.786404</td>
          <td>21.041022</td>
          <td>20.657808</td>
          <td>27.993467</td>
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
          <td>20.198108</td>
          <td>27.289282</td>
          <td>19.054228</td>
          <td>25.007246</td>
          <td>24.203894</td>
          <td>24.204292</td>
          <td>17.755001</td>
          <td>25.990160</td>
          <td>22.590564</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.494406</td>
          <td>31.145934</td>
          <td>27.393752</td>
          <td>22.132232</td>
          <td>19.497104</td>
          <td>24.451911</td>
          <td>20.975014</td>
          <td>16.965583</td>
          <td>23.629693</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.087691</td>
          <td>18.748404</td>
          <td>25.451554</td>
          <td>23.173630</td>
          <td>22.142138</td>
          <td>24.050219</td>
          <td>24.064805</td>
          <td>26.151502</td>
          <td>26.865334</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.875727</td>
          <td>26.601001</td>
          <td>17.969086</td>
          <td>22.354431</td>
          <td>22.875707</td>
          <td>16.479526</td>
          <td>21.480665</td>
          <td>16.685155</td>
          <td>27.074366</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.000998</td>
          <td>25.702878</td>
          <td>22.743215</td>
          <td>21.187449</td>
          <td>22.529593</td>
          <td>25.512745</td>
          <td>19.922405</td>
          <td>23.950887</td>
          <td>18.913451</td>
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
          <td>26.638253</td>
          <td>0.419374</td>
          <td>27.195994</td>
          <td>0.249112</td>
          <td>23.320838</td>
          <td>0.008660</td>
          <td>14.781061</td>
          <td>0.005000</td>
          <td>23.914386</td>
          <td>0.038830</td>
          <td>22.207579</td>
          <td>0.019566</td>
          <td>26.308349</td>
          <td>23.842458</td>
          <td>20.138138</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.212259</td>
          <td>0.054266</td>
          <td>22.849335</td>
          <td>0.007362</td>
          <td>23.698957</td>
          <td>0.011112</td>
          <td>20.432574</td>
          <td>0.005095</td>
          <td>22.260295</td>
          <td>0.010004</td>
          <td>24.589182</td>
          <td>0.157592</td>
          <td>25.673589</td>
          <td>21.625713</td>
          <td>15.859870</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.416479</td>
          <td>0.027106</td>
          <td>22.558829</td>
          <td>0.006536</td>
          <td>24.137295</td>
          <td>0.015558</td>
          <td>19.280602</td>
          <td>0.005019</td>
          <td>21.355277</td>
          <td>0.006332</td>
          <td>22.082551</td>
          <td>0.017620</td>
          <td>26.644466</td>
          <td>26.560035</td>
          <td>20.304162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.369903</td>
          <td>0.005126</td>
          <td>17.361561</td>
          <td>0.005002</td>
          <td>27.064196</td>
          <td>0.198033</td>
          <td>27.708353</td>
          <td>0.511987</td>
          <td>22.478882</td>
          <td>0.011669</td>
          <td>22.642861</td>
          <td>0.028471</td>
          <td>24.174916</td>
          <td>17.279879</td>
          <td>20.922547</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.502809</td>
          <td>0.012941</td>
          <td>22.073326</td>
          <td>0.005731</td>
          <td>25.705257</td>
          <td>0.060739</td>
          <td>24.798706</td>
          <td>0.044398</td>
          <td>23.567850</td>
          <td>0.028613</td>
          <td>24.834527</td>
          <td>0.194087</td>
          <td>21.041022</td>
          <td>20.657808</td>
          <td>27.993467</td>
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
          <td>20.202450</td>
          <td>0.005361</td>
          <td>27.197029</td>
          <td>0.249324</td>
          <td>19.058057</td>
          <td>0.005007</td>
          <td>24.981126</td>
          <td>0.052204</td>
          <td>24.251785</td>
          <td>0.052379</td>
          <td>24.147810</td>
          <td>0.107574</td>
          <td>17.755001</td>
          <td>25.990160</td>
          <td>22.590564</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.513149</td>
          <td>0.029456</td>
          <td>28.474667</td>
          <td>0.660008</td>
          <td>27.086608</td>
          <td>0.201796</td>
          <td>22.123903</td>
          <td>0.006401</td>
          <td>19.490218</td>
          <td>0.005071</td>
          <td>24.409929</td>
          <td>0.135085</td>
          <td>20.975014</td>
          <td>16.965583</td>
          <td>23.629693</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.135262</td>
          <td>0.050712</td>
          <td>18.736212</td>
          <td>0.005009</td>
          <td>25.475405</td>
          <td>0.049529</td>
          <td>23.176769</td>
          <td>0.011364</td>
          <td>22.136165</td>
          <td>0.009226</td>
          <td>24.019074</td>
          <td>0.096108</td>
          <td>24.064805</td>
          <td>26.151502</td>
          <td>26.865334</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.872929</td>
          <td>0.008584</td>
          <td>26.547006</td>
          <td>0.144210</td>
          <td>17.976153</td>
          <td>0.005002</td>
          <td>22.353391</td>
          <td>0.006995</td>
          <td>22.894085</td>
          <td>0.016123</td>
          <td>16.479124</td>
          <td>0.005005</td>
          <td>21.480665</td>
          <td>16.685155</td>
          <td>27.074366</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.981121</td>
          <td>0.018806</td>
          <td>25.669709</td>
          <td>0.066966</td>
          <td>22.745418</td>
          <td>0.006560</td>
          <td>21.189405</td>
          <td>0.005313</td>
          <td>22.517571</td>
          <td>0.012007</td>
          <td>25.965086</td>
          <td>0.478619</td>
          <td>19.922405</td>
          <td>23.950887</td>
          <td>18.913451</td>
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
          <td>26.044764</td>
          <td>27.383412</td>
          <td>23.322843</td>
          <td>14.784527</td>
          <td>23.903733</td>
          <td>22.218464</td>
          <td>26.287370</td>
          <td>0.107275</td>
          <td>23.866548</td>
          <td>0.021431</td>
          <td>20.139824</td>
          <td>0.005046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.329931</td>
          <td>22.837009</td>
          <td>23.677828</td>
          <td>20.427291</td>
          <td>22.261060</td>
          <td>24.608374</td>
          <td>25.557264</td>
          <td>0.056235</td>
          <td>21.618535</td>
          <td>0.005657</td>
          <td>15.866403</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.442496</td>
          <td>22.559822</td>
          <td>24.146213</td>
          <td>19.280767</td>
          <td>21.360552</td>
          <td>22.054094</td>
          <td>26.812662</td>
          <td>0.168949</td>
          <td>26.468307</td>
          <td>0.209685</td>
          <td>20.302021</td>
          <td>0.005062</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.380816</td>
          <td>17.357390</td>
          <td>27.195320</td>
          <td>27.859933</td>
          <td>22.482327</td>
          <td>22.633213</td>
          <td>24.199640</td>
          <td>0.017078</td>
          <td>17.282017</td>
          <td>0.005000</td>
          <td>20.921292</td>
          <td>0.005190</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.510361</td>
          <td>22.076363</td>
          <td>25.661370</td>
          <td>24.765824</td>
          <td>23.561478</td>
          <td>24.786404</td>
          <td>21.049382</td>
          <td>0.005081</td>
          <td>20.650723</td>
          <td>0.005116</td>
          <td>28.986285</td>
          <td>1.251083</td>
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
          <td>20.198108</td>
          <td>27.289282</td>
          <td>19.054228</td>
          <td>25.007246</td>
          <td>24.203894</td>
          <td>24.204292</td>
          <td>17.758269</td>
          <td>0.005000</td>
          <td>25.889753</td>
          <td>0.127969</td>
          <td>22.592784</td>
          <td>0.008188</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.494406</td>
          <td>31.145934</td>
          <td>27.393752</td>
          <td>22.132232</td>
          <td>19.497104</td>
          <td>24.451911</td>
          <td>20.980277</td>
          <td>0.005071</td>
          <td>16.969626</td>
          <td>0.005000</td>
          <td>23.628754</td>
          <td>0.017499</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.087691</td>
          <td>18.748404</td>
          <td>25.451554</td>
          <td>23.173630</td>
          <td>22.142138</td>
          <td>24.050219</td>
          <td>24.068069</td>
          <td>0.015316</td>
          <td>26.451289</td>
          <td>0.206717</td>
          <td>28.422401</td>
          <td>0.898749</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.875727</td>
          <td>26.601001</td>
          <td>17.969086</td>
          <td>22.354431</td>
          <td>22.875707</td>
          <td>16.479526</td>
          <td>21.487091</td>
          <td>0.005179</td>
          <td>16.692794</td>
          <td>0.005000</td>
          <td>27.025149</td>
          <td>0.330437</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.000998</td>
          <td>25.702878</td>
          <td>22.743215</td>
          <td>21.187449</td>
          <td>22.529593</td>
          <td>25.512745</td>
          <td>19.922703</td>
          <td>0.005010</td>
          <td>23.933030</td>
          <td>0.022701</td>
          <td>18.919038</td>
          <td>0.005005</td>
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
          <td>26.044764</td>
          <td>27.383412</td>
          <td>23.322843</td>
          <td>14.784527</td>
          <td>23.903733</td>
          <td>22.218464</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.708301</td>
          <td>0.100092</td>
          <td>20.136777</td>
          <td>0.006571</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.329931</td>
          <td>22.837009</td>
          <td>23.677828</td>
          <td>20.427291</td>
          <td>22.261060</td>
          <td>24.608374</td>
          <td>26.492883</td>
          <td>0.997994</td>
          <td>21.637351</td>
          <td>0.016215</td>
          <td>15.860743</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.442496</td>
          <td>22.559822</td>
          <td>24.146213</td>
          <td>19.280767</td>
          <td>21.360552</td>
          <td>22.054094</td>
          <td>25.977768</td>
          <td>0.718312</td>
          <td>25.214391</td>
          <td>0.354568</td>
          <td>20.303555</td>
          <td>0.007050</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.380816</td>
          <td>17.357390</td>
          <td>27.195320</td>
          <td>27.859933</td>
          <td>22.482327</td>
          <td>22.633213</td>
          <td>24.200877</td>
          <td>0.182102</td>
          <td>17.267317</td>
          <td>0.005008</td>
          <td>20.924186</td>
          <td>0.010114</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.510361</td>
          <td>22.076363</td>
          <td>25.661370</td>
          <td>24.765824</td>
          <td>23.561478</td>
          <td>24.786404</td>
          <td>21.041384</td>
          <td>0.011839</td>
          <td>20.669183</td>
          <td>0.008078</td>
          <td>27.229753</td>
          <td>1.423367</td>
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
          <td>20.198108</td>
          <td>27.289282</td>
          <td>19.054228</td>
          <td>25.007246</td>
          <td>24.203894</td>
          <td>24.204292</td>
          <td>17.758961</td>
          <td>0.005027</td>
          <td>26.749640</td>
          <td>1.032468</td>
          <td>22.575113</td>
          <td>0.040002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.494406</td>
          <td>31.145934</td>
          <td>27.393752</td>
          <td>22.132232</td>
          <td>19.497104</td>
          <td>24.451911</td>
          <td>20.984134</td>
          <td>0.011343</td>
          <td>16.963279</td>
          <td>0.005004</td>
          <td>23.724228</td>
          <td>0.110790</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.087691</td>
          <td>18.748404</td>
          <td>25.451554</td>
          <td>23.173630</td>
          <td>22.142138</td>
          <td>24.050219</td>
          <td>24.323983</td>
          <td>0.202034</td>
          <td>26.989603</td>
          <td>1.185885</td>
          <td>25.593638</td>
          <td>0.510625</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.875727</td>
          <td>26.601001</td>
          <td>17.969086</td>
          <td>22.354431</td>
          <td>22.875707</td>
          <td>16.479526</td>
          <td>21.478526</td>
          <td>0.016779</td>
          <td>16.689329</td>
          <td>0.005003</td>
          <td>27.063188</td>
          <td>1.304275</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.000998</td>
          <td>25.702878</td>
          <td>22.743215</td>
          <td>21.187449</td>
          <td>22.529593</td>
          <td>25.512745</td>
          <td>19.935828</td>
          <td>0.006332</td>
          <td>24.142668</td>
          <td>0.146052</td>
          <td>18.907080</td>
          <td>0.005185</td>
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


