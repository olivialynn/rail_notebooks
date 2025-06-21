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
          <td>21.386231</td>
          <td>24.115336</td>
          <td>27.269575</td>
          <td>20.963266</td>
          <td>21.289629</td>
          <td>23.014215</td>
          <td>27.038912</td>
          <td>15.694706</td>
          <td>22.697877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.775696</td>
          <td>19.162724</td>
          <td>23.212446</td>
          <td>23.099702</td>
          <td>23.758658</td>
          <td>20.936715</td>
          <td>21.226976</td>
          <td>19.084234</td>
          <td>17.108225</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.835589</td>
          <td>22.437482</td>
          <td>20.717873</td>
          <td>24.410622</td>
          <td>25.781576</td>
          <td>22.151535</td>
          <td>21.224266</td>
          <td>26.813603</td>
          <td>27.428931</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.494527</td>
          <td>21.765309</td>
          <td>26.140463</td>
          <td>26.789407</td>
          <td>19.091876</td>
          <td>22.944524</td>
          <td>32.974216</td>
          <td>22.087915</td>
          <td>25.735143</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.619301</td>
          <td>24.756749</td>
          <td>21.015544</td>
          <td>25.761906</td>
          <td>20.615673</td>
          <td>28.434212</td>
          <td>18.698907</td>
          <td>24.125151</td>
          <td>23.895956</td>
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
          <td>19.096102</td>
          <td>26.213432</td>
          <td>24.852589</td>
          <td>27.240815</td>
          <td>24.873994</td>
          <td>24.989988</td>
          <td>26.641472</td>
          <td>20.241440</td>
          <td>23.775841</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.390311</td>
          <td>20.372368</td>
          <td>24.045743</td>
          <td>21.928610</td>
          <td>23.089546</td>
          <td>24.300340</td>
          <td>21.396284</td>
          <td>26.187796</td>
          <td>25.219275</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.563690</td>
          <td>16.535470</td>
          <td>23.746126</td>
          <td>26.357362</td>
          <td>20.592385</td>
          <td>24.124636</td>
          <td>26.515306</td>
          <td>22.399950</td>
          <td>22.463351</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.450230</td>
          <td>21.491834</td>
          <td>18.941587</td>
          <td>23.573170</td>
          <td>26.531711</td>
          <td>24.032291</td>
          <td>23.340907</td>
          <td>23.393063</td>
          <td>18.699261</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.309563</td>
          <td>22.708274</td>
          <td>24.624797</td>
          <td>24.625816</td>
          <td>20.776987</td>
          <td>20.736867</td>
          <td>28.378191</td>
          <td>21.753554</td>
          <td>19.028420</td>
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
          <td>21.391307</td>
          <td>0.006864</td>
          <td>24.117051</td>
          <td>0.017331</td>
          <td>27.321750</td>
          <td>0.245379</td>
          <td>20.956983</td>
          <td>0.005216</td>
          <td>21.288225</td>
          <td>0.006199</td>
          <td>22.989048</td>
          <td>0.038623</td>
          <td>27.038912</td>
          <td>15.694706</td>
          <td>22.697877</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.747838</td>
          <td>0.036109</td>
          <td>19.167517</td>
          <td>0.005014</td>
          <td>23.220481</td>
          <td>0.008173</td>
          <td>23.101988</td>
          <td>0.010772</td>
          <td>23.745589</td>
          <td>0.033449</td>
          <td>20.930575</td>
          <td>0.007817</td>
          <td>21.226976</td>
          <td>19.084234</td>
          <td>17.108225</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.846902</td>
          <td>0.039373</td>
          <td>22.444777</td>
          <td>0.006292</td>
          <td>20.717903</td>
          <td>0.005064</td>
          <td>24.453271</td>
          <td>0.032704</td>
          <td>25.867129</td>
          <td>0.213441</td>
          <td>22.158295</td>
          <td>0.018770</td>
          <td>21.224266</td>
          <td>26.813603</td>
          <td>27.428931</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.483157</td>
          <td>0.007116</td>
          <td>21.763802</td>
          <td>0.005454</td>
          <td>26.012064</td>
          <td>0.079691</td>
          <td>27.149009</td>
          <td>0.333861</td>
          <td>19.090090</td>
          <td>0.005040</td>
          <td>22.957677</td>
          <td>0.037565</td>
          <td>32.974216</td>
          <td>22.087915</td>
          <td>25.735143</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.623859</td>
          <td>0.007566</td>
          <td>24.768533</td>
          <td>0.030215</td>
          <td>21.020631</td>
          <td>0.005101</td>
          <td>25.730468</td>
          <td>0.101223</td>
          <td>20.614188</td>
          <td>0.005408</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.698907</td>
          <td>24.125151</td>
          <td>23.895956</td>
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
          <td>19.091975</td>
          <td>0.005091</td>
          <td>26.042346</td>
          <td>0.092996</td>
          <td>24.850009</td>
          <td>0.028497</td>
          <td>27.415884</td>
          <td>0.411081</td>
          <td>24.865862</td>
          <td>0.090180</td>
          <td>24.934586</td>
          <td>0.211084</td>
          <td>26.641472</td>
          <td>20.241440</td>
          <td>23.775841</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.385345</td>
          <td>0.005006</td>
          <td>20.376603</td>
          <td>0.005061</td>
          <td>24.031511</td>
          <td>0.014294</td>
          <td>21.933982</td>
          <td>0.006039</td>
          <td>23.092034</td>
          <td>0.018998</td>
          <td>24.156565</td>
          <td>0.108399</td>
          <td>21.396284</td>
          <td>26.187796</td>
          <td>25.219275</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.567687</td>
          <td>0.013581</td>
          <td>16.536023</td>
          <td>0.005001</td>
          <td>23.746737</td>
          <td>0.011503</td>
          <td>26.967922</td>
          <td>0.288818</td>
          <td>20.595340</td>
          <td>0.005396</td>
          <td>24.142105</td>
          <td>0.107039</td>
          <td>26.515306</td>
          <td>22.399950</td>
          <td>22.463351</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.532399</td>
          <td>0.171136</td>
          <td>21.490671</td>
          <td>0.005299</td>
          <td>18.943159</td>
          <td>0.005006</td>
          <td>23.571607</td>
          <td>0.015403</td>
          <td>26.073199</td>
          <td>0.253152</td>
          <td>24.231568</td>
          <td>0.115725</td>
          <td>23.340907</td>
          <td>23.393063</td>
          <td>18.699261</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.322027</td>
          <td>0.011371</td>
          <td>22.706697</td>
          <td>0.006916</td>
          <td>24.627783</td>
          <td>0.023489</td>
          <td>24.589401</td>
          <td>0.036881</td>
          <td>20.772230</td>
          <td>0.005526</td>
          <td>20.737940</td>
          <td>0.007118</td>
          <td>28.378191</td>
          <td>21.753554</td>
          <td>19.028420</td>
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
          <td>21.386231</td>
          <td>24.115336</td>
          <td>27.269575</td>
          <td>20.963266</td>
          <td>21.289629</td>
          <td>23.014215</td>
          <td>27.030325</td>
          <td>0.203113</td>
          <td>15.699731</td>
          <td>0.005000</td>
          <td>22.703805</td>
          <td>0.008750</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.775696</td>
          <td>19.162724</td>
          <td>23.212446</td>
          <td>23.099702</td>
          <td>23.758658</td>
          <td>20.936715</td>
          <td>21.229055</td>
          <td>0.005112</td>
          <td>19.081429</td>
          <td>0.005007</td>
          <td>17.103375</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.835589</td>
          <td>22.437482</td>
          <td>20.717873</td>
          <td>24.410622</td>
          <td>25.781576</td>
          <td>22.151535</td>
          <td>21.223774</td>
          <td>0.005111</td>
          <td>26.884192</td>
          <td>0.295191</td>
          <td>29.243172</td>
          <td>1.433185</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.494527</td>
          <td>21.765309</td>
          <td>26.140463</td>
          <td>26.789407</td>
          <td>19.091876</td>
          <td>22.944524</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.076597</td>
          <td>0.006424</td>
          <td>25.961316</td>
          <td>0.136152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.619301</td>
          <td>24.756749</td>
          <td>21.015544</td>
          <td>25.761906</td>
          <td>20.615673</td>
          <td>28.434212</td>
          <td>18.699312</td>
          <td>0.005001</td>
          <td>24.104290</td>
          <td>0.026362</td>
          <td>23.882888</td>
          <td>0.021736</td>
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
          <td>19.096102</td>
          <td>26.213432</td>
          <td>24.852589</td>
          <td>27.240815</td>
          <td>24.873994</td>
          <td>24.989988</td>
          <td>26.558074</td>
          <td>0.135771</td>
          <td>20.244359</td>
          <td>0.005055</td>
          <td>23.786068</td>
          <td>0.019998</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.390311</td>
          <td>20.372368</td>
          <td>24.045743</td>
          <td>21.928610</td>
          <td>23.089546</td>
          <td>24.300340</td>
          <td>21.402215</td>
          <td>0.005153</td>
          <td>26.167997</td>
          <td>0.162629</td>
          <td>25.112850</td>
          <td>0.064587</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.563690</td>
          <td>16.535470</td>
          <td>23.746126</td>
          <td>26.357362</td>
          <td>20.592385</td>
          <td>24.124636</td>
          <td>26.539893</td>
          <td>0.133652</td>
          <td>22.408332</td>
          <td>0.007413</td>
          <td>22.447383</td>
          <td>0.007562</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.450230</td>
          <td>21.491834</td>
          <td>18.941587</td>
          <td>23.573170</td>
          <td>26.531711</td>
          <td>24.032291</td>
          <td>23.334349</td>
          <td>0.008918</td>
          <td>23.398014</td>
          <td>0.014469</td>
          <td>18.707587</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.309563</td>
          <td>22.708274</td>
          <td>24.624797</td>
          <td>24.625816</td>
          <td>20.776987</td>
          <td>20.736867</td>
          <td>30.368860</td>
          <td>1.841428</td>
          <td>21.759351</td>
          <td>0.005837</td>
          <td>19.031555</td>
          <td>0.005006</td>
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
          <td>21.386231</td>
          <td>24.115336</td>
          <td>27.269575</td>
          <td>20.963266</td>
          <td>21.289629</td>
          <td>23.014215</td>
          <td>27.322705</td>
          <td>1.567589</td>
          <td>15.696372</td>
          <td>0.005000</td>
          <td>22.691234</td>
          <td>0.044363</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.775696</td>
          <td>19.162724</td>
          <td>23.212446</td>
          <td>23.099702</td>
          <td>23.758658</td>
          <td>20.936715</td>
          <td>21.228459</td>
          <td>0.013685</td>
          <td>19.081894</td>
          <td>0.005212</td>
          <td>17.102903</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.835589</td>
          <td>22.437482</td>
          <td>20.717873</td>
          <td>24.410622</td>
          <td>25.781576</td>
          <td>22.151535</td>
          <td>21.228892</td>
          <td>0.013690</td>
          <td>27.066512</td>
          <td>1.237596</td>
          <td>31.037983</td>
          <td>4.902501</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.494527</td>
          <td>21.765309</td>
          <td>26.140463</td>
          <td>26.789407</td>
          <td>19.091876</td>
          <td>22.944524</td>
          <td>28.441197</td>
          <td>2.507352</td>
          <td>22.054055</td>
          <td>0.023119</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.619301</td>
          <td>24.756749</td>
          <td>21.015544</td>
          <td>25.761906</td>
          <td>20.615673</td>
          <td>28.434212</td>
          <td>18.697460</td>
          <td>0.005152</td>
          <td>24.384710</td>
          <td>0.179623</td>
          <td>23.912258</td>
          <td>0.130491</td>
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
          <td>19.096102</td>
          <td>26.213432</td>
          <td>24.852589</td>
          <td>27.240815</td>
          <td>24.873994</td>
          <td>24.989988</td>
          <td>25.680345</td>
          <td>0.584485</td>
          <td>20.248511</td>
          <td>0.006601</td>
          <td>23.686545</td>
          <td>0.107198</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.390311</td>
          <td>20.372368</td>
          <td>24.045743</td>
          <td>21.928610</td>
          <td>23.089546</td>
          <td>24.300340</td>
          <td>21.414467</td>
          <td>0.015911</td>
          <td>27.702772</td>
          <td>1.707768</td>
          <td>24.653034</td>
          <td>0.244472</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.563690</td>
          <td>16.535470</td>
          <td>23.746126</td>
          <td>26.357362</td>
          <td>20.592385</td>
          <td>24.124636</td>
          <td>25.599525</td>
          <td>0.551570</td>
          <td>22.417885</td>
          <td>0.031829</td>
          <td>22.395099</td>
          <td>0.034084</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.450230</td>
          <td>21.491834</td>
          <td>18.941587</td>
          <td>23.573170</td>
          <td>26.531711</td>
          <td>24.032291</td>
          <td>23.365345</td>
          <td>0.088261</td>
          <td>23.358748</td>
          <td>0.073521</td>
          <td>18.698332</td>
          <td>0.005127</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.309563</td>
          <td>22.708274</td>
          <td>24.624797</td>
          <td>24.625816</td>
          <td>20.776987</td>
          <td>20.736867</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.762751</td>
          <td>0.018008</td>
          <td>19.030755</td>
          <td>0.005232</td>
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


