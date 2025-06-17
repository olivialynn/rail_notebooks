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
          <td>23.678401</td>
          <td>22.618883</td>
          <td>18.042774</td>
          <td>21.626920</td>
          <td>22.611660</td>
          <td>21.070543</td>
          <td>23.449180</td>
          <td>22.459003</td>
          <td>24.364838</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.853340</td>
          <td>23.498045</td>
          <td>20.779727</td>
          <td>22.513693</td>
          <td>28.799223</td>
          <td>21.531555</td>
          <td>22.886875</td>
          <td>20.092485</td>
          <td>18.441479</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.861052</td>
          <td>24.435105</td>
          <td>19.744862</td>
          <td>22.174592</td>
          <td>24.351274</td>
          <td>23.581631</td>
          <td>25.849485</td>
          <td>27.027019</td>
          <td>21.517090</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.178001</td>
          <td>25.050423</td>
          <td>20.846966</td>
          <td>19.126539</td>
          <td>22.733988</td>
          <td>19.653957</td>
          <td>22.052097</td>
          <td>24.321460</td>
          <td>22.634372</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.070038</td>
          <td>16.148853</td>
          <td>22.532086</td>
          <td>25.963880</td>
          <td>18.688122</td>
          <td>23.431982</td>
          <td>26.179874</td>
          <td>21.724371</td>
          <td>26.632733</td>
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
          <td>16.183750</td>
          <td>14.330705</td>
          <td>27.570896</td>
          <td>22.520205</td>
          <td>22.063323</td>
          <td>23.464934</td>
          <td>18.759740</td>
          <td>18.364938</td>
          <td>24.680466</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.972464</td>
          <td>27.990482</td>
          <td>17.699538</td>
          <td>21.888654</td>
          <td>25.297979</td>
          <td>19.787470</td>
          <td>19.198553</td>
          <td>21.922479</td>
          <td>21.170750</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.428245</td>
          <td>21.630924</td>
          <td>22.726794</td>
          <td>22.017520</td>
          <td>22.937800</td>
          <td>25.509088</td>
          <td>24.288108</td>
          <td>22.380106</td>
          <td>26.751170</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.538588</td>
          <td>25.582866</td>
          <td>19.860198</td>
          <td>23.438329</td>
          <td>23.025794</td>
          <td>28.215028</td>
          <td>21.731147</td>
          <td>24.363684</td>
          <td>26.051926</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.554277</td>
          <td>22.396741</td>
          <td>23.491868</td>
          <td>18.942561</td>
          <td>28.856204</td>
          <td>25.294712</td>
          <td>20.325464</td>
          <td>26.549700</td>
          <td>25.467143</td>
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
          <td>23.673582</td>
          <td>0.033847</td>
          <td>22.621838</td>
          <td>0.006688</td>
          <td>18.041604</td>
          <td>0.005002</td>
          <td>21.628013</td>
          <td>0.005636</td>
          <td>22.619413</td>
          <td>0.012968</td>
          <td>21.061023</td>
          <td>0.008398</td>
          <td>23.449180</td>
          <td>22.459003</td>
          <td>24.364838</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.711553</td>
          <td>0.443368</td>
          <td>23.491616</td>
          <td>0.010732</td>
          <td>20.781218</td>
          <td>0.005071</td>
          <td>22.507378</td>
          <td>0.007512</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.521534</td>
          <td>0.011351</td>
          <td>22.886875</td>
          <td>20.092485</td>
          <td>18.441479</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.869938</td>
          <td>0.005004</td>
          <td>24.443068</td>
          <td>0.022794</td>
          <td>19.747953</td>
          <td>0.005017</td>
          <td>22.169628</td>
          <td>0.006505</td>
          <td>24.370914</td>
          <td>0.058221</td>
          <td>23.624741</td>
          <td>0.067877</td>
          <td>25.849485</td>
          <td>27.027019</td>
          <td>21.517090</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.207022</td>
          <td>0.022687</td>
          <td>25.090313</td>
          <td>0.040099</td>
          <td>20.849070</td>
          <td>0.005078</td>
          <td>19.132160</td>
          <td>0.005015</td>
          <td>22.743262</td>
          <td>0.014282</td>
          <td>19.652272</td>
          <td>0.005382</td>
          <td>22.052097</td>
          <td>24.321460</td>
          <td>22.634372</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.079740</td>
          <td>0.048296</td>
          <td>16.150657</td>
          <td>0.005001</td>
          <td>22.531430</td>
          <td>0.006116</td>
          <td>25.847285</td>
          <td>0.112102</td>
          <td>18.695257</td>
          <td>0.005024</td>
          <td>23.443369</td>
          <td>0.057795</td>
          <td>26.179874</td>
          <td>21.724371</td>
          <td>26.632733</td>
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
          <td>16.182415</td>
          <td>0.005005</td>
          <td>14.323582</td>
          <td>0.005000</td>
          <td>28.317276</td>
          <td>0.533112</td>
          <td>22.515953</td>
          <td>0.007544</td>
          <td>22.054723</td>
          <td>0.008773</td>
          <td>23.565684</td>
          <td>0.064417</td>
          <td>18.759740</td>
          <td>18.364938</td>
          <td>24.680466</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.970508</td>
          <td>0.006036</td>
          <td>28.064263</td>
          <td>0.492057</td>
          <td>17.699012</td>
          <td>0.005002</td>
          <td>21.892125</td>
          <td>0.005972</td>
          <td>25.028764</td>
          <td>0.104031</td>
          <td>19.789077</td>
          <td>0.005476</td>
          <td>19.198553</td>
          <td>21.922479</td>
          <td>21.170750</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.429926</td>
          <td>0.005044</td>
          <td>21.627036</td>
          <td>0.005368</td>
          <td>22.723732</td>
          <td>0.006508</td>
          <td>22.026907</td>
          <td>0.006203</td>
          <td>22.949750</td>
          <td>0.016876</td>
          <td>25.938457</td>
          <td>0.469205</td>
          <td>24.288108</td>
          <td>22.380106</td>
          <td>26.751170</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.536945</td>
          <td>0.005567</td>
          <td>25.575881</td>
          <td>0.061632</td>
          <td>19.862029</td>
          <td>0.005020</td>
          <td>23.430202</td>
          <td>0.013763</td>
          <td>23.022184</td>
          <td>0.017920</td>
          <td>27.002699</td>
          <td>0.968584</td>
          <td>21.731147</td>
          <td>24.363684</td>
          <td>26.051926</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.547596</td>
          <td>0.013378</td>
          <td>22.392140</td>
          <td>0.006193</td>
          <td>23.484803</td>
          <td>0.009594</td>
          <td>18.937589</td>
          <td>0.005012</td>
          <td>27.293041</td>
          <td>0.642456</td>
          <td>25.326795</td>
          <td>0.291396</td>
          <td>20.325464</td>
          <td>26.549700</td>
          <td>25.467143</td>
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
          <td>23.678401</td>
          <td>22.618883</td>
          <td>18.042774</td>
          <td>21.626920</td>
          <td>22.611660</td>
          <td>21.070543</td>
          <td>23.440027</td>
          <td>0.009551</td>
          <td>22.458496</td>
          <td>0.007606</td>
          <td>24.382248</td>
          <td>0.033698</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.853340</td>
          <td>23.498045</td>
          <td>20.779727</td>
          <td>22.513693</td>
          <td>28.799223</td>
          <td>21.531555</td>
          <td>22.895151</td>
          <td>0.007023</td>
          <td>20.085283</td>
          <td>0.005041</td>
          <td>18.444889</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.861052</td>
          <td>24.435105</td>
          <td>19.744862</td>
          <td>22.174592</td>
          <td>24.351274</td>
          <td>23.581631</td>
          <td>25.849970</td>
          <td>0.072951</td>
          <td>26.609396</td>
          <td>0.235817</td>
          <td>21.524521</td>
          <td>0.005558</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.178001</td>
          <td>25.050423</td>
          <td>20.846966</td>
          <td>19.126539</td>
          <td>22.733988</td>
          <td>19.653957</td>
          <td>22.047288</td>
          <td>0.005487</td>
          <td>24.391543</td>
          <td>0.033977</td>
          <td>22.622593</td>
          <td>0.008331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.070038</td>
          <td>16.148853</td>
          <td>22.532086</td>
          <td>25.963880</td>
          <td>18.688122</td>
          <td>23.431982</td>
          <td>26.205615</td>
          <td>0.099856</td>
          <td>21.721510</td>
          <td>0.005785</td>
          <td>26.795671</td>
          <td>0.274768</td>
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
          <td>16.183750</td>
          <td>14.330705</td>
          <td>27.570896</td>
          <td>22.520205</td>
          <td>22.063323</td>
          <td>23.464934</td>
          <td>18.760204</td>
          <td>0.005001</td>
          <td>18.361109</td>
          <td>0.005002</td>
          <td>24.719610</td>
          <td>0.045500</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.972464</td>
          <td>27.990482</td>
          <td>17.699538</td>
          <td>21.888654</td>
          <td>25.297979</td>
          <td>19.787470</td>
          <td>19.194067</td>
          <td>0.005003</td>
          <td>21.926349</td>
          <td>0.006110</td>
          <td>21.174094</td>
          <td>0.005300</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.428245</td>
          <td>21.630924</td>
          <td>22.726794</td>
          <td>22.017520</td>
          <td>22.937800</td>
          <td>25.509088</td>
          <td>24.261532</td>
          <td>0.017989</td>
          <td>22.378775</td>
          <td>0.007305</td>
          <td>26.627001</td>
          <td>0.239275</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.538588</td>
          <td>25.582866</td>
          <td>19.860198</td>
          <td>23.438329</td>
          <td>23.025794</td>
          <td>28.215028</td>
          <td>21.722819</td>
          <td>0.005274</td>
          <td>24.321302</td>
          <td>0.031925</td>
          <td>26.044132</td>
          <td>0.146236</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.554277</td>
          <td>22.396741</td>
          <td>23.491868</td>
          <td>18.942561</td>
          <td>28.856204</td>
          <td>25.294712</td>
          <td>20.325006</td>
          <td>0.005021</td>
          <td>26.308977</td>
          <td>0.183356</td>
          <td>25.627992</td>
          <td>0.101837</td>
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
          <td>23.678401</td>
          <td>22.618883</td>
          <td>18.042774</td>
          <td>21.626920</td>
          <td>22.611660</td>
          <td>21.070543</td>
          <td>23.266348</td>
          <td>0.080871</td>
          <td>22.388936</td>
          <td>0.031023</td>
          <td>24.259507</td>
          <td>0.175820</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.853340</td>
          <td>23.498045</td>
          <td>20.779727</td>
          <td>22.513693</td>
          <td>28.799223</td>
          <td>21.531555</td>
          <td>22.817513</td>
          <td>0.054278</td>
          <td>20.079730</td>
          <td>0.006214</td>
          <td>18.444256</td>
          <td>0.005080</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.861052</td>
          <td>24.435105</td>
          <td>19.744862</td>
          <td>22.174592</td>
          <td>24.351274</td>
          <td>23.581631</td>
          <td>25.845816</td>
          <td>0.656451</td>
          <td>25.716231</td>
          <td>0.519156</td>
          <td>21.502892</td>
          <td>0.015760</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.178001</td>
          <td>25.050423</td>
          <td>20.846966</td>
          <td>19.126539</td>
          <td>22.733988</td>
          <td>19.653957</td>
          <td>22.026818</td>
          <td>0.026889</td>
          <td>24.631919</td>
          <td>0.221125</td>
          <td>22.602859</td>
          <td>0.041003</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.070038</td>
          <td>16.148853</td>
          <td>22.532086</td>
          <td>25.963880</td>
          <td>18.688122</td>
          <td>23.431982</td>
          <td>30.330204</td>
          <td>4.303597</td>
          <td>21.712529</td>
          <td>0.017263</td>
          <td>25.463229</td>
          <td>0.463517</td>
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
          <td>16.183750</td>
          <td>14.330705</td>
          <td>27.570896</td>
          <td>22.520205</td>
          <td>22.063323</td>
          <td>23.464934</td>
          <td>18.758467</td>
          <td>0.005170</td>
          <td>18.363228</td>
          <td>0.005057</td>
          <td>24.396939</td>
          <td>0.197492</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.972464</td>
          <td>27.990482</td>
          <td>17.699538</td>
          <td>21.888654</td>
          <td>25.297979</td>
          <td>19.787470</td>
          <td>19.199799</td>
          <td>0.005375</td>
          <td>21.905614</td>
          <td>0.020336</td>
          <td>21.186260</td>
          <td>0.012249</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.428245</td>
          <td>21.630924</td>
          <td>22.726794</td>
          <td>22.017520</td>
          <td>22.937800</td>
          <td>25.509088</td>
          <td>24.166987</td>
          <td>0.176941</td>
          <td>22.446369</td>
          <td>0.032642</td>
          <td>25.146772</td>
          <td>0.363686</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.538588</td>
          <td>25.582866</td>
          <td>19.860198</td>
          <td>23.438329</td>
          <td>23.025794</td>
          <td>28.215028</td>
          <td>21.720162</td>
          <td>0.020592</td>
          <td>24.812492</td>
          <td>0.256721</td>
          <td>25.413245</td>
          <td>0.446410</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.554277</td>
          <td>22.396741</td>
          <td>23.491868</td>
          <td>18.942561</td>
          <td>28.856204</td>
          <td>25.294712</td>
          <td>20.331792</td>
          <td>0.007501</td>
          <td>25.742044</td>
          <td>0.529039</td>
          <td>25.843424</td>
          <td>0.611188</td>
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


