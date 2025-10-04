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
          <td>20.334969</td>
          <td>27.546585</td>
          <td>20.051272</td>
          <td>27.069287</td>
          <td>25.320045</td>
          <td>22.508693</td>
          <td>29.404999</td>
          <td>25.597582</td>
          <td>19.433457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.708488</td>
          <td>21.546758</td>
          <td>25.509321</td>
          <td>23.983723</td>
          <td>23.136279</td>
          <td>20.442632</td>
          <td>22.205049</td>
          <td>18.467176</td>
          <td>26.276924</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.308249</td>
          <td>19.434200</td>
          <td>25.629342</td>
          <td>19.838420</td>
          <td>21.756306</td>
          <td>24.526741</td>
          <td>25.352480</td>
          <td>20.261840</td>
          <td>25.675894</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974207</td>
          <td>20.574634</td>
          <td>22.912519</td>
          <td>23.961521</td>
          <td>22.021879</td>
          <td>16.845323</td>
          <td>25.013204</td>
          <td>21.609776</td>
          <td>26.853218</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.023112</td>
          <td>25.206881</td>
          <td>21.456181</td>
          <td>18.110929</td>
          <td>24.245118</td>
          <td>22.591596</td>
          <td>23.275217</td>
          <td>22.915466</td>
          <td>19.289246</td>
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
          <td>24.307784</td>
          <td>19.550096</td>
          <td>23.417221</td>
          <td>23.246507</td>
          <td>19.794057</td>
          <td>23.627951</td>
          <td>27.417222</td>
          <td>22.393403</td>
          <td>19.293839</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.121883</td>
          <td>25.980865</td>
          <td>21.204368</td>
          <td>16.661174</td>
          <td>31.228454</td>
          <td>21.844474</td>
          <td>25.385548</td>
          <td>20.980184</td>
          <td>26.841673</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.239319</td>
          <td>21.512206</td>
          <td>19.200974</td>
          <td>20.211394</td>
          <td>18.030657</td>
          <td>20.346940</td>
          <td>22.738089</td>
          <td>20.382063</td>
          <td>23.054781</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.689179</td>
          <td>21.579766</td>
          <td>24.353747</td>
          <td>24.399156</td>
          <td>19.229098</td>
          <td>29.444340</td>
          <td>22.806233</td>
          <td>25.473955</td>
          <td>22.661469</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.294482</td>
          <td>22.356002</td>
          <td>25.406568</td>
          <td>25.275140</td>
          <td>27.021568</td>
          <td>23.502599</td>
          <td>29.101771</td>
          <td>27.846745</td>
          <td>21.518209</td>
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
          <td>20.341015</td>
          <td>0.005435</td>
          <td>28.287617</td>
          <td>0.578812</td>
          <td>20.049816</td>
          <td>0.005025</td>
          <td>26.726339</td>
          <td>0.237057</td>
          <td>25.450595</td>
          <td>0.149971</td>
          <td>22.488748</td>
          <td>0.024894</td>
          <td>29.404999</td>
          <td>25.597582</td>
          <td>19.433457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.706391</td>
          <td>0.034828</td>
          <td>21.534738</td>
          <td>0.005320</td>
          <td>25.564154</td>
          <td>0.053590</td>
          <td>23.993528</td>
          <td>0.021912</td>
          <td>23.130694</td>
          <td>0.019628</td>
          <td>20.441034</td>
          <td>0.006344</td>
          <td>22.205049</td>
          <td>18.467176</td>
          <td>26.276924</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.309847</td>
          <td>0.005118</td>
          <td>19.430222</td>
          <td>0.005019</td>
          <td>25.729646</td>
          <td>0.062067</td>
          <td>19.833780</td>
          <td>0.005040</td>
          <td>21.765446</td>
          <td>0.007488</td>
          <td>24.495987</td>
          <td>0.145485</td>
          <td>25.352480</td>
          <td>20.261840</td>
          <td>25.675894</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.970308</td>
          <td>0.005266</td>
          <td>20.578092</td>
          <td>0.005081</td>
          <td>22.907457</td>
          <td>0.006999</td>
          <td>23.983401</td>
          <td>0.021722</td>
          <td>22.014659</td>
          <td>0.008567</td>
          <td>16.839530</td>
          <td>0.005008</td>
          <td>25.013204</td>
          <td>21.609776</td>
          <td>26.853218</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.024583</td>
          <td>0.005286</td>
          <td>25.185288</td>
          <td>0.043611</td>
          <td>21.451169</td>
          <td>0.005197</td>
          <td>18.111746</td>
          <td>0.005005</td>
          <td>24.270806</td>
          <td>0.053271</td>
          <td>22.571527</td>
          <td>0.026751</td>
          <td>23.275217</td>
          <td>22.915466</td>
          <td>19.289246</td>
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
          <td>24.210686</td>
          <td>0.054191</td>
          <td>19.550170</td>
          <td>0.005022</td>
          <td>23.422512</td>
          <td>0.009218</td>
          <td>23.241918</td>
          <td>0.011920</td>
          <td>19.794859</td>
          <td>0.005112</td>
          <td>23.704226</td>
          <td>0.072823</td>
          <td>27.417222</td>
          <td>22.393403</td>
          <td>19.293839</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.119454</td>
          <td>0.005323</td>
          <td>26.113737</td>
          <td>0.099000</td>
          <td>21.203519</td>
          <td>0.005134</td>
          <td>16.652958</td>
          <td>0.005001</td>
          <td>28.228349</td>
          <td>1.159651</td>
          <td>21.857210</td>
          <td>0.014665</td>
          <td>25.385548</td>
          <td>20.980184</td>
          <td>26.841673</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.256746</td>
          <td>0.005388</td>
          <td>21.514402</td>
          <td>0.005310</td>
          <td>19.193464</td>
          <td>0.005009</td>
          <td>20.220130</td>
          <td>0.005069</td>
          <td>18.031104</td>
          <td>0.005010</td>
          <td>20.350656</td>
          <td>0.006166</td>
          <td>22.738089</td>
          <td>20.382063</td>
          <td>23.054781</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.687476</td>
          <td>0.005058</td>
          <td>21.583385</td>
          <td>0.005345</td>
          <td>24.388648</td>
          <td>0.019153</td>
          <td>24.343071</td>
          <td>0.029683</td>
          <td>19.235197</td>
          <td>0.005049</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.806233</td>
          <td>25.473955</td>
          <td>22.661469</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.297566</td>
          <td>0.006636</td>
          <td>22.341116</td>
          <td>0.006103</td>
          <td>25.431305</td>
          <td>0.047627</td>
          <td>25.224829</td>
          <td>0.064807</td>
          <td>26.835595</td>
          <td>0.461502</td>
          <td>23.473893</td>
          <td>0.059382</td>
          <td>29.101771</td>
          <td>27.846745</td>
          <td>21.518209</td>
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
          <td>20.334969</td>
          <td>27.546585</td>
          <td>20.051272</td>
          <td>27.069287</td>
          <td>25.320045</td>
          <td>22.508693</td>
          <td>30.681634</td>
          <td>2.103179</td>
          <td>25.735394</td>
          <td>0.111876</td>
          <td>19.421108</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.708488</td>
          <td>21.546758</td>
          <td>25.509321</td>
          <td>23.983723</td>
          <td>23.136279</td>
          <td>20.442632</td>
          <td>22.216777</td>
          <td>0.005655</td>
          <td>18.466347</td>
          <td>0.005002</td>
          <td>26.222385</td>
          <td>0.170354</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.308249</td>
          <td>19.434200</td>
          <td>25.629342</td>
          <td>19.838420</td>
          <td>21.756306</td>
          <td>24.526741</td>
          <td>25.391226</td>
          <td>0.048499</td>
          <td>20.255030</td>
          <td>0.005057</td>
          <td>25.445706</td>
          <td>0.086745</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974207</td>
          <td>20.574634</td>
          <td>22.912519</td>
          <td>23.961521</td>
          <td>22.021879</td>
          <td>16.845323</td>
          <td>25.003374</td>
          <td>0.034336</td>
          <td>21.601773</td>
          <td>0.005638</td>
          <td>27.017318</td>
          <td>0.328388</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.023112</td>
          <td>25.206881</td>
          <td>21.456181</td>
          <td>18.110929</td>
          <td>24.245118</td>
          <td>22.591596</td>
          <td>23.270417</td>
          <td>0.008573</td>
          <td>22.906805</td>
          <td>0.009993</td>
          <td>19.287516</td>
          <td>0.005010</td>
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
          <td>24.307784</td>
          <td>19.550096</td>
          <td>23.417221</td>
          <td>23.246507</td>
          <td>19.794057</td>
          <td>23.627951</td>
          <td>27.320479</td>
          <td>0.258407</td>
          <td>22.400212</td>
          <td>0.007383</td>
          <td>19.292947</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.121883</td>
          <td>25.980865</td>
          <td>21.204368</td>
          <td>16.661174</td>
          <td>31.228454</td>
          <td>21.844474</td>
          <td>25.424416</td>
          <td>0.049956</td>
          <td>20.977557</td>
          <td>0.005211</td>
          <td>26.829189</td>
          <td>0.282351</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.239319</td>
          <td>21.512206</td>
          <td>19.200974</td>
          <td>20.211394</td>
          <td>18.030657</td>
          <td>20.346940</td>
          <td>22.731806</td>
          <td>0.006558</td>
          <td>20.375959</td>
          <td>0.005071</td>
          <td>23.068174</td>
          <td>0.011211</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.689179</td>
          <td>21.579766</td>
          <td>24.353747</td>
          <td>24.399156</td>
          <td>19.229098</td>
          <td>29.444340</td>
          <td>22.810889</td>
          <td>0.006770</td>
          <td>25.481735</td>
          <td>0.089545</td>
          <td>22.663763</td>
          <td>0.008538</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.294482</td>
          <td>22.356002</td>
          <td>25.406568</td>
          <td>25.275140</td>
          <td>27.021568</td>
          <td>23.502599</td>
          <td>29.473894</td>
          <td>1.175471</td>
          <td>27.989125</td>
          <td>0.676315</td>
          <td>21.519994</td>
          <td>0.005553</td>
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
          <td>20.334969</td>
          <td>27.546585</td>
          <td>20.051272</td>
          <td>27.069287</td>
          <td>25.320045</td>
          <td>22.508693</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.432288</td>
          <td>0.005474</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.708488</td>
          <td>21.546758</td>
          <td>25.509321</td>
          <td>23.983723</td>
          <td>23.136279</td>
          <td>20.442632</td>
          <td>22.226791</td>
          <td>0.032081</td>
          <td>18.469073</td>
          <td>0.005070</td>
          <td>25.968216</td>
          <td>0.666671</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.308249</td>
          <td>19.434200</td>
          <td>25.629342</td>
          <td>19.838420</td>
          <td>21.756306</td>
          <td>24.526741</td>
          <td>24.980657</td>
          <td>0.345276</td>
          <td>20.263628</td>
          <td>0.006640</td>
          <td>25.430472</td>
          <td>0.452247</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.974207</td>
          <td>20.574634</td>
          <td>22.912519</td>
          <td>23.961521</td>
          <td>22.021879</td>
          <td>16.845323</td>
          <td>24.754561</td>
          <td>0.288212</td>
          <td>21.622585</td>
          <td>0.016018</td>
          <td>26.834333</td>
          <td>1.149471</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.023112</td>
          <td>25.206881</td>
          <td>21.456181</td>
          <td>18.110929</td>
          <td>24.245118</td>
          <td>22.591596</td>
          <td>23.294826</td>
          <td>0.082933</td>
          <td>22.933501</td>
          <td>0.050362</td>
          <td>19.281651</td>
          <td>0.005363</td>
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
          <td>24.307784</td>
          <td>19.550096</td>
          <td>23.417221</td>
          <td>23.246507</td>
          <td>19.794057</td>
          <td>23.627951</td>
          <td>25.015880</td>
          <td>0.354983</td>
          <td>22.421960</td>
          <td>0.031944</td>
          <td>19.296105</td>
          <td>0.005373</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.121883</td>
          <td>25.980865</td>
          <td>21.204368</td>
          <td>16.661174</td>
          <td>31.228454</td>
          <td>21.844474</td>
          <td>28.497370</td>
          <td>2.558074</td>
          <td>20.987232</td>
          <td>0.009860</td>
          <td>27.439381</td>
          <td>1.580351</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.239319</td>
          <td>21.512206</td>
          <td>19.200974</td>
          <td>20.211394</td>
          <td>18.030657</td>
          <td>20.346940</td>
          <td>22.658053</td>
          <td>0.047086</td>
          <td>20.379524</td>
          <td>0.006974</td>
          <td>22.983185</td>
          <td>0.057549</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.689179</td>
          <td>21.579766</td>
          <td>24.353747</td>
          <td>24.399156</td>
          <td>19.229098</td>
          <td>29.444340</td>
          <td>22.785848</td>
          <td>0.052768</td>
          <td>25.525671</td>
          <td>0.450614</td>
          <td>22.709695</td>
          <td>0.045099</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.294482</td>
          <td>22.356002</td>
          <td>25.406568</td>
          <td>25.275140</td>
          <td>27.021568</td>
          <td>23.502599</td>
          <td>27.628703</td>
          <td>1.808749</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.502554</td>
          <td>0.015755</td>
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


