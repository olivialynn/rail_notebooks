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
          <td>25.118001</td>
          <td>22.687016</td>
          <td>21.105566</td>
          <td>15.579860</td>
          <td>27.193416</td>
          <td>23.741949</td>
          <td>26.004930</td>
          <td>22.591131</td>
          <td>23.511870</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.234907</td>
          <td>23.024541</td>
          <td>23.969711</td>
          <td>22.602795</td>
          <td>21.373990</td>
          <td>22.404920</td>
          <td>29.638317</td>
          <td>25.928293</td>
          <td>21.729917</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.727592</td>
          <td>24.923343</td>
          <td>27.577185</td>
          <td>18.298563</td>
          <td>23.236338</td>
          <td>17.831236</td>
          <td>22.738497</td>
          <td>20.256648</td>
          <td>21.391688</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.087744</td>
          <td>24.116876</td>
          <td>24.819036</td>
          <td>21.288478</td>
          <td>20.635700</td>
          <td>18.147741</td>
          <td>24.722296</td>
          <td>23.454560</td>
          <td>26.488003</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.755118</td>
          <td>25.257239</td>
          <td>21.276279</td>
          <td>25.075001</td>
          <td>32.216889</td>
          <td>22.209350</td>
          <td>22.771720</td>
          <td>20.204799</td>
          <td>25.555596</td>
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
          <td>22.041186</td>
          <td>25.093454</td>
          <td>29.552173</td>
          <td>22.984622</td>
          <td>21.236261</td>
          <td>24.251494</td>
          <td>26.147360</td>
          <td>27.608306</td>
          <td>18.511129</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.805891</td>
          <td>25.546920</td>
          <td>23.958183</td>
          <td>23.651314</td>
          <td>19.400275</td>
          <td>29.200588</td>
          <td>21.188719</td>
          <td>30.881697</td>
          <td>24.641947</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.640898</td>
          <td>24.986292</td>
          <td>24.470148</td>
          <td>25.142581</td>
          <td>24.552584</td>
          <td>27.905536</td>
          <td>21.594743</td>
          <td>22.154532</td>
          <td>23.585866</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.336827</td>
          <td>24.755800</td>
          <td>25.127717</td>
          <td>25.485994</td>
          <td>19.583768</td>
          <td>19.781042</td>
          <td>24.833147</td>
          <td>20.092712</td>
          <td>20.994338</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.953256</td>
          <td>26.022768</td>
          <td>24.141181</td>
          <td>22.896212</td>
          <td>23.641612</td>
          <td>19.413779</td>
          <td>24.007008</td>
          <td>19.350243</td>
          <td>27.708062</td>
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
          <td>24.970654</td>
          <td>0.105544</td>
          <td>22.688030</td>
          <td>0.006863</td>
          <td>21.111035</td>
          <td>0.005116</td>
          <td>15.587464</td>
          <td>0.005000</td>
          <td>28.163137</td>
          <td>1.117299</td>
          <td>23.752731</td>
          <td>0.076014</td>
          <td>26.004930</td>
          <td>22.591131</td>
          <td>23.511870</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.257835</td>
          <td>0.023681</td>
          <td>23.026052</td>
          <td>0.008044</td>
          <td>23.968976</td>
          <td>0.013608</td>
          <td>22.587000</td>
          <td>0.007824</td>
          <td>21.374178</td>
          <td>0.006372</td>
          <td>22.348376</td>
          <td>0.022056</td>
          <td>29.638317</td>
          <td>25.928293</td>
          <td>21.729917</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.714629</td>
          <td>0.007901</td>
          <td>24.952466</td>
          <td>0.035508</td>
          <td>27.574758</td>
          <td>0.301479</td>
          <td>18.297274</td>
          <td>0.005006</td>
          <td>23.230093</td>
          <td>0.021357</td>
          <td>17.833534</td>
          <td>0.005026</td>
          <td>22.738497</td>
          <td>20.256648</td>
          <td>21.391688</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.091941</td>
          <td>0.006227</td>
          <td>24.127328</td>
          <td>0.017478</td>
          <td>24.818374</td>
          <td>0.027719</td>
          <td>21.287473</td>
          <td>0.005367</td>
          <td>20.627923</td>
          <td>0.005417</td>
          <td>18.156753</td>
          <td>0.005040</td>
          <td>24.722296</td>
          <td>23.454560</td>
          <td>26.488003</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.550927</td>
          <td>0.173846</td>
          <td>25.312627</td>
          <td>0.048815</td>
          <td>21.277315</td>
          <td>0.005150</td>
          <td>25.104049</td>
          <td>0.058224</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.237621</td>
          <td>0.020069</td>
          <td>22.771720</td>
          <td>20.204799</td>
          <td>25.555596</td>
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
          <td>22.055684</td>
          <td>0.009551</td>
          <td>25.034150</td>
          <td>0.038159</td>
          <td>31.068389</td>
          <td>2.384040</td>
          <td>22.983285</td>
          <td>0.009927</td>
          <td>21.239085</td>
          <td>0.006110</td>
          <td>24.337145</td>
          <td>0.126841</td>
          <td>26.147360</td>
          <td>27.608306</td>
          <td>18.511129</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.802826</td>
          <td>0.005009</td>
          <td>25.479795</td>
          <td>0.056605</td>
          <td>23.971677</td>
          <td>0.013637</td>
          <td>23.643455</td>
          <td>0.016330</td>
          <td>19.394135</td>
          <td>0.005062</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.188719</td>
          <td>30.881697</td>
          <td>24.641947</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.653599</td>
          <td>0.014497</td>
          <td>24.979217</td>
          <td>0.036355</td>
          <td>24.475743</td>
          <td>0.020619</td>
          <td>25.183784</td>
          <td>0.062491</td>
          <td>24.515237</td>
          <td>0.066169</td>
          <td>29.481139</td>
          <td>2.948698</td>
          <td>21.594743</td>
          <td>22.154532</td>
          <td>23.585866</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.336496</td>
          <td>0.006727</td>
          <td>24.778436</td>
          <td>0.030478</td>
          <td>25.103689</td>
          <td>0.035624</td>
          <td>25.403068</td>
          <td>0.075883</td>
          <td>19.583705</td>
          <td>0.005082</td>
          <td>19.790657</td>
          <td>0.005477</td>
          <td>24.833147</td>
          <td>20.092712</td>
          <td>20.994338</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.945068</td>
          <td>0.103219</td>
          <td>26.070570</td>
          <td>0.095326</td>
          <td>24.160937</td>
          <td>0.015859</td>
          <td>22.889890</td>
          <td>0.009338</td>
          <td>23.636539</td>
          <td>0.030389</td>
          <td>19.417666</td>
          <td>0.005262</td>
          <td>24.007008</td>
          <td>19.350243</td>
          <td>27.708062</td>
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
          <td>25.118001</td>
          <td>22.687016</td>
          <td>21.105566</td>
          <td>15.579860</td>
          <td>27.193416</td>
          <td>23.741949</td>
          <td>25.818014</td>
          <td>0.070912</td>
          <td>22.584713</td>
          <td>0.008150</td>
          <td>23.495480</td>
          <td>0.015664</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.234907</td>
          <td>23.024541</td>
          <td>23.969711</td>
          <td>22.602795</td>
          <td>21.373990</td>
          <td>22.404920</td>
          <td>28.745022</td>
          <td>0.751380</td>
          <td>26.034174</td>
          <td>0.144988</td>
          <td>21.727081</td>
          <td>0.005792</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.727592</td>
          <td>24.923343</td>
          <td>27.577185</td>
          <td>18.298563</td>
          <td>23.236338</td>
          <td>17.831236</td>
          <td>22.736496</td>
          <td>0.006570</td>
          <td>20.252108</td>
          <td>0.005056</td>
          <td>21.387805</td>
          <td>0.005439</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.087744</td>
          <td>24.116876</td>
          <td>24.819036</td>
          <td>21.288478</td>
          <td>20.635700</td>
          <td>18.147741</td>
          <td>24.734509</td>
          <td>0.027072</td>
          <td>23.458801</td>
          <td>0.015200</td>
          <td>26.427061</td>
          <td>0.202557</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.755118</td>
          <td>25.257239</td>
          <td>21.276279</td>
          <td>25.075001</td>
          <td>32.216889</td>
          <td>22.209350</td>
          <td>22.777746</td>
          <td>0.006678</td>
          <td>20.203994</td>
          <td>0.005051</td>
          <td>25.542470</td>
          <td>0.094465</td>
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
          <td>22.041186</td>
          <td>25.093454</td>
          <td>29.552173</td>
          <td>22.984622</td>
          <td>21.236261</td>
          <td>24.251494</td>
          <td>26.113579</td>
          <td>0.092093</td>
          <td>27.462137</td>
          <td>0.463138</td>
          <td>18.503299</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.805891</td>
          <td>25.546920</td>
          <td>23.958183</td>
          <td>23.651314</td>
          <td>19.400275</td>
          <td>29.200588</td>
          <td>21.188492</td>
          <td>0.005104</td>
          <td>29.627890</td>
          <td>1.727723</td>
          <td>24.604770</td>
          <td>0.041073</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.640898</td>
          <td>24.986292</td>
          <td>24.470148</td>
          <td>25.142581</td>
          <td>24.552584</td>
          <td>27.905536</td>
          <td>21.596804</td>
          <td>0.005218</td>
          <td>22.146849</td>
          <td>0.006596</td>
          <td>23.549217</td>
          <td>0.016375</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.336827</td>
          <td>24.755800</td>
          <td>25.127717</td>
          <td>25.485994</td>
          <td>19.583768</td>
          <td>19.781042</td>
          <td>24.808817</td>
          <td>0.028903</td>
          <td>20.089592</td>
          <td>0.005042</td>
          <td>20.981928</td>
          <td>0.005212</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.953256</td>
          <td>26.022768</td>
          <td>24.141181</td>
          <td>22.896212</td>
          <td>23.641612</td>
          <td>19.413779</td>
          <td>24.023297</td>
          <td>0.014767</td>
          <td>19.353534</td>
          <td>0.005011</td>
          <td>27.306209</td>
          <td>0.411495</td>
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
          <td>25.118001</td>
          <td>22.687016</td>
          <td>21.105566</td>
          <td>15.579860</td>
          <td>27.193416</td>
          <td>23.741949</td>
          <td>26.055690</td>
          <td>0.756721</td>
          <td>22.576812</td>
          <td>0.036651</td>
          <td>23.739486</td>
          <td>0.112277</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.234907</td>
          <td>23.024541</td>
          <td>23.969711</td>
          <td>22.602795</td>
          <td>21.373990</td>
          <td>22.404920</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.812743</td>
          <td>0.556854</td>
          <td>21.757979</td>
          <td>0.019524</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.727592</td>
          <td>24.923343</td>
          <td>27.577185</td>
          <td>18.298563</td>
          <td>23.236338</td>
          <td>17.831236</td>
          <td>22.861916</td>
          <td>0.056469</td>
          <td>20.264167</td>
          <td>0.006642</td>
          <td>21.403029</td>
          <td>0.014527</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.087744</td>
          <td>24.116876</td>
          <td>24.819036</td>
          <td>21.288478</td>
          <td>20.635700</td>
          <td>18.147741</td>
          <td>25.491465</td>
          <td>0.509810</td>
          <td>23.415356</td>
          <td>0.077303</td>
          <td>25.917775</td>
          <td>0.643818</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.755118</td>
          <td>25.257239</td>
          <td>21.276279</td>
          <td>25.075001</td>
          <td>32.216889</td>
          <td>22.209350</td>
          <td>22.790076</td>
          <td>0.052967</td>
          <td>20.215982</td>
          <td>0.006519</td>
          <td>25.282695</td>
          <td>0.404133</td>
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
          <td>22.041186</td>
          <td>25.093454</td>
          <td>29.552173</td>
          <td>22.984622</td>
          <td>21.236261</td>
          <td>24.251494</td>
          <td>26.739189</td>
          <td>1.152646</td>
          <td>28.967185</td>
          <td>2.805004</td>
          <td>18.510113</td>
          <td>0.005090</td>
        </tr>
        <tr>
          <th>996</th>
          <td>16.805891</td>
          <td>25.546920</td>
          <td>23.958183</td>
          <td>23.651314</td>
          <td>19.400275</td>
          <td>29.200588</td>
          <td>21.185402</td>
          <td>0.013227</td>
          <td>27.838843</td>
          <td>1.816979</td>
          <td>25.307462</td>
          <td>0.411890</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.640898</td>
          <td>24.986292</td>
          <td>24.470148</td>
          <td>25.142581</td>
          <td>24.552584</td>
          <td>27.905536</td>
          <td>21.583399</td>
          <td>0.018324</td>
          <td>22.163374</td>
          <td>0.025433</td>
          <td>23.555634</td>
          <td>0.095566</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.336827</td>
          <td>24.755800</td>
          <td>25.127717</td>
          <td>25.485994</td>
          <td>19.583768</td>
          <td>19.781042</td>
          <td>24.743438</td>
          <td>0.285629</td>
          <td>20.095918</td>
          <td>0.006247</td>
          <td>21.000586</td>
          <td>0.010674</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.953256</td>
          <td>26.022768</td>
          <td>24.141181</td>
          <td>22.896212</td>
          <td>23.641612</td>
          <td>19.413779</td>
          <td>23.890174</td>
          <td>0.139589</td>
          <td>19.356948</td>
          <td>0.005348</td>
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


