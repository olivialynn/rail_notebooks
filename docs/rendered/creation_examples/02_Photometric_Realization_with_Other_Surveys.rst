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
          <td>29.260508</td>
          <td>20.861059</td>
          <td>23.083835</td>
          <td>19.901728</td>
          <td>14.318524</td>
          <td>25.662325</td>
          <td>17.144958</td>
          <td>22.971195</td>
          <td>18.162335</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.570752</td>
          <td>22.762881</td>
          <td>22.099434</td>
          <td>20.995434</td>
          <td>30.572345</td>
          <td>24.739955</td>
          <td>22.361741</td>
          <td>18.176652</td>
          <td>21.742875</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.005636</td>
          <td>18.584763</td>
          <td>18.893481</td>
          <td>24.472040</td>
          <td>23.209546</td>
          <td>24.037657</td>
          <td>23.610764</td>
          <td>24.893811</td>
          <td>22.716431</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.025543</td>
          <td>20.901775</td>
          <td>25.530339</td>
          <td>22.802948</td>
          <td>23.498416</td>
          <td>24.073561</td>
          <td>22.647588</td>
          <td>26.195096</td>
          <td>22.085223</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.211077</td>
          <td>29.619616</td>
          <td>17.510367</td>
          <td>22.056824</td>
          <td>26.199241</td>
          <td>19.717672</td>
          <td>25.870734</td>
          <td>25.232189</td>
          <td>25.462595</td>
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
          <td>20.415609</td>
          <td>23.836251</td>
          <td>21.896342</td>
          <td>21.777365</td>
          <td>24.975164</td>
          <td>23.350077</td>
          <td>26.889123</td>
          <td>22.253885</td>
          <td>22.738359</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.918747</td>
          <td>26.100382</td>
          <td>23.672081</td>
          <td>24.880943</td>
          <td>18.856381</td>
          <td>21.753298</td>
          <td>20.793742</td>
          <td>24.235182</td>
          <td>22.636089</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.985319</td>
          <td>15.568970</td>
          <td>25.547296</td>
          <td>21.061630</td>
          <td>24.552481</td>
          <td>22.732129</td>
          <td>25.136521</td>
          <td>27.703780</td>
          <td>25.582257</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.807829</td>
          <td>16.753764</td>
          <td>20.974805</td>
          <td>25.223907</td>
          <td>23.417808</td>
          <td>23.064195</td>
          <td>18.831183</td>
          <td>22.710373</td>
          <td>23.339270</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.926540</td>
          <td>25.870196</td>
          <td>19.178568</td>
          <td>22.884028</td>
          <td>27.367916</td>
          <td>20.820510</td>
          <td>25.168662</td>
          <td>24.919768</td>
          <td>28.360526</td>
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
          <td>27.043556</td>
          <td>0.566178</td>
          <td>20.858824</td>
          <td>0.005119</td>
          <td>23.084355</td>
          <td>0.007603</td>
          <td>19.902625</td>
          <td>0.005044</td>
          <td>14.321093</td>
          <td>0.005000</td>
          <td>25.810744</td>
          <td>0.426101</td>
          <td>17.144958</td>
          <td>22.971195</td>
          <td>18.162335</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.540467</td>
          <td>0.030158</td>
          <td>22.761025</td>
          <td>0.007076</td>
          <td>22.094817</td>
          <td>0.005554</td>
          <td>20.996609</td>
          <td>0.005230</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.837321</td>
          <td>0.194544</td>
          <td>22.361741</td>
          <td>18.176652</td>
          <td>21.742875</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.001341</td>
          <td>0.006081</td>
          <td>18.584997</td>
          <td>0.005008</td>
          <td>18.892634</td>
          <td>0.005006</td>
          <td>24.449812</td>
          <td>0.032604</td>
          <td>23.219063</td>
          <td>0.021157</td>
          <td>24.072604</td>
          <td>0.100725</td>
          <td>23.610764</td>
          <td>24.893811</td>
          <td>22.716431</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.009168</td>
          <td>0.254745</td>
          <td>20.912288</td>
          <td>0.005128</td>
          <td>25.551348</td>
          <td>0.052984</td>
          <td>22.798114</td>
          <td>0.008819</td>
          <td>23.462846</td>
          <td>0.026107</td>
          <td>24.202509</td>
          <td>0.112833</td>
          <td>22.647588</td>
          <td>26.195096</td>
          <td>22.085223</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.276080</td>
          <td>0.057401</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.508310</td>
          <td>0.005001</td>
          <td>22.054783</td>
          <td>0.006257</td>
          <td>25.937499</td>
          <td>0.226322</td>
          <td>19.719170</td>
          <td>0.005425</td>
          <td>25.870734</td>
          <td>25.232189</td>
          <td>25.462595</td>
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
          <td>20.417358</td>
          <td>0.005482</td>
          <td>23.838931</td>
          <td>0.013869</td>
          <td>21.897875</td>
          <td>0.005403</td>
          <td>21.778796</td>
          <td>0.005811</td>
          <td>24.938306</td>
          <td>0.096105</td>
          <td>23.358964</td>
          <td>0.053624</td>
          <td>26.889123</td>
          <td>22.253885</td>
          <td>22.738359</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.914784</td>
          <td>0.005958</td>
          <td>26.394250</td>
          <td>0.126399</td>
          <td>23.688849</td>
          <td>0.011032</td>
          <td>24.907627</td>
          <td>0.048906</td>
          <td>18.857813</td>
          <td>0.005029</td>
          <td>21.751699</td>
          <td>0.013496</td>
          <td>20.793742</td>
          <td>24.235182</td>
          <td>22.636089</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.567431</td>
          <td>0.397214</td>
          <td>15.575986</td>
          <td>0.005000</td>
          <td>25.541243</td>
          <td>0.052510</td>
          <td>21.061889</td>
          <td>0.005255</td>
          <td>24.587795</td>
          <td>0.070559</td>
          <td>22.696716</td>
          <td>0.029847</td>
          <td>25.136521</td>
          <td>27.703780</td>
          <td>25.582257</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.803869</td>
          <td>0.016302</td>
          <td>16.750724</td>
          <td>0.005001</td>
          <td>20.974242</td>
          <td>0.005094</td>
          <td>25.275649</td>
          <td>0.067792</td>
          <td>23.395980</td>
          <td>0.024634</td>
          <td>23.092648</td>
          <td>0.042336</td>
          <td>18.831183</td>
          <td>22.710373</td>
          <td>23.339270</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.928482</td>
          <td>0.005976</td>
          <td>25.835657</td>
          <td>0.077535</td>
          <td>19.183678</td>
          <td>0.005009</td>
          <td>22.884804</td>
          <td>0.009308</td>
          <td>27.724911</td>
          <td>0.856678</td>
          <td>20.820955</td>
          <td>0.007398</td>
          <td>25.168662</td>
          <td>24.919768</td>
          <td>28.360526</td>
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
          <td>29.260508</td>
          <td>20.861059</td>
          <td>23.083835</td>
          <td>19.901728</td>
          <td>14.318524</td>
          <td>25.662325</td>
          <td>17.145212</td>
          <td>0.005000</td>
          <td>22.953896</td>
          <td>0.010326</td>
          <td>18.158351</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.570752</td>
          <td>22.762881</td>
          <td>22.099434</td>
          <td>20.995434</td>
          <td>30.572345</td>
          <td>24.739955</td>
          <td>22.364050</td>
          <td>0.005844</td>
          <td>18.180336</td>
          <td>0.005001</td>
          <td>21.732630</td>
          <td>0.005800</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.005636</td>
          <td>18.584763</td>
          <td>18.893481</td>
          <td>24.472040</td>
          <td>23.209546</td>
          <td>24.037657</td>
          <td>23.641958</td>
          <td>0.010998</td>
          <td>24.884983</td>
          <td>0.052727</td>
          <td>22.728674</td>
          <td>0.008887</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.025543</td>
          <td>20.901775</td>
          <td>25.530339</td>
          <td>22.802948</td>
          <td>23.498416</td>
          <td>24.073561</td>
          <td>22.638597</td>
          <td>0.006338</td>
          <td>26.402701</td>
          <td>0.198452</td>
          <td>22.085756</td>
          <td>0.006446</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.211077</td>
          <td>29.619616</td>
          <td>17.510367</td>
          <td>22.056824</td>
          <td>26.199241</td>
          <td>19.717672</td>
          <td>25.840180</td>
          <td>0.072320</td>
          <td>25.193920</td>
          <td>0.069411</td>
          <td>25.374160</td>
          <td>0.081432</td>
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
          <td>20.415609</td>
          <td>23.836251</td>
          <td>21.896342</td>
          <td>21.777365</td>
          <td>24.975164</td>
          <td>23.350077</td>
          <td>26.868915</td>
          <td>0.177231</td>
          <td>22.250997</td>
          <td>0.006886</td>
          <td>22.748464</td>
          <td>0.008998</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.918747</td>
          <td>26.100382</td>
          <td>23.672081</td>
          <td>24.880943</td>
          <td>18.856381</td>
          <td>21.753298</td>
          <td>20.799828</td>
          <td>0.005051</td>
          <td>24.229300</td>
          <td>0.029430</td>
          <td>22.638248</td>
          <td>0.008409</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.985319</td>
          <td>15.568970</td>
          <td>25.547296</td>
          <td>21.061630</td>
          <td>24.552481</td>
          <td>22.732129</td>
          <td>25.100267</td>
          <td>0.037424</td>
          <td>28.719237</td>
          <td>1.075700</td>
          <td>25.367486</td>
          <td>0.080953</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.807829</td>
          <td>16.753764</td>
          <td>20.974805</td>
          <td>25.223907</td>
          <td>23.417808</td>
          <td>23.064195</td>
          <td>18.822131</td>
          <td>0.005001</td>
          <td>22.718298</td>
          <td>0.008829</td>
          <td>23.354092</td>
          <td>0.013967</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.926540</td>
          <td>25.870196</td>
          <td>19.178568</td>
          <td>22.884028</td>
          <td>27.367916</td>
          <td>20.820510</td>
          <td>25.135381</td>
          <td>0.038612</td>
          <td>24.891666</td>
          <td>0.053042</td>
          <td>28.975406</td>
          <td>1.243653</td>
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
          <td>29.260508</td>
          <td>20.861059</td>
          <td>23.083835</td>
          <td>19.901728</td>
          <td>14.318524</td>
          <td>25.662325</td>
          <td>17.138471</td>
          <td>0.005009</td>
          <td>23.005315</td>
          <td>0.053691</td>
          <td>18.171973</td>
          <td>0.005049</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.570752</td>
          <td>22.762881</td>
          <td>22.099434</td>
          <td>20.995434</td>
          <td>30.572345</td>
          <td>24.739955</td>
          <td>22.359537</td>
          <td>0.036093</td>
          <td>18.178753</td>
          <td>0.005041</td>
          <td>21.741540</td>
          <td>0.019252</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.005636</td>
          <td>18.584763</td>
          <td>18.893481</td>
          <td>24.472040</td>
          <td>23.209546</td>
          <td>24.037657</td>
          <td>23.657553</td>
          <td>0.114062</td>
          <td>24.899211</td>
          <td>0.275561</td>
          <td>22.711101</td>
          <td>0.045156</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.025543</td>
          <td>20.901775</td>
          <td>25.530339</td>
          <td>22.802948</td>
          <td>23.498416</td>
          <td>24.073561</td>
          <td>22.615853</td>
          <td>0.045347</td>
          <td>25.781368</td>
          <td>0.544375</td>
          <td>22.069504</td>
          <td>0.025570</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.211077</td>
          <td>29.619616</td>
          <td>17.510367</td>
          <td>22.056824</td>
          <td>26.199241</td>
          <td>19.717672</td>
          <td>26.830992</td>
          <td>1.213564</td>
          <td>25.068700</td>
          <td>0.315909</td>
          <td>25.386882</td>
          <td>0.437594</td>
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
          <td>20.415609</td>
          <td>23.836251</td>
          <td>21.896342</td>
          <td>21.777365</td>
          <td>24.975164</td>
          <td>23.350077</td>
          <td>27.198247</td>
          <td>1.473818</td>
          <td>22.272787</td>
          <td>0.027999</td>
          <td>22.701301</td>
          <td>0.044763</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.918747</td>
          <td>26.100382</td>
          <td>23.672081</td>
          <td>24.880943</td>
          <td>18.856381</td>
          <td>21.753298</td>
          <td>20.794152</td>
          <td>0.009907</td>
          <td>24.078981</td>
          <td>0.138247</td>
          <td>22.649627</td>
          <td>0.042748</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.985319</td>
          <td>15.568970</td>
          <td>25.547296</td>
          <td>21.061630</td>
          <td>24.552481</td>
          <td>22.732129</td>
          <td>24.848576</td>
          <td>0.310865</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.017850</td>
          <td>0.689726</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.807829</td>
          <td>16.753764</td>
          <td>20.974805</td>
          <td>25.223907</td>
          <td>23.417808</td>
          <td>23.064195</td>
          <td>18.819346</td>
          <td>0.005190</td>
          <td>22.639653</td>
          <td>0.038759</td>
          <td>23.323770</td>
          <td>0.077881</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.926540</td>
          <td>25.870196</td>
          <td>19.178568</td>
          <td>22.884028</td>
          <td>27.367916</td>
          <td>20.820510</td>
          <td>25.382133</td>
          <td>0.470123</td>
          <td>24.991999</td>
          <td>0.297054</td>
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


