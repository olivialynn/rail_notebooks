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
          <td>23.740936</td>
          <td>13.685681</td>
          <td>20.799086</td>
          <td>25.984786</td>
          <td>24.035178</td>
          <td>22.186010</td>
          <td>17.579744</td>
          <td>20.581030</td>
          <td>26.739341</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.229630</td>
          <td>29.827841</td>
          <td>16.251403</td>
          <td>30.157393</td>
          <td>20.301416</td>
          <td>21.410122</td>
          <td>16.687863</td>
          <td>22.717470</td>
          <td>28.175532</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.732310</td>
          <td>24.157149</td>
          <td>22.009583</td>
          <td>21.614722</td>
          <td>24.574476</td>
          <td>27.992491</td>
          <td>26.855688</td>
          <td>24.301315</td>
          <td>21.626626</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.944161</td>
          <td>23.001021</td>
          <td>27.171725</td>
          <td>21.701175</td>
          <td>25.177037</td>
          <td>23.904951</td>
          <td>15.365173</td>
          <td>22.123339</td>
          <td>18.710954</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.915262</td>
          <td>24.780101</td>
          <td>25.078882</td>
          <td>25.590802</td>
          <td>26.734049</td>
          <td>23.556755</td>
          <td>23.669190</td>
          <td>22.970982</td>
          <td>26.931432</td>
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
          <td>24.674166</td>
          <td>22.423542</td>
          <td>19.297054</td>
          <td>25.397888</td>
          <td>18.475651</td>
          <td>16.123450</td>
          <td>23.428321</td>
          <td>22.425674</td>
          <td>24.241922</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.597389</td>
          <td>25.233722</td>
          <td>25.906116</td>
          <td>23.002804</td>
          <td>25.547716</td>
          <td>21.603156</td>
          <td>21.871898</td>
          <td>20.820311</td>
          <td>16.894532</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.369852</td>
          <td>30.145877</td>
          <td>19.599532</td>
          <td>22.038286</td>
          <td>21.040481</td>
          <td>25.129321</td>
          <td>29.108986</td>
          <td>23.982886</td>
          <td>19.120240</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.534840</td>
          <td>21.779648</td>
          <td>25.555759</td>
          <td>25.361484</td>
          <td>26.295411</td>
          <td>21.958973</td>
          <td>25.132905</td>
          <td>21.808320</td>
          <td>22.785123</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.099293</td>
          <td>20.854772</td>
          <td>23.914252</td>
          <td>22.384139</td>
          <td>19.045957</td>
          <td>25.916404</td>
          <td>19.706185</td>
          <td>22.781055</td>
          <td>20.290277</td>
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
          <td>23.748776</td>
          <td>0.036138</td>
          <td>13.689280</td>
          <td>0.005000</td>
          <td>20.795971</td>
          <td>0.005072</td>
          <td>26.106743</td>
          <td>0.140387</td>
          <td>23.974243</td>
          <td>0.040945</td>
          <td>22.177100</td>
          <td>0.019069</td>
          <td>17.579744</td>
          <td>20.581030</td>
          <td>26.739341</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.239232</td>
          <td>0.005379</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.241840</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.306921</td>
          <td>0.005250</td>
          <td>21.425841</td>
          <td>0.010604</td>
          <td>16.687863</td>
          <td>22.717470</td>
          <td>28.175532</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.665866</td>
          <td>0.080861</td>
          <td>24.175944</td>
          <td>0.018195</td>
          <td>22.011775</td>
          <td>0.005485</td>
          <td>21.618062</td>
          <td>0.005626</td>
          <td>24.528922</td>
          <td>0.066976</td>
          <td>25.800327</td>
          <td>0.422732</td>
          <td>26.855688</td>
          <td>24.301315</td>
          <td>21.626626</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.014881</td>
          <td>0.109683</td>
          <td>23.007920</td>
          <td>0.007967</td>
          <td>27.674387</td>
          <td>0.326468</td>
          <td>21.699068</td>
          <td>0.005713</td>
          <td>25.210631</td>
          <td>0.121903</td>
          <td>23.940013</td>
          <td>0.089660</td>
          <td>15.365173</td>
          <td>22.123339</td>
          <td>18.710954</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.913976</td>
          <td>0.005026</td>
          <td>24.740988</td>
          <td>0.029496</td>
          <td>25.060624</td>
          <td>0.034295</td>
          <td>25.741101</td>
          <td>0.102169</td>
          <td>26.374406</td>
          <td>0.322979</td>
          <td>23.577625</td>
          <td>0.065102</td>
          <td>23.669190</td>
          <td>22.970982</td>
          <td>26.931432</td>
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
          <td>24.649117</td>
          <td>0.079682</td>
          <td>22.417828</td>
          <td>0.006241</td>
          <td>19.299309</td>
          <td>0.005010</td>
          <td>25.428445</td>
          <td>0.077603</td>
          <td>18.476980</td>
          <td>0.005018</td>
          <td>16.123375</td>
          <td>0.005003</td>
          <td>23.428321</td>
          <td>22.425674</td>
          <td>24.241922</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.634787</td>
          <td>0.032725</td>
          <td>25.203251</td>
          <td>0.044310</td>
          <td>25.899131</td>
          <td>0.072122</td>
          <td>23.002492</td>
          <td>0.010056</td>
          <td>25.801400</td>
          <td>0.202014</td>
          <td>21.611893</td>
          <td>0.012131</td>
          <td>21.871898</td>
          <td>20.820311</td>
          <td>16.894532</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.217613</td>
          <td>0.130729</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.601840</td>
          <td>0.005014</td>
          <td>22.032555</td>
          <td>0.006214</td>
          <td>21.043098</td>
          <td>0.005813</td>
          <td>25.336443</td>
          <td>0.293673</td>
          <td>29.108986</td>
          <td>23.982886</td>
          <td>19.120240</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.458942</td>
          <td>0.067422</td>
          <td>21.776873</td>
          <td>0.005463</td>
          <td>25.460774</td>
          <td>0.048889</td>
          <td>25.365172</td>
          <td>0.073382</td>
          <td>26.028908</td>
          <td>0.244098</td>
          <td>21.968001</td>
          <td>0.016035</td>
          <td>25.132905</td>
          <td>21.808320</td>
          <td>22.785123</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.097192</td>
          <td>0.005314</td>
          <td>20.857618</td>
          <td>0.005119</td>
          <td>23.897082</td>
          <td>0.012873</td>
          <td>22.382696</td>
          <td>0.007085</td>
          <td>19.042687</td>
          <td>0.005038</td>
          <td>25.652945</td>
          <td>0.377375</td>
          <td>19.706185</td>
          <td>22.781055</td>
          <td>20.290277</td>
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
          <td>23.740936</td>
          <td>13.685681</td>
          <td>20.799086</td>
          <td>25.984786</td>
          <td>24.035178</td>
          <td>22.186010</td>
          <td>17.586883</td>
          <td>0.005000</td>
          <td>20.587118</td>
          <td>0.005104</td>
          <td>26.562695</td>
          <td>0.226859</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.229630</td>
          <td>29.827841</td>
          <td>16.251403</td>
          <td>30.157393</td>
          <td>20.301416</td>
          <td>21.410122</td>
          <td>16.683537</td>
          <td>0.005000</td>
          <td>22.721084</td>
          <td>0.008844</td>
          <td>27.590436</td>
          <td>0.509425</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.732310</td>
          <td>24.157149</td>
          <td>22.009583</td>
          <td>21.614722</td>
          <td>24.574476</td>
          <td>27.992491</td>
          <td>26.700382</td>
          <td>0.153480</td>
          <td>24.345261</td>
          <td>0.032610</td>
          <td>21.624620</td>
          <td>0.005664</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.944161</td>
          <td>23.001021</td>
          <td>27.171725</td>
          <td>21.701175</td>
          <td>25.177037</td>
          <td>23.904951</td>
          <td>15.362866</td>
          <td>0.005000</td>
          <td>22.121150</td>
          <td>0.006531</td>
          <td>18.704331</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.915262</td>
          <td>24.780101</td>
          <td>25.078882</td>
          <td>25.590802</td>
          <td>26.734049</td>
          <td>23.556755</td>
          <td>23.674970</td>
          <td>0.011267</td>
          <td>22.986646</td>
          <td>0.010568</td>
          <td>27.445211</td>
          <td>0.457290</td>
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
          <td>24.674166</td>
          <td>22.423542</td>
          <td>19.297054</td>
          <td>25.397888</td>
          <td>18.475651</td>
          <td>16.123450</td>
          <td>23.440541</td>
          <td>0.009554</td>
          <td>22.432399</td>
          <td>0.007504</td>
          <td>24.270007</td>
          <td>0.030508</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.597389</td>
          <td>25.233722</td>
          <td>25.906116</td>
          <td>23.002804</td>
          <td>25.547716</td>
          <td>21.603156</td>
          <td>21.871293</td>
          <td>0.005357</td>
          <td>20.825038</td>
          <td>0.005160</td>
          <td>16.888249</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.369852</td>
          <td>30.145877</td>
          <td>19.599532</td>
          <td>22.038286</td>
          <td>21.040481</td>
          <td>25.129321</td>
          <td>28.267506</td>
          <td>0.538930</td>
          <td>24.026070</td>
          <td>0.024617</td>
          <td>19.107652</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.534840</td>
          <td>21.779648</td>
          <td>25.555759</td>
          <td>25.361484</td>
          <td>26.295411</td>
          <td>21.958973</td>
          <td>25.154712</td>
          <td>0.039282</td>
          <td>21.810159</td>
          <td>0.005913</td>
          <td>22.790798</td>
          <td>0.009247</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.099293</td>
          <td>20.854772</td>
          <td>23.914252</td>
          <td>22.384139</td>
          <td>19.045957</td>
          <td>25.916404</td>
          <td>19.702275</td>
          <td>0.005007</td>
          <td>22.781855</td>
          <td>0.009193</td>
          <td>20.285819</td>
          <td>0.005060</td>
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
          <td>23.740936</td>
          <td>13.685681</td>
          <td>20.799086</td>
          <td>25.984786</td>
          <td>24.035178</td>
          <td>22.186010</td>
          <td>17.589976</td>
          <td>0.005020</td>
          <td>20.584937</td>
          <td>0.007712</td>
          <td>25.968397</td>
          <td>0.666754</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.229630</td>
          <td>29.827841</td>
          <td>16.251403</td>
          <td>30.157393</td>
          <td>20.301416</td>
          <td>21.410122</td>
          <td>16.683627</td>
          <td>0.005004</td>
          <td>22.770503</td>
          <td>0.043551</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.732310</td>
          <td>24.157149</td>
          <td>22.009583</td>
          <td>21.614722</td>
          <td>24.574476</td>
          <td>27.992491</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.166664</td>
          <td>0.149098</td>
          <td>21.639365</td>
          <td>0.017656</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.944161</td>
          <td>23.001021</td>
          <td>27.171725</td>
          <td>21.701175</td>
          <td>25.177037</td>
          <td>23.904951</td>
          <td>15.361205</td>
          <td>0.005000</td>
          <td>22.135443</td>
          <td>0.024819</td>
          <td>18.701255</td>
          <td>0.005128</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.915262</td>
          <td>24.780101</td>
          <td>25.078882</td>
          <td>25.590802</td>
          <td>26.734049</td>
          <td>23.556755</td>
          <td>23.468822</td>
          <td>0.096680</td>
          <td>22.940810</td>
          <td>0.050692</td>
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
          <td>24.674166</td>
          <td>22.423542</td>
          <td>19.297054</td>
          <td>25.397888</td>
          <td>18.475651</td>
          <td>16.123450</td>
          <td>23.373638</td>
          <td>0.088908</td>
          <td>22.389958</td>
          <td>0.031051</td>
          <td>24.161729</td>
          <td>0.161760</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.597389</td>
          <td>25.233722</td>
          <td>25.906116</td>
          <td>23.002804</td>
          <td>25.547716</td>
          <td>21.603156</td>
          <td>21.854116</td>
          <td>0.023120</td>
          <td>20.804767</td>
          <td>0.008755</td>
          <td>16.894033</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.369852</td>
          <td>30.145877</td>
          <td>19.599532</td>
          <td>22.038286</td>
          <td>21.040481</td>
          <td>25.129321</td>
          <td>26.288473</td>
          <td>0.879779</td>
          <td>23.998682</td>
          <td>0.128964</td>
          <td>19.121916</td>
          <td>0.005273</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.534840</td>
          <td>21.779648</td>
          <td>25.555759</td>
          <td>25.361484</td>
          <td>26.295411</td>
          <td>21.958973</td>
          <td>25.111287</td>
          <td>0.382441</td>
          <td>21.812978</td>
          <td>0.018790</td>
          <td>22.785676</td>
          <td>0.048260</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.099293</td>
          <td>20.854772</td>
          <td>23.914252</td>
          <td>22.384139</td>
          <td>19.045957</td>
          <td>25.916404</td>
          <td>19.701317</td>
          <td>0.005899</td>
          <td>22.815035</td>
          <td>0.045314</td>
          <td>20.297797</td>
          <td>0.007031</td>
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


