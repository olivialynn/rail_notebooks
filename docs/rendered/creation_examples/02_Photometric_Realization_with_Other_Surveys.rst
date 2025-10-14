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
          <td>26.437379</td>
          <td>28.948111</td>
          <td>21.849472</td>
          <td>23.754400</td>
          <td>22.330525</td>
          <td>24.946865</td>
          <td>19.829901</td>
          <td>28.143368</td>
          <td>22.216555</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.371690</td>
          <td>21.594629</td>
          <td>20.151776</td>
          <td>22.123165</td>
          <td>24.778483</td>
          <td>25.766884</td>
          <td>27.003458</td>
          <td>19.544163</td>
          <td>27.661635</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.854863</td>
          <td>21.732092</td>
          <td>17.735628</td>
          <td>21.429432</td>
          <td>18.533413</td>
          <td>20.763183</td>
          <td>24.789120</td>
          <td>21.339229</td>
          <td>24.360136</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.545462</td>
          <td>28.024716</td>
          <td>29.555571</td>
          <td>23.035614</td>
          <td>23.632886</td>
          <td>20.170800</td>
          <td>18.815501</td>
          <td>26.886607</td>
          <td>22.897311</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.452517</td>
          <td>18.580834</td>
          <td>19.574018</td>
          <td>24.940038</td>
          <td>25.913060</td>
          <td>24.247743</td>
          <td>16.669600</td>
          <td>21.175648</td>
          <td>21.991510</td>
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
          <td>20.038078</td>
          <td>27.604895</td>
          <td>21.982123</td>
          <td>28.619783</td>
          <td>19.894978</td>
          <td>23.786785</td>
          <td>26.265156</td>
          <td>28.055209</td>
          <td>24.901782</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.708328</td>
          <td>19.271825</td>
          <td>27.604497</td>
          <td>25.158632</td>
          <td>22.950778</td>
          <td>23.220390</td>
          <td>19.394853</td>
          <td>25.111526</td>
          <td>26.091408</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.440150</td>
          <td>21.748634</td>
          <td>23.622945</td>
          <td>23.437177</td>
          <td>20.905790</td>
          <td>19.310665</td>
          <td>18.740203</td>
          <td>24.025863</td>
          <td>25.918762</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.037731</td>
          <td>25.405315</td>
          <td>22.881362</td>
          <td>24.194116</td>
          <td>21.455568</td>
          <td>25.514606</td>
          <td>20.921185</td>
          <td>22.590249</td>
          <td>23.095031</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.222985</td>
          <td>21.233484</td>
          <td>28.863723</td>
          <td>23.526066</td>
          <td>25.933270</td>
          <td>25.288633</td>
          <td>28.318544</td>
          <td>20.903781</td>
          <td>21.646530</td>
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
          <td>26.264904</td>
          <td>0.313282</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.856467</td>
          <td>0.005377</td>
          <td>23.753348</td>
          <td>0.017883</td>
          <td>22.329305</td>
          <td>0.010486</td>
          <td>24.638002</td>
          <td>0.164303</td>
          <td>19.829901</td>
          <td>28.143368</td>
          <td>22.216555</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.894567</td>
          <td>0.508138</td>
          <td>21.598085</td>
          <td>0.005352</td>
          <td>20.151432</td>
          <td>0.005029</td>
          <td>22.116669</td>
          <td>0.006385</td>
          <td>24.788962</td>
          <td>0.084278</td>
          <td>25.694232</td>
          <td>0.389653</td>
          <td>27.003458</td>
          <td>19.544163</td>
          <td>27.661635</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.473642</td>
          <td>0.369374</td>
          <td>21.731334</td>
          <td>0.005432</td>
          <td>17.739196</td>
          <td>0.005002</td>
          <td>21.429900</td>
          <td>0.005462</td>
          <td>18.529491</td>
          <td>0.005019</td>
          <td>20.766536</td>
          <td>0.007211</td>
          <td>24.789120</td>
          <td>21.339229</td>
          <td>24.360136</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.544962</td>
          <td>0.030276</td>
          <td>27.413693</td>
          <td>0.297373</td>
          <td>29.275505</td>
          <td>1.008686</td>
          <td>23.033305</td>
          <td>0.010269</td>
          <td>23.653771</td>
          <td>0.030852</td>
          <td>20.172591</td>
          <td>0.005880</td>
          <td>18.815501</td>
          <td>26.886607</td>
          <td>22.897311</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.445224</td>
          <td>0.005006</td>
          <td>18.583023</td>
          <td>0.005008</td>
          <td>19.573877</td>
          <td>0.005014</td>
          <td>24.930139</td>
          <td>0.049893</td>
          <td>26.148198</td>
          <td>0.269165</td>
          <td>24.250230</td>
          <td>0.117620</td>
          <td>16.669600</td>
          <td>21.175648</td>
          <td>21.991510</td>
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
          <td>20.038122</td>
          <td>0.005291</td>
          <td>27.899588</td>
          <td>0.434934</td>
          <td>21.985245</td>
          <td>0.005464</td>
          <td>27.590686</td>
          <td>0.469245</td>
          <td>19.901290</td>
          <td>0.005132</td>
          <td>23.656904</td>
          <td>0.069838</td>
          <td>26.265156</td>
          <td>28.055209</td>
          <td>24.901782</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.700139</td>
          <td>0.007845</td>
          <td>19.282588</td>
          <td>0.005016</td>
          <td>27.395597</td>
          <td>0.260712</td>
          <td>25.098840</td>
          <td>0.057955</td>
          <td>22.939405</td>
          <td>0.016733</td>
          <td>23.168615</td>
          <td>0.045287</td>
          <td>19.394853</td>
          <td>25.111526</td>
          <td>26.091408</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.409665</td>
          <td>0.064562</td>
          <td>21.747775</td>
          <td>0.005443</td>
          <td>23.625094</td>
          <td>0.010547</td>
          <td>23.442559</td>
          <td>0.013897</td>
          <td>20.915525</td>
          <td>0.005663</td>
          <td>19.303024</td>
          <td>0.005219</td>
          <td>18.740203</td>
          <td>24.025863</td>
          <td>25.918762</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.105146</td>
          <td>0.118619</td>
          <td>25.347920</td>
          <td>0.050365</td>
          <td>22.887922</td>
          <td>0.006940</td>
          <td>24.175613</td>
          <td>0.025643</td>
          <td>21.443895</td>
          <td>0.006529</td>
          <td>26.129669</td>
          <td>0.540168</td>
          <td>20.921185</td>
          <td>22.590249</td>
          <td>23.095031</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.146995</td>
          <td>0.284971</td>
          <td>21.236050</td>
          <td>0.005205</td>
          <td>28.421213</td>
          <td>0.574614</td>
          <td>23.523251</td>
          <td>0.014816</td>
          <td>25.996351</td>
          <td>0.237628</td>
          <td>25.932989</td>
          <td>0.467290</td>
          <td>28.318544</td>
          <td>20.903781</td>
          <td>21.646530</td>
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
          <td>26.437379</td>
          <td>28.948111</td>
          <td>21.849472</td>
          <td>23.754400</td>
          <td>22.330525</td>
          <td>24.946865</td>
          <td>19.827581</td>
          <td>0.005009</td>
          <td>27.273259</td>
          <td>0.401209</td>
          <td>22.206069</td>
          <td>0.006756</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.371690</td>
          <td>21.594629</td>
          <td>20.151776</td>
          <td>22.123165</td>
          <td>24.778483</td>
          <td>25.766884</td>
          <td>27.070012</td>
          <td>0.209984</td>
          <td>19.545538</td>
          <td>0.005015</td>
          <td>30.695627</td>
          <td>2.647390</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.854863</td>
          <td>21.732092</td>
          <td>17.735628</td>
          <td>21.429432</td>
          <td>18.533413</td>
          <td>20.763183</td>
          <td>24.837902</td>
          <td>0.029654</td>
          <td>21.341995</td>
          <td>0.005404</td>
          <td>24.374972</td>
          <td>0.033481</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.545462</td>
          <td>28.024716</td>
          <td>29.555571</td>
          <td>23.035614</td>
          <td>23.632886</td>
          <td>20.170800</td>
          <td>18.821821</td>
          <td>0.005001</td>
          <td>27.145642</td>
          <td>0.363365</td>
          <td>22.902394</td>
          <td>0.009963</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.452517</td>
          <td>18.580834</td>
          <td>19.574018</td>
          <td>24.940038</td>
          <td>25.913060</td>
          <td>24.247743</td>
          <td>16.671084</td>
          <td>0.005000</td>
          <td>21.183376</td>
          <td>0.005305</td>
          <td>22.004774</td>
          <td>0.006265</td>
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
          <td>20.038078</td>
          <td>27.604895</td>
          <td>21.982123</td>
          <td>28.619783</td>
          <td>19.894978</td>
          <td>23.786785</td>
          <td>26.364162</td>
          <td>0.114722</td>
          <td>28.026283</td>
          <td>0.693700</td>
          <td>24.845779</td>
          <td>0.050917</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.708328</td>
          <td>19.271825</td>
          <td>27.604497</td>
          <td>25.158632</td>
          <td>22.950778</td>
          <td>23.220390</td>
          <td>19.394744</td>
          <td>0.005004</td>
          <td>25.024269</td>
          <td>0.059693</td>
          <td>25.988954</td>
          <td>0.139443</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.440150</td>
          <td>21.748634</td>
          <td>23.622945</td>
          <td>23.437177</td>
          <td>20.905790</td>
          <td>19.310665</td>
          <td>18.742555</td>
          <td>0.005001</td>
          <td>24.050232</td>
          <td>0.025142</td>
          <td>25.967679</td>
          <td>0.136903</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.037731</td>
          <td>25.405315</td>
          <td>22.881362</td>
          <td>24.194116</td>
          <td>21.455568</td>
          <td>25.514606</td>
          <td>20.919284</td>
          <td>0.005064</td>
          <td>22.577220</td>
          <td>0.008115</td>
          <td>23.091044</td>
          <td>0.011402</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.222985</td>
          <td>21.233484</td>
          <td>28.863723</td>
          <td>23.526066</td>
          <td>25.933270</td>
          <td>25.288633</td>
          <td>28.091794</td>
          <td>0.473528</td>
          <td>20.904923</td>
          <td>0.005185</td>
          <td>21.657014</td>
          <td>0.005702</td>
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
          <td>26.437379</td>
          <td>28.948111</td>
          <td>21.849472</td>
          <td>23.754400</td>
          <td>22.330525</td>
          <td>24.946865</td>
          <td>19.828693</td>
          <td>0.006115</td>
          <td>28.582584</td>
          <td>2.454705</td>
          <td>22.238790</td>
          <td>0.029678</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.371690</td>
          <td>21.594629</td>
          <td>20.151776</td>
          <td>22.123165</td>
          <td>24.778483</td>
          <td>25.766884</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.542441</td>
          <td>0.005483</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.854863</td>
          <td>21.732092</td>
          <td>17.735628</td>
          <td>21.429432</td>
          <td>18.533413</td>
          <td>20.763183</td>
          <td>24.746978</td>
          <td>0.286449</td>
          <td>21.333162</td>
          <td>0.012699</td>
          <td>24.217350</td>
          <td>0.169625</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.545462</td>
          <td>28.024716</td>
          <td>29.555571</td>
          <td>23.035614</td>
          <td>23.632886</td>
          <td>20.170800</td>
          <td>18.809836</td>
          <td>0.005186</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.844878</td>
          <td>0.050876</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.452517</td>
          <td>18.580834</td>
          <td>19.574018</td>
          <td>24.940038</td>
          <td>25.913060</td>
          <td>24.247743</td>
          <td>16.666214</td>
          <td>0.005004</td>
          <td>21.180641</td>
          <td>0.011314</td>
          <td>21.997204</td>
          <td>0.024004</td>
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
          <td>20.038078</td>
          <td>27.604895</td>
          <td>21.982123</td>
          <td>28.619783</td>
          <td>19.894978</td>
          <td>23.786785</td>
          <td>25.341846</td>
          <td>0.456135</td>
          <td>26.970428</td>
          <td>1.173180</td>
          <td>24.983091</td>
          <td>0.319560</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.708328</td>
          <td>19.271825</td>
          <td>27.604497</td>
          <td>25.158632</td>
          <td>22.950778</td>
          <td>23.220390</td>
          <td>19.396411</td>
          <td>0.005531</td>
          <td>24.831740</td>
          <td>0.260801</td>
          <td>26.868222</td>
          <td>1.171724</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.440150</td>
          <td>21.748634</td>
          <td>23.622945</td>
          <td>23.437177</td>
          <td>20.905790</td>
          <td>19.310665</td>
          <td>18.745837</td>
          <td>0.005166</td>
          <td>24.303704</td>
          <td>0.167663</td>
          <td>25.567058</td>
          <td>0.500730</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.037731</td>
          <td>25.405315</td>
          <td>22.881362</td>
          <td>24.194116</td>
          <td>21.455568</td>
          <td>25.514606</td>
          <td>20.922041</td>
          <td>0.010840</td>
          <td>22.533571</td>
          <td>0.035269</td>
          <td>22.986405</td>
          <td>0.057714</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.222985</td>
          <td>21.233484</td>
          <td>28.863723</td>
          <td>23.526066</td>
          <td>25.933270</td>
          <td>25.288633</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.907804</td>
          <td>0.009350</td>
          <td>21.659143</td>
          <td>0.017953</td>
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


