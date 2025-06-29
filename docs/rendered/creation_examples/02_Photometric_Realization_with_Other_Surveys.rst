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
          <td>22.670787</td>
          <td>28.624560</td>
          <td>20.195162</td>
          <td>20.698662</td>
          <td>25.822467</td>
          <td>20.491963</td>
          <td>25.838397</td>
          <td>23.628945</td>
          <td>21.227209</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.909416</td>
          <td>21.792206</td>
          <td>23.798500</td>
          <td>25.908272</td>
          <td>20.520896</td>
          <td>21.429594</td>
          <td>20.508867</td>
          <td>20.829932</td>
          <td>16.668617</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536435</td>
          <td>19.063426</td>
          <td>27.660839</td>
          <td>17.973805</td>
          <td>19.830040</td>
          <td>20.253452</td>
          <td>24.021735</td>
          <td>20.935994</td>
          <td>22.208526</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.099706</td>
          <td>23.576044</td>
          <td>21.527266</td>
          <td>22.517196</td>
          <td>21.755067</td>
          <td>23.458903</td>
          <td>23.497548</td>
          <td>23.070335</td>
          <td>23.019641</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.660859</td>
          <td>19.051882</td>
          <td>24.008997</td>
          <td>22.783294</td>
          <td>25.426481</td>
          <td>22.265436</td>
          <td>22.657447</td>
          <td>28.765785</td>
          <td>20.725367</td>
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
          <td>24.174597</td>
          <td>25.875369</td>
          <td>27.369675</td>
          <td>20.142432</td>
          <td>27.199025</td>
          <td>25.769922</td>
          <td>22.100905</td>
          <td>26.365384</td>
          <td>24.375301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.375442</td>
          <td>23.865021</td>
          <td>28.209448</td>
          <td>23.565754</td>
          <td>22.923628</td>
          <td>22.877856</td>
          <td>17.919013</td>
          <td>18.606234</td>
          <td>23.790529</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.783605</td>
          <td>19.867189</td>
          <td>19.053849</td>
          <td>22.032866</td>
          <td>19.631967</td>
          <td>25.458199</td>
          <td>24.594211</td>
          <td>19.954764</td>
          <td>23.082701</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.349646</td>
          <td>28.957708</td>
          <td>15.758355</td>
          <td>22.705440</td>
          <td>24.359148</td>
          <td>25.156255</td>
          <td>26.735582</td>
          <td>17.981427</td>
          <td>23.050265</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.149657</td>
          <td>25.080052</td>
          <td>24.398447</td>
          <td>26.135915</td>
          <td>21.926355</td>
          <td>27.053709</td>
          <td>21.705891</td>
          <td>31.708931</td>
          <td>21.644171</td>
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
          <td>22.672727</td>
          <td>0.014712</td>
          <td>29.289802</td>
          <td>1.106974</td>
          <td>20.197525</td>
          <td>0.005031</td>
          <td>20.694421</td>
          <td>0.005143</td>
          <td>26.264090</td>
          <td>0.295668</td>
          <td>20.492746</td>
          <td>0.006456</td>
          <td>25.838397</td>
          <td>23.628945</td>
          <td>21.227209</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.298248</td>
          <td>0.321719</td>
          <td>21.789681</td>
          <td>0.005472</td>
          <td>23.797911</td>
          <td>0.011945</td>
          <td>25.992478</td>
          <td>0.127185</td>
          <td>20.509166</td>
          <td>0.005345</td>
          <td>21.419498</td>
          <td>0.010557</td>
          <td>20.508867</td>
          <td>20.829932</td>
          <td>16.668617</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.535473</td>
          <td>0.005007</td>
          <td>19.057716</td>
          <td>0.005013</td>
          <td>28.151859</td>
          <td>0.471909</td>
          <td>17.975369</td>
          <td>0.005004</td>
          <td>19.825427</td>
          <td>0.005118</td>
          <td>20.250136</td>
          <td>0.005995</td>
          <td>24.021735</td>
          <td>20.935994</td>
          <td>22.208526</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.078198</td>
          <td>0.048230</td>
          <td>23.564200</td>
          <td>0.011295</td>
          <td>21.525059</td>
          <td>0.005222</td>
          <td>22.507083</td>
          <td>0.007511</td>
          <td>21.752714</td>
          <td>0.007441</td>
          <td>23.479812</td>
          <td>0.059694</td>
          <td>23.497548</td>
          <td>23.070335</td>
          <td>23.019641</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.652080</td>
          <td>0.007666</td>
          <td>19.059247</td>
          <td>0.005013</td>
          <td>24.026218</td>
          <td>0.014234</td>
          <td>22.781770</td>
          <td>0.008733</td>
          <td>25.782549</td>
          <td>0.198841</td>
          <td>22.255238</td>
          <td>0.020371</td>
          <td>22.657447</td>
          <td>28.765785</td>
          <td>20.725367</td>
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
          <td>24.146776</td>
          <td>0.051228</td>
          <td>25.841097</td>
          <td>0.077908</td>
          <td>28.586833</td>
          <td>0.645736</td>
          <td>20.136954</td>
          <td>0.005061</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.430350</td>
          <td>0.316654</td>
          <td>22.100905</td>
          <td>26.365384</td>
          <td>24.375301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.254075</td>
          <td>0.656613</td>
          <td>23.853260</td>
          <td>0.014025</td>
          <td>27.502667</td>
          <td>0.284449</td>
          <td>23.584595</td>
          <td>0.015566</td>
          <td>22.931078</td>
          <td>0.016619</td>
          <td>22.807040</td>
          <td>0.032887</td>
          <td>17.919013</td>
          <td>18.606234</td>
          <td>23.790529</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.776637</td>
          <td>0.005790</td>
          <td>19.869758</td>
          <td>0.005032</td>
          <td>19.051371</td>
          <td>0.005007</td>
          <td>22.028925</td>
          <td>0.006207</td>
          <td>19.631384</td>
          <td>0.005088</td>
          <td>24.957426</td>
          <td>0.215149</td>
          <td>24.594211</td>
          <td>19.954764</td>
          <td>23.082701</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.127276</td>
          <td>0.280461</td>
          <td>inf</td>
          <td>inf</td>
          <td>15.758937</td>
          <td>0.005000</td>
          <td>22.698181</td>
          <td>0.008316</td>
          <td>24.420300</td>
          <td>0.060828</td>
          <td>25.750260</td>
          <td>0.406845</td>
          <td>26.735582</td>
          <td>17.981427</td>
          <td>23.050265</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.175485</td>
          <td>0.052538</td>
          <td>25.103138</td>
          <td>0.040556</td>
          <td>24.415011</td>
          <td>0.019584</td>
          <td>25.850275</td>
          <td>0.112395</td>
          <td>21.925313</td>
          <td>0.008140</td>
          <td>26.683477</td>
          <td>0.791719</td>
          <td>21.705891</td>
          <td>31.708931</td>
          <td>21.644171</td>
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
          <td>22.670787</td>
          <td>28.624560</td>
          <td>20.195162</td>
          <td>20.698662</td>
          <td>25.822467</td>
          <td>20.491963</td>
          <td>25.781018</td>
          <td>0.068621</td>
          <td>23.639397</td>
          <td>0.017657</td>
          <td>21.227492</td>
          <td>0.005330</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.909416</td>
          <td>21.792206</td>
          <td>23.798500</td>
          <td>25.908272</td>
          <td>20.520896</td>
          <td>21.429594</td>
          <td>20.505752</td>
          <td>0.005030</td>
          <td>20.828352</td>
          <td>0.005161</td>
          <td>16.665749</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536435</td>
          <td>19.063426</td>
          <td>27.660839</td>
          <td>17.973805</td>
          <td>19.830040</td>
          <td>20.253452</td>
          <td>24.012279</td>
          <td>0.014636</td>
          <td>20.940165</td>
          <td>0.005197</td>
          <td>22.206303</td>
          <td>0.006757</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.099706</td>
          <td>23.576044</td>
          <td>21.527266</td>
          <td>22.517196</td>
          <td>21.755067</td>
          <td>23.458903</td>
          <td>23.505876</td>
          <td>0.009987</td>
          <td>23.079697</td>
          <td>0.011306</td>
          <td>23.020913</td>
          <td>0.010831</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.660859</td>
          <td>19.051882</td>
          <td>24.008997</td>
          <td>22.783294</td>
          <td>25.426481</td>
          <td>22.265436</td>
          <td>22.655206</td>
          <td>0.006375</td>
          <td>28.400079</td>
          <td>0.886239</td>
          <td>20.724961</td>
          <td>0.005133</td>
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
          <td>24.174597</td>
          <td>25.875369</td>
          <td>27.369675</td>
          <td>20.142432</td>
          <td>27.199025</td>
          <td>25.769922</td>
          <td>22.101041</td>
          <td>0.005535</td>
          <td>26.883083</td>
          <td>0.294927</td>
          <td>24.343667</td>
          <td>0.032564</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.375442</td>
          <td>23.865021</td>
          <td>28.209448</td>
          <td>23.565754</td>
          <td>22.923628</td>
          <td>22.877856</td>
          <td>17.920241</td>
          <td>0.005000</td>
          <td>18.606924</td>
          <td>0.005003</td>
          <td>23.812423</td>
          <td>0.020455</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.783605</td>
          <td>19.867189</td>
          <td>19.053849</td>
          <td>22.032866</td>
          <td>19.631967</td>
          <td>25.458199</td>
          <td>24.608659</td>
          <td>0.024245</td>
          <td>19.959308</td>
          <td>0.005033</td>
          <td>23.069164</td>
          <td>0.011219</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.349646</td>
          <td>28.957708</td>
          <td>15.758355</td>
          <td>22.705440</td>
          <td>24.359148</td>
          <td>25.156255</td>
          <td>26.767321</td>
          <td>0.162535</td>
          <td>17.985099</td>
          <td>0.005001</td>
          <td>23.036213</td>
          <td>0.010952</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.149657</td>
          <td>25.080052</td>
          <td>24.398447</td>
          <td>26.135915</td>
          <td>21.926355</td>
          <td>27.053709</td>
          <td>21.703632</td>
          <td>0.005264</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.641831</td>
          <td>0.005684</td>
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
          <td>22.670787</td>
          <td>28.624560</td>
          <td>20.195162</td>
          <td>20.698662</td>
          <td>25.822467</td>
          <td>20.491963</td>
          <td>27.131978</td>
          <td>1.424992</td>
          <td>23.787262</td>
          <td>0.107265</td>
          <td>21.220081</td>
          <td>0.012571</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.909416</td>
          <td>21.792206</td>
          <td>23.798500</td>
          <td>25.908272</td>
          <td>20.520896</td>
          <td>21.429594</td>
          <td>20.515977</td>
          <td>0.008299</td>
          <td>20.829798</td>
          <td>0.008893</td>
          <td>16.661971</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536435</td>
          <td>19.063426</td>
          <td>27.660839</td>
          <td>17.973805</td>
          <td>19.830040</td>
          <td>20.253452</td>
          <td>23.956165</td>
          <td>0.147758</td>
          <td>20.935021</td>
          <td>0.009519</td>
          <td>22.182206</td>
          <td>0.028232</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.099706</td>
          <td>23.576044</td>
          <td>21.527266</td>
          <td>22.517196</td>
          <td>21.755067</td>
          <td>23.458903</td>
          <td>23.304949</td>
          <td>0.083678</td>
          <td>23.051459</td>
          <td>0.055945</td>
          <td>22.933513</td>
          <td>0.055058</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.660859</td>
          <td>19.051882</td>
          <td>24.008997</td>
          <td>22.783294</td>
          <td>25.426481</td>
          <td>22.265436</td>
          <td>22.683190</td>
          <td>0.048153</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.720591</td>
          <td>0.008842</td>
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
          <td>24.174597</td>
          <td>25.875369</td>
          <td>27.369675</td>
          <td>20.142432</td>
          <td>27.199025</td>
          <td>25.769922</td>
          <td>22.097678</td>
          <td>0.028620</td>
          <td>25.314534</td>
          <td>0.383406</td>
          <td>24.303397</td>
          <td>0.182491</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.375442</td>
          <td>23.865021</td>
          <td>28.209448</td>
          <td>23.565754</td>
          <td>22.923628</td>
          <td>22.877856</td>
          <td>17.922806</td>
          <td>0.005037</td>
          <td>18.605967</td>
          <td>0.005089</td>
          <td>23.716698</td>
          <td>0.110063</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.783605</td>
          <td>19.867189</td>
          <td>19.053849</td>
          <td>22.032866</td>
          <td>19.631967</td>
          <td>25.458199</td>
          <td>24.669351</td>
          <td>0.268940</td>
          <td>19.961193</td>
          <td>0.005995</td>
          <td>22.938649</td>
          <td>0.055310</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.349646</td>
          <td>28.957708</td>
          <td>15.758355</td>
          <td>22.705440</td>
          <td>24.359148</td>
          <td>25.156255</td>
          <td>29.499446</td>
          <td>3.496284</td>
          <td>17.981849</td>
          <td>0.005028</td>
          <td>23.150147</td>
          <td>0.066764</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.149657</td>
          <td>25.080052</td>
          <td>24.398447</td>
          <td>26.135915</td>
          <td>21.926355</td>
          <td>27.053709</td>
          <td>21.709182</td>
          <td>0.020399</td>
          <td>28.562764</td>
          <td>2.436968</td>
          <td>21.629971</td>
          <td>0.017517</td>
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


