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
          <td>21.303040</td>
          <td>26.955411</td>
          <td>23.502981</td>
          <td>21.729469</td>
          <td>20.712193</td>
          <td>22.791109</td>
          <td>16.153800</td>
          <td>21.651180</td>
          <td>25.128473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.530605</td>
          <td>20.372370</td>
          <td>25.208681</td>
          <td>24.123378</td>
          <td>17.882258</td>
          <td>22.662770</td>
          <td>26.595836</td>
          <td>25.839274</td>
          <td>21.795158</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.012813</td>
          <td>24.425708</td>
          <td>21.355692</td>
          <td>25.143500</td>
          <td>17.106335</td>
          <td>23.746742</td>
          <td>23.622289</td>
          <td>25.400476</td>
          <td>22.926877</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.544109</td>
          <td>19.565876</td>
          <td>25.103485</td>
          <td>22.575802</td>
          <td>24.271672</td>
          <td>19.358764</td>
          <td>22.457656</td>
          <td>18.089727</td>
          <td>22.161857</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.652786</td>
          <td>28.775541</td>
          <td>22.762351</td>
          <td>29.340570</td>
          <td>20.872940</td>
          <td>23.131804</td>
          <td>17.217919</td>
          <td>22.952624</td>
          <td>24.128698</td>
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
          <td>25.225606</td>
          <td>23.737844</td>
          <td>21.737120</td>
          <td>27.029893</td>
          <td>24.305502</td>
          <td>25.704547</td>
          <td>23.570553</td>
          <td>22.757484</td>
          <td>22.784404</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.856501</td>
          <td>24.344420</td>
          <td>23.210810</td>
          <td>21.193258</td>
          <td>22.321472</td>
          <td>29.756530</td>
          <td>21.695570</td>
          <td>25.660157</td>
          <td>19.944849</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.558759</td>
          <td>25.519549</td>
          <td>23.231673</td>
          <td>23.321073</td>
          <td>19.544652</td>
          <td>20.419525</td>
          <td>25.409073</td>
          <td>21.907412</td>
          <td>24.612423</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.292889</td>
          <td>22.296901</td>
          <td>21.034124</td>
          <td>22.800452</td>
          <td>22.812254</td>
          <td>20.344184</td>
          <td>24.203782</td>
          <td>30.626483</td>
          <td>22.606926</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.937652</td>
          <td>25.457145</td>
          <td>25.845608</td>
          <td>24.391602</td>
          <td>22.024633</td>
          <td>25.482208</td>
          <td>22.626631</td>
          <td>20.378856</td>
          <td>21.789880</td>
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
          <td>21.289215</td>
          <td>0.006617</td>
          <td>27.027482</td>
          <td>0.216680</td>
          <td>23.507684</td>
          <td>0.009739</td>
          <td>21.719735</td>
          <td>0.005737</td>
          <td>20.710772</td>
          <td>0.005477</td>
          <td>22.855521</td>
          <td>0.034323</td>
          <td>16.153800</td>
          <td>21.651180</td>
          <td>25.128473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.549884</td>
          <td>0.173692</td>
          <td>20.371207</td>
          <td>0.005061</td>
          <td>25.232549</td>
          <td>0.039928</td>
          <td>24.140882</td>
          <td>0.024881</td>
          <td>17.877432</td>
          <td>0.005009</td>
          <td>22.672479</td>
          <td>0.029219</td>
          <td>26.595836</td>
          <td>25.839274</td>
          <td>21.795158</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.012448</td>
          <td>0.109451</td>
          <td>24.438024</td>
          <td>0.022696</td>
          <td>21.363912</td>
          <td>0.005172</td>
          <td>25.196744</td>
          <td>0.063214</td>
          <td>17.104773</td>
          <td>0.005004</td>
          <td>23.693152</td>
          <td>0.072114</td>
          <td>23.622289</td>
          <td>25.400476</td>
          <td>22.926877</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.562769</td>
          <td>0.030745</td>
          <td>19.562696</td>
          <td>0.005022</td>
          <td>25.079025</td>
          <td>0.034857</td>
          <td>22.586605</td>
          <td>0.007822</td>
          <td>24.244097</td>
          <td>0.052022</td>
          <td>19.367741</td>
          <td>0.005242</td>
          <td>22.457656</td>
          <td>18.089727</td>
          <td>22.161857</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.665407</td>
          <td>0.005057</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.774768</td>
          <td>0.006632</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.884735</td>
          <td>0.005631</td>
          <td>23.093239</td>
          <td>0.042358</td>
          <td>17.217919</td>
          <td>22.952624</td>
          <td>24.128698</td>
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
          <td>25.167089</td>
          <td>0.125150</td>
          <td>23.753015</td>
          <td>0.012983</td>
          <td>21.738011</td>
          <td>0.005312</td>
          <td>27.020954</td>
          <td>0.301428</td>
          <td>24.248021</td>
          <td>0.052204</td>
          <td>25.868792</td>
          <td>0.445276</td>
          <td>23.570553</td>
          <td>22.757484</td>
          <td>22.784404</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.881361</td>
          <td>0.017343</td>
          <td>24.403248</td>
          <td>0.022032</td>
          <td>23.209268</td>
          <td>0.008122</td>
          <td>21.191964</td>
          <td>0.005314</td>
          <td>22.313299</td>
          <td>0.010371</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.695570</td>
          <td>25.660157</td>
          <td>19.944849</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.584637</td>
          <td>0.013755</td>
          <td>25.493202</td>
          <td>0.057281</td>
          <td>23.225889</td>
          <td>0.008198</td>
          <td>23.302146</td>
          <td>0.012471</td>
          <td>19.549586</td>
          <td>0.005078</td>
          <td>20.415712</td>
          <td>0.006292</td>
          <td>25.409073</td>
          <td>21.907412</td>
          <td>24.612423</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.287910</td>
          <td>0.011107</td>
          <td>22.290956</td>
          <td>0.006022</td>
          <td>21.032340</td>
          <td>0.005103</td>
          <td>22.798921</td>
          <td>0.008824</td>
          <td>22.848253</td>
          <td>0.015534</td>
          <td>20.343493</td>
          <td>0.006153</td>
          <td>24.203782</td>
          <td>30.626483</td>
          <td>22.606926</td>
        </tr>
        <tr>
          <th>999</th>
          <td>inf</td>
          <td>inf</td>
          <td>25.521526</td>
          <td>0.058736</td>
          <td>25.814890</td>
          <td>0.066939</td>
          <td>24.373103</td>
          <td>0.030476</td>
          <td>22.012618</td>
          <td>0.008556</td>
          <td>26.513967</td>
          <td>0.707287</td>
          <td>22.626631</td>
          <td>20.378856</td>
          <td>21.789880</td>
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
          <td>21.303040</td>
          <td>26.955411</td>
          <td>23.502981</td>
          <td>21.729469</td>
          <td>20.712193</td>
          <td>22.791109</td>
          <td>16.154349</td>
          <td>0.005000</td>
          <td>21.652013</td>
          <td>0.005696</td>
          <td>25.153355</td>
          <td>0.066955</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.530605</td>
          <td>20.372370</td>
          <td>25.208681</td>
          <td>24.123378</td>
          <td>17.882258</td>
          <td>22.662770</td>
          <td>26.616565</td>
          <td>0.142804</td>
          <td>25.770368</td>
          <td>0.115344</td>
          <td>21.791988</td>
          <td>0.005885</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.012813</td>
          <td>24.425708</td>
          <td>21.355692</td>
          <td>25.143500</td>
          <td>17.106335</td>
          <td>23.746742</td>
          <td>23.624648</td>
          <td>0.010860</td>
          <td>25.531697</td>
          <td>0.093574</td>
          <td>22.925379</td>
          <td>0.010122</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.544109</td>
          <td>19.565876</td>
          <td>25.103485</td>
          <td>22.575802</td>
          <td>24.271672</td>
          <td>19.358764</td>
          <td>22.464032</td>
          <td>0.006000</td>
          <td>18.090637</td>
          <td>0.005001</td>
          <td>22.153361</td>
          <td>0.006613</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.652786</td>
          <td>28.775541</td>
          <td>22.762351</td>
          <td>29.340570</td>
          <td>20.872940</td>
          <td>23.131804</td>
          <td>17.222562</td>
          <td>0.005000</td>
          <td>22.950881</td>
          <td>0.010304</td>
          <td>24.140025</td>
          <td>0.027203</td>
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
          <td>25.225606</td>
          <td>23.737844</td>
          <td>21.737120</td>
          <td>27.029893</td>
          <td>24.305502</td>
          <td>25.704547</td>
          <td>23.572451</td>
          <td>0.010462</td>
          <td>22.753856</td>
          <td>0.009029</td>
          <td>22.772739</td>
          <td>0.009139</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.856501</td>
          <td>24.344420</td>
          <td>23.210810</td>
          <td>21.193258</td>
          <td>22.321472</td>
          <td>29.756530</td>
          <td>21.701397</td>
          <td>0.005263</td>
          <td>25.591013</td>
          <td>0.098584</td>
          <td>19.947844</td>
          <td>0.005032</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.558759</td>
          <td>25.519549</td>
          <td>23.231673</td>
          <td>23.321073</td>
          <td>19.544652</td>
          <td>20.419525</td>
          <td>25.441577</td>
          <td>0.050726</td>
          <td>21.919084</td>
          <td>0.006097</td>
          <td>24.629598</td>
          <td>0.041992</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.292889</td>
          <td>22.296901</td>
          <td>21.034124</td>
          <td>22.800452</td>
          <td>22.812254</td>
          <td>20.344184</td>
          <td>24.202359</td>
          <td>0.017116</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.602230</td>
          <td>0.008233</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.937652</td>
          <td>25.457145</td>
          <td>25.845608</td>
          <td>24.391602</td>
          <td>22.024633</td>
          <td>25.482208</td>
          <td>22.633726</td>
          <td>0.006327</td>
          <td>20.378393</td>
          <td>0.005071</td>
          <td>21.786266</td>
          <td>0.005877</td>
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
          <td>21.303040</td>
          <td>26.955411</td>
          <td>23.502981</td>
          <td>21.729469</td>
          <td>20.712193</td>
          <td>22.791109</td>
          <td>16.158726</td>
          <td>0.005001</td>
          <td>21.655164</td>
          <td>0.016456</td>
          <td>25.082648</td>
          <td>0.345819</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.530605</td>
          <td>20.372370</td>
          <td>25.208681</td>
          <td>24.123378</td>
          <td>17.882258</td>
          <td>22.662770</td>
          <td>30.755545</td>
          <td>4.722234</td>
          <td>25.483097</td>
          <td>0.436340</td>
          <td>21.789136</td>
          <td>0.020051</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.012813</td>
          <td>24.425708</td>
          <td>21.355692</td>
          <td>25.143500</td>
          <td>17.106335</td>
          <td>23.746742</td>
          <td>23.513254</td>
          <td>0.100528</td>
          <td>25.078378</td>
          <td>0.318360</td>
          <td>22.970352</td>
          <td>0.056895</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.544109</td>
          <td>19.565876</td>
          <td>25.103485</td>
          <td>22.575802</td>
          <td>24.271672</td>
          <td>19.358764</td>
          <td>22.411279</td>
          <td>0.037792</td>
          <td>18.090786</td>
          <td>0.005035</td>
          <td>22.184832</td>
          <td>0.028298</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.652786</td>
          <td>28.775541</td>
          <td>22.762351</td>
          <td>29.340570</td>
          <td>20.872940</td>
          <td>23.131804</td>
          <td>17.220008</td>
          <td>0.005010</td>
          <td>22.905397</td>
          <td>0.049116</td>
          <td>24.087731</td>
          <td>0.151822</td>
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
          <td>25.225606</td>
          <td>23.737844</td>
          <td>21.737120</td>
          <td>27.029893</td>
          <td>24.305502</td>
          <td>25.704547</td>
          <td>23.470001</td>
          <td>0.096780</td>
          <td>22.697251</td>
          <td>0.040799</td>
          <td>22.735083</td>
          <td>0.046132</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.856501</td>
          <td>24.344420</td>
          <td>23.210810</td>
          <td>21.193258</td>
          <td>22.321472</td>
          <td>29.756530</td>
          <td>21.684158</td>
          <td>0.019966</td>
          <td>25.917508</td>
          <td>0.600109</td>
          <td>19.942046</td>
          <td>0.006140</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.558759</td>
          <td>25.519549</td>
          <td>23.231673</td>
          <td>23.321073</td>
          <td>19.544652</td>
          <td>20.419525</td>
          <td>24.870433</td>
          <td>0.316347</td>
          <td>21.862321</td>
          <td>0.019596</td>
          <td>24.488185</td>
          <td>0.213201</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.292889</td>
          <td>22.296901</td>
          <td>21.034124</td>
          <td>22.800452</td>
          <td>22.812254</td>
          <td>20.344184</td>
          <td>24.081790</td>
          <td>0.164557</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.584302</td>
          <td>0.040331</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.937652</td>
          <td>25.457145</td>
          <td>25.845608</td>
          <td>24.391602</td>
          <td>22.024633</td>
          <td>25.482208</td>
          <td>22.642277</td>
          <td>0.046428</td>
          <td>20.376862</td>
          <td>0.006965</td>
          <td>21.789819</td>
          <td>0.020063</td>
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


