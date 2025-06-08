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
          <td>23.372259</td>
          <td>24.093846</td>
          <td>26.501633</td>
          <td>22.383456</td>
          <td>25.511400</td>
          <td>24.363360</td>
          <td>22.807514</td>
          <td>22.987921</td>
          <td>27.069194</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.090670</td>
          <td>20.907259</td>
          <td>27.286319</td>
          <td>22.707655</td>
          <td>21.318102</td>
          <td>27.221491</td>
          <td>23.557872</td>
          <td>15.055855</td>
          <td>27.846296</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.783997</td>
          <td>22.102058</td>
          <td>17.882944</td>
          <td>23.404825</td>
          <td>21.618610</td>
          <td>29.610300</td>
          <td>23.179962</td>
          <td>26.220121</td>
          <td>27.932926</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.999183</td>
          <td>20.327469</td>
          <td>21.099019</td>
          <td>30.548210</td>
          <td>24.615194</td>
          <td>23.843024</td>
          <td>21.843345</td>
          <td>24.202709</td>
          <td>22.683166</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.286777</td>
          <td>22.342359</td>
          <td>18.556179</td>
          <td>23.627634</td>
          <td>24.866671</td>
          <td>23.025616</td>
          <td>22.080503</td>
          <td>21.687553</td>
          <td>25.059799</td>
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
          <td>24.520540</td>
          <td>23.648429</td>
          <td>25.016488</td>
          <td>21.161005</td>
          <td>28.542051</td>
          <td>21.651750</td>
          <td>22.893903</td>
          <td>17.242472</td>
          <td>22.116446</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.026586</td>
          <td>21.327264</td>
          <td>27.735674</td>
          <td>18.158788</td>
          <td>24.579845</td>
          <td>27.230707</td>
          <td>21.460429</td>
          <td>24.206564</td>
          <td>23.396535</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.664960</td>
          <td>20.656846</td>
          <td>22.177050</td>
          <td>23.938074</td>
          <td>20.491606</td>
          <td>20.712607</td>
          <td>24.829158</td>
          <td>19.088020</td>
          <td>20.198326</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.770342</td>
          <td>24.630636</td>
          <td>21.293152</td>
          <td>18.495248</td>
          <td>22.123258</td>
          <td>22.054154</td>
          <td>24.406425</td>
          <td>25.940808</td>
          <td>25.721227</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.247313</td>
          <td>20.435847</td>
          <td>24.664137</td>
          <td>27.633619</td>
          <td>28.174137</td>
          <td>24.805029</td>
          <td>22.183800</td>
          <td>21.670169</td>
          <td>23.237695</td>
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
          <td>23.370425</td>
          <td>0.026059</td>
          <td>24.092709</td>
          <td>0.016989</td>
          <td>26.680870</td>
          <td>0.142894</td>
          <td>22.389997</td>
          <td>0.007108</td>
          <td>25.416256</td>
          <td>0.145612</td>
          <td>24.442722</td>
          <td>0.138963</td>
          <td>22.807514</td>
          <td>22.987921</td>
          <td>27.069194</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.080708</td>
          <td>0.009700</td>
          <td>20.905868</td>
          <td>0.005127</td>
          <td>27.594025</td>
          <td>0.306178</td>
          <td>22.713036</td>
          <td>0.008387</td>
          <td>21.315951</td>
          <td>0.006253</td>
          <td>25.897053</td>
          <td>0.454862</td>
          <td>23.557872</td>
          <td>15.055855</td>
          <td>27.846296</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.791624</td>
          <td>0.005212</td>
          <td>22.101407</td>
          <td>0.005763</td>
          <td>17.892649</td>
          <td>0.005002</td>
          <td>23.404387</td>
          <td>0.013489</td>
          <td>21.614451</td>
          <td>0.006985</td>
          <td>29.924311</td>
          <td>3.367524</td>
          <td>23.179962</td>
          <td>26.220121</td>
          <td>27.932926</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.992175</td>
          <td>0.006067</td>
          <td>20.322661</td>
          <td>0.005057</td>
          <td>21.105806</td>
          <td>0.005115</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.635590</td>
          <td>0.073606</td>
          <td>23.929089</td>
          <td>0.088802</td>
          <td>21.843345</td>
          <td>24.202709</td>
          <td>22.683166</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.564384</td>
          <td>0.808367</td>
          <td>22.341972</td>
          <td>0.006105</td>
          <td>18.558060</td>
          <td>0.005004</td>
          <td>23.631817</td>
          <td>0.016176</td>
          <td>24.974794</td>
          <td>0.099230</td>
          <td>22.988285</td>
          <td>0.038597</td>
          <td>22.080503</td>
          <td>21.687553</td>
          <td>25.059799</td>
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
          <td>24.504998</td>
          <td>0.070208</td>
          <td>23.630470</td>
          <td>0.011849</td>
          <td>25.063390</td>
          <td>0.034379</td>
          <td>21.156746</td>
          <td>0.005297</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.644165</td>
          <td>0.012428</td>
          <td>22.893903</td>
          <td>17.242472</td>
          <td>22.116446</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.285331</td>
          <td>0.318428</td>
          <td>21.323369</td>
          <td>0.005233</td>
          <td>27.777037</td>
          <td>0.354053</td>
          <td>18.169791</td>
          <td>0.005005</td>
          <td>24.525337</td>
          <td>0.066763</td>
          <td>25.791753</td>
          <td>0.419976</td>
          <td>21.460429</td>
          <td>24.206564</td>
          <td>23.396535</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.681251</td>
          <td>0.034074</td>
          <td>20.661445</td>
          <td>0.005090</td>
          <td>22.183571</td>
          <td>0.005640</td>
          <td>23.948759</td>
          <td>0.021088</td>
          <td>20.502874</td>
          <td>0.005341</td>
          <td>20.714899</td>
          <td>0.007046</td>
          <td>24.829158</td>
          <td>19.088020</td>
          <td>20.198326</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.831556</td>
          <td>0.038848</td>
          <td>24.613588</td>
          <td>0.026401</td>
          <td>21.293990</td>
          <td>0.005154</td>
          <td>18.490657</td>
          <td>0.005007</td>
          <td>22.138258</td>
          <td>0.009238</td>
          <td>22.030150</td>
          <td>0.016873</td>
          <td>24.406425</td>
          <td>25.940808</td>
          <td>25.721227</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.241358</td>
          <td>0.010762</td>
          <td>20.435341</td>
          <td>0.005066</td>
          <td>24.662928</td>
          <td>0.024214</td>
          <td>28.025961</td>
          <td>0.642483</td>
          <td>28.741068</td>
          <td>1.521941</td>
          <td>25.079152</td>
          <td>0.238031</td>
          <td>22.183800</td>
          <td>21.670169</td>
          <td>23.237695</td>
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
          <td>23.372259</td>
          <td>24.093846</td>
          <td>26.501633</td>
          <td>22.383456</td>
          <td>25.511400</td>
          <td>24.363360</td>
          <td>22.790394</td>
          <td>0.006713</td>
          <td>22.973521</td>
          <td>0.010470</td>
          <td>27.355417</td>
          <td>0.427259</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.090670</td>
          <td>20.907259</td>
          <td>27.286319</td>
          <td>22.707655</td>
          <td>21.318102</td>
          <td>27.221491</td>
          <td>23.567359</td>
          <td>0.010425</td>
          <td>15.048180</td>
          <td>0.005000</td>
          <td>29.073915</td>
          <td>1.311786</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.783997</td>
          <td>22.102058</td>
          <td>17.882944</td>
          <td>23.404825</td>
          <td>21.618610</td>
          <td>29.610300</td>
          <td>23.176573</td>
          <td>0.008112</td>
          <td>26.202777</td>
          <td>0.167531</td>
          <td>27.721177</td>
          <td>0.560246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.999183</td>
          <td>20.327469</td>
          <td>21.099019</td>
          <td>30.548210</td>
          <td>24.615194</td>
          <td>23.843024</td>
          <td>21.847728</td>
          <td>0.005342</td>
          <td>24.195981</td>
          <td>0.028577</td>
          <td>22.680002</td>
          <td>0.008623</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.286777</td>
          <td>22.342359</td>
          <td>18.556179</td>
          <td>23.627634</td>
          <td>24.866671</td>
          <td>23.025616</td>
          <td>22.079742</td>
          <td>0.005516</td>
          <td>21.697920</td>
          <td>0.005753</td>
          <td>25.114388</td>
          <td>0.064676</td>
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
          <td>24.520540</td>
          <td>23.648429</td>
          <td>25.016488</td>
          <td>21.161005</td>
          <td>28.542051</td>
          <td>21.651750</td>
          <td>22.884599</td>
          <td>0.006990</td>
          <td>17.248633</td>
          <td>0.005000</td>
          <td>22.133220</td>
          <td>0.006562</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.026586</td>
          <td>21.327264</td>
          <td>27.735674</td>
          <td>18.158788</td>
          <td>24.579845</td>
          <td>27.230707</td>
          <td>21.458857</td>
          <td>0.005170</td>
          <td>24.208268</td>
          <td>0.028889</td>
          <td>23.402781</td>
          <td>0.014524</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.664960</td>
          <td>20.656846</td>
          <td>22.177050</td>
          <td>23.938074</td>
          <td>20.491606</td>
          <td>20.712607</td>
          <td>24.817753</td>
          <td>0.029132</td>
          <td>19.091542</td>
          <td>0.005007</td>
          <td>20.196225</td>
          <td>0.005051</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.770342</td>
          <td>24.630636</td>
          <td>21.293152</td>
          <td>18.495248</td>
          <td>22.123258</td>
          <td>22.054154</td>
          <td>24.379187</td>
          <td>0.019881</td>
          <td>25.911127</td>
          <td>0.130363</td>
          <td>25.648922</td>
          <td>0.103723</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.247313</td>
          <td>20.435847</td>
          <td>24.664137</td>
          <td>27.633619</td>
          <td>28.174137</td>
          <td>24.805029</td>
          <td>22.182201</td>
          <td>0.005617</td>
          <td>21.673544</td>
          <td>0.005722</td>
          <td>23.225560</td>
          <td>0.012625</td>
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
          <td>23.372259</td>
          <td>24.093846</td>
          <td>26.501633</td>
          <td>22.383456</td>
          <td>25.511400</td>
          <td>24.363360</td>
          <td>22.757052</td>
          <td>0.051431</td>
          <td>22.933241</td>
          <td>0.050351</td>
          <td>27.089174</td>
          <td>1.322509</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.090670</td>
          <td>20.907259</td>
          <td>27.286319</td>
          <td>22.707655</td>
          <td>21.318102</td>
          <td>27.221491</td>
          <td>23.524984</td>
          <td>0.101568</td>
          <td>15.055366</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.783997</td>
          <td>22.102058</td>
          <td>17.882944</td>
          <td>23.404825</td>
          <td>21.618610</td>
          <td>29.610300</td>
          <td>23.172950</td>
          <td>0.074452</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.999183</td>
          <td>20.327469</td>
          <td>21.099019</td>
          <td>30.548210</td>
          <td>24.615194</td>
          <td>23.843024</td>
          <td>21.854533</td>
          <td>0.023129</td>
          <td>24.405911</td>
          <td>0.182880</td>
          <td>22.674794</td>
          <td>0.043718</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.286777</td>
          <td>22.342359</td>
          <td>18.556179</td>
          <td>23.627634</td>
          <td>24.866671</td>
          <td>23.025616</td>
          <td>22.069276</td>
          <td>0.027913</td>
          <td>21.721357</td>
          <td>0.017391</td>
          <td>24.980333</td>
          <td>0.318857</td>
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
          <td>24.520540</td>
          <td>23.648429</td>
          <td>25.016488</td>
          <td>21.161005</td>
          <td>28.542051</td>
          <td>21.651750</td>
          <td>22.798609</td>
          <td>0.053372</td>
          <td>17.248519</td>
          <td>0.005007</td>
          <td>22.086337</td>
          <td>0.025950</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.026586</td>
          <td>21.327264</td>
          <td>27.735674</td>
          <td>18.158788</td>
          <td>24.579845</td>
          <td>27.230707</td>
          <td>21.487319</td>
          <td>0.016903</td>
          <td>23.994940</td>
          <td>0.128547</td>
          <td>23.432800</td>
          <td>0.085762</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.664960</td>
          <td>20.656846</td>
          <td>22.177050</td>
          <td>23.938074</td>
          <td>20.491606</td>
          <td>20.712607</td>
          <td>24.400318</td>
          <td>0.215373</td>
          <td>19.083910</td>
          <td>0.005213</td>
          <td>20.195723</td>
          <td>0.006727</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.770342</td>
          <td>24.630636</td>
          <td>21.293152</td>
          <td>18.495248</td>
          <td>22.123258</td>
          <td>22.054154</td>
          <td>24.397997</td>
          <td>0.214956</td>
          <td>26.306668</td>
          <td>0.782606</td>
          <td>26.100913</td>
          <td>0.729574</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.247313</td>
          <td>20.435847</td>
          <td>24.664137</td>
          <td>27.633619</td>
          <td>28.174137</td>
          <td>24.805029</td>
          <td>22.177417</td>
          <td>0.030709</td>
          <td>21.698522</td>
          <td>0.017062</td>
          <td>23.297736</td>
          <td>0.076106</td>
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


