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
          <td>26.732313</td>
          <td>25.761231</td>
          <td>25.738066</td>
          <td>22.573244</td>
          <td>19.448511</td>
          <td>24.485991</td>
          <td>20.743079</td>
          <td>25.316307</td>
          <td>22.204769</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.980738</td>
          <td>17.127469</td>
          <td>20.904937</td>
          <td>29.224352</td>
          <td>16.484287</td>
          <td>21.210963</td>
          <td>19.002433</td>
          <td>24.300803</td>
          <td>22.401545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.318211</td>
          <td>24.522467</td>
          <td>21.982600</td>
          <td>19.138879</td>
          <td>26.559258</td>
          <td>22.115862</td>
          <td>21.925922</td>
          <td>23.022981</td>
          <td>23.435004</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.552324</td>
          <td>23.277053</td>
          <td>23.100985</td>
          <td>23.087052</td>
          <td>18.856428</td>
          <td>25.269100</td>
          <td>23.379616</td>
          <td>20.473295</td>
          <td>23.614024</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.317817</td>
          <td>26.233391</td>
          <td>22.082569</td>
          <td>16.397212</td>
          <td>24.320442</td>
          <td>24.634470</td>
          <td>20.471088</td>
          <td>25.659656</td>
          <td>18.321659</td>
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
          <td>19.384420</td>
          <td>23.726459</td>
          <td>21.231866</td>
          <td>23.208137</td>
          <td>20.934604</td>
          <td>21.502046</td>
          <td>26.435565</td>
          <td>18.323898</td>
          <td>20.123960</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.172797</td>
          <td>23.079794</td>
          <td>28.678315</td>
          <td>22.321831</td>
          <td>22.109179</td>
          <td>23.938334</td>
          <td>24.878219</td>
          <td>20.840737</td>
          <td>26.193016</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.314608</td>
          <td>20.690667</td>
          <td>23.397164</td>
          <td>21.137376</td>
          <td>20.605399</td>
          <td>24.105414</td>
          <td>23.555646</td>
          <td>21.143267</td>
          <td>22.961764</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.363523</td>
          <td>25.795839</td>
          <td>26.932648</td>
          <td>26.079657</td>
          <td>22.737335</td>
          <td>26.265638</td>
          <td>23.675056</td>
          <td>20.829273</td>
          <td>25.330876</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.136190</td>
          <td>27.495336</td>
          <td>23.601799</td>
          <td>19.403157</td>
          <td>29.476482</td>
          <td>23.996492</td>
          <td>21.109392</td>
          <td>29.210873</td>
          <td>21.089239</td>
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
          <td>26.584753</td>
          <td>0.402543</td>
          <td>25.736327</td>
          <td>0.071027</td>
          <td>25.696102</td>
          <td>0.060248</td>
          <td>22.568737</td>
          <td>0.007750</td>
          <td>19.456359</td>
          <td>0.005068</td>
          <td>24.348727</td>
          <td>0.128120</td>
          <td>20.743079</td>
          <td>25.316307</td>
          <td>22.204769</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.210914</td>
          <td>1.194484</td>
          <td>17.124278</td>
          <td>0.005002</td>
          <td>20.897179</td>
          <td>0.005084</td>
          <td>27.282717</td>
          <td>0.370864</td>
          <td>16.481054</td>
          <td>0.005002</td>
          <td>21.202692</td>
          <td>0.009145</td>
          <td>19.002433</td>
          <td>24.300803</td>
          <td>22.401545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.325873</td>
          <td>0.059973</td>
          <td>24.530079</td>
          <td>0.024562</td>
          <td>21.990631</td>
          <td>0.005468</td>
          <td>19.150058</td>
          <td>0.005016</td>
          <td>26.090137</td>
          <td>0.256692</td>
          <td>22.135959</td>
          <td>0.018422</td>
          <td>21.925922</td>
          <td>23.022981</td>
          <td>23.435004</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.461661</td>
          <td>0.067583</td>
          <td>23.269576</td>
          <td>0.009268</td>
          <td>23.100499</td>
          <td>0.007666</td>
          <td>23.087670</td>
          <td>0.010664</td>
          <td>18.854580</td>
          <td>0.005029</td>
          <td>25.015915</td>
          <td>0.225884</td>
          <td>23.379616</td>
          <td>20.473295</td>
          <td>23.614024</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.353188</td>
          <td>0.146886</td>
          <td>26.138226</td>
          <td>0.101145</td>
          <td>22.088819</td>
          <td>0.005549</td>
          <td>16.398186</td>
          <td>0.005001</td>
          <td>24.372815</td>
          <td>0.058319</td>
          <td>24.736064</td>
          <td>0.178593</td>
          <td>20.471088</td>
          <td>25.659656</td>
          <td>18.321659</td>
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
          <td>19.387307</td>
          <td>0.005129</td>
          <td>23.730811</td>
          <td>0.012767</td>
          <td>21.235805</td>
          <td>0.005141</td>
          <td>23.222707</td>
          <td>0.011752</td>
          <td>20.927115</td>
          <td>0.005675</td>
          <td>21.495767</td>
          <td>0.011142</td>
          <td>26.435565</td>
          <td>18.323898</td>
          <td>20.123960</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.177945</td>
          <td>0.006384</td>
          <td>23.078313</td>
          <td>0.008277</td>
          <td>29.045297</td>
          <td>0.875323</td>
          <td>22.331974</td>
          <td>0.006931</td>
          <td>22.117313</td>
          <td>0.009117</td>
          <td>24.061270</td>
          <td>0.099730</td>
          <td>24.878219</td>
          <td>20.840737</td>
          <td>26.193016</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.323613</td>
          <td>0.005120</td>
          <td>20.692664</td>
          <td>0.005094</td>
          <td>23.398597</td>
          <td>0.009080</td>
          <td>21.142749</td>
          <td>0.005291</td>
          <td>20.604857</td>
          <td>0.005402</td>
          <td>24.047458</td>
          <td>0.098530</td>
          <td>23.555646</td>
          <td>21.143267</td>
          <td>22.961764</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.380139</td>
          <td>0.011843</td>
          <td>25.756106</td>
          <td>0.072279</td>
          <td>26.776726</td>
          <td>0.155154</td>
          <td>25.940143</td>
          <td>0.121539</td>
          <td>22.760952</td>
          <td>0.014483</td>
          <td>26.042429</td>
          <td>0.506816</td>
          <td>23.675056</td>
          <td>20.829273</td>
          <td>25.330876</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.150637</td>
          <td>0.021638</td>
          <td>27.136624</td>
          <td>0.237220</td>
          <td>23.596967</td>
          <td>0.010344</td>
          <td>19.406319</td>
          <td>0.005022</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.932960</td>
          <td>0.089105</td>
          <td>21.109392</td>
          <td>29.210873</td>
          <td>21.089239</td>
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
          <td>26.732313</td>
          <td>25.761231</td>
          <td>25.738066</td>
          <td>22.573244</td>
          <td>19.448511</td>
          <td>24.485991</td>
          <td>20.749899</td>
          <td>0.005047</td>
          <td>25.191110</td>
          <td>0.069238</td>
          <td>22.205932</td>
          <td>0.006756</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.980738</td>
          <td>17.127469</td>
          <td>20.904937</td>
          <td>29.224352</td>
          <td>16.484287</td>
          <td>21.210963</td>
          <td>19.008606</td>
          <td>0.005002</td>
          <td>24.362730</td>
          <td>0.033119</td>
          <td>22.399178</td>
          <td>0.007379</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.318211</td>
          <td>24.522467</td>
          <td>21.982600</td>
          <td>19.138879</td>
          <td>26.559258</td>
          <td>22.115862</td>
          <td>21.927631</td>
          <td>0.005394</td>
          <td>23.012760</td>
          <td>0.010768</td>
          <td>23.439109</td>
          <td>0.014958</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.552324</td>
          <td>23.277053</td>
          <td>23.100985</td>
          <td>23.087052</td>
          <td>18.856428</td>
          <td>25.269100</td>
          <td>23.386419</td>
          <td>0.009220</td>
          <td>20.470558</td>
          <td>0.005084</td>
          <td>23.609241</td>
          <td>0.017215</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.317817</td>
          <td>26.233391</td>
          <td>22.082569</td>
          <td>16.397212</td>
          <td>24.320442</td>
          <td>24.634470</td>
          <td>20.472594</td>
          <td>0.005028</td>
          <td>25.657858</td>
          <td>0.104538</td>
          <td>18.317726</td>
          <td>0.005002</td>
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
          <td>19.384420</td>
          <td>23.726459</td>
          <td>21.231866</td>
          <td>23.208137</td>
          <td>20.934604</td>
          <td>21.502046</td>
          <td>26.604508</td>
          <td>0.141327</td>
          <td>18.328813</td>
          <td>0.005002</td>
          <td>20.122066</td>
          <td>0.005044</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.172797</td>
          <td>23.079794</td>
          <td>28.678315</td>
          <td>22.321831</td>
          <td>22.109179</td>
          <td>23.938334</td>
          <td>24.897183</td>
          <td>0.031251</td>
          <td>20.843755</td>
          <td>0.005165</td>
          <td>26.054772</td>
          <td>0.147581</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.314608</td>
          <td>20.690667</td>
          <td>23.397164</td>
          <td>21.137376</td>
          <td>20.605399</td>
          <td>24.105414</td>
          <td>23.570081</td>
          <td>0.010445</td>
          <td>21.145885</td>
          <td>0.005285</td>
          <td>22.956844</td>
          <td>0.010348</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.363523</td>
          <td>25.795839</td>
          <td>26.932648</td>
          <td>26.079657</td>
          <td>22.737335</td>
          <td>26.265638</td>
          <td>23.678508</td>
          <td>0.011296</td>
          <td>20.824207</td>
          <td>0.005160</td>
          <td>25.276639</td>
          <td>0.074696</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.136190</td>
          <td>27.495336</td>
          <td>23.601799</td>
          <td>19.403157</td>
          <td>29.476482</td>
          <td>23.996492</td>
          <td>21.114915</td>
          <td>0.005091</td>
          <td>28.081865</td>
          <td>0.720296</td>
          <td>21.082851</td>
          <td>0.005255</td>
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
          <td>26.732313</td>
          <td>25.761231</td>
          <td>25.738066</td>
          <td>22.573244</td>
          <td>19.448511</td>
          <td>24.485991</td>
          <td>20.731559</td>
          <td>0.009498</td>
          <td>24.894323</td>
          <td>0.274467</td>
          <td>22.194398</td>
          <td>0.028538</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.980738</td>
          <td>17.127469</td>
          <td>20.904937</td>
          <td>29.224352</td>
          <td>16.484287</td>
          <td>21.210963</td>
          <td>19.009013</td>
          <td>0.005267</td>
          <td>24.434401</td>
          <td>0.187343</td>
          <td>22.432452</td>
          <td>0.035234</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.318211</td>
          <td>24.522467</td>
          <td>21.982600</td>
          <td>19.138879</td>
          <td>26.559258</td>
          <td>22.115862</td>
          <td>21.916637</td>
          <td>0.024415</td>
          <td>23.088536</td>
          <td>0.057824</td>
          <td>23.539020</td>
          <td>0.094179</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.552324</td>
          <td>23.277053</td>
          <td>23.100985</td>
          <td>23.087052</td>
          <td>18.856428</td>
          <td>25.269100</td>
          <td>23.280970</td>
          <td>0.081924</td>
          <td>20.469812</td>
          <td>0.007273</td>
          <td>23.717551</td>
          <td>0.110145</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.317817</td>
          <td>26.233391</td>
          <td>22.082569</td>
          <td>16.397212</td>
          <td>24.320442</td>
          <td>24.634470</td>
          <td>20.478992</td>
          <td>0.008123</td>
          <td>26.426908</td>
          <td>0.846026</td>
          <td>18.321680</td>
          <td>0.005064</td>
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
          <td>19.384420</td>
          <td>23.726459</td>
          <td>21.231866</td>
          <td>23.208137</td>
          <td>20.934604</td>
          <td>21.502046</td>
          <td>26.085594</td>
          <td>0.771833</td>
          <td>18.321796</td>
          <td>0.005053</td>
          <td>20.126412</td>
          <td>0.006544</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.172797</td>
          <td>23.079794</td>
          <td>28.678315</td>
          <td>22.321831</td>
          <td>22.109179</td>
          <td>23.938334</td>
          <td>24.896030</td>
          <td>0.322873</td>
          <td>20.837900</td>
          <td>0.008938</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.314608</td>
          <td>20.690667</td>
          <td>23.397164</td>
          <td>21.137376</td>
          <td>20.605399</td>
          <td>24.105414</td>
          <td>23.833058</td>
          <td>0.132864</td>
          <td>21.147892</td>
          <td>0.011045</td>
          <td>22.903195</td>
          <td>0.053590</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.363523</td>
          <td>25.795839</td>
          <td>26.932648</td>
          <td>26.079657</td>
          <td>22.737335</td>
          <td>26.265638</td>
          <td>23.963984</td>
          <td>0.148755</td>
          <td>20.823156</td>
          <td>0.008856</td>
          <td>25.201209</td>
          <td>0.379459</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.136190</td>
          <td>27.495336</td>
          <td>23.601799</td>
          <td>19.403157</td>
          <td>29.476482</td>
          <td>23.996492</td>
          <td>21.108876</td>
          <td>0.012463</td>
          <td>29.504864</td>
          <td>3.310181</td>
          <td>21.089759</td>
          <td>0.011391</td>
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


