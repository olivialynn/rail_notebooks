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
          <td>26.891928</td>
          <td>25.931281</td>
          <td>23.144344</td>
          <td>27.305228</td>
          <td>22.265193</td>
          <td>22.747906</td>
          <td>25.832356</td>
          <td>24.650148</td>
          <td>21.853256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.107300</td>
          <td>19.623773</td>
          <td>23.859738</td>
          <td>21.487100</td>
          <td>19.210526</td>
          <td>21.822387</td>
          <td>21.655952</td>
          <td>23.543058</td>
          <td>20.127502</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.497872</td>
          <td>24.377850</td>
          <td>26.493820</td>
          <td>21.462221</td>
          <td>18.804347</td>
          <td>23.124403</td>
          <td>27.695994</td>
          <td>26.938246</td>
          <td>26.161323</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.208039</td>
          <td>25.542543</td>
          <td>21.856784</td>
          <td>26.001506</td>
          <td>22.324289</td>
          <td>20.646961</td>
          <td>25.185551</td>
          <td>25.867066</td>
          <td>27.180762</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.288516</td>
          <td>20.298738</td>
          <td>23.864835</td>
          <td>20.193966</td>
          <td>28.249252</td>
          <td>22.846176</td>
          <td>20.080795</td>
          <td>23.705760</td>
          <td>27.859743</td>
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
          <td>22.268062</td>
          <td>21.329918</td>
          <td>21.644113</td>
          <td>23.941736</td>
          <td>27.590533</td>
          <td>18.727659</td>
          <td>28.683879</td>
          <td>23.829393</td>
          <td>25.057529</td>
        </tr>
        <tr>
          <th>996</th>
          <td>14.866750</td>
          <td>20.221467</td>
          <td>25.379444</td>
          <td>21.210632</td>
          <td>30.317549</td>
          <td>22.893265</td>
          <td>24.492727</td>
          <td>19.597537</td>
          <td>25.273962</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.086007</td>
          <td>22.821571</td>
          <td>24.158277</td>
          <td>23.501996</td>
          <td>23.560346</td>
          <td>21.663820</td>
          <td>20.666492</td>
          <td>23.412891</td>
          <td>23.099497</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.108891</td>
          <td>21.210701</td>
          <td>21.391978</td>
          <td>22.228945</td>
          <td>20.352398</td>
          <td>19.371490</td>
          <td>21.058781</td>
          <td>22.145371</td>
          <td>22.090912</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.488830</td>
          <td>16.818275</td>
          <td>20.595327</td>
          <td>24.730813</td>
          <td>21.778439</td>
          <td>30.974104</td>
          <td>19.141357</td>
          <td>21.955014</td>
          <td>25.584463</td>
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
          <td>27.078737</td>
          <td>0.580597</td>
          <td>26.020060</td>
          <td>0.091195</td>
          <td>23.145457</td>
          <td>0.007847</td>
          <td>27.486240</td>
          <td>0.433742</td>
          <td>22.271259</td>
          <td>0.010078</td>
          <td>22.709810</td>
          <td>0.030192</td>
          <td>25.832356</td>
          <td>24.650148</td>
          <td>21.853256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.163605</td>
          <td>0.124774</td>
          <td>19.617776</td>
          <td>0.005024</td>
          <td>23.860829</td>
          <td>0.012522</td>
          <td>21.494026</td>
          <td>0.005512</td>
          <td>19.209418</td>
          <td>0.005047</td>
          <td>21.808134</td>
          <td>0.014105</td>
          <td>21.655952</td>
          <td>23.543058</td>
          <td>20.127502</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.500808</td>
          <td>0.005047</td>
          <td>24.353073</td>
          <td>0.021111</td>
          <td>26.493435</td>
          <td>0.121509</td>
          <td>21.460389</td>
          <td>0.005485</td>
          <td>18.807966</td>
          <td>0.005027</td>
          <td>23.150204</td>
          <td>0.044553</td>
          <td>27.695994</td>
          <td>26.938246</td>
          <td>26.161323</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.202274</td>
          <td>0.005361</td>
          <td>25.493272</td>
          <td>0.057285</td>
          <td>21.852588</td>
          <td>0.005375</td>
          <td>25.995344</td>
          <td>0.127501</td>
          <td>22.314093</td>
          <td>0.010377</td>
          <td>20.644773</td>
          <td>0.006840</td>
          <td>25.185551</td>
          <td>25.867066</td>
          <td>27.180762</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.282556</td>
          <td>0.005402</td>
          <td>20.299178</td>
          <td>0.005056</td>
          <td>23.873221</td>
          <td>0.012640</td>
          <td>20.194497</td>
          <td>0.005067</td>
          <td>27.743631</td>
          <td>0.866932</td>
          <td>22.836304</td>
          <td>0.033746</td>
          <td>20.080795</td>
          <td>23.705760</td>
          <td>27.859743</td>
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
          <td>22.264101</td>
          <td>0.010928</td>
          <td>21.335719</td>
          <td>0.005237</td>
          <td>21.654991</td>
          <td>0.005273</td>
          <td>23.945355</td>
          <td>0.021027</td>
          <td>27.093213</td>
          <td>0.557740</td>
          <td>18.734360</td>
          <td>0.005091</td>
          <td>28.683879</td>
          <td>23.829393</td>
          <td>25.057529</td>
        </tr>
        <tr>
          <th>996</th>
          <td>14.867207</td>
          <td>0.005001</td>
          <td>20.216533</td>
          <td>0.005050</td>
          <td>25.427694</td>
          <td>0.047474</td>
          <td>21.210879</td>
          <td>0.005324</td>
          <td>27.422815</td>
          <td>0.702324</td>
          <td>22.886783</td>
          <td>0.035284</td>
          <td>24.492727</td>
          <td>19.597537</td>
          <td>25.273962</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.075342</td>
          <td>0.005011</td>
          <td>22.831063</td>
          <td>0.007300</td>
          <td>24.141214</td>
          <td>0.015607</td>
          <td>23.516693</td>
          <td>0.014738</td>
          <td>23.574827</td>
          <td>0.028788</td>
          <td>21.680260</td>
          <td>0.012773</td>
          <td>20.666492</td>
          <td>23.412891</td>
          <td>23.099497</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.112399</td>
          <td>0.006263</td>
          <td>21.208092</td>
          <td>0.005196</td>
          <td>21.390633</td>
          <td>0.005179</td>
          <td>22.236680</td>
          <td>0.006669</td>
          <td>20.359523</td>
          <td>0.005272</td>
          <td>19.365846</td>
          <td>0.005242</td>
          <td>21.058781</td>
          <td>22.145371</td>
          <td>22.090912</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.497412</td>
          <td>0.012890</td>
          <td>16.811272</td>
          <td>0.005001</td>
          <td>20.587642</td>
          <td>0.005053</td>
          <td>24.779062</td>
          <td>0.043631</td>
          <td>21.780572</td>
          <td>0.007544</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.141357</td>
          <td>21.955014</td>
          <td>25.584463</td>
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
          <td>26.891928</td>
          <td>25.931281</td>
          <td>23.144344</td>
          <td>27.305228</td>
          <td>22.265193</td>
          <td>22.747906</td>
          <td>25.779614</td>
          <td>0.068535</td>
          <td>24.607860</td>
          <td>0.041186</td>
          <td>21.844860</td>
          <td>0.005968</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.107300</td>
          <td>19.623773</td>
          <td>23.859738</td>
          <td>21.487100</td>
          <td>19.210526</td>
          <td>21.822387</td>
          <td>21.658444</td>
          <td>0.005244</td>
          <td>23.542613</td>
          <td>0.016286</td>
          <td>20.129474</td>
          <td>0.005045</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.497872</td>
          <td>24.377850</td>
          <td>26.493820</td>
          <td>21.462221</td>
          <td>18.804347</td>
          <td>23.124403</td>
          <td>27.356437</td>
          <td>0.266121</td>
          <td>26.829253</td>
          <td>0.282365</td>
          <td>25.971808</td>
          <td>0.137393</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.208039</td>
          <td>25.542543</td>
          <td>21.856784</td>
          <td>26.001506</td>
          <td>22.324289</td>
          <td>20.646961</td>
          <td>25.210011</td>
          <td>0.041265</td>
          <td>25.765799</td>
          <td>0.114886</td>
          <td>26.551291</td>
          <td>0.224719</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.288516</td>
          <td>20.298738</td>
          <td>23.864835</td>
          <td>20.193966</td>
          <td>28.249252</td>
          <td>22.846176</td>
          <td>20.079245</td>
          <td>0.005014</td>
          <td>23.698507</td>
          <td>0.018560</td>
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
          <td>22.268062</td>
          <td>21.329918</td>
          <td>21.644113</td>
          <td>23.941736</td>
          <td>27.590533</td>
          <td>18.727659</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.833854</td>
          <td>0.020836</td>
          <td>25.039815</td>
          <td>0.060525</td>
        </tr>
        <tr>
          <th>996</th>
          <td>14.866750</td>
          <td>20.221467</td>
          <td>25.379444</td>
          <td>21.210632</td>
          <td>30.317549</td>
          <td>22.893265</td>
          <td>24.492042</td>
          <td>0.021908</td>
          <td>19.591866</td>
          <td>0.005017</td>
          <td>25.191043</td>
          <td>0.069234</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.086007</td>
          <td>22.821571</td>
          <td>24.158277</td>
          <td>23.501996</td>
          <td>23.560346</td>
          <td>21.663820</td>
          <td>20.668345</td>
          <td>0.005040</td>
          <td>23.426368</td>
          <td>0.014804</td>
          <td>23.085943</td>
          <td>0.011359</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.108891</td>
          <td>21.210701</td>
          <td>21.391978</td>
          <td>22.228945</td>
          <td>20.352398</td>
          <td>19.371490</td>
          <td>21.066772</td>
          <td>0.005083</td>
          <td>22.131476</td>
          <td>0.006557</td>
          <td>22.090675</td>
          <td>0.006457</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.488830</td>
          <td>16.818275</td>
          <td>20.595327</td>
          <td>24.730813</td>
          <td>21.778439</td>
          <td>30.974104</td>
          <td>19.139634</td>
          <td>0.005002</td>
          <td>21.952154</td>
          <td>0.006159</td>
          <td>25.475134</td>
          <td>0.089026</td>
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
          <td>26.891928</td>
          <td>25.931281</td>
          <td>23.144344</td>
          <td>27.305228</td>
          <td>22.265193</td>
          <td>22.747906</td>
          <td>25.088461</td>
          <td>0.375714</td>
          <td>24.670084</td>
          <td>0.228256</td>
          <td>21.891626</td>
          <td>0.021901</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.107300</td>
          <td>19.623773</td>
          <td>23.859738</td>
          <td>21.487100</td>
          <td>19.210526</td>
          <td>21.822387</td>
          <td>21.639047</td>
          <td>0.019211</td>
          <td>23.577372</td>
          <td>0.089202</td>
          <td>20.133175</td>
          <td>0.006562</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.497872</td>
          <td>24.377850</td>
          <td>26.493820</td>
          <td>21.462221</td>
          <td>18.804347</td>
          <td>23.124403</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.921979</td>
          <td>0.304305</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.208039</td>
          <td>25.542543</td>
          <td>21.856784</td>
          <td>26.001506</td>
          <td>22.324289</td>
          <td>20.646961</td>
          <td>24.304601</td>
          <td>0.198769</td>
          <td>25.913715</td>
          <td>0.598500</td>
          <td>25.677971</td>
          <td>0.543037</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.288516</td>
          <td>20.298738</td>
          <td>23.864835</td>
          <td>20.193966</td>
          <td>28.249252</td>
          <td>22.846176</td>
          <td>20.075751</td>
          <td>0.006673</td>
          <td>23.493147</td>
          <td>0.082810</td>
          <td>25.697181</td>
          <td>0.550637</td>
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
          <td>22.268062</td>
          <td>21.329918</td>
          <td>21.644113</td>
          <td>23.941736</td>
          <td>27.590533</td>
          <td>18.727659</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.848010</td>
          <td>0.113116</td>
          <td>25.018600</td>
          <td>0.328722</td>
        </tr>
        <tr>
          <th>996</th>
          <td>14.866750</td>
          <td>20.221467</td>
          <td>25.379444</td>
          <td>21.210632</td>
          <td>30.317549</td>
          <td>22.893265</td>
          <td>24.122951</td>
          <td>0.170436</td>
          <td>19.599641</td>
          <td>0.005534</td>
          <td>25.193624</td>
          <td>0.377227</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.086007</td>
          <td>22.821571</td>
          <td>24.158277</td>
          <td>23.501996</td>
          <td>23.560346</td>
          <td>21.663820</td>
          <td>20.679101</td>
          <td>0.009177</td>
          <td>23.433017</td>
          <td>0.078521</td>
          <td>23.031575</td>
          <td>0.060082</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.108891</td>
          <td>21.210701</td>
          <td>21.391978</td>
          <td>22.228945</td>
          <td>20.352398</td>
          <td>19.371490</td>
          <td>21.052103</td>
          <td>0.011935</td>
          <td>22.127829</td>
          <td>0.024655</td>
          <td>22.078172</td>
          <td>0.025765</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.488830</td>
          <td>16.818275</td>
          <td>20.595327</td>
          <td>24.730813</td>
          <td>21.778439</td>
          <td>30.974104</td>
          <td>19.146699</td>
          <td>0.005341</td>
          <td>21.976400</td>
          <td>0.021614</td>
          <td>25.668124</td>
          <td>0.539172</td>
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


