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
          <td>21.207142</td>
          <td>21.768060</td>
          <td>25.358228</td>
          <td>20.064263</td>
          <td>20.173220</td>
          <td>17.365904</td>
          <td>22.034751</td>
          <td>24.913340</td>
          <td>22.100588</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.118785</td>
          <td>18.050639</td>
          <td>24.979382</td>
          <td>17.926979</td>
          <td>23.814483</td>
          <td>17.889180</td>
          <td>23.135435</td>
          <td>23.542522</td>
          <td>22.820544</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.454343</td>
          <td>19.689854</td>
          <td>23.082243</td>
          <td>21.919061</td>
          <td>24.567844</td>
          <td>16.353748</td>
          <td>24.540258</td>
          <td>26.436896</td>
          <td>20.667897</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.352834</td>
          <td>24.370825</td>
          <td>24.191936</td>
          <td>21.061806</td>
          <td>19.928653</td>
          <td>24.881233</td>
          <td>17.845147</td>
          <td>22.835327</td>
          <td>27.231397</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.492550</td>
          <td>16.304880</td>
          <td>23.643824</td>
          <td>21.825289</td>
          <td>24.474537</td>
          <td>26.493543</td>
          <td>23.763956</td>
          <td>25.835877</td>
          <td>28.698167</td>
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
          <td>22.908378</td>
          <td>24.240435</td>
          <td>21.510670</td>
          <td>23.743783</td>
          <td>24.076295</td>
          <td>25.030318</td>
          <td>26.630887</td>
          <td>24.799529</td>
          <td>25.093559</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.859734</td>
          <td>23.360098</td>
          <td>22.800743</td>
          <td>20.753168</td>
          <td>22.896966</td>
          <td>23.802438</td>
          <td>24.009940</td>
          <td>25.914200</td>
          <td>19.341532</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.025613</td>
          <td>26.685066</td>
          <td>23.202417</td>
          <td>25.016719</td>
          <td>21.727052</td>
          <td>20.279292</td>
          <td>22.251402</td>
          <td>22.028524</td>
          <td>18.038506</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.011839</td>
          <td>22.020393</td>
          <td>23.647831</td>
          <td>26.059048</td>
          <td>22.048345</td>
          <td>20.217737</td>
          <td>26.439233</td>
          <td>22.796258</td>
          <td>25.273275</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.304238</td>
          <td>23.121427</td>
          <td>23.960430</td>
          <td>27.224560</td>
          <td>27.197381</td>
          <td>28.117011</td>
          <td>20.560695</td>
          <td>22.471569</td>
          <td>20.972696</td>
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
          <td>21.204266</td>
          <td>0.006436</td>
          <td>21.766305</td>
          <td>0.005456</td>
          <td>25.399647</td>
          <td>0.046307</td>
          <td>20.070725</td>
          <td>0.005056</td>
          <td>20.172072</td>
          <td>0.005202</td>
          <td>17.368616</td>
          <td>0.005014</td>
          <td>22.034751</td>
          <td>24.913340</td>
          <td>22.100588</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.053201</td>
          <td>0.047182</td>
          <td>18.055020</td>
          <td>0.005004</td>
          <td>25.020048</td>
          <td>0.033089</td>
          <td>17.927586</td>
          <td>0.005004</td>
          <td>23.781929</td>
          <td>0.034539</td>
          <td>17.888175</td>
          <td>0.005027</td>
          <td>23.135435</td>
          <td>23.542522</td>
          <td>22.820544</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.463516</td>
          <td>0.067694</td>
          <td>19.695005</td>
          <td>0.005026</td>
          <td>23.085945</td>
          <td>0.007609</td>
          <td>21.902098</td>
          <td>0.005987</td>
          <td>24.699635</td>
          <td>0.077892</td>
          <td>16.350846</td>
          <td>0.005004</td>
          <td>24.540258</td>
          <td>26.436896</td>
          <td>20.667897</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.622709</td>
          <td>0.184722</td>
          <td>24.355204</td>
          <td>0.021150</td>
          <td>24.198481</td>
          <td>0.016353</td>
          <td>21.060024</td>
          <td>0.005254</td>
          <td>19.931111</td>
          <td>0.005139</td>
          <td>25.036404</td>
          <td>0.229757</td>
          <td>17.845147</td>
          <td>22.835327</td>
          <td>27.231397</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.504622</td>
          <td>0.007179</td>
          <td>16.313236</td>
          <td>0.005001</td>
          <td>23.652340</td>
          <td>0.010750</td>
          <td>21.820806</td>
          <td>0.005867</td>
          <td>24.521373</td>
          <td>0.066529</td>
          <td>25.638555</td>
          <td>0.373172</td>
          <td>23.763956</td>
          <td>25.835877</td>
          <td>28.698167</td>
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
          <td>22.902593</td>
          <td>0.017643</td>
          <td>24.244488</td>
          <td>0.019265</td>
          <td>21.518065</td>
          <td>0.005219</td>
          <td>23.732733</td>
          <td>0.017578</td>
          <td>24.068488</td>
          <td>0.044514</td>
          <td>25.008248</td>
          <td>0.224449</td>
          <td>26.630887</td>
          <td>24.799529</td>
          <td>25.093559</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.808507</td>
          <td>0.038073</td>
          <td>23.359473</td>
          <td>0.009816</td>
          <td>22.793397</td>
          <td>0.006679</td>
          <td>20.740900</td>
          <td>0.005153</td>
          <td>22.898647</td>
          <td>0.016183</td>
          <td>23.855190</td>
          <td>0.083207</td>
          <td>24.009940</td>
          <td>25.914200</td>
          <td>19.341532</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.032633</td>
          <td>0.006130</td>
          <td>26.671808</td>
          <td>0.160485</td>
          <td>23.197418</td>
          <td>0.008069</td>
          <td>25.045657</td>
          <td>0.055283</td>
          <td>21.720041</td>
          <td>0.007326</td>
          <td>20.273691</td>
          <td>0.006033</td>
          <td>22.251402</td>
          <td>22.028524</td>
          <td>18.038506</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.861215</td>
          <td>0.225508</td>
          <td>22.020768</td>
          <td>0.005674</td>
          <td>23.650477</td>
          <td>0.010736</td>
          <td>26.179264</td>
          <td>0.149424</td>
          <td>22.051110</td>
          <td>0.008754</td>
          <td>20.220520</td>
          <td>0.005950</td>
          <td>26.439233</td>
          <td>22.796258</td>
          <td>25.273275</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.315249</td>
          <td>0.006677</td>
          <td>23.124800</td>
          <td>0.008497</td>
          <td>23.954434</td>
          <td>0.013455</td>
          <td>27.098309</td>
          <td>0.320678</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.560695</td>
          <td>22.471569</td>
          <td>20.972696</td>
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
          <td>21.207142</td>
          <td>21.768060</td>
          <td>25.358228</td>
          <td>20.064263</td>
          <td>20.173220</td>
          <td>17.365904</td>
          <td>22.033379</td>
          <td>0.005475</td>
          <td>24.892416</td>
          <td>0.053078</td>
          <td>22.094473</td>
          <td>0.006466</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.118785</td>
          <td>18.050639</td>
          <td>24.979382</td>
          <td>17.926979</td>
          <td>23.814483</td>
          <td>17.889180</td>
          <td>23.142923</td>
          <td>0.007960</td>
          <td>23.549089</td>
          <td>0.016373</td>
          <td>22.827450</td>
          <td>0.009472</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.454343</td>
          <td>19.689854</td>
          <td>23.082243</td>
          <td>21.919061</td>
          <td>24.567844</td>
          <td>16.353748</td>
          <td>24.537753</td>
          <td>0.022794</td>
          <td>26.813902</td>
          <td>0.278870</td>
          <td>20.663995</td>
          <td>0.005119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.352834</td>
          <td>24.370825</td>
          <td>24.191936</td>
          <td>21.061806</td>
          <td>19.928653</td>
          <td>24.881233</td>
          <td>17.846538</td>
          <td>0.005000</td>
          <td>22.840780</td>
          <td>0.009556</td>
          <td>26.539024</td>
          <td>0.222437</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.492550</td>
          <td>16.304880</td>
          <td>23.643824</td>
          <td>21.825289</td>
          <td>24.474537</td>
          <td>26.493543</td>
          <td>23.770306</td>
          <td>0.012101</td>
          <td>25.902908</td>
          <td>0.129438</td>
          <td>28.975348</td>
          <td>1.243613</td>
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
          <td>22.908378</td>
          <td>24.240435</td>
          <td>21.510670</td>
          <td>23.743783</td>
          <td>24.076295</td>
          <td>25.030318</td>
          <td>26.449259</td>
          <td>0.123547</td>
          <td>24.682916</td>
          <td>0.044035</td>
          <td>25.085844</td>
          <td>0.063055</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.859734</td>
          <td>23.360098</td>
          <td>22.800743</td>
          <td>20.753168</td>
          <td>22.896966</td>
          <td>23.802438</td>
          <td>23.989876</td>
          <td>0.014374</td>
          <td>25.789619</td>
          <td>0.117297</td>
          <td>19.334380</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.025613</td>
          <td>26.685066</td>
          <td>23.202417</td>
          <td>25.016719</td>
          <td>21.727052</td>
          <td>20.279292</td>
          <td>22.248790</td>
          <td>0.005692</td>
          <td>22.033783</td>
          <td>0.006327</td>
          <td>18.043233</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.011839</td>
          <td>22.020393</td>
          <td>23.647831</td>
          <td>26.059048</td>
          <td>22.048345</td>
          <td>20.217737</td>
          <td>26.465902</td>
          <td>0.125347</td>
          <td>22.801091</td>
          <td>0.009309</td>
          <td>25.201089</td>
          <td>0.069854</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.304238</td>
          <td>23.121427</td>
          <td>23.960430</td>
          <td>27.224560</td>
          <td>27.197381</td>
          <td>28.117011</td>
          <td>20.555909</td>
          <td>0.005033</td>
          <td>22.462314</td>
          <td>0.007621</td>
          <td>20.976732</td>
          <td>0.005210</td>
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
          <td>21.207142</td>
          <td>21.768060</td>
          <td>25.358228</td>
          <td>20.064263</td>
          <td>20.173220</td>
          <td>17.365904</td>
          <td>22.047716</td>
          <td>0.027388</td>
          <td>25.801121</td>
          <td>0.552206</td>
          <td>22.063524</td>
          <td>0.025437</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.118785</td>
          <td>18.050639</td>
          <td>24.979382</td>
          <td>17.926979</td>
          <td>23.814483</td>
          <td>17.889180</td>
          <td>23.130547</td>
          <td>0.071705</td>
          <td>23.452825</td>
          <td>0.079910</td>
          <td>22.877754</td>
          <td>0.052389</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.454343</td>
          <td>19.689854</td>
          <td>23.082243</td>
          <td>21.919061</td>
          <td>24.567844</td>
          <td>16.353748</td>
          <td>24.534881</td>
          <td>0.240838</td>
          <td>26.221675</td>
          <td>0.739782</td>
          <td>20.658853</td>
          <td>0.008513</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.352834</td>
          <td>24.370825</td>
          <td>24.191936</td>
          <td>21.061806</td>
          <td>19.928653</td>
          <td>24.881233</td>
          <td>17.839205</td>
          <td>0.005032</td>
          <td>22.819330</td>
          <td>0.045488</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.492550</td>
          <td>16.304880</td>
          <td>23.643824</td>
          <td>21.825289</td>
          <td>24.474537</td>
          <td>26.493543</td>
          <td>23.777477</td>
          <td>0.126613</td>
          <td>26.087335</td>
          <td>0.675485</td>
          <td>26.012302</td>
          <td>0.687121</td>
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
          <td>22.908378</td>
          <td>24.240435</td>
          <td>21.510670</td>
          <td>23.743783</td>
          <td>24.076295</td>
          <td>25.030318</td>
          <td>27.704156</td>
          <td>1.870336</td>
          <td>24.898327</td>
          <td>0.275363</td>
          <td>25.560183</td>
          <td>0.498194</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.859734</td>
          <td>23.360098</td>
          <td>22.800743</td>
          <td>20.753168</td>
          <td>22.896966</td>
          <td>23.802438</td>
          <td>23.942765</td>
          <td>0.146064</td>
          <td>25.509891</td>
          <td>0.445280</td>
          <td>19.339461</td>
          <td>0.005403</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.025613</td>
          <td>26.685066</td>
          <td>23.202417</td>
          <td>25.016719</td>
          <td>21.727052</td>
          <td>20.279292</td>
          <td>22.250551</td>
          <td>0.032764</td>
          <td>22.006963</td>
          <td>0.022193</td>
          <td>18.034287</td>
          <td>0.005038</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.011839</td>
          <td>22.020393</td>
          <td>23.647831</td>
          <td>26.059048</td>
          <td>22.048345</td>
          <td>20.217737</td>
          <td>25.619259</td>
          <td>0.559473</td>
          <td>22.820683</td>
          <td>0.045543</td>
          <td>25.183086</td>
          <td>0.374145</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.304238</td>
          <td>23.121427</td>
          <td>23.960430</td>
          <td>27.224560</td>
          <td>27.197381</td>
          <td>28.117011</td>
          <td>20.569636</td>
          <td>0.008569</td>
          <td>22.414877</td>
          <td>0.031744</td>
          <td>20.980270</td>
          <td>0.010520</td>
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


