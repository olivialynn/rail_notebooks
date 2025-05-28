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
          <td>21.831258</td>
          <td>23.028346</td>
          <td>28.589577</td>
          <td>23.861054</td>
          <td>23.111832</td>
          <td>29.271242</td>
          <td>21.336646</td>
          <td>20.130654</td>
          <td>24.152155</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.021007</td>
          <td>28.872908</td>
          <td>28.642188</td>
          <td>23.161870</td>
          <td>24.262689</td>
          <td>20.745633</td>
          <td>29.289283</td>
          <td>29.130679</td>
          <td>21.282722</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.873743</td>
          <td>25.929211</td>
          <td>26.123432</td>
          <td>21.756943</td>
          <td>24.657134</td>
          <td>24.252534</td>
          <td>26.533348</td>
          <td>26.587945</td>
          <td>24.723207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.497698</td>
          <td>25.164427</td>
          <td>24.953891</td>
          <td>22.700626</td>
          <td>25.901722</td>
          <td>27.195494</td>
          <td>21.469393</td>
          <td>18.983697</td>
          <td>22.997512</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.976675</td>
          <td>23.129548</td>
          <td>24.918256</td>
          <td>25.726265</td>
          <td>26.578601</td>
          <td>20.934747</td>
          <td>19.370540</td>
          <td>24.535970</td>
          <td>20.866363</td>
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
          <td>17.081842</td>
          <td>19.822631</td>
          <td>27.088093</td>
          <td>19.750104</td>
          <td>24.200047</td>
          <td>21.892643</td>
          <td>23.881065</td>
          <td>23.119526</td>
          <td>20.771587</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.034327</td>
          <td>18.958048</td>
          <td>23.860043</td>
          <td>24.776219</td>
          <td>24.440288</td>
          <td>23.458001</td>
          <td>23.080572</td>
          <td>18.588280</td>
          <td>25.720231</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.538537</td>
          <td>24.177222</td>
          <td>27.831640</td>
          <td>18.495173</td>
          <td>27.683777</td>
          <td>25.400233</td>
          <td>17.096256</td>
          <td>21.348670</td>
          <td>18.325865</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.871210</td>
          <td>22.980063</td>
          <td>24.056119</td>
          <td>18.781556</td>
          <td>20.013217</td>
          <td>23.294228</td>
          <td>20.303802</td>
          <td>23.365742</td>
          <td>19.134944</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.917254</td>
          <td>20.596728</td>
          <td>20.168702</td>
          <td>26.179393</td>
          <td>20.421718</td>
          <td>23.799122</td>
          <td>23.504330</td>
          <td>26.753460</td>
          <td>22.626769</td>
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
          <td>21.850076</td>
          <td>0.008478</td>
          <td>23.036186</td>
          <td>0.008088</td>
          <td>28.104093</td>
          <td>0.455317</td>
          <td>23.883784</td>
          <td>0.019955</td>
          <td>23.111857</td>
          <td>0.019318</td>
          <td>27.464271</td>
          <td>1.263942</td>
          <td>21.336646</td>
          <td>20.130654</td>
          <td>24.152155</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.985553</td>
          <td>0.044461</td>
          <td>27.541579</td>
          <td>0.329375</td>
          <td>28.542981</td>
          <td>0.626301</td>
          <td>23.160502</td>
          <td>0.011231</td>
          <td>24.245770</td>
          <td>0.052100</td>
          <td>20.754814</td>
          <td>0.007173</td>
          <td>29.289283</td>
          <td>29.130679</td>
          <td>21.282722</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.872544</td>
          <td>0.005235</td>
          <td>26.010990</td>
          <td>0.090471</td>
          <td>26.159481</td>
          <td>0.090742</td>
          <td>21.761795</td>
          <td>0.005789</td>
          <td>24.736518</td>
          <td>0.080469</td>
          <td>24.468355</td>
          <td>0.142066</td>
          <td>26.533348</td>
          <td>26.587945</td>
          <td>24.723207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.498871</td>
          <td>0.029095</td>
          <td>25.161803</td>
          <td>0.042714</td>
          <td>24.945778</td>
          <td>0.030995</td>
          <td>22.691815</td>
          <td>0.008286</td>
          <td>26.198447</td>
          <td>0.280389</td>
          <td>26.557305</td>
          <td>0.728244</td>
          <td>21.469393</td>
          <td>18.983697</td>
          <td>22.997512</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.184363</td>
          <td>0.293691</td>
          <td>23.139907</td>
          <td>0.008572</td>
          <td>24.914058</td>
          <td>0.030143</td>
          <td>25.599431</td>
          <td>0.090226</td>
          <td>25.847219</td>
          <td>0.209919</td>
          <td>20.915844</td>
          <td>0.007757</td>
          <td>19.370540</td>
          <td>24.535970</td>
          <td>20.866363</td>
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
          <td>17.089251</td>
          <td>0.005011</td>
          <td>19.823554</td>
          <td>0.005031</td>
          <td>26.873881</td>
          <td>0.168576</td>
          <td>19.756863</td>
          <td>0.005036</td>
          <td>24.273305</td>
          <td>0.053389</td>
          <td>21.901211</td>
          <td>0.015191</td>
          <td>23.881065</td>
          <td>23.119526</td>
          <td>20.771587</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.049190</td>
          <td>0.006156</td>
          <td>18.965418</td>
          <td>0.005011</td>
          <td>23.856832</td>
          <td>0.012484</td>
          <td>24.849332</td>
          <td>0.046439</td>
          <td>24.465100</td>
          <td>0.063293</td>
          <td>23.424426</td>
          <td>0.056832</td>
          <td>23.080572</td>
          <td>18.588280</td>
          <td>25.720231</td>
        </tr>
        <tr>
          <th>997</th>
          <td>inf</td>
          <td>inf</td>
          <td>24.157459</td>
          <td>0.017919</td>
          <td>27.813437</td>
          <td>0.364299</td>
          <td>18.492854</td>
          <td>0.005007</td>
          <td>26.876822</td>
          <td>0.475949</td>
          <td>25.055489</td>
          <td>0.233418</td>
          <td>17.096256</td>
          <td>21.348670</td>
          <td>18.325865</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.609415</td>
          <td>0.410231</td>
          <td>22.982939</td>
          <td>0.007864</td>
          <td>24.048468</td>
          <td>0.014487</td>
          <td>18.773488</td>
          <td>0.005010</td>
          <td>20.009523</td>
          <td>0.005156</td>
          <td>23.267461</td>
          <td>0.049440</td>
          <td>20.303802</td>
          <td>23.365742</td>
          <td>19.134944</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.876142</td>
          <td>0.040394</td>
          <td>20.594401</td>
          <td>0.005082</td>
          <td>20.158407</td>
          <td>0.005029</td>
          <td>25.887997</td>
          <td>0.116150</td>
          <td>20.420225</td>
          <td>0.005299</td>
          <td>23.816669</td>
          <td>0.080428</td>
          <td>23.504330</td>
          <td>26.753460</td>
          <td>22.626769</td>
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
          <td>21.831258</td>
          <td>23.028346</td>
          <td>28.589577</td>
          <td>23.861054</td>
          <td>23.111832</td>
          <td>29.271242</td>
          <td>21.340986</td>
          <td>0.005137</td>
          <td>20.130534</td>
          <td>0.005045</td>
          <td>24.146370</td>
          <td>0.027356</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.021007</td>
          <td>28.872908</td>
          <td>28.642188</td>
          <td>23.161870</td>
          <td>24.262689</td>
          <td>20.745633</td>
          <td>27.995334</td>
          <td>0.440405</td>
          <td>27.208715</td>
          <td>0.381678</td>
          <td>21.291282</td>
          <td>0.005370</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.873743</td>
          <td>25.929211</td>
          <td>26.123432</td>
          <td>21.756943</td>
          <td>24.657134</td>
          <td>24.252534</td>
          <td>26.637979</td>
          <td>0.145463</td>
          <td>26.611717</td>
          <td>0.236270</td>
          <td>24.726363</td>
          <td>0.045774</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.497698</td>
          <td>25.164427</td>
          <td>24.953891</td>
          <td>22.700626</td>
          <td>25.901722</td>
          <td>27.195494</td>
          <td>21.469475</td>
          <td>0.005173</td>
          <td>18.982902</td>
          <td>0.005005</td>
          <td>22.990794</td>
          <td>0.010600</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.976675</td>
          <td>23.129548</td>
          <td>24.918256</td>
          <td>25.726265</td>
          <td>26.578601</td>
          <td>20.934747</td>
          <td>19.370466</td>
          <td>0.005004</td>
          <td>24.545245</td>
          <td>0.038952</td>
          <td>20.879762</td>
          <td>0.005176</td>
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
          <td>17.081842</td>
          <td>19.822631</td>
          <td>27.088093</td>
          <td>19.750104</td>
          <td>24.200047</td>
          <td>21.892643</td>
          <td>23.873693</td>
          <td>0.013106</td>
          <td>23.126582</td>
          <td>0.011708</td>
          <td>20.769021</td>
          <td>0.005144</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.034327</td>
          <td>18.958048</td>
          <td>23.860043</td>
          <td>24.776219</td>
          <td>24.440288</td>
          <td>23.458001</td>
          <td>23.077866</td>
          <td>0.007683</td>
          <td>18.589347</td>
          <td>0.005003</td>
          <td>25.944296</td>
          <td>0.134163</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.538537</td>
          <td>24.177222</td>
          <td>27.831640</td>
          <td>18.495173</td>
          <td>27.683777</td>
          <td>25.400233</td>
          <td>17.087801</td>
          <td>0.005000</td>
          <td>21.343516</td>
          <td>0.005405</td>
          <td>18.334938</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.871210</td>
          <td>22.980063</td>
          <td>24.056119</td>
          <td>18.781556</td>
          <td>20.013217</td>
          <td>23.294228</td>
          <td>20.306943</td>
          <td>0.005021</td>
          <td>23.397668</td>
          <td>0.014464</td>
          <td>19.136622</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.917254</td>
          <td>20.596728</td>
          <td>20.168702</td>
          <td>26.179393</td>
          <td>20.421718</td>
          <td>23.799122</td>
          <td>23.514561</td>
          <td>0.010047</td>
          <td>26.334846</td>
          <td>0.187414</td>
          <td>22.619336</td>
          <td>0.008315</td>
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
          <td>21.831258</td>
          <td>23.028346</td>
          <td>28.589577</td>
          <td>23.861054</td>
          <td>23.111832</td>
          <td>29.271242</td>
          <td>21.350433</td>
          <td>0.015097</td>
          <td>20.141492</td>
          <td>0.006344</td>
          <td>24.282405</td>
          <td>0.179272</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.021007</td>
          <td>28.872908</td>
          <td>28.642188</td>
          <td>23.161870</td>
          <td>24.262689</td>
          <td>20.745633</td>
          <td>25.347628</td>
          <td>0.458122</td>
          <td>26.201454</td>
          <td>0.729839</td>
          <td>21.280448</td>
          <td>0.013176</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.873743</td>
          <td>25.929211</td>
          <td>26.123432</td>
          <td>21.756943</td>
          <td>24.657134</td>
          <td>24.252534</td>
          <td>30.291757</td>
          <td>4.265894</td>
          <td>25.919433</td>
          <td>0.600927</td>
          <td>24.706473</td>
          <td>0.255456</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.497698</td>
          <td>25.164427</td>
          <td>24.953891</td>
          <td>22.700626</td>
          <td>25.901722</td>
          <td>27.195494</td>
          <td>21.456117</td>
          <td>0.016469</td>
          <td>18.980105</td>
          <td>0.005177</td>
          <td>22.946127</td>
          <td>0.055680</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.976675</td>
          <td>23.129548</td>
          <td>24.918256</td>
          <td>25.726265</td>
          <td>26.578601</td>
          <td>20.934747</td>
          <td>19.373337</td>
          <td>0.005510</td>
          <td>24.274907</td>
          <td>0.163592</td>
          <td>20.866181</td>
          <td>0.009720</td>
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
          <td>17.081842</td>
          <td>19.822631</td>
          <td>27.088093</td>
          <td>19.750104</td>
          <td>24.200047</td>
          <td>21.892643</td>
          <td>23.647920</td>
          <td>0.113107</td>
          <td>23.205410</td>
          <td>0.064162</td>
          <td>20.768973</td>
          <td>0.009117</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.034327</td>
          <td>18.958048</td>
          <td>23.860043</td>
          <td>24.776219</td>
          <td>24.440288</td>
          <td>23.458001</td>
          <td>23.027011</td>
          <td>0.065406</td>
          <td>18.584371</td>
          <td>0.005086</td>
          <td>25.307978</td>
          <td>0.412053</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.538537</td>
          <td>24.177222</td>
          <td>27.831640</td>
          <td>18.495173</td>
          <td>27.683777</td>
          <td>25.400233</td>
          <td>17.090645</td>
          <td>0.005008</td>
          <td>21.338298</td>
          <td>0.012750</td>
          <td>18.329210</td>
          <td>0.005065</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.871210</td>
          <td>22.980063</td>
          <td>24.056119</td>
          <td>18.781556</td>
          <td>20.013217</td>
          <td>23.294228</td>
          <td>20.306119</td>
          <td>0.007405</td>
          <td>23.204763</td>
          <td>0.064125</td>
          <td>19.137323</td>
          <td>0.005281</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.917254</td>
          <td>20.596728</td>
          <td>20.168702</td>
          <td>26.179393</td>
          <td>20.421718</td>
          <td>23.799122</td>
          <td>23.517720</td>
          <td>0.100923</td>
          <td>28.021325</td>
          <td>1.967486</td>
          <td>22.630813</td>
          <td>0.042037</td>
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


