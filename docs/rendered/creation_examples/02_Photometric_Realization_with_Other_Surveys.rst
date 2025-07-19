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
          <td>17.805083</td>
          <td>26.456931</td>
          <td>22.784570</td>
          <td>21.071740</td>
          <td>19.372245</td>
          <td>22.206585</td>
          <td>21.824974</td>
          <td>22.949115</td>
          <td>20.574373</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.776463</td>
          <td>30.594003</td>
          <td>23.892064</td>
          <td>29.226631</td>
          <td>24.881620</td>
          <td>23.377169</td>
          <td>21.257626</td>
          <td>25.893228</td>
          <td>22.958333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.751530</td>
          <td>21.995271</td>
          <td>22.527203</td>
          <td>24.875962</td>
          <td>23.029736</td>
          <td>25.166940</td>
          <td>26.465972</td>
          <td>22.167477</td>
          <td>20.981851</td>
        </tr>
        <tr>
          <th>3</th>
          <td>33.307651</td>
          <td>21.885966</td>
          <td>19.381886</td>
          <td>26.458601</td>
          <td>20.165643</td>
          <td>24.094057</td>
          <td>23.352689</td>
          <td>21.155936</td>
          <td>24.504644</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.686217</td>
          <td>20.775916</td>
          <td>25.988071</td>
          <td>20.947180</td>
          <td>25.643924</td>
          <td>26.260881</td>
          <td>24.243633</td>
          <td>22.855883</td>
          <td>21.635502</td>
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
          <td>27.013869</td>
          <td>25.018544</td>
          <td>22.554348</td>
          <td>20.846662</td>
          <td>26.668947</td>
          <td>23.322762</td>
          <td>22.434725</td>
          <td>20.618849</td>
          <td>21.185383</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.627575</td>
          <td>22.205780</td>
          <td>23.113269</td>
          <td>18.041789</td>
          <td>19.966436</td>
          <td>25.311361</td>
          <td>24.675650</td>
          <td>20.708243</td>
          <td>25.387712</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.637280</td>
          <td>18.175695</td>
          <td>26.686664</td>
          <td>25.487720</td>
          <td>23.397444</td>
          <td>22.492340</td>
          <td>27.973058</td>
          <td>20.374608</td>
          <td>23.234769</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.303993</td>
          <td>19.111586</td>
          <td>25.199388</td>
          <td>30.071928</td>
          <td>22.003347</td>
          <td>21.797322</td>
          <td>25.704615</td>
          <td>24.897821</td>
          <td>19.321218</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.795478</td>
          <td>21.210168</td>
          <td>23.824416</td>
          <td>18.261661</td>
          <td>22.960118</td>
          <td>19.704976</td>
          <td>23.728514</td>
          <td>20.672299</td>
          <td>28.913505</td>
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
          <td>17.806049</td>
          <td>0.005023</td>
          <td>26.539824</td>
          <td>0.143322</td>
          <td>22.783320</td>
          <td>0.006654</td>
          <td>21.070525</td>
          <td>0.005259</td>
          <td>19.371746</td>
          <td>0.005060</td>
          <td>22.225213</td>
          <td>0.019859</td>
          <td>21.824974</td>
          <td>22.949115</td>
          <td>20.574373</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.776728</td>
          <td>0.005064</td>
          <td>29.495434</td>
          <td>1.242807</td>
          <td>23.892501</td>
          <td>0.012828</td>
          <td>27.746611</td>
          <td>0.526524</td>
          <td>24.828948</td>
          <td>0.087298</td>
          <td>23.308979</td>
          <td>0.051296</td>
          <td>21.257626</td>
          <td>25.893228</td>
          <td>22.958333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.755595</td>
          <td>0.005203</td>
          <td>21.996561</td>
          <td>0.005649</td>
          <td>22.531490</td>
          <td>0.006116</td>
          <td>24.816467</td>
          <td>0.045103</td>
          <td>23.025950</td>
          <td>0.017976</td>
          <td>25.330437</td>
          <td>0.292254</td>
          <td>26.465972</td>
          <td>22.167477</td>
          <td>20.981851</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.818518</td>
          <td>0.949091</td>
          <td>21.880608</td>
          <td>0.005543</td>
          <td>19.386947</td>
          <td>0.005011</td>
          <td>26.393184</td>
          <td>0.179344</td>
          <td>20.168732</td>
          <td>0.005201</td>
          <td>24.217868</td>
          <td>0.114353</td>
          <td>23.352689</td>
          <td>21.155936</td>
          <td>24.504644</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.812951</td>
          <td>0.945851</td>
          <td>20.773774</td>
          <td>0.005105</td>
          <td>25.985367</td>
          <td>0.077834</td>
          <td>20.950579</td>
          <td>0.005214</td>
          <td>25.878546</td>
          <td>0.215485</td>
          <td>27.329611</td>
          <td>1.173151</td>
          <td>24.243633</td>
          <td>22.855883</td>
          <td>21.635502</td>
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
          <td>27.451210</td>
          <td>0.750454</td>
          <td>25.031139</td>
          <td>0.038058</td>
          <td>22.554831</td>
          <td>0.006158</td>
          <td>20.849616</td>
          <td>0.005182</td>
          <td>26.547299</td>
          <td>0.370129</td>
          <td>23.424957</td>
          <td>0.056858</td>
          <td>22.434725</td>
          <td>20.618849</td>
          <td>21.185383</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.664751</td>
          <td>0.080782</td>
          <td>22.210277</td>
          <td>0.005903</td>
          <td>23.092687</td>
          <td>0.007635</td>
          <td>18.039388</td>
          <td>0.005004</td>
          <td>19.968382</td>
          <td>0.005147</td>
          <td>25.423429</td>
          <td>0.314909</td>
          <td>24.675650</td>
          <td>20.708243</td>
          <td>25.387712</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.639567</td>
          <td>0.007621</td>
          <td>18.177929</td>
          <td>0.005005</td>
          <td>26.641826</td>
          <td>0.138166</td>
          <td>25.480574</td>
          <td>0.081257</td>
          <td>23.418494</td>
          <td>0.025120</td>
          <td>22.439679</td>
          <td>0.023859</td>
          <td>27.973058</td>
          <td>20.374608</td>
          <td>23.234769</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.298473</td>
          <td>0.005005</td>
          <td>19.120809</td>
          <td>0.005014</td>
          <td>25.210785</td>
          <td>0.039165</td>
          <td>27.824688</td>
          <td>0.557184</td>
          <td>21.986453</td>
          <td>0.008427</td>
          <td>21.777909</td>
          <td>0.013774</td>
          <td>25.704615</td>
          <td>24.897821</td>
          <td>19.321218</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.788949</td>
          <td>0.005023</td>
          <td>21.208489</td>
          <td>0.005197</td>
          <td>23.822099</td>
          <td>0.012162</td>
          <td>18.263627</td>
          <td>0.005006</td>
          <td>22.949199</td>
          <td>0.016868</td>
          <td>19.706022</td>
          <td>0.005416</td>
          <td>23.728514</td>
          <td>20.672299</td>
          <td>28.913505</td>
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
          <td>17.805083</td>
          <td>26.456931</td>
          <td>22.784570</td>
          <td>21.071740</td>
          <td>19.372245</td>
          <td>22.206585</td>
          <td>21.827466</td>
          <td>0.005330</td>
          <td>22.943864</td>
          <td>0.010254</td>
          <td>20.575756</td>
          <td>0.005102</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.776463</td>
          <td>30.594003</td>
          <td>23.892064</td>
          <td>29.226631</td>
          <td>24.881620</td>
          <td>23.377169</td>
          <td>21.261684</td>
          <td>0.005119</td>
          <td>25.871244</td>
          <td>0.125930</td>
          <td>22.964921</td>
          <td>0.010407</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.751530</td>
          <td>21.995271</td>
          <td>22.527203</td>
          <td>24.875962</td>
          <td>23.029736</td>
          <td>25.166940</td>
          <td>26.719483</td>
          <td>0.156014</td>
          <td>22.161193</td>
          <td>0.006634</td>
          <td>20.976062</td>
          <td>0.005210</td>
        </tr>
        <tr>
          <th>3</th>
          <td>33.307651</td>
          <td>21.885966</td>
          <td>19.381886</td>
          <td>26.458601</td>
          <td>20.165643</td>
          <td>24.094057</td>
          <td>23.347338</td>
          <td>0.008992</td>
          <td>21.158019</td>
          <td>0.005291</td>
          <td>24.520151</td>
          <td>0.038092</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.686217</td>
          <td>20.775916</td>
          <td>25.988071</td>
          <td>20.947180</td>
          <td>25.643924</td>
          <td>26.260881</td>
          <td>24.255514</td>
          <td>0.017898</td>
          <td>22.849288</td>
          <td>0.009610</td>
          <td>21.647348</td>
          <td>0.005691</td>
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
          <td>27.013869</td>
          <td>25.018544</td>
          <td>22.554348</td>
          <td>20.846662</td>
          <td>26.668947</td>
          <td>23.322762</td>
          <td>22.443615</td>
          <td>0.005966</td>
          <td>20.616984</td>
          <td>0.005109</td>
          <td>21.179357</td>
          <td>0.005303</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.627575</td>
          <td>22.205780</td>
          <td>23.113269</td>
          <td>18.041789</td>
          <td>19.966436</td>
          <td>25.311361</td>
          <td>24.655468</td>
          <td>0.025258</td>
          <td>20.713177</td>
          <td>0.005130</td>
          <td>25.313027</td>
          <td>0.077144</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.637280</td>
          <td>18.175695</td>
          <td>26.686664</td>
          <td>25.487720</td>
          <td>23.397444</td>
          <td>22.492340</td>
          <td>27.998731</td>
          <td>0.441539</td>
          <td>20.380193</td>
          <td>0.005071</td>
          <td>23.226471</td>
          <td>0.012633</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.303993</td>
          <td>19.111586</td>
          <td>25.199388</td>
          <td>30.071928</td>
          <td>22.003347</td>
          <td>21.797322</td>
          <td>25.726121</td>
          <td>0.065354</td>
          <td>24.927193</td>
          <td>0.054749</td>
          <td>19.313889</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.795478</td>
          <td>21.210168</td>
          <td>23.824416</td>
          <td>18.261661</td>
          <td>22.960118</td>
          <td>19.704976</td>
          <td>23.717954</td>
          <td>0.011632</td>
          <td>20.673400</td>
          <td>0.005121</td>
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
          <td>17.805083</td>
          <td>26.456931</td>
          <td>22.784570</td>
          <td>21.071740</td>
          <td>19.372245</td>
          <td>22.206585</td>
          <td>21.824264</td>
          <td>0.022529</td>
          <td>22.989672</td>
          <td>0.052948</td>
          <td>20.576506</td>
          <td>0.008112</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.776463</td>
          <td>30.594003</td>
          <td>23.892064</td>
          <td>29.226631</td>
          <td>24.881620</td>
          <td>23.377169</td>
          <td>21.248300</td>
          <td>0.013903</td>
          <td>27.127334</td>
          <td>1.279331</td>
          <td>22.901214</td>
          <td>0.053496</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.751530</td>
          <td>21.995271</td>
          <td>22.527203</td>
          <td>24.875962</td>
          <td>23.029736</td>
          <td>25.166940</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.193246</td>
          <td>0.026108</td>
          <td>21.000222</td>
          <td>0.010671</td>
        </tr>
        <tr>
          <th>3</th>
          <td>33.307651</td>
          <td>21.885966</td>
          <td>19.381886</td>
          <td>26.458601</td>
          <td>20.165643</td>
          <td>24.094057</td>
          <td>23.247664</td>
          <td>0.079546</td>
          <td>21.169044</td>
          <td>0.011218</td>
          <td>24.281308</td>
          <td>0.179105</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.686217</td>
          <td>20.775916</td>
          <td>25.988071</td>
          <td>20.947180</td>
          <td>25.643924</td>
          <td>26.260881</td>
          <td>24.222199</td>
          <td>0.185420</td>
          <td>22.790637</td>
          <td>0.044339</td>
          <td>21.620535</td>
          <td>0.017379</td>
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
          <td>27.013869</td>
          <td>25.018544</td>
          <td>22.554348</td>
          <td>20.846662</td>
          <td>26.668947</td>
          <td>23.322762</td>
          <td>22.453462</td>
          <td>0.039238</td>
          <td>20.603839</td>
          <td>0.007791</td>
          <td>21.201910</td>
          <td>0.012397</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.627575</td>
          <td>22.205780</td>
          <td>23.113269</td>
          <td>18.041789</td>
          <td>19.966436</td>
          <td>25.311361</td>
          <td>25.079360</td>
          <td>0.373061</td>
          <td>20.729079</td>
          <td>0.008363</td>
          <td>25.127534</td>
          <td>0.358246</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.637280</td>
          <td>18.175695</td>
          <td>26.686664</td>
          <td>25.487720</td>
          <td>23.397444</td>
          <td>22.492340</td>
          <td>27.312139</td>
          <td>1.559525</td>
          <td>20.384665</td>
          <td>0.006990</td>
          <td>23.308461</td>
          <td>0.076832</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.303993</td>
          <td>19.111586</td>
          <td>25.199388</td>
          <td>30.071928</td>
          <td>22.003347</td>
          <td>21.797322</td>
          <td>24.887082</td>
          <td>0.320578</td>
          <td>25.505892</td>
          <td>0.443936</td>
          <td>19.326861</td>
          <td>0.005394</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.795478</td>
          <td>21.210168</td>
          <td>23.824416</td>
          <td>18.261661</td>
          <td>22.960118</td>
          <td>19.704976</td>
          <td>23.890410</td>
          <td>0.139618</td>
          <td>20.666896</td>
          <td>0.008068</td>
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


