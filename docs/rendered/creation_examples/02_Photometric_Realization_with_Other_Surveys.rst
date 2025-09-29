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
          <td>23.585143</td>
          <td>19.900961</td>
          <td>25.282584</td>
          <td>16.858096</td>
          <td>22.426649</td>
          <td>19.522585</td>
          <td>25.371290</td>
          <td>25.349726</td>
          <td>25.273670</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.265587</td>
          <td>22.046237</td>
          <td>20.457694</td>
          <td>20.218665</td>
          <td>27.505218</td>
          <td>26.940306</td>
          <td>23.445708</td>
          <td>27.062251</td>
          <td>20.113713</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.978171</td>
          <td>25.188918</td>
          <td>18.504948</td>
          <td>19.876685</td>
          <td>24.065451</td>
          <td>23.331191</td>
          <td>24.567932</td>
          <td>28.361129</td>
          <td>26.131191</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.270650</td>
          <td>29.337990</td>
          <td>25.351899</td>
          <td>18.622050</td>
          <td>23.430548</td>
          <td>20.733837</td>
          <td>21.031339</td>
          <td>25.441480</td>
          <td>21.069947</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.234074</td>
          <td>23.685638</td>
          <td>20.949744</td>
          <td>25.644223</td>
          <td>28.283177</td>
          <td>23.257584</td>
          <td>29.100041</td>
          <td>24.094522</td>
          <td>23.621742</td>
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
          <td>18.129165</td>
          <td>22.678963</td>
          <td>23.876701</td>
          <td>17.970381</td>
          <td>22.733586</td>
          <td>22.181251</td>
          <td>21.504208</td>
          <td>25.003535</td>
          <td>21.125452</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.043975</td>
          <td>24.478822</td>
          <td>20.048860</td>
          <td>18.226384</td>
          <td>24.819161</td>
          <td>20.039941</td>
          <td>20.732798</td>
          <td>23.267801</td>
          <td>21.200186</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.498903</td>
          <td>25.087619</td>
          <td>25.255891</td>
          <td>22.322965</td>
          <td>25.394800</td>
          <td>21.437846</td>
          <td>25.391538</td>
          <td>24.676546</td>
          <td>23.588397</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.559011</td>
          <td>21.213300</td>
          <td>21.886139</td>
          <td>23.581679</td>
          <td>24.409725</td>
          <td>22.687500</td>
          <td>22.717805</td>
          <td>24.046458</td>
          <td>30.483883</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.324020</td>
          <td>18.743812</td>
          <td>19.806491</td>
          <td>26.040732</td>
          <td>25.519914</td>
          <td>20.043639</td>
          <td>16.553529</td>
          <td>23.806367</td>
          <td>24.800750</td>
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
          <td>23.582092</td>
          <td>0.031264</td>
          <td>19.903266</td>
          <td>0.005034</td>
          <td>25.206619</td>
          <td>0.039021</td>
          <td>16.860496</td>
          <td>0.005001</td>
          <td>22.447150</td>
          <td>0.011402</td>
          <td>19.522094</td>
          <td>0.005310</td>
          <td>25.371290</td>
          <td>25.349726</td>
          <td>25.273670</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.309925</td>
          <td>0.059137</td>
          <td>22.048505</td>
          <td>0.005704</td>
          <td>20.450700</td>
          <td>0.005044</td>
          <td>20.215764</td>
          <td>0.005069</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.159668</td>
          <td>0.552021</td>
          <td>23.445708</td>
          <td>27.062251</td>
          <td>20.113713</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.985844</td>
          <td>0.006058</td>
          <td>25.205364</td>
          <td>0.044392</td>
          <td>18.507278</td>
          <td>0.005004</td>
          <td>19.873449</td>
          <td>0.005042</td>
          <td>24.077420</td>
          <td>0.044868</td>
          <td>23.258114</td>
          <td>0.049031</td>
          <td>24.567932</td>
          <td>28.361129</td>
          <td>26.131191</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.258829</td>
          <td>0.023700</td>
          <td>28.975088</td>
          <td>0.916603</td>
          <td>25.373861</td>
          <td>0.045259</td>
          <td>18.623992</td>
          <td>0.005008</td>
          <td>23.454180</td>
          <td>0.025911</td>
          <td>20.723837</td>
          <td>0.007074</td>
          <td>21.031339</td>
          <td>25.441480</td>
          <td>21.069947</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.738255</td>
          <td>0.903061</td>
          <td>23.690478</td>
          <td>0.012386</td>
          <td>20.936175</td>
          <td>0.005089</td>
          <td>25.794310</td>
          <td>0.107037</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.234784</td>
          <td>0.048026</td>
          <td>29.100041</td>
          <td>24.094522</td>
          <td>23.621742</td>
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
          <td>18.126286</td>
          <td>0.005032</td>
          <td>22.688366</td>
          <td>0.006864</td>
          <td>23.874866</td>
          <td>0.012656</td>
          <td>17.965982</td>
          <td>0.005004</td>
          <td>22.720159</td>
          <td>0.014024</td>
          <td>22.180792</td>
          <td>0.019129</td>
          <td>21.504208</td>
          <td>25.003535</td>
          <td>21.125452</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.004699</td>
          <td>0.019173</td>
          <td>24.466941</td>
          <td>0.023265</td>
          <td>20.047395</td>
          <td>0.005025</td>
          <td>18.233721</td>
          <td>0.005005</td>
          <td>24.741471</td>
          <td>0.080822</td>
          <td>20.051158</td>
          <td>0.005725</td>
          <td>20.732798</td>
          <td>23.267801</td>
          <td>21.200186</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.494273</td>
          <td>0.007149</td>
          <td>25.087120</td>
          <td>0.039986</td>
          <td>25.243301</td>
          <td>0.040310</td>
          <td>22.322419</td>
          <td>0.006903</td>
          <td>25.444514</td>
          <td>0.149191</td>
          <td>21.462588</td>
          <td>0.010881</td>
          <td>25.391538</td>
          <td>24.676546</td>
          <td>23.588397</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.682029</td>
          <td>0.082016</td>
          <td>21.217001</td>
          <td>0.005199</td>
          <td>21.902166</td>
          <td>0.005406</td>
          <td>23.591221</td>
          <td>0.015650</td>
          <td>24.455193</td>
          <td>0.062740</td>
          <td>22.727572</td>
          <td>0.030666</td>
          <td>22.717805</td>
          <td>24.046458</td>
          <td>30.483883</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.322467</td>
          <td>0.025015</td>
          <td>18.748546</td>
          <td>0.005009</td>
          <td>19.802803</td>
          <td>0.005018</td>
          <td>25.866366</td>
          <td>0.113983</td>
          <td>25.415064</td>
          <td>0.145463</td>
          <td>20.041660</td>
          <td>0.005714</td>
          <td>16.553529</td>
          <td>23.806367</td>
          <td>24.800750</td>
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
          <td>23.585143</td>
          <td>19.900961</td>
          <td>25.282584</td>
          <td>16.858096</td>
          <td>22.426649</td>
          <td>19.522585</td>
          <td>25.410401</td>
          <td>0.049336</td>
          <td>25.311705</td>
          <td>0.077053</td>
          <td>25.384490</td>
          <td>0.082179</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.265587</td>
          <td>22.046237</td>
          <td>20.457694</td>
          <td>20.218665</td>
          <td>27.505218</td>
          <td>26.940306</td>
          <td>23.438274</td>
          <td>0.009540</td>
          <td>26.985598</td>
          <td>0.320199</td>
          <td>20.114557</td>
          <td>0.005044</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.978171</td>
          <td>25.188918</td>
          <td>18.504948</td>
          <td>19.876685</td>
          <td>24.065451</td>
          <td>23.331191</td>
          <td>24.587895</td>
          <td>0.023810</td>
          <td>28.604563</td>
          <td>1.005030</td>
          <td>25.874112</td>
          <td>0.126244</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.270650</td>
          <td>29.337990</td>
          <td>25.351899</td>
          <td>18.622050</td>
          <td>23.430548</td>
          <td>20.733837</td>
          <td>21.027930</td>
          <td>0.005078</td>
          <td>25.411782</td>
          <td>0.084185</td>
          <td>21.078920</td>
          <td>0.005253</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.234074</td>
          <td>23.685638</td>
          <td>20.949744</td>
          <td>25.644223</td>
          <td>28.283177</td>
          <td>23.257584</td>
          <td>28.535650</td>
          <td>0.651850</td>
          <td>24.086696</td>
          <td>0.025959</td>
          <td>23.627124</td>
          <td>0.017475</td>
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
          <td>18.129165</td>
          <td>22.678963</td>
          <td>23.876701</td>
          <td>17.970381</td>
          <td>22.733586</td>
          <td>22.181251</td>
          <td>21.503187</td>
          <td>0.005184</td>
          <td>25.053864</td>
          <td>0.061286</td>
          <td>21.123592</td>
          <td>0.005274</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.043975</td>
          <td>24.478822</td>
          <td>20.048860</td>
          <td>18.226384</td>
          <td>24.819161</td>
          <td>20.039941</td>
          <td>20.726766</td>
          <td>0.005045</td>
          <td>23.268913</td>
          <td>0.013058</td>
          <td>21.198945</td>
          <td>0.005313</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.498903</td>
          <td>25.087619</td>
          <td>25.255891</td>
          <td>22.322965</td>
          <td>25.394800</td>
          <td>21.437846</td>
          <td>25.319298</td>
          <td>0.045487</td>
          <td>24.631694</td>
          <td>0.042070</td>
          <td>23.604453</td>
          <td>0.017146</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.559011</td>
          <td>21.213300</td>
          <td>21.886139</td>
          <td>23.581679</td>
          <td>24.409725</td>
          <td>22.687500</td>
          <td>22.712216</td>
          <td>0.006509</td>
          <td>24.063182</td>
          <td>0.025429</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.324020</td>
          <td>18.743812</td>
          <td>19.806491</td>
          <td>26.040732</td>
          <td>25.519914</td>
          <td>20.043639</td>
          <td>16.563004</td>
          <td>0.005000</td>
          <td>23.807308</td>
          <td>0.020366</td>
          <td>24.822446</td>
          <td>0.049868</td>
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
          <td>23.585143</td>
          <td>19.900961</td>
          <td>25.282584</td>
          <td>16.858096</td>
          <td>22.426649</td>
          <td>19.522585</td>
          <td>25.810969</td>
          <td>0.640779</td>
          <td>25.129484</td>
          <td>0.331576</td>
          <td>25.452772</td>
          <td>0.459895</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.265587</td>
          <td>22.046237</td>
          <td>20.457694</td>
          <td>20.218665</td>
          <td>27.505218</td>
          <td>26.940306</td>
          <td>23.426148</td>
          <td>0.093118</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.125601</td>
          <td>0.006542</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.978171</td>
          <td>25.188918</td>
          <td>18.504948</td>
          <td>19.876685</td>
          <td>24.065451</td>
          <td>23.331191</td>
          <td>24.606738</td>
          <td>0.255512</td>
          <td>27.164533</td>
          <td>1.305215</td>
          <td>25.903053</td>
          <td>0.637258</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.270650</td>
          <td>29.337990</td>
          <td>25.351899</td>
          <td>18.622050</td>
          <td>23.430548</td>
          <td>20.733837</td>
          <td>21.008503</td>
          <td>0.011550</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.077941</td>
          <td>0.011292</td>
        </tr>
        <tr>
          <th>4</th>
          <td>29.234074</td>
          <td>23.685638</td>
          <td>20.949744</td>
          <td>25.644223</td>
          <td>28.283177</td>
          <td>23.257584</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.151740</td>
          <td>0.147197</td>
          <td>23.722503</td>
          <td>0.110623</td>
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
          <td>18.129165</td>
          <td>22.678963</td>
          <td>23.876701</td>
          <td>17.970381</td>
          <td>22.733586</td>
          <td>22.181251</td>
          <td>21.510746</td>
          <td>0.017237</td>
          <td>24.884785</td>
          <td>0.272344</td>
          <td>21.130775</td>
          <td>0.011745</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.043975</td>
          <td>24.478822</td>
          <td>20.048860</td>
          <td>18.226384</td>
          <td>24.819161</td>
          <td>20.039941</td>
          <td>20.722687</td>
          <td>0.009442</td>
          <td>23.268087</td>
          <td>0.067837</td>
          <td>21.199828</td>
          <td>0.012377</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.498903</td>
          <td>25.087619</td>
          <td>25.255891</td>
          <td>22.322965</td>
          <td>25.394800</td>
          <td>21.437846</td>
          <td>25.122316</td>
          <td>0.385726</td>
          <td>24.806042</td>
          <td>0.255366</td>
          <td>23.595602</td>
          <td>0.098982</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.559011</td>
          <td>21.213300</td>
          <td>21.886139</td>
          <td>23.581679</td>
          <td>24.409725</td>
          <td>22.687500</td>
          <td>22.700206</td>
          <td>0.048889</td>
          <td>24.206743</td>
          <td>0.154319</td>
          <td>27.960604</td>
          <td>2.000446</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.324020</td>
          <td>18.743812</td>
          <td>19.806491</td>
          <td>26.040732</td>
          <td>25.519914</td>
          <td>20.043639</td>
          <td>16.546642</td>
          <td>0.005003</td>
          <td>23.730054</td>
          <td>0.102021</td>
          <td>24.782531</td>
          <td>0.271845</td>
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


