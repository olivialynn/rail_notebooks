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
          <td>24.830458</td>
          <td>18.671596</td>
          <td>27.233500</td>
          <td>23.874603</td>
          <td>25.739305</td>
          <td>24.145561</td>
          <td>27.073790</td>
          <td>25.907755</td>
          <td>25.712520</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.072902</td>
          <td>18.772199</td>
          <td>23.457085</td>
          <td>16.278015</td>
          <td>20.890538</td>
          <td>28.215653</td>
          <td>19.819411</td>
          <td>20.016679</td>
          <td>24.843200</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.368898</td>
          <td>22.190638</td>
          <td>22.382570</td>
          <td>27.734757</td>
          <td>25.986085</td>
          <td>22.420473</td>
          <td>23.786990</td>
          <td>22.853951</td>
          <td>23.414377</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.257463</td>
          <td>28.211800</td>
          <td>24.536880</td>
          <td>26.586835</td>
          <td>22.320751</td>
          <td>22.656333</td>
          <td>26.272109</td>
          <td>21.482740</td>
          <td>19.761598</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.853227</td>
          <td>21.935157</td>
          <td>26.826255</td>
          <td>24.907800</td>
          <td>28.996040</td>
          <td>13.229757</td>
          <td>25.370612</td>
          <td>19.322039</td>
          <td>21.646457</td>
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
          <td>26.151461</td>
          <td>19.322712</td>
          <td>20.196659</td>
          <td>20.905331</td>
          <td>25.332587</td>
          <td>20.302639</td>
          <td>19.509536</td>
          <td>21.172956</td>
          <td>20.474423</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.496745</td>
          <td>21.172430</td>
          <td>24.440205</td>
          <td>24.642579</td>
          <td>24.995670</td>
          <td>25.475810</td>
          <td>22.096497</td>
          <td>23.490390</td>
          <td>23.331213</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.020889</td>
          <td>22.591103</td>
          <td>19.545526</td>
          <td>23.531875</td>
          <td>22.387958</td>
          <td>20.696441</td>
          <td>23.410525</td>
          <td>20.468292</td>
          <td>18.658454</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.630692</td>
          <td>22.771388</td>
          <td>23.931829</td>
          <td>24.912983</td>
          <td>23.575449</td>
          <td>20.465844</td>
          <td>23.052885</td>
          <td>19.048318</td>
          <td>26.179756</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.864555</td>
          <td>20.746920</td>
          <td>22.152576</td>
          <td>20.363411</td>
          <td>25.464692</td>
          <td>23.025613</td>
          <td>26.497948</td>
          <td>26.853195</td>
          <td>20.478425</td>
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
          <td>24.796809</td>
          <td>0.090688</td>
          <td>18.661600</td>
          <td>0.005008</td>
          <td>26.954577</td>
          <td>0.180535</td>
          <td>23.900679</td>
          <td>0.020243</td>
          <td>25.964710</td>
          <td>0.231488</td>
          <td>24.215225</td>
          <td>0.114090</td>
          <td>27.073790</td>
          <td>25.907755</td>
          <td>25.712520</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.069651</td>
          <td>0.005089</td>
          <td>18.765980</td>
          <td>0.005009</td>
          <td>23.452691</td>
          <td>0.009397</td>
          <td>16.277100</td>
          <td>0.005001</td>
          <td>20.897164</td>
          <td>0.005644</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.819411</td>
          <td>20.016679</td>
          <td>24.843200</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.366660</td>
          <td>0.005006</td>
          <td>22.191185</td>
          <td>0.005877</td>
          <td>22.386463</td>
          <td>0.005886</td>
          <td>27.307684</td>
          <td>0.378143</td>
          <td>25.982906</td>
          <td>0.235001</td>
          <td>22.404494</td>
          <td>0.023146</td>
          <td>23.786990</td>
          <td>22.853951</td>
          <td>23.414377</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.279561</td>
          <td>0.024120</td>
          <td>30.883764</td>
          <td>2.347676</td>
          <td>24.540865</td>
          <td>0.021798</td>
          <td>26.598295</td>
          <td>0.213132</td>
          <td>22.329325</td>
          <td>0.010487</td>
          <td>22.668003</td>
          <td>0.029105</td>
          <td>26.272109</td>
          <td>21.482740</td>
          <td>19.761598</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.854811</td>
          <td>0.005070</td>
          <td>21.929468</td>
          <td>0.005586</td>
          <td>26.556301</td>
          <td>0.128320</td>
          <td>24.926920</td>
          <td>0.049751</td>
          <td>27.301785</td>
          <td>0.646368</td>
          <td>13.227132</td>
          <td>0.005000</td>
          <td>25.370612</td>
          <td>19.322039</td>
          <td>21.646457</td>
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
          <td>26.037723</td>
          <td>0.260763</td>
          <td>19.328700</td>
          <td>0.005017</td>
          <td>20.196330</td>
          <td>0.005031</td>
          <td>20.908474</td>
          <td>0.005200</td>
          <td>25.194504</td>
          <td>0.120207</td>
          <td>20.302165</td>
          <td>0.006081</td>
          <td>19.509536</td>
          <td>21.172956</td>
          <td>20.474423</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.423181</td>
          <td>0.155947</td>
          <td>21.179948</td>
          <td>0.005189</td>
          <td>24.435060</td>
          <td>0.019919</td>
          <td>24.625196</td>
          <td>0.038068</td>
          <td>25.025445</td>
          <td>0.103730</td>
          <td>25.302652</td>
          <td>0.285766</td>
          <td>22.096497</td>
          <td>23.490390</td>
          <td>23.331213</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.215500</td>
          <td>0.301135</td>
          <td>22.596988</td>
          <td>0.006626</td>
          <td>19.550651</td>
          <td>0.005013</td>
          <td>23.515391</td>
          <td>0.014723</td>
          <td>22.412589</td>
          <td>0.011121</td>
          <td>20.695401</td>
          <td>0.006987</td>
          <td>23.410525</td>
          <td>20.468292</td>
          <td>18.658454</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.774929</td>
          <td>0.007119</td>
          <td>23.938860</td>
          <td>0.013293</td>
          <td>24.921739</td>
          <td>0.049522</td>
          <td>23.608827</td>
          <td>0.029659</td>
          <td>20.455815</td>
          <td>0.006375</td>
          <td>23.052885</td>
          <td>19.048318</td>
          <td>26.179756</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.819711</td>
          <td>0.038448</td>
          <td>20.743213</td>
          <td>0.005101</td>
          <td>22.149767</td>
          <td>0.005606</td>
          <td>20.373431</td>
          <td>0.005087</td>
          <td>25.334782</td>
          <td>0.135740</td>
          <td>23.098212</td>
          <td>0.042545</td>
          <td>26.497948</td>
          <td>26.853195</td>
          <td>20.478425</td>
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
          <td>24.830458</td>
          <td>18.671596</td>
          <td>27.233500</td>
          <td>23.874603</td>
          <td>25.739305</td>
          <td>24.145561</td>
          <td>27.357629</td>
          <td>0.266380</td>
          <td>26.015590</td>
          <td>0.142684</td>
          <td>25.658243</td>
          <td>0.104574</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.072902</td>
          <td>18.772199</td>
          <td>23.457085</td>
          <td>16.278015</td>
          <td>20.890538</td>
          <td>28.215653</td>
          <td>19.815236</td>
          <td>0.005008</td>
          <td>20.019441</td>
          <td>0.005037</td>
          <td>24.813753</td>
          <td>0.049483</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.368898</td>
          <td>22.190638</td>
          <td>22.382570</td>
          <td>27.734757</td>
          <td>25.986085</td>
          <td>22.420473</td>
          <td>23.775889</td>
          <td>0.012152</td>
          <td>22.846222</td>
          <td>0.009591</td>
          <td>23.438426</td>
          <td>0.014950</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.257463</td>
          <td>28.211800</td>
          <td>24.536880</td>
          <td>26.586835</td>
          <td>22.320751</td>
          <td>22.656333</td>
          <td>26.356259</td>
          <td>0.113933</td>
          <td>21.473505</td>
          <td>0.005510</td>
          <td>19.759347</td>
          <td>0.005023</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.853227</td>
          <td>21.935157</td>
          <td>26.826255</td>
          <td>24.907800</td>
          <td>28.996040</td>
          <td>13.229757</td>
          <td>25.428824</td>
          <td>0.050153</td>
          <td>19.323776</td>
          <td>0.005010</td>
          <td>21.643534</td>
          <td>0.005686</td>
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
          <td>26.151461</td>
          <td>19.322712</td>
          <td>20.196659</td>
          <td>20.905331</td>
          <td>25.332587</td>
          <td>20.302639</td>
          <td>19.502526</td>
          <td>0.005005</td>
          <td>21.186733</td>
          <td>0.005307</td>
          <td>20.482146</td>
          <td>0.005086</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.496745</td>
          <td>21.172430</td>
          <td>24.440205</td>
          <td>24.642579</td>
          <td>24.995670</td>
          <td>25.475810</td>
          <td>22.085790</td>
          <td>0.005521</td>
          <td>23.490370</td>
          <td>0.015598</td>
          <td>23.321712</td>
          <td>0.013612</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.020889</td>
          <td>22.591103</td>
          <td>19.545526</td>
          <td>23.531875</td>
          <td>22.387958</td>
          <td>20.696441</td>
          <td>23.418057</td>
          <td>0.009413</td>
          <td>20.474844</td>
          <td>0.005084</td>
          <td>18.655630</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.630692</td>
          <td>22.771388</td>
          <td>23.931829</td>
          <td>24.912983</td>
          <td>23.575449</td>
          <td>20.465844</td>
          <td>23.045446</td>
          <td>0.007554</td>
          <td>19.056878</td>
          <td>0.005006</td>
          <td>26.079773</td>
          <td>0.150788</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.864555</td>
          <td>20.746920</td>
          <td>22.152576</td>
          <td>20.363411</td>
          <td>25.464692</td>
          <td>23.025613</td>
          <td>26.356742</td>
          <td>0.113981</td>
          <td>27.895416</td>
          <td>0.633874</td>
          <td>20.477377</td>
          <td>0.005085</td>
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
          <td>24.830458</td>
          <td>18.671596</td>
          <td>27.233500</td>
          <td>23.874603</td>
          <td>25.739305</td>
          <td>24.145561</td>
          <td>26.063730</td>
          <td>0.760764</td>
          <td>25.673880</td>
          <td>0.503255</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.072902</td>
          <td>18.772199</td>
          <td>23.457085</td>
          <td>16.278015</td>
          <td>20.890538</td>
          <td>28.215653</td>
          <td>19.822795</td>
          <td>0.006104</td>
          <td>20.020293</td>
          <td>0.006099</td>
          <td>24.749339</td>
          <td>0.264582</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.368898</td>
          <td>22.190638</td>
          <td>22.382570</td>
          <td>27.734757</td>
          <td>25.986085</td>
          <td>22.420473</td>
          <td>23.662963</td>
          <td>0.114602</td>
          <td>22.821914</td>
          <td>0.045593</td>
          <td>23.320353</td>
          <td>0.077646</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.257463</td>
          <td>28.211800</td>
          <td>24.536880</td>
          <td>26.586835</td>
          <td>22.320751</td>
          <td>22.656333</td>
          <td>25.633561</td>
          <td>0.565255</td>
          <td>21.483271</td>
          <td>0.014298</td>
          <td>19.765740</td>
          <td>0.005846</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.853227</td>
          <td>21.935157</td>
          <td>26.826255</td>
          <td>24.907800</td>
          <td>28.996040</td>
          <td>13.229757</td>
          <td>25.231635</td>
          <td>0.419580</td>
          <td>19.329148</td>
          <td>0.005331</td>
          <td>21.640403</td>
          <td>0.017672</td>
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
          <td>26.151461</td>
          <td>19.322712</td>
          <td>20.196659</td>
          <td>20.905331</td>
          <td>25.332587</td>
          <td>20.302639</td>
          <td>19.511613</td>
          <td>0.005649</td>
          <td>21.176736</td>
          <td>0.011282</td>
          <td>20.486370</td>
          <td>0.007718</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.496745</td>
          <td>21.172430</td>
          <td>24.440205</td>
          <td>24.642579</td>
          <td>24.995670</td>
          <td>25.475810</td>
          <td>22.120789</td>
          <td>0.029210</td>
          <td>23.560845</td>
          <td>0.087911</td>
          <td>23.446697</td>
          <td>0.086820</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.020889</td>
          <td>22.591103</td>
          <td>19.545526</td>
          <td>23.531875</td>
          <td>22.387958</td>
          <td>20.696441</td>
          <td>23.303375</td>
          <td>0.083562</td>
          <td>20.477874</td>
          <td>0.007302</td>
          <td>18.660522</td>
          <td>0.005119</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.630692</td>
          <td>22.771388</td>
          <td>23.931829</td>
          <td>24.912983</td>
          <td>23.575449</td>
          <td>20.465844</td>
          <td>23.069184</td>
          <td>0.067903</td>
          <td>19.048900</td>
          <td>0.005200</td>
          <td>27.406924</td>
          <td>1.555553</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.864555</td>
          <td>20.746920</td>
          <td>22.152576</td>
          <td>20.363411</td>
          <td>25.464692</td>
          <td>23.025613</td>
          <td>26.048432</td>
          <td>0.753084</td>
          <td>27.441919</td>
          <td>1.506420</td>
          <td>20.478394</td>
          <td>0.007686</td>
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


