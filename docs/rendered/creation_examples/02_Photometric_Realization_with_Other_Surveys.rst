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
          <td>23.757542</td>
          <td>21.922052</td>
          <td>24.974657</td>
          <td>23.809748</td>
          <td>24.778852</td>
          <td>25.591741</td>
          <td>21.236403</td>
          <td>21.190175</td>
          <td>20.968713</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.859909</td>
          <td>21.505056</td>
          <td>28.176642</td>
          <td>17.486255</td>
          <td>26.366688</td>
          <td>21.759347</td>
          <td>25.671585</td>
          <td>21.085216</td>
          <td>26.708914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.540612</td>
          <td>24.953302</td>
          <td>24.186426</td>
          <td>19.647438</td>
          <td>25.853584</td>
          <td>24.941343</td>
          <td>26.470114</td>
          <td>22.536475</td>
          <td>21.252877</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.718616</td>
          <td>19.948641</td>
          <td>22.331085</td>
          <td>23.982547</td>
          <td>19.495133</td>
          <td>26.876261</td>
          <td>24.567914</td>
          <td>22.915181</td>
          <td>27.816441</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.262325</td>
          <td>24.766528</td>
          <td>20.492319</td>
          <td>21.571059</td>
          <td>22.829105</td>
          <td>23.779701</td>
          <td>23.104036</td>
          <td>22.073179</td>
          <td>21.058143</td>
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
          <td>21.081209</td>
          <td>26.241361</td>
          <td>25.351154</td>
          <td>21.745124</td>
          <td>18.589233</td>
          <td>25.219284</td>
          <td>21.821678</td>
          <td>19.970967</td>
          <td>22.812178</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.603131</td>
          <td>21.007275</td>
          <td>27.348369</td>
          <td>26.080688</td>
          <td>23.528707</td>
          <td>18.298022</td>
          <td>24.424957</td>
          <td>23.532199</td>
          <td>24.407914</td>
        </tr>
        <tr>
          <th>997</th>
          <td>14.782608</td>
          <td>21.245918</td>
          <td>22.141351</td>
          <td>24.513056</td>
          <td>21.733797</td>
          <td>24.074817</td>
          <td>24.522714</td>
          <td>23.761439</td>
          <td>23.829831</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.429385</td>
          <td>21.725518</td>
          <td>21.232133</td>
          <td>24.817561</td>
          <td>23.780090</td>
          <td>28.383280</td>
          <td>18.916321</td>
          <td>23.298350</td>
          <td>23.128086</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.785518</td>
          <td>19.883186</td>
          <td>24.706564</td>
          <td>22.448203</td>
          <td>28.890098</td>
          <td>23.427291</td>
          <td>20.527756</td>
          <td>21.118318</td>
          <td>27.006559</td>
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
          <td>23.773090</td>
          <td>0.036913</td>
          <td>21.919152</td>
          <td>0.005576</td>
          <td>25.001892</td>
          <td>0.032564</td>
          <td>23.826946</td>
          <td>0.019019</td>
          <td>24.736446</td>
          <td>0.080464</td>
          <td>26.637577</td>
          <td>0.768203</td>
          <td>21.236403</td>
          <td>21.190175</td>
          <td>20.968713</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.850551</td>
          <td>0.039499</td>
          <td>21.506017</td>
          <td>0.005306</td>
          <td>27.906359</td>
          <td>0.391589</td>
          <td>17.481277</td>
          <td>0.005002</td>
          <td>26.974292</td>
          <td>0.511537</td>
          <td>21.768182</td>
          <td>0.013670</td>
          <td>25.671585</td>
          <td>21.085216</td>
          <td>26.708914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.534573</td>
          <td>0.005565</td>
          <td>24.964910</td>
          <td>0.035899</td>
          <td>24.194962</td>
          <td>0.016306</td>
          <td>19.642917</td>
          <td>0.005030</td>
          <td>26.125131</td>
          <td>0.264147</td>
          <td>24.929078</td>
          <td>0.210114</td>
          <td>26.470114</td>
          <td>22.536475</td>
          <td>21.252877</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.712884</td>
          <td>0.005060</td>
          <td>19.954224</td>
          <td>0.005036</td>
          <td>22.324215</td>
          <td>0.005802</td>
          <td>23.977134</td>
          <td>0.021606</td>
          <td>19.498706</td>
          <td>0.005072</td>
          <td>25.695202</td>
          <td>0.389945</td>
          <td>24.567914</td>
          <td>22.915181</td>
          <td>27.816441</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.266967</td>
          <td>0.005393</td>
          <td>24.748200</td>
          <td>0.029683</td>
          <td>20.489432</td>
          <td>0.005046</td>
          <td>21.569172</td>
          <td>0.005578</td>
          <td>22.832504</td>
          <td>0.015337</td>
          <td>23.672430</td>
          <td>0.070804</td>
          <td>23.104036</td>
          <td>22.073179</td>
          <td>21.058143</td>
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
          <td>21.078588</td>
          <td>0.006205</td>
          <td>26.173280</td>
          <td>0.104292</td>
          <td>25.303300</td>
          <td>0.042512</td>
          <td>21.746076</td>
          <td>0.005769</td>
          <td>18.588863</td>
          <td>0.005020</td>
          <td>25.496462</td>
          <td>0.333750</td>
          <td>21.821678</td>
          <td>19.970967</td>
          <td>22.812178</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.560873</td>
          <td>0.073742</td>
          <td>21.012751</td>
          <td>0.005148</td>
          <td>27.099221</td>
          <td>0.203943</td>
          <td>26.081047</td>
          <td>0.137310</td>
          <td>23.539934</td>
          <td>0.027923</td>
          <td>18.295815</td>
          <td>0.005048</td>
          <td>24.424957</td>
          <td>23.532199</td>
          <td>24.407914</td>
        </tr>
        <tr>
          <th>997</th>
          <td>14.775203</td>
          <td>0.005001</td>
          <td>21.248224</td>
          <td>0.005208</td>
          <td>22.152544</td>
          <td>0.005608</td>
          <td>24.544074</td>
          <td>0.035432</td>
          <td>21.724997</td>
          <td>0.007343</td>
          <td>24.205893</td>
          <td>0.113166</td>
          <td>24.522714</td>
          <td>23.761439</td>
          <td>23.829831</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.421506</td>
          <td>0.027223</td>
          <td>21.723383</td>
          <td>0.005427</td>
          <td>21.240791</td>
          <td>0.005142</td>
          <td>24.745601</td>
          <td>0.042355</td>
          <td>23.756678</td>
          <td>0.033778</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.916321</td>
          <td>23.298350</td>
          <td>23.128086</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.773482</td>
          <td>0.015915</td>
          <td>19.886074</td>
          <td>0.005033</td>
          <td>24.705312</td>
          <td>0.025119</td>
          <td>22.442470</td>
          <td>0.007281</td>
          <td>27.837736</td>
          <td>0.919685</td>
          <td>23.377587</td>
          <td>0.054517</td>
          <td>20.527756</td>
          <td>21.118318</td>
          <td>27.006559</td>
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
          <td>23.757542</td>
          <td>21.922052</td>
          <td>24.974657</td>
          <td>23.809748</td>
          <td>24.778852</td>
          <td>25.591741</td>
          <td>21.238757</td>
          <td>0.005114</td>
          <td>21.190203</td>
          <td>0.005309</td>
          <td>20.977716</td>
          <td>0.005211</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.859909</td>
          <td>21.505056</td>
          <td>28.176642</td>
          <td>17.486255</td>
          <td>26.366688</td>
          <td>21.759347</td>
          <td>25.660257</td>
          <td>0.061636</td>
          <td>21.086652</td>
          <td>0.005256</td>
          <td>26.503477</td>
          <td>0.215942</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.540612</td>
          <td>24.953302</td>
          <td>24.186426</td>
          <td>19.647438</td>
          <td>25.853584</td>
          <td>24.941343</td>
          <td>26.269014</td>
          <td>0.105565</td>
          <td>22.525055</td>
          <td>0.007882</td>
          <td>21.256059</td>
          <td>0.005347</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.718616</td>
          <td>19.948641</td>
          <td>22.331085</td>
          <td>23.982547</td>
          <td>19.495133</td>
          <td>26.876261</td>
          <td>24.523307</td>
          <td>0.022510</td>
          <td>22.909127</td>
          <td>0.010009</td>
          <td>26.948973</td>
          <td>0.310964</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.262325</td>
          <td>24.766528</td>
          <td>20.492319</td>
          <td>21.571059</td>
          <td>22.829105</td>
          <td>23.779701</td>
          <td>23.101561</td>
          <td>0.007782</td>
          <td>22.076111</td>
          <td>0.006423</td>
          <td>21.057290</td>
          <td>0.005243</td>
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
          <td>21.081209</td>
          <td>26.241361</td>
          <td>25.351154</td>
          <td>21.745124</td>
          <td>18.589233</td>
          <td>25.219284</td>
          <td>21.823085</td>
          <td>0.005327</td>
          <td>19.969101</td>
          <td>0.005033</td>
          <td>22.826538</td>
          <td>0.009466</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.603131</td>
          <td>21.007275</td>
          <td>27.348369</td>
          <td>26.080688</td>
          <td>23.528707</td>
          <td>18.298022</td>
          <td>24.460375</td>
          <td>0.021317</td>
          <td>23.551756</td>
          <td>0.016410</td>
          <td>24.352688</td>
          <td>0.032826</td>
        </tr>
        <tr>
          <th>997</th>
          <td>14.782608</td>
          <td>21.245918</td>
          <td>22.141351</td>
          <td>24.513056</td>
          <td>21.733797</td>
          <td>24.074817</td>
          <td>24.537514</td>
          <td>0.022789</td>
          <td>23.752838</td>
          <td>0.019438</td>
          <td>23.850904</td>
          <td>0.021144</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.429385</td>
          <td>21.725518</td>
          <td>21.232133</td>
          <td>24.817561</td>
          <td>23.780090</td>
          <td>28.383280</td>
          <td>18.917950</td>
          <td>0.005002</td>
          <td>23.303149</td>
          <td>0.013414</td>
          <td>23.118837</td>
          <td>0.011640</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.785518</td>
          <td>19.883186</td>
          <td>24.706564</td>
          <td>22.448203</td>
          <td>28.890098</td>
          <td>23.427291</td>
          <td>20.526557</td>
          <td>0.005031</td>
          <td>21.121500</td>
          <td>0.005273</td>
          <td>27.571607</td>
          <td>0.502412</td>
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
          <td>23.757542</td>
          <td>21.922052</td>
          <td>24.974657</td>
          <td>23.809748</td>
          <td>24.778852</td>
          <td>25.591741</td>
          <td>21.224206</td>
          <td>0.013639</td>
          <td>21.175693</td>
          <td>0.011273</td>
          <td>20.955689</td>
          <td>0.010339</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.859909</td>
          <td>21.505056</td>
          <td>28.176642</td>
          <td>17.486255</td>
          <td>26.366688</td>
          <td>21.759347</td>
          <td>25.605196</td>
          <td>0.553832</td>
          <td>21.098986</td>
          <td>0.010662</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.540612</td>
          <td>24.953302</td>
          <td>24.186426</td>
          <td>19.647438</td>
          <td>25.853584</td>
          <td>24.941343</td>
          <td>25.933003</td>
          <td>0.696877</td>
          <td>22.472091</td>
          <td>0.033396</td>
          <td>21.227159</td>
          <td>0.012640</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.718616</td>
          <td>19.948641</td>
          <td>22.331085</td>
          <td>23.982547</td>
          <td>19.495133</td>
          <td>26.876261</td>
          <td>25.098567</td>
          <td>0.378680</td>
          <td>22.940361</td>
          <td>0.050671</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.262325</td>
          <td>24.766528</td>
          <td>20.492319</td>
          <td>21.571059</td>
          <td>22.829105</td>
          <td>23.779701</td>
          <td>23.263611</td>
          <td>0.080676</td>
          <td>22.062566</td>
          <td>0.023291</td>
          <td>21.070978</td>
          <td>0.011234</td>
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
          <td>21.081209</td>
          <td>26.241361</td>
          <td>25.351154</td>
          <td>21.745124</td>
          <td>18.589233</td>
          <td>25.219284</td>
          <td>21.819951</td>
          <td>0.022445</td>
          <td>19.971796</td>
          <td>0.006013</td>
          <td>22.876242</td>
          <td>0.052318</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.603131</td>
          <td>21.007275</td>
          <td>27.348369</td>
          <td>26.080688</td>
          <td>23.528707</td>
          <td>18.298022</td>
          <td>24.480131</td>
          <td>0.230167</td>
          <td>23.531629</td>
          <td>0.085673</td>
          <td>24.231161</td>
          <td>0.171632</td>
        </tr>
        <tr>
          <th>997</th>
          <td>14.782608</td>
          <td>21.245918</td>
          <td>22.141351</td>
          <td>24.513056</td>
          <td>21.733797</td>
          <td>24.074817</td>
          <td>24.554634</td>
          <td>0.244795</td>
          <td>23.852507</td>
          <td>0.113561</td>
          <td>23.671754</td>
          <td>0.105819</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.429385</td>
          <td>21.725518</td>
          <td>21.232133</td>
          <td>24.817561</td>
          <td>23.780090</td>
          <td>28.383280</td>
          <td>18.918967</td>
          <td>0.005227</td>
          <td>23.296379</td>
          <td>0.069563</td>
          <td>23.142934</td>
          <td>0.066338</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.785518</td>
          <td>19.883186</td>
          <td>24.706564</td>
          <td>22.448203</td>
          <td>28.890098</td>
          <td>23.427291</td>
          <td>20.521232</td>
          <td>0.008325</td>
          <td>21.099470</td>
          <td>0.010665</td>
          <td>26.033734</td>
          <td>0.697224</td>
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


