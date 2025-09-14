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
          <td>26.898195</td>
          <td>26.273514</td>
          <td>22.600331</td>
          <td>21.206029</td>
          <td>23.436843</td>
          <td>22.123731</td>
          <td>20.269266</td>
          <td>24.333749</td>
          <td>21.697071</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.080755</td>
          <td>21.068687</td>
          <td>29.171182</td>
          <td>30.386141</td>
          <td>25.449230</td>
          <td>22.891427</td>
          <td>20.112287</td>
          <td>18.703972</td>
          <td>25.092185</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.829268</td>
          <td>19.624197</td>
          <td>21.160491</td>
          <td>22.017059</td>
          <td>25.315810</td>
          <td>23.428736</td>
          <td>26.396395</td>
          <td>25.972861</td>
          <td>22.297769</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.638562</td>
          <td>19.865237</td>
          <td>17.447230</td>
          <td>26.695968</td>
          <td>20.961536</td>
          <td>17.645332</td>
          <td>24.601519</td>
          <td>22.135249</td>
          <td>19.096081</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.772442</td>
          <td>28.427140</td>
          <td>25.255164</td>
          <td>27.073398</td>
          <td>24.813250</td>
          <td>25.039420</td>
          <td>20.559921</td>
          <td>23.300691</td>
          <td>22.658688</td>
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
          <td>29.786404</td>
          <td>22.848267</td>
          <td>25.585893</td>
          <td>21.760396</td>
          <td>24.018676</td>
          <td>20.875625</td>
          <td>18.776122</td>
          <td>24.328754</td>
          <td>17.929261</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558703</td>
          <td>17.475258</td>
          <td>23.986565</td>
          <td>27.582082</td>
          <td>23.222228</td>
          <td>20.866050</td>
          <td>20.648753</td>
          <td>24.854527</td>
          <td>19.811267</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.297964</td>
          <td>20.541944</td>
          <td>17.046203</td>
          <td>25.964654</td>
          <td>21.286188</td>
          <td>19.459101</td>
          <td>24.283832</td>
          <td>17.444254</td>
          <td>19.495940</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.134913</td>
          <td>20.436796</td>
          <td>23.450380</td>
          <td>22.424678</td>
          <td>25.914341</td>
          <td>24.812831</td>
          <td>18.798726</td>
          <td>23.617958</td>
          <td>25.084235</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.197721</td>
          <td>26.919086</td>
          <td>23.442614</td>
          <td>28.245718</td>
          <td>26.946701</td>
          <td>23.372449</td>
          <td>20.668828</td>
          <td>24.198475</td>
          <td>22.251362</td>
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
          <td>26.608998</td>
          <td>0.410100</td>
          <td>26.177547</td>
          <td>0.104682</td>
          <td>22.593440</td>
          <td>0.006230</td>
          <td>21.199972</td>
          <td>0.005319</td>
          <td>23.400419</td>
          <td>0.024729</td>
          <td>22.122117</td>
          <td>0.018210</td>
          <td>20.269266</td>
          <td>24.333749</td>
          <td>21.697071</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.083947</td>
          <td>0.006214</td>
          <td>21.070102</td>
          <td>0.005161</td>
          <td>28.870565</td>
          <td>0.782125</td>
          <td>28.754462</td>
          <td>1.028500</td>
          <td>25.568188</td>
          <td>0.165842</td>
          <td>22.941056</td>
          <td>0.037017</td>
          <td>20.112287</td>
          <td>18.703972</td>
          <td>25.092185</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.828614</td>
          <td>0.005849</td>
          <td>19.635456</td>
          <td>0.005024</td>
          <td>21.163237</td>
          <td>0.005126</td>
          <td>22.029720</td>
          <td>0.006209</td>
          <td>25.340983</td>
          <td>0.136468</td>
          <td>23.451615</td>
          <td>0.058219</td>
          <td>26.396395</td>
          <td>25.972861</td>
          <td>22.297769</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.649656</td>
          <td>0.007657</td>
          <td>19.858536</td>
          <td>0.005032</td>
          <td>17.452520</td>
          <td>0.005001</td>
          <td>26.767095</td>
          <td>0.245164</td>
          <td>20.961731</td>
          <td>0.005714</td>
          <td>17.638074</td>
          <td>0.005020</td>
          <td>24.601519</td>
          <td>22.135249</td>
          <td>19.096081</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.771436</td>
          <td>0.088697</td>
          <td>28.298916</td>
          <td>0.583493</td>
          <td>25.332995</td>
          <td>0.043647</td>
          <td>27.317545</td>
          <td>0.381051</td>
          <td>24.663987</td>
          <td>0.075477</td>
          <td>24.743656</td>
          <td>0.179746</td>
          <td>20.559921</td>
          <td>23.300691</td>
          <td>22.658688</td>
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
          <td>28.191999</td>
          <td>1.181905</td>
          <td>22.853027</td>
          <td>0.007375</td>
          <td>25.651724</td>
          <td>0.057922</td>
          <td>21.760731</td>
          <td>0.005788</td>
          <td>24.035834</td>
          <td>0.043243</td>
          <td>20.891229</td>
          <td>0.007660</td>
          <td>18.776122</td>
          <td>24.328754</td>
          <td>17.929261</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.535892</td>
          <td>0.072141</td>
          <td>17.481845</td>
          <td>0.005003</td>
          <td>23.980195</td>
          <td>0.013728</td>
          <td>27.798342</td>
          <td>0.546688</td>
          <td>23.233229</td>
          <td>0.021415</td>
          <td>20.852479</td>
          <td>0.007512</td>
          <td>20.648753</td>
          <td>24.854527</td>
          <td>19.811267</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.283812</td>
          <td>0.057793</td>
          <td>20.540820</td>
          <td>0.005077</td>
          <td>17.043864</td>
          <td>0.005001</td>
          <td>26.245101</td>
          <td>0.158097</td>
          <td>21.291095</td>
          <td>0.006205</td>
          <td>19.455687</td>
          <td>0.005279</td>
          <td>24.283832</td>
          <td>17.444254</td>
          <td>19.495940</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.152216</td>
          <td>0.021667</td>
          <td>20.437149</td>
          <td>0.005067</td>
          <td>23.460166</td>
          <td>0.009442</td>
          <td>22.424890</td>
          <td>0.007222</td>
          <td>25.633292</td>
          <td>0.175287</td>
          <td>24.753516</td>
          <td>0.181253</td>
          <td>18.798726</td>
          <td>23.617958</td>
          <td>25.084235</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.194237</td>
          <td>0.006416</td>
          <td>26.811498</td>
          <td>0.180724</td>
          <td>23.441333</td>
          <td>0.009329</td>
          <td>28.055262</td>
          <td>0.655664</td>
          <td>26.212108</td>
          <td>0.283510</td>
          <td>23.275694</td>
          <td>0.049803</td>
          <td>20.668828</td>
          <td>24.198475</td>
          <td>22.251362</td>
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
          <td>26.898195</td>
          <td>26.273514</td>
          <td>22.600331</td>
          <td>21.206029</td>
          <td>23.436843</td>
          <td>22.123731</td>
          <td>20.263707</td>
          <td>0.005019</td>
          <td>24.349076</td>
          <td>0.032721</td>
          <td>21.690944</td>
          <td>0.005744</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.080755</td>
          <td>21.068687</td>
          <td>29.171182</td>
          <td>30.386141</td>
          <td>25.449230</td>
          <td>22.891427</td>
          <td>20.108890</td>
          <td>0.005014</td>
          <td>18.698334</td>
          <td>0.005003</td>
          <td>25.192387</td>
          <td>0.069317</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.829268</td>
          <td>19.624197</td>
          <td>21.160491</td>
          <td>22.017059</td>
          <td>25.315810</td>
          <td>23.428736</td>
          <td>26.358252</td>
          <td>0.114132</td>
          <td>26.359020</td>
          <td>0.191280</td>
          <td>22.292956</td>
          <td>0.007016</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.638562</td>
          <td>19.865237</td>
          <td>17.447230</td>
          <td>26.695968</td>
          <td>20.961536</td>
          <td>17.645332</td>
          <td>24.576516</td>
          <td>0.023575</td>
          <td>22.138196</td>
          <td>0.006574</td>
          <td>19.105627</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.772442</td>
          <td>28.427140</td>
          <td>25.255164</td>
          <td>27.073398</td>
          <td>24.813250</td>
          <td>25.039420</td>
          <td>20.560392</td>
          <td>0.005033</td>
          <td>23.294420</td>
          <td>0.013322</td>
          <td>22.656198</td>
          <td>0.008500</td>
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
          <td>29.786404</td>
          <td>22.848267</td>
          <td>25.585893</td>
          <td>21.760396</td>
          <td>24.018676</td>
          <td>20.875625</td>
          <td>18.776721</td>
          <td>0.005001</td>
          <td>24.328809</td>
          <td>0.032138</td>
          <td>17.931922</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558703</td>
          <td>17.475258</td>
          <td>23.986565</td>
          <td>27.582082</td>
          <td>23.222228</td>
          <td>20.866050</td>
          <td>20.659013</td>
          <td>0.005039</td>
          <td>24.819424</td>
          <td>0.049734</td>
          <td>19.805553</td>
          <td>0.005025</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.297964</td>
          <td>20.541944</td>
          <td>17.046203</td>
          <td>25.964654</td>
          <td>21.286188</td>
          <td>19.459101</td>
          <td>24.252175</td>
          <td>0.017848</td>
          <td>17.442335</td>
          <td>0.005000</td>
          <td>19.506097</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.134913</td>
          <td>20.436796</td>
          <td>23.450380</td>
          <td>22.424678</td>
          <td>25.914341</td>
          <td>24.812831</td>
          <td>18.797639</td>
          <td>0.005001</td>
          <td>23.643971</td>
          <td>0.017725</td>
          <td>25.107417</td>
          <td>0.064276</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.197721</td>
          <td>26.919086</td>
          <td>23.442614</td>
          <td>28.245718</td>
          <td>26.946701</td>
          <td>23.372449</td>
          <td>20.677930</td>
          <td>0.005041</td>
          <td>24.201585</td>
          <td>0.028719</td>
          <td>22.251263</td>
          <td>0.006887</td>
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
          <td>26.898195</td>
          <td>26.273514</td>
          <td>22.600331</td>
          <td>21.206029</td>
          <td>23.436843</td>
          <td>22.123731</td>
          <td>20.262497</td>
          <td>0.007248</td>
          <td>24.530287</td>
          <td>0.203107</td>
          <td>21.695296</td>
          <td>0.018510</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.080755</td>
          <td>21.068687</td>
          <td>29.171182</td>
          <td>30.386141</td>
          <td>25.449230</td>
          <td>22.891427</td>
          <td>20.123462</td>
          <td>0.006806</td>
          <td>18.711822</td>
          <td>0.005108</td>
          <td>24.967474</td>
          <td>0.315600</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.829268</td>
          <td>19.624197</td>
          <td>21.160491</td>
          <td>22.017059</td>
          <td>25.315810</td>
          <td>23.428736</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.156364</td>
          <td>0.708005</td>
          <td>22.337092</td>
          <td>0.032375</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.638562</td>
          <td>19.865237</td>
          <td>17.447230</td>
          <td>26.695968</td>
          <td>20.961536</td>
          <td>17.645332</td>
          <td>24.434048</td>
          <td>0.221517</td>
          <td>22.141204</td>
          <td>0.024945</td>
          <td>19.100397</td>
          <td>0.005263</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.772442</td>
          <td>28.427140</td>
          <td>25.255164</td>
          <td>27.073398</td>
          <td>24.813250</td>
          <td>25.039420</td>
          <td>20.567328</td>
          <td>0.008557</td>
          <td>23.257131</td>
          <td>0.067180</td>
          <td>22.701569</td>
          <td>0.044774</td>
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
          <td>29.786404</td>
          <td>22.848267</td>
          <td>25.585893</td>
          <td>21.760396</td>
          <td>24.018676</td>
          <td>20.875625</td>
          <td>18.774237</td>
          <td>0.005175</td>
          <td>24.236796</td>
          <td>0.158346</td>
          <td>17.926498</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558703</td>
          <td>17.475258</td>
          <td>23.986565</td>
          <td>27.582082</td>
          <td>23.222228</td>
          <td>20.866050</td>
          <td>20.645057</td>
          <td>0.008979</td>
          <td>24.655292</td>
          <td>0.225468</td>
          <td>19.810224</td>
          <td>0.005913</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.297964</td>
          <td>20.541944</td>
          <td>17.046203</td>
          <td>25.964654</td>
          <td>21.286188</td>
          <td>19.459101</td>
          <td>24.220251</td>
          <td>0.185114</td>
          <td>17.443934</td>
          <td>0.005011</td>
          <td>19.497106</td>
          <td>0.005532</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.134913</td>
          <td>20.436796</td>
          <td>23.450380</td>
          <td>22.424678</td>
          <td>25.914341</td>
          <td>24.812831</td>
          <td>18.801412</td>
          <td>0.005184</td>
          <td>23.586174</td>
          <td>0.089896</td>
          <td>24.535841</td>
          <td>0.221848</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.197721</td>
          <td>26.919086</td>
          <td>23.442614</td>
          <td>28.245718</td>
          <td>26.946701</td>
          <td>23.372449</td>
          <td>20.657252</td>
          <td>0.009049</td>
          <td>24.252881</td>
          <td>0.160541</td>
          <td>22.293936</td>
          <td>0.031161</td>
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


