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
          <td>28.165087</td>
          <td>19.682399</td>
          <td>27.429155</td>
          <td>22.853937</td>
          <td>22.232981</td>
          <td>21.891000</td>
          <td>15.149110</td>
          <td>23.786432</td>
          <td>21.538564</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.503102</td>
          <td>23.220179</td>
          <td>25.316469</td>
          <td>22.951187</td>
          <td>21.233494</td>
          <td>19.336199</td>
          <td>21.371645</td>
          <td>27.368928</td>
          <td>20.935410</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.393685</td>
          <td>24.494829</td>
          <td>28.417101</td>
          <td>26.014209</td>
          <td>19.023294</td>
          <td>24.762336</td>
          <td>22.437831</td>
          <td>21.004987</td>
          <td>21.917794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866021</td>
          <td>26.001750</td>
          <td>26.725884</td>
          <td>21.485779</td>
          <td>20.537018</td>
          <td>20.708087</td>
          <td>31.933935</td>
          <td>23.476947</td>
          <td>22.968494</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.224888</td>
          <td>26.561472</td>
          <td>23.901706</td>
          <td>20.384650</td>
          <td>19.975021</td>
          <td>25.051449</td>
          <td>27.309610</td>
          <td>23.661250</td>
          <td>29.409544</td>
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
          <td>24.942362</td>
          <td>22.259956</td>
          <td>20.550556</td>
          <td>23.662899</td>
          <td>19.657615</td>
          <td>19.568463</td>
          <td>22.764763</td>
          <td>21.464093</td>
          <td>21.723649</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.682825</td>
          <td>19.892478</td>
          <td>17.872080</td>
          <td>19.761312</td>
          <td>25.882433</td>
          <td>22.527636</td>
          <td>24.300416</td>
          <td>22.433428</td>
          <td>20.891118</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.134842</td>
          <td>26.656063</td>
          <td>23.501101</td>
          <td>27.862554</td>
          <td>25.950961</td>
          <td>26.805479</td>
          <td>22.176528</td>
          <td>23.470519</td>
          <td>23.470700</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.810034</td>
          <td>26.143891</td>
          <td>22.423155</td>
          <td>20.429836</td>
          <td>23.893208</td>
          <td>22.132392</td>
          <td>30.045722</td>
          <td>25.035145</td>
          <td>24.045173</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.377870</td>
          <td>21.166412</td>
          <td>25.571469</td>
          <td>21.210386</td>
          <td>22.860989</td>
          <td>23.315176</td>
          <td>23.049976</td>
          <td>25.266238</td>
          <td>22.495109</td>
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
          <td>27.739233</td>
          <td>0.903613</td>
          <td>19.683975</td>
          <td>0.005026</td>
          <td>27.411887</td>
          <td>0.264207</td>
          <td>22.873518</td>
          <td>0.009241</td>
          <td>22.221941</td>
          <td>0.009751</td>
          <td>21.898461</td>
          <td>0.015157</td>
          <td>15.149110</td>
          <td>23.786432</td>
          <td>21.538564</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.470967</td>
          <td>0.028405</td>
          <td>23.228244</td>
          <td>0.009034</td>
          <td>25.338014</td>
          <td>0.043842</td>
          <td>22.946988</td>
          <td>0.009691</td>
          <td>21.237812</td>
          <td>0.006108</td>
          <td>19.336015</td>
          <td>0.005231</td>
          <td>21.371645</td>
          <td>27.368928</td>
          <td>20.935410</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.400672</td>
          <td>0.026742</td>
          <td>24.509903</td>
          <td>0.024139</td>
          <td>27.901868</td>
          <td>0.390232</td>
          <td>25.812600</td>
          <td>0.108760</td>
          <td>19.023196</td>
          <td>0.005037</td>
          <td>24.610651</td>
          <td>0.160511</td>
          <td>22.437831</td>
          <td>21.004987</td>
          <td>21.917794</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.858179</td>
          <td>0.017023</td>
          <td>26.116697</td>
          <td>0.099257</td>
          <td>26.804408</td>
          <td>0.158873</td>
          <td>21.481188</td>
          <td>0.005502</td>
          <td>20.545863</td>
          <td>0.005366</td>
          <td>20.716812</td>
          <td>0.007052</td>
          <td>31.933935</td>
          <td>23.476947</td>
          <td>22.968494</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.532075</td>
          <td>0.171089</td>
          <td>26.503299</td>
          <td>0.138886</td>
          <td>23.909899</td>
          <td>0.013000</td>
          <td>20.378451</td>
          <td>0.005088</td>
          <td>19.976523</td>
          <td>0.005149</td>
          <td>25.212550</td>
          <td>0.265589</td>
          <td>27.309610</td>
          <td>23.661250</td>
          <td>29.409544</td>
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
          <td>24.824036</td>
          <td>0.092873</td>
          <td>22.249760</td>
          <td>0.005959</td>
          <td>20.553328</td>
          <td>0.005051</td>
          <td>23.647775</td>
          <td>0.016388</td>
          <td>19.656718</td>
          <td>0.005091</td>
          <td>19.564449</td>
          <td>0.005332</td>
          <td>22.764763</td>
          <td>21.464093</td>
          <td>21.723649</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.678981</td>
          <td>0.007765</td>
          <td>19.886662</td>
          <td>0.005033</td>
          <td>17.868083</td>
          <td>0.005002</td>
          <td>19.763769</td>
          <td>0.005036</td>
          <td>25.792016</td>
          <td>0.200429</td>
          <td>22.511707</td>
          <td>0.025395</td>
          <td>24.300416</td>
          <td>22.433428</td>
          <td>20.891118</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.134327</td>
          <td>0.005330</td>
          <td>26.678693</td>
          <td>0.161431</td>
          <td>23.492359</td>
          <td>0.009642</td>
          <td>27.981838</td>
          <td>0.623002</td>
          <td>26.341901</td>
          <td>0.314714</td>
          <td>26.021766</td>
          <td>0.499157</td>
          <td>22.176528</td>
          <td>23.470519</td>
          <td>23.470700</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.814096</td>
          <td>0.008315</td>
          <td>26.241117</td>
          <td>0.110652</td>
          <td>22.418394</td>
          <td>0.005932</td>
          <td>20.426189</td>
          <td>0.005094</td>
          <td>23.904206</td>
          <td>0.038482</td>
          <td>22.151008</td>
          <td>0.018656</td>
          <td>30.045722</td>
          <td>25.035145</td>
          <td>24.045173</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.378191</td>
          <td>0.005457</td>
          <td>21.166158</td>
          <td>0.005185</td>
          <td>25.505274</td>
          <td>0.050860</td>
          <td>21.210745</td>
          <td>0.005324</td>
          <td>22.867216</td>
          <td>0.015774</td>
          <td>23.300895</td>
          <td>0.050929</td>
          <td>23.049976</td>
          <td>25.266238</td>
          <td>22.495109</td>
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
          <td>28.165087</td>
          <td>19.682399</td>
          <td>27.429155</td>
          <td>22.853937</td>
          <td>22.232981</td>
          <td>21.891000</td>
          <td>15.155917</td>
          <td>0.005000</td>
          <td>23.757210</td>
          <td>0.019511</td>
          <td>21.528211</td>
          <td>0.005561</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.503102</td>
          <td>23.220179</td>
          <td>25.316469</td>
          <td>22.951187</td>
          <td>21.233494</td>
          <td>19.336199</td>
          <td>21.365199</td>
          <td>0.005143</td>
          <td>27.079615</td>
          <td>0.344992</td>
          <td>20.934738</td>
          <td>0.005195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.393685</td>
          <td>24.494829</td>
          <td>28.417101</td>
          <td>26.014209</td>
          <td>19.023294</td>
          <td>24.762336</td>
          <td>22.428124</td>
          <td>0.005941</td>
          <td>21.002940</td>
          <td>0.005220</td>
          <td>21.920981</td>
          <td>0.006101</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866021</td>
          <td>26.001750</td>
          <td>26.725884</td>
          <td>21.485779</td>
          <td>20.537018</td>
          <td>20.708087</td>
          <td>28.086005</td>
          <td>0.471485</td>
          <td>23.476666</td>
          <td>0.015424</td>
          <td>22.966544</td>
          <td>0.010419</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.224888</td>
          <td>26.561472</td>
          <td>23.901706</td>
          <td>20.384650</td>
          <td>19.975021</td>
          <td>25.051449</td>
          <td>27.684238</td>
          <td>0.346253</td>
          <td>23.639187</td>
          <td>0.017654</td>
          <td>30.439752</td>
          <td>2.416418</td>
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
          <td>24.942362</td>
          <td>22.259956</td>
          <td>20.550556</td>
          <td>23.662899</td>
          <td>19.657615</td>
          <td>19.568463</td>
          <td>22.755707</td>
          <td>0.006619</td>
          <td>21.473329</td>
          <td>0.005510</td>
          <td>21.721582</td>
          <td>0.005785</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.682825</td>
          <td>19.892478</td>
          <td>17.872080</td>
          <td>19.761312</td>
          <td>25.882433</td>
          <td>22.527636</td>
          <td>24.294176</td>
          <td>0.018492</td>
          <td>22.431768</td>
          <td>0.007501</td>
          <td>20.891389</td>
          <td>0.005180</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.134842</td>
          <td>26.656063</td>
          <td>23.501101</td>
          <td>27.862554</td>
          <td>25.950961</td>
          <td>26.805479</td>
          <td>22.185184</td>
          <td>0.005620</td>
          <td>23.489925</td>
          <td>0.015593</td>
          <td>23.490473</td>
          <td>0.015600</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.810034</td>
          <td>26.143891</td>
          <td>22.423155</td>
          <td>20.429836</td>
          <td>23.893208</td>
          <td>22.132392</td>
          <td>28.393964</td>
          <td>0.590174</td>
          <td>25.050345</td>
          <td>0.061094</td>
          <td>24.012649</td>
          <td>0.024330</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.377870</td>
          <td>21.166412</td>
          <td>25.571469</td>
          <td>21.210386</td>
          <td>22.860989</td>
          <td>23.315176</td>
          <td>23.050214</td>
          <td>0.007573</td>
          <td>25.204603</td>
          <td>0.070073</td>
          <td>22.491228</td>
          <td>0.007738</td>
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
          <td>28.165087</td>
          <td>19.682399</td>
          <td>27.429155</td>
          <td>22.853937</td>
          <td>22.232981</td>
          <td>21.891000</td>
          <td>15.148880</td>
          <td>0.005000</td>
          <td>23.617027</td>
          <td>0.092373</td>
          <td>21.532836</td>
          <td>0.016154</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.503102</td>
          <td>23.220179</td>
          <td>25.316469</td>
          <td>22.951187</td>
          <td>21.233494</td>
          <td>19.336199</td>
          <td>21.384280</td>
          <td>0.015521</td>
          <td>29.538616</td>
          <td>3.342356</td>
          <td>20.940065</td>
          <td>0.010227</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.393685</td>
          <td>24.494829</td>
          <td>28.417101</td>
          <td>26.014209</td>
          <td>19.023294</td>
          <td>24.762336</td>
          <td>22.475616</td>
          <td>0.040020</td>
          <td>21.004120</td>
          <td>0.009975</td>
          <td>21.938980</td>
          <td>0.022818</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866021</td>
          <td>26.001750</td>
          <td>26.725884</td>
          <td>21.485779</td>
          <td>20.537018</td>
          <td>20.708087</td>
          <td>26.423468</td>
          <td>0.956800</td>
          <td>23.563798</td>
          <td>0.088140</td>
          <td>22.949644</td>
          <td>0.055855</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.224888</td>
          <td>26.561472</td>
          <td>23.901706</td>
          <td>20.384650</td>
          <td>19.975021</td>
          <td>25.051449</td>
          <td>26.711940</td>
          <td>1.134899</td>
          <td>23.672385</td>
          <td>0.096983</td>
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
          <td>24.942362</td>
          <td>22.259956</td>
          <td>20.550556</td>
          <td>23.662899</td>
          <td>19.657615</td>
          <td>19.568463</td>
          <td>22.778810</td>
          <td>0.052438</td>
          <td>21.487533</td>
          <td>0.014347</td>
          <td>21.720715</td>
          <td>0.018914</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.682825</td>
          <td>19.892478</td>
          <td>17.872080</td>
          <td>19.761312</td>
          <td>25.882433</td>
          <td>22.527636</td>
          <td>24.939512</td>
          <td>0.334223</td>
          <td>22.404032</td>
          <td>0.031441</td>
          <td>20.903607</td>
          <td>0.009971</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.134842</td>
          <td>26.656063</td>
          <td>23.501101</td>
          <td>27.862554</td>
          <td>25.950961</td>
          <td>26.805479</td>
          <td>22.246303</td>
          <td>0.032640</td>
          <td>23.678972</td>
          <td>0.097547</td>
          <td>23.534367</td>
          <td>0.093794</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.810034</td>
          <td>26.143891</td>
          <td>22.423155</td>
          <td>20.429836</td>
          <td>23.893208</td>
          <td>22.132392</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.341009</td>
          <td>0.391349</td>
          <td>24.251093</td>
          <td>0.174567</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.377870</td>
          <td>21.166412</td>
          <td>25.571469</td>
          <td>21.210386</td>
          <td>22.860989</td>
          <td>23.315176</td>
          <td>22.984223</td>
          <td>0.062964</td>
          <td>25.030252</td>
          <td>0.306332</td>
          <td>22.553303</td>
          <td>0.039233</td>
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


