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
          <td>19.702774</td>
          <td>30.370503</td>
          <td>24.061069</td>
          <td>23.929271</td>
          <td>19.092523</td>
          <td>26.318932</td>
          <td>27.510851</td>
          <td>22.352958</td>
          <td>23.213335</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.870895</td>
          <td>25.011521</td>
          <td>23.406309</td>
          <td>24.623601</td>
          <td>22.943607</td>
          <td>25.070375</td>
          <td>21.219384</td>
          <td>21.748864</td>
          <td>21.246151</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.376083</td>
          <td>29.387350</td>
          <td>21.922115</td>
          <td>25.681362</td>
          <td>17.816971</td>
          <td>24.660008</td>
          <td>21.689772</td>
          <td>25.610883</td>
          <td>22.507566</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.160167</td>
          <td>21.125298</td>
          <td>20.204310</td>
          <td>27.005577</td>
          <td>22.027310</td>
          <td>20.879040</td>
          <td>24.429419</td>
          <td>22.443984</td>
          <td>21.568422</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.219562</td>
          <td>19.505957</td>
          <td>23.234059</td>
          <td>27.114757</td>
          <td>23.426770</td>
          <td>20.774425</td>
          <td>24.212093</td>
          <td>25.590589</td>
          <td>20.026443</td>
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
          <td>23.345699</td>
          <td>28.727862</td>
          <td>22.225559</td>
          <td>24.258637</td>
          <td>24.640529</td>
          <td>27.306906</td>
          <td>23.255113</td>
          <td>20.717761</td>
          <td>20.922909</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.307005</td>
          <td>26.816540</td>
          <td>16.805532</td>
          <td>20.051929</td>
          <td>20.032392</td>
          <td>23.387059</td>
          <td>27.094655</td>
          <td>25.100913</td>
          <td>22.363926</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.029308</td>
          <td>28.908141</td>
          <td>21.477551</td>
          <td>24.222872</td>
          <td>28.910025</td>
          <td>24.513340</td>
          <td>21.458866</td>
          <td>21.737977</td>
          <td>23.015674</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.915732</td>
          <td>22.600498</td>
          <td>22.049245</td>
          <td>15.501000</td>
          <td>22.934801</td>
          <td>23.680952</td>
          <td>20.810241</td>
          <td>26.311146</td>
          <td>27.491183</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.936116</td>
          <td>24.519752</td>
          <td>18.810637</td>
          <td>21.391943</td>
          <td>24.879226</td>
          <td>23.029867</td>
          <td>27.757917</td>
          <td>28.279362</td>
          <td>24.853591</td>
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
          <td>19.709884</td>
          <td>0.005191</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.059333</td>
          <td>0.014613</td>
          <td>23.956920</td>
          <td>0.021236</td>
          <td>19.088160</td>
          <td>0.005040</td>
          <td>25.951906</td>
          <td>0.473940</td>
          <td>27.510851</td>
          <td>22.352958</td>
          <td>23.213335</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.860376</td>
          <td>0.005231</td>
          <td>25.090329</td>
          <td>0.040099</td>
          <td>23.402928</td>
          <td>0.009105</td>
          <td>24.659934</td>
          <td>0.039257</td>
          <td>22.949782</td>
          <td>0.016876</td>
          <td>24.886632</td>
          <td>0.202776</td>
          <td>21.219384</td>
          <td>21.748864</td>
          <td>21.246151</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.313112</td>
          <td>0.141924</td>
          <td>29.719653</td>
          <td>1.400523</td>
          <td>21.914581</td>
          <td>0.005414</td>
          <td>25.730785</td>
          <td>0.101251</td>
          <td>17.821522</td>
          <td>0.005008</td>
          <td>24.708958</td>
          <td>0.174532</td>
          <td>21.689772</td>
          <td>25.610883</td>
          <td>22.507566</td>
        </tr>
        <tr>
          <th>3</th>
          <td>inf</td>
          <td>inf</td>
          <td>21.122127</td>
          <td>0.005173</td>
          <td>20.199433</td>
          <td>0.005031</td>
          <td>26.561394</td>
          <td>0.206654</td>
          <td>22.011327</td>
          <td>0.008550</td>
          <td>20.875709</td>
          <td>0.007600</td>
          <td>24.429419</td>
          <td>22.443984</td>
          <td>21.568422</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.212440</td>
          <td>0.005013</td>
          <td>19.513309</td>
          <td>0.005021</td>
          <td>23.248819</td>
          <td>0.008305</td>
          <td>27.627474</td>
          <td>0.482292</td>
          <td>23.457165</td>
          <td>0.025978</td>
          <td>20.778205</td>
          <td>0.007250</td>
          <td>24.212093</td>
          <td>25.590589</td>
          <td>20.026443</td>
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
          <td>23.366833</td>
          <td>0.025979</td>
          <td>28.006430</td>
          <td>0.471349</td>
          <td>22.225485</td>
          <td>0.005684</td>
          <td>24.228500</td>
          <td>0.026852</td>
          <td>24.667458</td>
          <td>0.075709</td>
          <td>26.272373</td>
          <td>0.598336</td>
          <td>23.255113</td>
          <td>20.717761</td>
          <td>20.922909</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.237971</td>
          <td>0.055508</td>
          <td>26.880475</td>
          <td>0.191564</td>
          <td>16.800077</td>
          <td>0.005001</td>
          <td>20.061727</td>
          <td>0.005055</td>
          <td>20.030844</td>
          <td>0.005162</td>
          <td>23.373776</td>
          <td>0.054333</td>
          <td>27.094655</td>
          <td>25.100913</td>
          <td>22.363926</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.044658</td>
          <td>0.046829</td>
          <td>28.863164</td>
          <td>0.854230</td>
          <td>21.467848</td>
          <td>0.005203</td>
          <td>24.236788</td>
          <td>0.027046</td>
          <td>29.833554</td>
          <td>2.430029</td>
          <td>24.610985</td>
          <td>0.160557</td>
          <td>21.458866</td>
          <td>21.737977</td>
          <td>23.015674</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.924978</td>
          <td>0.005251</td>
          <td>22.603458</td>
          <td>0.006642</td>
          <td>22.045676</td>
          <td>0.005512</td>
          <td>15.501224</td>
          <td>0.005000</td>
          <td>22.939170</td>
          <td>0.016730</td>
          <td>23.830689</td>
          <td>0.081429</td>
          <td>20.810241</td>
          <td>26.311146</td>
          <td>27.491183</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.934911</td>
          <td>0.008889</td>
          <td>24.538728</td>
          <td>0.024746</td>
          <td>18.813610</td>
          <td>0.005006</td>
          <td>21.400500</td>
          <td>0.005440</td>
          <td>24.691510</td>
          <td>0.077335</td>
          <td>23.044628</td>
          <td>0.040572</td>
          <td>27.757917</td>
          <td>28.279362</td>
          <td>24.853591</td>
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
          <td>19.702774</td>
          <td>30.370503</td>
          <td>24.061069</td>
          <td>23.929271</td>
          <td>19.092523</td>
          <td>26.318932</td>
          <td>27.385539</td>
          <td>0.272512</td>
          <td>22.359195</td>
          <td>0.007236</td>
          <td>23.223591</td>
          <td>0.012605</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.870895</td>
          <td>25.011521</td>
          <td>23.406309</td>
          <td>24.623601</td>
          <td>22.943607</td>
          <td>25.070375</td>
          <td>21.223140</td>
          <td>0.005111</td>
          <td>21.738286</td>
          <td>0.005808</td>
          <td>21.248410</td>
          <td>0.005342</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.376083</td>
          <td>29.387350</td>
          <td>21.922115</td>
          <td>25.681362</td>
          <td>17.816971</td>
          <td>24.660008</td>
          <td>21.684705</td>
          <td>0.005255</td>
          <td>25.620092</td>
          <td>0.101133</td>
          <td>22.507475</td>
          <td>0.007806</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.160167</td>
          <td>21.125298</td>
          <td>20.204310</td>
          <td>27.005577</td>
          <td>22.027310</td>
          <td>20.879040</td>
          <td>24.411142</td>
          <td>0.020433</td>
          <td>22.442295</td>
          <td>0.007542</td>
          <td>21.567688</td>
          <td>0.005601</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.219562</td>
          <td>19.505957</td>
          <td>23.234059</td>
          <td>27.114757</td>
          <td>23.426770</td>
          <td>20.774425</td>
          <td>24.243202</td>
          <td>0.017713</td>
          <td>25.334524</td>
          <td>0.078626</td>
          <td>20.021964</td>
          <td>0.005037</td>
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
          <td>23.345699</td>
          <td>28.727862</td>
          <td>22.225559</td>
          <td>24.258637</td>
          <td>24.640529</td>
          <td>27.306906</td>
          <td>23.268654</td>
          <td>0.008564</td>
          <td>20.717502</td>
          <td>0.005131</td>
          <td>20.931535</td>
          <td>0.005194</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.307005</td>
          <td>26.816540</td>
          <td>16.805532</td>
          <td>20.051929</td>
          <td>20.032392</td>
          <td>23.387059</td>
          <td>27.365194</td>
          <td>0.268030</td>
          <td>25.115713</td>
          <td>0.064752</td>
          <td>22.357184</td>
          <td>0.007229</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.029308</td>
          <td>28.908141</td>
          <td>21.477551</td>
          <td>24.222872</td>
          <td>28.910025</td>
          <td>24.513340</td>
          <td>21.453600</td>
          <td>0.005168</td>
          <td>21.727727</td>
          <td>0.005793</td>
          <td>23.019710</td>
          <td>0.010822</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.915732</td>
          <td>22.600498</td>
          <td>22.049245</td>
          <td>15.501000</td>
          <td>22.934801</td>
          <td>23.680952</td>
          <td>20.815443</td>
          <td>0.005053</td>
          <td>26.535548</td>
          <td>0.221794</td>
          <td>27.547765</td>
          <td>0.493641</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.936116</td>
          <td>24.519752</td>
          <td>18.810637</td>
          <td>21.391943</td>
          <td>24.879226</td>
          <td>23.029867</td>
          <td>27.637148</td>
          <td>0.333597</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.856451</td>
          <td>0.051403</td>
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
          <td>19.702774</td>
          <td>30.370503</td>
          <td>24.061069</td>
          <td>23.929271</td>
          <td>19.092523</td>
          <td>26.318932</td>
          <td>26.795283</td>
          <td>1.189663</td>
          <td>22.345356</td>
          <td>0.029850</td>
          <td>23.193683</td>
          <td>0.069397</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.870895</td>
          <td>25.011521</td>
          <td>23.406309</td>
          <td>24.623601</td>
          <td>22.943607</td>
          <td>25.070375</td>
          <td>21.206937</td>
          <td>0.013454</td>
          <td>21.747570</td>
          <td>0.017779</td>
          <td>21.281621</td>
          <td>0.013188</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.376083</td>
          <td>29.387350</td>
          <td>21.922115</td>
          <td>25.681362</td>
          <td>17.816971</td>
          <td>24.660008</td>
          <td>21.683174</td>
          <td>0.019949</td>
          <td>26.028532</td>
          <td>0.648643</td>
          <td>22.501247</td>
          <td>0.037457</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.160167</td>
          <td>21.125298</td>
          <td>20.204310</td>
          <td>27.005577</td>
          <td>22.027310</td>
          <td>20.879040</td>
          <td>24.827768</td>
          <td>0.305722</td>
          <td>22.482617</td>
          <td>0.033709</td>
          <td>21.577890</td>
          <td>0.016770</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.219562</td>
          <td>19.505957</td>
          <td>23.234059</td>
          <td>27.114757</td>
          <td>23.426770</td>
          <td>20.774425</td>
          <td>24.346472</td>
          <td>0.205884</td>
          <td>25.937396</td>
          <td>0.608598</td>
          <td>20.026255</td>
          <td>0.006311</td>
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
          <td>23.345699</td>
          <td>28.727862</td>
          <td>22.225559</td>
          <td>24.258637</td>
          <td>24.640529</td>
          <td>27.306906</td>
          <td>23.302947</td>
          <td>0.083530</td>
          <td>20.720226</td>
          <td>0.008320</td>
          <td>20.919841</td>
          <td>0.010084</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.307005</td>
          <td>26.816540</td>
          <td>16.805532</td>
          <td>20.051929</td>
          <td>20.032392</td>
          <td>23.387059</td>
          <td>26.957570</td>
          <td>1.300350</td>
          <td>25.490811</td>
          <td>0.438899</td>
          <td>22.400864</td>
          <td>0.034259</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.029308</td>
          <td>28.908141</td>
          <td>21.477551</td>
          <td>24.222872</td>
          <td>28.910025</td>
          <td>24.513340</td>
          <td>21.447225</td>
          <td>0.016348</td>
          <td>21.738748</td>
          <td>0.017647</td>
          <td>23.009849</td>
          <td>0.058931</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.915732</td>
          <td>22.600498</td>
          <td>22.049245</td>
          <td>15.501000</td>
          <td>22.934801</td>
          <td>23.680952</td>
          <td>20.812729</td>
          <td>0.010034</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.287210</td>
          <td>5.149289</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.936116</td>
          <td>24.519752</td>
          <td>18.810637</td>
          <td>21.391943</td>
          <td>24.879226</td>
          <td>23.029867</td>
          <td>25.782288</td>
          <td>0.628088</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.289814</td>
          <td>0.180402</td>
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


