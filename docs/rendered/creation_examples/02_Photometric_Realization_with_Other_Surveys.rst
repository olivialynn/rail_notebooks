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
          <td>21.760198</td>
          <td>23.642715</td>
          <td>19.186263</td>
          <td>18.242736</td>
          <td>19.624346</td>
          <td>22.492626</td>
          <td>28.052765</td>
          <td>20.438205</td>
          <td>23.561601</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.232934</td>
          <td>30.250664</td>
          <td>25.755244</td>
          <td>21.517991</td>
          <td>23.757247</td>
          <td>24.175190</td>
          <td>23.867250</td>
          <td>20.039597</td>
          <td>26.590164</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.437900</td>
          <td>20.618988</td>
          <td>23.565110</td>
          <td>25.787691</td>
          <td>23.904834</td>
          <td>25.093160</td>
          <td>23.541233</td>
          <td>23.035449</td>
          <td>23.247262</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.590695</td>
          <td>24.160024</td>
          <td>23.017718</td>
          <td>24.481619</td>
          <td>18.441371</td>
          <td>19.799997</td>
          <td>20.594189</td>
          <td>23.659628</td>
          <td>25.583926</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.018935</td>
          <td>25.860243</td>
          <td>24.285466</td>
          <td>21.691552</td>
          <td>22.434355</td>
          <td>22.786679</td>
          <td>24.316636</td>
          <td>22.190709</td>
          <td>20.943416</td>
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
          <td>26.733297</td>
          <td>17.923574</td>
          <td>26.220492</td>
          <td>24.201427</td>
          <td>25.417217</td>
          <td>27.007341</td>
          <td>23.610520</td>
          <td>19.320905</td>
          <td>17.351237</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.976488</td>
          <td>24.514021</td>
          <td>20.216850</td>
          <td>24.992255</td>
          <td>23.984426</td>
          <td>22.422652</td>
          <td>24.992914</td>
          <td>25.227597</td>
          <td>24.566468</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.280974</td>
          <td>26.330806</td>
          <td>27.926608</td>
          <td>19.774508</td>
          <td>22.813523</td>
          <td>21.262489</td>
          <td>20.874494</td>
          <td>20.778436</td>
          <td>21.519516</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.014423</td>
          <td>21.425930</td>
          <td>22.331156</td>
          <td>15.954789</td>
          <td>28.223129</td>
          <td>20.253224</td>
          <td>22.575903</td>
          <td>29.074841</td>
          <td>20.768745</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.330558</td>
          <td>27.068073</td>
          <td>22.942143</td>
          <td>20.258910</td>
          <td>24.559723</td>
          <td>22.751647</td>
          <td>21.058490</td>
          <td>24.897714</td>
          <td>24.993618</td>
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
          <td>21.761844</td>
          <td>0.008092</td>
          <td>23.621965</td>
          <td>0.011776</td>
          <td>19.184911</td>
          <td>0.005009</td>
          <td>18.244764</td>
          <td>0.005005</td>
          <td>19.623717</td>
          <td>0.005087</td>
          <td>22.471615</td>
          <td>0.024527</td>
          <td>28.052765</td>
          <td>20.438205</td>
          <td>23.561601</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.233313</td>
          <td>0.010704</td>
          <td>28.261879</td>
          <td>0.568255</td>
          <td>25.774537</td>
          <td>0.064588</td>
          <td>21.509774</td>
          <td>0.005525</td>
          <td>23.733775</td>
          <td>0.033103</td>
          <td>24.024999</td>
          <td>0.096609</td>
          <td>23.867250</td>
          <td>20.039597</td>
          <td>26.590164</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.425510</td>
          <td>0.012233</td>
          <td>20.623160</td>
          <td>0.005086</td>
          <td>23.547844</td>
          <td>0.010003</td>
          <td>25.842389</td>
          <td>0.111625</td>
          <td>23.886803</td>
          <td>0.037894</td>
          <td>25.155977</td>
          <td>0.253573</td>
          <td>23.541233</td>
          <td>23.035449</td>
          <td>23.247262</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.593418</td>
          <td>0.005019</td>
          <td>24.168425</td>
          <td>0.018082</td>
          <td>23.032037</td>
          <td>0.007409</td>
          <td>24.421858</td>
          <td>0.031811</td>
          <td>18.442076</td>
          <td>0.005017</td>
          <td>19.812295</td>
          <td>0.005494</td>
          <td>20.594189</td>
          <td>23.659628</td>
          <td>25.583926</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.004862</td>
          <td>0.009261</td>
          <td>25.714083</td>
          <td>0.069645</td>
          <td>24.287206</td>
          <td>0.017595</td>
          <td>21.683401</td>
          <td>0.005695</td>
          <td>22.426911</td>
          <td>0.011236</td>
          <td>22.762659</td>
          <td>0.031627</td>
          <td>24.316636</td>
          <td>22.190709</td>
          <td>20.943416</td>
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
          <td>26.277958</td>
          <td>0.316562</td>
          <td>17.928974</td>
          <td>0.005004</td>
          <td>26.137540</td>
          <td>0.089008</td>
          <td>24.170434</td>
          <td>0.025528</td>
          <td>25.396534</td>
          <td>0.143162</td>
          <td>25.832996</td>
          <td>0.433370</td>
          <td>23.610520</td>
          <td>19.320905</td>
          <td>17.351237</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>24.524689</td>
          <td>0.024448</td>
          <td>20.213735</td>
          <td>0.005031</td>
          <td>24.889354</td>
          <td>0.048118</td>
          <td>24.026993</td>
          <td>0.042905</td>
          <td>22.404159</td>
          <td>0.023139</td>
          <td>24.992914</td>
          <td>25.227597</td>
          <td>24.566468</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.344552</td>
          <td>1.285417</td>
          <td>26.207322</td>
          <td>0.107439</td>
          <td>28.105122</td>
          <td>0.455669</td>
          <td>19.782228</td>
          <td>0.005037</td>
          <td>22.802270</td>
          <td>0.014969</td>
          <td>21.261539</td>
          <td>0.009494</td>
          <td>20.874494</td>
          <td>20.778436</td>
          <td>21.519516</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.008481</td>
          <td>0.019233</td>
          <td>21.421252</td>
          <td>0.005270</td>
          <td>22.336324</td>
          <td>0.005818</td>
          <td>15.943104</td>
          <td>0.005001</td>
          <td>27.113438</td>
          <td>0.565908</td>
          <td>20.260478</td>
          <td>0.006012</td>
          <td>22.575903</td>
          <td>29.074841</td>
          <td>20.768745</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.327898</td>
          <td>0.005427</td>
          <td>27.359029</td>
          <td>0.284540</td>
          <td>22.949396</td>
          <td>0.007129</td>
          <td>20.262061</td>
          <td>0.005074</td>
          <td>24.569982</td>
          <td>0.069456</td>
          <td>22.719154</td>
          <td>0.030440</td>
          <td>21.058490</td>
          <td>24.897714</td>
          <td>24.993618</td>
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
          <td>21.760198</td>
          <td>23.642715</td>
          <td>19.186263</td>
          <td>18.242736</td>
          <td>19.624346</td>
          <td>22.492626</td>
          <td>28.348356</td>
          <td>0.571283</td>
          <td>20.438661</td>
          <td>0.005079</td>
          <td>23.593054</td>
          <td>0.016984</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.232934</td>
          <td>30.250664</td>
          <td>25.755244</td>
          <td>21.517991</td>
          <td>23.757247</td>
          <td>24.175190</td>
          <td>23.857428</td>
          <td>0.012941</td>
          <td>20.033089</td>
          <td>0.005038</td>
          <td>26.497407</td>
          <td>0.214850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.437900</td>
          <td>20.618988</td>
          <td>23.565110</td>
          <td>25.787691</td>
          <td>23.904834</td>
          <td>25.093160</td>
          <td>23.532521</td>
          <td>0.010173</td>
          <td>23.037386</td>
          <td>0.010961</td>
          <td>23.238501</td>
          <td>0.012752</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.590695</td>
          <td>24.160024</td>
          <td>23.017718</td>
          <td>24.481619</td>
          <td>18.441371</td>
          <td>19.799997</td>
          <td>20.597075</td>
          <td>0.005035</td>
          <td>23.650620</td>
          <td>0.017824</td>
          <td>25.591589</td>
          <td>0.098634</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.018935</td>
          <td>25.860243</td>
          <td>24.285466</td>
          <td>21.691552</td>
          <td>22.434355</td>
          <td>22.786679</td>
          <td>24.309028</td>
          <td>0.018727</td>
          <td>22.182575</td>
          <td>0.006691</td>
          <td>20.941248</td>
          <td>0.005197</td>
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
          <td>26.733297</td>
          <td>17.923574</td>
          <td>26.220492</td>
          <td>24.201427</td>
          <td>25.417217</td>
          <td>27.007341</td>
          <td>23.617916</td>
          <td>0.010808</td>
          <td>19.318512</td>
          <td>0.005010</td>
          <td>17.352566</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.976488</td>
          <td>24.514021</td>
          <td>20.216850</td>
          <td>24.992255</td>
          <td>23.984426</td>
          <td>22.422652</td>
          <td>24.963756</td>
          <td>0.033150</td>
          <td>25.244208</td>
          <td>0.072579</td>
          <td>24.568653</td>
          <td>0.039773</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.280974</td>
          <td>26.330806</td>
          <td>27.926608</td>
          <td>19.774508</td>
          <td>22.813523</td>
          <td>21.262489</td>
          <td>20.867548</td>
          <td>0.005058</td>
          <td>20.771004</td>
          <td>0.005145</td>
          <td>21.524527</td>
          <td>0.005558</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.014423</td>
          <td>21.425930</td>
          <td>22.331156</td>
          <td>15.954789</td>
          <td>28.223129</td>
          <td>20.253224</td>
          <td>22.568516</td>
          <td>0.006191</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.767891</td>
          <td>0.005144</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.330558</td>
          <td>27.068073</td>
          <td>22.942143</td>
          <td>20.258910</td>
          <td>24.559723</td>
          <td>22.751647</td>
          <td>21.073037</td>
          <td>0.005084</td>
          <td>24.834702</td>
          <td>0.050416</td>
          <td>24.956192</td>
          <td>0.056181</td>
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
          <td>21.760198</td>
          <td>23.642715</td>
          <td>19.186263</td>
          <td>18.242736</td>
          <td>19.624346</td>
          <td>22.492626</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.445234</td>
          <td>0.007188</td>
          <td>23.405873</td>
          <td>0.083747</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.232934</td>
          <td>30.250664</td>
          <td>25.755244</td>
          <td>21.517991</td>
          <td>23.757247</td>
          <td>24.175190</td>
          <td>24.035129</td>
          <td>0.158120</td>
          <td>20.039470</td>
          <td>0.006135</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.437900</td>
          <td>20.618988</td>
          <td>23.565110</td>
          <td>25.787691</td>
          <td>23.904834</td>
          <td>25.093160</td>
          <td>23.487565</td>
          <td>0.098286</td>
          <td>22.999240</td>
          <td>0.053402</td>
          <td>23.296068</td>
          <td>0.075993</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.590695</td>
          <td>24.160024</td>
          <td>23.017718</td>
          <td>24.481619</td>
          <td>18.441371</td>
          <td>19.799997</td>
          <td>20.604719</td>
          <td>0.008755</td>
          <td>23.498286</td>
          <td>0.083187</td>
          <td>25.473841</td>
          <td>0.467216</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.018935</td>
          <td>25.860243</td>
          <td>24.285466</td>
          <td>21.691552</td>
          <td>22.434355</td>
          <td>22.786679</td>
          <td>24.295585</td>
          <td>0.197267</td>
          <td>22.193983</td>
          <td>0.026125</td>
          <td>20.947071</td>
          <td>0.010277</td>
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
          <td>26.733297</td>
          <td>17.923574</td>
          <td>26.220492</td>
          <td>24.201427</td>
          <td>25.417217</td>
          <td>27.007341</td>
          <td>23.582275</td>
          <td>0.106798</td>
          <td>19.312648</td>
          <td>0.005321</td>
          <td>17.342028</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.976488</td>
          <td>24.514021</td>
          <td>20.216850</td>
          <td>24.992255</td>
          <td>23.984426</td>
          <td>22.422652</td>
          <td>24.886283</td>
          <td>0.320374</td>
          <td>24.727291</td>
          <td>0.239332</td>
          <td>24.470157</td>
          <td>0.210010</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.280974</td>
          <td>26.330806</td>
          <td>27.926608</td>
          <td>19.774508</td>
          <td>22.813523</td>
          <td>21.262489</td>
          <td>20.878139</td>
          <td>0.010504</td>
          <td>20.782665</td>
          <td>0.008637</td>
          <td>21.530810</td>
          <td>0.016127</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.014423</td>
          <td>21.425930</td>
          <td>22.331156</td>
          <td>15.954789</td>
          <td>28.223129</td>
          <td>20.253224</td>
          <td>22.574853</td>
          <td>0.043720</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.775844</td>
          <td>0.009157</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.330558</td>
          <td>27.068073</td>
          <td>22.942143</td>
          <td>20.258910</td>
          <td>24.559723</td>
          <td>22.751647</td>
          <td>21.051309</td>
          <td>0.011928</td>
          <td>24.754791</td>
          <td>0.244826</td>
          <td>24.657864</td>
          <td>0.245447</td>
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


