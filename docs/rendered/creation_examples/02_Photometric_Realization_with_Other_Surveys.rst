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
          <td>21.937235</td>
          <td>17.094073</td>
          <td>25.846850</td>
          <td>23.150071</td>
          <td>27.309253</td>
          <td>25.121548</td>
          <td>21.938860</td>
          <td>26.081838</td>
          <td>21.587799</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.616848</td>
          <td>26.393153</td>
          <td>23.423353</td>
          <td>24.045128</td>
          <td>24.418128</td>
          <td>20.026445</td>
          <td>21.762743</td>
          <td>21.892966</td>
          <td>21.534308</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.963827</td>
          <td>23.727334</td>
          <td>22.786926</td>
          <td>21.177085</td>
          <td>25.885302</td>
          <td>28.178484</td>
          <td>15.355662</td>
          <td>25.282891</td>
          <td>23.848205</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.699611</td>
          <td>25.860087</td>
          <td>18.688641</td>
          <td>22.168238</td>
          <td>18.909632</td>
          <td>23.988360</td>
          <td>21.427460</td>
          <td>16.705149</td>
          <td>24.153181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.858382</td>
          <td>21.127579</td>
          <td>19.124769</td>
          <td>15.027143</td>
          <td>24.412863</td>
          <td>26.845610</td>
          <td>24.037554</td>
          <td>23.459878</td>
          <td>20.623803</td>
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
          <td>23.401236</td>
          <td>21.548130</td>
          <td>18.306742</td>
          <td>16.080918</td>
          <td>18.560196</td>
          <td>26.606813</td>
          <td>24.939282</td>
          <td>22.751961</td>
          <td>20.617528</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.446817</td>
          <td>23.112479</td>
          <td>21.706670</td>
          <td>25.898376</td>
          <td>23.974834</td>
          <td>20.426450</td>
          <td>21.324096</td>
          <td>21.289120</td>
          <td>22.659221</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.199822</td>
          <td>24.773676</td>
          <td>23.364239</td>
          <td>24.755236</td>
          <td>26.180620</td>
          <td>19.353352</td>
          <td>28.548563</td>
          <td>22.143152</td>
          <td>23.791952</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574903</td>
          <td>22.870877</td>
          <td>25.698804</td>
          <td>26.875151</td>
          <td>23.339615</td>
          <td>25.518013</td>
          <td>21.824349</td>
          <td>23.430189</td>
          <td>27.248871</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.515534</td>
          <td>23.229912</td>
          <td>28.315188</td>
          <td>27.240017</td>
          <td>19.752263</td>
          <td>26.428380</td>
          <td>22.203450</td>
          <td>12.460464</td>
          <td>24.527815</td>
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
          <td>21.934435</td>
          <td>0.008887</td>
          <td>17.101349</td>
          <td>0.005002</td>
          <td>25.771384</td>
          <td>0.064407</td>
          <td>23.160186</td>
          <td>0.011228</td>
          <td>28.457452</td>
          <td>1.315329</td>
          <td>24.956560</td>
          <td>0.214993</td>
          <td>21.938860</td>
          <td>26.081838</td>
          <td>21.587799</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.606803</td>
          <td>0.013988</td>
          <td>26.518483</td>
          <td>0.140714</td>
          <td>23.413301</td>
          <td>0.009164</td>
          <td>24.045797</td>
          <td>0.022918</td>
          <td>24.328568</td>
          <td>0.056073</td>
          <td>20.025003</td>
          <td>0.005695</td>
          <td>21.762743</td>
          <td>21.892966</td>
          <td>21.534308</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.006384</td>
          <td>0.108876</td>
          <td>23.705549</td>
          <td>0.012526</td>
          <td>22.791181</td>
          <td>0.006674</td>
          <td>21.184041</td>
          <td>0.005310</td>
          <td>25.712552</td>
          <td>0.187455</td>
          <td>inf</td>
          <td>inf</td>
          <td>15.355662</td>
          <td>25.282891</td>
          <td>23.848205</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.797823</td>
          <td>0.473021</td>
          <td>25.796376</td>
          <td>0.074895</td>
          <td>18.682825</td>
          <td>0.005005</td>
          <td>22.159013</td>
          <td>0.006480</td>
          <td>18.912068</td>
          <td>0.005031</td>
          <td>24.012597</td>
          <td>0.095563</td>
          <td>21.427460</td>
          <td>16.705149</td>
          <td>24.153181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.862991</td>
          <td>0.005232</td>
          <td>21.132512</td>
          <td>0.005176</td>
          <td>19.124082</td>
          <td>0.005008</td>
          <td>15.031220</td>
          <td>0.005000</td>
          <td>24.510889</td>
          <td>0.065914</td>
          <td>26.721334</td>
          <td>0.811478</td>
          <td>24.037554</td>
          <td>23.459878</td>
          <td>20.623803</td>
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
          <td>23.411929</td>
          <td>0.027001</td>
          <td>21.540847</td>
          <td>0.005323</td>
          <td>18.306897</td>
          <td>0.005003</td>
          <td>16.080475</td>
          <td>0.005001</td>
          <td>18.553640</td>
          <td>0.005020</td>
          <td>26.527907</td>
          <td>0.713981</td>
          <td>24.939282</td>
          <td>22.751961</td>
          <td>20.617528</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.583427</td>
          <td>0.402133</td>
          <td>23.115183</td>
          <td>0.008451</td>
          <td>21.707993</td>
          <td>0.005297</td>
          <td>26.044185</td>
          <td>0.133007</td>
          <td>23.949806</td>
          <td>0.040068</td>
          <td>20.425863</td>
          <td>0.006312</td>
          <td>21.324096</td>
          <td>21.289120</td>
          <td>22.659221</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.027305</td>
          <td>0.110874</td>
          <td>24.761256</td>
          <td>0.030023</td>
          <td>23.368021</td>
          <td>0.008910</td>
          <td>24.668597</td>
          <td>0.039560</td>
          <td>25.884520</td>
          <td>0.216561</td>
          <td>19.348571</td>
          <td>0.005235</td>
          <td>28.548563</td>
          <td>22.143152</td>
          <td>23.791952</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.575031</td>
          <td>0.007400</td>
          <td>22.874314</td>
          <td>0.007449</td>
          <td>25.698705</td>
          <td>0.060387</td>
          <td>26.621788</td>
          <td>0.217351</td>
          <td>23.337682</td>
          <td>0.023424</td>
          <td>25.006076</td>
          <td>0.224045</td>
          <td>21.824349</td>
          <td>23.430189</td>
          <td>27.248871</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.549791</td>
          <td>0.030402</td>
          <td>23.226398</td>
          <td>0.009024</td>
          <td>29.667439</td>
          <td>1.262205</td>
          <td>28.957453</td>
          <td>1.157194</td>
          <td>19.756515</td>
          <td>0.005106</td>
          <td>25.582019</td>
          <td>0.357039</td>
          <td>22.203450</td>
          <td>12.460464</td>
          <td>24.527815</td>
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
          <td>21.937235</td>
          <td>17.094073</td>
          <td>25.846850</td>
          <td>23.150071</td>
          <td>27.309253</td>
          <td>25.121548</td>
          <td>21.937127</td>
          <td>0.005401</td>
          <td>25.950888</td>
          <td>0.134930</td>
          <td>21.586812</td>
          <td>0.005622</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.616848</td>
          <td>26.393153</td>
          <td>23.423353</td>
          <td>24.045128</td>
          <td>24.418128</td>
          <td>20.026445</td>
          <td>21.764778</td>
          <td>0.005295</td>
          <td>21.887232</td>
          <td>0.006040</td>
          <td>21.536406</td>
          <td>0.005569</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.963827</td>
          <td>23.727334</td>
          <td>22.786926</td>
          <td>21.177085</td>
          <td>25.885302</td>
          <td>28.178484</td>
          <td>15.353461</td>
          <td>0.005000</td>
          <td>25.309875</td>
          <td>0.076929</td>
          <td>23.862095</td>
          <td>0.021349</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.699611</td>
          <td>25.860087</td>
          <td>18.688641</td>
          <td>22.168238</td>
          <td>18.909632</td>
          <td>23.988360</td>
          <td>21.415644</td>
          <td>0.005157</td>
          <td>16.703569</td>
          <td>0.005000</td>
          <td>24.137973</td>
          <td>0.027154</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.858382</td>
          <td>21.127579</td>
          <td>19.124769</td>
          <td>15.027143</td>
          <td>24.412863</td>
          <td>26.845610</td>
          <td>24.039157</td>
          <td>0.014959</td>
          <td>23.491886</td>
          <td>0.015618</td>
          <td>20.613830</td>
          <td>0.005109</td>
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
          <td>23.401236</td>
          <td>21.548130</td>
          <td>18.306742</td>
          <td>16.080918</td>
          <td>18.560196</td>
          <td>26.606813</td>
          <td>24.937192</td>
          <td>0.032378</td>
          <td>22.759255</td>
          <td>0.009060</td>
          <td>20.614093</td>
          <td>0.005109</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.446817</td>
          <td>23.112479</td>
          <td>21.706670</td>
          <td>25.898376</td>
          <td>23.974834</td>
          <td>20.426450</td>
          <td>21.319982</td>
          <td>0.005132</td>
          <td>21.283300</td>
          <td>0.005364</td>
          <td>22.664965</td>
          <td>0.008545</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.199822</td>
          <td>24.773676</td>
          <td>23.364239</td>
          <td>24.755236</td>
          <td>26.180620</td>
          <td>19.353352</td>
          <td>28.513548</td>
          <td>0.641929</td>
          <td>22.130163</td>
          <td>0.006554</td>
          <td>23.797554</td>
          <td>0.020196</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574903</td>
          <td>22.870877</td>
          <td>25.698804</td>
          <td>26.875151</td>
          <td>23.339615</td>
          <td>25.518013</td>
          <td>21.819721</td>
          <td>0.005325</td>
          <td>23.452954</td>
          <td>0.015128</td>
          <td>27.592625</td>
          <td>0.510245</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.515534</td>
          <td>23.229912</td>
          <td>28.315188</td>
          <td>27.240017</td>
          <td>19.752263</td>
          <td>26.428380</td>
          <td>22.207639</td>
          <td>0.005645</td>
          <td>12.462791</td>
          <td>0.005000</td>
          <td>24.535467</td>
          <td>0.038615</td>
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
          <td>21.937235</td>
          <td>17.094073</td>
          <td>25.846850</td>
          <td>23.150071</td>
          <td>27.309253</td>
          <td>25.121548</td>
          <td>21.932650</td>
          <td>0.024759</td>
          <td>28.194479</td>
          <td>2.114182</td>
          <td>21.611183</td>
          <td>0.017243</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.616848</td>
          <td>26.393153</td>
          <td>23.423353</td>
          <td>24.045128</td>
          <td>24.418128</td>
          <td>20.026445</td>
          <td>21.759825</td>
          <td>0.021307</td>
          <td>21.881620</td>
          <td>0.019922</td>
          <td>21.538223</td>
          <td>0.016226</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.963827</td>
          <td>23.727334</td>
          <td>22.786926</td>
          <td>21.177085</td>
          <td>25.885302</td>
          <td>28.178484</td>
          <td>15.354077</td>
          <td>0.005000</td>
          <td>25.668315</td>
          <td>0.501194</td>
          <td>23.954033</td>
          <td>0.135297</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.699611</td>
          <td>25.860087</td>
          <td>18.688641</td>
          <td>22.168238</td>
          <td>18.909632</td>
          <td>23.988360</td>
          <td>21.430156</td>
          <td>0.016118</td>
          <td>16.706684</td>
          <td>0.005003</td>
          <td>24.025227</td>
          <td>0.143874</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.858382</td>
          <td>21.127579</td>
          <td>19.124769</td>
          <td>15.027143</td>
          <td>24.412863</td>
          <td>26.845610</td>
          <td>24.149121</td>
          <td>0.174275</td>
          <td>23.461194</td>
          <td>0.080503</td>
          <td>20.623316</td>
          <td>0.008335</td>
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
          <td>23.401236</td>
          <td>21.548130</td>
          <td>18.306742</td>
          <td>16.080918</td>
          <td>18.560196</td>
          <td>26.606813</td>
          <td>24.592318</td>
          <td>0.252505</td>
          <td>22.741610</td>
          <td>0.042444</td>
          <td>20.618672</td>
          <td>0.008312</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.446817</td>
          <td>23.112479</td>
          <td>21.706670</td>
          <td>25.898376</td>
          <td>23.974834</td>
          <td>20.426450</td>
          <td>21.340720</td>
          <td>0.014978</td>
          <td>21.297019</td>
          <td>0.012350</td>
          <td>22.637227</td>
          <td>0.042278</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.199822</td>
          <td>24.773676</td>
          <td>23.364239</td>
          <td>24.755236</td>
          <td>26.180620</td>
          <td>19.353352</td>
          <td>26.240878</td>
          <td>0.853609</td>
          <td>22.188717</td>
          <td>0.026005</td>
          <td>23.798552</td>
          <td>0.118214</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574903</td>
          <td>22.870877</td>
          <td>25.698804</td>
          <td>26.875151</td>
          <td>23.339615</td>
          <td>25.518013</td>
          <td>21.838213</td>
          <td>0.022803</td>
          <td>23.536066</td>
          <td>0.086010</td>
          <td>26.320621</td>
          <td>0.842627</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.515534</td>
          <td>23.229912</td>
          <td>28.315188</td>
          <td>27.240017</td>
          <td>19.752263</td>
          <td>26.428380</td>
          <td>22.214465</td>
          <td>0.031732</td>
          <td>12.464678</td>
          <td>0.005000</td>
          <td>24.451189</td>
          <td>0.206700</td>
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


