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
          <td>25.736835</td>
          <td>19.917407</td>
          <td>20.986057</td>
          <td>26.910705</td>
          <td>18.677322</td>
          <td>20.935704</td>
          <td>31.119239</td>
          <td>25.302931</td>
          <td>22.434593</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.800972</td>
          <td>22.945456</td>
          <td>24.293907</td>
          <td>30.253762</td>
          <td>30.233437</td>
          <td>27.744562</td>
          <td>20.974163</td>
          <td>19.455174</td>
          <td>23.643818</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.574491</td>
          <td>19.110623</td>
          <td>19.323640</td>
          <td>20.625240</td>
          <td>17.940197</td>
          <td>22.390481</td>
          <td>16.478662</td>
          <td>27.425476</td>
          <td>24.965375</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.179016</td>
          <td>21.741618</td>
          <td>27.903255</td>
          <td>24.607075</td>
          <td>20.873944</td>
          <td>26.510423</td>
          <td>23.766059</td>
          <td>26.120283</td>
          <td>25.021864</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.646491</td>
          <td>23.930229</td>
          <td>28.636598</td>
          <td>25.516423</td>
          <td>22.028819</td>
          <td>19.960216</td>
          <td>20.518998</td>
          <td>22.198773</td>
          <td>23.252226</td>
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
          <td>19.035185</td>
          <td>18.266150</td>
          <td>24.620085</td>
          <td>21.012311</td>
          <td>23.465605</td>
          <td>19.670594</td>
          <td>22.811588</td>
          <td>22.608770</td>
          <td>24.764849</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.566236</td>
          <td>25.417824</td>
          <td>20.975074</td>
          <td>21.125867</td>
          <td>25.367561</td>
          <td>23.338445</td>
          <td>19.601345</td>
          <td>25.636688</td>
          <td>23.857578</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.868406</td>
          <td>25.958466</td>
          <td>19.280671</td>
          <td>23.715656</td>
          <td>26.487186</td>
          <td>22.698258</td>
          <td>21.965477</td>
          <td>19.240812</td>
          <td>20.198224</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.268297</td>
          <td>24.395181</td>
          <td>24.792173</td>
          <td>24.884775</td>
          <td>20.867344</td>
          <td>26.494393</td>
          <td>20.779516</td>
          <td>22.625476</td>
          <td>24.043867</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.912694</td>
          <td>25.545388</td>
          <td>23.825023</td>
          <td>16.876869</td>
          <td>24.723914</td>
          <td>21.222395</td>
          <td>22.147799</td>
          <td>23.698177</td>
          <td>22.585012</td>
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
          <td>25.991984</td>
          <td>0.251184</td>
          <td>19.912057</td>
          <td>0.005034</td>
          <td>20.985996</td>
          <td>0.005096</td>
          <td>26.992229</td>
          <td>0.294539</td>
          <td>18.680419</td>
          <td>0.005023</td>
          <td>20.932661</td>
          <td>0.007826</td>
          <td>31.119239</td>
          <td>25.302931</td>
          <td>22.434593</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.803296</td>
          <td>0.005215</td>
          <td>22.943437</td>
          <td>0.007706</td>
          <td>24.308631</td>
          <td>0.017911</td>
          <td>28.720941</td>
          <td>1.008104</td>
          <td>27.436415</td>
          <td>0.708822</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.974163</td>
          <td>19.455174</td>
          <td>23.643818</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.574162</td>
          <td>0.005051</td>
          <td>19.109874</td>
          <td>0.005013</td>
          <td>19.332410</td>
          <td>0.005010</td>
          <td>20.629557</td>
          <td>0.005129</td>
          <td>17.949082</td>
          <td>0.005009</td>
          <td>22.369817</td>
          <td>0.022466</td>
          <td>16.478662</td>
          <td>27.425476</td>
          <td>24.965375</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.601610</td>
          <td>0.407785</td>
          <td>21.740056</td>
          <td>0.005438</td>
          <td>28.397922</td>
          <td>0.565106</td>
          <td>24.659582</td>
          <td>0.039245</td>
          <td>20.869715</td>
          <td>0.005616</td>
          <td>27.231526</td>
          <td>1.109369</td>
          <td>23.766059</td>
          <td>26.120283</td>
          <td>25.021864</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.644760</td>
          <td>0.005658</td>
          <td>23.958304</td>
          <td>0.015238</td>
          <td>30.730222</td>
          <td>2.089154</td>
          <td>25.562339</td>
          <td>0.087329</td>
          <td>22.024780</td>
          <td>0.008618</td>
          <td>19.957796</td>
          <td>0.005624</td>
          <td>20.518998</td>
          <td>22.198773</td>
          <td>23.252226</td>
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
          <td>19.035526</td>
          <td>0.005086</td>
          <td>18.264683</td>
          <td>0.005006</td>
          <td>24.606140</td>
          <td>0.023055</td>
          <td>21.004842</td>
          <td>0.005233</td>
          <td>23.478318</td>
          <td>0.026461</td>
          <td>19.672415</td>
          <td>0.005394</td>
          <td>22.811588</td>
          <td>22.608770</td>
          <td>24.764849</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.567254</td>
          <td>0.005591</td>
          <td>25.406726</td>
          <td>0.053058</td>
          <td>20.976406</td>
          <td>0.005094</td>
          <td>21.121009</td>
          <td>0.005281</td>
          <td>25.668520</td>
          <td>0.180602</td>
          <td>23.371582</td>
          <td>0.054228</td>
          <td>19.601345</td>
          <td>25.636688</td>
          <td>23.857578</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.029183</td>
          <td>1.076714</td>
          <td>25.961131</td>
          <td>0.086593</td>
          <td>19.283635</td>
          <td>0.005010</td>
          <td>23.679402</td>
          <td>0.016820</td>
          <td>26.110212</td>
          <td>0.260946</td>
          <td>22.656721</td>
          <td>0.028819</td>
          <td>21.965477</td>
          <td>19.240812</td>
          <td>20.198224</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.268985</td>
          <td>0.005014</td>
          <td>24.358987</td>
          <td>0.021218</td>
          <td>24.835471</td>
          <td>0.028137</td>
          <td>24.901382</td>
          <td>0.048635</td>
          <td>20.864239</td>
          <td>0.005610</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.779516</td>
          <td>22.625476</td>
          <td>24.043867</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.945494</td>
          <td>0.018267</td>
          <td>25.545173</td>
          <td>0.059979</td>
          <td>23.827681</td>
          <td>0.012213</td>
          <td>16.881183</td>
          <td>0.005001</td>
          <td>24.722461</td>
          <td>0.079477</td>
          <td>21.223867</td>
          <td>0.009268</td>
          <td>22.147799</td>
          <td>23.698177</td>
          <td>22.585012</td>
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
          <td>25.736835</td>
          <td>19.917407</td>
          <td>20.986057</td>
          <td>26.910705</td>
          <td>18.677322</td>
          <td>20.935704</td>
          <td>28.846079</td>
          <td>0.803027</td>
          <td>25.283402</td>
          <td>0.075145</td>
          <td>22.430657</td>
          <td>0.007497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.800972</td>
          <td>22.945456</td>
          <td>24.293907</td>
          <td>30.253762</td>
          <td>30.233437</td>
          <td>27.744562</td>
          <td>20.982686</td>
          <td>0.005071</td>
          <td>19.451623</td>
          <td>0.005013</td>
          <td>23.656268</td>
          <td>0.017909</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.574491</td>
          <td>19.110623</td>
          <td>19.323640</td>
          <td>20.625240</td>
          <td>17.940197</td>
          <td>22.390481</td>
          <td>16.482510</td>
          <td>0.005000</td>
          <td>27.873834</td>
          <td>0.624383</td>
          <td>24.891969</td>
          <td>0.053057</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.179016</td>
          <td>21.741618</td>
          <td>27.903255</td>
          <td>24.607075</td>
          <td>20.873944</td>
          <td>26.510423</td>
          <td>23.760884</td>
          <td>0.012015</td>
          <td>26.237434</td>
          <td>0.172551</td>
          <td>25.030711</td>
          <td>0.060036</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.646491</td>
          <td>23.930229</td>
          <td>28.636598</td>
          <td>25.516423</td>
          <td>22.028819</td>
          <td>19.960216</td>
          <td>20.515143</td>
          <td>0.005030</td>
          <td>22.195601</td>
          <td>0.006727</td>
          <td>23.242702</td>
          <td>0.012793</td>
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
          <td>19.035185</td>
          <td>18.266150</td>
          <td>24.620085</td>
          <td>21.012311</td>
          <td>23.465605</td>
          <td>19.670594</td>
          <td>22.814845</td>
          <td>0.006781</td>
          <td>22.609203</td>
          <td>0.008266</td>
          <td>24.708620</td>
          <td>0.045056</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.566236</td>
          <td>25.417824</td>
          <td>20.975074</td>
          <td>21.125867</td>
          <td>25.367561</td>
          <td>23.338445</td>
          <td>19.598350</td>
          <td>0.005006</td>
          <td>25.560430</td>
          <td>0.095970</td>
          <td>23.856962</td>
          <td>0.021255</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.868406</td>
          <td>25.958466</td>
          <td>19.280671</td>
          <td>23.715656</td>
          <td>26.487186</td>
          <td>22.698258</td>
          <td>21.960276</td>
          <td>0.005418</td>
          <td>19.240205</td>
          <td>0.005009</td>
          <td>20.183609</td>
          <td>0.005050</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.268297</td>
          <td>24.395181</td>
          <td>24.792173</td>
          <td>24.884775</td>
          <td>20.867344</td>
          <td>26.494393</td>
          <td>20.782272</td>
          <td>0.005049</td>
          <td>22.624376</td>
          <td>0.008340</td>
          <td>24.058023</td>
          <td>0.025314</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.912694</td>
          <td>25.545388</td>
          <td>23.825023</td>
          <td>16.876869</td>
          <td>24.723914</td>
          <td>21.222395</td>
          <td>22.147292</td>
          <td>0.005580</td>
          <td>23.683405</td>
          <td>0.018325</td>
          <td>22.569834</td>
          <td>0.008081</td>
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
          <td>25.736835</td>
          <td>19.917407</td>
          <td>20.986057</td>
          <td>26.910705</td>
          <td>18.677322</td>
          <td>20.935704</td>
          <td>27.420619</td>
          <td>1.643175</td>
          <td>25.319487</td>
          <td>0.384881</td>
          <td>22.453752</td>
          <td>0.035907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.800972</td>
          <td>22.945456</td>
          <td>24.293907</td>
          <td>30.253762</td>
          <td>30.233437</td>
          <td>27.744562</td>
          <td>20.992812</td>
          <td>0.011417</td>
          <td>19.452646</td>
          <td>0.005412</td>
          <td>23.618952</td>
          <td>0.101032</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.574491</td>
          <td>19.110623</td>
          <td>19.323640</td>
          <td>20.625240</td>
          <td>17.940197</td>
          <td>22.390481</td>
          <td>16.480917</td>
          <td>0.005003</td>
          <td>27.288761</td>
          <td>1.393578</td>
          <td>24.431628</td>
          <td>0.203335</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.179016</td>
          <td>21.741618</td>
          <td>27.903255</td>
          <td>24.607075</td>
          <td>20.873944</td>
          <td>26.510423</td>
          <td>23.755547</td>
          <td>0.124224</td>
          <td>26.970743</td>
          <td>1.173388</td>
          <td>24.889571</td>
          <td>0.296474</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.646491</td>
          <td>23.930229</td>
          <td>28.636598</td>
          <td>25.516423</td>
          <td>22.028819</td>
          <td>19.960216</td>
          <td>20.505031</td>
          <td>0.008246</td>
          <td>22.187131</td>
          <td>0.025968</td>
          <td>23.322688</td>
          <td>0.077806</td>
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
          <td>19.035185</td>
          <td>18.266150</td>
          <td>24.620085</td>
          <td>21.012311</td>
          <td>23.465605</td>
          <td>19.670594</td>
          <td>22.856346</td>
          <td>0.056189</td>
          <td>22.595368</td>
          <td>0.037261</td>
          <td>24.799488</td>
          <td>0.275623</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.566236</td>
          <td>25.417824</td>
          <td>20.975074</td>
          <td>21.125867</td>
          <td>25.367561</td>
          <td>23.338445</td>
          <td>19.609277</td>
          <td>0.005768</td>
          <td>25.097959</td>
          <td>0.323369</td>
          <td>23.842090</td>
          <td>0.122780</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.868406</td>
          <td>25.958466</td>
          <td>19.280671</td>
          <td>23.715656</td>
          <td>26.487186</td>
          <td>22.698258</td>
          <td>21.931201</td>
          <td>0.024727</td>
          <td>19.246198</td>
          <td>0.005285</td>
          <td>20.197409</td>
          <td>0.006732</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.268297</td>
          <td>24.395181</td>
          <td>24.792173</td>
          <td>24.884775</td>
          <td>20.867344</td>
          <td>26.494393</td>
          <td>20.803397</td>
          <td>0.009970</td>
          <td>22.653706</td>
          <td>0.039247</td>
          <td>23.908141</td>
          <td>0.130027</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.912694</td>
          <td>25.545388</td>
          <td>23.825023</td>
          <td>16.876869</td>
          <td>24.723914</td>
          <td>21.222395</td>
          <td>22.139694</td>
          <td>0.029701</td>
          <td>23.669639</td>
          <td>0.096750</td>
          <td>22.602177</td>
          <td>0.040978</td>
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


