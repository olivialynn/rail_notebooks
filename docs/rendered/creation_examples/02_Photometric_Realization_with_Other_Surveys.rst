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
          <td>18.725428</td>
          <td>24.399663</td>
          <td>25.311801</td>
          <td>28.769341</td>
          <td>23.591156</td>
          <td>17.780518</td>
          <td>28.900498</td>
          <td>18.136830</td>
          <td>20.270760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.508699</td>
          <td>23.960244</td>
          <td>22.289428</td>
          <td>23.576174</td>
          <td>28.922254</td>
          <td>22.612407</td>
          <td>20.462057</td>
          <td>23.538415</td>
          <td>24.536914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.529464</td>
          <td>24.458368</td>
          <td>26.214194</td>
          <td>20.523772</td>
          <td>23.895809</td>
          <td>23.467580</td>
          <td>22.961779</td>
          <td>19.089753</td>
          <td>21.038654</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.229162</td>
          <td>18.949050</td>
          <td>19.752717</td>
          <td>25.950062</td>
          <td>24.674750</td>
          <td>29.502978</td>
          <td>30.390020</td>
          <td>25.184450</td>
          <td>18.886763</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.293251</td>
          <td>22.282948</td>
          <td>27.467756</td>
          <td>25.173156</td>
          <td>24.313506</td>
          <td>21.131268</td>
          <td>28.969342</td>
          <td>22.735897</td>
          <td>26.416098</td>
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
          <td>18.972000</td>
          <td>22.299579</td>
          <td>27.820251</td>
          <td>18.402108</td>
          <td>26.793546</td>
          <td>20.382886</td>
          <td>27.240830</td>
          <td>27.207820</td>
          <td>26.004181</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.079836</td>
          <td>19.043932</td>
          <td>22.047356</td>
          <td>26.084351</td>
          <td>30.372787</td>
          <td>24.430258</td>
          <td>22.861169</td>
          <td>24.726348</td>
          <td>27.264110</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.510525</td>
          <td>21.543432</td>
          <td>23.045952</td>
          <td>20.159242</td>
          <td>19.318659</td>
          <td>24.027095</td>
          <td>19.588628</td>
          <td>25.996481</td>
          <td>25.488711</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.564690</td>
          <td>24.317696</td>
          <td>18.973691</td>
          <td>20.667544</td>
          <td>23.878697</td>
          <td>20.198904</td>
          <td>25.093206</td>
          <td>26.267653</td>
          <td>22.948807</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.349120</td>
          <td>19.789898</td>
          <td>22.496389</td>
          <td>26.533412</td>
          <td>24.115390</td>
          <td>22.412482</td>
          <td>27.574233</td>
          <td>20.559816</td>
          <td>26.215968</td>
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
          <td>18.728026</td>
          <td>0.005061</td>
          <td>24.416147</td>
          <td>0.022275</td>
          <td>25.338173</td>
          <td>0.043848</td>
          <td>28.472375</td>
          <td>0.864674</td>
          <td>23.595827</td>
          <td>0.029323</td>
          <td>17.789392</td>
          <td>0.005024</td>
          <td>28.900498</td>
          <td>18.136830</td>
          <td>20.270760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.917223</td>
          <td>0.236201</td>
          <td>23.946265</td>
          <td>0.015092</td>
          <td>22.292205</td>
          <td>0.005762</td>
          <td>23.590265</td>
          <td>0.015638</td>
          <td>29.694647</td>
          <td>2.306816</td>
          <td>22.703511</td>
          <td>0.030025</td>
          <td>20.462057</td>
          <td>23.538415</td>
          <td>24.536914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.248512</td>
          <td>0.134256</td>
          <td>24.473559</td>
          <td>0.023397</td>
          <td>26.147364</td>
          <td>0.089781</td>
          <td>20.524977</td>
          <td>0.005110</td>
          <td>23.892241</td>
          <td>0.038076</td>
          <td>23.600615</td>
          <td>0.066442</td>
          <td>22.961779</td>
          <td>19.089753</td>
          <td>21.038654</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.390532</td>
          <td>0.720619</td>
          <td>18.943181</td>
          <td>0.005011</td>
          <td>19.749453</td>
          <td>0.005017</td>
          <td>26.126443</td>
          <td>0.142790</td>
          <td>24.637894</td>
          <td>0.073757</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.390020</td>
          <td>25.184450</td>
          <td>18.886763</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.338476</td>
          <td>0.060642</td>
          <td>22.283090</td>
          <td>0.006010</td>
          <td>27.430387</td>
          <td>0.268225</td>
          <td>25.130121</td>
          <td>0.059586</td>
          <td>24.286069</td>
          <td>0.053997</td>
          <td>21.132214</td>
          <td>0.008758</td>
          <td>28.969342</td>
          <td>22.735897</td>
          <td>26.416098</td>
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
          <td>18.980739</td>
          <td>0.005080</td>
          <td>22.300947</td>
          <td>0.006038</td>
          <td>28.338061</td>
          <td>0.541221</td>
          <td>18.403363</td>
          <td>0.005006</td>
          <td>26.650797</td>
          <td>0.401035</td>
          <td>20.371855</td>
          <td>0.006206</td>
          <td>27.240830</td>
          <td>27.207820</td>
          <td>26.004181</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.098035</td>
          <td>0.117891</td>
          <td>19.052314</td>
          <td>0.005013</td>
          <td>22.041738</td>
          <td>0.005509</td>
          <td>26.090123</td>
          <td>0.138390</td>
          <td>26.912521</td>
          <td>0.488748</td>
          <td>24.612990</td>
          <td>0.160832</td>
          <td>22.861169</td>
          <td>24.726348</td>
          <td>27.264110</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.519355</td>
          <td>0.013100</td>
          <td>21.543662</td>
          <td>0.005324</td>
          <td>23.050677</td>
          <td>0.007477</td>
          <td>20.159896</td>
          <td>0.005063</td>
          <td>19.328967</td>
          <td>0.005056</td>
          <td>24.041714</td>
          <td>0.098035</td>
          <td>19.588628</td>
          <td>25.996481</td>
          <td>25.488711</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.565792</td>
          <td>0.005160</td>
          <td>24.316172</td>
          <td>0.020462</td>
          <td>18.979037</td>
          <td>0.005007</td>
          <td>20.673127</td>
          <td>0.005138</td>
          <td>23.866282</td>
          <td>0.037212</td>
          <td>20.201217</td>
          <td>0.005921</td>
          <td>25.093206</td>
          <td>26.267653</td>
          <td>22.948807</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.323728</td>
          <td>0.059860</td>
          <td>19.789693</td>
          <td>0.005029</td>
          <td>22.481946</td>
          <td>0.006032</td>
          <td>26.481401</td>
          <td>0.193224</td>
          <td>24.100993</td>
          <td>0.045816</td>
          <td>22.423152</td>
          <td>0.023521</td>
          <td>27.574233</td>
          <td>20.559816</td>
          <td>26.215968</td>
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
          <td>18.725428</td>
          <td>24.399663</td>
          <td>25.311801</td>
          <td>28.769341</td>
          <td>23.591156</td>
          <td>17.780518</td>
          <td>29.805293</td>
          <td>1.405554</td>
          <td>18.141084</td>
          <td>0.005001</td>
          <td>20.271096</td>
          <td>0.005058</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.508699</td>
          <td>23.960244</td>
          <td>22.289428</td>
          <td>23.576174</td>
          <td>28.922254</td>
          <td>22.612407</td>
          <td>20.456834</td>
          <td>0.005027</td>
          <td>23.547417</td>
          <td>0.016351</td>
          <td>24.588035</td>
          <td>0.040465</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.529464</td>
          <td>24.458368</td>
          <td>26.214194</td>
          <td>20.523772</td>
          <td>23.895809</td>
          <td>23.467580</td>
          <td>22.965778</td>
          <td>0.007259</td>
          <td>19.093535</td>
          <td>0.005007</td>
          <td>21.038603</td>
          <td>0.005235</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.229162</td>
          <td>18.949050</td>
          <td>19.752717</td>
          <td>25.950062</td>
          <td>24.674750</td>
          <td>29.502978</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.211390</td>
          <td>0.070496</td>
          <td>18.886417</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.293251</td>
          <td>22.282948</td>
          <td>27.467756</td>
          <td>25.173156</td>
          <td>24.313506</td>
          <td>21.131268</td>
          <td>29.819941</td>
          <td>1.416208</td>
          <td>22.748972</td>
          <td>0.009001</td>
          <td>26.757614</td>
          <td>0.266376</td>
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
          <td>18.972000</td>
          <td>22.299579</td>
          <td>27.820251</td>
          <td>18.402108</td>
          <td>26.793546</td>
          <td>20.382886</td>
          <td>27.648179</td>
          <td>0.336526</td>
          <td>27.437246</td>
          <td>0.454560</td>
          <td>26.002107</td>
          <td>0.141035</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.079836</td>
          <td>19.043932</td>
          <td>22.047356</td>
          <td>26.084351</td>
          <td>30.372787</td>
          <td>24.430258</td>
          <td>22.868019</td>
          <td>0.006938</td>
          <td>24.728596</td>
          <td>0.045866</td>
          <td>28.115654</td>
          <td>0.736812</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.510525</td>
          <td>21.543432</td>
          <td>23.045952</td>
          <td>20.159242</td>
          <td>19.318659</td>
          <td>24.027095</td>
          <td>19.584882</td>
          <td>0.005005</td>
          <td>25.940969</td>
          <td>0.133777</td>
          <td>25.480756</td>
          <td>0.089468</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.564690</td>
          <td>24.317696</td>
          <td>18.973691</td>
          <td>20.667544</td>
          <td>23.878697</td>
          <td>20.198904</td>
          <td>25.139761</td>
          <td>0.038763</td>
          <td>26.393809</td>
          <td>0.196972</td>
          <td>22.961198</td>
          <td>0.010379</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.349120</td>
          <td>19.789898</td>
          <td>22.496389</td>
          <td>26.533412</td>
          <td>24.115390</td>
          <td>22.412482</td>
          <td>27.133075</td>
          <td>0.221338</td>
          <td>20.563998</td>
          <td>0.005099</td>
          <td>26.330982</td>
          <td>0.186802</td>
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
          <td>18.725428</td>
          <td>24.399663</td>
          <td>25.311801</td>
          <td>28.769341</td>
          <td>23.591156</td>
          <td>17.780518</td>
          <td>26.097455</td>
          <td>0.777884</td>
          <td>18.135273</td>
          <td>0.005038</td>
          <td>20.272218</td>
          <td>0.006951</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.508699</td>
          <td>23.960244</td>
          <td>22.289428</td>
          <td>23.576174</td>
          <td>28.922254</td>
          <td>22.612407</td>
          <td>20.467635</td>
          <td>0.008071</td>
          <td>23.537776</td>
          <td>0.086140</td>
          <td>24.265871</td>
          <td>0.176773</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.529464</td>
          <td>24.458368</td>
          <td>26.214194</td>
          <td>20.523772</td>
          <td>23.895809</td>
          <td>23.467580</td>
          <td>22.874078</td>
          <td>0.057084</td>
          <td>19.090153</td>
          <td>0.005215</td>
          <td>21.042441</td>
          <td>0.011002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.229162</td>
          <td>18.949050</td>
          <td>19.752717</td>
          <td>25.950062</td>
          <td>24.674750</td>
          <td>29.502978</td>
          <td>29.184431</td>
          <td>3.195770</td>
          <td>24.965258</td>
          <td>0.290715</td>
          <td>18.891287</td>
          <td>0.005180</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.293251</td>
          <td>22.282948</td>
          <td>27.467756</td>
          <td>25.173156</td>
          <td>24.313506</td>
          <td>21.131268</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.704847</td>
          <td>0.041076</td>
          <td>25.297396</td>
          <td>0.408723</td>
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
          <td>18.972000</td>
          <td>22.299579</td>
          <td>27.820251</td>
          <td>18.402108</td>
          <td>26.793546</td>
          <td>20.382886</td>
          <td>27.052602</td>
          <td>1.367557</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.142903</td>
          <td>2.155828</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.079836</td>
          <td>19.043932</td>
          <td>22.047356</td>
          <td>26.084351</td>
          <td>30.372787</td>
          <td>24.430258</td>
          <td>22.812008</td>
          <td>0.054013</td>
          <td>24.873147</td>
          <td>0.269774</td>
          <td>27.606866</td>
          <td>1.711014</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.510525</td>
          <td>21.543432</td>
          <td>23.045952</td>
          <td>20.159242</td>
          <td>19.318659</td>
          <td>24.027095</td>
          <td>19.596257</td>
          <td>0.005751</td>
          <td>26.496680</td>
          <td>0.884344</td>
          <td>25.882703</td>
          <td>0.628270</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.564690</td>
          <td>24.317696</td>
          <td>18.973691</td>
          <td>20.667544</td>
          <td>23.878697</td>
          <td>20.198904</td>
          <td>24.938712</td>
          <td>0.334011</td>
          <td>25.761577</td>
          <td>0.536614</td>
          <td>22.883571</td>
          <td>0.052661</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.349120</td>
          <td>19.789898</td>
          <td>22.496389</td>
          <td>26.533412</td>
          <td>24.115390</td>
          <td>22.412482</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.555077</td>
          <td>0.007592</td>
          <td>25.521034</td>
          <td>0.483953</td>
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


