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
          <td>25.594951</td>
          <td>23.710761</td>
          <td>22.127479</td>
          <td>24.306710</td>
          <td>22.326028</td>
          <td>19.581504</td>
          <td>26.472950</td>
          <td>21.642568</td>
          <td>26.354120</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.416095</td>
          <td>20.034283</td>
          <td>22.761375</td>
          <td>25.903602</td>
          <td>21.276993</td>
          <td>25.777043</td>
          <td>25.155456</td>
          <td>21.099612</td>
          <td>20.683909</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.878103</td>
          <td>21.862828</td>
          <td>22.018546</td>
          <td>23.626379</td>
          <td>21.655187</td>
          <td>21.873703</td>
          <td>15.985357</td>
          <td>26.011389</td>
          <td>23.102474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.769936</td>
          <td>18.893107</td>
          <td>23.573223</td>
          <td>22.959095</td>
          <td>26.259167</td>
          <td>21.860151</td>
          <td>22.409107</td>
          <td>26.104098</td>
          <td>22.547465</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.029069</td>
          <td>23.309193</td>
          <td>19.301584</td>
          <td>20.925764</td>
          <td>22.927249</td>
          <td>22.405153</td>
          <td>24.750312</td>
          <td>22.323275</td>
          <td>25.247917</td>
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
          <td>24.475493</td>
          <td>23.189946</td>
          <td>19.848360</td>
          <td>22.654307</td>
          <td>25.481862</td>
          <td>21.335590</td>
          <td>20.282415</td>
          <td>26.084485</td>
          <td>20.108689</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.610781</td>
          <td>23.525278</td>
          <td>26.090218</td>
          <td>22.913373</td>
          <td>20.024765</td>
          <td>27.443135</td>
          <td>30.088100</td>
          <td>23.188733</td>
          <td>21.773578</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.866458</td>
          <td>27.275457</td>
          <td>22.336351</td>
          <td>16.881642</td>
          <td>25.412394</td>
          <td>23.442565</td>
          <td>28.263121</td>
          <td>23.311196</td>
          <td>24.003204</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.574236</td>
          <td>29.449861</td>
          <td>20.803693</td>
          <td>20.156413</td>
          <td>20.932777</td>
          <td>21.486274</td>
          <td>23.863362</td>
          <td>24.995051</td>
          <td>26.701138</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.023829</td>
          <td>20.828024</td>
          <td>19.744481</td>
          <td>25.714379</td>
          <td>22.662160</td>
          <td>26.966082</td>
          <td>22.385614</td>
          <td>24.581353</td>
          <td>21.954916</td>
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
          <td>25.592167</td>
          <td>0.180020</td>
          <td>23.716341</td>
          <td>0.012628</td>
          <td>22.128414</td>
          <td>0.005585</td>
          <td>24.260852</td>
          <td>0.027621</td>
          <td>22.307181</td>
          <td>0.010328</td>
          <td>19.590152</td>
          <td>0.005346</td>
          <td>26.472950</td>
          <td>21.642568</td>
          <td>26.354120</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.367539</td>
          <td>1.301415</td>
          <td>20.038051</td>
          <td>0.005040</td>
          <td>22.762323</td>
          <td>0.006601</td>
          <td>26.037447</td>
          <td>0.132234</td>
          <td>21.289389</td>
          <td>0.006202</td>
          <td>25.085445</td>
          <td>0.239271</td>
          <td>25.155456</td>
          <td>21.099612</td>
          <td>20.683909</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.855350</td>
          <td>0.095449</td>
          <td>21.865368</td>
          <td>0.005531</td>
          <td>22.022562</td>
          <td>0.005493</td>
          <td>23.616105</td>
          <td>0.015969</td>
          <td>21.664279</td>
          <td>0.007140</td>
          <td>21.894073</td>
          <td>0.015104</td>
          <td>15.985357</td>
          <td>26.011389</td>
          <td>23.102474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.800838</td>
          <td>0.037819</td>
          <td>18.891866</td>
          <td>0.005011</td>
          <td>23.578431</td>
          <td>0.010213</td>
          <td>22.965997</td>
          <td>0.009813</td>
          <td>25.782063</td>
          <td>0.198760</td>
          <td>21.858861</td>
          <td>0.014684</td>
          <td>22.409107</td>
          <td>26.104098</td>
          <td>22.547465</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.041918</td>
          <td>0.046717</td>
          <td>23.309687</td>
          <td>0.009505</td>
          <td>19.302374</td>
          <td>0.005010</td>
          <td>20.923971</td>
          <td>0.005205</td>
          <td>22.918055</td>
          <td>0.016442</td>
          <td>22.356570</td>
          <td>0.022212</td>
          <td>24.750312</td>
          <td>22.323275</td>
          <td>25.247917</td>
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
          <td>24.409828</td>
          <td>0.064571</td>
          <td>23.174049</td>
          <td>0.008745</td>
          <td>19.853641</td>
          <td>0.005019</td>
          <td>22.652026</td>
          <td>0.008103</td>
          <td>25.693423</td>
          <td>0.184449</td>
          <td>21.352995</td>
          <td>0.010086</td>
          <td>20.282415</td>
          <td>26.084485</td>
          <td>20.108689</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.609715</td>
          <td>0.014019</td>
          <td>23.498958</td>
          <td>0.010787</td>
          <td>26.100708</td>
          <td>0.086169</td>
          <td>22.902118</td>
          <td>0.009412</td>
          <td>20.017880</td>
          <td>0.005158</td>
          <td>30.439910</td>
          <td>3.864500</td>
          <td>30.088100</td>
          <td>23.188733</td>
          <td>21.773578</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.868302</td>
          <td>0.005234</td>
          <td>27.665889</td>
          <td>0.363269</td>
          <td>22.330726</td>
          <td>0.005810</td>
          <td>16.879840</td>
          <td>0.005001</td>
          <td>25.345172</td>
          <td>0.136963</td>
          <td>23.444970</td>
          <td>0.057877</td>
          <td>28.263121</td>
          <td>23.311196</td>
          <td>24.003204</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.565422</td>
          <td>0.005051</td>
          <td>29.059631</td>
          <td>0.965594</td>
          <td>20.812186</td>
          <td>0.005074</td>
          <td>20.160222</td>
          <td>0.005063</td>
          <td>20.939309</td>
          <td>0.005689</td>
          <td>21.485462</td>
          <td>0.011060</td>
          <td>23.863362</td>
          <td>24.995051</td>
          <td>26.701138</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.023041</td>
          <td>0.009363</td>
          <td>20.832335</td>
          <td>0.005114</td>
          <td>19.744312</td>
          <td>0.005017</td>
          <td>25.581581</td>
          <td>0.088820</td>
          <td>22.641203</td>
          <td>0.013187</td>
          <td>28.160312</td>
          <td>1.786358</td>
          <td>22.385614</td>
          <td>24.581353</td>
          <td>21.954916</td>
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
          <td>25.594951</td>
          <td>23.710761</td>
          <td>22.127479</td>
          <td>24.306710</td>
          <td>22.326028</td>
          <td>19.581504</td>
          <td>26.426204</td>
          <td>0.121095</td>
          <td>21.655516</td>
          <td>0.005700</td>
          <td>26.697218</td>
          <td>0.253523</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.416095</td>
          <td>20.034283</td>
          <td>22.761375</td>
          <td>25.903602</td>
          <td>21.276993</td>
          <td>25.777043</td>
          <td>25.171980</td>
          <td>0.039891</td>
          <td>21.109366</td>
          <td>0.005267</td>
          <td>20.686744</td>
          <td>0.005124</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.878103</td>
          <td>21.862828</td>
          <td>22.018546</td>
          <td>23.626379</td>
          <td>21.655187</td>
          <td>21.873703</td>
          <td>15.976134</td>
          <td>0.005000</td>
          <td>26.193611</td>
          <td>0.166226</td>
          <td>23.118199</td>
          <td>0.011634</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.769936</td>
          <td>18.893107</td>
          <td>23.573223</td>
          <td>22.959095</td>
          <td>26.259167</td>
          <td>21.860151</td>
          <td>22.412666</td>
          <td>0.005917</td>
          <td>26.424239</td>
          <td>0.202078</td>
          <td>22.560311</td>
          <td>0.008038</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.029069</td>
          <td>23.309193</td>
          <td>19.301584</td>
          <td>20.925764</td>
          <td>22.927249</td>
          <td>22.405153</td>
          <td>24.769274</td>
          <td>0.027913</td>
          <td>22.336651</td>
          <td>0.007159</td>
          <td>25.163542</td>
          <td>0.067564</td>
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
          <td>24.475493</td>
          <td>23.189946</td>
          <td>19.848360</td>
          <td>22.654307</td>
          <td>25.481862</td>
          <td>21.335590</td>
          <td>20.280935</td>
          <td>0.005020</td>
          <td>26.389622</td>
          <td>0.196279</td>
          <td>20.102064</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.610781</td>
          <td>23.525278</td>
          <td>26.090218</td>
          <td>22.913373</td>
          <td>20.024765</td>
          <td>27.443135</td>
          <td>29.091966</td>
          <td>0.938459</td>
          <td>23.205007</td>
          <td>0.012426</td>
          <td>21.784276</td>
          <td>0.005874</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.866458</td>
          <td>27.275457</td>
          <td>22.336351</td>
          <td>16.881642</td>
          <td>25.412394</td>
          <td>23.442565</td>
          <td>28.454167</td>
          <td>0.615826</td>
          <td>23.317218</td>
          <td>0.013564</td>
          <td>23.990247</td>
          <td>0.023859</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.574236</td>
          <td>29.449861</td>
          <td>20.803693</td>
          <td>20.156413</td>
          <td>20.932777</td>
          <td>21.486274</td>
          <td>23.860929</td>
          <td>0.012976</td>
          <td>25.051823</td>
          <td>0.061175</td>
          <td>26.381313</td>
          <td>0.194910</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.023829</td>
          <td>20.828024</td>
          <td>19.744481</td>
          <td>25.714379</td>
          <td>22.662160</td>
          <td>26.966082</td>
          <td>22.383698</td>
          <td>0.005873</td>
          <td>24.593121</td>
          <td>0.040649</td>
          <td>21.950781</td>
          <td>0.006157</td>
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
          <td>25.594951</td>
          <td>23.710761</td>
          <td>22.127479</td>
          <td>24.306710</td>
          <td>22.326028</td>
          <td>19.581504</td>
          <td>26.150072</td>
          <td>0.805116</td>
          <td>21.642472</td>
          <td>0.016284</td>
          <td>26.636119</td>
          <td>1.024191</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.416095</td>
          <td>20.034283</td>
          <td>22.761375</td>
          <td>25.903602</td>
          <td>21.276993</td>
          <td>25.777043</td>
          <td>25.637072</td>
          <td>0.566681</td>
          <td>21.101242</td>
          <td>0.010679</td>
          <td>20.682034</td>
          <td>0.008634</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.878103</td>
          <td>21.862828</td>
          <td>22.018546</td>
          <td>23.626379</td>
          <td>21.655187</td>
          <td>21.873703</td>
          <td>15.983240</td>
          <td>0.005001</td>
          <td>25.218191</td>
          <td>0.355628</td>
          <td>23.056731</td>
          <td>0.061443</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.769936</td>
          <td>18.893107</td>
          <td>23.573223</td>
          <td>22.959095</td>
          <td>26.259167</td>
          <td>21.860151</td>
          <td>22.416785</td>
          <td>0.037978</td>
          <td>25.500921</td>
          <td>0.442271</td>
          <td>22.495246</td>
          <td>0.037257</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.029069</td>
          <td>23.309193</td>
          <td>19.301584</td>
          <td>20.925764</td>
          <td>22.927249</td>
          <td>22.405153</td>
          <td>24.980649</td>
          <td>0.345274</td>
          <td>22.334584</td>
          <td>0.029568</td>
          <td>25.555359</td>
          <td>0.496422</td>
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
          <td>24.475493</td>
          <td>23.189946</td>
          <td>19.848360</td>
          <td>22.654307</td>
          <td>25.481862</td>
          <td>21.335590</td>
          <td>20.283961</td>
          <td>0.007324</td>
          <td>25.371869</td>
          <td>0.400780</td>
          <td>20.104189</td>
          <td>0.006490</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.610781</td>
          <td>23.525278</td>
          <td>26.090218</td>
          <td>22.913373</td>
          <td>20.024765</td>
          <td>27.443135</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.113415</td>
          <td>0.059119</td>
          <td>21.770169</td>
          <td>0.019728</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.866458</td>
          <td>27.275457</td>
          <td>22.336351</td>
          <td>16.881642</td>
          <td>25.412394</td>
          <td>23.442565</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.436349</td>
          <td>0.078753</td>
          <td>24.231915</td>
          <td>0.171742</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.574236</td>
          <td>29.449861</td>
          <td>20.803693</td>
          <td>20.156413</td>
          <td>20.932777</td>
          <td>21.486274</td>
          <td>23.854495</td>
          <td>0.135352</td>
          <td>24.495948</td>
          <td>0.197327</td>
          <td>25.958636</td>
          <td>0.662286</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.023829</td>
          <td>20.828024</td>
          <td>19.744481</td>
          <td>25.714379</td>
          <td>22.662160</td>
          <td>26.966082</td>
          <td>22.326955</td>
          <td>0.035063</td>
          <td>25.071976</td>
          <td>0.316737</td>
          <td>21.948070</td>
          <td>0.022999</td>
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


