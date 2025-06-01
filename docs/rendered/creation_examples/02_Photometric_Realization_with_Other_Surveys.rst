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
          <td>29.155298</td>
          <td>21.891557</td>
          <td>19.926735</td>
          <td>24.421923</td>
          <td>20.704988</td>
          <td>26.399094</td>
          <td>25.508697</td>
          <td>26.515995</td>
          <td>19.554188</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.167770</td>
          <td>20.493404</td>
          <td>20.220273</td>
          <td>21.551628</td>
          <td>21.925103</td>
          <td>27.715337</td>
          <td>26.699616</td>
          <td>23.850463</td>
          <td>25.038439</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.566287</td>
          <td>24.509081</td>
          <td>23.217261</td>
          <td>20.928781</td>
          <td>26.953577</td>
          <td>15.132141</td>
          <td>21.482828</td>
          <td>22.847971</td>
          <td>22.201177</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.947636</td>
          <td>19.020921</td>
          <td>27.377398</td>
          <td>25.751673</td>
          <td>14.596418</td>
          <td>27.423638</td>
          <td>24.711120</td>
          <td>20.568191</td>
          <td>23.411624</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.812902</td>
          <td>21.367284</td>
          <td>28.381551</td>
          <td>25.273765</td>
          <td>26.284700</td>
          <td>21.296150</td>
          <td>25.561347</td>
          <td>26.364965</td>
          <td>26.332669</td>
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
          <td>19.708056</td>
          <td>21.566354</td>
          <td>22.557115</td>
          <td>25.619750</td>
          <td>28.068711</td>
          <td>23.327498</td>
          <td>22.854005</td>
          <td>25.610419</td>
          <td>24.122638</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.521334</td>
          <td>19.977890</td>
          <td>26.438586</td>
          <td>20.271870</td>
          <td>27.215646</td>
          <td>26.157555</td>
          <td>23.088782</td>
          <td>24.023517</td>
          <td>21.330768</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.695412</td>
          <td>25.002718</td>
          <td>22.039411</td>
          <td>24.747115</td>
          <td>22.138444</td>
          <td>26.217952</td>
          <td>26.919971</td>
          <td>19.781811</td>
          <td>28.701544</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.892421</td>
          <td>26.198519</td>
          <td>21.836956</td>
          <td>23.015576</td>
          <td>17.534666</td>
          <td>18.475268</td>
          <td>20.921746</td>
          <td>27.475505</td>
          <td>21.812379</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.347839</td>
          <td>28.548606</td>
          <td>22.533606</td>
          <td>22.996822</td>
          <td>21.503218</td>
          <td>26.081386</td>
          <td>22.798322</td>
          <td>22.398005</td>
          <td>24.645606</td>
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
          <td>26.726260</td>
          <td>0.448314</td>
          <td>21.886367</td>
          <td>0.005548</td>
          <td>19.923908</td>
          <td>0.005021</td>
          <td>24.464400</td>
          <td>0.033026</td>
          <td>20.712028</td>
          <td>0.005478</td>
          <td>26.070897</td>
          <td>0.517517</td>
          <td>25.508697</td>
          <td>26.515995</td>
          <td>19.554188</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.053818</td>
          <td>0.113456</td>
          <td>20.496442</td>
          <td>0.005072</td>
          <td>20.224939</td>
          <td>0.005032</td>
          <td>21.552659</td>
          <td>0.005563</td>
          <td>21.920964</td>
          <td>0.008121</td>
          <td>26.999612</td>
          <td>0.966763</td>
          <td>26.699616</td>
          <td>23.850463</td>
          <td>25.038439</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.915626</td>
          <td>0.516048</td>
          <td>24.500639</td>
          <td>0.023948</td>
          <td>23.207907</td>
          <td>0.008116</td>
          <td>20.923638</td>
          <td>0.005205</td>
          <td>26.998942</td>
          <td>0.520860</td>
          <td>15.140029</td>
          <td>0.005001</td>
          <td>21.482828</td>
          <td>22.847971</td>
          <td>22.201177</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.028686</td>
          <td>0.111007</td>
          <td>19.016057</td>
          <td>0.005012</td>
          <td>27.139006</td>
          <td>0.210851</td>
          <td>25.756650</td>
          <td>0.103569</td>
          <td>14.592895</td>
          <td>0.005000</td>
          <td>26.971533</td>
          <td>0.950301</td>
          <td>24.711120</td>
          <td>20.568191</td>
          <td>23.411624</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.812577</td>
          <td>0.005218</td>
          <td>21.358139</td>
          <td>0.005245</td>
          <td>28.824386</td>
          <td>0.758666</td>
          <td>25.337905</td>
          <td>0.071634</td>
          <td>26.300327</td>
          <td>0.304409</td>
          <td>21.297961</td>
          <td>0.009722</td>
          <td>25.561347</td>
          <td>26.364965</td>
          <td>26.332669</td>
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
          <td>19.713442</td>
          <td>0.005192</td>
          <td>21.578192</td>
          <td>0.005342</td>
          <td>22.558481</td>
          <td>0.006164</td>
          <td>25.752671</td>
          <td>0.103209</td>
          <td>26.246596</td>
          <td>0.291526</td>
          <td>23.356005</td>
          <td>0.053483</td>
          <td>22.854005</td>
          <td>25.610419</td>
          <td>24.122638</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.439911</td>
          <td>0.066303</td>
          <td>19.971060</td>
          <td>0.005037</td>
          <td>26.451870</td>
          <td>0.117197</td>
          <td>20.283562</td>
          <td>0.005076</td>
          <td>28.380298</td>
          <td>1.261731</td>
          <td>27.204755</td>
          <td>1.092312</td>
          <td>23.088782</td>
          <td>24.023517</td>
          <td>21.330768</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.678807</td>
          <td>0.007764</td>
          <td>24.999624</td>
          <td>0.037014</td>
          <td>22.041397</td>
          <td>0.005509</td>
          <td>24.709409</td>
          <td>0.041017</td>
          <td>22.147459</td>
          <td>0.009292</td>
          <td>26.029533</td>
          <td>0.502025</td>
          <td>26.919971</td>
          <td>19.781811</td>
          <td>28.701544</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.864667</td>
          <td>0.096229</td>
          <td>26.160099</td>
          <td>0.103098</td>
          <td>21.840752</td>
          <td>0.005368</td>
          <td>22.998948</td>
          <td>0.010032</td>
          <td>17.529082</td>
          <td>0.005006</td>
          <td>18.475175</td>
          <td>0.005062</td>
          <td>20.921746</td>
          <td>27.475505</td>
          <td>21.812379</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.222387</td>
          <td>0.642359</td>
          <td>28.873758</td>
          <td>0.860012</td>
          <td>22.535139</td>
          <td>0.006122</td>
          <td>23.001790</td>
          <td>0.010051</td>
          <td>21.514069</td>
          <td>0.006704</td>
          <td>25.746444</td>
          <td>0.405654</td>
          <td>22.798322</td>
          <td>22.398005</td>
          <td>24.645606</td>
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
          <td>29.155298</td>
          <td>21.891557</td>
          <td>19.926735</td>
          <td>24.421923</td>
          <td>20.704988</td>
          <td>26.399094</td>
          <td>25.519207</td>
          <td>0.054360</td>
          <td>26.656386</td>
          <td>0.245148</td>
          <td>19.546307</td>
          <td>0.005015</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.167770</td>
          <td>20.493404</td>
          <td>20.220273</td>
          <td>21.551628</td>
          <td>21.925103</td>
          <td>27.715337</td>
          <td>26.846483</td>
          <td>0.173884</td>
          <td>23.807082</td>
          <td>0.020362</td>
          <td>25.020498</td>
          <td>0.059493</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.566287</td>
          <td>24.509081</td>
          <td>23.217261</td>
          <td>20.928781</td>
          <td>26.953577</td>
          <td>15.132141</td>
          <td>21.491661</td>
          <td>0.005180</td>
          <td>22.835849</td>
          <td>0.009525</td>
          <td>22.201722</td>
          <td>0.006744</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.947636</td>
          <td>19.020921</td>
          <td>27.377398</td>
          <td>25.751673</td>
          <td>14.596418</td>
          <td>27.423638</td>
          <td>24.708363</td>
          <td>0.026457</td>
          <td>20.569126</td>
          <td>0.005100</td>
          <td>23.402611</td>
          <td>0.014522</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.812902</td>
          <td>21.367284</td>
          <td>28.381551</td>
          <td>25.273765</td>
          <td>26.284700</td>
          <td>21.296150</td>
          <td>25.633443</td>
          <td>0.060182</td>
          <td>26.204643</td>
          <td>0.167797</td>
          <td>26.006487</td>
          <td>0.141568</td>
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
          <td>19.708056</td>
          <td>21.566354</td>
          <td>22.557115</td>
          <td>25.619750</td>
          <td>28.068711</td>
          <td>23.327498</td>
          <td>22.844031</td>
          <td>0.006866</td>
          <td>25.612262</td>
          <td>0.100441</td>
          <td>24.108398</td>
          <td>0.026458</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.521334</td>
          <td>19.977890</td>
          <td>26.438586</td>
          <td>20.271870</td>
          <td>27.215646</td>
          <td>26.157555</td>
          <td>23.088677</td>
          <td>0.007728</td>
          <td>24.016936</td>
          <td>0.024421</td>
          <td>21.337378</td>
          <td>0.005401</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.695412</td>
          <td>25.002718</td>
          <td>22.039411</td>
          <td>24.747115</td>
          <td>22.138444</td>
          <td>26.217952</td>
          <td>27.392010</td>
          <td>0.273951</td>
          <td>19.780693</td>
          <td>0.005024</td>
          <td>32.665656</td>
          <td>4.535022</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.892421</td>
          <td>26.198519</td>
          <td>21.836956</td>
          <td>23.015576</td>
          <td>17.534666</td>
          <td>18.475268</td>
          <td>20.922914</td>
          <td>0.005064</td>
          <td>27.215742</td>
          <td>0.383765</td>
          <td>21.812608</td>
          <td>0.005917</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.347839</td>
          <td>28.548606</td>
          <td>22.533606</td>
          <td>22.996822</td>
          <td>21.503218</td>
          <td>26.081386</td>
          <td>22.791041</td>
          <td>0.006714</td>
          <td>22.407912</td>
          <td>0.007411</td>
          <td>24.594577</td>
          <td>0.040702</td>
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
          <td>29.155298</td>
          <td>21.891557</td>
          <td>19.926735</td>
          <td>24.421923</td>
          <td>20.704988</td>
          <td>26.399094</td>
          <td>24.961288</td>
          <td>0.340035</td>
          <td>25.641201</td>
          <td>0.491248</td>
          <td>19.555432</td>
          <td>0.005589</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.167770</td>
          <td>20.493404</td>
          <td>20.220273</td>
          <td>21.551628</td>
          <td>21.925103</td>
          <td>27.715337</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.981004</td>
          <td>0.127001</td>
          <td>25.181897</td>
          <td>0.373799</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.566287</td>
          <td>24.509081</td>
          <td>23.217261</td>
          <td>20.928781</td>
          <td>26.953577</td>
          <td>15.132141</td>
          <td>21.475262</td>
          <td>0.016734</td>
          <td>22.836106</td>
          <td>0.046174</td>
          <td>22.137353</td>
          <td>0.027139</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.947636</td>
          <td>19.020921</td>
          <td>27.377398</td>
          <td>25.751673</td>
          <td>14.596418</td>
          <td>27.423638</td>
          <td>24.571388</td>
          <td>0.248196</td>
          <td>20.554190</td>
          <td>0.007589</td>
          <td>23.482183</td>
          <td>0.089581</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.812902</td>
          <td>21.367284</td>
          <td>28.381551</td>
          <td>25.273765</td>
          <td>26.284700</td>
          <td>21.296150</td>
          <td>25.066812</td>
          <td>0.369427</td>
          <td>25.804953</td>
          <td>0.553735</td>
          <td>26.595312</td>
          <td>0.999455</td>
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
          <td>19.708056</td>
          <td>21.566354</td>
          <td>22.557115</td>
          <td>25.619750</td>
          <td>28.068711</td>
          <td>23.327498</td>
          <td>22.800876</td>
          <td>0.053480</td>
          <td>25.288081</td>
          <td>0.375604</td>
          <td>24.071628</td>
          <td>0.149736</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.521334</td>
          <td>19.977890</td>
          <td>26.438586</td>
          <td>20.271870</td>
          <td>27.215646</td>
          <td>26.157555</td>
          <td>23.111597</td>
          <td>0.070509</td>
          <td>23.863450</td>
          <td>0.114650</td>
          <td>21.324401</td>
          <td>0.013641</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.695412</td>
          <td>25.002718</td>
          <td>22.039411</td>
          <td>24.747115</td>
          <td>22.138444</td>
          <td>26.217952</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.787662</td>
          <td>0.005740</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.892421</td>
          <td>26.198519</td>
          <td>21.836956</td>
          <td>23.015576</td>
          <td>17.534666</td>
          <td>18.475268</td>
          <td>20.923473</td>
          <td>0.010851</td>
          <td>26.407665</td>
          <td>0.835653</td>
          <td>21.816351</td>
          <td>0.020525</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.347839</td>
          <td>28.548606</td>
          <td>22.533606</td>
          <td>22.996822</td>
          <td>21.503218</td>
          <td>26.081386</td>
          <td>22.826772</td>
          <td>0.054728</td>
          <td>22.405587</td>
          <td>0.031484</td>
          <td>24.807848</td>
          <td>0.277502</td>
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


