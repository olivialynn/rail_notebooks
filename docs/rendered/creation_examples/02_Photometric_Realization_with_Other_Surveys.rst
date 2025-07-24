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
          <td>28.181998</td>
          <td>23.448955</td>
          <td>22.822242</td>
          <td>22.013497</td>
          <td>22.812341</td>
          <td>25.306135</td>
          <td>22.995346</td>
          <td>24.507920</td>
          <td>21.716695</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.360377</td>
          <td>21.556209</td>
          <td>24.010393</td>
          <td>23.992579</td>
          <td>27.147400</td>
          <td>25.559215</td>
          <td>19.852835</td>
          <td>18.333988</td>
          <td>21.134299</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.021532</td>
          <td>29.081114</td>
          <td>24.099460</td>
          <td>24.744458</td>
          <td>22.650830</td>
          <td>21.352374</td>
          <td>27.830244</td>
          <td>20.745898</td>
          <td>22.306049</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.941411</td>
          <td>23.925294</td>
          <td>24.388599</td>
          <td>25.364896</td>
          <td>20.714967</td>
          <td>27.351239</td>
          <td>18.221767</td>
          <td>21.113241</td>
          <td>27.712699</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.432279</td>
          <td>21.502693</td>
          <td>25.404576</td>
          <td>23.205095</td>
          <td>16.758840</td>
          <td>19.686318</td>
          <td>20.252312</td>
          <td>21.567891</td>
          <td>27.742502</td>
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
          <td>20.543287</td>
          <td>18.523023</td>
          <td>18.115838</td>
          <td>24.285783</td>
          <td>25.485183</td>
          <td>20.299707</td>
          <td>20.239882</td>
          <td>23.036917</td>
          <td>23.353073</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.199577</td>
          <td>23.781941</td>
          <td>18.044524</td>
          <td>26.031037</td>
          <td>19.818177</td>
          <td>27.200344</td>
          <td>22.979051</td>
          <td>17.532569</td>
          <td>22.407614</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.625391</td>
          <td>21.299536</td>
          <td>19.918275</td>
          <td>31.306514</td>
          <td>25.269362</td>
          <td>24.837621</td>
          <td>20.697803</td>
          <td>19.702187</td>
          <td>23.106922</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.524046</td>
          <td>21.107468</td>
          <td>22.115093</td>
          <td>24.826826</td>
          <td>23.772155</td>
          <td>22.215274</td>
          <td>21.541097</td>
          <td>22.528247</td>
          <td>17.926102</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.873134</td>
          <td>14.617455</td>
          <td>28.300904</td>
          <td>18.334292</td>
          <td>23.514198</td>
          <td>28.368738</td>
          <td>21.934679</td>
          <td>20.670852</td>
          <td>19.920361</td>
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
          <td>28.970247</td>
          <td>1.754711</td>
          <td>23.440644</td>
          <td>0.010362</td>
          <td>22.827548</td>
          <td>0.006770</td>
          <td>22.020753</td>
          <td>0.006192</td>
          <td>22.848006</td>
          <td>0.015531</td>
          <td>25.412549</td>
          <td>0.312182</td>
          <td>22.995346</td>
          <td>24.507920</td>
          <td>21.716695</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.378804</td>
          <td>0.011832</td>
          <td>21.554727</td>
          <td>0.005330</td>
          <td>24.009561</td>
          <td>0.014048</td>
          <td>24.018068</td>
          <td>0.022378</td>
          <td>26.001429</td>
          <td>0.238626</td>
          <td>25.741167</td>
          <td>0.404012</td>
          <td>19.852835</td>
          <td>18.333988</td>
          <td>21.134299</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.031451</td>
          <td>0.006128</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.101412</td>
          <td>0.015114</td>
          <td>24.717156</td>
          <td>0.041299</td>
          <td>22.651271</td>
          <td>0.013290</td>
          <td>21.361253</td>
          <td>0.010143</td>
          <td>27.830244</td>
          <td>20.745898</td>
          <td>22.306049</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.940159</td>
          <td>0.005256</td>
          <td>23.938475</td>
          <td>0.014999</td>
          <td>24.376389</td>
          <td>0.018957</td>
          <td>25.412450</td>
          <td>0.076514</td>
          <td>20.713156</td>
          <td>0.005479</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.221767</td>
          <td>21.113241</td>
          <td>27.712699</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.398768</td>
          <td>0.152729</td>
          <td>21.502075</td>
          <td>0.005305</td>
          <td>25.357771</td>
          <td>0.044617</td>
          <td>23.232655</td>
          <td>0.011839</td>
          <td>16.762280</td>
          <td>0.005003</td>
          <td>19.684284</td>
          <td>0.005402</td>
          <td>20.252312</td>
          <td>21.567891</td>
          <td>27.742502</td>
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
          <td>20.543840</td>
          <td>0.005573</td>
          <td>18.528857</td>
          <td>0.005007</td>
          <td>18.116422</td>
          <td>0.005003</td>
          <td>24.306701</td>
          <td>0.028751</td>
          <td>25.246534</td>
          <td>0.125760</td>
          <td>20.288507</td>
          <td>0.006058</td>
          <td>20.239882</td>
          <td>23.036917</td>
          <td>23.353073</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.021731</td>
          <td>0.557371</td>
          <td>23.778305</td>
          <td>0.013236</td>
          <td>18.043757</td>
          <td>0.005002</td>
          <td>25.910689</td>
          <td>0.118467</td>
          <td>19.820512</td>
          <td>0.005117</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.979051</td>
          <td>17.532569</td>
          <td>22.407614</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.627666</td>
          <td>0.005173</td>
          <td>21.298328</td>
          <td>0.005225</td>
          <td>19.927779</td>
          <td>0.005021</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.207488</td>
          <td>0.121570</td>
          <td>24.672616</td>
          <td>0.169221</td>
          <td>20.697803</td>
          <td>19.702187</td>
          <td>23.106922</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.378685</td>
          <td>0.150128</td>
          <td>21.103105</td>
          <td>0.005169</td>
          <td>22.109400</td>
          <td>0.005568</td>
          <td>24.753468</td>
          <td>0.042651</td>
          <td>23.776783</td>
          <td>0.034382</td>
          <td>22.213258</td>
          <td>0.019660</td>
          <td>21.541097</td>
          <td>22.528247</td>
          <td>17.926102</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.152620</td>
          <td>0.123594</td>
          <td>14.618853</td>
          <td>0.005000</td>
          <td>28.904277</td>
          <td>0.799561</td>
          <td>18.327993</td>
          <td>0.005006</td>
          <td>23.491968</td>
          <td>0.026778</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.934679</td>
          <td>20.670852</td>
          <td>19.920361</td>
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
          <td>28.181998</td>
          <td>23.448955</td>
          <td>22.822242</td>
          <td>22.013497</td>
          <td>22.812341</td>
          <td>25.306135</td>
          <td>22.996145</td>
          <td>0.007368</td>
          <td>24.474654</td>
          <td>0.036581</td>
          <td>21.723064</td>
          <td>0.005787</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.360377</td>
          <td>21.556209</td>
          <td>24.010393</td>
          <td>23.992579</td>
          <td>27.147400</td>
          <td>25.559215</td>
          <td>19.856519</td>
          <td>0.005009</td>
          <td>18.330737</td>
          <td>0.005002</td>
          <td>21.138667</td>
          <td>0.005281</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.021532</td>
          <td>29.081114</td>
          <td>24.099460</td>
          <td>24.744458</td>
          <td>22.650830</td>
          <td>21.352374</td>
          <td>29.336948</td>
          <td>1.086869</td>
          <td>20.749027</td>
          <td>0.005139</td>
          <td>22.296245</td>
          <td>0.007026</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.941411</td>
          <td>23.925294</td>
          <td>24.388599</td>
          <td>25.364896</td>
          <td>20.714967</td>
          <td>27.351239</td>
          <td>18.230368</td>
          <td>0.005000</td>
          <td>21.108679</td>
          <td>0.005267</td>
          <td>27.758007</td>
          <td>0.575242</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.432279</td>
          <td>21.502693</td>
          <td>25.404576</td>
          <td>23.205095</td>
          <td>16.758840</td>
          <td>19.686318</td>
          <td>20.248725</td>
          <td>0.005019</td>
          <td>21.567573</td>
          <td>0.005601</td>
          <td>27.197444</td>
          <td>0.378350</td>
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
          <td>20.543287</td>
          <td>18.523023</td>
          <td>18.115838</td>
          <td>24.285783</td>
          <td>25.485183</td>
          <td>20.299707</td>
          <td>20.239268</td>
          <td>0.005018</td>
          <td>23.034362</td>
          <td>0.010937</td>
          <td>23.356090</td>
          <td>0.013990</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.199577</td>
          <td>23.781941</td>
          <td>18.044524</td>
          <td>26.031037</td>
          <td>19.818177</td>
          <td>27.200344</td>
          <td>22.975619</td>
          <td>0.007294</td>
          <td>17.527355</td>
          <td>0.005000</td>
          <td>22.412759</td>
          <td>0.007429</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.625391</td>
          <td>21.299536</td>
          <td>19.918275</td>
          <td>31.306514</td>
          <td>25.269362</td>
          <td>24.837621</td>
          <td>20.703174</td>
          <td>0.005043</td>
          <td>19.702789</td>
          <td>0.005021</td>
          <td>23.101920</td>
          <td>0.011494</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.524046</td>
          <td>21.107468</td>
          <td>22.115093</td>
          <td>24.826826</td>
          <td>23.772155</td>
          <td>22.215274</td>
          <td>21.547494</td>
          <td>0.005199</td>
          <td>22.522994</td>
          <td>0.007873</td>
          <td>17.922588</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.873134</td>
          <td>14.617455</td>
          <td>28.300904</td>
          <td>18.334292</td>
          <td>23.514198</td>
          <td>28.368738</td>
          <td>21.936773</td>
          <td>0.005401</td>
          <td>20.664862</td>
          <td>0.005119</td>
          <td>19.924509</td>
          <td>0.005031</td>
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
          <td>28.181998</td>
          <td>23.448955</td>
          <td>22.822242</td>
          <td>22.013497</td>
          <td>22.812341</td>
          <td>25.306135</td>
          <td>23.015433</td>
          <td>0.064736</td>
          <td>24.693993</td>
          <td>0.232828</td>
          <td>21.702634</td>
          <td>0.018625</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.360377</td>
          <td>21.556209</td>
          <td>24.010393</td>
          <td>23.992579</td>
          <td>27.147400</td>
          <td>25.559215</td>
          <td>19.849645</td>
          <td>0.006155</td>
          <td>18.322596</td>
          <td>0.005053</td>
          <td>21.159795</td>
          <td>0.012005</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.021532</td>
          <td>29.081114</td>
          <td>24.099460</td>
          <td>24.744458</td>
          <td>22.650830</td>
          <td>21.352374</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.753990</td>
          <td>0.008488</td>
          <td>22.309721</td>
          <td>0.031599</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.941411</td>
          <td>23.925294</td>
          <td>24.388599</td>
          <td>25.364896</td>
          <td>20.714967</td>
          <td>27.351239</td>
          <td>18.219156</td>
          <td>0.005064</td>
          <td>21.122208</td>
          <td>0.010841</td>
          <td>27.794166</td>
          <td>1.862137</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.432279</td>
          <td>21.502693</td>
          <td>25.404576</td>
          <td>23.205095</td>
          <td>16.758840</td>
          <td>19.686318</td>
          <td>20.252564</td>
          <td>0.007213</td>
          <td>21.580694</td>
          <td>0.015475</td>
          <td>27.396011</td>
          <td>1.547254</td>
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
          <td>20.543287</td>
          <td>18.523023</td>
          <td>18.115838</td>
          <td>24.285783</td>
          <td>25.485183</td>
          <td>20.299707</td>
          <td>20.234199</td>
          <td>0.007151</td>
          <td>22.944819</td>
          <td>0.050873</td>
          <td>23.419205</td>
          <td>0.084739</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.199577</td>
          <td>23.781941</td>
          <td>18.044524</td>
          <td>26.031037</td>
          <td>19.818177</td>
          <td>27.200344</td>
          <td>22.951600</td>
          <td>0.061163</td>
          <td>17.532463</td>
          <td>0.005012</td>
          <td>22.390444</td>
          <td>0.033944</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.625391</td>
          <td>21.299536</td>
          <td>19.918275</td>
          <td>31.306514</td>
          <td>25.269362</td>
          <td>24.837621</td>
          <td>20.702014</td>
          <td>0.009314</td>
          <td>19.694155</td>
          <td>0.005630</td>
          <td>23.088173</td>
          <td>0.063185</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.524046</td>
          <td>21.107468</td>
          <td>22.115093</td>
          <td>24.826826</td>
          <td>23.772155</td>
          <td>22.215274</td>
          <td>21.569997</td>
          <td>0.018118</td>
          <td>22.509010</td>
          <td>0.034508</td>
          <td>17.933960</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.873134</td>
          <td>14.617455</td>
          <td>28.300904</td>
          <td>18.334292</td>
          <td>23.514198</td>
          <td>28.368738</td>
          <td>21.923552</td>
          <td>0.024563</td>
          <td>20.678087</td>
          <td>0.008119</td>
          <td>19.920101</td>
          <td>0.006099</td>
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


