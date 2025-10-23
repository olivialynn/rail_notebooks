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
          <td>23.472829</td>
          <td>19.236707</td>
          <td>19.117777</td>
          <td>27.879279</td>
          <td>21.666091</td>
          <td>25.088738</td>
          <td>19.222138</td>
          <td>24.078075</td>
          <td>23.079513</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.967769</td>
          <td>21.707153</td>
          <td>18.835818</td>
          <td>29.176943</td>
          <td>22.644328</td>
          <td>21.996322</td>
          <td>26.459471</td>
          <td>26.914500</td>
          <td>18.475356</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.321756</td>
          <td>24.140461</td>
          <td>24.790944</td>
          <td>20.018251</td>
          <td>19.186283</td>
          <td>25.544792</td>
          <td>23.402284</td>
          <td>15.065526</td>
          <td>20.649671</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.279812</td>
          <td>21.420266</td>
          <td>21.598902</td>
          <td>30.129227</td>
          <td>13.412878</td>
          <td>23.239565</td>
          <td>22.280727</td>
          <td>20.850829</td>
          <td>22.808656</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.534237</td>
          <td>24.464611</td>
          <td>29.819416</td>
          <td>21.872950</td>
          <td>23.814487</td>
          <td>20.231337</td>
          <td>18.129247</td>
          <td>18.291235</td>
          <td>24.197230</td>
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
          <td>21.363238</td>
          <td>28.269660</td>
          <td>24.218484</td>
          <td>24.266430</td>
          <td>25.447907</td>
          <td>29.081532</td>
          <td>16.322228</td>
          <td>22.495834</td>
          <td>26.215243</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.295015</td>
          <td>16.692167</td>
          <td>22.005134</td>
          <td>23.674001</td>
          <td>19.629577</td>
          <td>21.430592</td>
          <td>25.417681</td>
          <td>21.605351</td>
          <td>23.895812</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.778413</td>
          <td>22.933828</td>
          <td>22.997520</td>
          <td>20.618150</td>
          <td>25.778255</td>
          <td>17.317788</td>
          <td>23.993781</td>
          <td>21.128471</td>
          <td>25.424862</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.608095</td>
          <td>23.880468</td>
          <td>22.334864</td>
          <td>27.768519</td>
          <td>23.522006</td>
          <td>20.162051</td>
          <td>21.673806</td>
          <td>19.932528</td>
          <td>27.279328</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.288816</td>
          <td>19.904284</td>
          <td>28.650437</td>
          <td>30.023330</td>
          <td>21.609836</td>
          <td>28.476982</td>
          <td>21.873155</td>
          <td>23.149625</td>
          <td>25.370202</td>
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
          <td>23.455454</td>
          <td>0.028028</td>
          <td>19.230679</td>
          <td>0.005015</td>
          <td>19.121515</td>
          <td>0.005008</td>
          <td>27.329287</td>
          <td>0.384538</td>
          <td>21.660754</td>
          <td>0.007129</td>
          <td>24.907144</td>
          <td>0.206292</td>
          <td>19.222138</td>
          <td>24.078075</td>
          <td>23.079513</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.964257</td>
          <td>0.005010</td>
          <td>21.711612</td>
          <td>0.005419</td>
          <td>18.842146</td>
          <td>0.005006</td>
          <td>29.133567</td>
          <td>1.275803</td>
          <td>22.666986</td>
          <td>0.013453</td>
          <td>22.013041</td>
          <td>0.016637</td>
          <td>26.459471</td>
          <td>26.914500</td>
          <td>18.475356</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.314490</td>
          <td>0.011312</td>
          <td>24.116656</td>
          <td>0.017326</td>
          <td>24.807757</td>
          <td>0.027463</td>
          <td>20.017662</td>
          <td>0.005051</td>
          <td>19.190000</td>
          <td>0.005046</td>
          <td>26.087298</td>
          <td>0.523762</td>
          <td>23.402284</td>
          <td>15.065526</td>
          <td>20.649671</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.644594</td>
          <td>0.851193</td>
          <td>21.421404</td>
          <td>0.005270</td>
          <td>21.593895</td>
          <td>0.005248</td>
          <td>28.570006</td>
          <td>0.919352</td>
          <td>13.413925</td>
          <td>0.005000</td>
          <td>23.198732</td>
          <td>0.046514</td>
          <td>22.280727</td>
          <td>20.850829</td>
          <td>22.808656</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.538653</td>
          <td>0.007284</td>
          <td>24.507952</td>
          <td>0.024099</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.869543</td>
          <td>0.005937</td>
          <td>23.771972</td>
          <td>0.034237</td>
          <td>20.223930</td>
          <td>0.005955</td>
          <td>18.129247</td>
          <td>18.291235</td>
          <td>24.197230</td>
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
          <td>21.355166</td>
          <td>0.006772</td>
          <td>28.862856</td>
          <td>0.854063</td>
          <td>24.224389</td>
          <td>0.016704</td>
          <td>24.253427</td>
          <td>0.027442</td>
          <td>25.436795</td>
          <td>0.148205</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.322228</td>
          <td>22.495834</td>
          <td>26.215243</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.320316</td>
          <td>0.024969</td>
          <td>16.701579</td>
          <td>0.005001</td>
          <td>21.998576</td>
          <td>0.005475</td>
          <td>23.670656</td>
          <td>0.016699</td>
          <td>19.626531</td>
          <td>0.005087</td>
          <td>21.444154</td>
          <td>0.010741</td>
          <td>25.417681</td>
          <td>21.605351</td>
          <td>23.895812</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.838863</td>
          <td>0.039097</td>
          <td>22.936510</td>
          <td>0.007679</td>
          <td>22.996756</td>
          <td>0.007286</td>
          <td>20.618057</td>
          <td>0.005127</td>
          <td>25.711150</td>
          <td>0.187233</td>
          <td>17.323097</td>
          <td>0.005013</td>
          <td>23.993781</td>
          <td>21.128471</td>
          <td>25.424862</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.605582</td>
          <td>0.076695</td>
          <td>23.881665</td>
          <td>0.014340</td>
          <td>22.324011</td>
          <td>0.005802</td>
          <td>29.198864</td>
          <td>1.321348</td>
          <td>23.506516</td>
          <td>0.027119</td>
          <td>20.156055</td>
          <td>0.005857</td>
          <td>21.673806</td>
          <td>19.932528</td>
          <td>27.279328</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.319181</td>
          <td>0.059620</td>
          <td>19.913114</td>
          <td>0.005034</td>
          <td>27.975507</td>
          <td>0.412984</td>
          <td>27.520483</td>
          <td>0.445134</td>
          <td>21.605388</td>
          <td>0.006958</td>
          <td>26.121735</td>
          <td>0.537066</td>
          <td>21.873155</td>
          <td>23.149625</td>
          <td>25.370202</td>
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
          <td>23.472829</td>
          <td>19.236707</td>
          <td>19.117777</td>
          <td>27.879279</td>
          <td>21.666091</td>
          <td>25.088738</td>
          <td>19.221413</td>
          <td>0.005003</td>
          <td>24.107620</td>
          <td>0.026440</td>
          <td>23.063117</td>
          <td>0.011169</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.967769</td>
          <td>21.707153</td>
          <td>18.835818</td>
          <td>29.176943</td>
          <td>22.644328</td>
          <td>21.996322</td>
          <td>26.307956</td>
          <td>0.109225</td>
          <td>26.688272</td>
          <td>0.251667</td>
          <td>18.479791</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.321756</td>
          <td>24.140461</td>
          <td>24.790944</td>
          <td>20.018251</td>
          <td>19.186283</td>
          <td>25.544792</td>
          <td>23.407945</td>
          <td>0.009351</td>
          <td>15.067230</td>
          <td>0.005000</td>
          <td>20.645215</td>
          <td>0.005115</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.279812</td>
          <td>21.420266</td>
          <td>21.598902</td>
          <td>30.129227</td>
          <td>13.412878</td>
          <td>23.239565</td>
          <td>22.289268</td>
          <td>0.005742</td>
          <td>20.849390</td>
          <td>0.005167</td>
          <td>22.808456</td>
          <td>0.009354</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.534237</td>
          <td>24.464611</td>
          <td>29.819416</td>
          <td>21.872950</td>
          <td>23.814487</td>
          <td>20.231337</td>
          <td>18.137990</td>
          <td>0.005000</td>
          <td>18.294496</td>
          <td>0.005002</td>
          <td>24.192295</td>
          <td>0.028485</td>
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
          <td>21.363238</td>
          <td>28.269660</td>
          <td>24.218484</td>
          <td>24.266430</td>
          <td>25.447907</td>
          <td>29.081532</td>
          <td>16.314049</td>
          <td>0.005000</td>
          <td>22.498426</td>
          <td>0.007768</td>
          <td>26.116154</td>
          <td>0.155570</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.295015</td>
          <td>16.692167</td>
          <td>22.005134</td>
          <td>23.674001</td>
          <td>19.629577</td>
          <td>21.430592</td>
          <td>25.397741</td>
          <td>0.048782</td>
          <td>21.606052</td>
          <td>0.005643</td>
          <td>23.913867</td>
          <td>0.022326</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.778413</td>
          <td>22.933828</td>
          <td>22.997520</td>
          <td>20.618150</td>
          <td>25.778255</td>
          <td>17.317788</td>
          <td>23.990336</td>
          <td>0.014379</td>
          <td>21.129729</td>
          <td>0.005277</td>
          <td>25.607104</td>
          <td>0.099987</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.608095</td>
          <td>23.880468</td>
          <td>22.334864</td>
          <td>27.768519</td>
          <td>23.522006</td>
          <td>20.162051</td>
          <td>21.670148</td>
          <td>0.005249</td>
          <td>19.927997</td>
          <td>0.005031</td>
          <td>27.918040</td>
          <td>0.643937</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.288816</td>
          <td>19.904284</td>
          <td>28.650437</td>
          <td>30.023330</td>
          <td>21.609836</td>
          <td>28.476982</td>
          <td>21.866919</td>
          <td>0.005354</td>
          <td>23.140050</td>
          <td>0.011827</td>
          <td>25.257586</td>
          <td>0.073445</td>
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
          <td>23.472829</td>
          <td>19.236707</td>
          <td>19.117777</td>
          <td>27.879279</td>
          <td>21.666091</td>
          <td>25.088738</td>
          <td>19.225172</td>
          <td>0.005392</td>
          <td>23.991978</td>
          <td>0.128217</td>
          <td>23.097054</td>
          <td>0.063687</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.967769</td>
          <td>21.707153</td>
          <td>18.835818</td>
          <td>29.176943</td>
          <td>22.644328</td>
          <td>21.996322</td>
          <td>26.616206</td>
          <td>1.073795</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.468046</td>
          <td>0.005083</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.321756</td>
          <td>24.140461</td>
          <td>24.790944</td>
          <td>20.018251</td>
          <td>19.186283</td>
          <td>25.544792</td>
          <td>23.483964</td>
          <td>0.097975</td>
          <td>15.059601</td>
          <td>0.005000</td>
          <td>20.641033</td>
          <td>0.008423</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.279812</td>
          <td>21.420266</td>
          <td>21.598902</td>
          <td>30.129227</td>
          <td>13.412878</td>
          <td>23.239565</td>
          <td>22.255573</td>
          <td>0.032910</td>
          <td>20.846995</td>
          <td>0.008990</td>
          <td>22.834101</td>
          <td>0.050389</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.534237</td>
          <td>24.464611</td>
          <td>29.819416</td>
          <td>21.872950</td>
          <td>23.814487</td>
          <td>20.231337</td>
          <td>18.125178</td>
          <td>0.005053</td>
          <td>18.299902</td>
          <td>0.005051</td>
          <td>24.315382</td>
          <td>0.184353</td>
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
          <td>21.363238</td>
          <td>28.269660</td>
          <td>24.218484</td>
          <td>24.266430</td>
          <td>25.447907</td>
          <td>29.081532</td>
          <td>16.326538</td>
          <td>0.005002</td>
          <td>22.442327</td>
          <td>0.032526</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.295015</td>
          <td>16.692167</td>
          <td>22.005134</td>
          <td>23.674001</td>
          <td>19.629577</td>
          <td>21.430592</td>
          <td>24.567150</td>
          <td>0.247332</td>
          <td>21.570064</td>
          <td>0.015341</td>
          <td>23.759211</td>
          <td>0.114227</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.778413</td>
          <td>22.933828</td>
          <td>22.997520</td>
          <td>20.618150</td>
          <td>25.778255</td>
          <td>17.317788</td>
          <td>24.201501</td>
          <td>0.182198</td>
          <td>21.115775</td>
          <td>0.010791</td>
          <td>25.015016</td>
          <td>0.327787</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.608095</td>
          <td>23.880468</td>
          <td>22.334864</td>
          <td>27.768519</td>
          <td>23.522006</td>
          <td>20.162051</td>
          <td>21.664844</td>
          <td>0.019639</td>
          <td>19.931241</td>
          <td>0.005946</td>
          <td>26.389090</td>
          <td>0.880121</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.288816</td>
          <td>19.904284</td>
          <td>28.650437</td>
          <td>30.023330</td>
          <td>21.609836</td>
          <td>28.476982</td>
          <td>21.892040</td>
          <td>0.023896</td>
          <td>23.095777</td>
          <td>0.058198</td>
          <td>25.904213</td>
          <td>0.637773</td>
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


