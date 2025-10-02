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
          <td>23.043484</td>
          <td>19.246179</td>
          <td>16.955689</td>
          <td>26.660080</td>
          <td>22.284531</td>
          <td>23.432347</td>
          <td>22.962544</td>
          <td>26.626692</td>
          <td>21.395465</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.158968</td>
          <td>18.836393</td>
          <td>19.888331</td>
          <td>21.525638</td>
          <td>22.303448</td>
          <td>28.324786</td>
          <td>26.479193</td>
          <td>21.369508</td>
          <td>22.406830</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.039144</td>
          <td>22.008170</td>
          <td>22.928224</td>
          <td>20.362448</td>
          <td>23.688640</td>
          <td>26.604621</td>
          <td>20.958302</td>
          <td>21.916676</td>
          <td>26.728050</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.214036</td>
          <td>24.627427</td>
          <td>19.436881</td>
          <td>24.791139</td>
          <td>21.485036</td>
          <td>24.111197</td>
          <td>25.955891</td>
          <td>20.041548</td>
          <td>23.084949</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.166828</td>
          <td>22.224627</td>
          <td>25.923978</td>
          <td>20.965687</td>
          <td>22.588274</td>
          <td>26.310902</td>
          <td>17.892907</td>
          <td>23.088922</td>
          <td>19.536793</td>
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
          <td>24.929119</td>
          <td>17.578318</td>
          <td>21.593462</td>
          <td>21.984312</td>
          <td>25.353628</td>
          <td>22.136997</td>
          <td>25.712944</td>
          <td>18.445411</td>
          <td>19.118571</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.514613</td>
          <td>22.772730</td>
          <td>24.045861</td>
          <td>24.585944</td>
          <td>20.814703</td>
          <td>24.878201</td>
          <td>23.286004</td>
          <td>22.166352</td>
          <td>19.260530</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.116328</td>
          <td>26.851097</td>
          <td>18.167302</td>
          <td>19.943209</td>
          <td>20.478489</td>
          <td>20.334514</td>
          <td>21.187214</td>
          <td>20.640894</td>
          <td>23.380731</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.085499</td>
          <td>21.980089</td>
          <td>20.593995</td>
          <td>21.931483</td>
          <td>24.516719</td>
          <td>23.052326</td>
          <td>24.233400</td>
          <td>22.462236</td>
          <td>24.864912</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.219295</td>
          <td>26.907583</td>
          <td>22.676603</td>
          <td>23.104068</td>
          <td>23.598750</td>
          <td>26.083193</td>
          <td>20.949257</td>
          <td>23.583193</td>
          <td>25.476632</td>
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
          <td>23.053006</td>
          <td>0.019951</td>
          <td>19.246587</td>
          <td>0.005016</td>
          <td>16.955822</td>
          <td>0.005001</td>
          <td>26.864654</td>
          <td>0.265582</td>
          <td>22.289844</td>
          <td>0.010206</td>
          <td>23.506474</td>
          <td>0.061123</td>
          <td>22.962544</td>
          <td>26.626692</td>
          <td>21.395465</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.465341</td>
          <td>0.161655</td>
          <td>18.838934</td>
          <td>0.005010</td>
          <td>19.891854</td>
          <td>0.005020</td>
          <td>21.528641</td>
          <td>0.005542</td>
          <td>22.302862</td>
          <td>0.010297</td>
          <td>27.201580</td>
          <td>1.090299</td>
          <td>26.479193</td>
          <td>21.369508</td>
          <td>22.406830</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.034951</td>
          <td>0.009431</td>
          <td>22.008896</td>
          <td>0.005662</td>
          <td>22.923083</td>
          <td>0.007047</td>
          <td>20.355340</td>
          <td>0.005085</td>
          <td>23.639365</td>
          <td>0.030464</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.958302</td>
          <td>21.916676</td>
          <td>26.728050</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.200986</td>
          <td>0.022572</td>
          <td>24.597275</td>
          <td>0.026030</td>
          <td>19.433947</td>
          <td>0.005011</td>
          <td>24.758247</td>
          <td>0.042832</td>
          <td>21.482475</td>
          <td>0.006623</td>
          <td>24.014299</td>
          <td>0.095706</td>
          <td>25.955891</td>
          <td>20.041548</td>
          <td>23.084949</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.165685</td>
          <td>0.006361</td>
          <td>22.222728</td>
          <td>0.005920</td>
          <td>25.963819</td>
          <td>0.076367</td>
          <td>20.961058</td>
          <td>0.005217</td>
          <td>22.559268</td>
          <td>0.012388</td>
          <td>25.568964</td>
          <td>0.353399</td>
          <td>17.892907</td>
          <td>23.088922</td>
          <td>19.536793</td>
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
          <td>25.014212</td>
          <td>0.109619</td>
          <td>17.574802</td>
          <td>0.005003</td>
          <td>21.596714</td>
          <td>0.005249</td>
          <td>21.982234</td>
          <td>0.006121</td>
          <td>25.553149</td>
          <td>0.163728</td>
          <td>22.119476</td>
          <td>0.018170</td>
          <td>25.712944</td>
          <td>18.445411</td>
          <td>19.118571</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.051433</td>
          <td>0.263697</td>
          <td>22.760504</td>
          <td>0.007074</td>
          <td>24.065684</td>
          <td>0.014688</td>
          <td>24.615823</td>
          <td>0.037753</td>
          <td>20.811120</td>
          <td>0.005560</td>
          <td>25.037406</td>
          <td>0.229948</td>
          <td>23.286004</td>
          <td>22.166352</td>
          <td>19.260530</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.083821</td>
          <td>0.116447</td>
          <td>26.965081</td>
          <td>0.205674</td>
          <td>18.167217</td>
          <td>0.005003</td>
          <td>19.942535</td>
          <td>0.005046</td>
          <td>20.474086</td>
          <td>0.005326</td>
          <td>20.333316</td>
          <td>0.006135</td>
          <td>21.187214</td>
          <td>20.640894</td>
          <td>23.380731</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.095088</td>
          <td>0.009787</td>
          <td>21.968984</td>
          <td>0.005622</td>
          <td>20.594314</td>
          <td>0.005054</td>
          <td>21.936311</td>
          <td>0.006043</td>
          <td>24.509666</td>
          <td>0.065843</td>
          <td>23.069698</td>
          <td>0.041483</td>
          <td>24.233400</td>
          <td>22.462236</td>
          <td>24.864912</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.205892</td>
          <td>0.006440</td>
          <td>27.013008</td>
          <td>0.214081</td>
          <td>22.687549</td>
          <td>0.006426</td>
          <td>23.088388</td>
          <td>0.010669</td>
          <td>23.591750</td>
          <td>0.029218</td>
          <td>26.791319</td>
          <td>0.848873</td>
          <td>20.949257</td>
          <td>23.583193</td>
          <td>25.476632</td>
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
          <td>23.043484</td>
          <td>19.246179</td>
          <td>16.955689</td>
          <td>26.660080</td>
          <td>22.284531</td>
          <td>23.432347</td>
          <td>22.970037</td>
          <td>0.007274</td>
          <td>26.511030</td>
          <td>0.217307</td>
          <td>21.389182</td>
          <td>0.005440</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.158968</td>
          <td>18.836393</td>
          <td>19.888331</td>
          <td>21.525638</td>
          <td>22.303448</td>
          <td>28.324786</td>
          <td>26.559392</td>
          <td>0.135926</td>
          <td>21.379466</td>
          <td>0.005432</td>
          <td>22.393263</td>
          <td>0.007357</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.039144</td>
          <td>22.008170</td>
          <td>22.928224</td>
          <td>20.362448</td>
          <td>23.688640</td>
          <td>26.604621</td>
          <td>20.949008</td>
          <td>0.005067</td>
          <td>21.918356</td>
          <td>0.006096</td>
          <td>27.324103</td>
          <td>0.417171</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.214036</td>
          <td>24.627427</td>
          <td>19.436881</td>
          <td>24.791139</td>
          <td>21.485036</td>
          <td>24.111197</td>
          <td>26.019681</td>
          <td>0.084774</td>
          <td>20.041713</td>
          <td>0.005038</td>
          <td>23.102812</td>
          <td>0.011502</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.166828</td>
          <td>22.224627</td>
          <td>25.923978</td>
          <td>20.965687</td>
          <td>22.588274</td>
          <td>26.310902</td>
          <td>17.890448</td>
          <td>0.005000</td>
          <td>23.074617</td>
          <td>0.011264</td>
          <td>19.529190</td>
          <td>0.005015</td>
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
          <td>24.929119</td>
          <td>17.578318</td>
          <td>21.593462</td>
          <td>21.984312</td>
          <td>25.353628</td>
          <td>22.136997</td>
          <td>25.834342</td>
          <td>0.071947</td>
          <td>18.446283</td>
          <td>0.005002</td>
          <td>19.126090</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.514613</td>
          <td>22.772730</td>
          <td>24.045861</td>
          <td>24.585944</td>
          <td>20.814703</td>
          <td>24.878201</td>
          <td>23.282564</td>
          <td>0.008636</td>
          <td>22.160424</td>
          <td>0.006632</td>
          <td>19.264438</td>
          <td>0.005009</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.116328</td>
          <td>26.851097</td>
          <td>18.167302</td>
          <td>19.943209</td>
          <td>20.478489</td>
          <td>20.334514</td>
          <td>21.188221</td>
          <td>0.005104</td>
          <td>20.641796</td>
          <td>0.005115</td>
          <td>23.406833</td>
          <td>0.014572</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.085499</td>
          <td>21.980089</td>
          <td>20.593995</td>
          <td>21.931483</td>
          <td>24.516719</td>
          <td>23.052326</td>
          <td>24.232343</td>
          <td>0.017552</td>
          <td>22.451224</td>
          <td>0.007577</td>
          <td>24.885257</td>
          <td>0.052740</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.219295</td>
          <td>26.907583</td>
          <td>22.676603</td>
          <td>23.104068</td>
          <td>23.598750</td>
          <td>26.083193</td>
          <td>20.955001</td>
          <td>0.005068</td>
          <td>23.599400</td>
          <td>0.017074</td>
          <td>25.402582</td>
          <td>0.083503</td>
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
          <td>23.043484</td>
          <td>19.246179</td>
          <td>16.955689</td>
          <td>26.660080</td>
          <td>22.284531</td>
          <td>23.432347</td>
          <td>22.998147</td>
          <td>0.063748</td>
          <td>25.422982</td>
          <td>0.416814</td>
          <td>21.387922</td>
          <td>0.014351</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.158968</td>
          <td>18.836393</td>
          <td>19.888331</td>
          <td>21.525638</td>
          <td>22.303448</td>
          <td>28.324786</td>
          <td>25.624686</td>
          <td>0.561662</td>
          <td>21.370635</td>
          <td>0.013075</td>
          <td>22.425976</td>
          <td>0.035032</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.039144</td>
          <td>22.008170</td>
          <td>22.928224</td>
          <td>20.362448</td>
          <td>23.688640</td>
          <td>26.604621</td>
          <td>20.971241</td>
          <td>0.011236</td>
          <td>21.942531</td>
          <td>0.020992</td>
          <td>26.475581</td>
          <td>0.929008</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.214036</td>
          <td>24.627427</td>
          <td>19.436881</td>
          <td>24.791139</td>
          <td>21.485036</td>
          <td>24.111197</td>
          <td>25.970271</td>
          <td>0.714690</td>
          <td>20.034484</td>
          <td>0.006126</td>
          <td>23.016543</td>
          <td>0.059284</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.166828</td>
          <td>22.224627</td>
          <td>25.923978</td>
          <td>20.965687</td>
          <td>22.588274</td>
          <td>26.310902</td>
          <td>17.888736</td>
          <td>0.005035</td>
          <td>23.111850</td>
          <td>0.059036</td>
          <td>19.537335</td>
          <td>0.005570</td>
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
          <td>24.929119</td>
          <td>17.578318</td>
          <td>21.593462</td>
          <td>21.984312</td>
          <td>25.353628</td>
          <td>22.136997</td>
          <td>25.486014</td>
          <td>0.507771</td>
          <td>18.443349</td>
          <td>0.005066</td>
          <td>19.120374</td>
          <td>0.005272</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.514613</td>
          <td>22.772730</td>
          <td>24.045861</td>
          <td>24.585944</td>
          <td>20.814703</td>
          <td>24.878201</td>
          <td>23.287581</td>
          <td>0.082404</td>
          <td>22.152292</td>
          <td>0.025188</td>
          <td>19.254376</td>
          <td>0.005346</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.116328</td>
          <td>26.851097</td>
          <td>18.167302</td>
          <td>19.943209</td>
          <td>20.478489</td>
          <td>20.334514</td>
          <td>21.213076</td>
          <td>0.013519</td>
          <td>20.653851</td>
          <td>0.008009</td>
          <td>23.396802</td>
          <td>0.083078</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.085499</td>
          <td>21.980089</td>
          <td>20.593995</td>
          <td>21.931483</td>
          <td>24.516719</td>
          <td>23.052326</td>
          <td>24.446841</td>
          <td>0.223889</td>
          <td>22.515504</td>
          <td>0.034708</td>
          <td>24.833049</td>
          <td>0.283236</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.219295</td>
          <td>26.907583</td>
          <td>22.676603</td>
          <td>23.104068</td>
          <td>23.598750</td>
          <td>26.083193</td>
          <td>20.947815</td>
          <td>0.011045</td>
          <td>23.474132</td>
          <td>0.081430</td>
          <td>25.243332</td>
          <td>0.392052</td>
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


