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
          <td>21.004853</td>
          <td>19.923727</td>
          <td>20.442098</td>
          <td>25.726012</td>
          <td>18.546038</td>
          <td>24.302033</td>
          <td>27.717633</td>
          <td>21.612491</td>
          <td>25.542649</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.440166</td>
          <td>21.124959</td>
          <td>19.554122</td>
          <td>16.848037</td>
          <td>21.387260</td>
          <td>21.704545</td>
          <td>21.653207</td>
          <td>24.375714</td>
          <td>15.862351</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.138158</td>
          <td>18.812575</td>
          <td>19.573578</td>
          <td>23.297329</td>
          <td>20.684827</td>
          <td>19.982900</td>
          <td>28.310457</td>
          <td>27.318786</td>
          <td>21.385452</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.476900</td>
          <td>20.845937</td>
          <td>27.571857</td>
          <td>19.494139</td>
          <td>23.092057</td>
          <td>20.330757</td>
          <td>19.743059</td>
          <td>23.446602</td>
          <td>19.430107</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.296692</td>
          <td>24.634313</td>
          <td>20.259377</td>
          <td>23.241553</td>
          <td>18.773855</td>
          <td>22.758855</td>
          <td>22.751277</td>
          <td>26.387978</td>
          <td>21.439515</td>
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
          <td>22.727749</td>
          <td>26.135534</td>
          <td>23.821783</td>
          <td>20.839956</td>
          <td>24.220425</td>
          <td>28.205828</td>
          <td>23.714995</td>
          <td>26.132177</td>
          <td>23.344048</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.307919</td>
          <td>20.425481</td>
          <td>23.498220</td>
          <td>26.312769</td>
          <td>29.406904</td>
          <td>21.187601</td>
          <td>27.150157</td>
          <td>25.278898</td>
          <td>19.917014</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.695625</td>
          <td>24.390445</td>
          <td>23.668564</td>
          <td>19.614965</td>
          <td>19.262112</td>
          <td>23.220333</td>
          <td>20.386422</td>
          <td>17.044609</td>
          <td>27.950696</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.204231</td>
          <td>20.183395</td>
          <td>20.776553</td>
          <td>23.298163</td>
          <td>24.162619</td>
          <td>23.575761</td>
          <td>26.553273</td>
          <td>21.874103</td>
          <td>25.276261</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.848251</td>
          <td>23.019471</td>
          <td>22.660215</td>
          <td>29.382305</td>
          <td>22.706707</td>
          <td>24.554919</td>
          <td>25.873725</td>
          <td>27.346237</td>
          <td>25.215631</td>
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
          <td>20.997308</td>
          <td>0.006075</td>
          <td>19.924743</td>
          <td>0.005035</td>
          <td>20.434338</td>
          <td>0.005043</td>
          <td>25.837185</td>
          <td>0.111119</td>
          <td>18.538677</td>
          <td>0.005019</td>
          <td>24.221807</td>
          <td>0.114746</td>
          <td>27.717633</td>
          <td>21.612491</td>
          <td>25.542649</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.439684</td>
          <td>0.005044</td>
          <td>21.129860</td>
          <td>0.005175</td>
          <td>19.559726</td>
          <td>0.005013</td>
          <td>16.854752</td>
          <td>0.005001</td>
          <td>21.387585</td>
          <td>0.006401</td>
          <td>21.719305</td>
          <td>0.013161</td>
          <td>21.653207</td>
          <td>24.375714</td>
          <td>15.862351</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.118138</td>
          <td>0.049954</td>
          <td>18.816248</td>
          <td>0.005010</td>
          <td>19.572857</td>
          <td>0.005014</td>
          <td>23.295319</td>
          <td>0.012406</td>
          <td>20.681880</td>
          <td>0.005455</td>
          <td>19.980028</td>
          <td>0.005647</td>
          <td>28.310457</td>
          <td>27.318786</td>
          <td>21.385452</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.478140</td>
          <td>0.005144</td>
          <td>20.850734</td>
          <td>0.005117</td>
          <td>28.401176</td>
          <td>0.566427</td>
          <td>19.498736</td>
          <td>0.005025</td>
          <td>23.093834</td>
          <td>0.019027</td>
          <td>20.331709</td>
          <td>0.006132</td>
          <td>19.743059</td>
          <td>23.446602</td>
          <td>19.430107</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.323591</td>
          <td>0.025039</td>
          <td>24.628092</td>
          <td>0.026735</td>
          <td>20.264648</td>
          <td>0.005034</td>
          <td>23.274486</td>
          <td>0.012214</td>
          <td>18.774933</td>
          <td>0.005026</td>
          <td>22.730112</td>
          <td>0.030735</td>
          <td>22.751277</td>
          <td>26.387978</td>
          <td>21.439515</td>
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
          <td>22.748992</td>
          <td>0.015611</td>
          <td>26.018659</td>
          <td>0.091083</td>
          <td>23.822969</td>
          <td>0.012170</td>
          <td>20.834439</td>
          <td>0.005178</td>
          <td>24.264885</td>
          <td>0.052991</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.714995</td>
          <td>26.132177</td>
          <td>23.344048</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.200893</td>
          <td>0.297623</td>
          <td>20.422287</td>
          <td>0.005065</td>
          <td>23.511218</td>
          <td>0.009762</td>
          <td>26.727404</td>
          <td>0.237266</td>
          <td>28.161298</td>
          <td>1.116118</td>
          <td>21.181844</td>
          <td>0.009027</td>
          <td>27.150157</td>
          <td>25.278898</td>
          <td>19.917014</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.601009</td>
          <td>0.076388</td>
          <td>24.427086</td>
          <td>0.022484</td>
          <td>23.670590</td>
          <td>0.010890</td>
          <td>19.620784</td>
          <td>0.005029</td>
          <td>19.266890</td>
          <td>0.005052</td>
          <td>23.217724</td>
          <td>0.047305</td>
          <td>20.386422</td>
          <td>17.044609</td>
          <td>27.950696</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.208855</td>
          <td>0.005364</td>
          <td>20.196198</td>
          <td>0.005049</td>
          <td>20.780213</td>
          <td>0.005070</td>
          <td>23.287946</td>
          <td>0.012338</td>
          <td>24.236654</td>
          <td>0.051680</td>
          <td>23.579412</td>
          <td>0.065205</td>
          <td>26.553273</td>
          <td>21.874103</td>
          <td>25.276261</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.846723</td>
          <td>0.005871</td>
          <td>23.026462</td>
          <td>0.008046</td>
          <td>22.646397</td>
          <td>0.006337</td>
          <td>28.173313</td>
          <td>0.710760</td>
          <td>22.709187</td>
          <td>0.013903</td>
          <td>24.668315</td>
          <td>0.168603</td>
          <td>25.873725</td>
          <td>27.346237</td>
          <td>25.215631</td>
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
          <td>21.004853</td>
          <td>19.923727</td>
          <td>20.442098</td>
          <td>25.726012</td>
          <td>18.546038</td>
          <td>24.302033</td>
          <td>28.308680</td>
          <td>0.555226</td>
          <td>21.618096</td>
          <td>0.005656</td>
          <td>25.427912</td>
          <td>0.085393</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.440166</td>
          <td>21.124959</td>
          <td>19.554122</td>
          <td>16.848037</td>
          <td>21.387260</td>
          <td>21.704545</td>
          <td>21.656011</td>
          <td>0.005243</td>
          <td>24.374807</td>
          <td>0.033476</td>
          <td>15.868841</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.138158</td>
          <td>18.812575</td>
          <td>19.573578</td>
          <td>23.297329</td>
          <td>20.684827</td>
          <td>19.982900</td>
          <td>28.419164</td>
          <td>0.600812</td>
          <td>27.420245</td>
          <td>0.448774</td>
          <td>21.389738</td>
          <td>0.005440</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.476900</td>
          <td>20.845937</td>
          <td>27.571857</td>
          <td>19.494139</td>
          <td>23.092057</td>
          <td>20.330757</td>
          <td>19.742239</td>
          <td>0.005007</td>
          <td>23.464029</td>
          <td>0.015265</td>
          <td>19.433614</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.296692</td>
          <td>24.634313</td>
          <td>20.259377</td>
          <td>23.241553</td>
          <td>18.773855</td>
          <td>22.758855</td>
          <td>22.746821</td>
          <td>0.006596</td>
          <td>26.297747</td>
          <td>0.181620</td>
          <td>21.434404</td>
          <td>0.005476</td>
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
          <td>22.727749</td>
          <td>26.135534</td>
          <td>23.821783</td>
          <td>20.839956</td>
          <td>24.220425</td>
          <td>28.205828</td>
          <td>23.715007</td>
          <td>0.011607</td>
          <td>26.244554</td>
          <td>0.173599</td>
          <td>23.349747</td>
          <td>0.013919</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.307919</td>
          <td>20.425481</td>
          <td>23.498220</td>
          <td>26.312769</td>
          <td>29.406904</td>
          <td>21.187601</td>
          <td>26.933283</td>
          <td>0.187166</td>
          <td>25.301300</td>
          <td>0.076346</td>
          <td>19.916873</td>
          <td>0.005030</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.695625</td>
          <td>24.390445</td>
          <td>23.668564</td>
          <td>19.614965</td>
          <td>19.262112</td>
          <td>23.220333</td>
          <td>20.390005</td>
          <td>0.005024</td>
          <td>17.043610</td>
          <td>0.005000</td>
          <td>26.891945</td>
          <td>0.297042</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.204231</td>
          <td>20.183395</td>
          <td>20.776553</td>
          <td>23.298163</td>
          <td>24.162619</td>
          <td>23.575761</td>
          <td>26.665388</td>
          <td>0.148935</td>
          <td>21.873234</td>
          <td>0.006016</td>
          <td>25.239832</td>
          <td>0.072298</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.848251</td>
          <td>23.019471</td>
          <td>22.660215</td>
          <td>29.382305</td>
          <td>22.706707</td>
          <td>24.554919</td>
          <td>25.978199</td>
          <td>0.081723</td>
          <td>27.402518</td>
          <td>0.442805</td>
          <td>25.256639</td>
          <td>0.073384</td>
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
          <td>21.004853</td>
          <td>19.923727</td>
          <td>20.442098</td>
          <td>25.726012</td>
          <td>18.546038</td>
          <td>24.302033</td>
          <td>27.422752</td>
          <td>1.644839</td>
          <td>21.626199</td>
          <td>0.016066</td>
          <td>25.326813</td>
          <td>0.418036</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.440166</td>
          <td>21.124959</td>
          <td>19.554122</td>
          <td>16.848037</td>
          <td>21.387260</td>
          <td>21.704545</td>
          <td>21.655113</td>
          <td>0.019476</td>
          <td>24.508743</td>
          <td>0.199463</td>
          <td>15.868375</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.138158</td>
          <td>18.812575</td>
          <td>19.573578</td>
          <td>23.297329</td>
          <td>20.684827</td>
          <td>19.982900</td>
          <td>26.128710</td>
          <td>0.793983</td>
          <td>25.882620</td>
          <td>0.585432</td>
          <td>21.386611</td>
          <td>0.014336</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.476900</td>
          <td>20.845937</td>
          <td>27.571857</td>
          <td>19.494139</td>
          <td>23.092057</td>
          <td>20.330757</td>
          <td>19.741301</td>
          <td>0.005962</td>
          <td>23.321400</td>
          <td>0.071125</td>
          <td>19.421291</td>
          <td>0.005465</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.296692</td>
          <td>24.634313</td>
          <td>20.259377</td>
          <td>23.241553</td>
          <td>18.773855</td>
          <td>22.758855</td>
          <td>22.700498</td>
          <td>0.048902</td>
          <td>26.724844</td>
          <td>1.017320</td>
          <td>21.443227</td>
          <td>0.015008</td>
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
          <td>22.727749</td>
          <td>26.135534</td>
          <td>23.821783</td>
          <td>20.839956</td>
          <td>24.220425</td>
          <td>28.205828</td>
          <td>24.005723</td>
          <td>0.154184</td>
          <td>25.823466</td>
          <td>0.561169</td>
          <td>23.412898</td>
          <td>0.084268</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.307919</td>
          <td>20.425481</td>
          <td>23.498220</td>
          <td>26.312769</td>
          <td>29.406904</td>
          <td>21.187601</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.454105</td>
          <td>0.426832</td>
          <td>19.911987</td>
          <td>0.006084</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.695625</td>
          <td>24.390445</td>
          <td>23.668564</td>
          <td>19.614965</td>
          <td>19.262112</td>
          <td>23.220333</td>
          <td>20.383567</td>
          <td>0.007707</td>
          <td>17.046667</td>
          <td>0.005005</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.204231</td>
          <td>20.183395</td>
          <td>20.776553</td>
          <td>23.298163</td>
          <td>24.162619</td>
          <td>23.575761</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.898821</td>
          <td>0.020218</td>
          <td>25.240585</td>
          <td>0.391221</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.848251</td>
          <td>23.019471</td>
          <td>22.660215</td>
          <td>29.382305</td>
          <td>22.706707</td>
          <td>24.554919</td>
          <td>28.233939</td>
          <td>2.322542</td>
          <td>26.900084</td>
          <td>1.127227</td>
          <td>24.937154</td>
          <td>0.308033</td>
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


