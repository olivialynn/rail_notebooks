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
          <td>21.631393</td>
          <td>23.736400</td>
          <td>20.712099</td>
          <td>23.053073</td>
          <td>19.785953</td>
          <td>22.634065</td>
          <td>22.274318</td>
          <td>18.145302</td>
          <td>19.789392</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.329684</td>
          <td>19.985216</td>
          <td>26.005125</td>
          <td>19.444365</td>
          <td>19.980360</td>
          <td>24.301589</td>
          <td>24.846964</td>
          <td>24.414411</td>
          <td>24.236251</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.422209</td>
          <td>23.895998</td>
          <td>25.506103</td>
          <td>20.772424</td>
          <td>23.166072</td>
          <td>21.676080</td>
          <td>23.343819</td>
          <td>24.883844</td>
          <td>24.726997</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.332571</td>
          <td>20.475037</td>
          <td>23.897817</td>
          <td>19.743265</td>
          <td>23.298570</td>
          <td>19.552486</td>
          <td>22.481440</td>
          <td>16.244140</td>
          <td>25.702754</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.619328</td>
          <td>26.542438</td>
          <td>21.132157</td>
          <td>21.495753</td>
          <td>22.598611</td>
          <td>24.238950</td>
          <td>23.095766</td>
          <td>21.883846</td>
          <td>25.686091</td>
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
          <td>18.099156</td>
          <td>23.863056</td>
          <td>24.338771</td>
          <td>26.241470</td>
          <td>22.284581</td>
          <td>22.753838</td>
          <td>24.003454</td>
          <td>19.345942</td>
          <td>19.317814</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.226159</td>
          <td>25.705258</td>
          <td>18.211915</td>
          <td>20.551878</td>
          <td>25.317549</td>
          <td>26.911212</td>
          <td>21.533544</td>
          <td>28.449987</td>
          <td>19.373751</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.772117</td>
          <td>25.457352</td>
          <td>26.549839</td>
          <td>17.606291</td>
          <td>25.103544</td>
          <td>23.116532</td>
          <td>24.979356</td>
          <td>22.671892</td>
          <td>21.481095</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.851213</td>
          <td>22.842061</td>
          <td>24.142094</td>
          <td>27.044070</td>
          <td>22.815582</td>
          <td>22.037037</td>
          <td>22.839422</td>
          <td>21.835083</td>
          <td>25.422035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.041837</td>
          <td>29.055261</td>
          <td>22.193259</td>
          <td>22.829130</td>
          <td>21.640246</td>
          <td>23.501373</td>
          <td>22.302290</td>
          <td>26.372993</td>
          <td>20.827378</td>
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
          <td>21.624170</td>
          <td>0.007567</td>
          <td>23.732382</td>
          <td>0.012782</td>
          <td>20.715513</td>
          <td>0.005064</td>
          <td>23.056956</td>
          <td>0.010438</td>
          <td>19.784579</td>
          <td>0.005111</td>
          <td>22.634602</td>
          <td>0.028266</td>
          <td>22.274318</td>
          <td>18.145302</td>
          <td>19.789392</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.767726</td>
          <td>0.919793</td>
          <td>19.983324</td>
          <td>0.005037</td>
          <td>25.987369</td>
          <td>0.077972</td>
          <td>19.448606</td>
          <td>0.005023</td>
          <td>19.974666</td>
          <td>0.005148</td>
          <td>24.424856</td>
          <td>0.136837</td>
          <td>24.846964</td>
          <td>24.414411</td>
          <td>24.236251</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.420249</td>
          <td>0.005484</td>
          <td>23.919500</td>
          <td>0.014775</td>
          <td>25.639293</td>
          <td>0.057286</td>
          <td>20.767582</td>
          <td>0.005160</td>
          <td>23.158485</td>
          <td>0.020095</td>
          <td>21.662007</td>
          <td>0.012597</td>
          <td>23.343819</td>
          <td>24.883844</td>
          <td>24.726997</td>
        </tr>
        <tr>
          <th>3</th>
          <td>inf</td>
          <td>inf</td>
          <td>20.482101</td>
          <td>0.005071</td>
          <td>23.900995</td>
          <td>0.012911</td>
          <td>19.744925</td>
          <td>0.005035</td>
          <td>23.289073</td>
          <td>0.022464</td>
          <td>19.556588</td>
          <td>0.005328</td>
          <td>22.481440</td>
          <td>16.244140</td>
          <td>25.702754</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.810486</td>
          <td>0.091780</td>
          <td>26.455033</td>
          <td>0.133222</td>
          <td>21.142723</td>
          <td>0.005122</td>
          <td>21.498211</td>
          <td>0.005516</td>
          <td>22.571548</td>
          <td>0.012503</td>
          <td>24.256895</td>
          <td>0.118304</td>
          <td>23.095766</td>
          <td>21.883846</td>
          <td>25.686091</td>
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
          <td>18.080736</td>
          <td>0.005031</td>
          <td>23.872666</td>
          <td>0.014240</td>
          <td>24.344601</td>
          <td>0.018458</td>
          <td>26.038410</td>
          <td>0.132344</td>
          <td>22.292061</td>
          <td>0.010221</td>
          <td>22.739354</td>
          <td>0.030985</td>
          <td>24.003454</td>
          <td>19.345942</td>
          <td>19.317814</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.702773</td>
          <td>0.440436</td>
          <td>25.779243</td>
          <td>0.073771</td>
          <td>18.210923</td>
          <td>0.005003</td>
          <td>20.556864</td>
          <td>0.005115</td>
          <td>25.152398</td>
          <td>0.115884</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.533544</td>
          <td>28.449987</td>
          <td>19.373751</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.771099</td>
          <td>0.005064</td>
          <td>25.531183</td>
          <td>0.059240</td>
          <td>26.656622</td>
          <td>0.139940</td>
          <td>17.616333</td>
          <td>0.005003</td>
          <td>25.034066</td>
          <td>0.104515</td>
          <td>23.104071</td>
          <td>0.042767</td>
          <td>24.979356</td>
          <td>22.671892</td>
          <td>21.481095</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.899298</td>
          <td>0.041221</td>
          <td>22.836293</td>
          <td>0.007318</td>
          <td>24.184737</td>
          <td>0.016170</td>
          <td>26.771129</td>
          <td>0.245980</td>
          <td>22.820405</td>
          <td>0.015189</td>
          <td>22.026767</td>
          <td>0.016826</td>
          <td>22.839422</td>
          <td>21.835083</td>
          <td>25.422035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.956849</td>
          <td>0.104283</td>
          <td>28.471376</td>
          <td>0.658512</td>
          <td>22.203378</td>
          <td>0.005660</td>
          <td>22.833369</td>
          <td>0.009012</td>
          <td>21.641355</td>
          <td>0.007067</td>
          <td>23.451908</td>
          <td>0.058235</td>
          <td>22.302290</td>
          <td>26.372993</td>
          <td>20.827378</td>
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
          <td>21.631393</td>
          <td>23.736400</td>
          <td>20.712099</td>
          <td>23.053073</td>
          <td>19.785953</td>
          <td>22.634065</td>
          <td>22.268914</td>
          <td>0.005717</td>
          <td>18.148628</td>
          <td>0.005001</td>
          <td>19.781492</td>
          <td>0.005024</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.329684</td>
          <td>19.985216</td>
          <td>26.005125</td>
          <td>19.444365</td>
          <td>19.980360</td>
          <td>24.301589</td>
          <td>24.859968</td>
          <td>0.030238</td>
          <td>24.404866</td>
          <td>0.034381</td>
          <td>24.203246</td>
          <td>0.028761</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.422209</td>
          <td>23.895998</td>
          <td>25.506103</td>
          <td>20.772424</td>
          <td>23.166072</td>
          <td>21.676080</td>
          <td>23.351338</td>
          <td>0.009015</td>
          <td>24.834727</td>
          <td>0.050417</td>
          <td>24.735223</td>
          <td>0.046137</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.332571</td>
          <td>20.475037</td>
          <td>23.897817</td>
          <td>19.743265</td>
          <td>23.298570</td>
          <td>19.552486</td>
          <td>22.485210</td>
          <td>0.006036</td>
          <td>16.241023</td>
          <td>0.005000</td>
          <td>25.785691</td>
          <td>0.116896</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.619328</td>
          <td>26.542438</td>
          <td>21.132157</td>
          <td>21.495753</td>
          <td>22.598611</td>
          <td>24.238950</td>
          <td>23.093586</td>
          <td>0.007748</td>
          <td>21.893119</td>
          <td>0.006050</td>
          <td>25.757022</td>
          <td>0.114009</td>
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
          <td>18.099156</td>
          <td>23.863056</td>
          <td>24.338771</td>
          <td>26.241470</td>
          <td>22.284581</td>
          <td>22.753838</td>
          <td>23.994218</td>
          <td>0.014424</td>
          <td>19.342286</td>
          <td>0.005011</td>
          <td>19.317919</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.226159</td>
          <td>25.705258</td>
          <td>18.211915</td>
          <td>20.551878</td>
          <td>25.317549</td>
          <td>26.911212</td>
          <td>21.527172</td>
          <td>0.005192</td>
          <td>27.423433</td>
          <td>0.449854</td>
          <td>19.379101</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.772117</td>
          <td>25.457352</td>
          <td>26.549839</td>
          <td>17.606291</td>
          <td>25.103544</td>
          <td>23.116532</td>
          <td>24.930021</td>
          <td>0.032173</td>
          <td>22.686587</td>
          <td>0.008658</td>
          <td>21.485798</td>
          <td>0.005521</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.851213</td>
          <td>22.842061</td>
          <td>24.142094</td>
          <td>27.044070</td>
          <td>22.815582</td>
          <td>22.037037</td>
          <td>22.839966</td>
          <td>0.006854</td>
          <td>21.837355</td>
          <td>0.005956</td>
          <td>25.423001</td>
          <td>0.085023</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.041837</td>
          <td>29.055261</td>
          <td>22.193259</td>
          <td>22.829130</td>
          <td>21.640246</td>
          <td>23.501373</td>
          <td>22.305503</td>
          <td>0.005763</td>
          <td>26.004111</td>
          <td>0.141279</td>
          <td>20.830015</td>
          <td>0.005161</td>
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
          <td>21.631393</td>
          <td>23.736400</td>
          <td>20.712099</td>
          <td>23.053073</td>
          <td>19.785953</td>
          <td>22.634065</td>
          <td>22.252617</td>
          <td>0.032824</td>
          <td>18.146154</td>
          <td>0.005039</td>
          <td>19.787357</td>
          <td>0.005878</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.329684</td>
          <td>19.985216</td>
          <td>26.005125</td>
          <td>19.444365</td>
          <td>19.980360</td>
          <td>24.301589</td>
          <td>24.242565</td>
          <td>0.188640</td>
          <td>24.293677</td>
          <td>0.166235</td>
          <td>24.471133</td>
          <td>0.210182</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.422209</td>
          <td>23.895998</td>
          <td>25.506103</td>
          <td>20.772424</td>
          <td>23.166072</td>
          <td>21.676080</td>
          <td>23.300609</td>
          <td>0.083358</td>
          <td>24.846198</td>
          <td>0.263904</td>
          <td>25.271380</td>
          <td>0.400629</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.332571</td>
          <td>20.475037</td>
          <td>23.897817</td>
          <td>19.743265</td>
          <td>23.298570</td>
          <td>19.552486</td>
          <td>22.488511</td>
          <td>0.040482</td>
          <td>16.241626</td>
          <td>0.005001</td>
          <td>25.442054</td>
          <td>0.456206</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.619328</td>
          <td>26.542438</td>
          <td>21.132157</td>
          <td>21.495753</td>
          <td>22.598611</td>
          <td>24.238950</td>
          <td>23.021976</td>
          <td>0.065114</td>
          <td>21.885648</td>
          <td>0.019991</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>18.099156</td>
          <td>23.863056</td>
          <td>24.338771</td>
          <td>26.241470</td>
          <td>22.284581</td>
          <td>22.753838</td>
          <td>24.207547</td>
          <td>0.183134</td>
          <td>19.347376</td>
          <td>0.005342</td>
          <td>19.319998</td>
          <td>0.005389</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.226159</td>
          <td>25.705258</td>
          <td>18.211915</td>
          <td>20.551878</td>
          <td>25.317549</td>
          <td>26.911212</td>
          <td>21.529642</td>
          <td>0.017512</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.386122</td>
          <td>0.005437</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.772117</td>
          <td>25.457352</td>
          <td>26.549839</td>
          <td>17.606291</td>
          <td>25.103544</td>
          <td>23.116532</td>
          <td>24.893621</td>
          <td>0.322254</td>
          <td>22.624072</td>
          <td>0.038225</td>
          <td>21.472521</td>
          <td>0.015372</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.851213</td>
          <td>22.842061</td>
          <td>24.142094</td>
          <td>27.044070</td>
          <td>22.815582</td>
          <td>22.037037</td>
          <td>22.827609</td>
          <td>0.054769</td>
          <td>21.847930</td>
          <td>0.019357</td>
          <td>25.744518</td>
          <td>0.569714</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.041837</td>
          <td>29.055261</td>
          <td>22.193259</td>
          <td>22.829130</td>
          <td>21.640246</td>
          <td>23.501373</td>
          <td>22.297220</td>
          <td>0.034149</td>
          <td>26.200469</td>
          <td>0.729357</td>
          <td>20.832065</td>
          <td>0.009501</td>
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


