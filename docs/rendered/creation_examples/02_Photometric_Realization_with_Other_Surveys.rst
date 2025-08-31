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
          <td>28.922123</td>
          <td>26.539839</td>
          <td>20.164916</td>
          <td>24.570023</td>
          <td>26.720147</td>
          <td>24.805046</td>
          <td>20.587812</td>
          <td>15.264246</td>
          <td>22.897116</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.852105</td>
          <td>27.064536</td>
          <td>29.993295</td>
          <td>25.082287</td>
          <td>24.168653</td>
          <td>23.975316</td>
          <td>19.155970</td>
          <td>25.106908</td>
          <td>23.150404</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.689272</td>
          <td>28.881117</td>
          <td>27.991957</td>
          <td>26.740200</td>
          <td>24.741627</td>
          <td>22.896606</td>
          <td>22.025185</td>
          <td>31.077576</td>
          <td>20.636644</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.717666</td>
          <td>25.970525</td>
          <td>24.472501</td>
          <td>18.898489</td>
          <td>25.323788</td>
          <td>26.712363</td>
          <td>22.056413</td>
          <td>25.334257</td>
          <td>21.012807</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.308117</td>
          <td>20.793923</td>
          <td>16.897567</td>
          <td>25.796011</td>
          <td>25.500709</td>
          <td>26.291280</td>
          <td>23.241840</td>
          <td>24.065641</td>
          <td>22.922245</td>
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
          <td>23.531546</td>
          <td>26.680287</td>
          <td>25.924933</td>
          <td>26.749033</td>
          <td>23.544405</td>
          <td>23.786986</td>
          <td>26.250758</td>
          <td>22.520159</td>
          <td>30.468451</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.608658</td>
          <td>24.012314</td>
          <td>24.630620</td>
          <td>19.513208</td>
          <td>24.412089</td>
          <td>30.398965</td>
          <td>23.172954</td>
          <td>22.045923</td>
          <td>19.948984</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.837960</td>
          <td>22.531705</td>
          <td>22.425833</td>
          <td>20.759702</td>
          <td>20.208064</td>
          <td>25.923626</td>
          <td>20.898901</td>
          <td>21.765132</td>
          <td>24.736590</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.682414</td>
          <td>26.825508</td>
          <td>28.018015</td>
          <td>22.479107</td>
          <td>25.007829</td>
          <td>19.443000</td>
          <td>25.510076</td>
          <td>25.060040</td>
          <td>26.300815</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.495554</td>
          <td>18.749613</td>
          <td>22.962109</td>
          <td>27.800403</td>
          <td>22.579442</td>
          <td>18.878141</td>
          <td>18.565963</td>
          <td>19.335020</td>
          <td>20.880703</td>
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
          <td>26.977258</td>
          <td>0.539750</td>
          <td>26.650916</td>
          <td>0.157646</td>
          <td>20.168051</td>
          <td>0.005029</td>
          <td>24.602231</td>
          <td>0.037302</td>
          <td>27.801564</td>
          <td>0.899170</td>
          <td>24.833106</td>
          <td>0.193855</td>
          <td>20.587812</td>
          <td>15.264246</td>
          <td>22.897116</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.851810</td>
          <td>0.008486</td>
          <td>26.798520</td>
          <td>0.178749</td>
          <td>30.675322</td>
          <td>2.042446</td>
          <td>25.138387</td>
          <td>0.060025</td>
          <td>24.283463</td>
          <td>0.053873</td>
          <td>23.920623</td>
          <td>0.088143</td>
          <td>19.155970</td>
          <td>25.106908</td>
          <td>23.150404</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.768862</td>
          <td>0.036777</td>
          <td>28.768954</td>
          <td>0.803945</td>
          <td>28.411192</td>
          <td>0.570508</td>
          <td>26.448442</td>
          <td>0.187927</td>
          <td>24.822505</td>
          <td>0.086805</td>
          <td>22.890502</td>
          <td>0.035400</td>
          <td>22.025185</td>
          <td>31.077576</td>
          <td>20.636644</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.092189</td>
          <td>1.116758</td>
          <td>25.885207</td>
          <td>0.080996</td>
          <td>24.480999</td>
          <td>0.020711</td>
          <td>18.896940</td>
          <td>0.005012</td>
          <td>25.101487</td>
          <td>0.110854</td>
          <td>26.207143</td>
          <td>0.571185</td>
          <td>22.056413</td>
          <td>25.334257</td>
          <td>21.012807</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.991322</td>
          <td>0.251047</td>
          <td>20.788381</td>
          <td>0.005108</td>
          <td>16.899147</td>
          <td>0.005001</td>
          <td>25.739793</td>
          <td>0.102053</td>
          <td>25.581309</td>
          <td>0.167707</td>
          <td>25.399397</td>
          <td>0.308913</td>
          <td>23.241840</td>
          <td>24.065641</td>
          <td>22.922245</td>
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
          <td>23.569288</td>
          <td>0.030919</td>
          <td>27.029683</td>
          <td>0.217078</td>
          <td>25.945143</td>
          <td>0.075117</td>
          <td>27.011940</td>
          <td>0.299251</td>
          <td>23.521008</td>
          <td>0.027465</td>
          <td>23.862224</td>
          <td>0.083725</td>
          <td>26.250758</td>
          <td>22.520159</td>
          <td>30.468451</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.612445</td>
          <td>0.005629</td>
          <td>24.007006</td>
          <td>0.015845</td>
          <td>24.607433</td>
          <td>0.023081</td>
          <td>19.516290</td>
          <td>0.005026</td>
          <td>24.368065</td>
          <td>0.058074</td>
          <td>27.915356</td>
          <td>1.593180</td>
          <td>23.172954</td>
          <td>22.045923</td>
          <td>19.948984</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.112420</td>
          <td>0.277105</td>
          <td>22.537572</td>
          <td>0.006487</td>
          <td>22.428166</td>
          <td>0.005947</td>
          <td>20.764267</td>
          <td>0.005159</td>
          <td>20.206254</td>
          <td>0.005213</td>
          <td>25.687425</td>
          <td>0.387606</td>
          <td>20.898901</td>
          <td>21.765132</td>
          <td>24.736590</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.686460</td>
          <td>0.005186</td>
          <td>26.930194</td>
          <td>0.199745</td>
          <td>28.511069</td>
          <td>0.612433</td>
          <td>22.470109</td>
          <td>0.007377</td>
          <td>25.067667</td>
          <td>0.107630</td>
          <td>19.446415</td>
          <td>0.005275</td>
          <td>25.510076</td>
          <td>25.060040</td>
          <td>26.300815</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.486471</td>
          <td>0.028786</td>
          <td>18.752784</td>
          <td>0.005009</td>
          <td>22.954739</td>
          <td>0.007147</td>
          <td>27.839576</td>
          <td>0.563183</td>
          <td>22.609949</td>
          <td>0.012874</td>
          <td>18.882801</td>
          <td>0.005114</td>
          <td>18.565963</td>
          <td>19.335020</td>
          <td>20.880703</td>
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
          <td>28.922123</td>
          <td>26.539839</td>
          <td>20.164916</td>
          <td>24.570023</td>
          <td>26.720147</td>
          <td>24.805046</td>
          <td>20.586332</td>
          <td>0.005035</td>
          <td>15.261590</td>
          <td>0.005000</td>
          <td>22.896679</td>
          <td>0.009924</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.852105</td>
          <td>27.064536</td>
          <td>29.993295</td>
          <td>25.082287</td>
          <td>24.168653</td>
          <td>23.975316</td>
          <td>19.157940</td>
          <td>0.005002</td>
          <td>25.113247</td>
          <td>0.064610</td>
          <td>23.154407</td>
          <td>0.011956</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.689272</td>
          <td>28.881117</td>
          <td>27.991957</td>
          <td>26.740200</td>
          <td>24.741627</td>
          <td>22.896606</td>
          <td>22.023590</td>
          <td>0.005467</td>
          <td>27.871977</td>
          <td>0.623571</td>
          <td>20.636036</td>
          <td>0.005113</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.717666</td>
          <td>25.970525</td>
          <td>24.472501</td>
          <td>18.898489</td>
          <td>25.323788</td>
          <td>26.712363</td>
          <td>22.050737</td>
          <td>0.005490</td>
          <td>25.392602</td>
          <td>0.082770</td>
          <td>21.016935</td>
          <td>0.005226</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.308117</td>
          <td>20.793923</td>
          <td>16.897567</td>
          <td>25.796011</td>
          <td>25.500709</td>
          <td>26.291280</td>
          <td>23.232893</td>
          <td>0.008382</td>
          <td>24.064392</td>
          <td>0.025456</td>
          <td>22.910245</td>
          <td>0.010017</td>
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
          <td>23.531546</td>
          <td>26.680287</td>
          <td>25.924933</td>
          <td>26.749033</td>
          <td>23.544405</td>
          <td>23.786986</td>
          <td>26.274537</td>
          <td>0.106077</td>
          <td>22.521424</td>
          <td>0.007866</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.608658</td>
          <td>24.012314</td>
          <td>24.630620</td>
          <td>19.513208</td>
          <td>24.412089</td>
          <td>30.398965</td>
          <td>23.168566</td>
          <td>0.008075</td>
          <td>22.046940</td>
          <td>0.006356</td>
          <td>19.956003</td>
          <td>0.005033</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.837960</td>
          <td>22.531705</td>
          <td>22.425833</td>
          <td>20.759702</td>
          <td>20.208064</td>
          <td>25.923626</td>
          <td>20.892374</td>
          <td>0.005061</td>
          <td>21.764101</td>
          <td>0.005844</td>
          <td>24.732055</td>
          <td>0.046007</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.682414</td>
          <td>26.825508</td>
          <td>28.018015</td>
          <td>22.479107</td>
          <td>25.007829</td>
          <td>19.443000</td>
          <td>25.446808</td>
          <td>0.050963</td>
          <td>24.944982</td>
          <td>0.055623</td>
          <td>26.379700</td>
          <td>0.194645</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.495554</td>
          <td>18.749613</td>
          <td>22.962109</td>
          <td>27.800403</td>
          <td>22.579442</td>
          <td>18.878141</td>
          <td>18.575834</td>
          <td>0.005001</td>
          <td>19.340391</td>
          <td>0.005011</td>
          <td>20.885834</td>
          <td>0.005178</td>
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
          <td>28.922123</td>
          <td>26.539839</td>
          <td>20.164916</td>
          <td>24.570023</td>
          <td>26.720147</td>
          <td>24.805046</td>
          <td>20.581477</td>
          <td>0.008631</td>
          <td>15.264756</td>
          <td>0.005000</td>
          <td>22.880894</td>
          <td>0.052536</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.852105</td>
          <td>27.064536</td>
          <td>29.993295</td>
          <td>25.082287</td>
          <td>24.168653</td>
          <td>23.975316</td>
          <td>19.159187</td>
          <td>0.005349</td>
          <td>25.178964</td>
          <td>0.344815</td>
          <td>23.065915</td>
          <td>0.061947</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.689272</td>
          <td>28.881117</td>
          <td>27.991957</td>
          <td>26.740200</td>
          <td>24.741627</td>
          <td>22.896606</td>
          <td>22.036398</td>
          <td>0.027117</td>
          <td>26.354038</td>
          <td>0.807195</td>
          <td>20.632537</td>
          <td>0.008380</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.717666</td>
          <td>25.970525</td>
          <td>24.472501</td>
          <td>18.898489</td>
          <td>25.323788</td>
          <td>26.712363</td>
          <td>22.042229</td>
          <td>0.027256</td>
          <td>25.285357</td>
          <td>0.374808</td>
          <td>21.008232</td>
          <td>0.010733</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.308117</td>
          <td>20.793923</td>
          <td>16.897567</td>
          <td>25.796011</td>
          <td>25.500709</td>
          <td>26.291280</td>
          <td>23.348129</td>
          <td>0.086930</td>
          <td>23.882762</td>
          <td>0.116598</td>
          <td>22.862505</td>
          <td>0.051681</td>
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
          <td>23.531546</td>
          <td>26.680287</td>
          <td>25.924933</td>
          <td>26.749033</td>
          <td>23.544405</td>
          <td>23.786986</td>
          <td>26.341071</td>
          <td>0.909299</td>
          <td>22.499685</td>
          <td>0.034224</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.608658</td>
          <td>24.012314</td>
          <td>24.630620</td>
          <td>19.513208</td>
          <td>24.412089</td>
          <td>30.398965</td>
          <td>23.302515</td>
          <td>0.083499</td>
          <td>22.014023</td>
          <td>0.022329</td>
          <td>19.953591</td>
          <td>0.006162</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.837960</td>
          <td>22.531705</td>
          <td>22.425833</td>
          <td>20.759702</td>
          <td>20.208064</td>
          <td>25.923626</td>
          <td>20.907590</td>
          <td>0.010728</td>
          <td>21.759224</td>
          <td>0.017954</td>
          <td>24.635260</td>
          <td>0.240913</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.682414</td>
          <td>26.825508</td>
          <td>28.018015</td>
          <td>22.479107</td>
          <td>25.007829</td>
          <td>19.443000</td>
          <td>25.562200</td>
          <td>0.536857</td>
          <td>24.743889</td>
          <td>0.242635</td>
          <td>25.823764</td>
          <td>0.602769</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.495554</td>
          <td>18.749613</td>
          <td>22.962109</td>
          <td>27.800403</td>
          <td>22.579442</td>
          <td>18.878141</td>
          <td>18.561968</td>
          <td>0.005119</td>
          <td>19.332809</td>
          <td>0.005333</td>
          <td>20.881212</td>
          <td>0.009820</td>
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


