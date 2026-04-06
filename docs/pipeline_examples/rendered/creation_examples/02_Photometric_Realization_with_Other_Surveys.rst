Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


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
          <td>20.747000</td>
          <td>22.249063</td>
          <td>23.685588</td>
          <td>22.721838</td>
          <td>23.771914</td>
          <td>22.698360</td>
          <td>21.160142</td>
          <td>25.351397</td>
          <td>20.687696</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.084322</td>
          <td>20.063632</td>
          <td>16.944874</td>
          <td>15.781950</td>
          <td>24.201453</td>
          <td>21.605015</td>
          <td>25.259374</td>
          <td>20.598820</td>
          <td>26.631856</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.431844</td>
          <td>20.889241</td>
          <td>20.798437</td>
          <td>21.781607</td>
          <td>26.457272</td>
          <td>23.171717</td>
          <td>15.959734</td>
          <td>21.899959</td>
          <td>24.915304</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.010054</td>
          <td>22.958542</td>
          <td>22.016040</td>
          <td>21.333530</td>
          <td>27.134535</td>
          <td>28.643858</td>
          <td>21.106356</td>
          <td>16.916463</td>
          <td>22.738952</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.583002</td>
          <td>21.020391</td>
          <td>20.932884</td>
          <td>23.507425</td>
          <td>23.737179</td>
          <td>22.200160</td>
          <td>24.656101</td>
          <td>21.623564</td>
          <td>18.163103</td>
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
          <td>24.202943</td>
          <td>21.510214</td>
          <td>26.167983</td>
          <td>19.955693</td>
          <td>22.952157</td>
          <td>24.654113</td>
          <td>21.521533</td>
          <td>24.078578</td>
          <td>18.662254</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.497608</td>
          <td>26.714294</td>
          <td>26.262877</td>
          <td>23.009455</td>
          <td>26.579545</td>
          <td>27.431487</td>
          <td>23.884850</td>
          <td>24.383581</td>
          <td>21.427621</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.104230</td>
          <td>20.331776</td>
          <td>23.979348</td>
          <td>19.681189</td>
          <td>23.442473</td>
          <td>21.094317</td>
          <td>22.955014</td>
          <td>19.577914</td>
          <td>23.564479</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.312059</td>
          <td>27.299600</td>
          <td>22.305423</td>
          <td>21.983220</td>
          <td>21.543244</td>
          <td>26.309859</td>
          <td>31.015647</td>
          <td>24.034146</td>
          <td>18.362046</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.555251</td>
          <td>23.668204</td>
          <td>18.656786</td>
          <td>28.555834</td>
          <td>22.930563</td>
          <td>24.111705</td>
          <td>25.056557</td>
          <td>22.354667</td>
          <td>24.444780</td>
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
          <td>20.750825</td>
          <td>0.005762</td>
          <td>22.243010</td>
          <td>0.005949</td>
          <td>23.681741</td>
          <td>0.010976</td>
          <td>22.709282</td>
          <td>0.008369</td>
          <td>23.811459</td>
          <td>0.035451</td>
          <td>22.712921</td>
          <td>0.030274</td>
          <td>21.160142</td>
          <td>25.351397</td>
          <td>20.687696</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.081561</td>
          <td>0.020429</td>
          <td>20.070760</td>
          <td>0.005041</td>
          <td>16.948315</td>
          <td>0.005001</td>
          <td>15.775185</td>
          <td>0.005000</td>
          <td>24.216171</td>
          <td>0.050749</td>
          <td>21.628484</td>
          <td>0.012283</td>
          <td>25.259374</td>
          <td>20.598820</td>
          <td>26.631856</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.136428</td>
          <td>0.604842</td>
          <td>20.884444</td>
          <td>0.005123</td>
          <td>20.807263</td>
          <td>0.005073</td>
          <td>21.793243</td>
          <td>0.005830</td>
          <td>26.354289</td>
          <td>0.317842</td>
          <td>23.213599</td>
          <td>0.047132</td>
          <td>15.959734</td>
          <td>21.899959</td>
          <td>24.915304</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.007933</td>
          <td>0.005083</td>
          <td>22.958581</td>
          <td>0.007766</td>
          <td>22.011876</td>
          <td>0.005485</td>
          <td>21.332760</td>
          <td>0.005395</td>
          <td>28.137906</td>
          <td>1.101154</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.106356</td>
          <td>16.916463</td>
          <td>22.738952</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.121572</td>
          <td>0.279168</td>
          <td>21.014044</td>
          <td>0.005148</td>
          <td>20.930070</td>
          <td>0.005088</td>
          <td>23.499075</td>
          <td>0.014533</td>
          <td>23.730799</td>
          <td>0.033016</td>
          <td>22.197388</td>
          <td>0.019398</td>
          <td>24.656101</td>
          <td>21.623564</td>
          <td>18.163103</td>
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
          <td>24.261546</td>
          <td>0.056672</td>
          <td>21.508067</td>
          <td>0.005307</td>
          <td>26.179410</td>
          <td>0.092346</td>
          <td>19.959944</td>
          <td>0.005047</td>
          <td>22.936423</td>
          <td>0.016692</td>
          <td>24.515391</td>
          <td>0.147932</td>
          <td>21.521533</td>
          <td>24.078578</td>
          <td>18.662254</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.493380</td>
          <td>0.007146</td>
          <td>26.658106</td>
          <td>0.158618</td>
          <td>26.344310</td>
          <td>0.106703</td>
          <td>23.010101</td>
          <td>0.010108</td>
          <td>26.769258</td>
          <td>0.438997</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.884850</td>
          <td>24.383581</td>
          <td>21.427621</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.128239</td>
          <td>0.021237</td>
          <td>20.332590</td>
          <td>0.005058</td>
          <td>23.993031</td>
          <td>0.013867</td>
          <td>19.673554</td>
          <td>0.005032</td>
          <td>23.402850</td>
          <td>0.024781</td>
          <td>21.105829</td>
          <td>0.008621</td>
          <td>22.955014</td>
          <td>19.577914</td>
          <td>23.564479</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>27.289237</td>
          <td>0.268863</td>
          <td>22.287629</td>
          <td>0.005756</td>
          <td>21.976953</td>
          <td>0.006112</td>
          <td>21.535949</td>
          <td>0.006762</td>
          <td>28.200245</td>
          <td>1.818698</td>
          <td>31.015647</td>
          <td>24.034146</td>
          <td>18.362046</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.561852</td>
          <td>0.030721</td>
          <td>23.667747</td>
          <td>0.012178</td>
          <td>18.655385</td>
          <td>0.005005</td>
          <td>29.272457</td>
          <td>1.373661</td>
          <td>22.936912</td>
          <td>0.016699</td>
          <td>24.133329</td>
          <td>0.106221</td>
          <td>25.056557</td>
          <td>22.354667</td>
          <td>24.444780</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


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
          <td>20.747000</td>
          <td>22.249063</td>
          <td>23.685588</td>
          <td>22.721838</td>
          <td>23.771914</td>
          <td>22.698360</td>
          <td>21.155293</td>
          <td>0.005098</td>
          <td>25.240994</td>
          <td>0.072372</td>
          <td>20.695000</td>
          <td>0.005126</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.084322</td>
          <td>20.063632</td>
          <td>16.944874</td>
          <td>15.781950</td>
          <td>24.201453</td>
          <td>21.605015</td>
          <td>25.248763</td>
          <td>0.042715</td>
          <td>20.598641</td>
          <td>0.005106</td>
          <td>27.160960</td>
          <td>0.367743</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.431844</td>
          <td>20.889241</td>
          <td>20.798437</td>
          <td>21.781607</td>
          <td>26.457272</td>
          <td>23.171717</td>
          <td>15.965282</td>
          <td>0.005000</td>
          <td>21.906039</td>
          <td>0.006073</td>
          <td>24.859326</td>
          <td>0.051535</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.010054</td>
          <td>22.958542</td>
          <td>22.016040</td>
          <td>21.333530</td>
          <td>27.134535</td>
          <td>28.643858</td>
          <td>21.105800</td>
          <td>0.005089</td>
          <td>16.912381</td>
          <td>0.005000</td>
          <td>22.718919</td>
          <td>0.008833</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.583002</td>
          <td>21.020391</td>
          <td>20.932884</td>
          <td>23.507425</td>
          <td>23.737179</td>
          <td>22.200160</td>
          <td>24.639040</td>
          <td>0.024897</td>
          <td>21.617250</td>
          <td>0.005655</td>
          <td>18.166718</td>
          <td>0.005001</td>
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
          <td>24.202943</td>
          <td>21.510214</td>
          <td>26.167983</td>
          <td>19.955693</td>
          <td>22.952157</td>
          <td>24.654113</td>
          <td>21.522889</td>
          <td>0.005191</td>
          <td>24.105958</td>
          <td>0.026401</td>
          <td>18.660518</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.497608</td>
          <td>26.714294</td>
          <td>26.262877</td>
          <td>23.009455</td>
          <td>26.579545</td>
          <td>27.431487</td>
          <td>23.882262</td>
          <td>0.013195</td>
          <td>24.341216</td>
          <td>0.032494</td>
          <td>21.429635</td>
          <td>0.005472</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.104230</td>
          <td>20.331776</td>
          <td>23.979348</td>
          <td>19.681189</td>
          <td>23.442473</td>
          <td>21.094317</td>
          <td>22.950311</td>
          <td>0.007205</td>
          <td>19.571725</td>
          <td>0.005016</td>
          <td>23.572609</td>
          <td>0.016697</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.312059</td>
          <td>27.299600</td>
          <td>22.305423</td>
          <td>21.983220</td>
          <td>21.543244</td>
          <td>26.309859</td>
          <td>30.195930</td>
          <td>1.702349</td>
          <td>23.975413</td>
          <td>0.023553</td>
          <td>18.358650</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.555251</td>
          <td>23.668204</td>
          <td>18.656786</td>
          <td>28.555834</td>
          <td>22.930563</td>
          <td>24.111705</td>
          <td>25.015199</td>
          <td>0.034698</td>
          <td>22.369373</td>
          <td>0.007272</td>
          <td>24.465424</td>
          <td>0.036282</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


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
          <td>20.747000</td>
          <td>22.249063</td>
          <td>23.685588</td>
          <td>22.721838</td>
          <td>23.771914</td>
          <td>22.698360</td>
          <td>21.197857</td>
          <td>0.013358</td>
          <td>25.544295</td>
          <td>0.456976</td>
          <td>20.691479</td>
          <td>0.008684</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.084322</td>
          <td>20.063632</td>
          <td>16.944874</td>
          <td>15.781950</td>
          <td>24.201453</td>
          <td>21.605015</td>
          <td>24.950775</td>
          <td>0.337218</td>
          <td>20.595985</td>
          <td>0.007758</td>
          <td>25.652471</td>
          <td>0.533073</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.431844</td>
          <td>20.889241</td>
          <td>20.798437</td>
          <td>21.781607</td>
          <td>26.457272</td>
          <td>23.171717</td>
          <td>15.958041</td>
          <td>0.005001</td>
          <td>21.889012</td>
          <td>0.020049</td>
          <td>25.676114</td>
          <td>0.542306</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.010054</td>
          <td>22.958542</td>
          <td>22.016040</td>
          <td>21.333530</td>
          <td>27.134535</td>
          <td>28.643858</td>
          <td>21.099809</td>
          <td>0.012377</td>
          <td>16.920780</td>
          <td>0.005004</td>
          <td>22.710532</td>
          <td>0.045133</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.583002</td>
          <td>21.020391</td>
          <td>20.932884</td>
          <td>23.507425</td>
          <td>23.737179</td>
          <td>22.200160</td>
          <td>24.760037</td>
          <td>0.289491</td>
          <td>21.613804</td>
          <td>0.015902</td>
          <td>18.156415</td>
          <td>0.005047</td>
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
          <td>24.202943</td>
          <td>21.510214</td>
          <td>26.167983</td>
          <td>19.955693</td>
          <td>22.952157</td>
          <td>24.654113</td>
          <td>21.557299</td>
          <td>0.017925</td>
          <td>24.104031</td>
          <td>0.141269</td>
          <td>18.665656</td>
          <td>0.005120</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.497608</td>
          <td>26.714294</td>
          <td>26.262877</td>
          <td>23.009455</td>
          <td>26.579545</td>
          <td>27.431487</td>
          <td>23.859882</td>
          <td>0.135984</td>
          <td>24.282087</td>
          <td>0.164599</td>
          <td>21.422484</td>
          <td>0.014758</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.104230</td>
          <td>20.331776</td>
          <td>23.979348</td>
          <td>19.681189</td>
          <td>23.442473</td>
          <td>21.094317</td>
          <td>22.959854</td>
          <td>0.061614</td>
          <td>19.578090</td>
          <td>0.005514</td>
          <td>23.657910</td>
          <td>0.104543</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.312059</td>
          <td>27.299600</td>
          <td>22.305423</td>
          <td>21.983220</td>
          <td>21.543244</td>
          <td>26.309859</td>
          <td>27.330162</td>
          <td>1.573290</td>
          <td>23.842511</td>
          <td>0.112574</td>
          <td>18.363716</td>
          <td>0.005069</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.555251</td>
          <td>23.668204</td>
          <td>18.656786</td>
          <td>28.555834</td>
          <td>22.930563</td>
          <td>24.111705</td>
          <td>24.402197</td>
          <td>0.215711</td>
          <td>22.387875</td>
          <td>0.030994</td>
          <td>24.419007</td>
          <td>0.201191</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


