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
          <td>22.921103</td>
          <td>21.222804</td>
          <td>25.964298</td>
          <td>22.640582</td>
          <td>23.401199</td>
          <td>20.235727</td>
          <td>23.997698</td>
          <td>23.476725</td>
          <td>19.843065</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.057828</td>
          <td>21.613438</td>
          <td>21.043642</td>
          <td>22.669930</td>
          <td>20.728339</td>
          <td>20.094598</td>
          <td>20.394164</td>
          <td>24.001444</td>
          <td>24.081241</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.357232</td>
          <td>25.108490</td>
          <td>24.689811</td>
          <td>29.209549</td>
          <td>22.573222</td>
          <td>23.257196</td>
          <td>21.107641</td>
          <td>22.583102</td>
          <td>21.653047</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.917013</td>
          <td>23.621510</td>
          <td>25.456096</td>
          <td>19.584499</td>
          <td>21.199209</td>
          <td>23.946982</td>
          <td>22.283866</td>
          <td>23.125673</td>
          <td>25.775260</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.818106</td>
          <td>19.410268</td>
          <td>27.231975</td>
          <td>23.443288</td>
          <td>25.529726</td>
          <td>23.375750</td>
          <td>19.066350</td>
          <td>22.374388</td>
          <td>24.604244</td>
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
          <td>22.124713</td>
          <td>20.356784</td>
          <td>21.280461</td>
          <td>23.531718</td>
          <td>22.420174</td>
          <td>29.166336</td>
          <td>21.965959</td>
          <td>22.085326</td>
          <td>20.973996</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.825579</td>
          <td>21.046970</td>
          <td>16.918842</td>
          <td>24.811726</td>
          <td>22.749566</td>
          <td>20.650157</td>
          <td>21.762875</td>
          <td>25.252554</td>
          <td>23.400559</td>
        </tr>
        <tr>
          <th>997</th>
          <td>15.006222</td>
          <td>23.152228</td>
          <td>20.410877</td>
          <td>23.648098</td>
          <td>23.188432</td>
          <td>26.393612</td>
          <td>24.399589</td>
          <td>20.763125</td>
          <td>23.455795</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.952488</td>
          <td>21.487548</td>
          <td>19.567330</td>
          <td>20.117167</td>
          <td>23.402273</td>
          <td>21.110759</td>
          <td>21.522024</td>
          <td>25.818347</td>
          <td>24.811722</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.730736</td>
          <td>18.726071</td>
          <td>28.705431</td>
          <td>19.865524</td>
          <td>20.454933</td>
          <td>22.613683</td>
          <td>25.472151</td>
          <td>25.405352</td>
          <td>26.739015</td>
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
          <td>22.902249</td>
          <td>0.017638</td>
          <td>21.216202</td>
          <td>0.005199</td>
          <td>25.843541</td>
          <td>0.068659</td>
          <td>22.638976</td>
          <td>0.008045</td>
          <td>23.413168</td>
          <td>0.025004</td>
          <td>20.234548</td>
          <td>0.005971</td>
          <td>23.997698</td>
          <td>23.476725</td>
          <td>19.843065</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.060765</td>
          <td>0.009581</td>
          <td>21.621475</td>
          <td>0.005365</td>
          <td>21.041393</td>
          <td>0.005104</td>
          <td>22.675562</td>
          <td>0.008210</td>
          <td>20.720719</td>
          <td>0.005485</td>
          <td>20.091129</td>
          <td>0.005773</td>
          <td>20.394164</td>
          <td>24.001444</td>
          <td>24.081241</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.336397</td>
          <td>0.060531</td>
          <td>25.206265</td>
          <td>0.044428</td>
          <td>24.677750</td>
          <td>0.024526</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.561286</td>
          <td>0.012407</td>
          <td>23.263773</td>
          <td>0.049278</td>
          <td>21.107641</td>
          <td>22.583102</td>
          <td>21.653047</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.907558</td>
          <td>0.041521</td>
          <td>23.638658</td>
          <td>0.011920</td>
          <td>25.417689</td>
          <td>0.047054</td>
          <td>19.585564</td>
          <td>0.005028</td>
          <td>21.201084</td>
          <td>0.006045</td>
          <td>23.974093</td>
          <td>0.092386</td>
          <td>22.283866</td>
          <td>23.125673</td>
          <td>25.775260</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.816353</td>
          <td>0.005219</td>
          <td>19.405720</td>
          <td>0.005019</td>
          <td>27.380688</td>
          <td>0.257550</td>
          <td>23.429842</td>
          <td>0.013759</td>
          <td>25.443496</td>
          <td>0.149060</td>
          <td>23.275663</td>
          <td>0.049801</td>
          <td>19.066350</td>
          <td>22.374388</td>
          <td>24.604244</td>
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
          <td>22.116476</td>
          <td>0.009920</td>
          <td>20.367238</td>
          <td>0.005061</td>
          <td>21.283727</td>
          <td>0.005152</td>
          <td>23.536321</td>
          <td>0.014972</td>
          <td>22.414949</td>
          <td>0.011140</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.965959</td>
          <td>22.085326</td>
          <td>20.973996</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.812582</td>
          <td>0.038209</td>
          <td>21.037717</td>
          <td>0.005153</td>
          <td>16.913857</td>
          <td>0.005001</td>
          <td>24.778032</td>
          <td>0.043591</td>
          <td>22.741283</td>
          <td>0.014259</td>
          <td>20.660608</td>
          <td>0.006885</td>
          <td>21.762875</td>
          <td>25.252554</td>
          <td>23.400559</td>
        </tr>
        <tr>
          <th>997</th>
          <td>15.006478</td>
          <td>0.005002</td>
          <td>23.154195</td>
          <td>0.008643</td>
          <td>20.406545</td>
          <td>0.005041</td>
          <td>23.644182</td>
          <td>0.016340</td>
          <td>23.197426</td>
          <td>0.020771</td>
          <td>26.762203</td>
          <td>0.833179</td>
          <td>24.399589</td>
          <td>20.763125</td>
          <td>23.455795</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.028977</td>
          <td>0.046189</td>
          <td>21.487585</td>
          <td>0.005298</td>
          <td>19.562136</td>
          <td>0.005013</td>
          <td>20.111462</td>
          <td>0.005059</td>
          <td>23.424932</td>
          <td>0.025261</td>
          <td>21.120304</td>
          <td>0.008695</td>
          <td>21.522024</td>
          <td>25.818347</td>
          <td>24.811722</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.355167</td>
          <td>0.703620</td>
          <td>18.719902</td>
          <td>0.005009</td>
          <td>32.612500</td>
          <td>3.832356</td>
          <td>19.859609</td>
          <td>0.005041</td>
          <td>20.451854</td>
          <td>0.005315</td>
          <td>22.590259</td>
          <td>0.027192</td>
          <td>25.472151</td>
          <td>25.405352</td>
          <td>26.739015</td>
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
          <td>22.921103</td>
          <td>21.222804</td>
          <td>25.964298</td>
          <td>22.640582</td>
          <td>23.401199</td>
          <td>20.235727</td>
          <td>24.005165</td>
          <td>0.014552</td>
          <td>23.467329</td>
          <td>0.015307</td>
          <td>19.837845</td>
          <td>0.005026</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.057828</td>
          <td>21.613438</td>
          <td>21.043642</td>
          <td>22.669930</td>
          <td>20.728339</td>
          <td>20.094598</td>
          <td>20.388506</td>
          <td>0.005024</td>
          <td>24.032956</td>
          <td>0.024765</td>
          <td>24.089692</td>
          <td>0.026027</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.357232</td>
          <td>25.108490</td>
          <td>24.689811</td>
          <td>29.209549</td>
          <td>22.573222</td>
          <td>23.257196</td>
          <td>21.110421</td>
          <td>0.005090</td>
          <td>22.594638</td>
          <td>0.008197</td>
          <td>21.654980</td>
          <td>0.005700</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.917013</td>
          <td>23.621510</td>
          <td>25.456096</td>
          <td>19.584499</td>
          <td>21.199209</td>
          <td>23.946982</td>
          <td>22.280021</td>
          <td>0.005731</td>
          <td>23.123384</td>
          <td>0.011680</td>
          <td>25.847696</td>
          <td>0.123379</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.818106</td>
          <td>19.410268</td>
          <td>27.231975</td>
          <td>23.443288</td>
          <td>25.529726</td>
          <td>23.375750</td>
          <td>19.067719</td>
          <td>0.005002</td>
          <td>22.380198</td>
          <td>0.007310</td>
          <td>24.637134</td>
          <td>0.042275</td>
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
          <td>22.124713</td>
          <td>20.356784</td>
          <td>21.280461</td>
          <td>23.531718</td>
          <td>22.420174</td>
          <td>29.166336</td>
          <td>21.967891</td>
          <td>0.005423</td>
          <td>22.090003</td>
          <td>0.006456</td>
          <td>20.977189</td>
          <td>0.005210</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.825579</td>
          <td>21.046970</td>
          <td>16.918842</td>
          <td>24.811726</td>
          <td>22.749566</td>
          <td>20.650157</td>
          <td>21.761395</td>
          <td>0.005293</td>
          <td>25.325314</td>
          <td>0.077987</td>
          <td>23.437412</td>
          <td>0.014938</td>
        </tr>
        <tr>
          <th>997</th>
          <td>15.006222</td>
          <td>23.152228</td>
          <td>20.410877</td>
          <td>23.648098</td>
          <td>23.188432</td>
          <td>26.393612</td>
          <td>24.391952</td>
          <td>0.020099</td>
          <td>20.761379</td>
          <td>0.005142</td>
          <td>23.435431</td>
          <td>0.014914</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.952488</td>
          <td>21.487548</td>
          <td>19.567330</td>
          <td>20.117167</td>
          <td>23.402273</td>
          <td>21.110759</td>
          <td>21.515462</td>
          <td>0.005188</td>
          <td>26.035533</td>
          <td>0.145157</td>
          <td>24.860051</td>
          <td>0.051568</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.730736</td>
          <td>18.726071</td>
          <td>28.705431</td>
          <td>19.865524</td>
          <td>20.454933</td>
          <td>22.613683</td>
          <td>25.494958</td>
          <td>0.053198</td>
          <td>25.485883</td>
          <td>0.089873</td>
          <td>26.899133</td>
          <td>0.298766</td>
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
          <td>22.921103</td>
          <td>21.222804</td>
          <td>25.964298</td>
          <td>22.640582</td>
          <td>23.401199</td>
          <td>20.235727</td>
          <td>24.222528</td>
          <td>0.185471</td>
          <td>23.475411</td>
          <td>0.081522</td>
          <td>19.839200</td>
          <td>0.005959</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.057828</td>
          <td>21.613438</td>
          <td>21.043642</td>
          <td>22.669930</td>
          <td>20.728339</td>
          <td>20.094598</td>
          <td>20.413599</td>
          <td>0.007832</td>
          <td>24.031781</td>
          <td>0.132717</td>
          <td>23.897145</td>
          <td>0.128793</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.357232</td>
          <td>25.108490</td>
          <td>24.689811</td>
          <td>29.209549</td>
          <td>22.573222</td>
          <td>23.257196</td>
          <td>21.119900</td>
          <td>0.012570</td>
          <td>22.570206</td>
          <td>0.036437</td>
          <td>21.649949</td>
          <td>0.017814</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.917013</td>
          <td>23.621510</td>
          <td>25.456096</td>
          <td>19.584499</td>
          <td>21.199209</td>
          <td>23.946982</td>
          <td>22.255141</td>
          <td>0.032897</td>
          <td>23.071377</td>
          <td>0.056947</td>
          <td>25.961030</td>
          <td>0.663380</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.818106</td>
          <td>19.410268</td>
          <td>27.231975</td>
          <td>23.443288</td>
          <td>25.529726</td>
          <td>23.375750</td>
          <td>19.072135</td>
          <td>0.005299</td>
          <td>22.432387</td>
          <td>0.032240</td>
          <td>24.435054</td>
          <td>0.203921</td>
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
          <td>22.124713</td>
          <td>20.356784</td>
          <td>21.280461</td>
          <td>23.531718</td>
          <td>22.420174</td>
          <td>29.166336</td>
          <td>21.992915</td>
          <td>0.026101</td>
          <td>22.072794</td>
          <td>0.023499</td>
          <td>20.958853</td>
          <td>0.010362</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.825579</td>
          <td>21.046970</td>
          <td>16.918842</td>
          <td>24.811726</td>
          <td>22.749566</td>
          <td>20.650157</td>
          <td>21.755144</td>
          <td>0.021221</td>
          <td>24.779121</td>
          <td>0.249781</td>
          <td>23.436236</td>
          <td>0.086022</td>
        </tr>
        <tr>
          <th>997</th>
          <td>15.006222</td>
          <td>23.152228</td>
          <td>20.410877</td>
          <td>23.648098</td>
          <td>23.188432</td>
          <td>26.393612</td>
          <td>24.491269</td>
          <td>0.232303</td>
          <td>20.755846</td>
          <td>0.008498</td>
          <td>23.400005</td>
          <td>0.083314</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.952488</td>
          <td>21.487548</td>
          <td>19.567330</td>
          <td>20.117167</td>
          <td>23.402273</td>
          <td>21.110759</td>
          <td>21.552025</td>
          <td>0.017845</td>
          <td>25.349141</td>
          <td>0.393816</td>
          <td>24.961439</td>
          <td>0.314081</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.730736</td>
          <td>18.726071</td>
          <td>28.705431</td>
          <td>19.865524</td>
          <td>20.454933</td>
          <td>22.613683</td>
          <td>25.452648</td>
          <td>0.495428</td>
          <td>25.617577</td>
          <td>0.482711</td>
          <td>25.259753</td>
          <td>0.397055</td>
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


