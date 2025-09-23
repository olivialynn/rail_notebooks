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
          <td>15.629304</td>
          <td>24.485752</td>
          <td>22.895969</td>
          <td>26.206428</td>
          <td>26.249992</td>
          <td>18.991517</td>
          <td>26.065269</td>
          <td>28.616900</td>
          <td>23.995896</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.501996</td>
          <td>24.629489</td>
          <td>27.489514</td>
          <td>22.567553</td>
          <td>24.885363</td>
          <td>24.411922</td>
          <td>19.884378</td>
          <td>22.368658</td>
          <td>23.653606</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.285185</td>
          <td>21.233844</td>
          <td>22.070605</td>
          <td>23.316150</td>
          <td>17.000618</td>
          <td>20.290112</td>
          <td>18.079446</td>
          <td>21.664521</td>
          <td>23.046145</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.584035</td>
          <td>23.187159</td>
          <td>25.700550</td>
          <td>24.474084</td>
          <td>17.724925</td>
          <td>21.505372</td>
          <td>19.202377</td>
          <td>26.299869</td>
          <td>28.748204</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.255959</td>
          <td>25.265289</td>
          <td>26.742070</td>
          <td>23.994556</td>
          <td>27.857027</td>
          <td>19.859936</td>
          <td>24.839625</td>
          <td>23.193308</td>
          <td>24.450718</td>
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
          <td>25.039312</td>
          <td>24.199893</td>
          <td>22.519263</td>
          <td>25.639170</td>
          <td>25.236066</td>
          <td>20.357024</td>
          <td>22.368201</td>
          <td>20.151623</td>
          <td>24.072607</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.314079</td>
          <td>25.596450</td>
          <td>26.690298</td>
          <td>23.855417</td>
          <td>20.198065</td>
          <td>22.652398</td>
          <td>22.409178</td>
          <td>26.679922</td>
          <td>18.467228</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.896052</td>
          <td>24.488133</td>
          <td>23.136684</td>
          <td>21.682581</td>
          <td>20.654823</td>
          <td>21.511164</td>
          <td>27.972212</td>
          <td>24.693432</td>
          <td>22.361020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.462015</td>
          <td>23.262272</td>
          <td>21.312694</td>
          <td>23.387417</td>
          <td>29.029450</td>
          <td>23.005463</td>
          <td>28.201173</td>
          <td>22.927708</td>
          <td>21.012230</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.359221</td>
          <td>19.040289</td>
          <td>25.984521</td>
          <td>23.944207</td>
          <td>24.120868</td>
          <td>14.652887</td>
          <td>23.340134</td>
          <td>26.455211</td>
          <td>24.696576</td>
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
          <td>15.634035</td>
          <td>0.005003</td>
          <td>24.463393</td>
          <td>0.023194</td>
          <td>22.889689</td>
          <td>0.006946</td>
          <td>26.401288</td>
          <td>0.180580</td>
          <td>27.313898</td>
          <td>0.651818</td>
          <td>18.981293</td>
          <td>0.005133</td>
          <td>26.065269</td>
          <td>28.616900</td>
          <td>23.995896</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.503194</td>
          <td>0.005148</td>
          <td>24.632449</td>
          <td>0.026836</td>
          <td>27.481455</td>
          <td>0.279601</td>
          <td>22.549686</td>
          <td>0.007674</td>
          <td>24.871053</td>
          <td>0.090593</td>
          <td>24.392011</td>
          <td>0.133010</td>
          <td>19.884378</td>
          <td>22.368658</td>
          <td>23.653606</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.959261</td>
          <td>0.104502</td>
          <td>21.233386</td>
          <td>0.005204</td>
          <td>22.067944</td>
          <td>0.005531</td>
          <td>23.329758</td>
          <td>0.012735</td>
          <td>17.001882</td>
          <td>0.005003</td>
          <td>20.287221</td>
          <td>0.006055</td>
          <td>18.079446</td>
          <td>21.664521</td>
          <td>23.046145</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.574840</td>
          <td>0.007400</td>
          <td>23.181761</td>
          <td>0.008785</td>
          <td>25.767063</td>
          <td>0.064161</td>
          <td>24.477449</td>
          <td>0.033408</td>
          <td>17.724960</td>
          <td>0.005007</td>
          <td>21.512634</td>
          <td>0.011278</td>
          <td>19.202377</td>
          <td>26.299869</td>
          <td>28.748204</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.269951</td>
          <td>0.010972</td>
          <td>25.254902</td>
          <td>0.046382</td>
          <td>26.758650</td>
          <td>0.152769</td>
          <td>24.018784</td>
          <td>0.022392</td>
          <td>26.770249</td>
          <td>0.439327</td>
          <td>19.847633</td>
          <td>0.005523</td>
          <td>24.839625</td>
          <td>23.193308</td>
          <td>24.450718</td>
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
          <td>24.960348</td>
          <td>0.104601</td>
          <td>24.191430</td>
          <td>0.018431</td>
          <td>22.528380</td>
          <td>0.006110</td>
          <td>25.555517</td>
          <td>0.086806</td>
          <td>25.047668</td>
          <td>0.105765</td>
          <td>20.358797</td>
          <td>0.006181</td>
          <td>22.368201</td>
          <td>20.151623</td>
          <td>24.072607</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.294165</td>
          <td>0.058322</td>
          <td>25.705547</td>
          <td>0.069121</td>
          <td>26.547904</td>
          <td>0.127390</td>
          <td>23.846251</td>
          <td>0.019331</td>
          <td>20.195874</td>
          <td>0.005209</td>
          <td>22.732575</td>
          <td>0.030801</td>
          <td>22.409178</td>
          <td>26.679922</td>
          <td>18.467228</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.897741</td>
          <td>0.041165</td>
          <td>24.492038</td>
          <td>0.023771</td>
          <td>23.146074</td>
          <td>0.007849</td>
          <td>21.674737</td>
          <td>0.005686</td>
          <td>20.647447</td>
          <td>0.005431</td>
          <td>21.512272</td>
          <td>0.011275</td>
          <td>27.972212</td>
          <td>24.693432</td>
          <td>22.361020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.958977</td>
          <td>0.532633</td>
          <td>23.268039</td>
          <td>0.009259</td>
          <td>21.326833</td>
          <td>0.005162</td>
          <td>23.403357</td>
          <td>0.013478</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.925525</td>
          <td>0.036512</td>
          <td>28.201173</td>
          <td>22.927708</td>
          <td>21.012230</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.354905</td>
          <td>0.005124</td>
          <td>19.036176</td>
          <td>0.005012</td>
          <td>26.035269</td>
          <td>0.081339</td>
          <td>23.905100</td>
          <td>0.020319</td>
          <td>24.113715</td>
          <td>0.046337</td>
          <td>14.646596</td>
          <td>0.005001</td>
          <td>23.340134</td>
          <td>26.455211</td>
          <td>24.696576</td>
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
          <td>15.629304</td>
          <td>24.485752</td>
          <td>22.895969</td>
          <td>26.206428</td>
          <td>26.249992</td>
          <td>18.991517</td>
          <td>26.038247</td>
          <td>0.086175</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.004180</td>
          <td>0.024151</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.501996</td>
          <td>24.629489</td>
          <td>27.489514</td>
          <td>22.567553</td>
          <td>24.885363</td>
          <td>24.411922</td>
          <td>19.882359</td>
          <td>0.005009</td>
          <td>22.367592</td>
          <td>0.007266</td>
          <td>23.639795</td>
          <td>0.017663</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.285185</td>
          <td>21.233844</td>
          <td>22.070605</td>
          <td>23.316150</td>
          <td>17.000618</td>
          <td>20.290112</td>
          <td>18.075078</td>
          <td>0.005000</td>
          <td>21.663948</td>
          <td>0.005711</td>
          <td>23.050881</td>
          <td>0.011070</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.584035</td>
          <td>23.187159</td>
          <td>25.700550</td>
          <td>24.474084</td>
          <td>17.724925</td>
          <td>21.505372</td>
          <td>19.201001</td>
          <td>0.005003</td>
          <td>26.519129</td>
          <td>0.218780</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.255959</td>
          <td>25.265289</td>
          <td>26.742070</td>
          <td>23.994556</td>
          <td>27.857027</td>
          <td>19.859936</td>
          <td>24.797719</td>
          <td>0.028621</td>
          <td>23.205808</td>
          <td>0.012434</td>
          <td>24.434523</td>
          <td>0.035299</td>
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
          <td>25.039312</td>
          <td>24.199893</td>
          <td>22.519263</td>
          <td>25.639170</td>
          <td>25.236066</td>
          <td>20.357024</td>
          <td>22.376302</td>
          <td>0.005862</td>
          <td>20.154463</td>
          <td>0.005047</td>
          <td>24.106592</td>
          <td>0.026416</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.314079</td>
          <td>25.596450</td>
          <td>26.690298</td>
          <td>23.855417</td>
          <td>20.198065</td>
          <td>22.652398</td>
          <td>22.405944</td>
          <td>0.005906</td>
          <td>27.253572</td>
          <td>0.395166</td>
          <td>18.453615</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.896052</td>
          <td>24.488133</td>
          <td>23.136684</td>
          <td>21.682581</td>
          <td>20.654823</td>
          <td>21.511164</td>
          <td>27.689149</td>
          <td>0.347596</td>
          <td>24.693007</td>
          <td>0.044433</td>
          <td>22.364089</td>
          <td>0.007253</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.462015</td>
          <td>23.262272</td>
          <td>21.312694</td>
          <td>23.387417</td>
          <td>29.029450</td>
          <td>23.005463</td>
          <td>28.250896</td>
          <td>0.532462</td>
          <td>22.928019</td>
          <td>0.010141</td>
          <td>21.013161</td>
          <td>0.005225</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.359221</td>
          <td>19.040289</td>
          <td>25.984521</td>
          <td>23.944207</td>
          <td>24.120868</td>
          <td>14.652887</td>
          <td>23.345826</td>
          <td>0.008983</td>
          <td>26.427586</td>
          <td>0.202646</td>
          <td>24.661611</td>
          <td>0.043207</td>
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
          <td>15.629304</td>
          <td>24.485752</td>
          <td>22.895969</td>
          <td>26.206428</td>
          <td>26.249992</td>
          <td>18.991517</td>
          <td>26.845504</td>
          <td>1.223352</td>
          <td>27.305367</td>
          <td>1.405608</td>
          <td>24.046934</td>
          <td>0.146589</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.501996</td>
          <td>24.629489</td>
          <td>27.489514</td>
          <td>22.567553</td>
          <td>24.885363</td>
          <td>24.411922</td>
          <td>19.881573</td>
          <td>0.006218</td>
          <td>22.348408</td>
          <td>0.029931</td>
          <td>23.580644</td>
          <td>0.097690</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.285185</td>
          <td>21.233844</td>
          <td>22.070605</td>
          <td>23.316150</td>
          <td>17.000618</td>
          <td>20.290112</td>
          <td>18.072225</td>
          <td>0.005049</td>
          <td>21.639383</td>
          <td>0.016242</td>
          <td>22.960555</td>
          <td>0.056400</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.584035</td>
          <td>23.187159</td>
          <td>25.700550</td>
          <td>24.474084</td>
          <td>17.724925</td>
          <td>21.505372</td>
          <td>19.196541</td>
          <td>0.005373</td>
          <td>25.346804</td>
          <td>0.393106</td>
          <td>26.353910</td>
          <td>0.860723</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.255959</td>
          <td>25.265289</td>
          <td>26.742070</td>
          <td>23.994556</td>
          <td>27.857027</td>
          <td>19.859936</td>
          <td>24.894971</td>
          <td>0.322601</td>
          <td>23.234325</td>
          <td>0.065832</td>
          <td>24.719087</td>
          <td>0.258112</td>
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
          <td>25.039312</td>
          <td>24.199893</td>
          <td>22.519263</td>
          <td>25.639170</td>
          <td>25.236066</td>
          <td>20.357024</td>
          <td>22.353790</td>
          <td>0.035909</td>
          <td>20.147626</td>
          <td>0.006358</td>
          <td>24.280600</td>
          <td>0.178998</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.314079</td>
          <td>25.596450</td>
          <td>26.690298</td>
          <td>23.855417</td>
          <td>20.198065</td>
          <td>22.652398</td>
          <td>22.377160</td>
          <td>0.036663</td>
          <td>27.977254</td>
          <td>1.930735</td>
          <td>18.460592</td>
          <td>0.005082</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.896052</td>
          <td>24.488133</td>
          <td>23.136684</td>
          <td>21.682581</td>
          <td>20.654823</td>
          <td>21.511164</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.051345</td>
          <td>0.311555</td>
          <td>22.402768</td>
          <td>0.034317</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.462015</td>
          <td>23.262272</td>
          <td>21.312694</td>
          <td>23.387417</td>
          <td>29.029450</td>
          <td>23.005463</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.892519</td>
          <td>0.048555</td>
          <td>20.994937</td>
          <td>0.010631</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.359221</td>
          <td>19.040289</td>
          <td>25.984521</td>
          <td>23.944207</td>
          <td>24.120868</td>
          <td>14.652887</td>
          <td>23.471171</td>
          <td>0.096880</td>
          <td>27.562784</td>
          <td>1.598340</td>
          <td>24.408379</td>
          <td>0.199402</td>
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


