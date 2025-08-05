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
          <td>20.773542</td>
          <td>20.109167</td>
          <td>27.840134</td>
          <td>20.354303</td>
          <td>22.593589</td>
          <td>24.289279</td>
          <td>22.214591</td>
          <td>22.288867</td>
          <td>22.411895</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.506606</td>
          <td>21.332445</td>
          <td>21.725287</td>
          <td>23.380683</td>
          <td>21.369849</td>
          <td>24.030432</td>
          <td>29.307208</td>
          <td>27.223049</td>
          <td>27.646343</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.910567</td>
          <td>21.348195</td>
          <td>23.076603</td>
          <td>24.037825</td>
          <td>26.082534</td>
          <td>26.085767</td>
          <td>23.558094</td>
          <td>24.043185</td>
          <td>22.231625</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.898859</td>
          <td>24.169216</td>
          <td>29.160423</td>
          <td>19.539080</td>
          <td>18.046586</td>
          <td>21.578655</td>
          <td>26.875664</td>
          <td>24.221688</td>
          <td>20.690827</td>
        </tr>
        <tr>
          <th>4</th>
          <td>14.701788</td>
          <td>24.865995</td>
          <td>24.326521</td>
          <td>23.454226</td>
          <td>25.496923</td>
          <td>24.236635</td>
          <td>31.108190</td>
          <td>25.642572</td>
          <td>17.718634</td>
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
          <td>20.178298</td>
          <td>19.246676</td>
          <td>25.358689</td>
          <td>18.849029</td>
          <td>17.655867</td>
          <td>21.317267</td>
          <td>18.455154</td>
          <td>22.855470</td>
          <td>27.711379</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.117711</td>
          <td>21.670792</td>
          <td>19.359187</td>
          <td>29.273090</td>
          <td>23.914363</td>
          <td>23.686170</td>
          <td>21.536473</td>
          <td>19.263586</td>
          <td>21.960967</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.537205</td>
          <td>22.810912</td>
          <td>21.539062</td>
          <td>20.418506</td>
          <td>25.464075</td>
          <td>21.654261</td>
          <td>15.037090</td>
          <td>21.939859</td>
          <td>19.646113</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.613021</td>
          <td>24.817423</td>
          <td>23.780940</td>
          <td>21.737709</td>
          <td>21.528368</td>
          <td>20.467685</td>
          <td>26.849602</td>
          <td>26.677617</td>
          <td>25.977284</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.436143</td>
          <td>22.526714</td>
          <td>24.900622</td>
          <td>21.877280</td>
          <td>20.913627</td>
          <td>24.020262</td>
          <td>24.954986</td>
          <td>25.000169</td>
          <td>23.435212</td>
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
          <td>20.767096</td>
          <td>0.005780</td>
          <td>20.112081</td>
          <td>0.005044</td>
          <td>28.437268</td>
          <td>0.581238</td>
          <td>20.354695</td>
          <td>0.005085</td>
          <td>22.587555</td>
          <td>0.012656</td>
          <td>24.093339</td>
          <td>0.102570</td>
          <td>22.214591</td>
          <td>22.288867</td>
          <td>22.411895</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.548076</td>
          <td>0.173426</td>
          <td>21.324346</td>
          <td>0.005233</td>
          <td>21.721423</td>
          <td>0.005304</td>
          <td>23.365627</td>
          <td>0.013090</td>
          <td>21.371825</td>
          <td>0.006367</td>
          <td>24.047744</td>
          <td>0.098555</td>
          <td>29.307208</td>
          <td>27.223049</td>
          <td>27.646343</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.970590</td>
          <td>0.043880</td>
          <td>21.357654</td>
          <td>0.005245</td>
          <td>23.084329</td>
          <td>0.007603</td>
          <td>24.046699</td>
          <td>0.022936</td>
          <td>26.387805</td>
          <td>0.326440</td>
          <td>26.538479</td>
          <td>0.719087</td>
          <td>23.558094</td>
          <td>24.043185</td>
          <td>22.231625</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.373833</td>
          <td>1.305813</td>
          <td>24.156724</td>
          <td>0.017908</td>
          <td>27.895698</td>
          <td>0.388374</td>
          <td>19.538931</td>
          <td>0.005026</td>
          <td>18.046653</td>
          <td>0.005010</td>
          <td>21.574818</td>
          <td>0.011802</td>
          <td>26.875664</td>
          <td>24.221688</td>
          <td>20.690827</td>
        </tr>
        <tr>
          <th>4</th>
          <td>14.697927</td>
          <td>0.005001</td>
          <td>24.871450</td>
          <td>0.033066</td>
          <td>24.351511</td>
          <td>0.018565</td>
          <td>23.464193</td>
          <td>0.014136</td>
          <td>25.390573</td>
          <td>0.142429</td>
          <td>24.107824</td>
          <td>0.103879</td>
          <td>31.108190</td>
          <td>25.642572</td>
          <td>17.718634</td>
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
          <td>20.178222</td>
          <td>0.005350</td>
          <td>19.247822</td>
          <td>0.005016</td>
          <td>25.310840</td>
          <td>0.042798</td>
          <td>18.852691</td>
          <td>0.005011</td>
          <td>17.650819</td>
          <td>0.005007</td>
          <td>21.314242</td>
          <td>0.009828</td>
          <td>18.455154</td>
          <td>22.855470</td>
          <td>27.711379</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.115663</td>
          <td>0.005322</td>
          <td>21.670823</td>
          <td>0.005394</td>
          <td>19.358906</td>
          <td>0.005010</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.917698</td>
          <td>0.038944</td>
          <td>23.732808</td>
          <td>0.074687</td>
          <td>21.536473</td>
          <td>19.263586</td>
          <td>21.960967</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.589711</td>
          <td>0.179647</td>
          <td>22.816339</td>
          <td>0.007251</td>
          <td>21.535967</td>
          <td>0.005226</td>
          <td>20.404896</td>
          <td>0.005091</td>
          <td>25.332921</td>
          <td>0.135522</td>
          <td>21.649276</td>
          <td>0.012476</td>
          <td>15.037090</td>
          <td>21.939859</td>
          <td>19.646113</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.615094</td>
          <td>0.005054</td>
          <td>24.862164</td>
          <td>0.032798</td>
          <td>23.769033</td>
          <td>0.011693</td>
          <td>21.736556</td>
          <td>0.005758</td>
          <td>21.513713</td>
          <td>0.006703</td>
          <td>20.454132</td>
          <td>0.006372</td>
          <td>26.849602</td>
          <td>26.677617</td>
          <td>25.977284</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.443178</td>
          <td>0.005499</td>
          <td>22.532627</td>
          <td>0.006476</td>
          <td>24.903899</td>
          <td>0.029875</td>
          <td>21.888543</td>
          <td>0.005966</td>
          <td>20.906804</td>
          <td>0.005654</td>
          <td>23.885761</td>
          <td>0.085479</td>
          <td>24.954986</td>
          <td>25.000169</td>
          <td>23.435212</td>
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
          <td>20.773542</td>
          <td>20.109167</td>
          <td>27.840134</td>
          <td>20.354303</td>
          <td>22.593589</td>
          <td>24.289279</td>
          <td>22.219061</td>
          <td>0.005658</td>
          <td>22.284166</td>
          <td>0.006988</td>
          <td>22.420035</td>
          <td>0.007457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.506606</td>
          <td>21.332445</td>
          <td>21.725287</td>
          <td>23.380683</td>
          <td>21.369849</td>
          <td>24.030432</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.071270</td>
          <td>0.342727</td>
          <td>27.370230</td>
          <td>0.432099</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.910567</td>
          <td>21.348195</td>
          <td>23.076603</td>
          <td>24.037825</td>
          <td>26.082534</td>
          <td>26.085767</td>
          <td>23.551643</td>
          <td>0.010310</td>
          <td>24.038199</td>
          <td>0.024879</td>
          <td>22.233510</td>
          <td>0.006835</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.898859</td>
          <td>24.169216</td>
          <td>29.160423</td>
          <td>19.539080</td>
          <td>18.046586</td>
          <td>21.578655</td>
          <td>26.949651</td>
          <td>0.189773</td>
          <td>24.198737</td>
          <td>0.028647</td>
          <td>20.689479</td>
          <td>0.005125</td>
        </tr>
        <tr>
          <th>4</th>
          <td>14.701788</td>
          <td>24.865995</td>
          <td>24.326521</td>
          <td>23.454226</td>
          <td>25.496923</td>
          <td>24.236635</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.733670</td>
          <td>0.111708</td>
          <td>17.716155</td>
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
          <td>20.178298</td>
          <td>19.246676</td>
          <td>25.358689</td>
          <td>18.849029</td>
          <td>17.655867</td>
          <td>21.317267</td>
          <td>18.453433</td>
          <td>0.005001</td>
          <td>22.852901</td>
          <td>0.009634</td>
          <td>30.578972</td>
          <td>2.541433</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.117711</td>
          <td>21.670792</td>
          <td>19.359187</td>
          <td>29.273090</td>
          <td>23.914363</td>
          <td>23.686170</td>
          <td>21.529532</td>
          <td>0.005193</td>
          <td>19.259162</td>
          <td>0.005009</td>
          <td>21.961374</td>
          <td>0.006177</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.537205</td>
          <td>22.810912</td>
          <td>21.539062</td>
          <td>20.418506</td>
          <td>25.464075</td>
          <td>21.654261</td>
          <td>15.044665</td>
          <td>0.005000</td>
          <td>21.939228</td>
          <td>0.006135</td>
          <td>19.649695</td>
          <td>0.005019</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.613021</td>
          <td>24.817423</td>
          <td>23.780940</td>
          <td>21.737709</td>
          <td>21.528368</td>
          <td>20.467685</td>
          <td>26.993872</td>
          <td>0.196983</td>
          <td>26.457775</td>
          <td>0.207844</td>
          <td>26.173138</td>
          <td>0.163345</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.436143</td>
          <td>22.526714</td>
          <td>24.900622</td>
          <td>21.877280</td>
          <td>20.913627</td>
          <td>24.020262</td>
          <td>24.978486</td>
          <td>0.033586</td>
          <td>24.962967</td>
          <td>0.056522</td>
          <td>23.434002</td>
          <td>0.014896</td>
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
          <td>20.773542</td>
          <td>20.109167</td>
          <td>27.840134</td>
          <td>20.354303</td>
          <td>22.593589</td>
          <td>24.289279</td>
          <td>22.203014</td>
          <td>0.031412</td>
          <td>22.333324</td>
          <td>0.029535</td>
          <td>22.371759</td>
          <td>0.033386</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.506606</td>
          <td>21.332445</td>
          <td>21.725287</td>
          <td>23.380683</td>
          <td>21.369849</td>
          <td>24.030432</td>
          <td>27.624281</td>
          <td>1.805164</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.910567</td>
          <td>21.348195</td>
          <td>23.076603</td>
          <td>24.037825</td>
          <td>26.082534</td>
          <td>26.085767</td>
          <td>23.491786</td>
          <td>0.098651</td>
          <td>23.917154</td>
          <td>0.120144</td>
          <td>22.306516</td>
          <td>0.031510</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.898859</td>
          <td>24.169216</td>
          <td>29.160423</td>
          <td>19.539080</td>
          <td>18.046586</td>
          <td>21.578655</td>
          <td>29.700696</td>
          <td>3.690170</td>
          <td>24.134277</td>
          <td>0.145000</td>
          <td>20.688902</td>
          <td>0.008670</td>
        </tr>
        <tr>
          <th>4</th>
          <td>14.701788</td>
          <td>24.865995</td>
          <td>24.326521</td>
          <td>23.454226</td>
          <td>25.496923</td>
          <td>24.236635</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.598133</td>
          <td>0.475772</td>
          <td>17.722555</td>
          <td>0.005021</td>
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
          <td>20.178298</td>
          <td>19.246676</td>
          <td>25.358689</td>
          <td>18.849029</td>
          <td>17.655867</td>
          <td>21.317267</td>
          <td>18.450485</td>
          <td>0.005097</td>
          <td>22.771433</td>
          <td>0.043587</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.117711</td>
          <td>21.670792</td>
          <td>19.359187</td>
          <td>29.273090</td>
          <td>23.914363</td>
          <td>23.686170</td>
          <td>21.542940</td>
          <td>0.017709</td>
          <td>19.258500</td>
          <td>0.005292</td>
          <td>21.984703</td>
          <td>0.023744</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.537205</td>
          <td>22.810912</td>
          <td>21.539062</td>
          <td>20.418506</td>
          <td>25.464075</td>
          <td>21.654261</td>
          <td>15.044033</td>
          <td>0.005000</td>
          <td>21.916981</td>
          <td>0.020536</td>
          <td>19.642259</td>
          <td>0.005684</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.613021</td>
          <td>24.817423</td>
          <td>23.780940</td>
          <td>21.737709</td>
          <td>21.528368</td>
          <td>20.467685</td>
          <td>26.143456</td>
          <td>0.801657</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.466215</td>
          <td>0.923632</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.436143</td>
          <td>22.526714</td>
          <td>24.900622</td>
          <td>21.877280</td>
          <td>20.913627</td>
          <td>24.020262</td>
          <td>24.780604</td>
          <td>0.294338</td>
          <td>24.711414</td>
          <td>0.236211</td>
          <td>23.578172</td>
          <td>0.097478</td>
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


