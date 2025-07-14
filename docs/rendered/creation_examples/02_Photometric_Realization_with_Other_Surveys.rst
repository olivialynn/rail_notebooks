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
          <td>25.915798</td>
          <td>22.105948</td>
          <td>21.975290</td>
          <td>21.860914</td>
          <td>16.227986</td>
          <td>26.055708</td>
          <td>20.861817</td>
          <td>23.806940</td>
          <td>21.419205</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.088963</td>
          <td>21.763451</td>
          <td>24.910353</td>
          <td>25.593992</td>
          <td>24.070189</td>
          <td>22.506283</td>
          <td>21.960769</td>
          <td>20.308368</td>
          <td>26.458699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.700704</td>
          <td>24.935824</td>
          <td>24.962549</td>
          <td>22.959740</td>
          <td>22.349959</td>
          <td>22.756217</td>
          <td>24.303815</td>
          <td>25.865115</td>
          <td>22.364486</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.087939</td>
          <td>27.144283</td>
          <td>18.275348</td>
          <td>24.776088</td>
          <td>16.364994</td>
          <td>23.828070</td>
          <td>26.080507</td>
          <td>16.742970</td>
          <td>25.254365</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.279279</td>
          <td>21.895514</td>
          <td>22.290652</td>
          <td>15.860101</td>
          <td>26.871495</td>
          <td>21.271808</td>
          <td>28.676756</td>
          <td>22.216704</td>
          <td>20.529464</td>
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
          <td>27.545084</td>
          <td>25.507629</td>
          <td>22.940584</td>
          <td>21.038288</td>
          <td>21.305685</td>
          <td>29.201838</td>
          <td>21.109376</td>
          <td>25.357947</td>
          <td>21.042384</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.411495</td>
          <td>24.951063</td>
          <td>30.995662</td>
          <td>22.591597</td>
          <td>24.059103</td>
          <td>23.176530</td>
          <td>26.871192</td>
          <td>17.555367</td>
          <td>30.239544</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.269188</td>
          <td>22.035514</td>
          <td>21.429996</td>
          <td>25.447610</td>
          <td>25.114602</td>
          <td>29.763816</td>
          <td>23.620510</td>
          <td>23.616132</td>
          <td>23.594620</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.119019</td>
          <td>26.677347</td>
          <td>23.133373</td>
          <td>25.603393</td>
          <td>26.894120</td>
          <td>20.733535</td>
          <td>26.089971</td>
          <td>19.982572</td>
          <td>24.910075</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.835087</td>
          <td>23.766389</td>
          <td>31.205786</td>
          <td>19.605013</td>
          <td>20.165794</td>
          <td>24.053547</td>
          <td>21.398476</td>
          <td>24.358043</td>
          <td>31.180583</td>
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
          <td>25.783221</td>
          <td>0.211347</td>
          <td>22.091595</td>
          <td>0.005752</td>
          <td>21.973230</td>
          <td>0.005456</td>
          <td>21.857579</td>
          <td>0.005920</td>
          <td>16.220515</td>
          <td>0.005001</td>
          <td>25.743607</td>
          <td>0.404771</td>
          <td>20.861817</td>
          <td>23.806940</td>
          <td>21.419205</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.092950</td>
          <td>0.005031</td>
          <td>21.766688</td>
          <td>0.005456</td>
          <td>24.932223</td>
          <td>0.030628</td>
          <td>25.821934</td>
          <td>0.109650</td>
          <td>24.025264</td>
          <td>0.042839</td>
          <td>22.508863</td>
          <td>0.025332</td>
          <td>21.960769</td>
          <td>20.308368</td>
          <td>26.458699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.714708</td>
          <td>2.386406</td>
          <td>24.915192</td>
          <td>0.034362</td>
          <td>24.880693</td>
          <td>0.029273</td>
          <td>22.964348</td>
          <td>0.009802</td>
          <td>22.335718</td>
          <td>0.010533</td>
          <td>22.750608</td>
          <td>0.031293</td>
          <td>24.303815</td>
          <td>25.865115</td>
          <td>22.364486</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.583318</td>
          <td>0.818343</td>
          <td>26.885152</td>
          <td>0.192320</td>
          <td>18.278295</td>
          <td>0.005003</td>
          <td>24.817567</td>
          <td>0.045147</td>
          <td>16.368455</td>
          <td>0.005002</td>
          <td>23.804070</td>
          <td>0.079539</td>
          <td>26.080507</td>
          <td>16.742970</td>
          <td>25.254365</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.286602</td>
          <td>0.011097</td>
          <td>21.892936</td>
          <td>0.005554</td>
          <td>22.292281</td>
          <td>0.005762</td>
          <td>15.855187</td>
          <td>0.005000</td>
          <td>27.194017</td>
          <td>0.599353</td>
          <td>21.277746</td>
          <td>0.009595</td>
          <td>28.676756</td>
          <td>22.216704</td>
          <td>20.529464</td>
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
          <td>27.279366</td>
          <td>0.668153</td>
          <td>25.498895</td>
          <td>0.057571</td>
          <td>22.946685</td>
          <td>0.007121</td>
          <td>21.040368</td>
          <td>0.005247</td>
          <td>21.301397</td>
          <td>0.006225</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.109376</td>
          <td>25.357947</td>
          <td>21.042384</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.421815</td>
          <td>0.027230</td>
          <td>24.841626</td>
          <td>0.032212</td>
          <td>28.060749</td>
          <td>0.440671</td>
          <td>22.584537</td>
          <td>0.007814</td>
          <td>24.107752</td>
          <td>0.046092</td>
          <td>23.131375</td>
          <td>0.043815</td>
          <td>26.871192</td>
          <td>17.555367</td>
          <td>30.239544</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.256698</td>
          <td>0.010874</td>
          <td>22.026893</td>
          <td>0.005680</td>
          <td>21.427782</td>
          <td>0.005190</td>
          <td>25.465314</td>
          <td>0.080171</td>
          <td>24.928693</td>
          <td>0.095298</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.620510</td>
          <td>23.616132</td>
          <td>23.594620</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.098555</td>
          <td>0.049101</td>
          <td>26.727184</td>
          <td>0.168243</td>
          <td>23.132358</td>
          <td>0.007793</td>
          <td>25.561652</td>
          <td>0.087276</td>
          <td>25.887265</td>
          <td>0.217057</td>
          <td>20.740337</td>
          <td>0.007126</td>
          <td>26.089971</td>
          <td>19.982572</td>
          <td>24.910075</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.838711</td>
          <td>0.005225</td>
          <td>23.761186</td>
          <td>0.013064</td>
          <td>30.306511</td>
          <td>1.738603</td>
          <td>19.598096</td>
          <td>0.005029</td>
          <td>20.165645</td>
          <td>0.005200</td>
          <td>23.952996</td>
          <td>0.090689</td>
          <td>21.398476</td>
          <td>24.358043</td>
          <td>31.180583</td>
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
          <td>25.915798</td>
          <td>22.105948</td>
          <td>21.975290</td>
          <td>21.860914</td>
          <td>16.227986</td>
          <td>26.055708</td>
          <td>20.867656</td>
          <td>0.005058</td>
          <td>23.804561</td>
          <td>0.020318</td>
          <td>21.424633</td>
          <td>0.005468</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.088963</td>
          <td>21.763451</td>
          <td>24.910353</td>
          <td>25.593992</td>
          <td>24.070189</td>
          <td>22.506283</td>
          <td>21.957473</td>
          <td>0.005416</td>
          <td>20.311880</td>
          <td>0.005063</td>
          <td>26.156657</td>
          <td>0.161060</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.700704</td>
          <td>24.935824</td>
          <td>24.962549</td>
          <td>22.959740</td>
          <td>22.349959</td>
          <td>22.756217</td>
          <td>24.346050</td>
          <td>0.019326</td>
          <td>25.757169</td>
          <td>0.114024</td>
          <td>22.365046</td>
          <td>0.007257</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.087939</td>
          <td>27.144283</td>
          <td>18.275348</td>
          <td>24.776088</td>
          <td>16.364994</td>
          <td>23.828070</td>
          <td>26.037781</td>
          <td>0.086140</td>
          <td>16.740873</td>
          <td>0.005000</td>
          <td>25.165134</td>
          <td>0.067659</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.279279</td>
          <td>21.895514</td>
          <td>22.290652</td>
          <td>15.860101</td>
          <td>26.871495</td>
          <td>21.271808</td>
          <td>28.189488</td>
          <td>0.509070</td>
          <td>22.202768</td>
          <td>0.006747</td>
          <td>20.529634</td>
          <td>0.005093</td>
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
          <td>27.545084</td>
          <td>25.507629</td>
          <td>22.940584</td>
          <td>21.038288</td>
          <td>21.305685</td>
          <td>29.201838</td>
          <td>21.100420</td>
          <td>0.005089</td>
          <td>25.404170</td>
          <td>0.083621</td>
          <td>21.039701</td>
          <td>0.005236</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.411495</td>
          <td>24.951063</td>
          <td>30.995662</td>
          <td>22.591597</td>
          <td>24.059103</td>
          <td>23.176530</td>
          <td>26.997831</td>
          <td>0.197640</td>
          <td>17.554126</td>
          <td>0.005000</td>
          <td>29.337015</td>
          <td>1.502742</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.269188</td>
          <td>22.035514</td>
          <td>21.429996</td>
          <td>25.447610</td>
          <td>25.114602</td>
          <td>29.763816</td>
          <td>23.618471</td>
          <td>0.010812</td>
          <td>23.620113</td>
          <td>0.017373</td>
          <td>23.586905</td>
          <td>0.016897</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.119019</td>
          <td>26.677347</td>
          <td>23.133373</td>
          <td>25.603393</td>
          <td>26.894120</td>
          <td>20.733535</td>
          <td>26.127159</td>
          <td>0.093201</td>
          <td>19.987432</td>
          <td>0.005035</td>
          <td>24.962287</td>
          <td>0.056487</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.835087</td>
          <td>23.766389</td>
          <td>31.205786</td>
          <td>19.605013</td>
          <td>20.165794</td>
          <td>24.053547</td>
          <td>21.392870</td>
          <td>0.005151</td>
          <td>24.326950</td>
          <td>0.032085</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>25.915798</td>
          <td>22.105948</td>
          <td>21.975290</td>
          <td>21.860914</td>
          <td>16.227986</td>
          <td>26.055708</td>
          <td>20.869367</td>
          <td>0.010439</td>
          <td>23.870436</td>
          <td>0.115351</td>
          <td>21.399825</td>
          <td>0.014490</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.088963</td>
          <td>21.763451</td>
          <td>24.910353</td>
          <td>25.593992</td>
          <td>24.070189</td>
          <td>22.506283</td>
          <td>21.958330</td>
          <td>0.025321</td>
          <td>20.307719</td>
          <td>0.006761</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.700704</td>
          <td>24.935824</td>
          <td>24.962549</td>
          <td>22.959740</td>
          <td>22.349959</td>
          <td>22.756217</td>
          <td>24.610386</td>
          <td>0.256278</td>
          <td>25.375529</td>
          <td>0.401911</td>
          <td>22.334736</td>
          <td>0.032307</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.087939</td>
          <td>27.144283</td>
          <td>18.275348</td>
          <td>24.776088</td>
          <td>16.364994</td>
          <td>23.828070</td>
          <td>25.822362</td>
          <td>0.645873</td>
          <td>16.748213</td>
          <td>0.005003</td>
          <td>25.882130</td>
          <td>0.628019</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.279279</td>
          <td>21.895514</td>
          <td>22.290652</td>
          <td>15.860101</td>
          <td>26.871495</td>
          <td>21.271808</td>
          <td>27.065402</td>
          <td>1.376741</td>
          <td>22.201707</td>
          <td>0.026303</td>
          <td>20.535836</td>
          <td>0.007929</td>
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
          <td>27.545084</td>
          <td>25.507629</td>
          <td>22.940584</td>
          <td>21.038288</td>
          <td>21.305685</td>
          <td>29.201838</td>
          <td>21.098769</td>
          <td>0.012367</td>
          <td>25.909869</td>
          <td>0.596872</td>
          <td>21.047532</td>
          <td>0.011043</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.411495</td>
          <td>24.951063</td>
          <td>30.995662</td>
          <td>22.591597</td>
          <td>24.059103</td>
          <td>23.176530</td>
          <td>27.087175</td>
          <td>1.392431</td>
          <td>17.552374</td>
          <td>0.005013</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.269188</td>
          <td>22.035514</td>
          <td>21.429996</td>
          <td>25.447610</td>
          <td>25.114602</td>
          <td>29.763816</td>
          <td>23.503347</td>
          <td>0.099658</td>
          <td>23.782759</td>
          <td>0.106843</td>
          <td>23.601094</td>
          <td>0.099461</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.119019</td>
          <td>26.677347</td>
          <td>23.133373</td>
          <td>25.603393</td>
          <td>26.894120</td>
          <td>20.733535</td>
          <td>26.303398</td>
          <td>0.888092</td>
          <td>19.983086</td>
          <td>0.006033</td>
          <td>25.399893</td>
          <td>0.441927</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.835087</td>
          <td>23.766389</td>
          <td>31.205786</td>
          <td>19.605013</td>
          <td>20.165794</td>
          <td>24.053547</td>
          <td>21.370612</td>
          <td>0.015348</td>
          <td>24.366782</td>
          <td>0.176910</td>
          <td>inf</td>
          <td>inf</td>
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


