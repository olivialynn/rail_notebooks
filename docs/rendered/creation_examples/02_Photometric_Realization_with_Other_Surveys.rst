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
          <td>23.790220</td>
          <td>23.675417</td>
          <td>23.153200</td>
          <td>22.999895</td>
          <td>24.522497</td>
          <td>22.287376</td>
          <td>27.638444</td>
          <td>25.248904</td>
          <td>19.941539</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.728669</td>
          <td>25.326203</td>
          <td>23.410710</td>
          <td>24.272311</td>
          <td>27.150043</td>
          <td>24.301872</td>
          <td>22.736707</td>
          <td>21.119980</td>
          <td>23.804822</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.782314</td>
          <td>24.161604</td>
          <td>21.891643</td>
          <td>24.775808</td>
          <td>25.645490</td>
          <td>17.456446</td>
          <td>20.692022</td>
          <td>24.377430</td>
          <td>27.490459</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.469389</td>
          <td>24.699077</td>
          <td>24.197041</td>
          <td>27.939690</td>
          <td>26.078043</td>
          <td>25.060010</td>
          <td>20.903719</td>
          <td>17.432699</td>
          <td>24.948555</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.378300</td>
          <td>18.994520</td>
          <td>21.570982</td>
          <td>27.123408</td>
          <td>26.097261</td>
          <td>23.027784</td>
          <td>26.897399</td>
          <td>24.219014</td>
          <td>27.491108</td>
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
          <td>27.347334</td>
          <td>22.259014</td>
          <td>22.556064</td>
          <td>22.174826</td>
          <td>20.561073</td>
          <td>20.561853</td>
          <td>25.735927</td>
          <td>21.392983</td>
          <td>25.470794</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.939337</td>
          <td>20.437328</td>
          <td>22.561912</td>
          <td>28.210319</td>
          <td>24.578172</td>
          <td>22.943648</td>
          <td>20.186286</td>
          <td>22.377023</td>
          <td>22.694742</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.243619</td>
          <td>18.926255</td>
          <td>22.089410</td>
          <td>25.932153</td>
          <td>25.646382</td>
          <td>27.230090</td>
          <td>23.460711</td>
          <td>19.416132</td>
          <td>25.956851</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.310916</td>
          <td>27.861844</td>
          <td>22.942203</td>
          <td>20.853702</td>
          <td>23.724812</td>
          <td>19.637152</td>
          <td>21.911944</td>
          <td>23.820535</td>
          <td>21.577762</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.824902</td>
          <td>23.071150</td>
          <td>23.662127</td>
          <td>19.833728</td>
          <td>21.428526</td>
          <td>21.101201</td>
          <td>26.847140</td>
          <td>22.302961</td>
          <td>21.003955</td>
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
          <td>23.715444</td>
          <td>0.035103</td>
          <td>23.656052</td>
          <td>0.012074</td>
          <td>23.149784</td>
          <td>0.007865</td>
          <td>23.001291</td>
          <td>0.010048</td>
          <td>24.587890</td>
          <td>0.070565</td>
          <td>22.311507</td>
          <td>0.021371</td>
          <td>27.638444</td>
          <td>25.248904</td>
          <td>19.941539</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.729289</td>
          <td>0.005740</td>
          <td>25.277369</td>
          <td>0.047314</td>
          <td>23.395676</td>
          <td>0.009064</td>
          <td>24.243898</td>
          <td>0.027215</td>
          <td>26.648800</td>
          <td>0.400419</td>
          <td>24.191039</td>
          <td>0.111710</td>
          <td>22.736707</td>
          <td>21.119980</td>
          <td>23.804822</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.718211</td>
          <td>0.084658</td>
          <td>24.138928</td>
          <td>0.017646</td>
          <td>21.891344</td>
          <td>0.005399</td>
          <td>24.757252</td>
          <td>0.042795</td>
          <td>25.885425</td>
          <td>0.216724</td>
          <td>17.464200</td>
          <td>0.005016</td>
          <td>20.692022</td>
          <td>24.377430</td>
          <td>27.490459</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.456601</td>
          <td>0.012511</td>
          <td>24.701749</td>
          <td>0.028504</td>
          <td>24.212964</td>
          <td>0.016548</td>
          <td>27.918469</td>
          <td>0.595795</td>
          <td>26.957169</td>
          <td>0.505138</td>
          <td>25.285575</td>
          <td>0.281842</td>
          <td>20.903719</td>
          <td>17.432699</td>
          <td>24.948555</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.403414</td>
          <td>0.026805</td>
          <td>18.994719</td>
          <td>0.005012</td>
          <td>21.568026</td>
          <td>0.005238</td>
          <td>27.056701</td>
          <td>0.310195</td>
          <td>25.822564</td>
          <td>0.205631</td>
          <td>23.045869</td>
          <td>0.040616</td>
          <td>26.897399</td>
          <td>24.219014</td>
          <td>27.491108</td>
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
          <td>29.002768</td>
          <td>1.780844</td>
          <td>22.261414</td>
          <td>0.005977</td>
          <td>22.563246</td>
          <td>0.006173</td>
          <td>22.173499</td>
          <td>0.006514</td>
          <td>20.551123</td>
          <td>0.005369</td>
          <td>20.555946</td>
          <td>0.006606</td>
          <td>25.735927</td>
          <td>21.392983</td>
          <td>25.470794</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.950756</td>
          <td>0.018346</td>
          <td>20.433513</td>
          <td>0.005066</td>
          <td>22.557392</td>
          <td>0.006162</td>
          <td>28.271758</td>
          <td>0.759154</td>
          <td>24.511354</td>
          <td>0.065941</td>
          <td>22.944767</td>
          <td>0.037139</td>
          <td>20.186286</td>
          <td>22.377023</td>
          <td>22.694742</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.234202</td>
          <td>0.005108</td>
          <td>18.929575</td>
          <td>0.005011</td>
          <td>22.085380</td>
          <td>0.005546</td>
          <td>26.096371</td>
          <td>0.139137</td>
          <td>25.557166</td>
          <td>0.164290</td>
          <td>26.591596</td>
          <td>0.745133</td>
          <td>23.460711</td>
          <td>19.416132</td>
          <td>25.956851</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.188209</td>
          <td>0.127454</td>
          <td>27.960336</td>
          <td>0.455349</td>
          <td>22.949331</td>
          <td>0.007129</td>
          <td>20.854325</td>
          <td>0.005183</td>
          <td>23.774003</td>
          <td>0.034298</td>
          <td>19.623139</td>
          <td>0.005364</td>
          <td>21.911944</td>
          <td>23.820535</td>
          <td>21.577762</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.808877</td>
          <td>0.038085</td>
          <td>23.075373</td>
          <td>0.008264</td>
          <td>23.677589</td>
          <td>0.010944</td>
          <td>19.834491</td>
          <td>0.005040</td>
          <td>21.422510</td>
          <td>0.006479</td>
          <td>21.089871</td>
          <td>0.008540</td>
          <td>26.847140</td>
          <td>22.302961</td>
          <td>21.003955</td>
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
          <td>23.790220</td>
          <td>23.675417</td>
          <td>23.153200</td>
          <td>22.999895</td>
          <td>24.522497</td>
          <td>22.287376</td>
          <td>28.211143</td>
          <td>0.517226</td>
          <td>25.290471</td>
          <td>0.075618</td>
          <td>19.938819</td>
          <td>0.005032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.728669</td>
          <td>25.326203</td>
          <td>23.410710</td>
          <td>24.272311</td>
          <td>27.150043</td>
          <td>24.301872</td>
          <td>22.742243</td>
          <td>0.006585</td>
          <td>21.127639</td>
          <td>0.005276</td>
          <td>23.807349</td>
          <td>0.020367</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.782314</td>
          <td>24.161604</td>
          <td>21.891643</td>
          <td>24.775808</td>
          <td>25.645490</td>
          <td>17.456446</td>
          <td>20.704606</td>
          <td>0.005043</td>
          <td>24.425391</td>
          <td>0.035014</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.469389</td>
          <td>24.699077</td>
          <td>24.197041</td>
          <td>27.939690</td>
          <td>26.078043</td>
          <td>25.060010</td>
          <td>20.897050</td>
          <td>0.005061</td>
          <td>17.430236</td>
          <td>0.005000</td>
          <td>24.988223</td>
          <td>0.057807</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.378300</td>
          <td>18.994520</td>
          <td>21.570982</td>
          <td>27.123408</td>
          <td>26.097261</td>
          <td>23.027784</td>
          <td>27.013472</td>
          <td>0.200258</td>
          <td>24.211579</td>
          <td>0.028973</td>
          <td>28.228650</td>
          <td>0.793952</td>
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
          <td>27.347334</td>
          <td>22.259014</td>
          <td>22.556064</td>
          <td>22.174826</td>
          <td>20.561073</td>
          <td>20.561853</td>
          <td>25.655613</td>
          <td>0.061381</td>
          <td>21.393478</td>
          <td>0.005443</td>
          <td>25.456045</td>
          <td>0.087539</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.939337</td>
          <td>20.437328</td>
          <td>22.561912</td>
          <td>28.210319</td>
          <td>24.578172</td>
          <td>22.943648</td>
          <td>20.186620</td>
          <td>0.005017</td>
          <td>22.365768</td>
          <td>0.007259</td>
          <td>22.690481</td>
          <td>0.008678</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.243619</td>
          <td>18.926255</td>
          <td>22.089410</td>
          <td>25.932153</td>
          <td>25.646382</td>
          <td>27.230090</td>
          <td>23.462820</td>
          <td>0.009698</td>
          <td>19.411350</td>
          <td>0.005012</td>
          <td>25.833333</td>
          <td>0.121848</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.310916</td>
          <td>27.861844</td>
          <td>22.942203</td>
          <td>20.853702</td>
          <td>23.724812</td>
          <td>19.637152</td>
          <td>21.917141</td>
          <td>0.005387</td>
          <td>23.819627</td>
          <td>0.020582</td>
          <td>21.571983</td>
          <td>0.005606</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.824902</td>
          <td>23.071150</td>
          <td>23.662127</td>
          <td>19.833728</td>
          <td>21.428526</td>
          <td>21.101201</td>
          <td>26.685921</td>
          <td>0.151586</td>
          <td>22.306795</td>
          <td>0.007060</td>
          <td>21.002307</td>
          <td>0.005220</td>
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
          <td>23.790220</td>
          <td>23.675417</td>
          <td>23.153200</td>
          <td>22.999895</td>
          <td>24.522497</td>
          <td>22.287376</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.317131</td>
          <td>0.787992</td>
          <td>19.950029</td>
          <td>0.006155</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.728669</td>
          <td>25.326203</td>
          <td>23.410710</td>
          <td>24.272311</td>
          <td>27.150043</td>
          <td>24.301872</td>
          <td>22.707379</td>
          <td>0.049203</td>
          <td>21.131399</td>
          <td>0.010914</td>
          <td>24.103250</td>
          <td>0.153858</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.782314</td>
          <td>24.161604</td>
          <td>21.891643</td>
          <td>24.775808</td>
          <td>25.645490</td>
          <td>17.456446</td>
          <td>20.688338</td>
          <td>0.009232</td>
          <td>24.386335</td>
          <td>0.179871</td>
          <td>26.066541</td>
          <td>0.712892</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.469389</td>
          <td>24.699077</td>
          <td>24.197041</td>
          <td>27.939690</td>
          <td>26.078043</td>
          <td>25.060010</td>
          <td>20.896976</td>
          <td>0.010646</td>
          <td>17.421790</td>
          <td>0.005010</td>
          <td>25.068358</td>
          <td>0.341940</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.378300</td>
          <td>18.994520</td>
          <td>21.570982</td>
          <td>27.123408</td>
          <td>26.097261</td>
          <td>23.027784</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.101553</td>
          <td>0.140967</td>
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
          <td>27.347334</td>
          <td>22.259014</td>
          <td>22.556064</td>
          <td>22.174826</td>
          <td>20.561073</td>
          <td>20.561853</td>
          <td>24.864308</td>
          <td>0.314802</td>
          <td>21.383928</td>
          <td>0.013212</td>
          <td>25.444507</td>
          <td>0.457049</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.939337</td>
          <td>20.437328</td>
          <td>22.561912</td>
          <td>28.210319</td>
          <td>24.578172</td>
          <td>22.943648</td>
          <td>20.197845</td>
          <td>0.007032</td>
          <td>22.366024</td>
          <td>0.030401</td>
          <td>22.656691</td>
          <td>0.043018</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.243619</td>
          <td>18.926255</td>
          <td>22.089410</td>
          <td>25.932153</td>
          <td>25.646382</td>
          <td>27.230090</td>
          <td>23.407893</td>
          <td>0.091633</td>
          <td>19.406491</td>
          <td>0.005380</td>
          <td>25.910624</td>
          <td>0.640626</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.310916</td>
          <td>27.861844</td>
          <td>22.942203</td>
          <td>20.853702</td>
          <td>23.724812</td>
          <td>19.637152</td>
          <td>21.858405</td>
          <td>0.023207</td>
          <td>23.955900</td>
          <td>0.124262</td>
          <td>21.584937</td>
          <td>0.016869</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.824902</td>
          <td>23.071150</td>
          <td>23.662127</td>
          <td>19.833728</td>
          <td>21.428526</td>
          <td>21.101201</td>
          <td>26.291122</td>
          <td>0.881251</td>
          <td>22.334700</td>
          <td>0.029571</td>
          <td>21.009958</td>
          <td>0.010746</td>
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


