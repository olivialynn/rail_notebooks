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
          <td>22.976314</td>
          <td>15.237344</td>
          <td>22.261419</td>
          <td>21.057618</td>
          <td>24.462972</td>
          <td>25.786487</td>
          <td>20.573817</td>
          <td>19.784385</td>
          <td>26.814139</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.408759</td>
          <td>18.465648</td>
          <td>28.625058</td>
          <td>23.432489</td>
          <td>27.716107</td>
          <td>23.732086</td>
          <td>22.648875</td>
          <td>17.391585</td>
          <td>22.164940</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.214976</td>
          <td>23.971683</td>
          <td>21.094147</td>
          <td>27.237877</td>
          <td>23.565965</td>
          <td>18.473139</td>
          <td>19.874310</td>
          <td>21.214838</td>
          <td>19.565986</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.389760</td>
          <td>21.191483</td>
          <td>21.241283</td>
          <td>21.442380</td>
          <td>20.293576</td>
          <td>29.681273</td>
          <td>25.214597</td>
          <td>19.679727</td>
          <td>20.792901</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.960325</td>
          <td>22.837489</td>
          <td>23.999294</td>
          <td>24.627840</td>
          <td>24.248253</td>
          <td>22.208587</td>
          <td>23.645816</td>
          <td>17.726680</td>
          <td>23.683317</td>
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
          <td>21.838057</td>
          <td>22.003143</td>
          <td>25.010468</td>
          <td>22.345451</td>
          <td>20.107643</td>
          <td>26.083106</td>
          <td>24.847928</td>
          <td>23.178997</td>
          <td>23.496301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.818312</td>
          <td>23.450555</td>
          <td>22.629998</td>
          <td>19.064800</td>
          <td>24.599554</td>
          <td>26.109175</td>
          <td>25.933324</td>
          <td>18.051631</td>
          <td>18.522354</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916318</td>
          <td>21.321138</td>
          <td>22.972747</td>
          <td>25.652597</td>
          <td>21.839273</td>
          <td>23.592935</td>
          <td>16.486404</td>
          <td>18.521045</td>
          <td>22.016884</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178371</td>
          <td>25.786893</td>
          <td>21.864606</td>
          <td>22.140835</td>
          <td>25.863410</td>
          <td>15.453892</td>
          <td>22.496530</td>
          <td>22.090662</td>
          <td>21.187229</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.143385</td>
          <td>28.879582</td>
          <td>22.009183</td>
          <td>28.347195</td>
          <td>20.889330</td>
          <td>24.771826</td>
          <td>25.906532</td>
          <td>18.448945</td>
          <td>25.090969</td>
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
          <td>23.004385</td>
          <td>0.019168</td>
          <td>15.238781</td>
          <td>0.005000</td>
          <td>22.273998</td>
          <td>0.005740</td>
          <td>21.058064</td>
          <td>0.005254</td>
          <td>24.460667</td>
          <td>0.063045</td>
          <td>25.644133</td>
          <td>0.374796</td>
          <td>20.573817</td>
          <td>19.784385</td>
          <td>26.814139</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.412035</td>
          <td>0.005478</td>
          <td>18.460644</td>
          <td>0.005007</td>
          <td>27.810573</td>
          <td>0.363483</td>
          <td>23.450482</td>
          <td>0.013984</td>
          <td>26.554900</td>
          <td>0.372329</td>
          <td>23.557866</td>
          <td>0.063972</td>
          <td>22.648875</td>
          <td>17.391585</td>
          <td>22.164940</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.190774</td>
          <td>0.053250</td>
          <td>23.977542</td>
          <td>0.015474</td>
          <td>21.088971</td>
          <td>0.005112</td>
          <td>27.187144</td>
          <td>0.344080</td>
          <td>23.583300</td>
          <td>0.029002</td>
          <td>18.474982</td>
          <td>0.005062</td>
          <td>19.874310</td>
          <td>21.214838</td>
          <td>19.565986</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.400133</td>
          <td>0.005471</td>
          <td>21.194132</td>
          <td>0.005193</td>
          <td>21.234560</td>
          <td>0.005140</td>
          <td>21.433515</td>
          <td>0.005464</td>
          <td>20.288335</td>
          <td>0.005242</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.214597</td>
          <td>19.679727</td>
          <td>20.792901</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.682176</td>
          <td>0.433621</td>
          <td>22.825900</td>
          <td>0.007283</td>
          <td>24.010016</td>
          <td>0.014053</td>
          <td>24.592350</td>
          <td>0.036977</td>
          <td>24.212897</td>
          <td>0.050601</td>
          <td>22.217037</td>
          <td>0.019723</td>
          <td>23.645816</td>
          <td>17.726680</td>
          <td>23.683317</td>
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
          <td>21.849938</td>
          <td>0.008477</td>
          <td>21.998916</td>
          <td>0.005652</td>
          <td>25.000817</td>
          <td>0.032533</td>
          <td>22.347332</td>
          <td>0.006977</td>
          <td>20.105260</td>
          <td>0.005182</td>
          <td>26.324635</td>
          <td>0.620779</td>
          <td>24.847928</td>
          <td>23.178997</td>
          <td>23.496301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.814575</td>
          <td>0.008317</td>
          <td>23.442501</td>
          <td>0.010375</td>
          <td>22.620303</td>
          <td>0.006283</td>
          <td>19.066227</td>
          <td>0.005014</td>
          <td>24.546826</td>
          <td>0.068046</td>
          <td>25.434483</td>
          <td>0.317700</td>
          <td>25.933324</td>
          <td>18.051631</td>
          <td>18.522354</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.920703</td>
          <td>0.005966</td>
          <td>21.318676</td>
          <td>0.005231</td>
          <td>22.974428</td>
          <td>0.007211</td>
          <td>25.587438</td>
          <td>0.089279</td>
          <td>21.842576</td>
          <td>0.007786</td>
          <td>23.589342</td>
          <td>0.065782</td>
          <td>16.486404</td>
          <td>18.521045</td>
          <td>22.016884</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.180866</td>
          <td>0.005351</td>
          <td>25.572101</td>
          <td>0.061426</td>
          <td>21.874046</td>
          <td>0.005388</td>
          <td>22.138611</td>
          <td>0.006434</td>
          <td>26.392305</td>
          <td>0.327609</td>
          <td>15.448249</td>
          <td>0.005002</td>
          <td>22.496530</td>
          <td>22.090662</td>
          <td>21.187229</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.095687</td>
          <td>0.048977</td>
          <td>28.325018</td>
          <td>0.594416</td>
          <td>22.008439</td>
          <td>0.005482</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.890624</td>
          <td>0.005637</td>
          <td>24.941553</td>
          <td>0.212316</td>
          <td>25.906532</td>
          <td>18.448945</td>
          <td>25.090969</td>
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
          <td>22.976314</td>
          <td>15.237344</td>
          <td>22.261419</td>
          <td>21.057618</td>
          <td>24.462972</td>
          <td>25.786487</td>
          <td>20.577588</td>
          <td>0.005034</td>
          <td>19.788730</td>
          <td>0.005024</td>
          <td>27.116844</td>
          <td>0.355252</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.408759</td>
          <td>18.465648</td>
          <td>28.625058</td>
          <td>23.432489</td>
          <td>27.716107</td>
          <td>23.732086</td>
          <td>22.652632</td>
          <td>0.006369</td>
          <td>17.389604</td>
          <td>0.005000</td>
          <td>22.158919</td>
          <td>0.006628</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.214976</td>
          <td>23.971683</td>
          <td>21.094147</td>
          <td>27.237877</td>
          <td>23.565965</td>
          <td>18.473139</td>
          <td>19.879688</td>
          <td>0.005009</td>
          <td>21.217155</td>
          <td>0.005324</td>
          <td>19.560015</td>
          <td>0.005016</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.389760</td>
          <td>21.191483</td>
          <td>21.241283</td>
          <td>21.442380</td>
          <td>20.293576</td>
          <td>29.681273</td>
          <td>25.211504</td>
          <td>0.041320</td>
          <td>19.682188</td>
          <td>0.005020</td>
          <td>20.789531</td>
          <td>0.005150</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.960325</td>
          <td>22.837489</td>
          <td>23.999294</td>
          <td>24.627840</td>
          <td>24.248253</td>
          <td>22.208587</td>
          <td>23.657890</td>
          <td>0.011126</td>
          <td>17.715036</td>
          <td>0.005001</td>
          <td>23.662001</td>
          <td>0.017996</td>
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
          <td>21.838057</td>
          <td>22.003143</td>
          <td>25.010468</td>
          <td>22.345451</td>
          <td>20.107643</td>
          <td>26.083106</td>
          <td>24.828709</td>
          <td>0.029415</td>
          <td>23.176912</td>
          <td>0.012162</td>
          <td>23.499218</td>
          <td>0.015712</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.818312</td>
          <td>23.450555</td>
          <td>22.629998</td>
          <td>19.064800</td>
          <td>24.599554</td>
          <td>26.109175</td>
          <td>25.944666</td>
          <td>0.079335</td>
          <td>18.054330</td>
          <td>0.005001</td>
          <td>18.524743</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916318</td>
          <td>21.321138</td>
          <td>22.972747</td>
          <td>25.652597</td>
          <td>21.839273</td>
          <td>23.592935</td>
          <td>16.490601</td>
          <td>0.005000</td>
          <td>18.527343</td>
          <td>0.005002</td>
          <td>22.014009</td>
          <td>0.006285</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178371</td>
          <td>25.786893</td>
          <td>21.864606</td>
          <td>22.140835</td>
          <td>25.863410</td>
          <td>15.453892</td>
          <td>22.512131</td>
          <td>0.006084</td>
          <td>22.082379</td>
          <td>0.006438</td>
          <td>21.185708</td>
          <td>0.005306</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.143385</td>
          <td>28.879582</td>
          <td>22.009183</td>
          <td>28.347195</td>
          <td>20.889330</td>
          <td>24.771826</td>
          <td>25.817421</td>
          <td>0.070874</td>
          <td>18.443434</td>
          <td>0.005002</td>
          <td>25.067510</td>
          <td>0.062035</td>
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
          <td>22.976314</td>
          <td>15.237344</td>
          <td>22.261419</td>
          <td>21.057618</td>
          <td>24.462972</td>
          <td>25.786487</td>
          <td>20.562856</td>
          <td>0.008534</td>
          <td>19.778607</td>
          <td>0.005729</td>
          <td>26.615425</td>
          <td>1.011601</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.408759</td>
          <td>18.465648</td>
          <td>28.625058</td>
          <td>23.432489</td>
          <td>27.716107</td>
          <td>23.732086</td>
          <td>22.522171</td>
          <td>0.041715</td>
          <td>17.392147</td>
          <td>0.005010</td>
          <td>22.166383</td>
          <td>0.027842</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.214976</td>
          <td>23.971683</td>
          <td>21.094147</td>
          <td>27.237877</td>
          <td>23.565965</td>
          <td>18.473139</td>
          <td>19.871562</td>
          <td>0.006197</td>
          <td>21.243427</td>
          <td>0.011857</td>
          <td>19.563545</td>
          <td>0.005597</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.389760</td>
          <td>21.191483</td>
          <td>21.241283</td>
          <td>21.442380</td>
          <td>20.293576</td>
          <td>29.681273</td>
          <td>25.099812</td>
          <td>0.379047</td>
          <td>19.681190</td>
          <td>0.005616</td>
          <td>20.798391</td>
          <td>0.009292</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.960325</td>
          <td>22.837489</td>
          <td>23.999294</td>
          <td>24.627840</td>
          <td>24.248253</td>
          <td>22.208587</td>
          <td>23.608602</td>
          <td>0.109287</td>
          <td>17.727441</td>
          <td>0.005018</td>
          <td>23.694900</td>
          <td>0.107985</td>
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
          <td>21.838057</td>
          <td>22.003143</td>
          <td>25.010468</td>
          <td>22.345451</td>
          <td>20.107643</td>
          <td>26.083106</td>
          <td>24.773230</td>
          <td>0.292592</td>
          <td>23.142955</td>
          <td>0.060694</td>
          <td>23.743196</td>
          <td>0.112641</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.818312</td>
          <td>23.450555</td>
          <td>22.629998</td>
          <td>19.064800</td>
          <td>24.599554</td>
          <td>26.109175</td>
          <td>25.713507</td>
          <td>0.598412</td>
          <td>18.053288</td>
          <td>0.005032</td>
          <td>18.516902</td>
          <td>0.005091</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.916318</td>
          <td>21.321138</td>
          <td>22.972747</td>
          <td>25.652597</td>
          <td>21.839273</td>
          <td>23.592935</td>
          <td>16.484407</td>
          <td>0.005003</td>
          <td>18.520158</td>
          <td>0.005076</td>
          <td>22.034312</td>
          <td>0.024795</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.178371</td>
          <td>25.786893</td>
          <td>21.864606</td>
          <td>22.140835</td>
          <td>25.863410</td>
          <td>15.453892</td>
          <td>22.428835</td>
          <td>0.038387</td>
          <td>22.131491</td>
          <td>0.024734</td>
          <td>21.178481</td>
          <td>0.012176</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.143385</td>
          <td>28.879582</td>
          <td>22.009183</td>
          <td>28.347195</td>
          <td>20.889330</td>
          <td>24.771826</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.453251</td>
          <td>0.005068</td>
          <td>28.013620</td>
          <td>2.045234</td>
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


