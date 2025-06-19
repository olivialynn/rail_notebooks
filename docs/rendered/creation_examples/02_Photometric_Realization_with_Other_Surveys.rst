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
          <td>19.978765</td>
          <td>24.806629</td>
          <td>16.638694</td>
          <td>20.244856</td>
          <td>26.692374</td>
          <td>24.155987</td>
          <td>18.765479</td>
          <td>21.902792</td>
          <td>14.554621</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.901003</td>
          <td>22.049032</td>
          <td>21.556436</td>
          <td>21.639552</td>
          <td>23.165709</td>
          <td>19.413198</td>
          <td>18.157593</td>
          <td>26.464283</td>
          <td>18.525654</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.452693</td>
          <td>18.739237</td>
          <td>24.894999</td>
          <td>22.792536</td>
          <td>17.913057</td>
          <td>20.140808</td>
          <td>22.298881</td>
          <td>22.250315</td>
          <td>26.552971</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.696564</td>
          <td>22.742419</td>
          <td>24.186944</td>
          <td>24.297405</td>
          <td>24.418867</td>
          <td>21.807196</td>
          <td>25.320334</td>
          <td>23.932970</td>
          <td>19.638828</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.133185</td>
          <td>22.702593</td>
          <td>20.379300</td>
          <td>25.607187</td>
          <td>29.015359</td>
          <td>29.315011</td>
          <td>20.890907</td>
          <td>26.505135</td>
          <td>18.516007</td>
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
          <td>24.881074</td>
          <td>22.064416</td>
          <td>25.014096</td>
          <td>20.937880</td>
          <td>25.313413</td>
          <td>23.289246</td>
          <td>21.204410</td>
          <td>27.801522</td>
          <td>17.313106</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.493976</td>
          <td>21.226449</td>
          <td>23.838705</td>
          <td>22.296022</td>
          <td>20.466794</td>
          <td>24.137737</td>
          <td>23.499953</td>
          <td>25.031445</td>
          <td>25.147330</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.594490</td>
          <td>25.836610</td>
          <td>21.753790</td>
          <td>22.792434</td>
          <td>27.107535</td>
          <td>26.084213</td>
          <td>22.117996</td>
          <td>16.097135</td>
          <td>23.993457</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.563074</td>
          <td>22.611652</td>
          <td>21.294632</td>
          <td>25.416113</td>
          <td>24.186504</td>
          <td>26.344367</td>
          <td>21.911512</td>
          <td>21.493844</td>
          <td>21.128738</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.959950</td>
          <td>27.621450</td>
          <td>21.054500</td>
          <td>26.863410</td>
          <td>22.292281</td>
          <td>25.502317</td>
          <td>24.464675</td>
          <td>23.734977</td>
          <td>18.785409</td>
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
          <td>19.978564</td>
          <td>0.005269</td>
          <td>24.774016</td>
          <td>0.030360</td>
          <td>16.637015</td>
          <td>0.005001</td>
          <td>20.249963</td>
          <td>0.005072</td>
          <td>30.798945</td>
          <td>3.324962</td>
          <td>24.155834</td>
          <td>0.108330</td>
          <td>18.765479</td>
          <td>21.902792</td>
          <td>14.554621</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.894998</td>
          <td>0.008690</td>
          <td>22.037903</td>
          <td>0.005692</td>
          <td>21.558974</td>
          <td>0.005234</td>
          <td>21.646215</td>
          <td>0.005655</td>
          <td>23.158251</td>
          <td>0.020091</td>
          <td>19.415799</td>
          <td>0.005262</td>
          <td>18.157593</td>
          <td>26.464283</td>
          <td>18.525654</td>
        </tr>
        <tr>
          <th>2</th>
          <td>inf</td>
          <td>inf</td>
          <td>18.739897</td>
          <td>0.005009</td>
          <td>24.966354</td>
          <td>0.031561</td>
          <td>22.806580</td>
          <td>0.008865</td>
          <td>17.910988</td>
          <td>0.005009</td>
          <td>20.138327</td>
          <td>0.005833</td>
          <td>22.298881</td>
          <td>22.250315</td>
          <td>26.552971</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.746896</td>
          <td>0.036079</td>
          <td>22.727076</td>
          <td>0.006974</td>
          <td>24.199760</td>
          <td>0.016370</td>
          <td>24.266229</td>
          <td>0.027751</td>
          <td>24.567249</td>
          <td>0.069288</td>
          <td>21.812956</td>
          <td>0.014159</td>
          <td>25.320334</td>
          <td>23.932970</td>
          <td>19.638828</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.126170</td>
          <td>0.021201</td>
          <td>22.698633</td>
          <td>0.006893</td>
          <td>20.378479</td>
          <td>0.005039</td>
          <td>25.601112</td>
          <td>0.090359</td>
          <td>27.439044</td>
          <td>0.710083</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.890907</td>
          <td>26.505135</td>
          <td>18.516007</td>
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
          <td>24.862274</td>
          <td>0.096028</td>
          <td>22.076337</td>
          <td>0.005734</td>
          <td>24.975390</td>
          <td>0.031813</td>
          <td>20.936389</td>
          <td>0.005209</td>
          <td>25.434449</td>
          <td>0.147907</td>
          <td>23.311499</td>
          <td>0.051411</td>
          <td>21.204410</td>
          <td>27.801522</td>
          <td>17.313106</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.866985</td>
          <td>0.977585</td>
          <td>21.221105</td>
          <td>0.005200</td>
          <td>23.855988</td>
          <td>0.012476</td>
          <td>22.290192</td>
          <td>0.006812</td>
          <td>20.462182</td>
          <td>0.005320</td>
          <td>24.384498</td>
          <td>0.132149</td>
          <td>23.499953</td>
          <td>25.031445</td>
          <td>25.147330</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.608644</td>
          <td>0.031992</td>
          <td>26.008201</td>
          <td>0.090250</td>
          <td>21.754886</td>
          <td>0.005320</td>
          <td>22.790682</td>
          <td>0.008780</td>
          <td>26.848923</td>
          <td>0.466133</td>
          <td>25.683498</td>
          <td>0.386429</td>
          <td>22.117996</td>
          <td>16.097135</td>
          <td>23.993457</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.571071</td>
          <td>0.005594</td>
          <td>22.626966</td>
          <td>0.006701</td>
          <td>21.290965</td>
          <td>0.005153</td>
          <td>25.461051</td>
          <td>0.079870</td>
          <td>24.165677</td>
          <td>0.048524</td>
          <td>26.146213</td>
          <td>0.546680</td>
          <td>21.911512</td>
          <td>21.493844</td>
          <td>21.128738</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.221717</td>
          <td>0.642060</td>
          <td>27.736424</td>
          <td>0.383778</td>
          <td>21.058810</td>
          <td>0.005107</td>
          <td>26.589293</td>
          <td>0.211535</td>
          <td>22.295092</td>
          <td>0.010242</td>
          <td>25.441790</td>
          <td>0.319557</td>
          <td>24.464675</td>
          <td>23.734977</td>
          <td>18.785409</td>
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
          <td>19.978765</td>
          <td>24.806629</td>
          <td>16.638694</td>
          <td>20.244856</td>
          <td>26.692374</td>
          <td>24.155987</td>
          <td>18.761831</td>
          <td>0.005001</td>
          <td>21.904949</td>
          <td>0.006071</td>
          <td>14.551018</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.901003</td>
          <td>22.049032</td>
          <td>21.556436</td>
          <td>21.639552</td>
          <td>23.165709</td>
          <td>19.413198</td>
          <td>18.151517</td>
          <td>0.005000</td>
          <td>26.132351</td>
          <td>0.157744</td>
          <td>18.526220</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.452693</td>
          <td>18.739237</td>
          <td>24.894999</td>
          <td>22.792536</td>
          <td>17.913057</td>
          <td>20.140808</td>
          <td>22.294535</td>
          <td>0.005749</td>
          <td>22.260416</td>
          <td>0.006915</td>
          <td>26.712319</td>
          <td>0.256684</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.696564</td>
          <td>22.742419</td>
          <td>24.186944</td>
          <td>24.297405</td>
          <td>24.418867</td>
          <td>21.807196</td>
          <td>25.328794</td>
          <td>0.045874</td>
          <td>23.969274</td>
          <td>0.023427</td>
          <td>19.640001</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.133185</td>
          <td>22.702593</td>
          <td>20.379300</td>
          <td>25.607187</td>
          <td>29.015359</td>
          <td>29.315011</td>
          <td>20.891242</td>
          <td>0.005060</td>
          <td>26.377505</td>
          <td>0.194285</td>
          <td>18.516586</td>
          <td>0.005002</td>
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
          <td>24.881074</td>
          <td>22.064416</td>
          <td>25.014096</td>
          <td>20.937880</td>
          <td>25.313413</td>
          <td>23.289246</td>
          <td>21.202989</td>
          <td>0.005107</td>
          <td>27.398607</td>
          <td>0.441497</td>
          <td>17.317497</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.493976</td>
          <td>21.226449</td>
          <td>23.838705</td>
          <td>22.296022</td>
          <td>20.466794</td>
          <td>24.137737</td>
          <td>23.493489</td>
          <td>0.009902</td>
          <td>25.023021</td>
          <td>0.059627</td>
          <td>25.109870</td>
          <td>0.064417</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.594490</td>
          <td>25.836610</td>
          <td>21.753790</td>
          <td>22.792434</td>
          <td>27.107535</td>
          <td>26.084213</td>
          <td>22.113615</td>
          <td>0.005547</td>
          <td>16.102273</td>
          <td>0.005000</td>
          <td>23.987334</td>
          <td>0.023799</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.563074</td>
          <td>22.611652</td>
          <td>21.294632</td>
          <td>25.416113</td>
          <td>24.186504</td>
          <td>26.344367</td>
          <td>21.919525</td>
          <td>0.005389</td>
          <td>21.486268</td>
          <td>0.005522</td>
          <td>21.133112</td>
          <td>0.005279</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.959950</td>
          <td>27.621450</td>
          <td>21.054500</td>
          <td>26.863410</td>
          <td>22.292281</td>
          <td>25.502317</td>
          <td>24.501572</td>
          <td>0.022090</td>
          <td>23.721076</td>
          <td>0.018919</td>
          <td>18.781864</td>
          <td>0.005004</td>
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
          <td>19.978765</td>
          <td>24.806629</td>
          <td>16.638694</td>
          <td>20.244856</td>
          <td>26.692374</td>
          <td>24.155987</td>
          <td>18.764863</td>
          <td>0.005172</td>
          <td>21.880810</td>
          <td>0.019908</td>
          <td>14.552183</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.901003</td>
          <td>22.049032</td>
          <td>21.556436</td>
          <td>21.639552</td>
          <td>23.165709</td>
          <td>19.413198</td>
          <td>18.160376</td>
          <td>0.005057</td>
          <td>25.505541</td>
          <td>0.443819</td>
          <td>18.523144</td>
          <td>0.005092</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.452693</td>
          <td>18.739237</td>
          <td>24.894999</td>
          <td>22.792536</td>
          <td>17.913057</td>
          <td>20.140808</td>
          <td>22.322867</td>
          <td>0.034935</td>
          <td>22.312719</td>
          <td>0.029002</td>
          <td>27.032474</td>
          <td>1.282892</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.696564</td>
          <td>22.742419</td>
          <td>24.186944</td>
          <td>24.297405</td>
          <td>24.418867</td>
          <td>21.807196</td>
          <td>24.646342</td>
          <td>0.263935</td>
          <td>23.846850</td>
          <td>0.113001</td>
          <td>19.636863</td>
          <td>0.005678</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.133185</td>
          <td>22.702593</td>
          <td>20.379300</td>
          <td>25.607187</td>
          <td>29.015359</td>
          <td>29.315011</td>
          <td>20.882909</td>
          <td>0.010540</td>
          <td>29.736636</td>
          <td>3.532012</td>
          <td>18.513806</td>
          <td>0.005091</td>
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
          <td>24.881074</td>
          <td>22.064416</td>
          <td>25.014096</td>
          <td>20.937880</td>
          <td>25.313413</td>
          <td>23.289246</td>
          <td>21.188184</td>
          <td>0.013256</td>
          <td>26.887004</td>
          <td>1.118796</td>
          <td>17.314027</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.493976</td>
          <td>21.226449</td>
          <td>23.838705</td>
          <td>22.296022</td>
          <td>20.466794</td>
          <td>24.137737</td>
          <td>23.304815</td>
          <td>0.083668</td>
          <td>24.702660</td>
          <td>0.234505</td>
          <td>25.006205</td>
          <td>0.325499</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.594490</td>
          <td>25.836610</td>
          <td>21.753790</td>
          <td>22.792434</td>
          <td>27.107535</td>
          <td>26.084213</td>
          <td>22.087034</td>
          <td>0.028353</td>
          <td>16.098094</td>
          <td>0.005001</td>
          <td>24.285393</td>
          <td>0.179727</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.563074</td>
          <td>22.611652</td>
          <td>21.294632</td>
          <td>25.416113</td>
          <td>24.186504</td>
          <td>26.344367</td>
          <td>21.893654</td>
          <td>0.023930</td>
          <td>21.488209</td>
          <td>0.014355</td>
          <td>21.115777</td>
          <td>0.011613</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.959950</td>
          <td>27.621450</td>
          <td>21.054500</td>
          <td>26.863410</td>
          <td>22.292281</td>
          <td>25.502317</td>
          <td>23.984606</td>
          <td>0.151415</td>
          <td>23.696656</td>
          <td>0.099074</td>
          <td>18.782872</td>
          <td>0.005148</td>
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


