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
          <td>17.253686</td>
          <td>20.581017</td>
          <td>20.894196</td>
          <td>29.783857</td>
          <td>24.532602</td>
          <td>21.443881</td>
          <td>16.185152</td>
          <td>26.303009</td>
          <td>25.504736</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.684318</td>
          <td>22.487910</td>
          <td>23.006787</td>
          <td>22.232885</td>
          <td>23.972656</td>
          <td>19.621381</td>
          <td>21.417333</td>
          <td>25.075169</td>
          <td>22.291840</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.101733</td>
          <td>21.909094</td>
          <td>23.989030</td>
          <td>20.952363</td>
          <td>22.230752</td>
          <td>21.185498</td>
          <td>23.503716</td>
          <td>18.812763</td>
          <td>20.109746</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.471259</td>
          <td>22.987603</td>
          <td>23.833263</td>
          <td>25.147798</td>
          <td>25.899339</td>
          <td>22.845383</td>
          <td>17.192534</td>
          <td>24.047280</td>
          <td>28.019000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.943556</td>
          <td>21.138836</td>
          <td>25.389235</td>
          <td>21.480903</td>
          <td>27.525508</td>
          <td>21.388140</td>
          <td>26.461952</td>
          <td>20.671908</td>
          <td>23.677934</td>
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
          <td>22.198749</td>
          <td>17.762992</td>
          <td>24.737502</td>
          <td>24.420177</td>
          <td>23.304506</td>
          <td>26.033592</td>
          <td>21.466589</td>
          <td>20.964485</td>
          <td>24.850737</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.034416</td>
          <td>26.569314</td>
          <td>21.990821</td>
          <td>22.395460</td>
          <td>24.719533</td>
          <td>23.034772</td>
          <td>17.485257</td>
          <td>23.970158</td>
          <td>24.149235</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.861052</td>
          <td>23.128162</td>
          <td>18.797581</td>
          <td>20.075632</td>
          <td>22.702853</td>
          <td>24.412095</td>
          <td>25.958731</td>
          <td>28.784883</td>
          <td>25.833020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.837654</td>
          <td>19.106595</td>
          <td>16.765136</td>
          <td>18.035115</td>
          <td>25.038968</td>
          <td>20.315218</td>
          <td>26.622670</td>
          <td>22.439124</td>
          <td>19.803157</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.335881</td>
          <td>22.912424</td>
          <td>26.109306</td>
          <td>22.563357</td>
          <td>22.782685</td>
          <td>20.264307</td>
          <td>18.841587</td>
          <td>22.460830</td>
          <td>22.228550</td>
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
          <td>17.254317</td>
          <td>0.005013</td>
          <td>20.588784</td>
          <td>0.005082</td>
          <td>20.892532</td>
          <td>0.005083</td>
          <td>28.784351</td>
          <td>1.046893</td>
          <td>24.371133</td>
          <td>0.058232</td>
          <td>21.445360</td>
          <td>0.010750</td>
          <td>16.185152</td>
          <td>26.303009</td>
          <td>25.504736</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.676940</td>
          <td>0.014759</td>
          <td>22.491164</td>
          <td>0.006387</td>
          <td>23.003707</td>
          <td>0.007310</td>
          <td>22.236670</td>
          <td>0.006669</td>
          <td>24.024486</td>
          <td>0.042810</td>
          <td>19.619907</td>
          <td>0.005363</td>
          <td>21.417333</td>
          <td>25.075169</td>
          <td>22.291840</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.097016</td>
          <td>0.020692</td>
          <td>21.910038</td>
          <td>0.005568</td>
          <td>23.985998</td>
          <td>0.013790</td>
          <td>20.958545</td>
          <td>0.005216</td>
          <td>22.223362</td>
          <td>0.009760</td>
          <td>21.176160</td>
          <td>0.008995</td>
          <td>23.503716</td>
          <td>18.812763</td>
          <td>20.109746</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.478760</td>
          <td>0.005524</td>
          <td>22.992880</td>
          <td>0.007904</td>
          <td>23.832656</td>
          <td>0.012259</td>
          <td>25.168731</td>
          <td>0.061663</td>
          <td>25.735270</td>
          <td>0.191083</td>
          <td>22.870421</td>
          <td>0.034777</td>
          <td>17.192534</td>
          <td>24.047280</td>
          <td>28.019000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.986961</td>
          <td>0.044516</td>
          <td>21.141841</td>
          <td>0.005178</td>
          <td>25.432476</td>
          <td>0.047676</td>
          <td>21.482659</td>
          <td>0.005503</td>
          <td>27.881567</td>
          <td>0.944939</td>
          <td>21.380880</td>
          <td>0.010279</td>
          <td>26.461952</td>
          <td>20.671908</td>
          <td>23.677934</td>
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
          <td>22.210769</td>
          <td>0.010545</td>
          <td>17.761482</td>
          <td>0.005003</td>
          <td>24.731070</td>
          <td>0.025688</td>
          <td>24.485570</td>
          <td>0.033648</td>
          <td>23.310838</td>
          <td>0.022888</td>
          <td>25.402322</td>
          <td>0.309637</td>
          <td>21.466589</td>
          <td>20.964485</td>
          <td>24.850737</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.631931</td>
          <td>0.844334</td>
          <td>26.879053</td>
          <td>0.191335</td>
          <td>21.989737</td>
          <td>0.005468</td>
          <td>22.379505</td>
          <td>0.007075</td>
          <td>24.737183</td>
          <td>0.080517</td>
          <td>23.046535</td>
          <td>0.040640</td>
          <td>17.485257</td>
          <td>23.970158</td>
          <td>24.149235</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.164587</td>
          <td>1.163806</td>
          <td>23.118640</td>
          <td>0.008468</td>
          <td>18.787854</td>
          <td>0.005005</td>
          <td>20.074328</td>
          <td>0.005056</td>
          <td>22.693013</td>
          <td>0.013728</td>
          <td>24.372090</td>
          <td>0.130738</td>
          <td>25.958731</td>
          <td>28.784883</td>
          <td>25.833020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.849719</td>
          <td>0.005069</td>
          <td>19.105971</td>
          <td>0.005013</td>
          <td>16.753099</td>
          <td>0.005001</td>
          <td>18.039910</td>
          <td>0.005004</td>
          <td>25.077024</td>
          <td>0.108513</td>
          <td>20.320224</td>
          <td>0.006112</td>
          <td>26.622670</td>
          <td>22.439124</td>
          <td>19.803157</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.362362</td>
          <td>0.025880</td>
          <td>22.912602</td>
          <td>0.007589</td>
          <td>26.054210</td>
          <td>0.082709</td>
          <td>22.556698</td>
          <td>0.007701</td>
          <td>22.794688</td>
          <td>0.014878</td>
          <td>20.264639</td>
          <td>0.006018</td>
          <td>18.841587</td>
          <td>22.460830</td>
          <td>22.228550</td>
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
          <td>17.253686</td>
          <td>20.581017</td>
          <td>20.894196</td>
          <td>29.783857</td>
          <td>24.532602</td>
          <td>21.443881</td>
          <td>16.190337</td>
          <td>0.005000</td>
          <td>26.331686</td>
          <td>0.186914</td>
          <td>25.520298</td>
          <td>0.092640</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.684318</td>
          <td>22.487910</td>
          <td>23.006787</td>
          <td>22.232885</td>
          <td>23.972656</td>
          <td>19.621381</td>
          <td>21.424662</td>
          <td>0.005160</td>
          <td>25.166032</td>
          <td>0.067713</td>
          <td>22.292969</td>
          <td>0.007016</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.101733</td>
          <td>21.909094</td>
          <td>23.989030</td>
          <td>20.952363</td>
          <td>22.230752</td>
          <td>21.185498</td>
          <td>23.499014</td>
          <td>0.009940</td>
          <td>18.820683</td>
          <td>0.005004</td>
          <td>20.107059</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.471259</td>
          <td>22.987603</td>
          <td>23.833263</td>
          <td>25.147798</td>
          <td>25.899339</td>
          <td>22.845383</td>
          <td>17.192023</td>
          <td>0.005000</td>
          <td>24.040264</td>
          <td>0.024924</td>
          <td>28.538179</td>
          <td>0.965440</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.943556</td>
          <td>21.138836</td>
          <td>25.389235</td>
          <td>21.480903</td>
          <td>27.525508</td>
          <td>21.388140</td>
          <td>26.402307</td>
          <td>0.118601</td>
          <td>20.673331</td>
          <td>0.005121</td>
          <td>23.673213</td>
          <td>0.018167</td>
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
          <td>22.198749</td>
          <td>17.762992</td>
          <td>24.737502</td>
          <td>24.420177</td>
          <td>23.304506</td>
          <td>26.033592</td>
          <td>21.458228</td>
          <td>0.005170</td>
          <td>20.961192</td>
          <td>0.005204</td>
          <td>24.969476</td>
          <td>0.056850</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.034416</td>
          <td>26.569314</td>
          <td>21.990821</td>
          <td>22.395460</td>
          <td>24.719533</td>
          <td>23.034772</td>
          <td>17.482692</td>
          <td>0.005000</td>
          <td>23.980125</td>
          <td>0.023650</td>
          <td>24.151158</td>
          <td>0.027471</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.861052</td>
          <td>23.128162</td>
          <td>18.797581</td>
          <td>20.075632</td>
          <td>22.702853</td>
          <td>24.412095</td>
          <td>25.972417</td>
          <td>0.081306</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.068737</td>
          <td>0.149364</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.837654</td>
          <td>19.106595</td>
          <td>16.765136</td>
          <td>18.035115</td>
          <td>25.038968</td>
          <td>20.315218</td>
          <td>26.769456</td>
          <td>0.162832</td>
          <td>22.446000</td>
          <td>0.007556</td>
          <td>19.793217</td>
          <td>0.005024</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.335881</td>
          <td>22.912424</td>
          <td>26.109306</td>
          <td>22.563357</td>
          <td>22.782685</td>
          <td>20.264307</td>
          <td>18.840860</td>
          <td>0.005001</td>
          <td>22.475012</td>
          <td>0.007672</td>
          <td>22.223217</td>
          <td>0.006805</td>
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
          <td>17.253686</td>
          <td>20.581017</td>
          <td>20.894196</td>
          <td>29.783857</td>
          <td>24.532602</td>
          <td>21.443881</td>
          <td>16.184686</td>
          <td>0.005002</td>
          <td>24.852624</td>
          <td>0.265293</td>
          <td>24.932580</td>
          <td>0.306905</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.684318</td>
          <td>22.487910</td>
          <td>23.006787</td>
          <td>22.232885</td>
          <td>23.972656</td>
          <td>19.621381</td>
          <td>21.397589</td>
          <td>0.015691</td>
          <td>25.362809</td>
          <td>0.397992</td>
          <td>22.236460</td>
          <td>0.029617</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.101733</td>
          <td>21.909094</td>
          <td>23.989030</td>
          <td>20.952363</td>
          <td>22.230752</td>
          <td>21.185498</td>
          <td>23.529332</td>
          <td>0.101956</td>
          <td>18.811903</td>
          <td>0.005130</td>
          <td>20.120820</td>
          <td>0.006531</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.471259</td>
          <td>22.987603</td>
          <td>23.833263</td>
          <td>25.147798</td>
          <td>25.899339</td>
          <td>22.845383</td>
          <td>17.190271</td>
          <td>0.005010</td>
          <td>24.065774</td>
          <td>0.136678</td>
          <td>26.583948</td>
          <td>0.992632</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.943556</td>
          <td>21.138836</td>
          <td>25.389235</td>
          <td>21.480903</td>
          <td>27.525508</td>
          <td>21.388140</td>
          <td>27.092568</td>
          <td>1.396331</td>
          <td>20.677418</td>
          <td>0.008116</td>
          <td>23.758178</td>
          <td>0.114124</td>
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
          <td>22.198749</td>
          <td>17.762992</td>
          <td>24.737502</td>
          <td>24.420177</td>
          <td>23.304506</td>
          <td>26.033592</td>
          <td>21.461819</td>
          <td>0.016547</td>
          <td>20.964589</td>
          <td>0.009710</td>
          <td>24.830435</td>
          <td>0.282636</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.034416</td>
          <td>26.569314</td>
          <td>21.990821</td>
          <td>22.395460</td>
          <td>24.719533</td>
          <td>23.034772</td>
          <td>17.482976</td>
          <td>0.005016</td>
          <td>24.124118</td>
          <td>0.143737</td>
          <td>24.179914</td>
          <td>0.164294</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.861052</td>
          <td>23.128162</td>
          <td>18.797581</td>
          <td>20.075632</td>
          <td>22.702853</td>
          <td>24.412095</td>
          <td>25.230692</td>
          <td>0.419277</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.904801</td>
          <td>0.300132</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.837654</td>
          <td>19.106595</td>
          <td>16.765136</td>
          <td>18.035115</td>
          <td>25.038968</td>
          <td>20.315218</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.430459</td>
          <td>0.032185</td>
          <td>19.797809</td>
          <td>0.005894</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.335881</td>
          <td>22.912424</td>
          <td>26.109306</td>
          <td>22.563357</td>
          <td>22.782685</td>
          <td>20.264307</td>
          <td>18.841024</td>
          <td>0.005197</td>
          <td>22.413693</td>
          <td>0.031711</td>
          <td>22.180153</td>
          <td>0.028181</td>
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


