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
          <td>22.021529</td>
          <td>23.421019</td>
          <td>23.831544</td>
          <td>22.703940</td>
          <td>23.406944</td>
          <td>25.249895</td>
          <td>21.547028</td>
          <td>18.989690</td>
          <td>17.049648</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.577055</td>
          <td>23.988245</td>
          <td>25.390733</td>
          <td>22.767275</td>
          <td>20.998126</td>
          <td>27.657868</td>
          <td>23.035786</td>
          <td>25.804694</td>
          <td>26.101191</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.057467</td>
          <td>20.021911</td>
          <td>19.881636</td>
          <td>22.035471</td>
          <td>26.386787</td>
          <td>19.766451</td>
          <td>22.392638</td>
          <td>23.442248</td>
          <td>21.845793</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.241538</td>
          <td>24.194937</td>
          <td>20.520920</td>
          <td>23.006823</td>
          <td>21.478939</td>
          <td>25.145753</td>
          <td>23.053634</td>
          <td>25.745308</td>
          <td>25.073268</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.530613</td>
          <td>26.708311</td>
          <td>18.218044</td>
          <td>20.374130</td>
          <td>24.822193</td>
          <td>19.371461</td>
          <td>23.737689</td>
          <td>21.460317</td>
          <td>18.332291</td>
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
          <td>25.276026</td>
          <td>26.165888</td>
          <td>25.379647</td>
          <td>21.115022</td>
          <td>22.204453</td>
          <td>24.949037</td>
          <td>23.071378</td>
          <td>26.365277</td>
          <td>27.157886</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.521794</td>
          <td>22.350207</td>
          <td>17.899396</td>
          <td>25.681503</td>
          <td>21.858857</td>
          <td>24.980588</td>
          <td>19.619387</td>
          <td>19.951149</td>
          <td>26.130456</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.079401</td>
          <td>25.552535</td>
          <td>24.871311</td>
          <td>20.943198</td>
          <td>21.499440</td>
          <td>26.314721</td>
          <td>25.483390</td>
          <td>20.595949</td>
          <td>24.761564</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.140791</td>
          <td>19.666603</td>
          <td>24.908061</td>
          <td>28.901955</td>
          <td>25.580978</td>
          <td>27.880479</td>
          <td>24.171086</td>
          <td>24.092345</td>
          <td>19.652858</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.545047</td>
          <td>24.910536</td>
          <td>21.710083</td>
          <td>18.436603</td>
          <td>25.877135</td>
          <td>24.329588</td>
          <td>25.639602</td>
          <td>19.580849</td>
          <td>22.619473</td>
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
          <td>22.003318</td>
          <td>0.009253</td>
          <td>23.409525</td>
          <td>0.010147</td>
          <td>23.847438</td>
          <td>0.012396</td>
          <td>22.696217</td>
          <td>0.008306</td>
          <td>23.385279</td>
          <td>0.024407</td>
          <td>24.874355</td>
          <td>0.200697</td>
          <td>21.547028</td>
          <td>18.989690</td>
          <td>17.049648</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.579603</td>
          <td>0.005601</td>
          <td>24.003026</td>
          <td>0.015795</td>
          <td>25.461997</td>
          <td>0.048942</td>
          <td>22.783812</td>
          <td>0.008743</td>
          <td>21.001741</td>
          <td>0.005761</td>
          <td>26.153846</td>
          <td>0.549705</td>
          <td>23.035786</td>
          <td>25.804694</td>
          <td>26.101191</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.054952</td>
          <td>0.006165</td>
          <td>20.023255</td>
          <td>0.005039</td>
          <td>19.880234</td>
          <td>0.005020</td>
          <td>22.031907</td>
          <td>0.006213</td>
          <td>25.841292</td>
          <td>0.208881</td>
          <td>19.767764</td>
          <td>0.005460</td>
          <td>22.392638</td>
          <td>23.442248</td>
          <td>21.845793</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.711343</td>
          <td>0.887954</td>
          <td>24.208715</td>
          <td>0.018698</td>
          <td>20.520793</td>
          <td>0.005048</td>
          <td>23.013316</td>
          <td>0.010130</td>
          <td>21.480369</td>
          <td>0.006618</td>
          <td>25.258360</td>
          <td>0.275685</td>
          <td>23.053634</td>
          <td>25.745308</td>
          <td>25.073268</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.533838</td>
          <td>0.007269</td>
          <td>26.663878</td>
          <td>0.159402</td>
          <td>18.217472</td>
          <td>0.005003</td>
          <td>20.375151</td>
          <td>0.005087</td>
          <td>24.660017</td>
          <td>0.075213</td>
          <td>19.366337</td>
          <td>0.005242</td>
          <td>23.737689</td>
          <td>21.460317</td>
          <td>18.332291</td>
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
          <td>25.278107</td>
          <td>0.137719</td>
          <td>26.177300</td>
          <td>0.104659</td>
          <td>25.397862</td>
          <td>0.046234</td>
          <td>21.128466</td>
          <td>0.005284</td>
          <td>22.205060</td>
          <td>0.009644</td>
          <td>24.860569</td>
          <td>0.198386</td>
          <td>23.071378</td>
          <td>26.365277</td>
          <td>27.157886</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.528656</td>
          <td>0.005561</td>
          <td>22.343294</td>
          <td>0.006107</td>
          <td>17.904795</td>
          <td>0.005002</td>
          <td>25.683273</td>
          <td>0.097121</td>
          <td>21.866219</td>
          <td>0.007883</td>
          <td>24.869661</td>
          <td>0.199907</td>
          <td>19.619387</td>
          <td>19.951149</td>
          <td>26.130456</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.117827</td>
          <td>0.049940</td>
          <td>25.448254</td>
          <td>0.055046</td>
          <td>24.872355</td>
          <td>0.029060</td>
          <td>20.943348</td>
          <td>0.005211</td>
          <td>21.492390</td>
          <td>0.006648</td>
          <td>26.164130</td>
          <td>0.553801</td>
          <td>25.483390</td>
          <td>20.595949</td>
          <td>24.761564</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.553799</td>
          <td>0.174269</td>
          <td>19.661722</td>
          <td>0.005025</td>
          <td>24.924088</td>
          <td>0.030410</td>
          <td>29.657314</td>
          <td>1.662977</td>
          <td>25.529619</td>
          <td>0.160472</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.171086</td>
          <td>24.092345</td>
          <td>19.652858</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.539946</td>
          <td>0.007288</td>
          <td>24.909459</td>
          <td>0.034190</td>
          <td>21.712117</td>
          <td>0.005299</td>
          <td>18.432728</td>
          <td>0.005007</td>
          <td>25.670238</td>
          <td>0.180865</td>
          <td>24.456747</td>
          <td>0.140653</td>
          <td>25.639602</td>
          <td>19.580849</td>
          <td>22.619473</td>
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
          <td>22.021529</td>
          <td>23.421019</td>
          <td>23.831544</td>
          <td>22.703940</td>
          <td>23.406944</td>
          <td>25.249895</td>
          <td>21.549713</td>
          <td>0.005200</td>
          <td>18.991278</td>
          <td>0.005006</td>
          <td>17.048676</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.577055</td>
          <td>23.988245</td>
          <td>25.390733</td>
          <td>22.767275</td>
          <td>20.998126</td>
          <td>27.657868</td>
          <td>23.033772</td>
          <td>0.007509</td>
          <td>25.990271</td>
          <td>0.139601</td>
          <td>26.000733</td>
          <td>0.140867</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.057467</td>
          <td>20.021911</td>
          <td>19.881636</td>
          <td>22.035471</td>
          <td>26.386787</td>
          <td>19.766451</td>
          <td>22.393786</td>
          <td>0.005888</td>
          <td>23.457351</td>
          <td>0.015182</td>
          <td>21.839696</td>
          <td>0.005960</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.241538</td>
          <td>24.194937</td>
          <td>20.520920</td>
          <td>23.006823</td>
          <td>21.478939</td>
          <td>25.145753</td>
          <td>23.068647</td>
          <td>0.007646</td>
          <td>25.878767</td>
          <td>0.126755</td>
          <td>25.077303</td>
          <td>0.062577</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.530613</td>
          <td>26.708311</td>
          <td>18.218044</td>
          <td>20.374130</td>
          <td>24.822193</td>
          <td>19.371461</td>
          <td>23.732759</td>
          <td>0.011762</td>
          <td>21.460889</td>
          <td>0.005499</td>
          <td>18.338198</td>
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
          <td>25.276026</td>
          <td>26.165888</td>
          <td>25.379647</td>
          <td>21.115022</td>
          <td>22.204453</td>
          <td>24.949037</td>
          <td>23.073074</td>
          <td>0.007664</td>
          <td>26.401517</td>
          <td>0.198254</td>
          <td>26.592144</td>
          <td>0.232471</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.521794</td>
          <td>22.350207</td>
          <td>17.899396</td>
          <td>25.681503</td>
          <td>21.858857</td>
          <td>24.980588</td>
          <td>19.621320</td>
          <td>0.005006</td>
          <td>19.947471</td>
          <td>0.005032</td>
          <td>26.258160</td>
          <td>0.175619</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.079401</td>
          <td>25.552535</td>
          <td>24.871311</td>
          <td>20.943198</td>
          <td>21.499440</td>
          <td>26.314721</td>
          <td>25.501432</td>
          <td>0.053506</td>
          <td>20.598539</td>
          <td>0.005106</td>
          <td>24.810602</td>
          <td>0.049345</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.140791</td>
          <td>19.666603</td>
          <td>24.908061</td>
          <td>28.901955</td>
          <td>25.580978</td>
          <td>27.880479</td>
          <td>24.169359</td>
          <td>0.016651</td>
          <td>24.116737</td>
          <td>0.026652</td>
          <td>19.652128</td>
          <td>0.005019</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.545047</td>
          <td>24.910536</td>
          <td>21.710083</td>
          <td>18.436603</td>
          <td>25.877135</td>
          <td>24.329588</td>
          <td>25.576709</td>
          <td>0.057218</td>
          <td>19.585430</td>
          <td>0.005017</td>
          <td>22.617470</td>
          <td>0.008306</td>
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
          <td>22.021529</td>
          <td>23.421019</td>
          <td>23.831544</td>
          <td>22.703940</td>
          <td>23.406944</td>
          <td>25.249895</td>
          <td>21.552988</td>
          <td>0.017860</td>
          <td>18.993997</td>
          <td>0.005181</td>
          <td>17.044290</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.577055</td>
          <td>23.988245</td>
          <td>25.390733</td>
          <td>22.767275</td>
          <td>20.998126</td>
          <td>27.657868</td>
          <td>23.022471</td>
          <td>0.065142</td>
          <td>25.952717</td>
          <td>0.615199</td>
          <td>25.789018</td>
          <td>0.588103</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.057467</td>
          <td>20.021911</td>
          <td>19.881636</td>
          <td>22.035471</td>
          <td>26.386787</td>
          <td>19.766451</td>
          <td>22.458820</td>
          <td>0.039426</td>
          <td>23.403989</td>
          <td>0.076529</td>
          <td>21.848199</td>
          <td>0.021095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.241538</td>
          <td>24.194937</td>
          <td>20.520920</td>
          <td>23.006823</td>
          <td>21.478939</td>
          <td>25.145753</td>
          <td>23.088288</td>
          <td>0.069065</td>
          <td>25.804004</td>
          <td>0.553356</td>
          <td>25.807687</td>
          <td>0.595950</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.530613</td>
          <td>26.708311</td>
          <td>18.218044</td>
          <td>20.374130</td>
          <td>24.822193</td>
          <td>19.371461</td>
          <td>23.904966</td>
          <td>0.141383</td>
          <td>21.441029</td>
          <td>0.013822</td>
          <td>18.340079</td>
          <td>0.005066</td>
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
          <td>25.276026</td>
          <td>26.165888</td>
          <td>25.379647</td>
          <td>21.115022</td>
          <td>22.204453</td>
          <td>24.949037</td>
          <td>23.051862</td>
          <td>0.066866</td>
          <td>26.123922</td>
          <td>0.692585</td>
          <td>27.850149</td>
          <td>1.908257</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.521794</td>
          <td>22.350207</td>
          <td>17.899396</td>
          <td>25.681503</td>
          <td>21.858857</td>
          <td>24.980588</td>
          <td>19.617600</td>
          <td>0.005779</td>
          <td>19.944274</td>
          <td>0.005967</td>
          <td>25.902999</td>
          <td>0.637234</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.079401</td>
          <td>25.552535</td>
          <td>24.871311</td>
          <td>20.943198</td>
          <td>21.499440</td>
          <td>26.314721</td>
          <td>25.320646</td>
          <td>0.448910</td>
          <td>20.593738</td>
          <td>0.007749</td>
          <td>24.935946</td>
          <td>0.307735</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.140791</td>
          <td>19.666603</td>
          <td>24.908061</td>
          <td>28.901955</td>
          <td>25.580978</td>
          <td>27.880479</td>
          <td>24.320486</td>
          <td>0.201442</td>
          <td>24.673995</td>
          <td>0.228998</td>
          <td>19.648110</td>
          <td>0.005691</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.545047</td>
          <td>24.910536</td>
          <td>21.710083</td>
          <td>18.436603</td>
          <td>25.877135</td>
          <td>24.329588</td>
          <td>27.896173</td>
          <td>2.030457</td>
          <td>19.576310</td>
          <td>0.005512</td>
          <td>22.584856</td>
          <td>0.040351</td>
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


