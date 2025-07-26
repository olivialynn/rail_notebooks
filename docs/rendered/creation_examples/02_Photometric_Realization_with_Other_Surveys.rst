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
          <td>21.140797</td>
          <td>24.213059</td>
          <td>24.233564</td>
          <td>16.373993</td>
          <td>18.001652</td>
          <td>18.286621</td>
          <td>20.994188</td>
          <td>24.713549</td>
          <td>21.268815</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.595468</td>
          <td>21.495796</td>
          <td>22.850334</td>
          <td>19.428074</td>
          <td>19.379729</td>
          <td>23.029490</td>
          <td>19.575808</td>
          <td>26.001170</td>
          <td>20.564964</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.514821</td>
          <td>29.215412</td>
          <td>18.602363</td>
          <td>27.015566</td>
          <td>22.480447</td>
          <td>24.719299</td>
          <td>24.109562</td>
          <td>23.789267</td>
          <td>27.254934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.970308</td>
          <td>25.257545</td>
          <td>20.997965</td>
          <td>23.013340</td>
          <td>22.368784</td>
          <td>24.297614</td>
          <td>21.073587</td>
          <td>24.016394</td>
          <td>24.185595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.329683</td>
          <td>21.045527</td>
          <td>27.308779</td>
          <td>24.591263</td>
          <td>20.601829</td>
          <td>23.311688</td>
          <td>25.691292</td>
          <td>27.177233</td>
          <td>21.756006</td>
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
          <td>20.074828</td>
          <td>22.571505</td>
          <td>22.833553</td>
          <td>18.910686</td>
          <td>21.386457</td>
          <td>25.711767</td>
          <td>21.263572</td>
          <td>22.516688</td>
          <td>15.346702</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558313</td>
          <td>25.371288</td>
          <td>26.145671</td>
          <td>24.593467</td>
          <td>26.689451</td>
          <td>20.145742</td>
          <td>16.320384</td>
          <td>21.514989</td>
          <td>26.098367</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.536390</td>
          <td>18.432500</td>
          <td>25.483115</td>
          <td>21.673213</td>
          <td>15.147175</td>
          <td>22.055230</td>
          <td>23.442564</td>
          <td>25.609836</td>
          <td>22.486447</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.464702</td>
          <td>21.805805</td>
          <td>26.612127</td>
          <td>24.447908</td>
          <td>21.586938</td>
          <td>24.148251</td>
          <td>22.197352</td>
          <td>27.335803</td>
          <td>24.798231</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.318475</td>
          <td>23.096989</td>
          <td>20.497383</td>
          <td>19.438522</td>
          <td>24.453352</td>
          <td>23.802413</td>
          <td>20.706768</td>
          <td>22.534977</td>
          <td>19.111192</td>
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
          <td>21.141919</td>
          <td>0.006316</td>
          <td>24.208584</td>
          <td>0.018696</td>
          <td>24.230874</td>
          <td>0.016794</td>
          <td>16.370190</td>
          <td>0.005001</td>
          <td>17.997418</td>
          <td>0.005010</td>
          <td>18.281553</td>
          <td>0.005047</td>
          <td>20.994188</td>
          <td>24.713549</td>
          <td>21.268815</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.436115</td>
          <td>0.742953</td>
          <td>21.499100</td>
          <td>0.005303</td>
          <td>22.851678</td>
          <td>0.006836</td>
          <td>19.418425</td>
          <td>0.005022</td>
          <td>19.374203</td>
          <td>0.005060</td>
          <td>23.073592</td>
          <td>0.041627</td>
          <td>19.575808</td>
          <td>26.001170</td>
          <td>20.564964</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.456475</td>
          <td>0.067276</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.596142</td>
          <td>0.005004</td>
          <td>27.424248</td>
          <td>0.413723</td>
          <td>22.464235</td>
          <td>0.011544</td>
          <td>24.864515</td>
          <td>0.199045</td>
          <td>24.109562</td>
          <td>23.789267</td>
          <td>27.254934</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.963961</td>
          <td>0.043626</td>
          <td>25.198343</td>
          <td>0.044118</td>
          <td>20.997222</td>
          <td>0.005098</td>
          <td>23.027817</td>
          <td>0.010231</td>
          <td>22.369352</td>
          <td>0.010784</td>
          <td>24.434321</td>
          <td>0.137960</td>
          <td>21.073587</td>
          <td>24.016394</td>
          <td>24.185595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.347954</td>
          <td>0.006755</td>
          <td>21.044312</td>
          <td>0.005155</td>
          <td>27.214975</td>
          <td>0.224634</td>
          <td>24.564199</td>
          <td>0.036068</td>
          <td>20.607428</td>
          <td>0.005404</td>
          <td>23.319336</td>
          <td>0.051770</td>
          <td>25.691292</td>
          <td>27.177233</td>
          <td>21.756006</td>
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
          <td>20.072153</td>
          <td>0.005304</td>
          <td>22.564913</td>
          <td>0.006550</td>
          <td>22.838935</td>
          <td>0.006801</td>
          <td>18.915193</td>
          <td>0.005012</td>
          <td>21.399359</td>
          <td>0.006427</td>
          <td>25.241615</td>
          <td>0.271956</td>
          <td>21.263572</td>
          <td>22.516688</td>
          <td>15.346702</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.437481</td>
          <td>0.066161</td>
          <td>25.243545</td>
          <td>0.045918</td>
          <td>26.225168</td>
          <td>0.096132</td>
          <td>24.591320</td>
          <td>0.036944</td>
          <td>26.923471</td>
          <td>0.492728</td>
          <td>20.153662</td>
          <td>0.005854</td>
          <td>16.320384</td>
          <td>21.514989</td>
          <td>26.098367</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.571440</td>
          <td>0.074430</td>
          <td>18.425673</td>
          <td>0.005006</td>
          <td>25.566063</td>
          <td>0.053680</td>
          <td>21.673465</td>
          <td>0.005684</td>
          <td>15.147706</td>
          <td>0.005001</td>
          <td>22.052961</td>
          <td>0.017193</td>
          <td>23.442564</td>
          <td>25.609836</td>
          <td>22.486447</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.441569</td>
          <td>0.066399</td>
          <td>21.809313</td>
          <td>0.005487</td>
          <td>26.576444</td>
          <td>0.130577</td>
          <td>24.464849</td>
          <td>0.033039</td>
          <td>21.592741</td>
          <td>0.006921</td>
          <td>23.867862</td>
          <td>0.084142</td>
          <td>22.197352</td>
          <td>27.335803</td>
          <td>24.798231</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.309623</td>
          <td>0.006664</td>
          <td>23.094709</td>
          <td>0.008354</td>
          <td>20.494581</td>
          <td>0.005046</td>
          <td>19.446913</td>
          <td>0.005023</td>
          <td>24.568032</td>
          <td>0.069336</td>
          <td>23.769827</td>
          <td>0.077170</td>
          <td>20.706768</td>
          <td>22.534977</td>
          <td>19.111192</td>
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
          <td>21.140797</td>
          <td>24.213059</td>
          <td>24.233564</td>
          <td>16.373993</td>
          <td>18.001652</td>
          <td>18.286621</td>
          <td>20.999515</td>
          <td>0.005074</td>
          <td>24.764024</td>
          <td>0.047337</td>
          <td>21.268934</td>
          <td>0.005355</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.595468</td>
          <td>21.495796</td>
          <td>22.850334</td>
          <td>19.428074</td>
          <td>19.379729</td>
          <td>23.029490</td>
          <td>19.575492</td>
          <td>0.005005</td>
          <td>25.962909</td>
          <td>0.136340</td>
          <td>20.573518</td>
          <td>0.005101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.514821</td>
          <td>29.215412</td>
          <td>18.602363</td>
          <td>27.015566</td>
          <td>22.480447</td>
          <td>24.719299</td>
          <td>24.109394</td>
          <td>0.015844</td>
          <td>23.743083</td>
          <td>0.019277</td>
          <td>27.445607</td>
          <td>0.457427</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.970308</td>
          <td>25.257545</td>
          <td>20.997965</td>
          <td>23.013340</td>
          <td>22.368784</td>
          <td>24.297614</td>
          <td>21.077672</td>
          <td>0.005085</td>
          <td>24.005465</td>
          <td>0.024178</td>
          <td>24.183420</td>
          <td>0.028263</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.329683</td>
          <td>21.045527</td>
          <td>27.308779</td>
          <td>24.591263</td>
          <td>20.601829</td>
          <td>23.311688</td>
          <td>25.635639</td>
          <td>0.060300</td>
          <td>27.341671</td>
          <td>0.422806</td>
          <td>21.757609</td>
          <td>0.005835</td>
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
          <td>20.074828</td>
          <td>22.571505</td>
          <td>22.833553</td>
          <td>18.910686</td>
          <td>21.386457</td>
          <td>25.711767</td>
          <td>21.269245</td>
          <td>0.005120</td>
          <td>22.515102</td>
          <td>0.007839</td>
          <td>15.352379</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558313</td>
          <td>25.371288</td>
          <td>26.145671</td>
          <td>24.593467</td>
          <td>26.689451</td>
          <td>20.145742</td>
          <td>16.321547</td>
          <td>0.005000</td>
          <td>21.518278</td>
          <td>0.005552</td>
          <td>26.101335</td>
          <td>0.153605</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.536390</td>
          <td>18.432500</td>
          <td>25.483115</td>
          <td>21.673213</td>
          <td>15.147175</td>
          <td>22.055230</td>
          <td>23.465647</td>
          <td>0.009717</td>
          <td>25.450127</td>
          <td>0.087084</td>
          <td>22.474464</td>
          <td>0.007670</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.464702</td>
          <td>21.805805</td>
          <td>26.612127</td>
          <td>24.447908</td>
          <td>21.586938</td>
          <td>24.148251</td>
          <td>22.206107</td>
          <td>0.005643</td>
          <td>26.833687</td>
          <td>0.283382</td>
          <td>24.868777</td>
          <td>0.051971</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.318475</td>
          <td>23.096989</td>
          <td>20.497383</td>
          <td>19.438522</td>
          <td>24.453352</td>
          <td>23.802413</td>
          <td>20.712060</td>
          <td>0.005043</td>
          <td>22.544802</td>
          <td>0.007968</td>
          <td>19.111890</td>
          <td>0.005007</td>
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
          <td>21.140797</td>
          <td>24.213059</td>
          <td>24.233564</td>
          <td>16.373993</td>
          <td>18.001652</td>
          <td>18.286621</td>
          <td>21.000735</td>
          <td>0.011484</td>
          <td>24.536024</td>
          <td>0.204087</td>
          <td>21.276472</td>
          <td>0.013135</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.595468</td>
          <td>21.495796</td>
          <td>22.850334</td>
          <td>19.428074</td>
          <td>19.379729</td>
          <td>23.029490</td>
          <td>19.569042</td>
          <td>0.005717</td>
          <td>26.045507</td>
          <td>0.656311</td>
          <td>20.560055</td>
          <td>0.008037</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.514821</td>
          <td>29.215412</td>
          <td>18.602363</td>
          <td>27.015566</td>
          <td>22.480447</td>
          <td>24.719299</td>
          <td>24.092487</td>
          <td>0.166066</td>
          <td>23.714525</td>
          <td>0.100640</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.970308</td>
          <td>25.257545</td>
          <td>20.997965</td>
          <td>23.013340</td>
          <td>22.368784</td>
          <td>24.297614</td>
          <td>21.061257</td>
          <td>0.012018</td>
          <td>23.785821</td>
          <td>0.107130</td>
          <td>24.191055</td>
          <td>0.165864</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.329683</td>
          <td>21.045527</td>
          <td>27.308779</td>
          <td>24.591263</td>
          <td>20.601829</td>
          <td>23.311688</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.676605</td>
          <td>0.504266</td>
          <td>21.770037</td>
          <td>0.019726</td>
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
          <td>20.074828</td>
          <td>22.571505</td>
          <td>22.833553</td>
          <td>18.910686</td>
          <td>21.386457</td>
          <td>25.711767</td>
          <td>21.301516</td>
          <td>0.014509</td>
          <td>22.485916</td>
          <td>0.033808</td>
          <td>15.344636</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.558313</td>
          <td>25.371288</td>
          <td>26.145671</td>
          <td>24.593467</td>
          <td>26.689451</td>
          <td>20.145742</td>
          <td>16.317217</td>
          <td>0.005002</td>
          <td>21.485892</td>
          <td>0.014328</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.536390</td>
          <td>18.432500</td>
          <td>25.483115</td>
          <td>21.673213</td>
          <td>15.147175</td>
          <td>22.055230</td>
          <td>23.461049</td>
          <td>0.096022</td>
          <td>26.854209</td>
          <td>1.097818</td>
          <td>22.488105</td>
          <td>0.037021</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.464702</td>
          <td>21.805805</td>
          <td>26.612127</td>
          <td>24.447908</td>
          <td>21.586938</td>
          <td>24.148251</td>
          <td>22.205270</td>
          <td>0.031475</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.540720</td>
          <td>0.222751</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.318475</td>
          <td>23.096989</td>
          <td>20.497383</td>
          <td>19.438522</td>
          <td>24.453352</td>
          <td>23.802413</td>
          <td>20.703288</td>
          <td>0.009322</td>
          <td>22.490507</td>
          <td>0.033946</td>
          <td>19.114857</td>
          <td>0.005270</td>
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


