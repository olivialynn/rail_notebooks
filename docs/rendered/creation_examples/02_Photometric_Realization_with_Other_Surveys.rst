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
          <td>21.600285</td>
          <td>22.211212</td>
          <td>22.366020</td>
          <td>23.912386</td>
          <td>25.819375</td>
          <td>20.489041</td>
          <td>22.557040</td>
          <td>22.303042</td>
          <td>22.809665</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.004457</td>
          <td>20.295468</td>
          <td>23.061801</td>
          <td>20.854938</td>
          <td>23.375987</td>
          <td>26.162985</td>
          <td>25.581386</td>
          <td>19.851411</td>
          <td>32.049432</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.021675</td>
          <td>22.659454</td>
          <td>24.387666</td>
          <td>21.311826</td>
          <td>23.122928</td>
          <td>21.557232</td>
          <td>19.706379</td>
          <td>24.856198</td>
          <td>22.199455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.741081</td>
          <td>24.113025</td>
          <td>26.947669</td>
          <td>25.434167</td>
          <td>22.993714</td>
          <td>24.345914</td>
          <td>23.514285</td>
          <td>22.230996</td>
          <td>20.907478</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.905062</td>
          <td>16.423798</td>
          <td>20.817971</td>
          <td>20.074055</td>
          <td>24.464032</td>
          <td>21.620976</td>
          <td>21.617960</td>
          <td>28.736806</td>
          <td>22.901240</td>
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
          <td>28.734384</td>
          <td>21.803859</td>
          <td>28.377700</td>
          <td>20.464912</td>
          <td>24.483494</td>
          <td>24.216046</td>
          <td>24.832056</td>
          <td>19.824305</td>
          <td>21.856274</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.042235</td>
          <td>26.346415</td>
          <td>28.189524</td>
          <td>23.746700</td>
          <td>22.105478</td>
          <td>24.243824</td>
          <td>21.618190</td>
          <td>18.948769</td>
          <td>28.989164</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.380305</td>
          <td>20.338909</td>
          <td>24.608150</td>
          <td>21.483308</td>
          <td>22.082617</td>
          <td>18.090909</td>
          <td>25.522344</td>
          <td>23.767485</td>
          <td>24.033583</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.514965</td>
          <td>23.713411</td>
          <td>22.065376</td>
          <td>22.007360</td>
          <td>20.135617</td>
          <td>22.333707</td>
          <td>26.985481</td>
          <td>24.739030</td>
          <td>25.410448</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.382581</td>
          <td>20.341546</td>
          <td>25.069870</td>
          <td>21.293419</td>
          <td>20.065958</td>
          <td>21.112805</td>
          <td>23.075004</td>
          <td>25.032900</td>
          <td>28.973212</td>
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
          <td>21.582940</td>
          <td>0.007426</td>
          <td>22.211390</td>
          <td>0.005904</td>
          <td>22.364495</td>
          <td>0.005855</td>
          <td>23.937225</td>
          <td>0.020882</td>
          <td>26.443033</td>
          <td>0.341043</td>
          <td>20.494119</td>
          <td>0.006460</td>
          <td>22.557040</td>
          <td>22.303042</td>
          <td>22.809665</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.001183</td>
          <td>0.005082</td>
          <td>20.300324</td>
          <td>0.005056</td>
          <td>23.061884</td>
          <td>0.007518</td>
          <td>20.859213</td>
          <td>0.005185</td>
          <td>23.335043</td>
          <td>0.023371</td>
          <td>26.634411</td>
          <td>0.766599</td>
          <td>25.581386</td>
          <td>19.851411</td>
          <td>32.049432</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.070980</td>
          <td>0.115158</td>
          <td>22.663508</td>
          <td>0.006797</td>
          <td>24.362214</td>
          <td>0.018732</td>
          <td>21.315997</td>
          <td>0.005384</td>
          <td>23.140198</td>
          <td>0.019786</td>
          <td>21.562654</td>
          <td>0.011696</td>
          <td>19.706379</td>
          <td>24.856198</td>
          <td>22.199455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.746115</td>
          <td>0.008027</td>
          <td>24.100934</td>
          <td>0.017103</td>
          <td>27.267182</td>
          <td>0.234572</td>
          <td>25.474061</td>
          <td>0.080792</td>
          <td>23.017513</td>
          <td>0.017850</td>
          <td>24.388467</td>
          <td>0.132603</td>
          <td>23.514285</td>
          <td>22.230996</td>
          <td>20.907478</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.903853</td>
          <td>0.017661</td>
          <td>16.414458</td>
          <td>0.005001</td>
          <td>20.809513</td>
          <td>0.005074</td>
          <td>20.070133</td>
          <td>0.005056</td>
          <td>24.415314</td>
          <td>0.060560</td>
          <td>21.645236</td>
          <td>0.012438</td>
          <td>21.617960</td>
          <td>28.736806</td>
          <td>22.901240</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>21.801533</td>
          <td>0.005481</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.459915</td>
          <td>0.005099</td>
          <td>24.448889</td>
          <td>0.062390</td>
          <td>24.174821</td>
          <td>0.110140</td>
          <td>24.832056</td>
          <td>19.824305</td>
          <td>21.856274</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.969766</td>
          <td>0.105463</td>
          <td>26.646882</td>
          <td>0.157104</td>
          <td>27.754726</td>
          <td>0.347894</td>
          <td>23.754584</td>
          <td>0.017901</td>
          <td>22.109234</td>
          <td>0.009071</td>
          <td>24.344877</td>
          <td>0.127694</td>
          <td>21.618190</td>
          <td>18.948769</td>
          <td>28.989164</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.980350</td>
          <td>0.540961</td>
          <td>20.340594</td>
          <td>0.005059</td>
          <td>24.616949</td>
          <td>0.023271</td>
          <td>21.480601</td>
          <td>0.005501</td>
          <td>22.075467</td>
          <td>0.008884</td>
          <td>18.094732</td>
          <td>0.005036</td>
          <td>25.522344</td>
          <td>23.767485</td>
          <td>24.033583</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.520539</td>
          <td>0.005151</td>
          <td>23.718539</td>
          <td>0.012649</td>
          <td>22.056170</td>
          <td>0.005521</td>
          <td>22.011902</td>
          <td>0.006175</td>
          <td>20.127278</td>
          <td>0.005188</td>
          <td>22.332508</td>
          <td>0.021758</td>
          <td>26.985481</td>
          <td>24.739030</td>
          <td>25.410448</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.395142</td>
          <td>0.005006</td>
          <td>20.346714</td>
          <td>0.005059</td>
          <td>25.077951</td>
          <td>0.034824</td>
          <td>21.284943</td>
          <td>0.005365</td>
          <td>20.069015</td>
          <td>0.005172</td>
          <td>21.109014</td>
          <td>0.008637</td>
          <td>23.075004</td>
          <td>25.032900</td>
          <td>28.973212</td>
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
          <td>21.600285</td>
          <td>22.211212</td>
          <td>22.366020</td>
          <td>23.912386</td>
          <td>25.819375</td>
          <td>20.489041</td>
          <td>22.560827</td>
          <td>0.006176</td>
          <td>22.304969</td>
          <td>0.007054</td>
          <td>22.792313</td>
          <td>0.009256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.004457</td>
          <td>20.295468</td>
          <td>23.061801</td>
          <td>20.854938</td>
          <td>23.375987</td>
          <td>26.162985</td>
          <td>25.520852</td>
          <td>0.054440</td>
          <td>19.850978</td>
          <td>0.005027</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.021675</td>
          <td>22.659454</td>
          <td>24.387666</td>
          <td>21.311826</td>
          <td>23.122928</td>
          <td>21.557232</td>
          <td>19.713214</td>
          <td>0.005007</td>
          <td>24.877486</td>
          <td>0.052376</td>
          <td>22.198909</td>
          <td>0.006736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.741081</td>
          <td>24.113025</td>
          <td>26.947669</td>
          <td>25.434167</td>
          <td>22.993714</td>
          <td>24.345914</td>
          <td>23.514133</td>
          <td>0.010044</td>
          <td>22.229809</td>
          <td>0.006824</td>
          <td>20.906121</td>
          <td>0.005185</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.905062</td>
          <td>16.423798</td>
          <td>20.817971</td>
          <td>20.074055</td>
          <td>24.464032</td>
          <td>21.620976</td>
          <td>21.624249</td>
          <td>0.005229</td>
          <td>27.142717</td>
          <td>0.362534</td>
          <td>22.911775</td>
          <td>0.010028</td>
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
          <td>28.734384</td>
          <td>21.803859</td>
          <td>28.377700</td>
          <td>20.464912</td>
          <td>24.483494</td>
          <td>24.216046</td>
          <td>24.830149</td>
          <td>0.029452</td>
          <td>19.824997</td>
          <td>0.005026</td>
          <td>21.852009</td>
          <td>0.005980</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.042235</td>
          <td>26.346415</td>
          <td>28.189524</td>
          <td>23.746700</td>
          <td>22.105478</td>
          <td>24.243824</td>
          <td>21.626483</td>
          <td>0.005230</td>
          <td>18.952886</td>
          <td>0.005005</td>
          <td>29.579713</td>
          <td>1.689532</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.380305</td>
          <td>20.338909</td>
          <td>24.608150</td>
          <td>21.483308</td>
          <td>22.082617</td>
          <td>18.090909</td>
          <td>25.542093</td>
          <td>0.055480</td>
          <td>23.797079</td>
          <td>0.020188</td>
          <td>24.027940</td>
          <td>0.024657</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.514965</td>
          <td>23.713411</td>
          <td>22.065376</td>
          <td>22.007360</td>
          <td>20.135617</td>
          <td>22.333707</td>
          <td>26.983902</td>
          <td>0.195336</td>
          <td>24.695092</td>
          <td>0.044516</td>
          <td>25.566014</td>
          <td>0.096442</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.382581</td>
          <td>20.341546</td>
          <td>25.069870</td>
          <td>21.293419</td>
          <td>20.065958</td>
          <td>21.112805</td>
          <td>23.082292</td>
          <td>0.007702</td>
          <td>25.062212</td>
          <td>0.061743</td>
          <td>27.856507</td>
          <td>0.616839</td>
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
          <td>21.600285</td>
          <td>22.211212</td>
          <td>22.366020</td>
          <td>23.912386</td>
          <td>25.819375</td>
          <td>20.489041</td>
          <td>22.583086</td>
          <td>0.044042</td>
          <td>22.253067</td>
          <td>0.027517</td>
          <td>22.822525</td>
          <td>0.049872</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.004457</td>
          <td>20.295468</td>
          <td>23.061801</td>
          <td>20.854938</td>
          <td>23.375987</td>
          <td>26.162985</td>
          <td>26.565719</td>
          <td>1.042363</td>
          <td>19.855597</td>
          <td>0.005832</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.021675</td>
          <td>22.659454</td>
          <td>24.387666</td>
          <td>21.311826</td>
          <td>23.122928</td>
          <td>21.557232</td>
          <td>19.711342</td>
          <td>0.005915</td>
          <td>25.063180</td>
          <td>0.314519</td>
          <td>22.183014</td>
          <td>0.028253</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.741081</td>
          <td>24.113025</td>
          <td>26.947669</td>
          <td>25.434167</td>
          <td>22.993714</td>
          <td>24.345914</td>
          <td>23.550877</td>
          <td>0.103901</td>
          <td>22.193649</td>
          <td>0.026117</td>
          <td>20.895368</td>
          <td>0.009915</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.905062</td>
          <td>16.423798</td>
          <td>20.817971</td>
          <td>20.074055</td>
          <td>24.464032</td>
          <td>21.620976</td>
          <td>21.601740</td>
          <td>0.018611</td>
          <td>26.592371</td>
          <td>0.938693</td>
          <td>22.920506</td>
          <td>0.054423</td>
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
          <td>28.734384</td>
          <td>21.803859</td>
          <td>28.377700</td>
          <td>20.464912</td>
          <td>24.483494</td>
          <td>24.216046</td>
          <td>24.416792</td>
          <td>0.218354</td>
          <td>19.827745</td>
          <td>0.005793</td>
          <td>21.799900</td>
          <td>0.020237</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.042235</td>
          <td>26.346415</td>
          <td>28.189524</td>
          <td>23.746700</td>
          <td>22.105478</td>
          <td>24.243824</td>
          <td>21.644300</td>
          <td>0.019297</td>
          <td>18.948790</td>
          <td>0.005167</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.380305</td>
          <td>20.338909</td>
          <td>24.608150</td>
          <td>21.483308</td>
          <td>22.082617</td>
          <td>18.090909</td>
          <td>24.905486</td>
          <td>0.325313</td>
          <td>23.965654</td>
          <td>0.125320</td>
          <td>24.187305</td>
          <td>0.165334</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.514965</td>
          <td>23.713411</td>
          <td>22.065376</td>
          <td>22.007360</td>
          <td>20.135617</td>
          <td>22.333707</td>
          <td>25.526083</td>
          <td>0.522911</td>
          <td>24.607766</td>
          <td>0.216716</td>
          <td>25.667442</td>
          <td>0.538905</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.382581</td>
          <td>20.341546</td>
          <td>25.069870</td>
          <td>21.293419</td>
          <td>20.065958</td>
          <td>21.112805</td>
          <td>23.284741</td>
          <td>0.082197</td>
          <td>25.118621</td>
          <td>0.328728</td>
          <td>27.198150</td>
          <td>1.400373</td>
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


