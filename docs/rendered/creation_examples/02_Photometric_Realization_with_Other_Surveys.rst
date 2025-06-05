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
          <td>23.451617</td>
          <td>19.722363</td>
          <td>24.313431</td>
          <td>23.282069</td>
          <td>22.291720</td>
          <td>23.408437</td>
          <td>24.689445</td>
          <td>20.423463</td>
          <td>24.473912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.735226</td>
          <td>20.036914</td>
          <td>27.230196</td>
          <td>22.097412</td>
          <td>26.019418</td>
          <td>20.461524</td>
          <td>20.808941</td>
          <td>23.207794</td>
          <td>26.219134</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.685272</td>
          <td>20.749321</td>
          <td>17.270839</td>
          <td>19.247020</td>
          <td>24.720774</td>
          <td>25.313623</td>
          <td>19.366446</td>
          <td>24.028880</td>
          <td>22.250324</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.479002</td>
          <td>21.117964</td>
          <td>24.598079</td>
          <td>24.126539</td>
          <td>23.755744</td>
          <td>23.367258</td>
          <td>28.422847</td>
          <td>20.742615</td>
          <td>19.562294</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.824941</td>
          <td>21.149085</td>
          <td>19.353649</td>
          <td>19.243603</td>
          <td>24.662155</td>
          <td>25.362919</td>
          <td>21.679387</td>
          <td>21.700539</td>
          <td>23.410486</td>
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
          <td>21.154030</td>
          <td>25.519727</td>
          <td>22.978749</td>
          <td>18.266854</td>
          <td>25.471964</td>
          <td>22.103887</td>
          <td>21.694971</td>
          <td>26.997842</td>
          <td>19.522768</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.542303</td>
          <td>17.718911</td>
          <td>23.150489</td>
          <td>21.238544</td>
          <td>18.655748</td>
          <td>22.000960</td>
          <td>22.287470</td>
          <td>20.455288</td>
          <td>24.861999</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.206184</td>
          <td>28.985588</td>
          <td>18.562202</td>
          <td>21.028890</td>
          <td>20.596944</td>
          <td>21.978578</td>
          <td>24.580734</td>
          <td>23.805350</td>
          <td>26.317781</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.856998</td>
          <td>23.300778</td>
          <td>25.087269</td>
          <td>27.481376</td>
          <td>21.772763</td>
          <td>23.267766</td>
          <td>28.832052</td>
          <td>28.643594</td>
          <td>21.518397</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.216733</td>
          <td>22.900849</td>
          <td>23.331819</td>
          <td>19.789278</td>
          <td>24.099205</td>
          <td>22.747076</td>
          <td>22.900673</td>
          <td>30.600504</td>
          <td>24.322542</td>
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
          <td>23.510799</td>
          <td>0.029396</td>
          <td>19.717800</td>
          <td>0.005027</td>
          <td>24.295600</td>
          <td>0.017718</td>
          <td>23.279973</td>
          <td>0.012264</td>
          <td>22.289184</td>
          <td>0.010201</td>
          <td>23.479660</td>
          <td>0.059686</td>
          <td>24.689445</td>
          <td>20.423463</td>
          <td>24.473912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.797054</td>
          <td>0.090708</td>
          <td>20.037336</td>
          <td>0.005040</td>
          <td>27.170024</td>
          <td>0.216383</td>
          <td>22.092149</td>
          <td>0.006333</td>
          <td>26.004616</td>
          <td>0.239255</td>
          <td>20.464982</td>
          <td>0.006395</td>
          <td>20.808941</td>
          <td>23.207794</td>
          <td>26.219134</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.685447</td>
          <td>0.005696</td>
          <td>20.746785</td>
          <td>0.005101</td>
          <td>17.273284</td>
          <td>0.005001</td>
          <td>19.244769</td>
          <td>0.005018</td>
          <td>24.756455</td>
          <td>0.081897</td>
          <td>25.806330</td>
          <td>0.424671</td>
          <td>19.366446</td>
          <td>24.028880</td>
          <td>22.250324</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.480628</td>
          <td>0.005525</td>
          <td>21.114784</td>
          <td>0.005171</td>
          <td>24.591635</td>
          <td>0.022769</td>
          <td>24.084100</td>
          <td>0.023688</td>
          <td>23.748726</td>
          <td>0.033542</td>
          <td>23.362548</td>
          <td>0.053795</td>
          <td>28.422847</td>
          <td>20.742615</td>
          <td>19.562294</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.808363</td>
          <td>0.008290</td>
          <td>21.155711</td>
          <td>0.005182</td>
          <td>19.354381</td>
          <td>0.005010</td>
          <td>19.245591</td>
          <td>0.005018</td>
          <td>24.794724</td>
          <td>0.084707</td>
          <td>25.554842</td>
          <td>0.349496</td>
          <td>21.679387</td>
          <td>21.700539</td>
          <td>23.410486</td>
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
          <td>21.153904</td>
          <td>0.006339</td>
          <td>25.603206</td>
          <td>0.063140</td>
          <td>22.982851</td>
          <td>0.007239</td>
          <td>18.269650</td>
          <td>0.005006</td>
          <td>25.473490</td>
          <td>0.152946</td>
          <td>22.114613</td>
          <td>0.018097</td>
          <td>21.694971</td>
          <td>26.997842</td>
          <td>19.522768</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.547634</td>
          <td>0.030346</td>
          <td>17.724588</td>
          <td>0.005003</td>
          <td>23.153200</td>
          <td>0.007879</td>
          <td>21.234283</td>
          <td>0.005337</td>
          <td>18.651198</td>
          <td>0.005022</td>
          <td>22.006170</td>
          <td>0.016543</td>
          <td>22.287470</td>
          <td>20.455288</td>
          <td>24.861999</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.204638</td>
          <td>0.005104</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.564164</td>
          <td>0.005004</td>
          <td>21.024193</td>
          <td>0.005240</td>
          <td>20.593943</td>
          <td>0.005395</td>
          <td>22.012337</td>
          <td>0.016627</td>
          <td>24.580734</td>
          <td>23.805350</td>
          <td>26.317781</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.758587</td>
          <td>0.459343</td>
          <td>23.306012</td>
          <td>0.009483</td>
          <td>25.140636</td>
          <td>0.036807</td>
          <td>27.132069</td>
          <td>0.329405</td>
          <td>21.775502</td>
          <td>0.007525</td>
          <td>23.366978</td>
          <td>0.054007</td>
          <td>28.832052</td>
          <td>28.643594</td>
          <td>21.518397</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.257482</td>
          <td>0.311431</td>
          <td>22.893508</td>
          <td>0.007518</td>
          <td>23.323716</td>
          <td>0.008675</td>
          <td>19.794331</td>
          <td>0.005037</td>
          <td>24.072039</td>
          <td>0.044654</td>
          <td>22.751131</td>
          <td>0.031308</td>
          <td>22.900673</td>
          <td>30.600504</td>
          <td>24.322542</td>
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
          <td>23.451617</td>
          <td>19.722363</td>
          <td>24.313431</td>
          <td>23.282069</td>
          <td>22.291720</td>
          <td>23.408437</td>
          <td>24.678453</td>
          <td>0.025772</td>
          <td>20.421687</td>
          <td>0.005077</td>
          <td>24.485153</td>
          <td>0.036924</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.735226</td>
          <td>20.036914</td>
          <td>27.230196</td>
          <td>22.097412</td>
          <td>26.019418</td>
          <td>20.461524</td>
          <td>20.812199</td>
          <td>0.005052</td>
          <td>23.212873</td>
          <td>0.012502</td>
          <td>26.128589</td>
          <td>0.157237</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.685272</td>
          <td>20.749321</td>
          <td>17.270839</td>
          <td>19.247020</td>
          <td>24.720774</td>
          <td>25.313623</td>
          <td>19.369392</td>
          <td>0.005004</td>
          <td>24.032834</td>
          <td>0.024763</td>
          <td>22.246671</td>
          <td>0.006874</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.479002</td>
          <td>21.117964</td>
          <td>24.598079</td>
          <td>24.126539</td>
          <td>23.755744</td>
          <td>23.367258</td>
          <td>28.480206</td>
          <td>0.627174</td>
          <td>20.751650</td>
          <td>0.005140</td>
          <td>19.552393</td>
          <td>0.005016</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.824941</td>
          <td>21.149085</td>
          <td>19.353649</td>
          <td>19.243603</td>
          <td>24.662155</td>
          <td>25.362919</td>
          <td>21.681513</td>
          <td>0.005254</td>
          <td>21.693514</td>
          <td>0.005748</td>
          <td>23.388359</td>
          <td>0.014356</td>
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
          <td>21.154030</td>
          <td>25.519727</td>
          <td>22.978749</td>
          <td>18.266854</td>
          <td>25.471964</td>
          <td>22.103887</td>
          <td>21.695620</td>
          <td>0.005260</td>
          <td>27.222008</td>
          <td>0.385634</td>
          <td>19.524508</td>
          <td>0.005015</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.542303</td>
          <td>17.718911</td>
          <td>23.150489</td>
          <td>21.238544</td>
          <td>18.655748</td>
          <td>22.000960</td>
          <td>22.274084</td>
          <td>0.005723</td>
          <td>20.457905</td>
          <td>0.005082</td>
          <td>24.929103</td>
          <td>0.054842</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.206184</td>
          <td>28.985588</td>
          <td>18.562202</td>
          <td>21.028890</td>
          <td>20.596944</td>
          <td>21.978578</td>
          <td>24.563380</td>
          <td>0.023307</td>
          <td>23.802652</td>
          <td>0.020285</td>
          <td>26.288792</td>
          <td>0.180246</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.856998</td>
          <td>23.300778</td>
          <td>25.087269</td>
          <td>27.481376</td>
          <td>21.772763</td>
          <td>23.267766</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.516685</td>
          <td>0.005550</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.216733</td>
          <td>22.900849</td>
          <td>23.331819</td>
          <td>19.789278</td>
          <td>24.099205</td>
          <td>22.747076</td>
          <td>22.906757</td>
          <td>0.007060</td>
          <td>28.565871</td>
          <td>0.981836</td>
          <td>24.336449</td>
          <td>0.032357</td>
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
          <td>23.451617</td>
          <td>19.722363</td>
          <td>24.313431</td>
          <td>23.282069</td>
          <td>22.291720</td>
          <td>23.408437</td>
          <td>24.704398</td>
          <td>0.276725</td>
          <td>20.434805</td>
          <td>0.007153</td>
          <td>24.469456</td>
          <td>0.209887</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.735226</td>
          <td>20.036914</td>
          <td>27.230196</td>
          <td>22.097412</td>
          <td>26.019418</td>
          <td>20.461524</td>
          <td>20.808031</td>
          <td>0.010002</td>
          <td>23.192147</td>
          <td>0.063409</td>
          <td>25.869714</td>
          <td>0.622583</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.685272</td>
          <td>20.749321</td>
          <td>17.270839</td>
          <td>19.247020</td>
          <td>24.720774</td>
          <td>25.313623</td>
          <td>19.352715</td>
          <td>0.005492</td>
          <td>24.293275</td>
          <td>0.166178</td>
          <td>22.274142</td>
          <td>0.030620</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.479002</td>
          <td>21.117964</td>
          <td>24.598079</td>
          <td>24.126539</td>
          <td>23.755744</td>
          <td>23.367258</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.726747</td>
          <td>0.008352</td>
          <td>19.551970</td>
          <td>0.005585</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.824941</td>
          <td>21.149085</td>
          <td>19.353649</td>
          <td>19.243603</td>
          <td>24.662155</td>
          <td>25.362919</td>
          <td>21.678904</td>
          <td>0.019876</td>
          <td>21.705023</td>
          <td>0.017155</td>
          <td>23.275966</td>
          <td>0.074652</td>
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
          <td>21.154030</td>
          <td>25.519727</td>
          <td>22.978749</td>
          <td>18.266854</td>
          <td>25.471964</td>
          <td>22.103887</td>
          <td>21.719111</td>
          <td>0.020573</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.525893</td>
          <td>0.005559</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.542303</td>
          <td>17.718911</td>
          <td>23.150489</td>
          <td>21.238544</td>
          <td>18.655748</td>
          <td>22.000960</td>
          <td>22.314857</td>
          <td>0.034688</td>
          <td>20.445798</td>
          <td>0.007190</td>
          <td>24.769257</td>
          <td>0.268919</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.206184</td>
          <td>28.985588</td>
          <td>18.562202</td>
          <td>21.028890</td>
          <td>20.596944</td>
          <td>21.978578</td>
          <td>24.226433</td>
          <td>0.186085</td>
          <td>23.809305</td>
          <td>0.109354</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.856998</td>
          <td>23.300778</td>
          <td>25.087269</td>
          <td>27.481376</td>
          <td>21.772763</td>
          <td>23.267766</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.239308</td>
          <td>0.748530</td>
          <td>21.520304</td>
          <td>0.015988</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.216733</td>
          <td>22.900849</td>
          <td>23.331819</td>
          <td>19.789278</td>
          <td>24.099205</td>
          <td>22.747076</td>
          <td>22.938507</td>
          <td>0.060454</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.693368</td>
          <td>0.252723</td>
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


