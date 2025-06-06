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
          <td>30.031258</td>
          <td>17.191882</td>
          <td>24.889885</td>
          <td>24.220883</td>
          <td>24.016296</td>
          <td>22.016165</td>
          <td>22.126700</td>
          <td>22.553807</td>
          <td>21.258569</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.256566</td>
          <td>22.093002</td>
          <td>23.231394</td>
          <td>24.295086</td>
          <td>20.224759</td>
          <td>22.683413</td>
          <td>24.666483</td>
          <td>26.108378</td>
          <td>17.636451</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.776272</td>
          <td>26.366157</td>
          <td>28.593254</td>
          <td>20.230417</td>
          <td>25.729343</td>
          <td>22.062712</td>
          <td>26.757006</td>
          <td>22.001221</td>
          <td>18.491529</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.459887</td>
          <td>23.979001</td>
          <td>22.140016</td>
          <td>19.736142</td>
          <td>23.617306</td>
          <td>21.145141</td>
          <td>22.661724</td>
          <td>26.341636</td>
          <td>23.919586</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.588676</td>
          <td>27.975101</td>
          <td>16.695041</td>
          <td>18.098525</td>
          <td>24.452312</td>
          <td>24.737150</td>
          <td>20.741170</td>
          <td>22.867001</td>
          <td>20.983780</td>
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
          <td>20.412578</td>
          <td>25.377557</td>
          <td>20.844437</td>
          <td>21.918571</td>
          <td>24.183313</td>
          <td>26.121024</td>
          <td>19.377147</td>
          <td>13.416602</td>
          <td>21.466465</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.241010</td>
          <td>24.198261</td>
          <td>20.754567</td>
          <td>25.112757</td>
          <td>26.257187</td>
          <td>23.273130</td>
          <td>24.476742</td>
          <td>27.986746</td>
          <td>25.619116</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.891903</td>
          <td>21.907343</td>
          <td>21.447504</td>
          <td>20.030690</td>
          <td>25.571044</td>
          <td>19.974696</td>
          <td>25.696455</td>
          <td>23.136304</td>
          <td>21.673969</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.281391</td>
          <td>23.414690</td>
          <td>19.333246</td>
          <td>23.850651</td>
          <td>28.559619</td>
          <td>27.462186</td>
          <td>22.909602</td>
          <td>26.765439</td>
          <td>23.977622</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.284950</td>
          <td>24.388150</td>
          <td>18.412388</td>
          <td>28.990883</td>
          <td>26.713242</td>
          <td>26.501261</td>
          <td>25.972356</td>
          <td>26.980472</td>
          <td>20.475108</td>
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
          <td>29.521843</td>
          <td>2.216740</td>
          <td>17.186905</td>
          <td>0.005002</td>
          <td>24.878291</td>
          <td>0.029212</td>
          <td>24.202702</td>
          <td>0.026255</td>
          <td>24.016382</td>
          <td>0.042503</td>
          <td>21.999579</td>
          <td>0.016454</td>
          <td>22.126700</td>
          <td>22.553807</td>
          <td>21.258569</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.256401</td>
          <td>0.005388</td>
          <td>22.094991</td>
          <td>0.005756</td>
          <td>23.241657</td>
          <td>0.008271</td>
          <td>24.279529</td>
          <td>0.028076</td>
          <td>20.225764</td>
          <td>0.005220</td>
          <td>22.658336</td>
          <td>0.028859</td>
          <td>24.666483</td>
          <td>26.108378</td>
          <td>17.636451</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.130019</td>
          <td>0.602112</td>
          <td>26.110629</td>
          <td>0.098731</td>
          <td>29.871609</td>
          <td>1.406546</td>
          <td>20.227122</td>
          <td>0.005070</td>
          <td>25.806683</td>
          <td>0.202911</td>
          <td>22.069338</td>
          <td>0.017428</td>
          <td>26.757006</td>
          <td>22.001221</td>
          <td>18.491529</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.526685</td>
          <td>0.384911</td>
          <td>23.963892</td>
          <td>0.015306</td>
          <td>22.137990</td>
          <td>0.005594</td>
          <td>19.736790</td>
          <td>0.005035</td>
          <td>23.564475</td>
          <td>0.028528</td>
          <td>21.150995</td>
          <td>0.008858</td>
          <td>22.661724</td>
          <td>26.341636</td>
          <td>23.919586</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.587736</td>
          <td>0.013787</td>
          <td>27.472127</td>
          <td>0.311645</td>
          <td>16.691827</td>
          <td>0.005001</td>
          <td>18.099328</td>
          <td>0.005005</td>
          <td>24.480716</td>
          <td>0.064175</td>
          <td>24.767934</td>
          <td>0.183478</td>
          <td>20.741170</td>
          <td>22.867001</td>
          <td>20.983780</td>
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
          <td>20.413576</td>
          <td>0.005479</td>
          <td>25.344824</td>
          <td>0.050227</td>
          <td>20.839077</td>
          <td>0.005077</td>
          <td>21.914560</td>
          <td>0.006007</td>
          <td>24.121908</td>
          <td>0.046675</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.377147</td>
          <td>13.416602</td>
          <td>21.466465</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.250483</td>
          <td>0.010828</td>
          <td>24.207501</td>
          <td>0.018679</td>
          <td>20.751087</td>
          <td>0.005067</td>
          <td>25.080567</td>
          <td>0.057023</td>
          <td>26.447865</td>
          <td>0.342346</td>
          <td>23.284358</td>
          <td>0.050187</td>
          <td>24.476742</td>
          <td>27.986746</td>
          <td>25.619116</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.812211</td>
          <td>0.216513</td>
          <td>21.918206</td>
          <td>0.005576</td>
          <td>21.447332</td>
          <td>0.005196</td>
          <td>20.034284</td>
          <td>0.005053</td>
          <td>25.481488</td>
          <td>0.153998</td>
          <td>19.972557</td>
          <td>0.005639</td>
          <td>25.696455</td>
          <td>23.136304</td>
          <td>21.673969</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.277316</td>
          <td>0.005005</td>
          <td>23.422136</td>
          <td>0.010233</td>
          <td>19.334090</td>
          <td>0.005010</td>
          <td>23.845487</td>
          <td>0.019319</td>
          <td>26.437875</td>
          <td>0.339655</td>
          <td>26.747870</td>
          <td>0.825525</td>
          <td>22.909602</td>
          <td>26.765439</td>
          <td>23.977622</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.283946</td>
          <td>0.006605</td>
          <td>24.395142</td>
          <td>0.021880</td>
          <td>18.405314</td>
          <td>0.005004</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.816099</td>
          <td>0.454793</td>
          <td>27.610713</td>
          <td>1.366741</td>
          <td>25.972356</td>
          <td>26.980472</td>
          <td>20.475108</td>
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
          <td>30.031258</td>
          <td>17.191882</td>
          <td>24.889885</td>
          <td>24.220883</td>
          <td>24.016296</td>
          <td>22.016165</td>
          <td>22.143518</td>
          <td>0.005577</td>
          <td>22.543650</td>
          <td>0.007963</td>
          <td>21.257883</td>
          <td>0.005348</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.256566</td>
          <td>22.093002</td>
          <td>23.231394</td>
          <td>24.295086</td>
          <td>20.224759</td>
          <td>22.683413</td>
          <td>24.630346</td>
          <td>0.024709</td>
          <td>26.435980</td>
          <td>0.204080</td>
          <td>17.635783</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.776272</td>
          <td>26.366157</td>
          <td>28.593254</td>
          <td>20.230417</td>
          <td>25.729343</td>
          <td>22.062712</td>
          <td>26.768085</td>
          <td>0.162641</td>
          <td>22.000966</td>
          <td>0.006257</td>
          <td>18.498104</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.459887</td>
          <td>23.979001</td>
          <td>22.140016</td>
          <td>19.736142</td>
          <td>23.617306</td>
          <td>21.145141</td>
          <td>22.671883</td>
          <td>0.006413</td>
          <td>26.500338</td>
          <td>0.215377</td>
          <td>23.884868</td>
          <td>0.021773</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.588676</td>
          <td>27.975101</td>
          <td>16.695041</td>
          <td>18.098525</td>
          <td>24.452312</td>
          <td>24.737150</td>
          <td>20.743303</td>
          <td>0.005046</td>
          <td>22.869108</td>
          <td>0.009740</td>
          <td>20.988774</td>
          <td>0.005215</td>
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
          <td>20.412578</td>
          <td>25.377557</td>
          <td>20.844437</td>
          <td>21.918571</td>
          <td>24.183313</td>
          <td>26.121024</td>
          <td>19.379732</td>
          <td>0.005004</td>
          <td>13.410801</td>
          <td>0.005000</td>
          <td>21.461440</td>
          <td>0.005499</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.241010</td>
          <td>24.198261</td>
          <td>20.754567</td>
          <td>25.112757</td>
          <td>26.257187</td>
          <td>23.273130</td>
          <td>24.487474</td>
          <td>0.021822</td>
          <td>27.025209</td>
          <td>0.330453</td>
          <td>25.725011</td>
          <td>0.110866</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.891903</td>
          <td>21.907343</td>
          <td>21.447504</td>
          <td>20.030690</td>
          <td>25.571044</td>
          <td>19.974696</td>
          <td>25.707800</td>
          <td>0.064298</td>
          <td>23.152313</td>
          <td>0.011937</td>
          <td>21.671550</td>
          <td>0.005720</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.281391</td>
          <td>23.414690</td>
          <td>19.333246</td>
          <td>23.850651</td>
          <td>28.559619</td>
          <td>27.462186</td>
          <td>22.904421</td>
          <td>0.007053</td>
          <td>26.439040</td>
          <td>0.204604</td>
          <td>23.971655</td>
          <td>0.023476</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.284950</td>
          <td>24.388150</td>
          <td>18.412388</td>
          <td>28.990883</td>
          <td>26.713242</td>
          <td>26.501261</td>
          <td>25.974428</td>
          <td>0.081451</td>
          <td>27.118576</td>
          <td>0.355736</td>
          <td>20.481972</td>
          <td>0.005086</td>
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
          <td>30.031258</td>
          <td>17.191882</td>
          <td>24.889885</td>
          <td>24.220883</td>
          <td>24.016296</td>
          <td>22.016165</td>
          <td>22.106064</td>
          <td>0.028833</td>
          <td>22.538241</td>
          <td>0.035416</td>
          <td>21.260054</td>
          <td>0.012967</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.256566</td>
          <td>22.093002</td>
          <td>23.231394</td>
          <td>24.295086</td>
          <td>20.224759</td>
          <td>22.683413</td>
          <td>25.022669</td>
          <td>0.356881</td>
          <td>28.343557</td>
          <td>2.243202</td>
          <td>17.639423</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.776272</td>
          <td>26.366157</td>
          <td>28.593254</td>
          <td>20.230417</td>
          <td>25.729343</td>
          <td>22.062712</td>
          <td>26.061164</td>
          <td>0.759472</td>
          <td>22.008513</td>
          <td>0.022223</td>
          <td>18.493335</td>
          <td>0.005087</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.459887</td>
          <td>23.979001</td>
          <td>22.140016</td>
          <td>19.736142</td>
          <td>23.617306</td>
          <td>21.145141</td>
          <td>22.650731</td>
          <td>0.046780</td>
          <td>25.295316</td>
          <td>0.377724</td>
          <td>23.879319</td>
          <td>0.126816</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.588676</td>
          <td>27.975101</td>
          <td>16.695041</td>
          <td>18.098525</td>
          <td>24.452312</td>
          <td>24.737150</td>
          <td>20.738189</td>
          <td>0.009539</td>
          <td>22.918185</td>
          <td>0.049679</td>
          <td>20.977234</td>
          <td>0.010498</td>
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
          <td>20.412578</td>
          <td>25.377557</td>
          <td>20.844437</td>
          <td>21.918571</td>
          <td>24.183313</td>
          <td>26.121024</td>
          <td>19.375908</td>
          <td>0.005512</td>
          <td>13.414569</td>
          <td>0.005000</td>
          <td>21.448098</td>
          <td>0.015068</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.241010</td>
          <td>24.198261</td>
          <td>20.754567</td>
          <td>25.112757</td>
          <td>26.257187</td>
          <td>23.273130</td>
          <td>24.573999</td>
          <td>0.248730</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.508037</td>
          <td>1.633376</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.891903</td>
          <td>21.907343</td>
          <td>21.447504</td>
          <td>20.030690</td>
          <td>25.571044</td>
          <td>19.974696</td>
          <td>25.399516</td>
          <td>0.476263</td>
          <td>23.143226</td>
          <td>0.060709</td>
          <td>21.701412</td>
          <td>0.018606</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.281391</td>
          <td>23.414690</td>
          <td>19.333246</td>
          <td>23.850651</td>
          <td>28.559619</td>
          <td>27.462186</td>
          <td>22.844105</td>
          <td>0.055580</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.162005</td>
          <td>0.161798</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.284950</td>
          <td>24.388150</td>
          <td>18.412388</td>
          <td>28.990883</td>
          <td>26.713242</td>
          <td>26.501261</td>
          <td>25.400275</td>
          <td>0.476533</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.469266</td>
          <td>0.007649</td>
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


