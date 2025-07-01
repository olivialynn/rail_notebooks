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
          <td>20.723806</td>
          <td>19.689536</td>
          <td>18.196654</td>
          <td>19.031511</td>
          <td>27.813979</td>
          <td>19.580286</td>
          <td>22.884979</td>
          <td>22.820132</td>
          <td>22.633599</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.381439</td>
          <td>19.816516</td>
          <td>25.620505</td>
          <td>23.663116</td>
          <td>17.124172</td>
          <td>18.853032</td>
          <td>28.791766</td>
          <td>22.276992</td>
          <td>19.891014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.112769</td>
          <td>14.414841</td>
          <td>20.071296</td>
          <td>21.730730</td>
          <td>27.860440</td>
          <td>24.213735</td>
          <td>25.023527</td>
          <td>22.997774</td>
          <td>20.143152</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.978402</td>
          <td>25.192154</td>
          <td>21.650151</td>
          <td>19.363676</td>
          <td>27.757959</td>
          <td>24.735369</td>
          <td>23.180177</td>
          <td>23.862083</td>
          <td>21.453049</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.940529</td>
          <td>22.594908</td>
          <td>23.012994</td>
          <td>24.069308</td>
          <td>22.584905</td>
          <td>26.133791</td>
          <td>15.814690</td>
          <td>23.610985</td>
          <td>23.765968</td>
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
          <td>24.730889</td>
          <td>20.472660</td>
          <td>21.130936</td>
          <td>27.668702</td>
          <td>27.358423</td>
          <td>26.755434</td>
          <td>21.027403</td>
          <td>21.229973</td>
          <td>21.622442</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.207209</td>
          <td>27.069843</td>
          <td>19.230317</td>
          <td>19.009907</td>
          <td>24.764885</td>
          <td>22.303639</td>
          <td>22.193229</td>
          <td>20.216719</td>
          <td>22.184400</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.065462</td>
          <td>24.563469</td>
          <td>22.927805</td>
          <td>26.737417</td>
          <td>24.485355</td>
          <td>32.209668</td>
          <td>30.300191</td>
          <td>24.670921</td>
          <td>24.066085</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.998546</td>
          <td>21.590378</td>
          <td>22.558216</td>
          <td>23.245185</td>
          <td>22.180537</td>
          <td>22.531680</td>
          <td>27.758445</td>
          <td>20.342119</td>
          <td>24.590568</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.014481</td>
          <td>24.580167</td>
          <td>22.786230</td>
          <td>20.209357</td>
          <td>24.346184</td>
          <td>18.317677</td>
          <td>22.150009</td>
          <td>20.127893</td>
          <td>22.596184</td>
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
          <td>20.726963</td>
          <td>0.005737</td>
          <td>19.694763</td>
          <td>0.005026</td>
          <td>18.195843</td>
          <td>0.005003</td>
          <td>19.038022</td>
          <td>0.005014</td>
          <td>28.672377</td>
          <td>1.470570</td>
          <td>19.577304</td>
          <td>0.005339</td>
          <td>22.884979</td>
          <td>22.820132</td>
          <td>22.633599</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.385134</td>
          <td>0.005129</td>
          <td>19.818354</td>
          <td>0.005030</td>
          <td>25.637889</td>
          <td>0.057215</td>
          <td>23.664999</td>
          <td>0.016621</td>
          <td>17.128304</td>
          <td>0.005004</td>
          <td>18.850991</td>
          <td>0.005109</td>
          <td>28.791766</td>
          <td>22.276992</td>
          <td>19.891014</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.171274</td>
          <td>0.052344</td>
          <td>14.421151</td>
          <td>0.005000</td>
          <td>20.070618</td>
          <td>0.005026</td>
          <td>21.729742</td>
          <td>0.005749</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.228326</td>
          <td>0.115399</td>
          <td>25.023527</td>
          <td>22.997774</td>
          <td>20.143152</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.977728</td>
          <td>0.005010</td>
          <td>25.220083</td>
          <td>0.044974</td>
          <td>21.649532</td>
          <td>0.005271</td>
          <td>19.354847</td>
          <td>0.005021</td>
          <td>27.081078</td>
          <td>0.552884</td>
          <td>24.906180</td>
          <td>0.206126</td>
          <td>23.180177</td>
          <td>23.862083</td>
          <td>21.453049</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.918228</td>
          <td>0.017868</td>
          <td>22.590619</td>
          <td>0.006611</td>
          <td>23.017038</td>
          <td>0.007356</td>
          <td>24.092599</td>
          <td>0.023862</td>
          <td>22.610182</td>
          <td>0.012876</td>
          <td>25.907780</td>
          <td>0.458544</td>
          <td>15.814690</td>
          <td>23.610985</td>
          <td>23.765968</td>
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
          <td>24.776888</td>
          <td>0.089122</td>
          <td>20.463536</td>
          <td>0.005069</td>
          <td>21.140521</td>
          <td>0.005121</td>
          <td>27.678837</td>
          <td>0.500987</td>
          <td>27.153215</td>
          <td>0.582236</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.027403</td>
          <td>21.229973</td>
          <td>21.622442</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.205496</td>
          <td>0.005362</td>
          <td>27.051847</td>
          <td>0.221121</td>
          <td>19.221706</td>
          <td>0.005009</td>
          <td>19.010304</td>
          <td>0.005013</td>
          <td>24.912788</td>
          <td>0.093976</td>
          <td>22.290930</td>
          <td>0.020999</td>
          <td>22.193229</td>
          <td>20.216719</td>
          <td>22.184400</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.858710</td>
          <td>0.095730</td>
          <td>24.545080</td>
          <td>0.024882</td>
          <td>22.924529</td>
          <td>0.007051</td>
          <td>27.468940</td>
          <td>0.428077</td>
          <td>24.448788</td>
          <td>0.062384</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.300191</td>
          <td>24.670921</td>
          <td>24.066085</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.987939</td>
          <td>0.005272</td>
          <td>21.593748</td>
          <td>0.005350</td>
          <td>22.560317</td>
          <td>0.006168</td>
          <td>23.247765</td>
          <td>0.011972</td>
          <td>22.182796</td>
          <td>0.009505</td>
          <td>22.525923</td>
          <td>0.025710</td>
          <td>27.758445</td>
          <td>20.342119</td>
          <td>24.590568</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.007000</td>
          <td>0.019209</td>
          <td>24.572883</td>
          <td>0.025487</td>
          <td>22.781601</td>
          <td>0.006649</td>
          <td>20.206484</td>
          <td>0.005068</td>
          <td>24.292852</td>
          <td>0.054324</td>
          <td>18.313324</td>
          <td>0.005049</td>
          <td>22.150009</td>
          <td>20.127893</td>
          <td>22.596184</td>
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
          <td>20.723806</td>
          <td>19.689536</td>
          <td>18.196654</td>
          <td>19.031511</td>
          <td>27.813979</td>
          <td>19.580286</td>
          <td>22.880617</td>
          <td>0.006977</td>
          <td>22.824120</td>
          <td>0.009451</td>
          <td>22.626092</td>
          <td>0.008349</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.381439</td>
          <td>19.816516</td>
          <td>25.620505</td>
          <td>23.663116</td>
          <td>17.124172</td>
          <td>18.853032</td>
          <td>28.948476</td>
          <td>0.857752</td>
          <td>22.264985</td>
          <td>0.006929</td>
          <td>19.890772</td>
          <td>0.005029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.112769</td>
          <td>14.414841</td>
          <td>20.071296</td>
          <td>21.730730</td>
          <td>27.860440</td>
          <td>24.213735</td>
          <td>24.981432</td>
          <td>0.033674</td>
          <td>22.989374</td>
          <td>0.010589</td>
          <td>20.137989</td>
          <td>0.005046</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.978402</td>
          <td>25.192154</td>
          <td>21.650151</td>
          <td>19.363676</td>
          <td>27.757959</td>
          <td>24.735369</td>
          <td>23.188864</td>
          <td>0.008170</td>
          <td>23.903476</td>
          <td>0.022126</td>
          <td>21.453438</td>
          <td>0.005492</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.940529</td>
          <td>22.594908</td>
          <td>23.012994</td>
          <td>24.069308</td>
          <td>22.584905</td>
          <td>26.133791</td>
          <td>15.819955</td>
          <td>0.005000</td>
          <td>23.615984</td>
          <td>0.017313</td>
          <td>23.796194</td>
          <td>0.020173</td>
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
          <td>24.730889</td>
          <td>20.472660</td>
          <td>21.130936</td>
          <td>27.668702</td>
          <td>27.358423</td>
          <td>26.755434</td>
          <td>21.019832</td>
          <td>0.005076</td>
          <td>21.225228</td>
          <td>0.005329</td>
          <td>21.624309</td>
          <td>0.005664</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.207209</td>
          <td>27.069843</td>
          <td>19.230317</td>
          <td>19.009907</td>
          <td>24.764885</td>
          <td>22.303639</td>
          <td>22.191330</td>
          <td>0.005627</td>
          <td>20.218331</td>
          <td>0.005053</td>
          <td>22.192252</td>
          <td>0.006718</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.065462</td>
          <td>24.563469</td>
          <td>22.927805</td>
          <td>26.737417</td>
          <td>24.485355</td>
          <td>32.209668</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.683535</td>
          <td>0.044060</td>
          <td>24.083971</td>
          <td>0.025897</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.998546</td>
          <td>21.590378</td>
          <td>22.558216</td>
          <td>23.245185</td>
          <td>22.180537</td>
          <td>22.531680</td>
          <td>27.660103</td>
          <td>0.339716</td>
          <td>20.350978</td>
          <td>0.005067</td>
          <td>24.553836</td>
          <td>0.039251</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.014481</td>
          <td>24.580167</td>
          <td>22.786230</td>
          <td>20.209357</td>
          <td>24.346184</td>
          <td>18.317677</td>
          <td>22.154860</td>
          <td>0.005588</td>
          <td>20.120773</td>
          <td>0.005044</td>
          <td>22.601108</td>
          <td>0.008228</td>
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
          <td>20.723806</td>
          <td>19.689536</td>
          <td>18.196654</td>
          <td>19.031511</td>
          <td>27.813979</td>
          <td>19.580286</td>
          <td>22.933096</td>
          <td>0.060164</td>
          <td>22.823273</td>
          <td>0.045648</td>
          <td>22.598106</td>
          <td>0.040830</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.381439</td>
          <td>19.816516</td>
          <td>25.620505</td>
          <td>23.663116</td>
          <td>17.124172</td>
          <td>18.853032</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.253017</td>
          <td>0.027516</td>
          <td>19.881028</td>
          <td>0.006029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.112769</td>
          <td>14.414841</td>
          <td>20.071296</td>
          <td>21.730730</td>
          <td>27.860440</td>
          <td>24.213735</td>
          <td>24.755166</td>
          <td>0.288353</td>
          <td>23.013860</td>
          <td>0.054102</td>
          <td>20.133950</td>
          <td>0.006563</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.978402</td>
          <td>25.192154</td>
          <td>21.650151</td>
          <td>19.363676</td>
          <td>27.757959</td>
          <td>24.735369</td>
          <td>23.149780</td>
          <td>0.072939</td>
          <td>23.815760</td>
          <td>0.109973</td>
          <td>21.435821</td>
          <td>0.014918</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.940529</td>
          <td>22.594908</td>
          <td>23.012994</td>
          <td>24.069308</td>
          <td>22.584905</td>
          <td>26.133791</td>
          <td>15.809389</td>
          <td>0.005001</td>
          <td>23.468706</td>
          <td>0.081040</td>
          <td>23.768339</td>
          <td>0.115141</td>
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
          <td>24.730889</td>
          <td>20.472660</td>
          <td>21.130936</td>
          <td>27.668702</td>
          <td>27.358423</td>
          <td>26.755434</td>
          <td>21.008469</td>
          <td>0.011550</td>
          <td>21.233878</td>
          <td>0.011772</td>
          <td>21.644336</td>
          <td>0.017730</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.207209</td>
          <td>27.069843</td>
          <td>19.230317</td>
          <td>19.009907</td>
          <td>24.764885</td>
          <td>22.303639</td>
          <td>22.208302</td>
          <td>0.031560</td>
          <td>20.222463</td>
          <td>0.006535</td>
          <td>22.182695</td>
          <td>0.028245</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.065462</td>
          <td>24.563469</td>
          <td>22.927805</td>
          <td>26.737417</td>
          <td>24.485355</td>
          <td>32.209668</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.504558</td>
          <td>0.198762</td>
          <td>24.285607</td>
          <td>0.179760</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.998546</td>
          <td>21.590378</td>
          <td>22.558216</td>
          <td>23.245185</td>
          <td>22.180537</td>
          <td>22.531680</td>
          <td>25.472867</td>
          <td>0.502879</td>
          <td>20.340036</td>
          <td>0.006854</td>
          <td>24.428548</td>
          <td>0.202810</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.014481</td>
          <td>24.580167</td>
          <td>22.786230</td>
          <td>20.209357</td>
          <td>24.346184</td>
          <td>18.317677</td>
          <td>22.167490</td>
          <td>0.030440</td>
          <td>20.141203</td>
          <td>0.006344</td>
          <td>22.596238</td>
          <td>0.040762</td>
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


