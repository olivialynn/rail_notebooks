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
          <td>26.846463</td>
          <td>23.379415</td>
          <td>20.476706</td>
          <td>22.086830</td>
          <td>22.964259</td>
          <td>29.294367</td>
          <td>22.968611</td>
          <td>25.416813</td>
          <td>23.383791</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.785605</td>
          <td>24.978579</td>
          <td>25.404618</td>
          <td>22.351844</td>
          <td>24.121736</td>
          <td>24.094356</td>
          <td>23.915138</td>
          <td>23.530413</td>
          <td>23.309947</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.949424</td>
          <td>22.567964</td>
          <td>18.182410</td>
          <td>29.098719</td>
          <td>20.794782</td>
          <td>28.872692</td>
          <td>25.333942</td>
          <td>19.347991</td>
          <td>26.084123</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.229866</td>
          <td>20.063558</td>
          <td>21.983917</td>
          <td>23.333465</td>
          <td>21.321330</td>
          <td>20.715530</td>
          <td>24.015006</td>
          <td>24.347518</td>
          <td>28.194529</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.519889</td>
          <td>19.882792</td>
          <td>28.915809</td>
          <td>24.702912</td>
          <td>20.494598</td>
          <td>24.346208</td>
          <td>21.538750</td>
          <td>23.902017</td>
          <td>18.767941</td>
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
          <td>24.983497</td>
          <td>26.075854</td>
          <td>26.556739</td>
          <td>26.900710</td>
          <td>25.946519</td>
          <td>21.339982</td>
          <td>22.818980</td>
          <td>16.388861</td>
          <td>20.899265</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.597567</td>
          <td>25.856704</td>
          <td>25.521450</td>
          <td>26.920610</td>
          <td>27.000038</td>
          <td>22.433892</td>
          <td>23.614514</td>
          <td>23.192148</td>
          <td>27.026402</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.903110</td>
          <td>17.980748</td>
          <td>20.256216</td>
          <td>21.548300</td>
          <td>20.089170</td>
          <td>19.268244</td>
          <td>21.759512</td>
          <td>25.559710</td>
          <td>19.805645</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.415246</td>
          <td>28.469845</td>
          <td>26.455921</td>
          <td>27.484980</td>
          <td>23.712129</td>
          <td>22.432459</td>
          <td>17.157416</td>
          <td>26.816269</td>
          <td>19.796217</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.782519</td>
          <td>23.750562</td>
          <td>23.167039</td>
          <td>25.646895</td>
          <td>20.293372</td>
          <td>22.865980</td>
          <td>22.069162</td>
          <td>24.445880</td>
          <td>15.646491</td>
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
          <td>26.696370</td>
          <td>0.438309</td>
          <td>23.379408</td>
          <td>0.009946</td>
          <td>20.474113</td>
          <td>0.005045</td>
          <td>22.088919</td>
          <td>0.006327</td>
          <td>22.986156</td>
          <td>0.017391</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.968611</td>
          <td>25.416813</td>
          <td>23.383791</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.719243</td>
          <td>0.035220</td>
          <td>25.023742</td>
          <td>0.037810</td>
          <td>25.411689</td>
          <td>0.046804</td>
          <td>22.341066</td>
          <td>0.006958</td>
          <td>24.140812</td>
          <td>0.047465</td>
          <td>23.959577</td>
          <td>0.091215</td>
          <td>23.915138</td>
          <td>23.530413</td>
          <td>23.309947</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.920554</td>
          <td>0.041996</td>
          <td>22.566045</td>
          <td>0.006553</td>
          <td>18.183436</td>
          <td>0.005003</td>
          <td>27.965156</td>
          <td>0.615752</td>
          <td>20.803243</td>
          <td>0.005553</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.333942</td>
          <td>19.347991</td>
          <td>26.084123</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.382274</td>
          <td>0.150590</td>
          <td>20.062185</td>
          <td>0.005041</td>
          <td>21.988095</td>
          <td>0.005467</td>
          <td>23.329468</td>
          <td>0.012732</td>
          <td>21.319825</td>
          <td>0.006260</td>
          <td>20.702159</td>
          <td>0.007007</td>
          <td>24.015006</td>
          <td>24.347518</td>
          <td>28.194529</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.579726</td>
          <td>0.074973</td>
          <td>19.876896</td>
          <td>0.005033</td>
          <td>28.093888</td>
          <td>0.451833</td>
          <td>24.691613</td>
          <td>0.040375</td>
          <td>20.502096</td>
          <td>0.005341</td>
          <td>24.437678</td>
          <td>0.138360</td>
          <td>21.538750</td>
          <td>23.902017</td>
          <td>18.767941</td>
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
          <td>25.417935</td>
          <td>0.155250</td>
          <td>25.985989</td>
          <td>0.088506</td>
          <td>26.682324</td>
          <td>0.143073</td>
          <td>27.405010</td>
          <td>0.407667</td>
          <td>26.005407</td>
          <td>0.239412</td>
          <td>21.349101</td>
          <td>0.010060</td>
          <td>22.818980</td>
          <td>16.388861</td>
          <td>20.899265</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.598214</td>
          <td>0.005167</td>
          <td>25.927750</td>
          <td>0.084088</td>
          <td>25.523979</td>
          <td>0.051712</td>
          <td>26.806882</td>
          <td>0.253316</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.419470</td>
          <td>0.023447</td>
          <td>23.614514</td>
          <td>23.192148</td>
          <td>27.026402</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.896217</td>
          <td>0.005933</td>
          <td>17.976587</td>
          <td>0.005004</td>
          <td>20.256457</td>
          <td>0.005033</td>
          <td>21.552839</td>
          <td>0.005563</td>
          <td>20.087121</td>
          <td>0.005177</td>
          <td>19.269492</td>
          <td>0.005208</td>
          <td>21.759512</td>
          <td>25.559710</td>
          <td>19.805645</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.104047</td>
          <td>0.591142</td>
          <td>27.767733</td>
          <td>0.393188</td>
          <td>26.373586</td>
          <td>0.109467</td>
          <td>27.321394</td>
          <td>0.382191</td>
          <td>23.719753</td>
          <td>0.032696</td>
          <td>22.418613</td>
          <td>0.023429</td>
          <td>17.157416</td>
          <td>26.816269</td>
          <td>19.796217</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.795206</td>
          <td>0.037633</td>
          <td>23.742936</td>
          <td>0.012884</td>
          <td>23.162459</td>
          <td>0.007918</td>
          <td>25.788345</td>
          <td>0.106480</td>
          <td>20.294539</td>
          <td>0.005245</td>
          <td>22.914719</td>
          <td>0.036165</td>
          <td>22.069162</td>
          <td>24.445880</td>
          <td>15.646491</td>
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
          <td>26.846463</td>
          <td>23.379415</td>
          <td>20.476706</td>
          <td>22.086830</td>
          <td>22.964259</td>
          <td>29.294367</td>
          <td>22.968191</td>
          <td>0.007268</td>
          <td>25.308120</td>
          <td>0.076809</td>
          <td>23.374195</td>
          <td>0.014194</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.785605</td>
          <td>24.978579</td>
          <td>25.404618</td>
          <td>22.351844</td>
          <td>24.121736</td>
          <td>24.094356</td>
          <td>23.898592</td>
          <td>0.013365</td>
          <td>23.527338</td>
          <td>0.016081</td>
          <td>23.302162</td>
          <td>0.013403</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.949424</td>
          <td>22.567964</td>
          <td>18.182410</td>
          <td>29.098719</td>
          <td>20.794782</td>
          <td>28.872692</td>
          <td>25.352432</td>
          <td>0.046851</td>
          <td>19.352626</td>
          <td>0.005011</td>
          <td>26.006401</td>
          <td>0.141558</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.229866</td>
          <td>20.063558</td>
          <td>21.983917</td>
          <td>23.333465</td>
          <td>21.321330</td>
          <td>20.715530</td>
          <td>24.019320</td>
          <td>0.014720</td>
          <td>24.300533</td>
          <td>0.031343</td>
          <td>27.344276</td>
          <td>0.423647</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.519889</td>
          <td>19.882792</td>
          <td>28.915809</td>
          <td>24.702912</td>
          <td>20.494598</td>
          <td>24.346208</td>
          <td>21.540226</td>
          <td>0.005197</td>
          <td>23.923850</td>
          <td>0.022521</td>
          <td>18.764494</td>
          <td>0.005004</td>
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
          <td>24.983497</td>
          <td>26.075854</td>
          <td>26.556739</td>
          <td>26.900710</td>
          <td>25.946519</td>
          <td>21.339982</td>
          <td>22.825885</td>
          <td>0.006813</td>
          <td>16.378121</td>
          <td>0.005000</td>
          <td>20.895942</td>
          <td>0.005182</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.597567</td>
          <td>25.856704</td>
          <td>25.521450</td>
          <td>26.920610</td>
          <td>27.000038</td>
          <td>22.433892</td>
          <td>23.608941</td>
          <td>0.010738</td>
          <td>23.185867</td>
          <td>0.012245</td>
          <td>27.037938</td>
          <td>0.333807</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.903110</td>
          <td>17.980748</td>
          <td>20.256216</td>
          <td>21.548300</td>
          <td>20.089170</td>
          <td>19.268244</td>
          <td>21.749410</td>
          <td>0.005287</td>
          <td>25.642025</td>
          <td>0.103098</td>
          <td>19.795534</td>
          <td>0.005024</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.415246</td>
          <td>28.469845</td>
          <td>26.455921</td>
          <td>27.484980</td>
          <td>23.712129</td>
          <td>22.432459</td>
          <td>17.156333</td>
          <td>0.005000</td>
          <td>26.697961</td>
          <td>0.253678</td>
          <td>19.797047</td>
          <td>0.005024</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.782519</td>
          <td>23.750562</td>
          <td>23.167039</td>
          <td>25.646895</td>
          <td>20.293372</td>
          <td>22.865980</td>
          <td>22.061893</td>
          <td>0.005500</td>
          <td>24.430821</td>
          <td>0.035183</td>
          <td>15.651783</td>
          <td>0.005000</td>
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
          <td>26.846463</td>
          <td>23.379415</td>
          <td>20.476706</td>
          <td>22.086830</td>
          <td>22.964259</td>
          <td>29.294367</td>
          <td>22.917526</td>
          <td>0.059336</td>
          <td>25.053003</td>
          <td>0.311968</td>
          <td>23.412240</td>
          <td>0.084219</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.785605</td>
          <td>24.978579</td>
          <td>25.404618</td>
          <td>22.351844</td>
          <td>24.121736</td>
          <td>24.094356</td>
          <td>24.098129</td>
          <td>0.166868</td>
          <td>23.740059</td>
          <td>0.102920</td>
          <td>23.377864</td>
          <td>0.081699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.949424</td>
          <td>22.567964</td>
          <td>18.182410</td>
          <td>29.098719</td>
          <td>20.794782</td>
          <td>28.872692</td>
          <td>25.600592</td>
          <td>0.551995</td>
          <td>19.348133</td>
          <td>0.005342</td>
          <td>27.283714</td>
          <td>1.463043</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.229866</td>
          <td>20.063558</td>
          <td>21.983917</td>
          <td>23.333465</td>
          <td>21.321330</td>
          <td>20.715530</td>
          <td>23.807422</td>
          <td>0.129945</td>
          <td>24.458070</td>
          <td>0.191127</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.519889</td>
          <td>19.882792</td>
          <td>28.915809</td>
          <td>24.702912</td>
          <td>20.494598</td>
          <td>24.346208</td>
          <td>21.558776</td>
          <td>0.017947</td>
          <td>24.142741</td>
          <td>0.146061</td>
          <td>18.762671</td>
          <td>0.005143</td>
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
          <td>24.983497</td>
          <td>26.075854</td>
          <td>26.556739</td>
          <td>26.900710</td>
          <td>25.946519</td>
          <td>21.339982</td>
          <td>22.830918</td>
          <td>0.054931</td>
          <td>16.393814</td>
          <td>0.005002</td>
          <td>20.900835</td>
          <td>0.009952</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.597567</td>
          <td>25.856704</td>
          <td>25.521450</td>
          <td>26.920610</td>
          <td>27.000038</td>
          <td>22.433892</td>
          <td>23.788229</td>
          <td>0.127800</td>
          <td>23.254556</td>
          <td>0.067027</td>
          <td>28.830714</td>
          <td>2.771330</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.903110</td>
          <td>17.980748</td>
          <td>20.256216</td>
          <td>21.548300</td>
          <td>20.089170</td>
          <td>19.268244</td>
          <td>21.742184</td>
          <td>0.020986</td>
          <td>25.287664</td>
          <td>0.375482</td>
          <td>19.799701</td>
          <td>0.005897</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.415246</td>
          <td>28.469845</td>
          <td>26.455921</td>
          <td>27.484980</td>
          <td>23.712129</td>
          <td>22.432459</td>
          <td>17.155145</td>
          <td>0.005009</td>
          <td>27.228622</td>
          <td>1.350437</td>
          <td>19.793226</td>
          <td>0.005887</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.782519</td>
          <td>23.750562</td>
          <td>23.167039</td>
          <td>25.646895</td>
          <td>20.293372</td>
          <td>22.865980</td>
          <td>22.080911</td>
          <td>0.028200</td>
          <td>24.422425</td>
          <td>0.185455</td>
          <td>15.651438</td>
          <td>0.005000</td>
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


