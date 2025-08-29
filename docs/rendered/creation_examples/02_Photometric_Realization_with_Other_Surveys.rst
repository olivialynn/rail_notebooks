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
          <td>23.575443</td>
          <td>23.580278</td>
          <td>22.274118</td>
          <td>15.405415</td>
          <td>23.393822</td>
          <td>22.605057</td>
          <td>24.484778</td>
          <td>21.712819</td>
          <td>25.025249</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.547674</td>
          <td>19.827405</td>
          <td>22.415690</td>
          <td>23.013851</td>
          <td>23.609838</td>
          <td>20.877426</td>
          <td>24.815458</td>
          <td>26.285805</td>
          <td>20.105879</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.576288</td>
          <td>22.654364</td>
          <td>26.820418</td>
          <td>21.375382</td>
          <td>21.764110</td>
          <td>24.066682</td>
          <td>20.296811</td>
          <td>30.931490</td>
          <td>23.572458</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.010404</td>
          <td>20.221942</td>
          <td>22.397134</td>
          <td>20.640014</td>
          <td>21.284339</td>
          <td>19.951591</td>
          <td>20.551737</td>
          <td>22.470684</td>
          <td>21.395576</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.564686</td>
          <td>22.773870</td>
          <td>21.836955</td>
          <td>24.490891</td>
          <td>28.583925</td>
          <td>19.539712</td>
          <td>22.718003</td>
          <td>23.298183</td>
          <td>23.111782</td>
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
          <td>22.332101</td>
          <td>22.991677</td>
          <td>24.632827</td>
          <td>23.343244</td>
          <td>17.280326</td>
          <td>20.043880</td>
          <td>19.885759</td>
          <td>24.679366</td>
          <td>28.264059</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.783875</td>
          <td>25.633852</td>
          <td>22.287521</td>
          <td>27.875378</td>
          <td>20.389564</td>
          <td>20.851934</td>
          <td>23.931281</td>
          <td>25.585645</td>
          <td>28.132649</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.120565</td>
          <td>21.694198</td>
          <td>19.189825</td>
          <td>17.306137</td>
          <td>20.675108</td>
          <td>21.785271</td>
          <td>22.403402</td>
          <td>22.157051</td>
          <td>22.358178</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.947357</td>
          <td>18.574222</td>
          <td>19.805801</td>
          <td>21.541929</td>
          <td>21.709440</td>
          <td>25.230418</td>
          <td>23.014008</td>
          <td>24.930150</td>
          <td>20.199336</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.295035</td>
          <td>24.705754</td>
          <td>25.811110</td>
          <td>21.965747</td>
          <td>24.041107</td>
          <td>22.606975</td>
          <td>26.680099</td>
          <td>19.355258</td>
          <td>27.076788</td>
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
          <td>23.565961</td>
          <td>0.030831</td>
          <td>23.586142</td>
          <td>0.011474</td>
          <td>22.275228</td>
          <td>0.005741</td>
          <td>15.410034</td>
          <td>0.005000</td>
          <td>23.356409</td>
          <td>0.023805</td>
          <td>22.579320</td>
          <td>0.026934</td>
          <td>24.484778</td>
          <td>21.712819</td>
          <td>25.025249</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.539806</td>
          <td>0.005049</td>
          <td>19.827318</td>
          <td>0.005031</td>
          <td>22.423432</td>
          <td>0.005940</td>
          <td>23.013672</td>
          <td>0.010133</td>
          <td>23.581098</td>
          <td>0.028947</td>
          <td>20.878082</td>
          <td>0.007609</td>
          <td>24.815458</td>
          <td>26.285805</td>
          <td>20.105879</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.809157</td>
          <td>0.215964</td>
          <td>22.654506</td>
          <td>0.006773</td>
          <td>26.589214</td>
          <td>0.132028</td>
          <td>21.383148</td>
          <td>0.005428</td>
          <td>21.761764</td>
          <td>0.007474</td>
          <td>24.180400</td>
          <td>0.110678</td>
          <td>20.296811</td>
          <td>30.931490</td>
          <td>23.572458</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.209041</td>
          <td>0.129766</td>
          <td>20.220574</td>
          <td>0.005050</td>
          <td>22.399894</td>
          <td>0.005905</td>
          <td>20.642673</td>
          <td>0.005132</td>
          <td>21.291948</td>
          <td>0.006207</td>
          <td>19.953240</td>
          <td>0.005619</td>
          <td>20.551737</td>
          <td>22.470684</td>
          <td>21.395576</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.759028</td>
          <td>0.459495</td>
          <td>22.766379</td>
          <td>0.007092</td>
          <td>21.839128</td>
          <td>0.005367</td>
          <td>24.498541</td>
          <td>0.034036</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.537346</td>
          <td>0.005318</td>
          <td>22.718003</td>
          <td>23.298183</td>
          <td>23.111782</td>
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
          <td>22.333532</td>
          <td>0.011462</td>
          <td>23.000773</td>
          <td>0.007937</td>
          <td>24.656813</td>
          <td>0.024086</td>
          <td>23.350574</td>
          <td>0.012939</td>
          <td>17.275765</td>
          <td>0.005004</td>
          <td>20.047385</td>
          <td>0.005720</td>
          <td>19.885759</td>
          <td>24.679366</td>
          <td>28.264059</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.781047</td>
          <td>0.005022</td>
          <td>25.613016</td>
          <td>0.063691</td>
          <td>22.290488</td>
          <td>0.005760</td>
          <td>27.957505</td>
          <td>0.612448</td>
          <td>20.395679</td>
          <td>0.005288</td>
          <td>20.853746</td>
          <td>0.007517</td>
          <td>23.931281</td>
          <td>25.585645</td>
          <td>28.132649</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.980597</td>
          <td>0.248848</td>
          <td>21.694650</td>
          <td>0.005408</td>
          <td>19.188608</td>
          <td>0.005009</td>
          <td>17.304567</td>
          <td>0.005002</td>
          <td>20.682267</td>
          <td>0.005455</td>
          <td>21.811071</td>
          <td>0.014138</td>
          <td>22.403402</td>
          <td>22.157051</td>
          <td>22.358178</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.689667</td>
          <td>0.195431</td>
          <td>18.570580</td>
          <td>0.005008</td>
          <td>19.805598</td>
          <td>0.005018</td>
          <td>21.536921</td>
          <td>0.005549</td>
          <td>21.722255</td>
          <td>0.007333</td>
          <td>25.556573</td>
          <td>0.349973</td>
          <td>23.014008</td>
          <td>24.930150</td>
          <td>20.199336</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.265980</td>
          <td>0.136290</td>
          <td>24.716513</td>
          <td>0.028873</td>
          <td>25.749451</td>
          <td>0.063167</td>
          <td>21.968296</td>
          <td>0.006097</td>
          <td>24.003309</td>
          <td>0.042013</td>
          <td>22.623094</td>
          <td>0.027983</td>
          <td>26.680099</td>
          <td>19.355258</td>
          <td>27.076788</td>
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
          <td>23.575443</td>
          <td>23.580278</td>
          <td>22.274118</td>
          <td>15.405415</td>
          <td>23.393822</td>
          <td>22.605057</td>
          <td>24.490335</td>
          <td>0.021876</td>
          <td>21.717746</td>
          <td>0.005780</td>
          <td>25.001030</td>
          <td>0.058470</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.547674</td>
          <td>19.827405</td>
          <td>22.415690</td>
          <td>23.013851</td>
          <td>23.609838</td>
          <td>20.877426</td>
          <td>24.809633</td>
          <td>0.028924</td>
          <td>26.496122</td>
          <td>0.214619</td>
          <td>20.101251</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.576288</td>
          <td>22.654364</td>
          <td>26.820418</td>
          <td>21.375382</td>
          <td>21.764110</td>
          <td>24.066682</td>
          <td>20.305888</td>
          <td>0.005021</td>
          <td>27.961009</td>
          <td>0.663370</td>
          <td>23.583070</td>
          <td>0.016843</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.010404</td>
          <td>20.221942</td>
          <td>22.397134</td>
          <td>20.640014</td>
          <td>21.284339</td>
          <td>19.951591</td>
          <td>20.549614</td>
          <td>0.005032</td>
          <td>22.467093</td>
          <td>0.007640</td>
          <td>21.390560</td>
          <td>0.005441</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.564686</td>
          <td>22.773870</td>
          <td>21.836955</td>
          <td>24.490891</td>
          <td>28.583925</td>
          <td>19.539712</td>
          <td>22.718439</td>
          <td>0.006525</td>
          <td>23.277845</td>
          <td>0.013149</td>
          <td>23.105419</td>
          <td>0.011524</td>
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
          <td>22.332101</td>
          <td>22.991677</td>
          <td>24.632827</td>
          <td>23.343244</td>
          <td>17.280326</td>
          <td>20.043880</td>
          <td>19.885609</td>
          <td>0.005010</td>
          <td>24.771440</td>
          <td>0.047651</td>
          <td>28.982814</td>
          <td>1.248710</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.783875</td>
          <td>25.633852</td>
          <td>22.287521</td>
          <td>27.875378</td>
          <td>20.389564</td>
          <td>20.851934</td>
          <td>23.932930</td>
          <td>0.013734</td>
          <td>25.628024</td>
          <td>0.101839</td>
          <td>28.775628</td>
          <td>1.111493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.120565</td>
          <td>21.694198</td>
          <td>19.189825</td>
          <td>17.306137</td>
          <td>20.675108</td>
          <td>21.785271</td>
          <td>22.411171</td>
          <td>0.005914</td>
          <td>22.152909</td>
          <td>0.006612</td>
          <td>22.349594</td>
          <td>0.007203</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.947357</td>
          <td>18.574222</td>
          <td>19.805801</td>
          <td>21.541929</td>
          <td>21.709440</td>
          <td>25.230418</td>
          <td>23.005428</td>
          <td>0.007402</td>
          <td>24.979244</td>
          <td>0.057347</td>
          <td>20.205437</td>
          <td>0.005052</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.295035</td>
          <td>24.705754</td>
          <td>25.811110</td>
          <td>21.965747</td>
          <td>24.041107</td>
          <td>22.606975</td>
          <td>26.925035</td>
          <td>0.185865</td>
          <td>19.355461</td>
          <td>0.005011</td>
          <td>26.847533</td>
          <td>0.286578</td>
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
          <td>23.575443</td>
          <td>23.580278</td>
          <td>22.274118</td>
          <td>15.405415</td>
          <td>23.393822</td>
          <td>22.605057</td>
          <td>24.284651</td>
          <td>0.195459</td>
          <td>21.726289</td>
          <td>0.017463</td>
          <td>25.087658</td>
          <td>0.347187</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.547674</td>
          <td>19.827405</td>
          <td>22.415690</td>
          <td>23.013851</td>
          <td>23.609838</td>
          <td>20.877426</td>
          <td>24.589312</td>
          <td>0.251882</td>
          <td>26.611727</td>
          <td>0.949938</td>
          <td>20.112080</td>
          <td>0.006509</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.576288</td>
          <td>22.654364</td>
          <td>26.820418</td>
          <td>21.375382</td>
          <td>21.764110</td>
          <td>24.066682</td>
          <td>20.297331</td>
          <td>0.007372</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.708331</td>
          <td>0.109261</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.010404</td>
          <td>20.221942</td>
          <td>22.397134</td>
          <td>20.640014</td>
          <td>21.284339</td>
          <td>19.951591</td>
          <td>20.540623</td>
          <td>0.008421</td>
          <td>22.443964</td>
          <td>0.032573</td>
          <td>21.390079</td>
          <td>0.014376</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.564686</td>
          <td>22.773870</td>
          <td>21.836955</td>
          <td>24.490891</td>
          <td>28.583925</td>
          <td>19.539712</td>
          <td>22.779811</td>
          <td>0.052485</td>
          <td>23.316314</td>
          <td>0.070805</td>
          <td>23.137159</td>
          <td>0.065998</td>
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
          <td>22.332101</td>
          <td>22.991677</td>
          <td>24.632827</td>
          <td>23.343244</td>
          <td>17.280326</td>
          <td>20.043880</td>
          <td>19.888648</td>
          <td>0.006232</td>
          <td>24.373578</td>
          <td>0.177934</td>
          <td>25.349882</td>
          <td>0.425461</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.783875</td>
          <td>25.633852</td>
          <td>22.287521</td>
          <td>27.875378</td>
          <td>20.389564</td>
          <td>20.851934</td>
          <td>24.020934</td>
          <td>0.156209</td>
          <td>25.869488</td>
          <td>0.579978</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.120565</td>
          <td>21.694198</td>
          <td>19.189825</td>
          <td>17.306137</td>
          <td>20.675108</td>
          <td>21.785271</td>
          <td>22.397003</td>
          <td>0.037315</td>
          <td>22.158883</td>
          <td>0.025334</td>
          <td>22.331872</td>
          <td>0.032226</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.947357</td>
          <td>18.574222</td>
          <td>19.805801</td>
          <td>21.541929</td>
          <td>21.709440</td>
          <td>25.230418</td>
          <td>22.992333</td>
          <td>0.063420</td>
          <td>24.981713</td>
          <td>0.294602</td>
          <td>20.196808</td>
          <td>0.006730</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.295035</td>
          <td>24.705754</td>
          <td>25.811110</td>
          <td>21.965747</td>
          <td>24.041107</td>
          <td>22.606975</td>
          <td>28.479328</td>
          <td>2.541755</td>
          <td>19.348802</td>
          <td>0.005343</td>
          <td>25.726495</td>
          <td>0.562392</td>
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


