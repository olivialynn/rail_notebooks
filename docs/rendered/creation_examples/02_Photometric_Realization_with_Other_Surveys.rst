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
          <td>25.423949</td>
          <td>24.299774</td>
          <td>22.316558</td>
          <td>22.509463</td>
          <td>23.815093</td>
          <td>20.802580</td>
          <td>22.616284</td>
          <td>19.137405</td>
          <td>17.474313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.914337</td>
          <td>23.695942</td>
          <td>22.167966</td>
          <td>23.145401</td>
          <td>29.272109</td>
          <td>24.342202</td>
          <td>21.205686</td>
          <td>30.089619</td>
          <td>23.969244</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.106836</td>
          <td>25.041600</td>
          <td>26.846751</td>
          <td>22.417651</td>
          <td>18.702777</td>
          <td>23.860663</td>
          <td>25.658738</td>
          <td>19.038112</td>
          <td>20.764598</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.432824</td>
          <td>21.808301</td>
          <td>23.473235</td>
          <td>24.993651</td>
          <td>23.377725</td>
          <td>24.017697</td>
          <td>28.668410</td>
          <td>17.351019</td>
          <td>17.293819</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.247210</td>
          <td>21.892409</td>
          <td>25.608684</td>
          <td>23.543598</td>
          <td>21.906789</td>
          <td>27.480256</td>
          <td>18.174905</td>
          <td>25.275882</td>
          <td>26.373916</td>
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
          <td>17.162017</td>
          <td>19.534083</td>
          <td>22.121316</td>
          <td>26.579565</td>
          <td>21.206421</td>
          <td>24.041805</td>
          <td>23.049642</td>
          <td>24.100046</td>
          <td>20.539832</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.003744</td>
          <td>21.523527</td>
          <td>17.746034</td>
          <td>25.800040</td>
          <td>24.026021</td>
          <td>25.801730</td>
          <td>20.829552</td>
          <td>25.107066</td>
          <td>20.462392</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.836348</td>
          <td>22.972286</td>
          <td>24.293644</td>
          <td>23.829536</td>
          <td>25.536420</td>
          <td>23.176411</td>
          <td>23.834338</td>
          <td>22.172390</td>
          <td>22.902797</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.950070</td>
          <td>21.396718</td>
          <td>20.959458</td>
          <td>24.372343</td>
          <td>16.886995</td>
          <td>25.942154</td>
          <td>21.397707</td>
          <td>21.299541</td>
          <td>24.755993</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.933902</td>
          <td>23.730826</td>
          <td>19.798484</td>
          <td>31.300279</td>
          <td>21.756825</td>
          <td>23.118072</td>
          <td>25.173489</td>
          <td>17.036916</td>
          <td>24.733329</td>
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
          <td>25.442313</td>
          <td>0.158513</td>
          <td>24.319678</td>
          <td>0.020523</td>
          <td>22.324945</td>
          <td>0.005803</td>
          <td>22.514411</td>
          <td>0.007538</td>
          <td>23.785958</td>
          <td>0.034662</td>
          <td>20.803065</td>
          <td>0.007335</td>
          <td>22.616284</td>
          <td>19.137405</td>
          <td>17.474313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.895098</td>
          <td>0.005932</td>
          <td>23.701148</td>
          <td>0.012485</td>
          <td>22.168933</td>
          <td>0.005625</td>
          <td>23.140399</td>
          <td>0.011070</td>
          <td>28.368472</td>
          <td>1.253619</td>
          <td>24.243977</td>
          <td>0.116982</td>
          <td>21.205686</td>
          <td>30.089619</td>
          <td>23.969244</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.134982</td>
          <td>0.050699</td>
          <td>25.076847</td>
          <td>0.039625</td>
          <td>27.310651</td>
          <td>0.243146</td>
          <td>22.407640</td>
          <td>0.007165</td>
          <td>18.698696</td>
          <td>0.005024</td>
          <td>23.896361</td>
          <td>0.086281</td>
          <td>25.658738</td>
          <td>19.038112</td>
          <td>20.764598</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.433528</td>
          <td>0.012304</td>
          <td>21.807056</td>
          <td>0.005485</td>
          <td>23.477958</td>
          <td>0.009551</td>
          <td>25.019194</td>
          <td>0.053999</td>
          <td>23.343639</td>
          <td>0.023545</td>
          <td>24.122588</td>
          <td>0.105228</td>
          <td>28.668410</td>
          <td>17.351019</td>
          <td>17.293819</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.204679</td>
          <td>0.053905</td>
          <td>21.885425</td>
          <td>0.005547</td>
          <td>25.536486</td>
          <td>0.052289</td>
          <td>23.545024</td>
          <td>0.015077</td>
          <td>21.923108</td>
          <td>0.008130</td>
          <td>27.133187</td>
          <td>1.047470</td>
          <td>18.174905</td>
          <td>25.275882</td>
          <td>26.373916</td>
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
          <td>17.168686</td>
          <td>0.005012</td>
          <td>19.542081</td>
          <td>0.005022</td>
          <td>22.120405</td>
          <td>0.005578</td>
          <td>26.287016</td>
          <td>0.163860</td>
          <td>21.200532</td>
          <td>0.006045</td>
          <td>24.140428</td>
          <td>0.106882</td>
          <td>23.049642</td>
          <td>24.100046</td>
          <td>20.539832</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.008117</td>
          <td>0.009279</td>
          <td>21.526438</td>
          <td>0.005316</td>
          <td>17.748782</td>
          <td>0.005002</td>
          <td>25.698567</td>
          <td>0.098432</td>
          <td>24.073965</td>
          <td>0.044731</td>
          <td>25.450794</td>
          <td>0.321858</td>
          <td>20.829552</td>
          <td>25.107066</td>
          <td>20.462392</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.862382</td>
          <td>0.017081</td>
          <td>22.982876</td>
          <td>0.007863</td>
          <td>24.297589</td>
          <td>0.017747</td>
          <td>23.815606</td>
          <td>0.018839</td>
          <td>25.640672</td>
          <td>0.176388</td>
          <td>23.162741</td>
          <td>0.045052</td>
          <td>23.834338</td>
          <td>22.172390</td>
          <td>22.902797</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.967813</td>
          <td>0.105283</td>
          <td>21.395538</td>
          <td>0.005260</td>
          <td>20.962104</td>
          <td>0.005092</td>
          <td>24.376863</td>
          <td>0.030577</td>
          <td>16.886262</td>
          <td>0.005003</td>
          <td>25.390283</td>
          <td>0.306665</td>
          <td>21.397707</td>
          <td>21.299541</td>
          <td>24.755993</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.940058</td>
          <td>0.005992</td>
          <td>23.740621</td>
          <td>0.012862</td>
          <td>19.795415</td>
          <td>0.005018</td>
          <td>29.066790</td>
          <td>1.230093</td>
          <td>21.747499</td>
          <td>0.007423</td>
          <td>23.018511</td>
          <td>0.039644</td>
          <td>25.173489</td>
          <td>17.036916</td>
          <td>24.733329</td>
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
          <td>25.423949</td>
          <td>24.299774</td>
          <td>22.316558</td>
          <td>22.509463</td>
          <td>23.815093</td>
          <td>20.802580</td>
          <td>22.619581</td>
          <td>0.006297</td>
          <td>19.141149</td>
          <td>0.005007</td>
          <td>17.466404</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.914337</td>
          <td>23.695942</td>
          <td>22.167966</td>
          <td>23.145401</td>
          <td>29.272109</td>
          <td>24.342202</td>
          <td>21.206902</td>
          <td>0.005107</td>
          <td>32.367083</td>
          <td>4.241709</td>
          <td>23.995879</td>
          <td>0.023976</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.106836</td>
          <td>25.041600</td>
          <td>26.846751</td>
          <td>22.417651</td>
          <td>18.702777</td>
          <td>23.860663</td>
          <td>25.696111</td>
          <td>0.063633</td>
          <td>19.034635</td>
          <td>0.005006</td>
          <td>20.757300</td>
          <td>0.005141</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.432824</td>
          <td>21.808301</td>
          <td>23.473235</td>
          <td>24.993651</td>
          <td>23.377725</td>
          <td>24.017697</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.342991</td>
          <td>0.005000</td>
          <td>17.281656</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.247210</td>
          <td>21.892409</td>
          <td>25.608684</td>
          <td>23.543598</td>
          <td>21.906789</td>
          <td>27.480256</td>
          <td>18.176831</td>
          <td>0.005000</td>
          <td>25.433290</td>
          <td>0.085799</td>
          <td>26.155275</td>
          <td>0.160870</td>
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
          <td>17.162017</td>
          <td>19.534083</td>
          <td>22.121316</td>
          <td>26.579565</td>
          <td>21.206421</td>
          <td>24.041805</td>
          <td>23.042823</td>
          <td>0.007544</td>
          <td>24.123069</td>
          <td>0.026801</td>
          <td>20.543161</td>
          <td>0.005096</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.003744</td>
          <td>21.523527</td>
          <td>17.746034</td>
          <td>25.800040</td>
          <td>24.026021</td>
          <td>25.801730</td>
          <td>20.827940</td>
          <td>0.005054</td>
          <td>25.114833</td>
          <td>0.064701</td>
          <td>20.460849</td>
          <td>0.005082</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.836348</td>
          <td>22.972286</td>
          <td>24.293644</td>
          <td>23.829536</td>
          <td>25.536420</td>
          <td>23.176411</td>
          <td>23.826219</td>
          <td>0.012631</td>
          <td>22.169274</td>
          <td>0.006655</td>
          <td>22.919840</td>
          <td>0.010084</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.950070</td>
          <td>21.396718</td>
          <td>20.959458</td>
          <td>24.372343</td>
          <td>16.886995</td>
          <td>25.942154</td>
          <td>21.404951</td>
          <td>0.005154</td>
          <td>21.298958</td>
          <td>0.005375</td>
          <td>24.784950</td>
          <td>0.048229</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.933902</td>
          <td>23.730826</td>
          <td>19.798484</td>
          <td>31.300279</td>
          <td>21.756825</td>
          <td>23.118072</td>
          <td>25.195274</td>
          <td>0.040727</td>
          <td>17.036159</td>
          <td>0.005000</td>
          <td>24.782391</td>
          <td>0.048119</td>
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
          <td>25.423949</td>
          <td>24.299774</td>
          <td>22.316558</td>
          <td>22.509463</td>
          <td>23.815093</td>
          <td>20.802580</td>
          <td>22.639437</td>
          <td>0.046311</td>
          <td>19.139862</td>
          <td>0.005236</td>
          <td>17.471750</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.914337</td>
          <td>23.695942</td>
          <td>22.167966</td>
          <td>23.145401</td>
          <td>29.272109</td>
          <td>24.342202</td>
          <td>21.195108</td>
          <td>0.013329</td>
          <td>26.118494</td>
          <td>0.690029</td>
          <td>23.982075</td>
          <td>0.138617</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.106836</td>
          <td>25.041600</td>
          <td>26.846751</td>
          <td>22.417651</td>
          <td>18.702777</td>
          <td>23.860663</td>
          <td>25.834745</td>
          <td>0.651442</td>
          <td>19.041300</td>
          <td>0.005197</td>
          <td>20.771545</td>
          <td>0.009132</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.432824</td>
          <td>21.808301</td>
          <td>23.473235</td>
          <td>24.993651</td>
          <td>23.377725</td>
          <td>24.017697</td>
          <td>26.028342</td>
          <td>0.743081</td>
          <td>17.357457</td>
          <td>0.005009</td>
          <td>17.292664</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.247210</td>
          <td>21.892409</td>
          <td>25.608684</td>
          <td>23.543598</td>
          <td>21.906789</td>
          <td>27.480256</td>
          <td>18.176535</td>
          <td>0.005059</td>
          <td>25.588833</td>
          <td>0.472482</td>
          <td>26.432035</td>
          <td>0.904183</td>
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
          <td>17.162017</td>
          <td>19.534083</td>
          <td>22.121316</td>
          <td>26.579565</td>
          <td>21.206421</td>
          <td>24.041805</td>
          <td>23.030216</td>
          <td>0.065592</td>
          <td>24.103715</td>
          <td>0.141230</td>
          <td>20.533088</td>
          <td>0.007917</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.003744</td>
          <td>21.523527</td>
          <td>17.746034</td>
          <td>25.800040</td>
          <td>24.026021</td>
          <td>25.801730</td>
          <td>20.821315</td>
          <td>0.010094</td>
          <td>25.498112</td>
          <td>0.441332</td>
          <td>20.474114</td>
          <td>0.007668</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.836348</td>
          <td>22.972286</td>
          <td>24.293644</td>
          <td>23.829536</td>
          <td>25.536420</td>
          <td>23.176411</td>
          <td>23.710699</td>
          <td>0.119471</td>
          <td>22.141411</td>
          <td>0.024949</td>
          <td>22.826534</td>
          <td>0.050050</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.950070</td>
          <td>21.396718</td>
          <td>20.959458</td>
          <td>24.372343</td>
          <td>16.886995</td>
          <td>25.942154</td>
          <td>21.399411</td>
          <td>0.015715</td>
          <td>21.279298</td>
          <td>0.012184</td>
          <td>24.874818</td>
          <td>0.292968</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.933902</td>
          <td>23.730826</td>
          <td>19.798484</td>
          <td>31.300279</td>
          <td>21.756825</td>
          <td>23.118072</td>
          <td>24.926167</td>
          <td>0.330704</td>
          <td>17.037111</td>
          <td>0.005005</td>
          <td>24.835201</td>
          <td>0.283730</td>
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


