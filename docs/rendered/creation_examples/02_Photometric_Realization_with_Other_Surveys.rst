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
          <td>22.999087</td>
          <td>19.827322</td>
          <td>23.436222</td>
          <td>19.470544</td>
          <td>21.480642</td>
          <td>22.896605</td>
          <td>19.326726</td>
          <td>21.229961</td>
          <td>20.815980</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.338769</td>
          <td>20.435810</td>
          <td>20.406403</td>
          <td>19.939985</td>
          <td>27.483828</td>
          <td>23.066149</td>
          <td>21.064251</td>
          <td>21.044894</td>
          <td>27.757122</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.808945</td>
          <td>29.186742</td>
          <td>24.763384</td>
          <td>24.648434</td>
          <td>20.071030</td>
          <td>20.986725</td>
          <td>24.908888</td>
          <td>20.239676</td>
          <td>27.723530</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.398863</td>
          <td>25.843276</td>
          <td>18.930314</td>
          <td>23.050058</td>
          <td>22.366167</td>
          <td>21.843719</td>
          <td>24.937499</td>
          <td>16.628751</td>
          <td>29.137378</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.891690</td>
          <td>29.953443</td>
          <td>22.295838</td>
          <td>24.497738</td>
          <td>26.582205</td>
          <td>27.374276</td>
          <td>26.635613</td>
          <td>21.174449</td>
          <td>21.713301</td>
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
          <td>25.683760</td>
          <td>23.756449</td>
          <td>22.136960</td>
          <td>15.857509</td>
          <td>23.648701</td>
          <td>18.751863</td>
          <td>21.698186</td>
          <td>22.874143</td>
          <td>22.882404</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.261053</td>
          <td>26.458399</td>
          <td>22.121343</td>
          <td>23.389360</td>
          <td>16.508453</td>
          <td>24.557016</td>
          <td>25.037956</td>
          <td>21.892349</td>
          <td>25.101050</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.071384</td>
          <td>23.301799</td>
          <td>17.834030</td>
          <td>24.901812</td>
          <td>24.566540</td>
          <td>20.263828</td>
          <td>25.629264</td>
          <td>25.607783</td>
          <td>17.895820</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.078002</td>
          <td>26.117789</td>
          <td>25.476236</td>
          <td>27.659266</td>
          <td>22.751918</td>
          <td>21.582411</td>
          <td>22.151251</td>
          <td>22.469817</td>
          <td>19.389154</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.063131</td>
          <td>30.985074</td>
          <td>22.194029</td>
          <td>21.858148</td>
          <td>23.818743</td>
          <td>21.300723</td>
          <td>23.673206</td>
          <td>19.133579</td>
          <td>22.820031</td>
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
          <td>23.002946</td>
          <td>0.019145</td>
          <td>19.813154</td>
          <td>0.005030</td>
          <td>23.433186</td>
          <td>0.009280</td>
          <td>19.470400</td>
          <td>0.005024</td>
          <td>21.479066</td>
          <td>0.006614</td>
          <td>22.939112</td>
          <td>0.036954</td>
          <td>19.326726</td>
          <td>21.229961</td>
          <td>20.815980</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.339776</td>
          <td>0.005434</td>
          <td>20.429723</td>
          <td>0.005066</td>
          <td>20.410525</td>
          <td>0.005041</td>
          <td>19.938022</td>
          <td>0.005046</td>
          <td>26.941265</td>
          <td>0.499251</td>
          <td>23.108034</td>
          <td>0.042918</td>
          <td>21.064251</td>
          <td>21.044894</td>
          <td>27.757122</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.805473</td>
          <td>0.008277</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.744023</td>
          <td>0.025979</td>
          <td>24.590687</td>
          <td>0.036923</td>
          <td>20.062597</td>
          <td>0.005170</td>
          <td>20.985042</td>
          <td>0.008048</td>
          <td>24.908888</td>
          <td>20.239676</td>
          <td>27.723530</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.443104</td>
          <td>0.027732</td>
          <td>26.096964</td>
          <td>0.097557</td>
          <td>18.932041</td>
          <td>0.005006</td>
          <td>23.046281</td>
          <td>0.010361</td>
          <td>22.363411</td>
          <td>0.010739</td>
          <td>21.878319</td>
          <td>0.014914</td>
          <td>24.937499</td>
          <td>16.628751</td>
          <td>29.137378</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.600310</td>
          <td>0.407379</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.294284</td>
          <td>0.005764</td>
          <td>24.510603</td>
          <td>0.034400</td>
          <td>26.386188</td>
          <td>0.326020</td>
          <td>26.375280</td>
          <td>0.643118</td>
          <td>26.635613</td>
          <td>21.174449</td>
          <td>21.713301</td>
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
          <td>25.695287</td>
          <td>0.196355</td>
          <td>23.749455</td>
          <td>0.012948</td>
          <td>22.137012</td>
          <td>0.005593</td>
          <td>15.860846</td>
          <td>0.005000</td>
          <td>23.642559</td>
          <td>0.030550</td>
          <td>18.749050</td>
          <td>0.005093</td>
          <td>21.698186</td>
          <td>22.874143</td>
          <td>22.882404</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.262618</td>
          <td>0.005391</td>
          <td>26.266319</td>
          <td>0.113108</td>
          <td>22.118940</td>
          <td>0.005576</td>
          <td>23.386122</td>
          <td>0.013299</td>
          <td>16.505325</td>
          <td>0.005002</td>
          <td>24.657299</td>
          <td>0.167029</td>
          <td>25.037956</td>
          <td>21.892349</td>
          <td>25.101050</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.929812</td>
          <td>0.101856</td>
          <td>23.309593</td>
          <td>0.009505</td>
          <td>17.845547</td>
          <td>0.005002</td>
          <td>24.910023</td>
          <td>0.049010</td>
          <td>24.530363</td>
          <td>0.067061</td>
          <td>20.267525</td>
          <td>0.006023</td>
          <td>25.629264</td>
          <td>25.607783</td>
          <td>17.895820</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.068714</td>
          <td>0.020212</td>
          <td>26.087065</td>
          <td>0.096714</td>
          <td>25.442123</td>
          <td>0.048086</td>
          <td>26.938973</td>
          <td>0.282130</td>
          <td>22.754803</td>
          <td>0.014413</td>
          <td>21.595071</td>
          <td>0.011980</td>
          <td>22.151251</td>
          <td>22.469817</td>
          <td>19.389154</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.058809</td>
          <td>0.006172</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.192870</td>
          <td>0.005649</td>
          <td>21.856439</td>
          <td>0.005918</td>
          <td>23.855265</td>
          <td>0.036851</td>
          <td>21.300224</td>
          <td>0.009737</td>
          <td>23.673206</td>
          <td>19.133579</td>
          <td>22.820031</td>
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
          <td>22.999087</td>
          <td>19.827322</td>
          <td>23.436222</td>
          <td>19.470544</td>
          <td>21.480642</td>
          <td>22.896605</td>
          <td>19.315315</td>
          <td>0.005003</td>
          <td>21.222614</td>
          <td>0.005327</td>
          <td>20.825828</td>
          <td>0.005160</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.338769</td>
          <td>20.435810</td>
          <td>20.406403</td>
          <td>19.939985</td>
          <td>27.483828</td>
          <td>23.066149</td>
          <td>21.056458</td>
          <td>0.005082</td>
          <td>21.038903</td>
          <td>0.005235</td>
          <td>28.446036</td>
          <td>0.912118</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.808945</td>
          <td>29.186742</td>
          <td>24.763384</td>
          <td>24.648434</td>
          <td>20.071030</td>
          <td>20.986725</td>
          <td>24.955640</td>
          <td>0.032912</td>
          <td>20.240149</td>
          <td>0.005055</td>
          <td>27.151978</td>
          <td>0.365170</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.398863</td>
          <td>25.843276</td>
          <td>18.930314</td>
          <td>23.050058</td>
          <td>22.366167</td>
          <td>21.843719</td>
          <td>24.955203</td>
          <td>0.032899</td>
          <td>16.625993</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.891690</td>
          <td>29.953443</td>
          <td>22.295838</td>
          <td>24.497738</td>
          <td>26.582205</td>
          <td>27.374276</td>
          <td>26.423230</td>
          <td>0.120782</td>
          <td>21.178276</td>
          <td>0.005302</td>
          <td>21.703555</td>
          <td>0.005761</td>
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
          <td>25.683760</td>
          <td>23.756449</td>
          <td>22.136960</td>
          <td>15.857509</td>
          <td>23.648701</td>
          <td>18.751863</td>
          <td>21.701784</td>
          <td>0.005263</td>
          <td>22.874806</td>
          <td>0.009777</td>
          <td>22.885339</td>
          <td>0.009847</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.261053</td>
          <td>26.458399</td>
          <td>22.121343</td>
          <td>23.389360</td>
          <td>16.508453</td>
          <td>24.557016</td>
          <td>25.091638</td>
          <td>0.037138</td>
          <td>21.893254</td>
          <td>0.006051</td>
          <td>25.015627</td>
          <td>0.059235</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.071384</td>
          <td>23.301799</td>
          <td>17.834030</td>
          <td>24.901812</td>
          <td>24.566540</td>
          <td>20.263828</td>
          <td>25.594028</td>
          <td>0.058107</td>
          <td>25.698468</td>
          <td>0.108322</td>
          <td>17.899288</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.078002</td>
          <td>26.117789</td>
          <td>25.476236</td>
          <td>27.659266</td>
          <td>22.751918</td>
          <td>21.582411</td>
          <td>22.146820</td>
          <td>0.005580</td>
          <td>22.471603</td>
          <td>0.007658</td>
          <td>19.391692</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.063131</td>
          <td>30.985074</td>
          <td>22.194029</td>
          <td>21.858148</td>
          <td>23.818743</td>
          <td>21.300723</td>
          <td>23.687672</td>
          <td>0.011373</td>
          <td>19.137754</td>
          <td>0.005007</td>
          <td>22.806122</td>
          <td>0.009339</td>
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
          <td>22.999087</td>
          <td>19.827322</td>
          <td>23.436222</td>
          <td>19.470544</td>
          <td>21.480642</td>
          <td>22.896605</td>
          <td>19.330067</td>
          <td>0.005472</td>
          <td>21.242138</td>
          <td>0.011845</td>
          <td>20.810578</td>
          <td>0.009367</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.338769</td>
          <td>20.435810</td>
          <td>20.406403</td>
          <td>19.939985</td>
          <td>27.483828</td>
          <td>23.066149</td>
          <td>21.072598</td>
          <td>0.012122</td>
          <td>21.067686</td>
          <td>0.010427</td>
          <td>25.475841</td>
          <td>0.467916</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.808945</td>
          <td>29.186742</td>
          <td>24.763384</td>
          <td>24.648434</td>
          <td>20.071030</td>
          <td>20.986725</td>
          <td>24.771270</td>
          <td>0.292130</td>
          <td>20.237236</td>
          <td>0.006572</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.398863</td>
          <td>25.843276</td>
          <td>18.930314</td>
          <td>23.050058</td>
          <td>22.366167</td>
          <td>21.843719</td>
          <td>24.824654</td>
          <td>0.304959</td>
          <td>16.629956</td>
          <td>0.005002</td>
          <td>28.226930</td>
          <td>2.228696</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.891690</td>
          <td>29.953443</td>
          <td>22.295838</td>
          <td>24.497738</td>
          <td>26.582205</td>
          <td>27.374276</td>
          <td>26.578143</td>
          <td>1.050047</td>
          <td>21.159107</td>
          <td>0.011136</td>
          <td>21.700066</td>
          <td>0.018585</td>
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
          <td>25.683760</td>
          <td>23.756449</td>
          <td>22.136960</td>
          <td>15.857509</td>
          <td>23.648701</td>
          <td>18.751863</td>
          <td>21.705367</td>
          <td>0.020332</td>
          <td>22.939394</td>
          <td>0.050628</td>
          <td>22.770754</td>
          <td>0.047622</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.261053</td>
          <td>26.458399</td>
          <td>22.121343</td>
          <td>23.389360</td>
          <td>16.508453</td>
          <td>24.557016</td>
          <td>25.181398</td>
          <td>0.403730</td>
          <td>21.896868</td>
          <td>0.020184</td>
          <td>25.310598</td>
          <td>0.412881</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.071384</td>
          <td>23.301799</td>
          <td>17.834030</td>
          <td>24.901812</td>
          <td>24.566540</td>
          <td>20.263828</td>
          <td>24.758139</td>
          <td>0.289047</td>
          <td>25.213788</td>
          <td>0.354400</td>
          <td>17.896746</td>
          <td>0.005029</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.078002</td>
          <td>26.117789</td>
          <td>25.476236</td>
          <td>27.659266</td>
          <td>22.751918</td>
          <td>21.582411</td>
          <td>22.109316</td>
          <td>0.028915</td>
          <td>22.499819</td>
          <td>0.034228</td>
          <td>19.386542</td>
          <td>0.005438</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.063131</td>
          <td>30.985074</td>
          <td>22.194029</td>
          <td>21.858148</td>
          <td>23.818743</td>
          <td>21.300723</td>
          <td>23.588950</td>
          <td>0.107424</td>
          <td>19.141078</td>
          <td>0.005236</td>
          <td>22.780807</td>
          <td>0.048051</td>
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


