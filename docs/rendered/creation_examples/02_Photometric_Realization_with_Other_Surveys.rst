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
          <td>24.972382</td>
          <td>22.629714</td>
          <td>24.218297</td>
          <td>22.260338</td>
          <td>25.053363</td>
          <td>26.048331</td>
          <td>21.020672</td>
          <td>26.044951</td>
          <td>24.461332</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.712266</td>
          <td>25.043505</td>
          <td>21.512256</td>
          <td>26.689400</td>
          <td>20.902305</td>
          <td>19.959187</td>
          <td>19.053362</td>
          <td>28.158453</td>
          <td>22.503785</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.009063</td>
          <td>24.336182</td>
          <td>21.339717</td>
          <td>21.246975</td>
          <td>22.391253</td>
          <td>29.122794</td>
          <td>23.562464</td>
          <td>29.642847</td>
          <td>23.935775</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.741185</td>
          <td>18.321794</td>
          <td>23.359610</td>
          <td>22.765924</td>
          <td>27.258084</td>
          <td>19.333404</td>
          <td>26.644293</td>
          <td>24.948704</td>
          <td>30.495538</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.157303</td>
          <td>22.288223</td>
          <td>22.924155</td>
          <td>26.303416</td>
          <td>22.101506</td>
          <td>23.646530</td>
          <td>21.828157</td>
          <td>22.287719</td>
          <td>22.978120</td>
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
          <td>21.585156</td>
          <td>22.861385</td>
          <td>24.654946</td>
          <td>21.673594</td>
          <td>20.569019</td>
          <td>17.785903</td>
          <td>17.358287</td>
          <td>20.547613</td>
          <td>21.198582</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.021224</td>
          <td>26.502495</td>
          <td>28.141358</td>
          <td>22.651744</td>
          <td>19.849339</td>
          <td>25.717520</td>
          <td>18.542124</td>
          <td>26.593246</td>
          <td>33.201987</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.655964</td>
          <td>23.476139</td>
          <td>21.714976</td>
          <td>22.094037</td>
          <td>24.224151</td>
          <td>24.143843</td>
          <td>25.288495</td>
          <td>24.602538</td>
          <td>21.903946</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.431268</td>
          <td>23.428642</td>
          <td>21.796597</td>
          <td>20.910585</td>
          <td>22.730421</td>
          <td>28.281155</td>
          <td>22.335396</td>
          <td>19.547698</td>
          <td>23.700897</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.843528</td>
          <td>24.574605</td>
          <td>24.676849</td>
          <td>23.416297</td>
          <td>24.292166</td>
          <td>23.318922</td>
          <td>24.943414</td>
          <td>24.869939</td>
          <td>21.894024</td>
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
          <td>24.982967</td>
          <td>0.106681</td>
          <td>22.629612</td>
          <td>0.006708</td>
          <td>24.193604</td>
          <td>0.016288</td>
          <td>22.260668</td>
          <td>0.006732</td>
          <td>25.175594</td>
          <td>0.118246</td>
          <td>25.881064</td>
          <td>0.449418</td>
          <td>21.020672</td>
          <td>26.044951</td>
          <td>24.461332</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.694130</td>
          <td>0.082890</td>
          <td>24.982250</td>
          <td>0.036452</td>
          <td>21.514726</td>
          <td>0.005218</td>
          <td>26.859669</td>
          <td>0.264503</td>
          <td>20.906949</td>
          <td>0.005654</td>
          <td>19.955150</td>
          <td>0.005621</td>
          <td>19.053362</td>
          <td>28.158453</td>
          <td>22.503785</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.004526</td>
          <td>0.005083</td>
          <td>24.325917</td>
          <td>0.020631</td>
          <td>21.347979</td>
          <td>0.005168</td>
          <td>21.243708</td>
          <td>0.005342</td>
          <td>22.392196</td>
          <td>0.010960</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.562464</td>
          <td>29.642847</td>
          <td>23.935775</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.715200</td>
          <td>0.035096</td>
          <td>18.314838</td>
          <td>0.005006</td>
          <td>23.367225</td>
          <td>0.008906</td>
          <td>22.761520</td>
          <td>0.008628</td>
          <td>26.265791</td>
          <td>0.296073</td>
          <td>19.338250</td>
          <td>0.005231</td>
          <td>26.644293</td>
          <td>24.948704</td>
          <td>30.495538</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.165295</td>
          <td>0.006360</td>
          <td>22.288401</td>
          <td>0.006018</td>
          <td>22.923427</td>
          <td>0.007048</td>
          <td>26.455666</td>
          <td>0.189076</td>
          <td>22.085220</td>
          <td>0.008938</td>
          <td>23.613869</td>
          <td>0.067227</td>
          <td>21.828157</td>
          <td>22.287719</td>
          <td>22.978120</td>
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
          <td>21.592533</td>
          <td>0.007458</td>
          <td>22.857062</td>
          <td>0.007389</td>
          <td>24.658521</td>
          <td>0.024121</td>
          <td>21.683084</td>
          <td>0.005695</td>
          <td>20.570364</td>
          <td>0.005380</td>
          <td>17.790902</td>
          <td>0.005024</td>
          <td>17.358287</td>
          <td>20.547613</td>
          <td>21.198582</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.011661</td>
          <td>0.019283</td>
          <td>26.684736</td>
          <td>0.162266</td>
          <td>27.434220</td>
          <td>0.269065</td>
          <td>22.645038</td>
          <td>0.008072</td>
          <td>19.858131</td>
          <td>0.005124</td>
          <td>25.450636</td>
          <td>0.321818</td>
          <td>18.542124</td>
          <td>26.593246</td>
          <td>33.201987</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.674351</td>
          <td>0.014730</td>
          <td>23.489350</td>
          <td>0.010715</td>
          <td>21.720797</td>
          <td>0.005303</td>
          <td>22.088867</td>
          <td>0.006326</td>
          <td>24.171825</td>
          <td>0.048789</td>
          <td>23.942314</td>
          <td>0.089841</td>
          <td>25.288495</td>
          <td>24.602538</td>
          <td>21.903946</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.429052</td>
          <td>0.027400</td>
          <td>23.443506</td>
          <td>0.010382</td>
          <td>21.794942</td>
          <td>0.005342</td>
          <td>20.906305</td>
          <td>0.005199</td>
          <td>22.720888</td>
          <td>0.014032</td>
          <td>27.327606</td>
          <td>1.171828</td>
          <td>22.335396</td>
          <td>19.547698</td>
          <td>23.700897</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.840323</td>
          <td>0.005863</td>
          <td>24.557627</td>
          <td>0.025153</td>
          <td>24.651693</td>
          <td>0.023979</td>
          <td>23.433001</td>
          <td>0.013793</td>
          <td>24.296551</td>
          <td>0.054502</td>
          <td>23.304126</td>
          <td>0.051076</td>
          <td>24.943414</td>
          <td>24.869939</td>
          <td>21.894024</td>
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
          <td>24.972382</td>
          <td>22.629714</td>
          <td>24.218297</td>
          <td>22.260338</td>
          <td>25.053363</td>
          <td>26.048331</td>
          <td>21.016661</td>
          <td>0.005076</td>
          <td>25.984609</td>
          <td>0.138920</td>
          <td>24.418084</td>
          <td>0.034787</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.712266</td>
          <td>25.043505</td>
          <td>21.512256</td>
          <td>26.689400</td>
          <td>20.902305</td>
          <td>19.959187</td>
          <td>19.043382</td>
          <td>0.005002</td>
          <td>29.846690</td>
          <td>1.905395</td>
          <td>22.499325</td>
          <td>0.007772</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.009063</td>
          <td>24.336182</td>
          <td>21.339717</td>
          <td>21.246975</td>
          <td>22.391253</td>
          <td>29.122794</td>
          <td>23.550041</td>
          <td>0.010298</td>
          <td>29.197596</td>
          <td>1.399972</td>
          <td>23.900125</td>
          <td>0.022062</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.741185</td>
          <td>18.321794</td>
          <td>23.359610</td>
          <td>22.765924</td>
          <td>27.258084</td>
          <td>19.333404</td>
          <td>26.615113</td>
          <td>0.142626</td>
          <td>24.894127</td>
          <td>0.053159</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.157303</td>
          <td>22.288223</td>
          <td>22.924155</td>
          <td>26.303416</td>
          <td>22.101506</td>
          <td>23.646530</td>
          <td>21.843326</td>
          <td>0.005339</td>
          <td>22.298978</td>
          <td>0.007035</td>
          <td>22.994666</td>
          <td>0.010629</td>
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
          <td>21.585156</td>
          <td>22.861385</td>
          <td>24.654946</td>
          <td>21.673594</td>
          <td>20.569019</td>
          <td>17.785903</td>
          <td>17.354887</td>
          <td>0.005000</td>
          <td>20.542952</td>
          <td>0.005096</td>
          <td>21.197172</td>
          <td>0.005312</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.021224</td>
          <td>26.502495</td>
          <td>28.141358</td>
          <td>22.651744</td>
          <td>19.849339</td>
          <td>25.717520</td>
          <td>18.549306</td>
          <td>0.005001</td>
          <td>27.132936</td>
          <td>0.359766</td>
          <td>29.009513</td>
          <td>1.267026</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.655964</td>
          <td>23.476139</td>
          <td>21.714976</td>
          <td>22.094037</td>
          <td>24.224151</td>
          <td>24.143843</td>
          <td>25.299707</td>
          <td>0.044699</td>
          <td>24.603452</td>
          <td>0.041025</td>
          <td>21.912643</td>
          <td>0.006085</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.431268</td>
          <td>23.428642</td>
          <td>21.796597</td>
          <td>20.910585</td>
          <td>22.730421</td>
          <td>28.281155</td>
          <td>22.331939</td>
          <td>0.005799</td>
          <td>19.552414</td>
          <td>0.005016</td>
          <td>23.728112</td>
          <td>0.019033</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.843528</td>
          <td>24.574605</td>
          <td>24.676849</td>
          <td>23.416297</td>
          <td>24.292166</td>
          <td>23.318922</td>
          <td>24.960809</td>
          <td>0.033063</td>
          <td>24.924351</td>
          <td>0.054610</td>
          <td>21.899950</td>
          <td>0.006062</td>
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
          <td>24.972382</td>
          <td>22.629714</td>
          <td>24.218297</td>
          <td>22.260338</td>
          <td>25.053363</td>
          <td>26.048331</td>
          <td>21.029856</td>
          <td>0.011736</td>
          <td>26.128026</td>
          <td>0.694523</td>
          <td>24.441375</td>
          <td>0.205005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.712266</td>
          <td>25.043505</td>
          <td>21.512256</td>
          <td>26.689400</td>
          <td>20.902305</td>
          <td>19.959187</td>
          <td>19.045898</td>
          <td>0.005285</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.454552</td>
          <td>0.035933</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.009063</td>
          <td>24.336182</td>
          <td>21.339717</td>
          <td>21.246975</td>
          <td>22.391253</td>
          <td>29.122794</td>
          <td>23.687253</td>
          <td>0.117055</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.922926</td>
          <td>0.131703</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.741185</td>
          <td>18.321794</td>
          <td>23.359610</td>
          <td>22.765924</td>
          <td>27.258084</td>
          <td>19.333404</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.531882</td>
          <td>0.452728</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.157303</td>
          <td>22.288223</td>
          <td>22.924155</td>
          <td>26.303416</td>
          <td>22.101506</td>
          <td>23.646530</td>
          <td>21.797189</td>
          <td>0.022006</td>
          <td>22.330630</td>
          <td>0.029465</td>
          <td>22.972331</td>
          <td>0.056995</td>
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
          <td>21.585156</td>
          <td>22.861385</td>
          <td>24.654946</td>
          <td>21.673594</td>
          <td>20.569019</td>
          <td>17.785903</td>
          <td>17.363736</td>
          <td>0.005013</td>
          <td>20.547415</td>
          <td>0.007562</td>
          <td>21.182279</td>
          <td>0.012212</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.021224</td>
          <td>26.502495</td>
          <td>28.141358</td>
          <td>22.651744</td>
          <td>19.849339</td>
          <td>25.717520</td>
          <td>18.544570</td>
          <td>0.005115</td>
          <td>25.986485</td>
          <td>0.629933</td>
          <td>26.067245</td>
          <td>0.713231</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.655964</td>
          <td>23.476139</td>
          <td>21.714976</td>
          <td>22.094037</td>
          <td>24.224151</td>
          <td>24.143843</td>
          <td>25.180774</td>
          <td>0.403536</td>
          <td>24.285474</td>
          <td>0.165075</td>
          <td>21.925032</td>
          <td>0.022544</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.431268</td>
          <td>23.428642</td>
          <td>21.796597</td>
          <td>20.910585</td>
          <td>22.730421</td>
          <td>28.281155</td>
          <td>22.373024</td>
          <td>0.036528</td>
          <td>19.552230</td>
          <td>0.005491</td>
          <td>23.899982</td>
          <td>0.129110</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.843528</td>
          <td>24.574605</td>
          <td>24.676849</td>
          <td>23.416297</td>
          <td>24.292166</td>
          <td>23.318922</td>
          <td>25.604213</td>
          <td>0.553440</td>
          <td>25.223439</td>
          <td>0.357096</td>
          <td>21.921628</td>
          <td>0.022477</td>
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


