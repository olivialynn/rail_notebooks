Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


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
          <td>22.592771</td>
          <td>25.772394</td>
          <td>25.883902</td>
          <td>20.802655</td>
          <td>26.300639</td>
          <td>22.316392</td>
          <td>22.488680</td>
          <td>27.168765</td>
          <td>24.930708</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.730656</td>
          <td>13.623588</td>
          <td>23.215351</td>
          <td>20.753809</td>
          <td>20.370790</td>
          <td>24.460678</td>
          <td>22.258600</td>
          <td>21.681288</td>
          <td>21.843032</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.393935</td>
          <td>19.270263</td>
          <td>20.946593</td>
          <td>19.992976</td>
          <td>25.894789</td>
          <td>17.006665</td>
          <td>19.115662</td>
          <td>22.104229</td>
          <td>25.543010</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.111998</td>
          <td>23.601598</td>
          <td>21.963500</td>
          <td>21.322763</td>
          <td>23.178983</td>
          <td>20.408842</td>
          <td>23.890791</td>
          <td>21.252420</td>
          <td>22.562963</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.625284</td>
          <td>20.902819</td>
          <td>26.865352</td>
          <td>21.474095</td>
          <td>25.417051</td>
          <td>24.669178</td>
          <td>16.872956</td>
          <td>18.586848</td>
          <td>24.542632</td>
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
          <td>19.120719</td>
          <td>27.497973</td>
          <td>28.513292</td>
          <td>26.425235</td>
          <td>26.723250</td>
          <td>27.506768</td>
          <td>26.678440</td>
          <td>29.754957</td>
          <td>23.249771</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.184631</td>
          <td>22.501172</td>
          <td>23.110429</td>
          <td>20.735052</td>
          <td>21.900551</td>
          <td>18.036413</td>
          <td>26.964676</td>
          <td>23.347417</td>
          <td>24.925935</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214485</td>
          <td>24.124091</td>
          <td>21.180555</td>
          <td>26.541921</td>
          <td>21.508120</td>
          <td>17.003061</td>
          <td>24.592917</td>
          <td>28.242165</td>
          <td>19.640724</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.383474</td>
          <td>16.028738</td>
          <td>23.716013</td>
          <td>22.524019</td>
          <td>21.312426</td>
          <td>28.817778</td>
          <td>22.891953</td>
          <td>25.953929</td>
          <td>24.256074</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.972382</td>
          <td>20.339517</td>
          <td>20.285268</td>
          <td>16.923547</td>
          <td>20.428349</td>
          <td>22.215956</td>
          <td>23.930754</td>
          <td>17.642349</td>
          <td>26.581441</td>
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
          <td>22.585931</td>
          <td>0.013769</td>
          <td>25.876999</td>
          <td>0.080413</td>
          <td>25.875080</td>
          <td>0.070603</td>
          <td>20.797208</td>
          <td>0.005168</td>
          <td>26.331201</td>
          <td>0.312033</td>
          <td>22.300176</td>
          <td>0.021166</td>
          <td>22.488680</td>
          <td>27.168765</td>
          <td>24.930708</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.071807</td>
          <td>1.103711</td>
          <td>13.629712</td>
          <td>0.005000</td>
          <td>23.210838</td>
          <td>0.008129</td>
          <td>20.754108</td>
          <td>0.005157</td>
          <td>20.367231</td>
          <td>0.005275</td>
          <td>24.579311</td>
          <td>0.156266</td>
          <td>22.258600</td>
          <td>21.681288</td>
          <td>21.843032</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.380924</td>
          <td>0.005015</td>
          <td>19.274710</td>
          <td>0.005016</td>
          <td>20.950113</td>
          <td>0.005091</td>
          <td>19.991618</td>
          <td>0.005050</td>
          <td>25.628243</td>
          <td>0.174537</td>
          <td>17.003958</td>
          <td>0.005009</td>
          <td>19.115662</td>
          <td>22.104229</td>
          <td>25.543010</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.144386</td>
          <td>0.051120</td>
          <td>23.601554</td>
          <td>0.011602</td>
          <td>21.961104</td>
          <td>0.005447</td>
          <td>21.327807</td>
          <td>0.005391</td>
          <td>23.200466</td>
          <td>0.020825</td>
          <td>20.405573</td>
          <td>0.006271</td>
          <td>23.890791</td>
          <td>21.252420</td>
          <td>22.562963</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.639488</td>
          <td>0.032859</td>
          <td>20.900967</td>
          <td>0.005126</td>
          <td>26.848896</td>
          <td>0.165025</td>
          <td>21.471778</td>
          <td>0.005494</td>
          <td>25.583845</td>
          <td>0.168069</td>
          <td>24.655748</td>
          <td>0.166808</td>
          <td>16.872956</td>
          <td>18.586848</td>
          <td>24.542632</td>
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
          <td>19.126288</td>
          <td>0.005095</td>
          <td>27.225746</td>
          <td>0.255269</td>
          <td>32.561426</td>
          <td>3.782814</td>
          <td>26.038241</td>
          <td>0.132325</td>
          <td>26.895079</td>
          <td>0.482461</td>
          <td>28.172630</td>
          <td>1.796310</td>
          <td>26.678440</td>
          <td>29.754957</td>
          <td>23.249771</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.175740</td>
          <td>0.010305</td>
          <td>22.493145</td>
          <td>0.006391</td>
          <td>23.109100</td>
          <td>0.007699</td>
          <td>20.733719</td>
          <td>0.005152</td>
          <td>21.897257</td>
          <td>0.008016</td>
          <td>18.036433</td>
          <td>0.005034</td>
          <td>26.964676</td>
          <td>23.347417</td>
          <td>24.925935</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.255214</td>
          <td>0.023628</td>
          <td>24.130709</td>
          <td>0.017527</td>
          <td>21.183015</td>
          <td>0.005130</td>
          <td>26.848286</td>
          <td>0.262054</td>
          <td>21.509361</td>
          <td>0.006691</td>
          <td>17.012325</td>
          <td>0.005009</td>
          <td>24.592917</td>
          <td>28.242165</td>
          <td>19.640724</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.704993</td>
          <td>0.441176</td>
          <td>16.024649</td>
          <td>0.005001</td>
          <td>23.717816</td>
          <td>0.011264</td>
          <td>22.535429</td>
          <td>0.007618</td>
          <td>21.316448</td>
          <td>0.006254</td>
          <td>26.807463</td>
          <td>0.857658</td>
          <td>22.891953</td>
          <td>25.953929</td>
          <td>24.256074</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.069433</td>
          <td>0.115003</td>
          <td>20.338460</td>
          <td>0.005058</td>
          <td>20.279302</td>
          <td>0.005034</td>
          <td>16.924697</td>
          <td>0.005001</td>
          <td>20.436851</td>
          <td>0.005307</td>
          <td>22.214587</td>
          <td>0.019682</td>
          <td>23.930754</td>
          <td>17.642349</td>
          <td>26.581441</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


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
          <td>22.592771</td>
          <td>25.772394</td>
          <td>25.883902</td>
          <td>20.802655</td>
          <td>26.300639</td>
          <td>22.316392</td>
          <td>22.485183</td>
          <td>0.006036</td>
          <td>27.771633</td>
          <td>0.580866</td>
          <td>24.960474</td>
          <td>0.056396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.730656</td>
          <td>13.623588</td>
          <td>23.215351</td>
          <td>20.753809</td>
          <td>20.370790</td>
          <td>24.460678</td>
          <td>22.258150</td>
          <td>0.005704</td>
          <td>21.674505</td>
          <td>0.005724</td>
          <td>21.838768</td>
          <td>0.005958</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.393935</td>
          <td>19.270263</td>
          <td>20.946593</td>
          <td>19.992976</td>
          <td>25.894789</td>
          <td>17.006665</td>
          <td>19.117308</td>
          <td>0.005002</td>
          <td>22.100380</td>
          <td>0.006480</td>
          <td>25.770949</td>
          <td>0.115403</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.111998</td>
          <td>23.601598</td>
          <td>21.963500</td>
          <td>21.322763</td>
          <td>23.178983</td>
          <td>20.408842</td>
          <td>23.868967</td>
          <td>0.013058</td>
          <td>21.250088</td>
          <td>0.005343</td>
          <td>22.557787</td>
          <td>0.008026</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.625284</td>
          <td>20.902819</td>
          <td>26.865352</td>
          <td>21.474095</td>
          <td>25.417051</td>
          <td>24.669178</td>
          <td>16.870042</td>
          <td>0.005000</td>
          <td>18.592264</td>
          <td>0.005003</td>
          <td>24.483425</td>
          <td>0.036867</td>
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
          <td>19.120719</td>
          <td>27.497973</td>
          <td>28.513292</td>
          <td>26.425235</td>
          <td>26.723250</td>
          <td>27.506768</td>
          <td>26.785478</td>
          <td>0.165076</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.264960</td>
          <td>0.013017</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.184631</td>
          <td>22.501172</td>
          <td>23.110429</td>
          <td>20.735052</td>
          <td>21.900551</td>
          <td>18.036413</td>
          <td>26.806062</td>
          <td>0.168001</td>
          <td>23.345503</td>
          <td>0.013872</td>
          <td>24.899616</td>
          <td>0.053419</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214485</td>
          <td>24.124091</td>
          <td>21.180555</td>
          <td>26.541921</td>
          <td>21.508120</td>
          <td>17.003061</td>
          <td>24.585282</td>
          <td>0.023756</td>
          <td>27.873048</td>
          <td>0.624039</td>
          <td>19.636175</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.383474</td>
          <td>16.028738</td>
          <td>23.716013</td>
          <td>22.524019</td>
          <td>21.312426</td>
          <td>28.817778</td>
          <td>22.901795</td>
          <td>0.007044</td>
          <td>26.059167</td>
          <td>0.148140</td>
          <td>24.251610</td>
          <td>0.030016</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.972382</td>
          <td>20.339517</td>
          <td>20.285268</td>
          <td>16.923547</td>
          <td>20.428349</td>
          <td>22.215956</td>
          <td>23.942121</td>
          <td>0.013835</td>
          <td>17.641838</td>
          <td>0.005000</td>
          <td>26.770087</td>
          <td>0.269102</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


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
          <td>22.592771</td>
          <td>25.772394</td>
          <td>25.883902</td>
          <td>20.802655</td>
          <td>26.300639</td>
          <td>22.316392</td>
          <td>22.494532</td>
          <td>0.040700</td>
          <td>26.957504</td>
          <td>1.164660</td>
          <td>25.722971</td>
          <td>0.560969</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.730656</td>
          <td>13.623588</td>
          <td>23.215351</td>
          <td>20.753809</td>
          <td>20.370790</td>
          <td>24.460678</td>
          <td>22.271244</td>
          <td>0.033370</td>
          <td>21.694842</td>
          <td>0.017009</td>
          <td>21.861112</td>
          <td>0.021331</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.393935</td>
          <td>19.270263</td>
          <td>20.946593</td>
          <td>19.992976</td>
          <td>25.894789</td>
          <td>17.006665</td>
          <td>19.113104</td>
          <td>0.005321</td>
          <td>22.103021</td>
          <td>0.024126</td>
          <td>25.257472</td>
          <td>0.396357</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.111998</td>
          <td>23.601598</td>
          <td>21.963500</td>
          <td>21.322763</td>
          <td>23.178983</td>
          <td>20.408842</td>
          <td>24.111520</td>
          <td>0.168784</td>
          <td>21.275295</td>
          <td>0.012147</td>
          <td>22.572998</td>
          <td>0.039927</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.625284</td>
          <td>20.902819</td>
          <td>26.865352</td>
          <td>21.474095</td>
          <td>25.417051</td>
          <td>24.669178</td>
          <td>16.869623</td>
          <td>0.005005</td>
          <td>18.579789</td>
          <td>0.005085</td>
          <td>24.536466</td>
          <td>0.221964</td>
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
          <td>19.120719</td>
          <td>27.497973</td>
          <td>28.513292</td>
          <td>26.425235</td>
          <td>26.723250</td>
          <td>27.506768</td>
          <td>25.521106</td>
          <td>0.521012</td>
          <td>26.530555</td>
          <td>0.903347</td>
          <td>23.468744</td>
          <td>0.088526</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.184631</td>
          <td>22.501172</td>
          <td>23.110429</td>
          <td>20.735052</td>
          <td>21.900551</td>
          <td>18.036413</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.304258</td>
          <td>0.070051</td>
          <td>25.024232</td>
          <td>0.330196</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214485</td>
          <td>24.124091</td>
          <td>21.180555</td>
          <td>26.541921</td>
          <td>21.508120</td>
          <td>17.003061</td>
          <td>25.454285</td>
          <td>0.496028</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.637032</td>
          <td>0.005678</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.383474</td>
          <td>16.028738</td>
          <td>23.716013</td>
          <td>22.524019</td>
          <td>21.312426</td>
          <td>28.817778</td>
          <td>22.826731</td>
          <td>0.054726</td>
          <td>25.145441</td>
          <td>0.335797</td>
          <td>24.202156</td>
          <td>0.167442</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.972382</td>
          <td>20.339517</td>
          <td>20.285268</td>
          <td>16.923547</td>
          <td>20.428349</td>
          <td>22.215956</td>
          <td>23.926113</td>
          <td>0.143984</td>
          <td>17.636772</td>
          <td>0.005015</td>
          <td>inf</td>
          <td>inf</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


