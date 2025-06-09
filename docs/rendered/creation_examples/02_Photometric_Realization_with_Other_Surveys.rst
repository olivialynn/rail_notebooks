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
          <td>23.258566</td>
          <td>21.583472</td>
          <td>21.572643</td>
          <td>25.322508</td>
          <td>24.380030</td>
          <td>22.328409</td>
          <td>23.125329</td>
          <td>24.485853</td>
          <td>27.437665</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.568269</td>
          <td>26.950648</td>
          <td>29.402288</td>
          <td>22.473273</td>
          <td>19.317080</td>
          <td>22.718996</td>
          <td>23.896538</td>
          <td>22.248758</td>
          <td>16.544395</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.387246</td>
          <td>24.217885</td>
          <td>23.989508</td>
          <td>27.964217</td>
          <td>16.325389</td>
          <td>26.071910</td>
          <td>25.434930</td>
          <td>18.974612</td>
          <td>22.123756</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.946973</td>
          <td>20.725889</td>
          <td>19.303841</td>
          <td>23.158930</td>
          <td>20.485943</td>
          <td>26.063838</td>
          <td>22.499038</td>
          <td>24.692887</td>
          <td>21.183341</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.828001</td>
          <td>22.976979</td>
          <td>25.407152</td>
          <td>23.825255</td>
          <td>25.082953</td>
          <td>18.541815</td>
          <td>23.678173</td>
          <td>22.131692</td>
          <td>26.090311</td>
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
          <td>24.587455</td>
          <td>25.168550</td>
          <td>24.573111</td>
          <td>16.043238</td>
          <td>21.933947</td>
          <td>25.412809</td>
          <td>24.517653</td>
          <td>27.665257</td>
          <td>22.590258</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.865701</td>
          <td>27.068942</td>
          <td>26.233117</td>
          <td>26.618801</td>
          <td>19.495769</td>
          <td>20.805249</td>
          <td>22.274911</td>
          <td>24.965852</td>
          <td>22.021825</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.756339</td>
          <td>20.236823</td>
          <td>15.903555</td>
          <td>29.412058</td>
          <td>24.463277</td>
          <td>18.188956</td>
          <td>27.350236</td>
          <td>20.909376</td>
          <td>16.694886</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.781025</td>
          <td>19.132871</td>
          <td>17.614531</td>
          <td>24.592737</td>
          <td>22.743658</td>
          <td>24.745154</td>
          <td>26.509824</td>
          <td>23.996269</td>
          <td>23.819256</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.068990</td>
          <td>23.379125</td>
          <td>25.887052</td>
          <td>22.947889</td>
          <td>21.783223</td>
          <td>21.191779</td>
          <td>23.747203</td>
          <td>20.621473</td>
          <td>26.392623</td>
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
          <td>23.271614</td>
          <td>0.023958</td>
          <td>21.589110</td>
          <td>0.005348</td>
          <td>21.571276</td>
          <td>0.005239</td>
          <td>25.403247</td>
          <td>0.075895</td>
          <td>24.365616</td>
          <td>0.057948</td>
          <td>22.364220</td>
          <td>0.022358</td>
          <td>23.125329</td>
          <td>24.485853</td>
          <td>27.437665</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.754294</td>
          <td>0.206304</td>
          <td>26.878358</td>
          <td>0.191223</td>
          <td>31.392953</td>
          <td>2.676924</td>
          <td>22.485039</td>
          <td>0.007430</td>
          <td>19.311840</td>
          <td>0.005055</td>
          <td>22.725897</td>
          <td>0.030621</td>
          <td>23.896538</td>
          <td>22.248758</td>
          <td>16.544395</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.387254</td>
          <td>0.026436</td>
          <td>24.214836</td>
          <td>0.018793</td>
          <td>24.011946</td>
          <td>0.014075</td>
          <td>28.257367</td>
          <td>0.751941</td>
          <td>16.320522</td>
          <td>0.005002</td>
          <td>25.902073</td>
          <td>0.456582</td>
          <td>25.434930</td>
          <td>18.974612</td>
          <td>22.123756</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.922483</td>
          <td>0.017929</td>
          <td>20.735060</td>
          <td>0.005100</td>
          <td>19.299081</td>
          <td>0.005010</td>
          <td>23.166626</td>
          <td>0.011280</td>
          <td>20.486153</td>
          <td>0.005332</td>
          <td>26.531870</td>
          <td>0.715892</td>
          <td>22.499038</td>
          <td>24.692887</td>
          <td>21.183341</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.830144</td>
          <td>0.008387</td>
          <td>22.970695</td>
          <td>0.007814</td>
          <td>25.314762</td>
          <td>0.042947</td>
          <td>23.830991</td>
          <td>0.019084</td>
          <td>25.042638</td>
          <td>0.105301</td>
          <td>18.541959</td>
          <td>0.005068</td>
          <td>23.678173</td>
          <td>22.131692</td>
          <td>26.090311</td>
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
          <td>24.588697</td>
          <td>0.075566</td>
          <td>25.220947</td>
          <td>0.045009</td>
          <td>24.589423</td>
          <td>0.022726</td>
          <td>16.041814</td>
          <td>0.005001</td>
          <td>21.942056</td>
          <td>0.008217</td>
          <td>25.689924</td>
          <td>0.388356</td>
          <td>24.517653</td>
          <td>27.665257</td>
          <td>22.590258</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.858851</td>
          <td>0.005886</td>
          <td>26.986639</td>
          <td>0.209418</td>
          <td>26.243221</td>
          <td>0.097666</td>
          <td>26.538161</td>
          <td>0.202668</td>
          <td>19.493796</td>
          <td>0.005072</td>
          <td>20.813625</td>
          <td>0.007372</td>
          <td>22.274911</td>
          <td>24.965852</td>
          <td>22.021825</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.759942</td>
          <td>0.036492</td>
          <td>20.230688</td>
          <td>0.005051</td>
          <td>15.896318</td>
          <td>0.005000</td>
          <td>28.354550</td>
          <td>0.801581</td>
          <td>24.409170</td>
          <td>0.060230</td>
          <td>18.188577</td>
          <td>0.005041</td>
          <td>27.350236</td>
          <td>20.909376</td>
          <td>16.694886</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.799528</td>
          <td>0.016246</td>
          <td>19.130714</td>
          <td>0.005014</td>
          <td>17.614292</td>
          <td>0.005002</td>
          <td>24.614455</td>
          <td>0.037708</td>
          <td>22.736323</td>
          <td>0.014204</td>
          <td>24.817771</td>
          <td>0.191366</td>
          <td>26.509824</td>
          <td>23.996269</td>
          <td>23.819256</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.038185</td>
          <td>0.046564</td>
          <td>23.378987</td>
          <td>0.009943</td>
          <td>25.896846</td>
          <td>0.071976</td>
          <td>22.932653</td>
          <td>0.009600</td>
          <td>21.794910</td>
          <td>0.007598</td>
          <td>21.207356</td>
          <td>0.009172</td>
          <td>23.747203</td>
          <td>20.621473</td>
          <td>26.392623</td>
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
          <td>23.258566</td>
          <td>21.583472</td>
          <td>21.572643</td>
          <td>25.322508</td>
          <td>24.380030</td>
          <td>22.328409</td>
          <td>23.119718</td>
          <td>0.007859</td>
          <td>24.472536</td>
          <td>0.036512</td>
          <td>27.331748</td>
          <td>0.419616</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.568269</td>
          <td>26.950648</td>
          <td>29.402288</td>
          <td>22.473273</td>
          <td>19.317080</td>
          <td>22.718996</td>
          <td>23.870052</td>
          <td>0.013069</td>
          <td>22.245983</td>
          <td>0.006872</td>
          <td>16.546000</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.387246</td>
          <td>24.217885</td>
          <td>23.989508</td>
          <td>27.964217</td>
          <td>16.325389</td>
          <td>26.071910</td>
          <td>25.457732</td>
          <td>0.051462</td>
          <td>18.985738</td>
          <td>0.005005</td>
          <td>22.123490</td>
          <td>0.006537</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.946973</td>
          <td>20.725889</td>
          <td>19.303841</td>
          <td>23.158930</td>
          <td>20.485943</td>
          <td>26.063838</td>
          <td>22.495534</td>
          <td>0.006055</td>
          <td>24.710782</td>
          <td>0.045143</td>
          <td>21.188788</td>
          <td>0.005308</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.828001</td>
          <td>22.976979</td>
          <td>25.407152</td>
          <td>23.825255</td>
          <td>25.082953</td>
          <td>18.541815</td>
          <td>23.698796</td>
          <td>0.011467</td>
          <td>22.131911</td>
          <td>0.006558</td>
          <td>26.080020</td>
          <td>0.150820</td>
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
          <td>24.587455</td>
          <td>25.168550</td>
          <td>24.573111</td>
          <td>16.043238</td>
          <td>21.933947</td>
          <td>25.412809</td>
          <td>24.477775</td>
          <td>0.021640</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.582347</td>
          <td>0.008139</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.865701</td>
          <td>27.068942</td>
          <td>26.233117</td>
          <td>26.618801</td>
          <td>19.495769</td>
          <td>20.805249</td>
          <td>22.276713</td>
          <td>0.005726</td>
          <td>24.987949</td>
          <td>0.057793</td>
          <td>22.030162</td>
          <td>0.006319</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.756339</td>
          <td>20.236823</td>
          <td>15.903555</td>
          <td>29.412058</td>
          <td>24.463277</td>
          <td>18.188956</td>
          <td>27.152351</td>
          <td>0.224917</td>
          <td>20.911705</td>
          <td>0.005187</td>
          <td>16.684274</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.781025</td>
          <td>19.132871</td>
          <td>17.614531</td>
          <td>24.592737</td>
          <td>22.743658</td>
          <td>24.745154</td>
          <td>26.574406</td>
          <td>0.137701</td>
          <td>24.004069</td>
          <td>0.024148</td>
          <td>23.825704</td>
          <td>0.020690</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.068990</td>
          <td>23.379125</td>
          <td>25.887052</td>
          <td>22.947889</td>
          <td>21.783223</td>
          <td>21.191779</td>
          <td>23.732898</td>
          <td>0.011763</td>
          <td>20.619489</td>
          <td>0.005110</td>
          <td>26.335026</td>
          <td>0.187442</td>
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
          <td>23.258566</td>
          <td>21.583472</td>
          <td>21.572643</td>
          <td>25.322508</td>
          <td>24.380030</td>
          <td>22.328409</td>
          <td>23.083195</td>
          <td>0.068753</td>
          <td>24.484293</td>
          <td>0.195400</td>
          <td>26.216320</td>
          <td>0.787574</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.568269</td>
          <td>26.950648</td>
          <td>29.402288</td>
          <td>22.473273</td>
          <td>19.317080</td>
          <td>22.718996</td>
          <td>23.809894</td>
          <td>0.130224</td>
          <td>22.199662</td>
          <td>0.026255</td>
          <td>16.546611</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.387246</td>
          <td>24.217885</td>
          <td>23.989508</td>
          <td>27.964217</td>
          <td>16.325389</td>
          <td>26.071910</td>
          <td>25.328186</td>
          <td>0.451469</td>
          <td>18.978867</td>
          <td>0.005176</td>
          <td>22.159982</td>
          <td>0.027685</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.946973</td>
          <td>20.725889</td>
          <td>19.303841</td>
          <td>23.158930</td>
          <td>20.485943</td>
          <td>26.063838</td>
          <td>22.473237</td>
          <td>0.039935</td>
          <td>24.449298</td>
          <td>0.189716</td>
          <td>21.188778</td>
          <td>0.012273</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.828001</td>
          <td>22.976979</td>
          <td>25.407152</td>
          <td>23.825255</td>
          <td>25.082953</td>
          <td>18.541815</td>
          <td>23.877607</td>
          <td>0.138083</td>
          <td>22.128126</td>
          <td>0.024661</td>
          <td>25.407426</td>
          <td>0.444452</td>
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
          <td>24.587455</td>
          <td>25.168550</td>
          <td>24.573111</td>
          <td>16.043238</td>
          <td>21.933947</td>
          <td>25.412809</td>
          <td>24.559260</td>
          <td>0.245730</td>
          <td>25.314840</td>
          <td>0.383496</td>
          <td>22.559720</td>
          <td>0.039458</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.865701</td>
          <td>27.068942</td>
          <td>26.233117</td>
          <td>26.618801</td>
          <td>19.495769</td>
          <td>20.805249</td>
          <td>22.287216</td>
          <td>0.033847</td>
          <td>24.785626</td>
          <td>0.251120</td>
          <td>22.000316</td>
          <td>0.024069</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.756339</td>
          <td>20.236823</td>
          <td>15.903555</td>
          <td>29.412058</td>
          <td>24.463277</td>
          <td>18.188956</td>
          <td>25.052108</td>
          <td>0.365207</td>
          <td>20.901880</td>
          <td>0.009314</td>
          <td>16.692611</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.781025</td>
          <td>19.132871</td>
          <td>17.614531</td>
          <td>24.592737</td>
          <td>22.743658</td>
          <td>24.745154</td>
          <td>26.363499</td>
          <td>0.922077</td>
          <td>23.876866</td>
          <td>0.116000</td>
          <td>23.760568</td>
          <td>0.114362</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.068990</td>
          <td>23.379125</td>
          <td>25.887052</td>
          <td>22.947889</td>
          <td>21.783223</td>
          <td>21.191779</td>
          <td>23.971443</td>
          <td>0.149712</td>
          <td>20.617604</td>
          <td>0.007850</td>
          <td>26.192139</td>
          <td>0.775168</td>
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


