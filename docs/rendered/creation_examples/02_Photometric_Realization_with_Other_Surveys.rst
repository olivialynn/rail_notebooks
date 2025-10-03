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
          <td>29.348226</td>
          <td>19.715707</td>
          <td>22.769843</td>
          <td>26.065787</td>
          <td>25.709202</td>
          <td>25.325727</td>
          <td>20.593897</td>
          <td>25.349770</td>
          <td>22.223534</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.168724</td>
          <td>27.808624</td>
          <td>24.215762</td>
          <td>21.481874</td>
          <td>22.875849</td>
          <td>22.904764</td>
          <td>18.306972</td>
          <td>24.141817</td>
          <td>21.204220</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.613400</td>
          <td>18.901499</td>
          <td>18.701504</td>
          <td>24.086406</td>
          <td>25.423317</td>
          <td>24.257070</td>
          <td>23.665962</td>
          <td>22.722045</td>
          <td>24.092778</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.170878</td>
          <td>25.421987</td>
          <td>19.207727</td>
          <td>26.011276</td>
          <td>24.012760</td>
          <td>13.748849</td>
          <td>21.185455</td>
          <td>26.697019</td>
          <td>25.271980</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.239059</td>
          <td>24.594859</td>
          <td>22.376020</td>
          <td>18.754089</td>
          <td>25.224503</td>
          <td>23.307997</td>
          <td>20.525401</td>
          <td>21.676209</td>
          <td>19.880212</td>
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
          <td>21.238580</td>
          <td>22.598038</td>
          <td>24.275604</td>
          <td>24.113012</td>
          <td>24.148368</td>
          <td>21.948178</td>
          <td>28.209958</td>
          <td>19.811238</td>
          <td>21.793866</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.729206</td>
          <td>24.310238</td>
          <td>16.253191</td>
          <td>25.753654</td>
          <td>20.432781</td>
          <td>24.275885</td>
          <td>23.416159</td>
          <td>25.649050</td>
          <td>26.724153</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.445097</td>
          <td>27.118783</td>
          <td>28.147668</td>
          <td>21.256817</td>
          <td>26.340882</td>
          <td>21.796383</td>
          <td>22.047052</td>
          <td>28.325527</td>
          <td>24.067998</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.227476</td>
          <td>13.889387</td>
          <td>23.313255</td>
          <td>26.399239</td>
          <td>19.682913</td>
          <td>24.378966</td>
          <td>25.860910</td>
          <td>22.881533</td>
          <td>20.351473</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.837781</td>
          <td>23.184739</td>
          <td>20.543254</td>
          <td>21.218065</td>
          <td>18.307509</td>
          <td>21.420028</td>
          <td>24.492128</td>
          <td>27.057944</td>
          <td>28.551763</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>19.715250</td>
          <td>0.005027</td>
          <td>22.767086</td>
          <td>0.006613</td>
          <td>25.991256</td>
          <td>0.127051</td>
          <td>26.013134</td>
          <td>0.240943</td>
          <td>25.932038</td>
          <td>0.466958</td>
          <td>20.593897</td>
          <td>25.349770</td>
          <td>22.223534</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.273629</td>
          <td>0.315471</td>
          <td>27.687569</td>
          <td>0.369472</td>
          <td>24.210918</td>
          <td>0.016520</td>
          <td>21.468848</td>
          <td>0.005492</td>
          <td>22.848222</td>
          <td>0.015533</td>
          <td>22.881183</td>
          <td>0.035109</td>
          <td>18.306972</td>
          <td>24.141817</td>
          <td>21.204220</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.800695</td>
          <td>1.621040</td>
          <td>18.899286</td>
          <td>0.005011</td>
          <td>18.709206</td>
          <td>0.005005</td>
          <td>24.043423</td>
          <td>0.022871</td>
          <td>25.398705</td>
          <td>0.143430</td>
          <td>24.227855</td>
          <td>0.115352</td>
          <td>23.665962</td>
          <td>22.722045</td>
          <td>24.092778</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.165401</td>
          <td>0.006360</td>
          <td>25.438600</td>
          <td>0.054577</td>
          <td>19.219295</td>
          <td>0.005009</td>
          <td>26.128548</td>
          <td>0.143049</td>
          <td>24.009006</td>
          <td>0.042226</td>
          <td>13.741921</td>
          <td>0.005000</td>
          <td>21.185455</td>
          <td>26.697019</td>
          <td>25.271980</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.069057</td>
          <td>1.101958</td>
          <td>24.647692</td>
          <td>0.027194</td>
          <td>22.372001</td>
          <td>0.005866</td>
          <td>18.753449</td>
          <td>0.005010</td>
          <td>25.317996</td>
          <td>0.133786</td>
          <td>23.294643</td>
          <td>0.050648</td>
          <td>20.525401</td>
          <td>21.676209</td>
          <td>19.880212</td>
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
          <td>21.243422</td>
          <td>0.006517</td>
          <td>22.595279</td>
          <td>0.006622</td>
          <td>24.243607</td>
          <td>0.016971</td>
          <td>24.118373</td>
          <td>0.024400</td>
          <td>24.069705</td>
          <td>0.044562</td>
          <td>21.947655</td>
          <td>0.015772</td>
          <td>28.209958</td>
          <td>19.811238</td>
          <td>21.793866</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.363972</td>
          <td>0.707826</td>
          <td>24.303112</td>
          <td>0.020238</td>
          <td>16.241432</td>
          <td>0.005000</td>
          <td>25.771114</td>
          <td>0.104888</td>
          <td>20.434375</td>
          <td>0.005306</td>
          <td>24.648462</td>
          <td>0.165775</td>
          <td>23.416159</td>
          <td>25.649050</td>
          <td>26.724153</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.443587</td>
          <td>0.005138</td>
          <td>26.767623</td>
          <td>0.174126</td>
          <td>27.419471</td>
          <td>0.265848</td>
          <td>21.247038</td>
          <td>0.005344</td>
          <td>25.862688</td>
          <td>0.212651</td>
          <td>21.820556</td>
          <td>0.014244</td>
          <td>22.047052</td>
          <td>28.325527</td>
          <td>24.067998</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.227609</td>
          <td>0.005107</td>
          <td>13.888988</td>
          <td>0.005000</td>
          <td>23.328334</td>
          <td>0.008699</td>
          <td>26.235548</td>
          <td>0.156810</td>
          <td>19.687322</td>
          <td>0.005095</td>
          <td>24.524651</td>
          <td>0.149113</td>
          <td>25.860910</td>
          <td>22.881533</td>
          <td>20.351473</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.840052</td>
          <td>0.005069</td>
          <td>23.185839</td>
          <td>0.008806</td>
          <td>20.542318</td>
          <td>0.005050</td>
          <td>21.213091</td>
          <td>0.005325</td>
          <td>18.296267</td>
          <td>0.005014</td>
          <td>21.423890</td>
          <td>0.010589</td>
          <td>24.492128</td>
          <td>27.057944</td>
          <td>28.551763</td>
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
          <td>29.348226</td>
          <td>19.715707</td>
          <td>22.769843</td>
          <td>26.065787</td>
          <td>25.709202</td>
          <td>25.325727</td>
          <td>20.587003</td>
          <td>0.005035</td>
          <td>25.443134</td>
          <td>0.086548</td>
          <td>22.227495</td>
          <td>0.006817</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.168724</td>
          <td>27.808624</td>
          <td>24.215762</td>
          <td>21.481874</td>
          <td>22.875849</td>
          <td>22.904764</td>
          <td>18.312191</td>
          <td>0.005001</td>
          <td>24.161544</td>
          <td>0.027723</td>
          <td>21.207790</td>
          <td>0.005318</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.613400</td>
          <td>18.901499</td>
          <td>18.701504</td>
          <td>24.086406</td>
          <td>25.423317</td>
          <td>24.257070</td>
          <td>23.678735</td>
          <td>0.011298</td>
          <td>22.729144</td>
          <td>0.008889</td>
          <td>24.126169</td>
          <td>0.026874</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.170878</td>
          <td>25.421987</td>
          <td>19.207727</td>
          <td>26.011276</td>
          <td>24.012760</td>
          <td>13.748849</td>
          <td>21.192757</td>
          <td>0.005105</td>
          <td>26.613837</td>
          <td>0.236685</td>
          <td>25.302354</td>
          <td>0.076418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.239059</td>
          <td>24.594859</td>
          <td>22.376020</td>
          <td>18.754089</td>
          <td>25.224503</td>
          <td>23.307997</td>
          <td>20.528990</td>
          <td>0.005031</td>
          <td>21.676380</td>
          <td>0.005726</td>
          <td>19.881304</td>
          <td>0.005028</td>
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
          <td>21.238580</td>
          <td>22.598038</td>
          <td>24.275604</td>
          <td>24.113012</td>
          <td>24.148368</td>
          <td>21.948178</td>
          <td>27.766452</td>
          <td>0.369323</td>
          <td>19.817512</td>
          <td>0.005025</td>
          <td>21.790536</td>
          <td>0.005883</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.729206</td>
          <td>24.310238</td>
          <td>16.253191</td>
          <td>25.753654</td>
          <td>20.432781</td>
          <td>24.275885</td>
          <td>23.424008</td>
          <td>0.009450</td>
          <td>25.780486</td>
          <td>0.116367</td>
          <td>27.146004</td>
          <td>0.363467</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.445097</td>
          <td>27.118783</td>
          <td>28.147668</td>
          <td>21.256817</td>
          <td>26.340882</td>
          <td>21.796383</td>
          <td>22.044152</td>
          <td>0.005484</td>
          <td>27.703734</td>
          <td>0.553249</td>
          <td>24.055506</td>
          <td>0.025259</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.227476</td>
          <td>13.889387</td>
          <td>23.313255</td>
          <td>26.399239</td>
          <td>19.682913</td>
          <td>24.378966</td>
          <td>25.883497</td>
          <td>0.075152</td>
          <td>22.876482</td>
          <td>0.009788</td>
          <td>20.358586</td>
          <td>0.005068</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.837781</td>
          <td>23.184739</td>
          <td>20.543254</td>
          <td>21.218065</td>
          <td>18.307509</td>
          <td>21.420028</td>
          <td>24.518982</td>
          <td>0.022426</td>
          <td>26.795637</td>
          <td>0.274761</td>
          <td>28.830584</td>
          <td>1.147025</td>
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
          <td>29.348226</td>
          <td>19.715707</td>
          <td>22.769843</td>
          <td>26.065787</td>
          <td>25.709202</td>
          <td>25.325727</td>
          <td>20.595154</td>
          <td>0.008703</td>
          <td>25.254325</td>
          <td>0.365841</td>
          <td>22.184552</td>
          <td>0.028291</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.168724</td>
          <td>27.808624</td>
          <td>24.215762</td>
          <td>21.481874</td>
          <td>22.875849</td>
          <td>22.904764</td>
          <td>18.307022</td>
          <td>0.005075</td>
          <td>24.292062</td>
          <td>0.166006</td>
          <td>21.188488</td>
          <td>0.012270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.613400</td>
          <td>18.901499</td>
          <td>18.701504</td>
          <td>24.086406</td>
          <td>25.423317</td>
          <td>24.257070</td>
          <td>23.803664</td>
          <td>0.129523</td>
          <td>22.729141</td>
          <td>0.041975</td>
          <td>24.202165</td>
          <td>0.167443</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.170878</td>
          <td>25.421987</td>
          <td>19.207727</td>
          <td>26.011276</td>
          <td>24.012760</td>
          <td>13.748849</td>
          <td>21.186436</td>
          <td>0.013238</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.314056</td>
          <td>0.413976</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.239059</td>
          <td>24.594859</td>
          <td>22.376020</td>
          <td>18.754089</td>
          <td>25.224503</td>
          <td>23.307997</td>
          <td>20.537853</td>
          <td>0.008407</td>
          <td>21.715297</td>
          <td>0.017303</td>
          <td>19.881251</td>
          <td>0.006030</td>
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
          <td>21.238580</td>
          <td>22.598038</td>
          <td>24.275604</td>
          <td>24.113012</td>
          <td>24.148368</td>
          <td>21.948178</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.808881</td>
          <td>0.005768</td>
          <td>21.804607</td>
          <td>0.020319</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.729206</td>
          <td>24.310238</td>
          <td>16.253191</td>
          <td>25.753654</td>
          <td>20.432781</td>
          <td>24.275885</td>
          <td>23.396191</td>
          <td>0.090693</td>
          <td>26.422976</td>
          <td>0.843899</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.445097</td>
          <td>27.118783</td>
          <td>28.147668</td>
          <td>21.256817</td>
          <td>26.340882</td>
          <td>21.796383</td>
          <td>22.035281</td>
          <td>0.027090</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.141621</td>
          <td>0.159001</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.227476</td>
          <td>13.889387</td>
          <td>23.313255</td>
          <td>26.399239</td>
          <td>19.682913</td>
          <td>24.378966</td>
          <td>25.552416</td>
          <td>0.533051</td>
          <td>22.952966</td>
          <td>0.051244</td>
          <td>20.344054</td>
          <td>0.007184</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.837781</td>
          <td>23.184739</td>
          <td>20.543254</td>
          <td>21.218065</td>
          <td>18.307509</td>
          <td>21.420028</td>
          <td>24.541543</td>
          <td>0.242166</td>
          <td>25.432523</td>
          <td>0.419864</td>
          <td>27.433331</td>
          <td>1.575716</td>
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


