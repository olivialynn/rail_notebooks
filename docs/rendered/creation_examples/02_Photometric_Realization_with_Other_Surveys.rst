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
          <td>22.360375</td>
          <td>24.634850</td>
          <td>22.585922</td>
          <td>21.973600</td>
          <td>22.244300</td>
          <td>24.743189</td>
          <td>23.937801</td>
          <td>24.920946</td>
          <td>20.916154</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.001796</td>
          <td>20.871351</td>
          <td>24.615336</td>
          <td>18.581374</td>
          <td>19.556219</td>
          <td>20.793641</td>
          <td>30.835829</td>
          <td>24.272620</td>
          <td>18.775558</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.277253</td>
          <td>23.568013</td>
          <td>21.140933</td>
          <td>26.909487</td>
          <td>19.857168</td>
          <td>24.269164</td>
          <td>21.260296</td>
          <td>26.179694</td>
          <td>20.284969</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.196251</td>
          <td>24.165385</td>
          <td>25.306405</td>
          <td>25.384536</td>
          <td>24.187545</td>
          <td>22.042459</td>
          <td>21.225563</td>
          <td>23.128945</td>
          <td>22.855406</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.911562</td>
          <td>26.712207</td>
          <td>20.459665</td>
          <td>22.022166</td>
          <td>20.142843</td>
          <td>25.670229</td>
          <td>22.388872</td>
          <td>25.778561</td>
          <td>25.666082</td>
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
          <td>25.494155</td>
          <td>22.575062</td>
          <td>16.890450</td>
          <td>22.589098</td>
          <td>25.909471</td>
          <td>19.898503</td>
          <td>21.937495</td>
          <td>22.148449</td>
          <td>21.643015</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.302353</td>
          <td>22.808175</td>
          <td>17.112567</td>
          <td>19.448939</td>
          <td>16.064571</td>
          <td>23.671965</td>
          <td>20.254174</td>
          <td>21.169679</td>
          <td>20.490372</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.161406</td>
          <td>20.452922</td>
          <td>26.291010</td>
          <td>25.420140</td>
          <td>23.624007</td>
          <td>21.144990</td>
          <td>18.610947</td>
          <td>20.695000</td>
          <td>22.736321</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.002004</td>
          <td>21.319200</td>
          <td>19.571222</td>
          <td>25.737369</td>
          <td>33.542508</td>
          <td>25.241909</td>
          <td>27.803897</td>
          <td>17.898294</td>
          <td>20.525620</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.333882</td>
          <td>23.101568</td>
          <td>25.454699</td>
          <td>28.111862</td>
          <td>13.949349</td>
          <td>23.184587</td>
          <td>21.423110</td>
          <td>21.332495</td>
          <td>18.488938</td>
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
          <td>22.351778</td>
          <td>0.011609</td>
          <td>24.650269</td>
          <td>0.027255</td>
          <td>22.598909</td>
          <td>0.006241</td>
          <td>21.983618</td>
          <td>0.006124</td>
          <td>22.260621</td>
          <td>0.010006</td>
          <td>24.496433</td>
          <td>0.145541</td>
          <td>23.937801</td>
          <td>24.920946</td>
          <td>20.916154</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.971760</td>
          <td>0.043926</td>
          <td>20.870980</td>
          <td>0.005121</td>
          <td>24.599360</td>
          <td>0.022921</td>
          <td>18.581104</td>
          <td>0.005008</td>
          <td>19.558641</td>
          <td>0.005079</td>
          <td>20.790086</td>
          <td>0.007290</td>
          <td>30.835829</td>
          <td>24.272620</td>
          <td>18.775558</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.262233</td>
          <td>0.023769</td>
          <td>23.575952</td>
          <td>0.011390</td>
          <td>21.144781</td>
          <td>0.005122</td>
          <td>27.251569</td>
          <td>0.361948</td>
          <td>19.855454</td>
          <td>0.005123</td>
          <td>24.186226</td>
          <td>0.111242</td>
          <td>21.260296</td>
          <td>26.179694</td>
          <td>20.284969</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.233933</td>
          <td>0.305619</td>
          <td>24.179745</td>
          <td>0.018253</td>
          <td>25.218381</td>
          <td>0.039430</td>
          <td>25.287662</td>
          <td>0.068518</td>
          <td>24.113114</td>
          <td>0.046312</td>
          <td>22.049452</td>
          <td>0.017144</td>
          <td>21.225563</td>
          <td>23.128945</td>
          <td>22.855406</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.939568</td>
          <td>0.042702</td>
          <td>26.823303</td>
          <td>0.182538</td>
          <td>20.460314</td>
          <td>0.005044</td>
          <td>22.032216</td>
          <td>0.006213</td>
          <td>20.142217</td>
          <td>0.005192</td>
          <td>25.388258</td>
          <td>0.306167</td>
          <td>22.388872</td>
          <td>25.778561</td>
          <td>25.666082</td>
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
          <td>25.605483</td>
          <td>0.182056</td>
          <td>22.578701</td>
          <td>0.006582</td>
          <td>16.884612</td>
          <td>0.005001</td>
          <td>22.577554</td>
          <td>0.007785</td>
          <td>25.973227</td>
          <td>0.233126</td>
          <td>19.908715</td>
          <td>0.005577</td>
          <td>21.937495</td>
          <td>22.148449</td>
          <td>21.643015</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.783091</td>
          <td>0.007144</td>
          <td>17.108969</td>
          <td>0.005001</td>
          <td>19.442589</td>
          <td>0.005023</td>
          <td>16.066620</td>
          <td>0.005001</td>
          <td>23.853039</td>
          <td>0.083050</td>
          <td>20.254174</td>
          <td>21.169679</td>
          <td>20.490372</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.166723</td>
          <td>0.052135</td>
          <td>20.451968</td>
          <td>0.005068</td>
          <td>26.293133</td>
          <td>0.102032</td>
          <td>25.548223</td>
          <td>0.086250</td>
          <td>23.615150</td>
          <td>0.029824</td>
          <td>21.149530</td>
          <td>0.008850</td>
          <td>18.610947</td>
          <td>20.695000</td>
          <td>22.736321</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.529252</td>
          <td>0.385677</td>
          <td>21.319036</td>
          <td>0.005232</td>
          <td>19.581949</td>
          <td>0.005014</td>
          <td>25.508248</td>
          <td>0.083265</td>
          <td>26.987826</td>
          <td>0.516640</td>
          <td>25.348649</td>
          <td>0.296576</td>
          <td>27.803897</td>
          <td>17.898294</td>
          <td>20.525620</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.355816</td>
          <td>0.025736</td>
          <td>23.101900</td>
          <td>0.008387</td>
          <td>25.365141</td>
          <td>0.044910</td>
          <td>27.853313</td>
          <td>0.568761</td>
          <td>13.953609</td>
          <td>0.005000</td>
          <td>23.192605</td>
          <td>0.046262</td>
          <td>21.423110</td>
          <td>21.332495</td>
          <td>18.488938</td>
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
          <td>22.360375</td>
          <td>24.634850</td>
          <td>22.585922</td>
          <td>21.973600</td>
          <td>22.244300</td>
          <td>24.743189</td>
          <td>23.922153</td>
          <td>0.013617</td>
          <td>24.891971</td>
          <td>0.053057</td>
          <td>20.915848</td>
          <td>0.005188</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.001796</td>
          <td>20.871351</td>
          <td>24.615336</td>
          <td>18.581374</td>
          <td>19.556219</td>
          <td>20.793641</td>
          <td>32.870780</td>
          <td>4.147430</td>
          <td>24.295201</td>
          <td>0.031196</td>
          <td>18.765958</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.277253</td>
          <td>23.568013</td>
          <td>21.140933</td>
          <td>26.909487</td>
          <td>19.857168</td>
          <td>24.269164</td>
          <td>21.258318</td>
          <td>0.005118</td>
          <td>26.102359</td>
          <td>0.153740</td>
          <td>20.284971</td>
          <td>0.005060</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.196251</td>
          <td>24.165385</td>
          <td>25.306405</td>
          <td>25.384536</td>
          <td>24.187545</td>
          <td>22.042459</td>
          <td>21.213811</td>
          <td>0.005109</td>
          <td>23.125817</td>
          <td>0.011701</td>
          <td>22.837346</td>
          <td>0.009534</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.911562</td>
          <td>26.712207</td>
          <td>20.459665</td>
          <td>22.022166</td>
          <td>20.142843</td>
          <td>25.670229</td>
          <td>22.391759</td>
          <td>0.005885</td>
          <td>25.751467</td>
          <td>0.113458</td>
          <td>25.886854</td>
          <td>0.127648</td>
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
          <td>25.494155</td>
          <td>22.575062</td>
          <td>16.890450</td>
          <td>22.589098</td>
          <td>25.909471</td>
          <td>19.898503</td>
          <td>21.936652</td>
          <td>0.005401</td>
          <td>22.150216</td>
          <td>0.006605</td>
          <td>21.632813</td>
          <td>0.005673</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.302353</td>
          <td>22.808175</td>
          <td>17.112567</td>
          <td>19.448939</td>
          <td>16.064571</td>
          <td>23.671965</td>
          <td>20.256826</td>
          <td>0.005019</td>
          <td>21.166382</td>
          <td>0.005296</td>
          <td>20.493179</td>
          <td>0.005087</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.161406</td>
          <td>20.452922</td>
          <td>26.291010</td>
          <td>25.420140</td>
          <td>23.624007</td>
          <td>21.144990</td>
          <td>18.615561</td>
          <td>0.005001</td>
          <td>20.697645</td>
          <td>0.005127</td>
          <td>22.747865</td>
          <td>0.008995</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.002004</td>
          <td>21.319200</td>
          <td>19.571222</td>
          <td>25.737369</td>
          <td>33.542508</td>
          <td>25.241909</td>
          <td>27.751541</td>
          <td>0.365045</td>
          <td>17.895953</td>
          <td>0.005001</td>
          <td>20.509105</td>
          <td>0.005090</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.333882</td>
          <td>23.101568</td>
          <td>25.454699</td>
          <td>28.111862</td>
          <td>13.949349</td>
          <td>23.184587</td>
          <td>21.423463</td>
          <td>0.005159</td>
          <td>21.341004</td>
          <td>0.005404</td>
          <td>18.485788</td>
          <td>0.005002</td>
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
          <td>22.360375</td>
          <td>24.634850</td>
          <td>22.585922</td>
          <td>21.973600</td>
          <td>22.244300</td>
          <td>24.743189</td>
          <td>23.869755</td>
          <td>0.137149</td>
          <td>24.918634</td>
          <td>0.279944</td>
          <td>20.906483</td>
          <td>0.009991</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.001796</td>
          <td>20.871351</td>
          <td>24.615336</td>
          <td>18.581374</td>
          <td>19.556219</td>
          <td>20.793641</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.524527</td>
          <td>0.202126</td>
          <td>18.777905</td>
          <td>0.005147</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.277253</td>
          <td>23.568013</td>
          <td>21.140933</td>
          <td>26.909487</td>
          <td>19.857168</td>
          <td>24.269164</td>
          <td>21.276337</td>
          <td>0.014218</td>
          <td>26.239056</td>
          <td>0.748404</td>
          <td>20.292747</td>
          <td>0.007015</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.196251</td>
          <td>24.165385</td>
          <td>25.306405</td>
          <td>25.384536</td>
          <td>24.187545</td>
          <td>22.042459</td>
          <td>21.199854</td>
          <td>0.013379</td>
          <td>23.047056</td>
          <td>0.055726</td>
          <td>22.873927</td>
          <td>0.052210</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.911562</td>
          <td>26.712207</td>
          <td>20.459665</td>
          <td>22.022166</td>
          <td>20.142843</td>
          <td>25.670229</td>
          <td>22.371118</td>
          <td>0.036466</td>
          <td>25.574676</td>
          <td>0.467508</td>
          <td>25.310977</td>
          <td>0.413001</td>
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
          <td>25.494155</td>
          <td>22.575062</td>
          <td>16.890450</td>
          <td>22.589098</td>
          <td>25.909471</td>
          <td>19.898503</td>
          <td>21.932535</td>
          <td>0.024756</td>
          <td>22.182690</td>
          <td>0.025867</td>
          <td>21.635657</td>
          <td>0.017601</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.302353</td>
          <td>22.808175</td>
          <td>17.112567</td>
          <td>19.448939</td>
          <td>16.064571</td>
          <td>23.671965</td>
          <td>20.257923</td>
          <td>0.007232</td>
          <td>21.187170</td>
          <td>0.011369</td>
          <td>20.499416</td>
          <td>0.007773</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.161406</td>
          <td>20.452922</td>
          <td>26.291010</td>
          <td>25.420140</td>
          <td>23.624007</td>
          <td>21.144990</td>
          <td>18.609420</td>
          <td>0.005130</td>
          <td>20.684174</td>
          <td>0.008148</td>
          <td>22.791319</td>
          <td>0.048503</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.002004</td>
          <td>21.319200</td>
          <td>19.571222</td>
          <td>25.737369</td>
          <td>33.542508</td>
          <td>25.241909</td>
          <td>25.737885</td>
          <td>0.608808</td>
          <td>17.890518</td>
          <td>0.005024</td>
          <td>20.525798</td>
          <td>0.007885</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.333882</td>
          <td>23.101568</td>
          <td>25.454699</td>
          <td>28.111862</td>
          <td>13.949349</td>
          <td>23.184587</td>
          <td>21.416705</td>
          <td>0.015940</td>
          <td>21.355897</td>
          <td>0.012925</td>
          <td>18.487499</td>
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


