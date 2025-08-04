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
          <td>19.520856</td>
          <td>23.870198</td>
          <td>23.153531</td>
          <td>21.982037</td>
          <td>21.023270</td>
          <td>22.343271</td>
          <td>19.082718</td>
          <td>20.658203</td>
          <td>20.217255</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.622481</td>
          <td>28.664806</td>
          <td>26.696519</td>
          <td>21.890495</td>
          <td>23.227258</td>
          <td>21.514604</td>
          <td>22.502947</td>
          <td>22.964827</td>
          <td>21.781145</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.384168</td>
          <td>20.932438</td>
          <td>20.596590</td>
          <td>28.803214</td>
          <td>26.240173</td>
          <td>25.929943</td>
          <td>21.671622</td>
          <td>28.297746</td>
          <td>23.001902</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.237910</td>
          <td>22.381653</td>
          <td>21.650512</td>
          <td>22.236811</td>
          <td>18.579598</td>
          <td>18.501996</td>
          <td>23.896197</td>
          <td>29.557738</td>
          <td>24.811445</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.310974</td>
          <td>26.164746</td>
          <td>26.845314</td>
          <td>23.884259</td>
          <td>25.005756</td>
          <td>20.155505</td>
          <td>26.268829</td>
          <td>24.636904</td>
          <td>19.785709</td>
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
          <td>24.186577</td>
          <td>22.509752</td>
          <td>20.529599</td>
          <td>21.289712</td>
          <td>25.645573</td>
          <td>22.135487</td>
          <td>26.079081</td>
          <td>23.359520</td>
          <td>22.076702</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.412055</td>
          <td>20.393825</td>
          <td>25.584938</td>
          <td>23.205361</td>
          <td>24.094070</td>
          <td>24.978251</td>
          <td>28.291263</td>
          <td>21.279134</td>
          <td>25.651260</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.169910</td>
          <td>22.961484</td>
          <td>23.347618</td>
          <td>20.521507</td>
          <td>22.442633</td>
          <td>24.644523</td>
          <td>24.190172</td>
          <td>21.935634</td>
          <td>24.779416</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.881872</td>
          <td>24.137014</td>
          <td>26.320081</td>
          <td>21.092875</td>
          <td>21.576399</td>
          <td>23.573959</td>
          <td>19.067294</td>
          <td>24.554645</td>
          <td>26.497314</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.477718</td>
          <td>26.242487</td>
          <td>25.320634</td>
          <td>24.700291</td>
          <td>25.941257</td>
          <td>23.107270</td>
          <td>29.688709</td>
          <td>20.097340</td>
          <td>21.103330</td>
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
          <td>19.508828</td>
          <td>0.005149</td>
          <td>23.852449</td>
          <td>0.014016</td>
          <td>23.152455</td>
          <td>0.007876</td>
          <td>21.976446</td>
          <td>0.006111</td>
          <td>21.020925</td>
          <td>0.005785</td>
          <td>22.316427</td>
          <td>0.021461</td>
          <td>19.082718</td>
          <td>20.658203</td>
          <td>20.217255</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.621367</td>
          <td>0.077765</td>
          <td>29.068826</td>
          <td>0.971019</td>
          <td>26.498948</td>
          <td>0.122093</td>
          <td>21.885978</td>
          <td>0.005962</td>
          <td>23.225447</td>
          <td>0.021273</td>
          <td>21.510883</td>
          <td>0.011264</td>
          <td>22.502947</td>
          <td>22.964827</td>
          <td>21.781145</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.385529</td>
          <td>0.063205</td>
          <td>20.932375</td>
          <td>0.005132</td>
          <td>20.598572</td>
          <td>0.005054</td>
          <td>28.187552</td>
          <td>0.717622</td>
          <td>25.863167</td>
          <td>0.212736</td>
          <td>25.670983</td>
          <td>0.382698</td>
          <td>21.671622</td>
          <td>28.297746</td>
          <td>23.001902</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.248546</td>
          <td>0.005384</td>
          <td>22.385211</td>
          <td>0.006180</td>
          <td>21.648546</td>
          <td>0.005270</td>
          <td>22.232029</td>
          <td>0.006657</td>
          <td>18.587512</td>
          <td>0.005020</td>
          <td>18.502316</td>
          <td>0.005065</td>
          <td>23.896197</td>
          <td>29.557738</td>
          <td>24.811445</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.309590</td>
          <td>0.005118</td>
          <td>26.368641</td>
          <td>0.123625</td>
          <td>26.895013</td>
          <td>0.171635</td>
          <td>23.884363</td>
          <td>0.019965</td>
          <td>25.082380</td>
          <td>0.109021</td>
          <td>20.156635</td>
          <td>0.005858</td>
          <td>26.268829</td>
          <td>24.636904</td>
          <td>19.785709</td>
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
          <td>24.234383</td>
          <td>0.055333</td>
          <td>22.515734</td>
          <td>0.006439</td>
          <td>20.529522</td>
          <td>0.005049</td>
          <td>21.286758</td>
          <td>0.005366</td>
          <td>25.558042</td>
          <td>0.164413</td>
          <td>22.116746</td>
          <td>0.018129</td>
          <td>26.079081</td>
          <td>23.359520</td>
          <td>22.076702</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.397661</td>
          <td>0.026673</td>
          <td>20.399415</td>
          <td>0.005063</td>
          <td>25.588046</td>
          <td>0.054738</td>
          <td>23.216479</td>
          <td>0.011698</td>
          <td>24.079940</td>
          <td>0.044968</td>
          <td>25.067934</td>
          <td>0.235834</td>
          <td>28.291263</td>
          <td>21.279134</td>
          <td>25.651260</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.215939</td>
          <td>0.054442</td>
          <td>22.967551</td>
          <td>0.007801</td>
          <td>23.350360</td>
          <td>0.008815</td>
          <td>20.519136</td>
          <td>0.005109</td>
          <td>22.433577</td>
          <td>0.011290</td>
          <td>24.767878</td>
          <td>0.183470</td>
          <td>24.190172</td>
          <td>21.935634</td>
          <td>24.779416</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.045236</td>
          <td>0.262367</td>
          <td>24.165770</td>
          <td>0.018042</td>
          <td>26.325535</td>
          <td>0.104966</td>
          <td>21.100636</td>
          <td>0.005272</td>
          <td>21.565113</td>
          <td>0.006842</td>
          <td>23.490147</td>
          <td>0.060244</td>
          <td>19.067294</td>
          <td>24.554645</td>
          <td>26.497314</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.683995</td>
          <td>0.194502</td>
          <td>26.084910</td>
          <td>0.096532</td>
          <td>25.302313</td>
          <td>0.042475</td>
          <td>24.752771</td>
          <td>0.042625</td>
          <td>26.319457</td>
          <td>0.309114</td>
          <td>23.171415</td>
          <td>0.045400</td>
          <td>29.688709</td>
          <td>20.097340</td>
          <td>21.103330</td>
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
          <td>19.520856</td>
          <td>23.870198</td>
          <td>23.153531</td>
          <td>21.982037</td>
          <td>21.023270</td>
          <td>22.343271</td>
          <td>19.080978</td>
          <td>0.005002</td>
          <td>20.653122</td>
          <td>0.005117</td>
          <td>20.211960</td>
          <td>0.005052</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.622481</td>
          <td>28.664806</td>
          <td>26.696519</td>
          <td>21.890495</td>
          <td>23.227258</td>
          <td>21.514604</td>
          <td>22.510105</td>
          <td>0.006081</td>
          <td>22.973641</td>
          <td>0.010471</td>
          <td>21.779312</td>
          <td>0.005866</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.384168</td>
          <td>20.932438</td>
          <td>20.596590</td>
          <td>28.803214</td>
          <td>26.240173</td>
          <td>25.929943</td>
          <td>21.673431</td>
          <td>0.005250</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.000072</td>
          <td>0.010670</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.237910</td>
          <td>22.381653</td>
          <td>21.650512</td>
          <td>22.236811</td>
          <td>18.579598</td>
          <td>18.501996</td>
          <td>23.882339</td>
          <td>0.013196</td>
          <td>27.479804</td>
          <td>0.469305</td>
          <td>24.759419</td>
          <td>0.047143</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.310974</td>
          <td>26.164746</td>
          <td>26.845314</td>
          <td>23.884259</td>
          <td>25.005756</td>
          <td>20.155505</td>
          <td>26.321298</td>
          <td>0.110507</td>
          <td>24.585320</td>
          <td>0.040368</td>
          <td>19.782230</td>
          <td>0.005024</td>
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
          <td>24.186577</td>
          <td>22.509752</td>
          <td>20.529599</td>
          <td>21.289712</td>
          <td>25.645573</td>
          <td>22.135487</td>
          <td>26.138903</td>
          <td>0.094169</td>
          <td>23.385971</td>
          <td>0.014329</td>
          <td>22.074932</td>
          <td>0.006420</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.412055</td>
          <td>20.393825</td>
          <td>25.584938</td>
          <td>23.205361</td>
          <td>24.094070</td>
          <td>24.978251</td>
          <td>28.050645</td>
          <td>0.459161</td>
          <td>21.279233</td>
          <td>0.005362</td>
          <td>25.677794</td>
          <td>0.106380</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.169910</td>
          <td>22.961484</td>
          <td>23.347618</td>
          <td>20.521507</td>
          <td>22.442633</td>
          <td>24.644523</td>
          <td>24.188949</td>
          <td>0.016926</td>
          <td>21.946431</td>
          <td>0.006148</td>
          <td>24.838235</td>
          <td>0.050575</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.881872</td>
          <td>24.137014</td>
          <td>26.320081</td>
          <td>21.092875</td>
          <td>21.576399</td>
          <td>23.573959</td>
          <td>19.065771</td>
          <td>0.005002</td>
          <td>24.566679</td>
          <td>0.039703</td>
          <td>26.550554</td>
          <td>0.224581</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.477718</td>
          <td>26.242487</td>
          <td>25.320634</td>
          <td>24.700291</td>
          <td>25.941257</td>
          <td>23.107270</td>
          <td>28.628923</td>
          <td>0.694947</td>
          <td>20.093329</td>
          <td>0.005042</td>
          <td>21.098681</td>
          <td>0.005262</td>
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
          <td>19.520856</td>
          <td>23.870198</td>
          <td>23.153531</td>
          <td>21.982037</td>
          <td>21.023270</td>
          <td>22.343271</td>
          <td>19.078686</td>
          <td>0.005302</td>
          <td>20.657728</td>
          <td>0.008026</td>
          <td>20.219394</td>
          <td>0.006794</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.622481</td>
          <td>28.664806</td>
          <td>26.696519</td>
          <td>21.890495</td>
          <td>23.227258</td>
          <td>21.514604</td>
          <td>22.529295</td>
          <td>0.041980</td>
          <td>22.918666</td>
          <td>0.049701</td>
          <td>21.777347</td>
          <td>0.019850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.384168</td>
          <td>20.932438</td>
          <td>20.596590</td>
          <td>28.803214</td>
          <td>26.240173</td>
          <td>25.929943</td>
          <td>21.674850</td>
          <td>0.019807</td>
          <td>26.438188</td>
          <td>0.852145</td>
          <td>23.135839</td>
          <td>0.065921</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.237910</td>
          <td>22.381653</td>
          <td>21.650512</td>
          <td>22.236811</td>
          <td>18.579598</td>
          <td>18.501996</td>
          <td>23.685950</td>
          <td>0.116923</td>
          <td>26.430110</td>
          <td>0.847760</td>
          <td>24.414635</td>
          <td>0.200453</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.310974</td>
          <td>26.164746</td>
          <td>26.845314</td>
          <td>23.884259</td>
          <td>25.005756</td>
          <td>20.155505</td>
          <td>25.856768</td>
          <td>0.661433</td>
          <td>24.547602</td>
          <td>0.206079</td>
          <td>19.789863</td>
          <td>0.005882</td>
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
          <td>24.186577</td>
          <td>22.509752</td>
          <td>20.529599</td>
          <td>21.289712</td>
          <td>25.645573</td>
          <td>22.135487</td>
          <td>25.060341</td>
          <td>0.367565</td>
          <td>23.422119</td>
          <td>0.077767</td>
          <td>22.068669</td>
          <td>0.025552</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.412055</td>
          <td>20.393825</td>
          <td>25.584938</td>
          <td>23.205361</td>
          <td>24.094070</td>
          <td>24.978251</td>
          <td>26.129974</td>
          <td>0.794639</td>
          <td>21.293666</td>
          <td>0.012319</td>
          <td>25.362160</td>
          <td>0.429457</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.169910</td>
          <td>22.961484</td>
          <td>23.347618</td>
          <td>20.521507</td>
          <td>22.442633</td>
          <td>24.644523</td>
          <td>24.196095</td>
          <td>0.181365</td>
          <td>21.923803</td>
          <td>0.020656</td>
          <td>24.442198</td>
          <td>0.205147</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.881872</td>
          <td>24.137014</td>
          <td>26.320081</td>
          <td>21.092875</td>
          <td>21.576399</td>
          <td>23.573959</td>
          <td>19.068431</td>
          <td>0.005297</td>
          <td>24.694627</td>
          <td>0.232950</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.477718</td>
          <td>26.242487</td>
          <td>25.320634</td>
          <td>24.700291</td>
          <td>25.941257</td>
          <td>23.107270</td>
          <td>26.104840</td>
          <td>0.781667</td>
          <td>20.093894</td>
          <td>0.006243</td>
          <td>21.109795</td>
          <td>0.011562</td>
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


