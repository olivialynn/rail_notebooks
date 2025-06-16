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
          <td>19.349305</td>
          <td>24.133362</td>
          <td>25.484255</td>
          <td>23.583603</td>
          <td>26.640888</td>
          <td>21.442011</td>
          <td>23.506549</td>
          <td>26.634157</td>
          <td>18.102586</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.779900</td>
          <td>18.613185</td>
          <td>18.908447</td>
          <td>21.860236</td>
          <td>28.735297</td>
          <td>22.367366</td>
          <td>19.180062</td>
          <td>27.661537</td>
          <td>24.593502</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.918754</td>
          <td>26.726365</td>
          <td>17.364377</td>
          <td>18.714475</td>
          <td>24.404398</td>
          <td>22.015429</td>
          <td>22.844571</td>
          <td>27.497557</td>
          <td>24.076865</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.087907</td>
          <td>22.832302</td>
          <td>23.586844</td>
          <td>26.198173</td>
          <td>23.814370</td>
          <td>26.231318</td>
          <td>17.505952</td>
          <td>21.434603</td>
          <td>22.841146</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.896291</td>
          <td>26.757536</td>
          <td>21.981399</td>
          <td>20.507917</td>
          <td>23.788199</td>
          <td>25.190384</td>
          <td>24.104608</td>
          <td>22.846067</td>
          <td>29.649545</td>
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
          <td>22.011341</td>
          <td>20.015700</td>
          <td>23.244597</td>
          <td>21.214748</td>
          <td>30.303230</td>
          <td>24.995756</td>
          <td>17.427506</td>
          <td>23.414802</td>
          <td>20.628322</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.063201</td>
          <td>22.092008</td>
          <td>21.879521</td>
          <td>21.413124</td>
          <td>13.857834</td>
          <td>26.418490</td>
          <td>22.288077</td>
          <td>21.188242</td>
          <td>22.925549</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786914</td>
          <td>29.721854</td>
          <td>16.731628</td>
          <td>18.774871</td>
          <td>25.803281</td>
          <td>26.052312</td>
          <td>21.602579</td>
          <td>25.672364</td>
          <td>22.807770</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.579075</td>
          <td>16.782243</td>
          <td>23.641109</td>
          <td>18.782108</td>
          <td>22.534197</td>
          <td>23.306634</td>
          <td>15.430848</td>
          <td>23.051619</td>
          <td>23.982208</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.093873</td>
          <td>22.986322</td>
          <td>24.509225</td>
          <td>24.022592</td>
          <td>24.967786</td>
          <td>20.453377</td>
          <td>23.741287</td>
          <td>21.507832</td>
          <td>19.867537</td>
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
          <td>19.349143</td>
          <td>0.005123</td>
          <td>24.102660</td>
          <td>0.017128</td>
          <td>25.535146</td>
          <td>0.052227</td>
          <td>23.551222</td>
          <td>0.015152</td>
          <td>27.299468</td>
          <td>0.645330</td>
          <td>21.445580</td>
          <td>0.010751</td>
          <td>23.506549</td>
          <td>26.634157</td>
          <td>18.102586</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.778725</td>
          <td>0.037095</td>
          <td>18.613320</td>
          <td>0.005008</td>
          <td>18.900783</td>
          <td>0.005006</td>
          <td>21.861782</td>
          <td>0.005926</td>
          <td>27.143896</td>
          <td>0.578379</td>
          <td>22.370682</td>
          <td>0.022482</td>
          <td>19.180062</td>
          <td>27.661537</td>
          <td>24.593502</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.888410</td>
          <td>0.040830</td>
          <td>26.620032</td>
          <td>0.153535</td>
          <td>17.369723</td>
          <td>0.005001</td>
          <td>18.713357</td>
          <td>0.005009</td>
          <td>24.443863</td>
          <td>0.062112</td>
          <td>22.030133</td>
          <td>0.016873</td>
          <td>22.844571</td>
          <td>27.497557</td>
          <td>24.076865</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.102660</td>
          <td>0.049279</td>
          <td>22.821135</td>
          <td>0.007267</td>
          <td>23.590266</td>
          <td>0.010296</td>
          <td>26.228739</td>
          <td>0.155898</td>
          <td>23.771810</td>
          <td>0.034232</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.505952</td>
          <td>21.434603</td>
          <td>22.841146</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.138756</td>
          <td>0.605836</td>
          <td>27.066274</td>
          <td>0.223789</td>
          <td>21.983143</td>
          <td>0.005463</td>
          <td>20.502490</td>
          <td>0.005106</td>
          <td>23.804607</td>
          <td>0.035238</td>
          <td>25.161077</td>
          <td>0.254636</td>
          <td>24.104608</td>
          <td>22.846067</td>
          <td>29.649545</td>
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
          <td>21.997677</td>
          <td>0.009222</td>
          <td>20.014146</td>
          <td>0.005039</td>
          <td>23.239275</td>
          <td>0.008260</td>
          <td>21.209112</td>
          <td>0.005323</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.785786</td>
          <td>0.186269</td>
          <td>17.427506</td>
          <td>23.414802</td>
          <td>20.628322</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.069627</td>
          <td>0.006190</td>
          <td>22.089302</td>
          <td>0.005749</td>
          <td>21.880443</td>
          <td>0.005392</td>
          <td>21.407325</td>
          <td>0.005445</td>
          <td>13.859542</td>
          <td>0.005000</td>
          <td>26.240062</td>
          <td>0.584768</td>
          <td>22.288077</td>
          <td>21.188242</td>
          <td>22.925549</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.787000</td>
          <td>0.008197</td>
          <td>28.590065</td>
          <td>0.714077</td>
          <td>16.732743</td>
          <td>0.005001</td>
          <td>18.771364</td>
          <td>0.005010</td>
          <td>25.550384</td>
          <td>0.163343</td>
          <td>26.220906</td>
          <td>0.576835</td>
          <td>21.602579</td>
          <td>25.672364</td>
          <td>22.807770</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.675107</td>
          <td>0.431302</td>
          <td>16.783653</td>
          <td>0.005001</td>
          <td>23.635362</td>
          <td>0.010623</td>
          <td>18.786569</td>
          <td>0.005010</td>
          <td>22.535540</td>
          <td>0.012169</td>
          <td>23.288871</td>
          <td>0.050389</td>
          <td>15.430848</td>
          <td>23.051619</td>
          <td>23.982208</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.557910</td>
          <td>0.394311</td>
          <td>22.982160</td>
          <td>0.007860</td>
          <td>24.540434</td>
          <td>0.021790</td>
          <td>24.058428</td>
          <td>0.023169</td>
          <td>24.970723</td>
          <td>0.098876</td>
          <td>20.461051</td>
          <td>0.006386</td>
          <td>23.741287</td>
          <td>21.507832</td>
          <td>19.867537</td>
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
          <td>19.349305</td>
          <td>24.133362</td>
          <td>25.484255</td>
          <td>23.583603</td>
          <td>26.640888</td>
          <td>21.442011</td>
          <td>23.491717</td>
          <td>0.009890</td>
          <td>26.735098</td>
          <td>0.261519</td>
          <td>18.097426</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.779900</td>
          <td>18.613185</td>
          <td>18.908447</td>
          <td>21.860236</td>
          <td>28.735297</td>
          <td>22.367366</td>
          <td>19.182750</td>
          <td>0.005003</td>
          <td>28.019628</td>
          <td>0.690563</td>
          <td>24.602753</td>
          <td>0.040999</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.918754</td>
          <td>26.726365</td>
          <td>17.364377</td>
          <td>18.714475</td>
          <td>24.404398</td>
          <td>22.015429</td>
          <td>22.845579</td>
          <td>0.006870</td>
          <td>27.274703</td>
          <td>0.401656</td>
          <td>24.065451</td>
          <td>0.025480</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.087907</td>
          <td>22.832302</td>
          <td>23.586844</td>
          <td>26.198173</td>
          <td>23.814370</td>
          <td>26.231318</td>
          <td>17.500302</td>
          <td>0.005000</td>
          <td>21.424557</td>
          <td>0.005468</td>
          <td>22.837303</td>
          <td>0.009534</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.896291</td>
          <td>26.757536</td>
          <td>21.981399</td>
          <td>20.507917</td>
          <td>23.788199</td>
          <td>25.190384</td>
          <td>24.118526</td>
          <td>0.015964</td>
          <td>22.853259</td>
          <td>0.009636</td>
          <td>28.046527</td>
          <td>0.703304</td>
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
          <td>22.011341</td>
          <td>20.015700</td>
          <td>23.244597</td>
          <td>21.214748</td>
          <td>30.303230</td>
          <td>24.995756</td>
          <td>17.431696</td>
          <td>0.005000</td>
          <td>23.407001</td>
          <td>0.014574</td>
          <td>20.632817</td>
          <td>0.005113</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.063201</td>
          <td>22.092008</td>
          <td>21.879521</td>
          <td>21.413124</td>
          <td>13.857834</td>
          <td>26.418490</td>
          <td>22.279619</td>
          <td>0.005730</td>
          <td>21.192394</td>
          <td>0.005310</td>
          <td>22.914560</td>
          <td>0.010047</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786914</td>
          <td>29.721854</td>
          <td>16.731628</td>
          <td>18.774871</td>
          <td>25.803281</td>
          <td>26.052312</td>
          <td>21.609355</td>
          <td>0.005223</td>
          <td>25.634809</td>
          <td>0.102447</td>
          <td>22.806076</td>
          <td>0.009339</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.579075</td>
          <td>16.782243</td>
          <td>23.641109</td>
          <td>18.782108</td>
          <td>22.534197</td>
          <td>23.306634</td>
          <td>15.434598</td>
          <td>0.005000</td>
          <td>23.051044</td>
          <td>0.011071</td>
          <td>23.999357</td>
          <td>0.024049</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.093873</td>
          <td>22.986322</td>
          <td>24.509225</td>
          <td>24.022592</td>
          <td>24.967786</td>
          <td>20.453377</td>
          <td>23.754200</td>
          <td>0.011954</td>
          <td>21.505596</td>
          <td>0.005540</td>
          <td>19.861677</td>
          <td>0.005027</td>
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
          <td>19.349305</td>
          <td>24.133362</td>
          <td>25.484255</td>
          <td>23.583603</td>
          <td>26.640888</td>
          <td>21.442011</td>
          <td>23.429460</td>
          <td>0.093390</td>
          <td>26.113078</td>
          <td>0.687485</td>
          <td>18.105519</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.779900</td>
          <td>18.613185</td>
          <td>18.908447</td>
          <td>21.860236</td>
          <td>28.735297</td>
          <td>22.367366</td>
          <td>19.175449</td>
          <td>0.005359</td>
          <td>27.253406</td>
          <td>1.368133</td>
          <td>24.400596</td>
          <td>0.198101</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.918754</td>
          <td>26.726365</td>
          <td>17.364377</td>
          <td>18.714475</td>
          <td>24.404398</td>
          <td>22.015429</td>
          <td>22.878184</td>
          <td>0.057293</td>
          <td>27.289594</td>
          <td>1.394180</td>
          <td>24.325439</td>
          <td>0.185929</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.087907</td>
          <td>22.832302</td>
          <td>23.586844</td>
          <td>26.198173</td>
          <td>23.814370</td>
          <td>26.231318</td>
          <td>17.509707</td>
          <td>0.005017</td>
          <td>21.416929</td>
          <td>0.013560</td>
          <td>22.804283</td>
          <td>0.049067</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.896291</td>
          <td>26.757536</td>
          <td>21.981399</td>
          <td>20.507917</td>
          <td>23.788199</td>
          <td>25.190384</td>
          <td>24.088979</td>
          <td>0.165570</td>
          <td>22.855512</td>
          <td>0.046979</td>
          <td>27.969818</td>
          <td>2.008206</td>
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
          <td>22.011341</td>
          <td>20.015700</td>
          <td>23.244597</td>
          <td>21.214748</td>
          <td>30.303230</td>
          <td>24.995756</td>
          <td>17.433790</td>
          <td>0.005015</td>
          <td>23.415375</td>
          <td>0.077304</td>
          <td>20.630584</td>
          <td>0.008371</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.063201</td>
          <td>22.092008</td>
          <td>21.879521</td>
          <td>21.413124</td>
          <td>13.857834</td>
          <td>26.418490</td>
          <td>22.256470</td>
          <td>0.032936</td>
          <td>21.189068</td>
          <td>0.011385</td>
          <td>22.924188</td>
          <td>0.054602</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786914</td>
          <td>29.721854</td>
          <td>16.731628</td>
          <td>18.774871</td>
          <td>25.803281</td>
          <td>26.052312</td>
          <td>21.614515</td>
          <td>0.018814</td>
          <td>25.925010</td>
          <td>0.603301</td>
          <td>22.768710</td>
          <td>0.047536</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.579075</td>
          <td>16.782243</td>
          <td>23.641109</td>
          <td>18.782108</td>
          <td>22.534197</td>
          <td>23.306634</td>
          <td>15.433882</td>
          <td>0.005000</td>
          <td>23.043309</td>
          <td>0.055540</td>
          <td>23.951630</td>
          <td>0.135017</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.093873</td>
          <td>22.986322</td>
          <td>24.509225</td>
          <td>24.022592</td>
          <td>24.967786</td>
          <td>20.453377</td>
          <td>24.070098</td>
          <td>0.162921</td>
          <td>21.536853</td>
          <td>0.014931</td>
          <td>19.871216</td>
          <td>0.006012</td>
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


