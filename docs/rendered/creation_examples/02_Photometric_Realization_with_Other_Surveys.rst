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
          <td>19.905756</td>
          <td>22.595966</td>
          <td>21.079622</td>
          <td>21.224496</td>
          <td>23.944839</td>
          <td>24.717981</td>
          <td>25.336248</td>
          <td>27.526966</td>
          <td>26.822995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.214478</td>
          <td>22.059164</td>
          <td>22.053685</td>
          <td>18.780520</td>
          <td>26.628718</td>
          <td>26.580842</td>
          <td>23.563985</td>
          <td>21.540648</td>
          <td>25.356531</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.002177</td>
          <td>22.840270</td>
          <td>21.952955</td>
          <td>26.448755</td>
          <td>27.632442</td>
          <td>26.751562</td>
          <td>18.138293</td>
          <td>17.843495</td>
          <td>27.354061</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.837799</td>
          <td>25.188032</td>
          <td>21.421728</td>
          <td>20.718911</td>
          <td>26.386796</td>
          <td>28.631238</td>
          <td>21.750971</td>
          <td>19.813719</td>
          <td>19.674127</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.026782</td>
          <td>27.537906</td>
          <td>19.816694</td>
          <td>24.154672</td>
          <td>22.435838</td>
          <td>17.646177</td>
          <td>19.911878</td>
          <td>25.214031</td>
          <td>18.788371</td>
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
          <td>21.423785</td>
          <td>26.562500</td>
          <td>22.991418</td>
          <td>19.902476</td>
          <td>21.429431</td>
          <td>22.724964</td>
          <td>21.965784</td>
          <td>21.623891</td>
          <td>27.297555</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.154670</td>
          <td>22.647636</td>
          <td>22.453978</td>
          <td>25.803136</td>
          <td>22.204539</td>
          <td>24.190869</td>
          <td>21.789051</td>
          <td>24.023200</td>
          <td>18.088282</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.347147</td>
          <td>23.640986</td>
          <td>25.741203</td>
          <td>22.729064</td>
          <td>20.485232</td>
          <td>23.713040</td>
          <td>26.274973</td>
          <td>21.877116</td>
          <td>23.125813</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.048422</td>
          <td>23.761783</td>
          <td>25.110742</td>
          <td>23.252867</td>
          <td>24.276911</td>
          <td>26.748183</td>
          <td>29.652060</td>
          <td>22.794092</td>
          <td>19.300932</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.588004</td>
          <td>22.921991</td>
          <td>20.905526</td>
          <td>20.707702</td>
          <td>23.551973</td>
          <td>23.965402</td>
          <td>20.786495</td>
          <td>24.480347</td>
          <td>22.746686</td>
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
          <td>19.911847</td>
          <td>0.005247</td>
          <td>22.607896</td>
          <td>0.006653</td>
          <td>21.083752</td>
          <td>0.005111</td>
          <td>21.216909</td>
          <td>0.005327</td>
          <td>23.928200</td>
          <td>0.039308</td>
          <td>24.779660</td>
          <td>0.185307</td>
          <td>25.336248</td>
          <td>27.526966</td>
          <td>26.822995</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.221834</td>
          <td>0.010623</td>
          <td>22.059459</td>
          <td>0.005716</td>
          <td>22.051166</td>
          <td>0.005517</td>
          <td>18.778029</td>
          <td>0.005010</td>
          <td>27.661704</td>
          <td>0.822648</td>
          <td>26.042165</td>
          <td>0.506717</td>
          <td>23.563985</td>
          <td>21.540648</td>
          <td>25.356531</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.999089</td>
          <td>0.009229</td>
          <td>22.842796</td>
          <td>0.007340</td>
          <td>21.948836</td>
          <td>0.005438</td>
          <td>26.542087</td>
          <td>0.203337</td>
          <td>27.736586</td>
          <td>0.863064</td>
          <td>25.301102</td>
          <td>0.285408</td>
          <td>18.138293</td>
          <td>17.843495</td>
          <td>27.354061</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.872913</td>
          <td>0.040280</td>
          <td>25.276265</td>
          <td>0.047268</td>
          <td>21.419477</td>
          <td>0.005188</td>
          <td>20.719384</td>
          <td>0.005148</td>
          <td>26.167736</td>
          <td>0.273481</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.750971</td>
          <td>19.813719</td>
          <td>19.674127</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.017590</td>
          <td>0.019377</td>
          <td>28.128755</td>
          <td>0.515990</td>
          <td>19.818152</td>
          <td>0.005019</td>
          <td>24.148721</td>
          <td>0.025051</td>
          <td>22.427351</td>
          <td>0.011239</td>
          <td>17.644385</td>
          <td>0.005020</td>
          <td>19.911878</td>
          <td>25.214031</td>
          <td>18.788371</td>
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
          <td>21.422126</td>
          <td>0.006945</td>
          <td>26.491241</td>
          <td>0.137450</td>
          <td>22.990552</td>
          <td>0.007265</td>
          <td>19.900546</td>
          <td>0.005044</td>
          <td>21.427636</td>
          <td>0.006491</td>
          <td>22.752184</td>
          <td>0.031337</td>
          <td>21.965784</td>
          <td>21.623891</td>
          <td>27.297555</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.151828</td>
          <td>0.006335</td>
          <td>22.644068</td>
          <td>0.006745</td>
          <td>22.459820</td>
          <td>0.005996</td>
          <td>25.689332</td>
          <td>0.097638</td>
          <td>22.200335</td>
          <td>0.009614</td>
          <td>24.216441</td>
          <td>0.114211</td>
          <td>21.789051</td>
          <td>24.023200</td>
          <td>18.088282</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.330151</td>
          <td>0.011435</td>
          <td>23.640772</td>
          <td>0.011939</td>
          <td>25.609064</td>
          <td>0.055769</td>
          <td>22.737383</td>
          <td>0.008506</td>
          <td>20.493123</td>
          <td>0.005336</td>
          <td>23.685050</td>
          <td>0.071599</td>
          <td>26.274973</td>
          <td>21.877116</td>
          <td>23.125813</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.047715</td>
          <td>0.005294</td>
          <td>23.754442</td>
          <td>0.012997</td>
          <td>25.133812</td>
          <td>0.036586</td>
          <td>23.254387</td>
          <td>0.012031</td>
          <td>24.166022</td>
          <td>0.048539</td>
          <td>26.564982</td>
          <td>0.732002</td>
          <td>29.652060</td>
          <td>22.794092</td>
          <td>19.300932</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.597342</td>
          <td>0.005053</td>
          <td>22.918567</td>
          <td>0.007611</td>
          <td>20.904952</td>
          <td>0.005085</td>
          <td>20.712620</td>
          <td>0.005147</td>
          <td>23.558602</td>
          <td>0.028382</td>
          <td>23.902424</td>
          <td>0.086743</td>
          <td>20.786495</td>
          <td>24.480347</td>
          <td>22.746686</td>
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
          <td>19.905756</td>
          <td>22.595966</td>
          <td>21.079622</td>
          <td>21.224496</td>
          <td>23.944839</td>
          <td>24.717981</td>
          <td>25.258803</td>
          <td>0.043099</td>
          <td>27.520345</td>
          <td>0.483705</td>
          <td>26.596798</td>
          <td>0.233370</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.214478</td>
          <td>22.059164</td>
          <td>22.053685</td>
          <td>18.780520</td>
          <td>26.628718</td>
          <td>26.580842</td>
          <td>23.561637</td>
          <td>0.010383</td>
          <td>21.544724</td>
          <td>0.005578</td>
          <td>25.407834</td>
          <td>0.083892</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.002177</td>
          <td>22.840270</td>
          <td>21.952955</td>
          <td>26.448755</td>
          <td>27.632442</td>
          <td>26.751562</td>
          <td>18.147269</td>
          <td>0.005000</td>
          <td>17.841939</td>
          <td>0.005001</td>
          <td>27.067211</td>
          <td>0.341630</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.837799</td>
          <td>25.188032</td>
          <td>21.421728</td>
          <td>20.718911</td>
          <td>26.386796</td>
          <td>28.631238</td>
          <td>21.742456</td>
          <td>0.005283</td>
          <td>19.805788</td>
          <td>0.005025</td>
          <td>19.677673</td>
          <td>0.005020</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.026782</td>
          <td>27.537906</td>
          <td>19.816694</td>
          <td>24.154672</td>
          <td>22.435838</td>
          <td>17.646177</td>
          <td>19.914645</td>
          <td>0.005010</td>
          <td>25.179704</td>
          <td>0.068541</td>
          <td>18.785395</td>
          <td>0.005004</td>
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
          <td>21.423785</td>
          <td>26.562500</td>
          <td>22.991418</td>
          <td>19.902476</td>
          <td>21.429431</td>
          <td>22.724964</td>
          <td>21.965613</td>
          <td>0.005422</td>
          <td>21.621545</td>
          <td>0.005660</td>
          <td>27.276452</td>
          <td>0.402197</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.154670</td>
          <td>22.647636</td>
          <td>22.453978</td>
          <td>25.803136</td>
          <td>22.204539</td>
          <td>24.190869</td>
          <td>21.789658</td>
          <td>0.005308</td>
          <td>24.052895</td>
          <td>0.025201</td>
          <td>18.091995</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.347147</td>
          <td>23.640986</td>
          <td>25.741203</td>
          <td>22.729064</td>
          <td>20.485232</td>
          <td>23.713040</td>
          <td>26.343366</td>
          <td>0.112658</td>
          <td>21.876747</td>
          <td>0.006022</td>
          <td>23.124369</td>
          <td>0.011688</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.048422</td>
          <td>23.761783</td>
          <td>25.110742</td>
          <td>23.252867</td>
          <td>24.276911</td>
          <td>26.748183</td>
          <td>28.210202</td>
          <td>0.516869</td>
          <td>22.783501</td>
          <td>0.009203</td>
          <td>19.298488</td>
          <td>0.005010</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.588004</td>
          <td>22.921991</td>
          <td>20.905526</td>
          <td>20.707702</td>
          <td>23.551973</td>
          <td>23.965402</td>
          <td>20.778284</td>
          <td>0.005049</td>
          <td>24.462737</td>
          <td>0.036195</td>
          <td>22.769001</td>
          <td>0.009117</td>
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
          <td>19.905756</td>
          <td>22.595966</td>
          <td>21.079622</td>
          <td>21.224496</td>
          <td>23.944839</td>
          <td>24.717981</td>
          <td>26.086459</td>
          <td>0.772273</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.214478</td>
          <td>22.059164</td>
          <td>22.053685</td>
          <td>18.780520</td>
          <td>26.628718</td>
          <td>26.580842</td>
          <td>23.465142</td>
          <td>0.096368</td>
          <td>21.527014</td>
          <td>0.014812</td>
          <td>24.717960</td>
          <td>0.257874</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.002177</td>
          <td>22.840270</td>
          <td>21.952955</td>
          <td>26.448755</td>
          <td>27.632442</td>
          <td>26.751562</td>
          <td>18.140892</td>
          <td>0.005055</td>
          <td>17.845113</td>
          <td>0.005022</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.837799</td>
          <td>25.188032</td>
          <td>21.421728</td>
          <td>20.718911</td>
          <td>26.386796</td>
          <td>28.631238</td>
          <td>21.747636</td>
          <td>0.021084</td>
          <td>19.823523</td>
          <td>0.005787</td>
          <td>19.672113</td>
          <td>0.005721</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.026782</td>
          <td>27.537906</td>
          <td>19.816694</td>
          <td>24.154672</td>
          <td>22.435838</td>
          <td>17.646177</td>
          <td>19.907923</td>
          <td>0.006272</td>
          <td>25.935014</td>
          <td>0.607576</td>
          <td>18.787040</td>
          <td>0.005149</td>
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
          <td>21.423785</td>
          <td>26.562500</td>
          <td>22.991418</td>
          <td>19.902476</td>
          <td>21.429431</td>
          <td>22.724964</td>
          <td>21.947720</td>
          <td>0.025087</td>
          <td>21.629280</td>
          <td>0.016107</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.154670</td>
          <td>22.647636</td>
          <td>22.453978</td>
          <td>25.803136</td>
          <td>22.204539</td>
          <td>24.190869</td>
          <td>21.745685</td>
          <td>0.021049</td>
          <td>24.020645</td>
          <td>0.131443</td>
          <td>18.089330</td>
          <td>0.005042</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.347147</td>
          <td>23.640986</td>
          <td>25.741203</td>
          <td>22.729064</td>
          <td>20.485232</td>
          <td>23.713040</td>
          <td>25.495787</td>
          <td>0.511432</td>
          <td>21.858129</td>
          <td>0.019526</td>
          <td>23.080744</td>
          <td>0.062769</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.048422</td>
          <td>23.761783</td>
          <td>25.110742</td>
          <td>23.252867</td>
          <td>24.276911</td>
          <td>26.748183</td>
          <td>25.826041</td>
          <td>0.647524</td>
          <td>22.772218</td>
          <td>0.043617</td>
          <td>19.297213</td>
          <td>0.005373</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.588004</td>
          <td>22.921991</td>
          <td>20.905526</td>
          <td>20.707702</td>
          <td>23.551973</td>
          <td>23.965402</td>
          <td>20.804092</td>
          <td>0.009975</td>
          <td>24.636325</td>
          <td>0.221938</td>
          <td>22.771411</td>
          <td>0.047650</td>
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


