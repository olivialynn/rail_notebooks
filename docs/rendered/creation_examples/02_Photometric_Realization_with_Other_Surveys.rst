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
          <td>22.717941</td>
          <td>20.771821</td>
          <td>27.254462</td>
          <td>19.567456</td>
          <td>22.156478</td>
          <td>18.897846</td>
          <td>26.680376</td>
          <td>23.853523</td>
          <td>21.314919</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.595197</td>
          <td>20.906272</td>
          <td>26.853808</td>
          <td>20.274475</td>
          <td>21.437945</td>
          <td>20.118930</td>
          <td>21.646058</td>
          <td>20.161665</td>
          <td>24.765221</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.377220</td>
          <td>20.942054</td>
          <td>24.876338</td>
          <td>28.474118</td>
          <td>24.124880</td>
          <td>20.304054</td>
          <td>23.689901</td>
          <td>23.873352</td>
          <td>24.969246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.176412</td>
          <td>23.690449</td>
          <td>19.864929</td>
          <td>23.884178</td>
          <td>20.867784</td>
          <td>23.378411</td>
          <td>25.646191</td>
          <td>23.380867</td>
          <td>19.902321</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.891928</td>
          <td>21.178496</td>
          <td>15.803901</td>
          <td>24.185409</td>
          <td>23.091882</td>
          <td>22.934481</td>
          <td>20.418052</td>
          <td>23.231064</td>
          <td>25.669053</td>
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
          <td>21.692944</td>
          <td>23.123406</td>
          <td>22.791682</td>
          <td>21.639006</td>
          <td>23.379157</td>
          <td>19.688442</td>
          <td>22.826069</td>
          <td>25.773335</td>
          <td>28.341781</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.554630</td>
          <td>23.818679</td>
          <td>19.000273</td>
          <td>25.832785</td>
          <td>24.995080</td>
          <td>22.599558</td>
          <td>21.508492</td>
          <td>26.196997</td>
          <td>23.752239</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.305149</td>
          <td>22.822199</td>
          <td>22.317372</td>
          <td>21.263043</td>
          <td>22.913853</td>
          <td>24.619558</td>
          <td>23.320693</td>
          <td>18.122319</td>
          <td>23.582830</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.387237</td>
          <td>29.366981</td>
          <td>19.086229</td>
          <td>27.245293</td>
          <td>21.448781</td>
          <td>23.121782</td>
          <td>27.473921</td>
          <td>22.296479</td>
          <td>18.492188</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.238070</td>
          <td>26.993428</td>
          <td>19.910327</td>
          <td>22.381317</td>
          <td>24.752458</td>
          <td>25.034801</td>
          <td>21.316308</td>
          <td>20.788893</td>
          <td>26.549097</td>
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
          <td>22.703013</td>
          <td>0.015060</td>
          <td>20.771271</td>
          <td>0.005105</td>
          <td>27.238831</td>
          <td>0.229127</td>
          <td>19.565602</td>
          <td>0.005027</td>
          <td>22.156397</td>
          <td>0.009345</td>
          <td>18.909917</td>
          <td>0.005119</td>
          <td>26.680376</td>
          <td>23.853523</td>
          <td>21.314919</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.596920</td>
          <td>0.005616</td>
          <td>20.903765</td>
          <td>0.005127</td>
          <td>26.862125</td>
          <td>0.166896</td>
          <td>20.273583</td>
          <td>0.005075</td>
          <td>21.434703</td>
          <td>0.006507</td>
          <td>20.110623</td>
          <td>0.005797</td>
          <td>21.646058</td>
          <td>20.161665</td>
          <td>24.765221</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.379415</td>
          <td>0.011837</td>
          <td>20.939570</td>
          <td>0.005133</td>
          <td>24.875239</td>
          <td>0.029134</td>
          <td>29.540806</td>
          <td>1.572741</td>
          <td>24.046251</td>
          <td>0.043644</td>
          <td>20.304810</td>
          <td>0.006085</td>
          <td>23.689901</td>
          <td>23.873352</td>
          <td>24.969246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.082710</td>
          <td>0.270500</td>
          <td>23.688556</td>
          <td>0.012368</td>
          <td>19.871221</td>
          <td>0.005020</td>
          <td>23.911624</td>
          <td>0.020432</td>
          <td>20.869728</td>
          <td>0.005616</td>
          <td>23.362919</td>
          <td>0.053812</td>
          <td>25.646191</td>
          <td>23.380867</td>
          <td>19.902321</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.899293</td>
          <td>0.005937</td>
          <td>21.177967</td>
          <td>0.005188</td>
          <td>15.796786</td>
          <td>0.005000</td>
          <td>24.177761</td>
          <td>0.025691</td>
          <td>23.102380</td>
          <td>0.019165</td>
          <td>22.939231</td>
          <td>0.036958</td>
          <td>20.418052</td>
          <td>23.231064</td>
          <td>25.669053</td>
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
          <td>21.711833</td>
          <td>0.007890</td>
          <td>23.129363</td>
          <td>0.008520</td>
          <td>22.786963</td>
          <td>0.006663</td>
          <td>21.640225</td>
          <td>0.005649</td>
          <td>23.436069</td>
          <td>0.025506</td>
          <td>19.694356</td>
          <td>0.005409</td>
          <td>22.826069</td>
          <td>25.773335</td>
          <td>28.341781</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.546878</td>
          <td>0.013371</td>
          <td>23.829617</td>
          <td>0.013770</td>
          <td>18.998175</td>
          <td>0.005007</td>
          <td>25.890464</td>
          <td>0.116400</td>
          <td>25.041845</td>
          <td>0.105228</td>
          <td>22.534813</td>
          <td>0.025910</td>
          <td>21.508492</td>
          <td>26.196997</td>
          <td>23.752239</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.182612</td>
          <td>1.928039</td>
          <td>22.815396</td>
          <td>0.007248</td>
          <td>22.329047</td>
          <td>0.005808</td>
          <td>21.265560</td>
          <td>0.005354</td>
          <td>22.921592</td>
          <td>0.016490</td>
          <td>25.215742</td>
          <td>0.266282</td>
          <td>23.320693</td>
          <td>18.122319</td>
          <td>23.582830</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.394960</td>
          <td>0.063732</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.082081</td>
          <td>0.005008</td>
          <td>27.350567</td>
          <td>0.390925</td>
          <td>21.456677</td>
          <td>0.006560</td>
          <td>23.096525</td>
          <td>0.042482</td>
          <td>27.473921</td>
          <td>22.296479</td>
          <td>18.492188</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.232887</td>
          <td>0.006495</td>
          <td>27.414636</td>
          <td>0.297598</td>
          <td>19.915503</td>
          <td>0.005021</td>
          <td>22.388364</td>
          <td>0.007103</td>
          <td>24.839122</td>
          <td>0.088084</td>
          <td>24.791910</td>
          <td>0.187235</td>
          <td>21.316308</td>
          <td>20.788893</td>
          <td>26.549097</td>
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
          <td>22.717941</td>
          <td>20.771821</td>
          <td>27.254462</td>
          <td>19.567456</td>
          <td>22.156478</td>
          <td>18.897846</td>
          <td>26.892423</td>
          <td>0.180802</td>
          <td>23.839176</td>
          <td>0.020931</td>
          <td>21.317466</td>
          <td>0.005387</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.595197</td>
          <td>20.906272</td>
          <td>26.853808</td>
          <td>20.274475</td>
          <td>21.437945</td>
          <td>20.118930</td>
          <td>21.651813</td>
          <td>0.005241</td>
          <td>20.162018</td>
          <td>0.005048</td>
          <td>24.802947</td>
          <td>0.049009</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.377220</td>
          <td>20.942054</td>
          <td>24.876338</td>
          <td>28.474118</td>
          <td>24.124880</td>
          <td>20.304054</td>
          <td>23.676417</td>
          <td>0.011279</td>
          <td>23.910240</td>
          <td>0.022256</td>
          <td>24.987856</td>
          <td>0.057789</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.176412</td>
          <td>23.690449</td>
          <td>19.864929</td>
          <td>23.884178</td>
          <td>20.867784</td>
          <td>23.378411</td>
          <td>25.638050</td>
          <td>0.060430</td>
          <td>23.378384</td>
          <td>0.014242</td>
          <td>19.900514</td>
          <td>0.005029</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.891928</td>
          <td>21.178496</td>
          <td>15.803901</td>
          <td>24.185409</td>
          <td>23.091882</td>
          <td>22.934481</td>
          <td>20.427621</td>
          <td>0.005026</td>
          <td>23.241818</td>
          <td>0.012785</td>
          <td>25.743431</td>
          <td>0.112664</td>
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
          <td>21.692944</td>
          <td>23.123406</td>
          <td>22.791682</td>
          <td>21.639006</td>
          <td>23.379157</td>
          <td>19.688442</td>
          <td>22.830490</td>
          <td>0.006826</td>
          <td>25.707172</td>
          <td>0.109150</td>
          <td>28.208945</td>
          <td>0.783776</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.554630</td>
          <td>23.818679</td>
          <td>19.000273</td>
          <td>25.832785</td>
          <td>24.995080</td>
          <td>22.599558</td>
          <td>21.508381</td>
          <td>0.005186</td>
          <td>26.382994</td>
          <td>0.195186</td>
          <td>23.739642</td>
          <td>0.019221</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.305149</td>
          <td>22.822199</td>
          <td>22.317372</td>
          <td>21.263043</td>
          <td>22.913853</td>
          <td>24.619558</td>
          <td>23.330810</td>
          <td>0.008899</td>
          <td>18.124441</td>
          <td>0.005001</td>
          <td>23.555133</td>
          <td>0.016456</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.387237</td>
          <td>29.366981</td>
          <td>19.086229</td>
          <td>27.245293</td>
          <td>21.448781</td>
          <td>23.121782</td>
          <td>27.188628</td>
          <td>0.231795</td>
          <td>22.307666</td>
          <td>0.007063</td>
          <td>18.498059</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.238070</td>
          <td>26.993428</td>
          <td>19.910327</td>
          <td>22.381317</td>
          <td>24.752458</td>
          <td>25.034801</td>
          <td>21.322712</td>
          <td>0.005133</td>
          <td>20.793692</td>
          <td>0.005151</td>
          <td>26.550679</td>
          <td>0.224605</td>
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
          <td>22.717941</td>
          <td>20.771821</td>
          <td>27.254462</td>
          <td>19.567456</td>
          <td>22.156478</td>
          <td>18.897846</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.796900</td>
          <td>0.108174</td>
          <td>21.324208</td>
          <td>0.013639</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.595197</td>
          <td>20.906272</td>
          <td>26.853808</td>
          <td>20.274475</td>
          <td>21.437945</td>
          <td>20.118930</td>
          <td>21.617807</td>
          <td>0.018867</td>
          <td>20.170201</td>
          <td>0.006409</td>
          <td>24.650303</td>
          <td>0.243922</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.377220</td>
          <td>20.942054</td>
          <td>24.876338</td>
          <td>28.474118</td>
          <td>24.124880</td>
          <td>20.304054</td>
          <td>23.528505</td>
          <td>0.101882</td>
          <td>23.799191</td>
          <td>0.108391</td>
          <td>25.052642</td>
          <td>0.337717</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.176412</td>
          <td>23.690449</td>
          <td>19.864929</td>
          <td>23.884178</td>
          <td>20.867784</td>
          <td>23.378411</td>
          <td>26.296467</td>
          <td>0.884225</td>
          <td>23.322068</td>
          <td>0.071167</td>
          <td>19.899388</td>
          <td>0.006061</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.891928</td>
          <td>21.178496</td>
          <td>15.803901</td>
          <td>24.185409</td>
          <td>23.091882</td>
          <td>22.934481</td>
          <td>20.422350</td>
          <td>0.007870</td>
          <td>23.215332</td>
          <td>0.064730</td>
          <td>27.869560</td>
          <td>1.924345</td>
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
          <td>21.692944</td>
          <td>23.123406</td>
          <td>22.791682</td>
          <td>21.639006</td>
          <td>23.379157</td>
          <td>19.688442</td>
          <td>22.777380</td>
          <td>0.052371</td>
          <td>26.106708</td>
          <td>0.684501</td>
          <td>29.975501</td>
          <td>3.859567</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.554630</td>
          <td>23.818679</td>
          <td>19.000273</td>
          <td>25.832785</td>
          <td>24.995080</td>
          <td>22.599558</td>
          <td>21.516679</td>
          <td>0.017323</td>
          <td>25.962025</td>
          <td>0.619234</td>
          <td>23.854593</td>
          <td>0.124121</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.305149</td>
          <td>22.822199</td>
          <td>22.317372</td>
          <td>21.263043</td>
          <td>22.913853</td>
          <td>24.619558</td>
          <td>23.356661</td>
          <td>0.087587</td>
          <td>18.126061</td>
          <td>0.005037</td>
          <td>23.587993</td>
          <td>0.098323</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.387237</td>
          <td>29.366981</td>
          <td>19.086229</td>
          <td>27.245293</td>
          <td>21.448781</td>
          <td>23.121782</td>
          <td>26.999564</td>
          <td>1.329836</td>
          <td>22.263613</td>
          <td>0.027774</td>
          <td>18.494904</td>
          <td>0.005088</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.238070</td>
          <td>26.993428</td>
          <td>19.910327</td>
          <td>22.381317</td>
          <td>24.752458</td>
          <td>25.034801</td>
          <td>21.326593</td>
          <td>0.014807</td>
          <td>20.799409</td>
          <td>0.008726</td>
          <td>25.537130</td>
          <td>0.489769</td>
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


