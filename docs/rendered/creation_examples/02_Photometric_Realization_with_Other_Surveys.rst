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
          <td>23.126094</td>
          <td>21.765563</td>
          <td>24.345552</td>
          <td>23.335631</td>
          <td>33.190005</td>
          <td>23.412083</td>
          <td>26.853549</td>
          <td>22.583124</td>
          <td>24.501439</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.300216</td>
          <td>26.376761</td>
          <td>26.300771</td>
          <td>25.583268</td>
          <td>25.947449</td>
          <td>19.951740</td>
          <td>22.670735</td>
          <td>22.698961</td>
          <td>23.429990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.928457</td>
          <td>17.434263</td>
          <td>23.571964</td>
          <td>23.512589</td>
          <td>23.446879</td>
          <td>23.734994</td>
          <td>20.747074</td>
          <td>26.064407</td>
          <td>22.774020</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.209107</td>
          <td>20.787868</td>
          <td>19.745689</td>
          <td>23.997376</td>
          <td>25.712104</td>
          <td>22.390197</td>
          <td>19.855140</td>
          <td>26.147193</td>
          <td>22.538018</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.820682</td>
          <td>22.015283</td>
          <td>21.497701</td>
          <td>24.134490</td>
          <td>23.766229</td>
          <td>22.478887</td>
          <td>20.845650</td>
          <td>25.564745</td>
          <td>25.298754</td>
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
          <td>22.798513</td>
          <td>20.269476</td>
          <td>27.045263</td>
          <td>21.177024</td>
          <td>23.587394</td>
          <td>19.099538</td>
          <td>19.636201</td>
          <td>22.212361</td>
          <td>24.584888</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.968858</td>
          <td>20.705901</td>
          <td>21.601957</td>
          <td>22.909944</td>
          <td>15.680377</td>
          <td>17.959925</td>
          <td>23.726013</td>
          <td>22.440163</td>
          <td>25.311998</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.682696</td>
          <td>23.569851</td>
          <td>20.351372</td>
          <td>26.929603</td>
          <td>23.491797</td>
          <td>18.316621</td>
          <td>21.151882</td>
          <td>21.291536</td>
          <td>22.556460</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.190910</td>
          <td>22.532240</td>
          <td>26.133386</td>
          <td>17.296324</td>
          <td>19.768827</td>
          <td>24.689066</td>
          <td>17.569524</td>
          <td>28.201759</td>
          <td>24.179384</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.251957</td>
          <td>22.397777</td>
          <td>19.915855</td>
          <td>21.922858</td>
          <td>29.057112</td>
          <td>24.027718</td>
          <td>22.616556</td>
          <td>20.617510</td>
          <td>15.521707</td>
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
          <td>23.082730</td>
          <td>0.020448</td>
          <td>21.762483</td>
          <td>0.005453</td>
          <td>24.354097</td>
          <td>0.018605</td>
          <td>23.346733</td>
          <td>0.012901</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.457649</td>
          <td>0.058532</td>
          <td>26.853549</td>
          <td>22.583124</td>
          <td>24.501439</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.523386</td>
          <td>0.787047</td>
          <td>26.572363</td>
          <td>0.147386</td>
          <td>26.237637</td>
          <td>0.097189</td>
          <td>25.621148</td>
          <td>0.091965</td>
          <td>26.047091</td>
          <td>0.247779</td>
          <td>19.948664</td>
          <td>0.005615</td>
          <td>22.670735</td>
          <td>22.698961</td>
          <td>23.429990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.936465</td>
          <td>0.005987</td>
          <td>17.433998</td>
          <td>0.005002</td>
          <td>23.604001</td>
          <td>0.010394</td>
          <td>23.501441</td>
          <td>0.014560</td>
          <td>23.458586</td>
          <td>0.026010</td>
          <td>23.788055</td>
          <td>0.078422</td>
          <td>20.747074</td>
          <td>26.064407</td>
          <td>22.774020</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.202726</td>
          <td>0.005104</td>
          <td>20.784462</td>
          <td>0.005107</td>
          <td>19.747645</td>
          <td>0.005017</td>
          <td>23.975984</td>
          <td>0.021585</td>
          <td>26.211729</td>
          <td>0.283423</td>
          <td>22.416070</td>
          <td>0.023378</td>
          <td>19.855140</td>
          <td>26.147193</td>
          <td>22.538018</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.073098</td>
          <td>0.578267</td>
          <td>22.015690</td>
          <td>0.005669</td>
          <td>21.506291</td>
          <td>0.005215</td>
          <td>24.128170</td>
          <td>0.024608</td>
          <td>23.783964</td>
          <td>0.034601</td>
          <td>22.495393</td>
          <td>0.025038</td>
          <td>20.845650</td>
          <td>25.564745</td>
          <td>25.298754</td>
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
          <td>22.815390</td>
          <td>0.016452</td>
          <td>20.269366</td>
          <td>0.005053</td>
          <td>27.672870</td>
          <td>0.326074</td>
          <td>21.177415</td>
          <td>0.005307</td>
          <td>23.566513</td>
          <td>0.028579</td>
          <td>19.097732</td>
          <td>0.005159</td>
          <td>19.636201</td>
          <td>22.212361</td>
          <td>24.584888</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.970501</td>
          <td>0.006036</td>
          <td>20.707089</td>
          <td>0.005096</td>
          <td>21.603988</td>
          <td>0.005252</td>
          <td>22.914759</td>
          <td>0.009489</td>
          <td>15.692750</td>
          <td>0.005001</td>
          <td>17.965505</td>
          <td>0.005030</td>
          <td>23.726013</td>
          <td>22.440163</td>
          <td>25.311998</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.701310</td>
          <td>0.007850</td>
          <td>23.551427</td>
          <td>0.011192</td>
          <td>20.341562</td>
          <td>0.005037</td>
          <td>27.053410</td>
          <td>0.309379</td>
          <td>23.524477</td>
          <td>0.027548</td>
          <td>18.325047</td>
          <td>0.005050</td>
          <td>21.151882</td>
          <td>21.291536</td>
          <td>22.556460</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.112584</td>
          <td>0.594731</td>
          <td>22.545047</td>
          <td>0.006504</td>
          <td>26.125675</td>
          <td>0.088084</td>
          <td>17.301478</td>
          <td>0.005002</td>
          <td>19.768653</td>
          <td>0.005108</td>
          <td>24.502324</td>
          <td>0.146280</td>
          <td>17.569524</td>
          <td>28.201759</td>
          <td>24.179384</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.576219</td>
          <td>0.177608</td>
          <td>22.391234</td>
          <td>0.006191</td>
          <td>19.911799</td>
          <td>0.005021</td>
          <td>21.921596</td>
          <td>0.006019</td>
          <td>27.501454</td>
          <td>0.740483</td>
          <td>23.933874</td>
          <td>0.089177</td>
          <td>22.616556</td>
          <td>20.617510</td>
          <td>15.521707</td>
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
          <td>23.126094</td>
          <td>21.765563</td>
          <td>24.345552</td>
          <td>23.335631</td>
          <td>33.190005</td>
          <td>23.412083</td>
          <td>26.940957</td>
          <td>0.188384</td>
          <td>22.577602</td>
          <td>0.008117</td>
          <td>24.482055</td>
          <td>0.036823</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.300216</td>
          <td>26.376761</td>
          <td>26.300771</td>
          <td>25.583268</td>
          <td>25.947449</td>
          <td>19.951740</td>
          <td>22.667082</td>
          <td>0.006402</td>
          <td>22.692963</td>
          <td>0.008692</td>
          <td>23.435528</td>
          <td>0.014915</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.928457</td>
          <td>17.434263</td>
          <td>23.571964</td>
          <td>23.512589</td>
          <td>23.446879</td>
          <td>23.734994</td>
          <td>20.745426</td>
          <td>0.005046</td>
          <td>26.145161</td>
          <td>0.159484</td>
          <td>22.769309</td>
          <td>0.009119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.209107</td>
          <td>20.787868</td>
          <td>19.745689</td>
          <td>23.997376</td>
          <td>25.712104</td>
          <td>22.390197</td>
          <td>19.850069</td>
          <td>0.005009</td>
          <td>26.212276</td>
          <td>0.168893</td>
          <td>22.522449</td>
          <td>0.007870</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.820682</td>
          <td>22.015283</td>
          <td>21.497701</td>
          <td>24.134490</td>
          <td>23.766229</td>
          <td>22.478887</td>
          <td>20.845965</td>
          <td>0.005056</td>
          <td>25.520719</td>
          <td>0.092674</td>
          <td>25.301075</td>
          <td>0.076331</td>
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
          <td>22.798513</td>
          <td>20.269476</td>
          <td>27.045263</td>
          <td>21.177024</td>
          <td>23.587394</td>
          <td>19.099538</td>
          <td>19.629289</td>
          <td>0.005006</td>
          <td>22.208560</td>
          <td>0.006763</td>
          <td>24.567158</td>
          <td>0.039720</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.968858</td>
          <td>20.705901</td>
          <td>21.601957</td>
          <td>22.909944</td>
          <td>15.680377</td>
          <td>17.959925</td>
          <td>23.724830</td>
          <td>0.011692</td>
          <td>22.443259</td>
          <td>0.007546</td>
          <td>25.214135</td>
          <td>0.070668</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.682696</td>
          <td>23.569851</td>
          <td>20.351372</td>
          <td>26.929603</td>
          <td>23.491797</td>
          <td>18.316621</td>
          <td>21.151364</td>
          <td>0.005097</td>
          <td>21.284130</td>
          <td>0.005365</td>
          <td>22.570438</td>
          <td>0.008084</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.190910</td>
          <td>22.532240</td>
          <td>26.133386</td>
          <td>17.296324</td>
          <td>19.768827</td>
          <td>24.689066</td>
          <td>17.561876</td>
          <td>0.005000</td>
          <td>28.065864</td>
          <td>0.712566</td>
          <td>24.196525</td>
          <td>0.028591</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.251957</td>
          <td>22.397777</td>
          <td>19.915855</td>
          <td>21.922858</td>
          <td>29.057112</td>
          <td>24.027718</td>
          <td>22.625354</td>
          <td>0.006309</td>
          <td>20.623175</td>
          <td>0.005111</td>
          <td>15.524089</td>
          <td>0.005000</td>
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
          <td>23.126094</td>
          <td>21.765563</td>
          <td>24.345552</td>
          <td>23.335631</td>
          <td>33.190005</td>
          <td>23.412083</td>
          <td>25.764287</td>
          <td>0.620218</td>
          <td>22.559807</td>
          <td>0.036101</td>
          <td>24.787006</td>
          <td>0.272837</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.300216</td>
          <td>26.376761</td>
          <td>26.300771</td>
          <td>25.583268</td>
          <td>25.947449</td>
          <td>19.951740</td>
          <td>22.632808</td>
          <td>0.046038</td>
          <td>22.681679</td>
          <td>0.040237</td>
          <td>23.505438</td>
          <td>0.091435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.928457</td>
          <td>17.434263</td>
          <td>23.571964</td>
          <td>23.512589</td>
          <td>23.446879</td>
          <td>23.734994</td>
          <td>20.748421</td>
          <td>0.009605</td>
          <td>26.598722</td>
          <td>0.942373</td>
          <td>22.713688</td>
          <td>0.045260</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.209107</td>
          <td>20.787868</td>
          <td>19.745689</td>
          <td>23.997376</td>
          <td>25.712104</td>
          <td>22.390197</td>
          <td>19.850527</td>
          <td>0.006156</td>
          <td>26.211694</td>
          <td>0.734862</td>
          <td>22.497120</td>
          <td>0.037319</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.820682</td>
          <td>22.015283</td>
          <td>21.497701</td>
          <td>24.134490</td>
          <td>23.766229</td>
          <td>22.478887</td>
          <td>20.840683</td>
          <td>0.010231</td>
          <td>25.122556</td>
          <td>0.329757</td>
          <td>28.139358</td>
          <td>2.152771</td>
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
          <td>22.798513</td>
          <td>20.269476</td>
          <td>27.045263</td>
          <td>21.177024</td>
          <td>23.587394</td>
          <td>19.099538</td>
          <td>19.632486</td>
          <td>0.005800</td>
          <td>22.248129</td>
          <td>0.027398</td>
          <td>24.489868</td>
          <td>0.213501</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.968858</td>
          <td>20.705901</td>
          <td>21.601957</td>
          <td>22.909944</td>
          <td>15.680377</td>
          <td>17.959925</td>
          <td>23.822235</td>
          <td>0.131624</td>
          <td>22.440256</td>
          <td>0.032466</td>
          <td>25.170396</td>
          <td>0.370462</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.682696</td>
          <td>23.569851</td>
          <td>20.351372</td>
          <td>26.929603</td>
          <td>23.491797</td>
          <td>18.316621</td>
          <td>21.145432</td>
          <td>0.012821</td>
          <td>21.290969</td>
          <td>0.012293</td>
          <td>22.585097</td>
          <td>0.040360</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.190910</td>
          <td>22.532240</td>
          <td>26.133386</td>
          <td>17.296324</td>
          <td>19.768827</td>
          <td>24.689066</td>
          <td>17.566716</td>
          <td>0.005019</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.290450</td>
          <td>0.180499</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.251957</td>
          <td>22.397777</td>
          <td>19.915855</td>
          <td>21.922858</td>
          <td>29.057112</td>
          <td>24.027718</td>
          <td>22.609501</td>
          <td>0.045091</td>
          <td>20.602856</td>
          <td>0.007787</td>
          <td>15.520232</td>
          <td>0.005000</td>
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


