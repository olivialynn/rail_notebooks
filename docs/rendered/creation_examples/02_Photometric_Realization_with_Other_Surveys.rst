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
          <td>19.307233</td>
          <td>25.493946</td>
          <td>24.314966</td>
          <td>19.341316</td>
          <td>23.062761</td>
          <td>17.301525</td>
          <td>22.158762</td>
          <td>28.404047</td>
          <td>13.548669</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.190851</td>
          <td>22.340852</td>
          <td>27.637128</td>
          <td>22.367582</td>
          <td>20.932581</td>
          <td>25.933584</td>
          <td>16.760697</td>
          <td>15.999670</td>
          <td>15.716736</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.468687</td>
          <td>22.761595</td>
          <td>18.835067</td>
          <td>17.888131</td>
          <td>21.061232</td>
          <td>19.272387</td>
          <td>26.031616</td>
          <td>22.787321</td>
          <td>24.726977</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.732829</td>
          <td>15.231048</td>
          <td>17.918841</td>
          <td>22.797259</td>
          <td>22.722300</td>
          <td>21.326085</td>
          <td>24.711629</td>
          <td>19.024543</td>
          <td>23.853571</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.097255</td>
          <td>24.712366</td>
          <td>20.990974</td>
          <td>21.985040</td>
          <td>19.718199</td>
          <td>17.703882</td>
          <td>24.610037</td>
          <td>25.832075</td>
          <td>20.697271</td>
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
          <td>20.405699</td>
          <td>16.619559</td>
          <td>26.977351</td>
          <td>14.248519</td>
          <td>21.416118</td>
          <td>22.856268</td>
          <td>21.497059</td>
          <td>22.816504</td>
          <td>23.298171</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.621764</td>
          <td>24.708062</td>
          <td>26.558141</td>
          <td>26.321639</td>
          <td>18.317362</td>
          <td>20.038199</td>
          <td>27.371366</td>
          <td>18.215126</td>
          <td>18.280594</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.531048</td>
          <td>21.332732</td>
          <td>25.889963</td>
          <td>22.918798</td>
          <td>26.262848</td>
          <td>21.568089</td>
          <td>26.200966</td>
          <td>19.890239</td>
          <td>23.935838</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.053932</td>
          <td>26.875711</td>
          <td>19.833202</td>
          <td>23.452538</td>
          <td>23.024416</td>
          <td>22.625790</td>
          <td>23.555913</td>
          <td>19.982033</td>
          <td>26.017810</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.196979</td>
          <td>23.737257</td>
          <td>22.187823</td>
          <td>21.258408</td>
          <td>23.784922</td>
          <td>27.265961</td>
          <td>18.614158</td>
          <td>21.045989</td>
          <td>18.516025</td>
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
          <td>19.303580</td>
          <td>0.005117</td>
          <td>25.501543</td>
          <td>0.057706</td>
          <td>24.293276</td>
          <td>0.017684</td>
          <td>19.338000</td>
          <td>0.005020</td>
          <td>23.087154</td>
          <td>0.018921</td>
          <td>17.304079</td>
          <td>0.005013</td>
          <td>22.158762</td>
          <td>28.404047</td>
          <td>13.548669</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.165230</td>
          <td>0.021904</td>
          <td>22.345214</td>
          <td>0.006110</td>
          <td>27.459387</td>
          <td>0.274634</td>
          <td>22.358894</td>
          <td>0.007012</td>
          <td>20.934179</td>
          <td>0.005683</td>
          <td>25.963784</td>
          <td>0.478155</td>
          <td>16.760697</td>
          <td>15.999670</td>
          <td>15.716736</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.463963</td>
          <td>0.012578</td>
          <td>22.761484</td>
          <td>0.007077</td>
          <td>18.833617</td>
          <td>0.005006</td>
          <td>17.883674</td>
          <td>0.005004</td>
          <td>21.061934</td>
          <td>0.005838</td>
          <td>19.273775</td>
          <td>0.005209</td>
          <td>26.031616</td>
          <td>22.787321</td>
          <td>24.726977</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.823076</td>
          <td>0.092795</td>
          <td>15.231428</td>
          <td>0.005000</td>
          <td>17.919632</td>
          <td>0.005002</td>
          <td>22.798763</td>
          <td>0.008823</td>
          <td>22.699800</td>
          <td>0.013801</td>
          <td>21.322475</td>
          <td>0.009882</td>
          <td>24.711629</td>
          <td>19.024543</td>
          <td>23.853571</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.077638</td>
          <td>0.020362</td>
          <td>24.705663</td>
          <td>0.028601</td>
          <td>20.992365</td>
          <td>0.005097</td>
          <td>21.988112</td>
          <td>0.006132</td>
          <td>19.710115</td>
          <td>0.005099</td>
          <td>17.703379</td>
          <td>0.005022</td>
          <td>24.610037</td>
          <td>25.832075</td>
          <td>20.697271</td>
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
          <td>20.404888</td>
          <td>0.005474</td>
          <td>16.629120</td>
          <td>0.005001</td>
          <td>27.036117</td>
          <td>0.193409</td>
          <td>14.252567</td>
          <td>0.005000</td>
          <td>21.419330</td>
          <td>0.006472</td>
          <td>22.833647</td>
          <td>0.033667</td>
          <td>21.497059</td>
          <td>22.816504</td>
          <td>23.298171</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.645259</td>
          <td>0.033024</td>
          <td>24.692502</td>
          <td>0.028275</td>
          <td>26.693109</td>
          <td>0.144407</td>
          <td>26.354900</td>
          <td>0.173611</td>
          <td>18.314806</td>
          <td>0.005014</td>
          <td>20.046425</td>
          <td>0.005719</td>
          <td>27.371366</td>
          <td>18.215126</td>
          <td>18.280594</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.524437</td>
          <td>0.007240</td>
          <td>21.341391</td>
          <td>0.005239</td>
          <td>25.907519</td>
          <td>0.072659</td>
          <td>22.917807</td>
          <td>0.009508</td>
          <td>26.550094</td>
          <td>0.370937</td>
          <td>21.577236</td>
          <td>0.011823</td>
          <td>26.200966</td>
          <td>19.890239</td>
          <td>23.935838</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.046399</td>
          <td>0.009497</td>
          <td>27.025913</td>
          <td>0.216397</td>
          <td>19.828804</td>
          <td>0.005019</td>
          <td>23.441726</td>
          <td>0.013888</td>
          <td>23.027496</td>
          <td>0.017999</td>
          <td>22.618420</td>
          <td>0.027869</td>
          <td>23.555913</td>
          <td>19.982033</td>
          <td>26.017810</td>
        </tr>
        <tr>
          <th>999</th>
          <td>31.140018</td>
          <td>3.719761</td>
          <td>23.725052</td>
          <td>0.012711</td>
          <td>22.181249</td>
          <td>0.005637</td>
          <td>21.256464</td>
          <td>0.005349</td>
          <td>23.762061</td>
          <td>0.033939</td>
          <td>25.892656</td>
          <td>0.453360</td>
          <td>18.614158</td>
          <td>21.045989</td>
          <td>18.516025</td>
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
          <td>19.307233</td>
          <td>25.493946</td>
          <td>24.314966</td>
          <td>19.341316</td>
          <td>23.062761</td>
          <td>17.301525</td>
          <td>22.167898</td>
          <td>0.005602</td>
          <td>inf</td>
          <td>inf</td>
          <td>13.544242</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.190851</td>
          <td>22.340852</td>
          <td>27.637128</td>
          <td>22.367582</td>
          <td>20.932581</td>
          <td>25.933584</td>
          <td>16.767405</td>
          <td>0.005000</td>
          <td>15.989225</td>
          <td>0.005000</td>
          <td>15.714517</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.468687</td>
          <td>22.761595</td>
          <td>18.835067</td>
          <td>17.888131</td>
          <td>21.061232</td>
          <td>19.272387</td>
          <td>25.979577</td>
          <td>0.081823</td>
          <td>22.801825</td>
          <td>0.009313</td>
          <td>24.787443</td>
          <td>0.048336</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.732829</td>
          <td>15.231048</td>
          <td>17.918841</td>
          <td>22.797259</td>
          <td>22.722300</td>
          <td>21.326085</td>
          <td>24.726978</td>
          <td>0.026893</td>
          <td>19.016380</td>
          <td>0.005006</td>
          <td>23.855020</td>
          <td>0.021219</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.097255</td>
          <td>24.712366</td>
          <td>20.990974</td>
          <td>21.985040</td>
          <td>19.718199</td>
          <td>17.703882</td>
          <td>24.599327</td>
          <td>0.024049</td>
          <td>25.984288</td>
          <td>0.138882</td>
          <td>20.706217</td>
          <td>0.005129</td>
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
          <td>20.405699</td>
          <td>16.619559</td>
          <td>26.977351</td>
          <td>14.248519</td>
          <td>21.416118</td>
          <td>22.856268</td>
          <td>21.498727</td>
          <td>0.005183</td>
          <td>22.828015</td>
          <td>0.009475</td>
          <td>23.307824</td>
          <td>0.013463</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.621764</td>
          <td>24.708062</td>
          <td>26.558141</td>
          <td>26.321639</td>
          <td>18.317362</td>
          <td>20.038199</td>
          <td>27.545000</td>
          <td>0.309976</td>
          <td>18.204200</td>
          <td>0.005001</td>
          <td>18.284863</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.531048</td>
          <td>21.332732</td>
          <td>25.889963</td>
          <td>22.918798</td>
          <td>26.262848</td>
          <td>21.568089</td>
          <td>26.379282</td>
          <td>0.116245</td>
          <td>19.891519</td>
          <td>0.005029</td>
          <td>23.948292</td>
          <td>0.023004</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.053932</td>
          <td>26.875711</td>
          <td>19.833202</td>
          <td>23.452538</td>
          <td>23.024416</td>
          <td>22.625790</td>
          <td>23.550513</td>
          <td>0.010302</td>
          <td>19.987357</td>
          <td>0.005035</td>
          <td>25.915969</td>
          <td>0.130912</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.196979</td>
          <td>23.737257</td>
          <td>22.187823</td>
          <td>21.258408</td>
          <td>23.784922</td>
          <td>27.265961</td>
          <td>18.618527</td>
          <td>0.005001</td>
          <td>21.041244</td>
          <td>0.005236</td>
          <td>18.511685</td>
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
          <td>19.307233</td>
          <td>25.493946</td>
          <td>24.314966</td>
          <td>19.341316</td>
          <td>23.062761</td>
          <td>17.301525</td>
          <td>22.176551</td>
          <td>0.030685</td>
          <td>inf</td>
          <td>inf</td>
          <td>13.550907</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.190851</td>
          <td>22.340852</td>
          <td>27.637128</td>
          <td>22.367582</td>
          <td>20.932581</td>
          <td>25.933584</td>
          <td>16.758331</td>
          <td>0.005004</td>
          <td>15.994527</td>
          <td>0.005001</td>
          <td>15.717248</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.468687</td>
          <td>22.761595</td>
          <td>18.835067</td>
          <td>17.888131</td>
          <td>21.061232</td>
          <td>19.272387</td>
          <td>24.796537</td>
          <td>0.298142</td>
          <td>22.705833</td>
          <td>0.041112</td>
          <td>24.601096</td>
          <td>0.234202</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.732829</td>
          <td>15.231048</td>
          <td>17.918841</td>
          <td>22.797259</td>
          <td>22.722300</td>
          <td>21.326085</td>
          <td>24.873595</td>
          <td>0.317147</td>
          <td>19.031731</td>
          <td>0.005194</td>
          <td>24.045738</td>
          <td>0.146438</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.097255</td>
          <td>24.712366</td>
          <td>20.990974</td>
          <td>21.985040</td>
          <td>19.718199</td>
          <td>17.703882</td>
          <td>24.066785</td>
          <td>0.162461</td>
          <td>27.061243</td>
          <td>1.234015</td>
          <td>20.705399</td>
          <td>0.008759</td>
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
          <td>20.405699</td>
          <td>16.619559</td>
          <td>26.977351</td>
          <td>14.248519</td>
          <td>21.416118</td>
          <td>22.856268</td>
          <td>21.496232</td>
          <td>0.017029</td>
          <td>22.717546</td>
          <td>0.041543</td>
          <td>23.423185</td>
          <td>0.085037</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.621764</td>
          <td>24.708062</td>
          <td>26.558141</td>
          <td>26.321639</td>
          <td>18.317362</td>
          <td>20.038199</td>
          <td>26.859978</td>
          <td>1.233156</td>
          <td>18.208226</td>
          <td>0.005043</td>
          <td>18.274944</td>
          <td>0.005059</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.531048</td>
          <td>21.332732</td>
          <td>25.889963</td>
          <td>22.918798</td>
          <td>26.262848</td>
          <td>21.568089</td>
          <td>25.449033</td>
          <td>0.494105</td>
          <td>19.889790</td>
          <td>0.005882</td>
          <td>24.024454</td>
          <td>0.143778</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.053932</td>
          <td>26.875711</td>
          <td>19.833202</td>
          <td>23.452538</td>
          <td>23.024416</td>
          <td>22.625790</td>
          <td>23.407173</td>
          <td>0.091575</td>
          <td>19.987815</td>
          <td>0.006041</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.196979</td>
          <td>23.737257</td>
          <td>22.187823</td>
          <td>21.258408</td>
          <td>23.784922</td>
          <td>27.265961</td>
          <td>18.623553</td>
          <td>0.005133</td>
          <td>21.048921</td>
          <td>0.010290</td>
          <td>18.518433</td>
          <td>0.005091</td>
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


