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
          <td>23.977872</td>
          <td>16.310388</td>
          <td>21.080921</td>
          <td>27.938732</td>
          <td>21.255207</td>
          <td>24.895981</td>
          <td>19.218116</td>
          <td>18.509263</td>
          <td>27.975693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.466232</td>
          <td>23.738694</td>
          <td>24.758151</td>
          <td>26.371748</td>
          <td>20.897686</td>
          <td>19.496666</td>
          <td>23.784648</td>
          <td>24.482567</td>
          <td>25.630613</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.665360</td>
          <td>22.480938</td>
          <td>26.582017</td>
          <td>22.216247</td>
          <td>23.062255</td>
          <td>27.708366</td>
          <td>24.089010</td>
          <td>28.682445</td>
          <td>24.720997</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.005719</td>
          <td>24.032934</td>
          <td>24.894927</td>
          <td>26.190616</td>
          <td>26.914225</td>
          <td>19.771218</td>
          <td>23.354148</td>
          <td>23.038325</td>
          <td>18.042702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.975389</td>
          <td>23.025205</td>
          <td>17.883507</td>
          <td>19.087399</td>
          <td>20.995780</td>
          <td>21.653675</td>
          <td>27.378912</td>
          <td>20.679977</td>
          <td>28.812849</td>
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
          <td>21.895871</td>
          <td>23.299155</td>
          <td>28.299937</td>
          <td>24.920971</td>
          <td>17.644815</td>
          <td>25.639419</td>
          <td>27.156866</td>
          <td>22.045665</td>
          <td>24.153487</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.148951</td>
          <td>20.978019</td>
          <td>23.508698</td>
          <td>24.128549</td>
          <td>21.936221</td>
          <td>26.424329</td>
          <td>19.309733</td>
          <td>25.065355</td>
          <td>27.483269</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.401172</td>
          <td>21.587285</td>
          <td>20.740098</td>
          <td>18.615972</td>
          <td>23.801548</td>
          <td>22.004180</td>
          <td>25.963404</td>
          <td>23.983987</td>
          <td>23.704755</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.162943</td>
          <td>20.481040</td>
          <td>20.464589</td>
          <td>20.691261</td>
          <td>24.417215</td>
          <td>18.350584</td>
          <td>23.356276</td>
          <td>22.331694</td>
          <td>22.696124</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.268326</td>
          <td>18.403384</td>
          <td>21.563581</td>
          <td>22.513245</td>
          <td>23.146299</td>
          <td>24.925948</td>
          <td>18.830450</td>
          <td>20.827172</td>
          <td>23.192361</td>
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
          <td>23.936444</td>
          <td>0.042586</td>
          <td>16.307021</td>
          <td>0.005001</td>
          <td>21.085290</td>
          <td>0.005112</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.247426</td>
          <td>0.006125</td>
          <td>25.000020</td>
          <td>0.222920</td>
          <td>19.218116</td>
          <td>18.509263</td>
          <td>27.975693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.473130</td>
          <td>0.012662</td>
          <td>23.740685</td>
          <td>0.012862</td>
          <td>24.762376</td>
          <td>0.026397</td>
          <td>26.189519</td>
          <td>0.150745</td>
          <td>20.894096</td>
          <td>0.005640</td>
          <td>19.496842</td>
          <td>0.005298</td>
          <td>23.784648</td>
          <td>24.482567</td>
          <td>25.630613</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.665517</td>
          <td>0.005020</td>
          <td>22.475739</td>
          <td>0.006355</td>
          <td>26.590575</td>
          <td>0.132183</td>
          <td>22.212438</td>
          <td>0.006608</td>
          <td>23.080029</td>
          <td>0.018808</td>
          <td>28.366322</td>
          <td>1.955536</td>
          <td>24.089010</td>
          <td>28.682445</td>
          <td>24.720997</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.045859</td>
          <td>0.019834</td>
          <td>24.020100</td>
          <td>0.016014</td>
          <td>24.912478</td>
          <td>0.030101</td>
          <td>26.195630</td>
          <td>0.151537</td>
          <td>26.205695</td>
          <td>0.282041</td>
          <td>19.774050</td>
          <td>0.005464</td>
          <td>23.354148</td>
          <td>23.038325</td>
          <td>18.042702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.969828</td>
          <td>0.009071</td>
          <td>23.020094</td>
          <td>0.008019</td>
          <td>17.885803</td>
          <td>0.005002</td>
          <td>19.090552</td>
          <td>0.005015</td>
          <td>20.994048</td>
          <td>0.005752</td>
          <td>21.655074</td>
          <td>0.012531</td>
          <td>27.378912</td>
          <td>20.679977</td>
          <td>28.812849</td>
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
          <td>21.884372</td>
          <td>0.008639</td>
          <td>23.288853</td>
          <td>0.009380</td>
          <td>28.362691</td>
          <td>0.550952</td>
          <td>24.912672</td>
          <td>0.049125</td>
          <td>17.642649</td>
          <td>0.005007</td>
          <td>25.241669</td>
          <td>0.271968</td>
          <td>27.156866</td>
          <td>22.045665</td>
          <td>24.153487</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.151041</td>
          <td>0.005033</td>
          <td>20.979757</td>
          <td>0.005141</td>
          <td>23.505558</td>
          <td>0.009726</td>
          <td>24.121287</td>
          <td>0.024462</td>
          <td>21.922822</td>
          <td>0.008129</td>
          <td>27.514559</td>
          <td>1.298776</td>
          <td>19.309733</td>
          <td>25.065355</td>
          <td>27.483269</td>
        </tr>
        <tr>
          <th>997</th>
          <td>inf</td>
          <td>inf</td>
          <td>21.581359</td>
          <td>0.005344</td>
          <td>20.741558</td>
          <td>0.005067</td>
          <td>18.610183</td>
          <td>0.005008</td>
          <td>23.808807</td>
          <td>0.035368</td>
          <td>22.041064</td>
          <td>0.017025</td>
          <td>25.963404</td>
          <td>23.983987</td>
          <td>23.704755</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.105106</td>
          <td>0.020832</td>
          <td>20.481858</td>
          <td>0.005071</td>
          <td>20.466905</td>
          <td>0.005045</td>
          <td>20.689865</td>
          <td>0.005142</td>
          <td>24.528426</td>
          <td>0.066946</td>
          <td>18.349873</td>
          <td>0.005052</td>
          <td>23.356276</td>
          <td>22.331694</td>
          <td>22.696124</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.267468</td>
          <td>0.006569</td>
          <td>18.400261</td>
          <td>0.005006</td>
          <td>21.563191</td>
          <td>0.005236</td>
          <td>22.513897</td>
          <td>0.007537</td>
          <td>23.127985</td>
          <td>0.019583</td>
          <td>24.620623</td>
          <td>0.161884</td>
          <td>18.830450</td>
          <td>20.827172</td>
          <td>23.192361</td>
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
          <td>23.977872</td>
          <td>16.310388</td>
          <td>21.080921</td>
          <td>27.938732</td>
          <td>21.255207</td>
          <td>24.895981</td>
          <td>19.221659</td>
          <td>0.005003</td>
          <td>18.504809</td>
          <td>0.005002</td>
          <td>28.002777</td>
          <td>0.682665</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.466232</td>
          <td>23.738694</td>
          <td>24.758151</td>
          <td>26.371748</td>
          <td>20.897686</td>
          <td>19.496666</td>
          <td>23.783313</td>
          <td>0.012221</td>
          <td>24.533881</td>
          <td>0.038560</td>
          <td>25.602745</td>
          <td>0.099605</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.665360</td>
          <td>22.480938</td>
          <td>26.582017</td>
          <td>22.216247</td>
          <td>23.062255</td>
          <td>27.708366</td>
          <td>24.099517</td>
          <td>0.015716</td>
          <td>27.395870</td>
          <td>0.440584</td>
          <td>24.690673</td>
          <td>0.044341</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.005719</td>
          <td>24.032934</td>
          <td>24.894927</td>
          <td>26.190616</td>
          <td>26.914225</td>
          <td>19.771218</td>
          <td>23.364335</td>
          <td>0.009090</td>
          <td>23.049992</td>
          <td>0.011062</td>
          <td>18.033473</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.975389</td>
          <td>23.025205</td>
          <td>17.883507</td>
          <td>19.087399</td>
          <td>20.995780</td>
          <td>21.653675</td>
          <td>27.587391</td>
          <td>0.320657</td>
          <td>20.676702</td>
          <td>0.005122</td>
          <td>31.180676</td>
          <td>3.097737</td>
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
          <td>21.895871</td>
          <td>23.299155</td>
          <td>28.299937</td>
          <td>24.920971</td>
          <td>17.644815</td>
          <td>25.639419</td>
          <td>26.888888</td>
          <td>0.180261</td>
          <td>22.039875</td>
          <td>0.006341</td>
          <td>24.165512</td>
          <td>0.027820</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.148951</td>
          <td>20.978019</td>
          <td>23.508698</td>
          <td>24.128549</td>
          <td>21.936221</td>
          <td>26.424329</td>
          <td>19.304131</td>
          <td>0.005003</td>
          <td>24.969170</td>
          <td>0.056835</td>
          <td>27.989886</td>
          <td>0.676667</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.401172</td>
          <td>21.587285</td>
          <td>20.740098</td>
          <td>18.615972</td>
          <td>23.801548</td>
          <td>22.004180</td>
          <td>25.887323</td>
          <td>0.075407</td>
          <td>23.994989</td>
          <td>0.023958</td>
          <td>23.726393</td>
          <td>0.019005</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.162943</td>
          <td>20.481040</td>
          <td>20.464589</td>
          <td>20.691261</td>
          <td>24.417215</td>
          <td>18.350584</td>
          <td>23.345487</td>
          <td>0.008981</td>
          <td>22.332771</td>
          <td>0.007146</td>
          <td>22.691231</td>
          <td>0.008682</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.268326</td>
          <td>18.403384</td>
          <td>21.563581</td>
          <td>22.513245</td>
          <td>23.146299</td>
          <td>24.925948</td>
          <td>18.830297</td>
          <td>0.005001</td>
          <td>20.824428</td>
          <td>0.005160</td>
          <td>23.200227</td>
          <td>0.012381</td>
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
          <td>23.977872</td>
          <td>16.310388</td>
          <td>21.080921</td>
          <td>27.938732</td>
          <td>21.255207</td>
          <td>24.895981</td>
          <td>19.221939</td>
          <td>0.005390</td>
          <td>18.519197</td>
          <td>0.005076</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.466232</td>
          <td>23.738694</td>
          <td>24.758151</td>
          <td>26.371748</td>
          <td>20.897686</td>
          <td>19.496666</td>
          <td>23.660803</td>
          <td>0.114386</td>
          <td>24.955473</td>
          <td>0.288425</td>
          <td>26.147015</td>
          <td>0.752376</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.665360</td>
          <td>22.480938</td>
          <td>26.582017</td>
          <td>22.216247</td>
          <td>23.062255</td>
          <td>27.708366</td>
          <td>24.041147</td>
          <td>0.158937</td>
          <td>25.260075</td>
          <td>0.367489</td>
          <td>25.013665</td>
          <td>0.327436</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.005719</td>
          <td>24.032934</td>
          <td>24.894927</td>
          <td>26.190616</td>
          <td>26.914225</td>
          <td>19.771218</td>
          <td>23.428590</td>
          <td>0.093319</td>
          <td>22.987379</td>
          <td>0.052840</td>
          <td>18.049918</td>
          <td>0.005039</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.975389</td>
          <td>23.025205</td>
          <td>17.883507</td>
          <td>19.087399</td>
          <td>20.995780</td>
          <td>21.653675</td>
          <td>26.192705</td>
          <td>0.827648</td>
          <td>20.675276</td>
          <td>0.008106</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>21.895871</td>
          <td>23.299155</td>
          <td>28.299937</td>
          <td>24.920971</td>
          <td>17.644815</td>
          <td>25.639419</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.053172</td>
          <td>0.023101</td>
          <td>23.945854</td>
          <td>0.134344</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.148951</td>
          <td>20.978019</td>
          <td>23.508698</td>
          <td>24.128549</td>
          <td>21.936221</td>
          <td>26.424329</td>
          <td>19.306979</td>
          <td>0.005454</td>
          <td>24.563868</td>
          <td>0.208907</td>
          <td>26.101335</td>
          <td>0.729781</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.401172</td>
          <td>21.587285</td>
          <td>20.740098</td>
          <td>18.615972</td>
          <td>23.801548</td>
          <td>22.004180</td>
          <td>26.220457</td>
          <td>0.842539</td>
          <td>23.788413</td>
          <td>0.107373</td>
          <td>24.012279</td>
          <td>0.142277</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.162943</td>
          <td>20.481040</td>
          <td>20.464589</td>
          <td>20.691261</td>
          <td>24.417215</td>
          <td>18.350584</td>
          <td>23.374384</td>
          <td>0.088967</td>
          <td>22.384989</td>
          <td>0.030915</td>
          <td>22.736373</td>
          <td>0.046185</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.268326</td>
          <td>18.403384</td>
          <td>21.563581</td>
          <td>22.513245</td>
          <td>23.146299</td>
          <td>24.925948</td>
          <td>18.831218</td>
          <td>0.005194</td>
          <td>20.829286</td>
          <td>0.008890</td>
          <td>23.232447</td>
          <td>0.071826</td>
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


