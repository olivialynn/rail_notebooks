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

    data = np.random.normal(23, 3, size = (1000,10))
    
    data_df = pd.DataFrame(data=data,    # values
                columns=['u', 'g', 'r', 'i', 'z', 'y', 'Y', 'J', 'H', 'F'])
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
          <th>F</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
          <td>22.120187</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
          <td>24.984402</td>
          <td>24.519918</td>
          <td>26.603928</td>
          <td>28.264883</td>
          <td>24.168218</td>
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
        </tr>
        <tr>
          <th>995</th>
          <td>22.498273</td>
          <td>19.345848</td>
          <td>30.636966</td>
          <td>17.298766</td>
          <td>25.278388</td>
          <td>21.281571</td>
          <td>23.401882</td>
          <td>20.084787</td>
          <td>17.316507</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.691973</td>
          <td>25.383890</td>
          <td>22.249757</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262651</td>
          <td>20.081021</td>
          <td>23.758420</td>
          <td>19.030635</td>
          <td>23.078506</td>
          <td>25.476684</td>
          <td>20.307481</td>
          <td>25.088338</td>
          <td>25.547408</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.510375</td>
          <td>18.692511</td>
          <td>19.689822</td>
          <td>23.657409</td>
          <td>22.711747</td>
          <td>22.648712</td>
          <td>23.722045</td>
          <td>21.819881</td>
          <td>20.238721</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>25.834554</td>
          <td>20.236373</td>
          <td>24.318103</td>
          <td>22.891500</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 10 columns</p>
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
          <th>F</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.586901</td>
          <td>0.013779</td>
          <td>22.971329</td>
          <td>0.007816</td>
          <td>25.821243</td>
          <td>0.067317</td>
          <td>18.275745</td>
          <td>0.005006</td>
          <td>23.014828</td>
          <td>0.017810</td>
          <td>23.575459</td>
          <td>0.064977</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.335639</td>
          <td>0.060490</td>
          <td>20.923165</td>
          <td>0.005130</td>
          <td>24.358093</td>
          <td>0.018667</td>
          <td>22.926871</td>
          <td>0.009564</td>
          <td>26.317936</td>
          <td>0.308738</td>
          <td>23.248818</td>
          <td>0.048628</td>
          <td>24.175778</td>
          <td>24.107270</td>
          <td>22.120187</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001457</td>
          <td>0.009242</td>
          <td>25.345579</td>
          <td>0.050260</td>
          <td>22.978983</td>
          <td>0.007226</td>
          <td>23.450554</td>
          <td>0.013985</td>
          <td>24.747995</td>
          <td>0.081288</td>
          <td>26.964466</td>
          <td>0.946185</td>
          <td>24.365248</td>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.627354</td>
          <td>0.005173</td>
          <td>18.620499</td>
          <td>0.005008</td>
          <td>22.601334</td>
          <td>0.006246</td>
          <td>25.264598</td>
          <td>0.067132</td>
          <td>22.210158</td>
          <td>0.009676</td>
          <td>23.041737</td>
          <td>0.040468</td>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.658593</td>
          <td>0.425923</td>
          <td>22.876168</td>
          <td>0.007456</td>
          <td>23.049111</td>
          <td>0.007471</td>
          <td>22.879821</td>
          <td>0.009279</td>
          <td>16.978398</td>
          <td>0.005003</td>
          <td>24.813351</td>
          <td>0.190654</td>
          <td>24.519918</td>
          <td>26.603928</td>
          <td>28.264883</td>
          <td>24.168218</td>
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
          <td>...</td>
        </tr>
        <tr>
          <th>995</th>
          <td>22.476477</td>
          <td>0.012693</td>
          <td>19.354023</td>
          <td>0.005018</td>
          <td>30.253551</td>
          <td>1.696534</td>
          <td>17.305162</td>
          <td>0.005002</td>
          <td>25.675261</td>
          <td>0.181636</td>
          <td>21.287228</td>
          <td>0.009654</td>
          <td>23.401882</td>
          <td>20.084787</td>
          <td>17.316507</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.145384</td>
          <td>0.021544</td>
          <td>23.227788</td>
          <td>0.009031</td>
          <td>23.947024</td>
          <td>0.013378</td>
          <td>25.696259</td>
          <td>0.098233</td>
          <td>24.586882</td>
          <td>0.070502</td>
          <td>24.254042</td>
          <td>0.118011</td>
          <td>21.691973</td>
          <td>25.383890</td>
          <td>22.249757</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.266927</td>
          <td>0.005014</td>
          <td>20.079713</td>
          <td>0.005042</td>
          <td>23.779829</td>
          <td>0.011786</td>
          <td>19.030976</td>
          <td>0.005014</td>
          <td>23.077821</td>
          <td>0.018773</td>
          <td>25.765709</td>
          <td>0.411694</td>
          <td>20.307481</td>
          <td>25.088338</td>
          <td>25.547408</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.510056</td>
          <td>0.005150</td>
          <td>18.688963</td>
          <td>0.005009</td>
          <td>19.679008</td>
          <td>0.005016</td>
          <td>23.640454</td>
          <td>0.016290</td>
          <td>22.697729</td>
          <td>0.013779</td>
          <td>22.685665</td>
          <td>0.029559</td>
          <td>23.722045</td>
          <td>21.819881</td>
          <td>20.238721</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.819414</td>
          <td>0.005220</td>
          <td>24.532799</td>
          <td>0.024620</td>
          <td>24.447567</td>
          <td>0.020131</td>
          <td>23.454425</td>
          <td>0.014028</td>
          <td>21.240289</td>
          <td>0.006112</td>
          <td>16.846887</td>
          <td>0.005008</td>
          <td>25.834554</td>
          <td>20.236373</td>
          <td>24.318103</td>
          <td>22.891500</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 16 columns</p>
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




.. image:: ../../../docs/rendered/creation_examples/photerr_demo_files/../../../docs/rendered/creation_examples/photerr_demo_8_0.png


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
          <th>F</th>
          <th>F_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.976586</td>
          <td>0.033529</td>
          <td>17.290152</td>
          <td>0.005000</td>
          <td>25.775700</td>
          <td>0.074634</td>
          <td>22.681741</td>
          <td>0.009499</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.172581</td>
          <td>0.016696</td>
          <td>24.107156</td>
          <td>0.016483</td>
          <td>22.119334</td>
          <td>0.005658</td>
          <td>23.423254</td>
          <td>0.016706</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.352102</td>
          <td>0.019426</td>
          <td>22.986838</td>
          <td>0.007521</td>
          <td>22.929880</td>
          <td>0.007494</td>
          <td>27.357700</td>
          <td>0.479176</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
          <td>19.547458</td>
          <td>0.005005</td>
          <td>22.305423</td>
          <td>0.005832</td>
          <td>24.278711</td>
          <td>0.019873</td>
          <td>12.207847</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
          <td>24.984402</td>
          <td>24.522994</td>
          <td>0.022504</td>
          <td>26.887109</td>
          <td>0.187773</td>
          <td>28.283431</td>
          <td>0.585770</td>
          <td>24.239086</td>
          <td>0.033903</td>
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
        </tr>
        <tr>
          <th>995</th>
          <td>22.498273</td>
          <td>19.345848</td>
          <td>30.636966</td>
          <td>17.298766</td>
          <td>25.278388</td>
          <td>21.281571</td>
          <td>23.396595</td>
          <td>0.009282</td>
          <td>20.079056</td>
          <td>0.005015</td>
          <td>17.319728</td>
          <td>0.005000</td>
          <td>20.854871</td>
          <td>0.005221</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.697644</td>
          <td>0.005261</td>
          <td>25.271109</td>
          <td>0.045560</td>
          <td>22.247279</td>
          <td>0.005820</td>
          <td>20.800299</td>
          <td>0.005200</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262651</td>
          <td>20.081021</td>
          <td>23.758420</td>
          <td>19.030635</td>
          <td>23.078506</td>
          <td>25.476684</td>
          <td>20.311631</td>
          <td>0.005021</td>
          <td>25.175031</td>
          <td>0.041821</td>
          <td>25.535396</td>
          <td>0.060287</td>
          <td>18.305701</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.510375</td>
          <td>18.692511</td>
          <td>19.689822</td>
          <td>23.657409</td>
          <td>22.711747</td>
          <td>22.648712</td>
          <td>23.717307</td>
          <td>0.011627</td>
          <td>21.822666</td>
          <td>0.005358</td>
          <td>20.244480</td>
          <td>0.005022</td>
          <td>25.237117</td>
          <td>0.082370</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>25.736205</td>
          <td>0.065942</td>
          <td>20.237178</td>
          <td>0.005020</td>
          <td>24.333927</td>
          <td>0.020837</td>
          <td>22.887043</td>
          <td>0.010959</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 14 columns</p>
    </div>



.. code:: ipython3

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    
    for band in "YJHF":
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




.. image:: ../../../docs/rendered/creation_examples/photerr_demo_files/../../../docs/rendered/creation_examples/photerr_demo_14_0.png


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
          <th>F</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.548843</td>
          <td>0.310931</td>
          <td>17.296313</td>
          <td>0.005014</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.037784</td>
          <td>0.204389</td>
          <td>23.969376</td>
          <td>0.162821</td>
          <td>22.109583</td>
          <td>0.041250</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.122522</td>
          <td>0.219400</td>
          <td>22.997457</td>
          <td>0.069630</td>
          <td>22.831414</td>
          <td>0.078410</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
          <td>19.545734</td>
          <td>0.006147</td>
          <td>22.292177</td>
          <td>0.037156</td>
          <td>23.945844</td>
          <td>0.205775</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
          <td>24.984402</td>
          <td>24.891411</td>
          <td>0.406849</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.363687</td>
          <td>1.919472</td>
          <td>24.168218</td>
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
        </tr>
        <tr>
          <th>995</th>
          <td>22.498273</td>
          <td>19.345848</td>
          <td>30.636966</td>
          <td>17.298766</td>
          <td>25.278388</td>
          <td>21.281571</td>
          <td>23.133012</td>
          <td>0.093682</td>
          <td>20.064033</td>
          <td>0.006926</td>
          <td>17.312639</td>
          <td>0.005025</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.672482</td>
          <td>0.025637</td>
          <td>25.576533</td>
          <td>0.582899</td>
          <td>22.221237</td>
          <td>0.045566</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262651</td>
          <td>20.081021</td>
          <td>23.758420</td>
          <td>19.030635</td>
          <td>23.078506</td>
          <td>25.476684</td>
          <td>20.297960</td>
          <td>0.008718</td>
          <td>24.851944</td>
          <td>0.337531</td>
          <td>25.505897</td>
          <td>0.684122</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.510375</td>
          <td>18.692511</td>
          <td>19.689822</td>
          <td>23.657409</td>
          <td>22.711747</td>
          <td>22.648712</td>
          <td>23.783619</td>
          <td>0.164814</td>
          <td>21.821403</td>
          <td>0.024517</td>
          <td>20.245272</td>
          <td>0.008980</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.232311</td>
          <td>0.007503</td>
          <td>24.395379</td>
          <td>0.297864</td>
          <td>22.891500</td>
        </tr>
      </tbody>
    </table>
    <p>1000 rows × 13 columns</p>
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




.. image:: ../../../docs/rendered/creation_examples/photerr_demo_files/../../../docs/rendered/creation_examples/photerr_demo_17_0.png


