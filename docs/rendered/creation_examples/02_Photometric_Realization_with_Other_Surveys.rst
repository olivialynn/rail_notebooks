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
          <td>21.818062</td>
          <td>26.214486</td>
          <td>20.877917</td>
          <td>26.022165</td>
          <td>20.598507</td>
          <td>17.918139</td>
          <td>24.434907</td>
          <td>19.708667</td>
          <td>20.479575</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.025057</td>
          <td>24.708676</td>
          <td>21.240154</td>
          <td>23.254721</td>
          <td>23.480529</td>
          <td>21.532593</td>
          <td>22.460109</td>
          <td>22.527596</td>
          <td>20.816039</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.408157</td>
          <td>23.296930</td>
          <td>26.453744</td>
          <td>20.752518</td>
          <td>16.955311</td>
          <td>22.260961</td>
          <td>21.733526</td>
          <td>22.703063</td>
          <td>18.403733</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.779853</td>
          <td>21.640651</td>
          <td>17.441402</td>
          <td>21.835802</td>
          <td>19.186013</td>
          <td>24.327900</td>
          <td>19.377728</td>
          <td>20.670458</td>
          <td>24.326921</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.428453</td>
          <td>24.254357</td>
          <td>24.086111</td>
          <td>22.108184</td>
          <td>23.883757</td>
          <td>20.624393</td>
          <td>21.977441</td>
          <td>22.040541</td>
          <td>24.217260</td>
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
          <td>23.794482</td>
          <td>22.972278</td>
          <td>25.691641</td>
          <td>20.811759</td>
          <td>24.208439</td>
          <td>21.465744</td>
          <td>23.148684</td>
          <td>26.406570</td>
          <td>18.771108</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.851897</td>
          <td>17.904474</td>
          <td>22.789905</td>
          <td>23.798372</td>
          <td>24.362355</td>
          <td>23.089135</td>
          <td>26.797185</td>
          <td>25.885375</td>
          <td>24.843920</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.788285</td>
          <td>22.790321</td>
          <td>23.320659</td>
          <td>25.823304</td>
          <td>23.447630</td>
          <td>20.127497</td>
          <td>24.648535</td>
          <td>21.012876</td>
          <td>25.598301</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.330063</td>
          <td>22.719615</td>
          <td>20.301839</td>
          <td>23.923643</td>
          <td>26.409430</td>
          <td>19.619684</td>
          <td>16.393269</td>
          <td>19.098611</td>
          <td>21.635562</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.085473</td>
          <td>20.286310</td>
          <td>19.182225</td>
          <td>23.957908</td>
          <td>23.939193</td>
          <td>21.983657</td>
          <td>21.670301</td>
          <td>25.530125</td>
          <td>19.641244</td>
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
          <td>21.830291</td>
          <td>0.008387</td>
          <td>26.159648</td>
          <td>0.103057</td>
          <td>20.883526</td>
          <td>0.005082</td>
          <td>25.851165</td>
          <td>0.112482</td>
          <td>20.605278</td>
          <td>0.005402</td>
          <td>17.913564</td>
          <td>0.005028</td>
          <td>24.434907</td>
          <td>19.708667</td>
          <td>20.479575</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.010862</td>
          <td>0.019270</td>
          <td>24.697874</td>
          <td>0.028408</td>
          <td>21.244321</td>
          <td>0.005143</td>
          <td>23.241748</td>
          <td>0.011919</td>
          <td>23.454275</td>
          <td>0.025913</td>
          <td>21.537344</td>
          <td>0.011482</td>
          <td>22.460109</td>
          <td>22.527596</td>
          <td>20.816039</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.410845</td>
          <td>0.005043</td>
          <td>23.290502</td>
          <td>0.009390</td>
          <td>26.437455</td>
          <td>0.115735</td>
          <td>20.746288</td>
          <td>0.005155</td>
          <td>16.954842</td>
          <td>0.005003</td>
          <td>22.258396</td>
          <td>0.020426</td>
          <td>21.733526</td>
          <td>22.703063</td>
          <td>18.403733</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.787095</td>
          <td>0.005802</td>
          <td>21.642121</td>
          <td>0.005377</td>
          <td>17.439286</td>
          <td>0.005001</td>
          <td>21.852111</td>
          <td>0.005912</td>
          <td>19.187480</td>
          <td>0.005046</td>
          <td>24.339613</td>
          <td>0.127112</td>
          <td>19.377728</td>
          <td>20.670458</td>
          <td>24.326921</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.481614</td>
          <td>0.163910</td>
          <td>24.232746</td>
          <td>0.019077</td>
          <td>24.085478</td>
          <td>0.014922</td>
          <td>22.109192</td>
          <td>0.006369</td>
          <td>23.856112</td>
          <td>0.036879</td>
          <td>20.616929</td>
          <td>0.006763</td>
          <td>21.977441</td>
          <td>22.040541</td>
          <td>24.217260</td>
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
          <td>23.838774</td>
          <td>0.039094</td>
          <td>22.971480</td>
          <td>0.007817</td>
          <td>25.661697</td>
          <td>0.058437</td>
          <td>20.815690</td>
          <td>0.005173</td>
          <td>24.255761</td>
          <td>0.052564</td>
          <td>21.467387</td>
          <td>0.010918</td>
          <td>23.148684</td>
          <td>26.406570</td>
          <td>18.771108</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.852242</td>
          <td>0.005878</td>
          <td>17.907073</td>
          <td>0.005004</td>
          <td>22.783668</td>
          <td>0.006654</td>
          <td>23.796692</td>
          <td>0.018542</td>
          <td>24.368928</td>
          <td>0.058118</td>
          <td>23.121970</td>
          <td>0.043451</td>
          <td>26.797185</td>
          <td>25.885375</td>
          <td>24.843920</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.793812</td>
          <td>0.005213</td>
          <td>22.797942</td>
          <td>0.007191</td>
          <td>23.337856</td>
          <td>0.008749</td>
          <td>25.697400</td>
          <td>0.098332</td>
          <td>23.439364</td>
          <td>0.025579</td>
          <td>20.122347</td>
          <td>0.005812</td>
          <td>24.648535</td>
          <td>21.012876</td>
          <td>25.598301</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.365018</td>
          <td>0.339201</td>
          <td>22.725171</td>
          <td>0.006969</td>
          <td>20.298221</td>
          <td>0.005035</td>
          <td>23.932198</td>
          <td>0.020793</td>
          <td>26.439052</td>
          <td>0.339971</td>
          <td>19.621619</td>
          <td>0.005364</td>
          <td>16.393269</td>
          <td>19.098611</td>
          <td>21.635562</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.084563</td>
          <td>0.005309</td>
          <td>20.282482</td>
          <td>0.005054</td>
          <td>19.183786</td>
          <td>0.005009</td>
          <td>23.951001</td>
          <td>0.021129</td>
          <td>23.927633</td>
          <td>0.039288</td>
          <td>21.979262</td>
          <td>0.016183</td>
          <td>21.670301</td>
          <td>25.530125</td>
          <td>19.641244</td>
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
          <td>21.818062</td>
          <td>26.214486</td>
          <td>20.877917</td>
          <td>26.022165</td>
          <td>20.598507</td>
          <td>17.918139</td>
          <td>24.446039</td>
          <td>0.021055</td>
          <td>19.703370</td>
          <td>0.005021</td>
          <td>20.479132</td>
          <td>0.005085</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.025057</td>
          <td>24.708676</td>
          <td>21.240154</td>
          <td>23.254721</td>
          <td>23.480529</td>
          <td>21.532593</td>
          <td>22.469067</td>
          <td>0.006009</td>
          <td>22.516498</td>
          <td>0.007845</td>
          <td>20.810996</td>
          <td>0.005156</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.408157</td>
          <td>23.296930</td>
          <td>26.453744</td>
          <td>20.752518</td>
          <td>16.955311</td>
          <td>22.260961</td>
          <td>21.742372</td>
          <td>0.005283</td>
          <td>22.706368</td>
          <td>0.008764</td>
          <td>18.402737</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.779853</td>
          <td>21.640651</td>
          <td>17.441402</td>
          <td>21.835802</td>
          <td>19.186013</td>
          <td>24.327900</td>
          <td>19.379227</td>
          <td>0.005004</td>
          <td>20.667874</td>
          <td>0.005120</td>
          <td>24.307756</td>
          <td>0.031544</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.428453</td>
          <td>24.254357</td>
          <td>24.086111</td>
          <td>22.108184</td>
          <td>23.883757</td>
          <td>20.624393</td>
          <td>21.984174</td>
          <td>0.005436</td>
          <td>22.036743</td>
          <td>0.006334</td>
          <td>24.221471</td>
          <td>0.029227</td>
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
          <td>23.794482</td>
          <td>22.972278</td>
          <td>25.691641</td>
          <td>20.811759</td>
          <td>24.208439</td>
          <td>21.465744</td>
          <td>23.151846</td>
          <td>0.008000</td>
          <td>26.834060</td>
          <td>0.283468</td>
          <td>18.773773</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.851897</td>
          <td>17.904474</td>
          <td>22.789905</td>
          <td>23.798372</td>
          <td>24.362355</td>
          <td>23.089135</td>
          <td>26.693050</td>
          <td>0.152517</td>
          <td>25.722418</td>
          <td>0.110615</td>
          <td>24.886538</td>
          <td>0.052800</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.788285</td>
          <td>22.790321</td>
          <td>23.320659</td>
          <td>25.823304</td>
          <td>23.447630</td>
          <td>20.127497</td>
          <td>24.626551</td>
          <td>0.024627</td>
          <td>21.031429</td>
          <td>0.005232</td>
          <td>25.568198</td>
          <td>0.096627</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.330063</td>
          <td>22.719615</td>
          <td>20.301839</td>
          <td>23.923643</td>
          <td>26.409430</td>
          <td>19.619684</td>
          <td>16.398023</td>
          <td>0.005000</td>
          <td>19.099347</td>
          <td>0.005007</td>
          <td>21.628740</td>
          <td>0.005669</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.085473</td>
          <td>20.286310</td>
          <td>19.182225</td>
          <td>23.957908</td>
          <td>23.939193</td>
          <td>21.983657</td>
          <td>21.672435</td>
          <td>0.005250</td>
          <td>25.679291</td>
          <td>0.106519</td>
          <td>19.635302</td>
          <td>0.005018</td>
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
          <td>21.818062</td>
          <td>26.214486</td>
          <td>20.877917</td>
          <td>26.022165</td>
          <td>20.598507</td>
          <td>17.918139</td>
          <td>24.179088</td>
          <td>0.178768</td>
          <td>19.717127</td>
          <td>0.005655</td>
          <td>20.473026</td>
          <td>0.007664</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.025057</td>
          <td>24.708676</td>
          <td>21.240154</td>
          <td>23.254721</td>
          <td>23.480529</td>
          <td>21.532593</td>
          <td>22.440039</td>
          <td>0.038772</td>
          <td>22.558636</td>
          <td>0.036064</td>
          <td>20.803823</td>
          <td>0.009325</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.408157</td>
          <td>23.296930</td>
          <td>26.453744</td>
          <td>20.752518</td>
          <td>16.955311</td>
          <td>22.260961</td>
          <td>21.748142</td>
          <td>0.021094</td>
          <td>22.733759</td>
          <td>0.042148</td>
          <td>18.396664</td>
          <td>0.005073</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.779853</td>
          <td>21.640651</td>
          <td>17.441402</td>
          <td>21.835802</td>
          <td>19.186013</td>
          <td>24.327900</td>
          <td>19.380842</td>
          <td>0.005517</td>
          <td>20.666623</td>
          <td>0.008067</td>
          <td>24.057110</td>
          <td>0.147878</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.428453</td>
          <td>24.254357</td>
          <td>24.086111</td>
          <td>22.108184</td>
          <td>23.883757</td>
          <td>20.624393</td>
          <td>22.036284</td>
          <td>0.027114</td>
          <td>22.025621</td>
          <td>0.022555</td>
          <td>24.563196</td>
          <td>0.226954</td>
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
          <td>23.794482</td>
          <td>22.972278</td>
          <td>25.691641</td>
          <td>20.811759</td>
          <td>24.208439</td>
          <td>21.465744</td>
          <td>23.162005</td>
          <td>0.073734</td>
          <td>25.320376</td>
          <td>0.385147</td>
          <td>18.769833</td>
          <td>0.005145</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.851897</td>
          <td>17.904474</td>
          <td>22.789905</td>
          <td>23.798372</td>
          <td>24.362355</td>
          <td>23.089135</td>
          <td>25.703222</td>
          <td>0.594066</td>
          <td>28.169112</td>
          <td>2.092471</td>
          <td>25.437799</td>
          <td>0.454749</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.788285</td>
          <td>22.790321</td>
          <td>23.320659</td>
          <td>25.823304</td>
          <td>23.447630</td>
          <td>20.127497</td>
          <td>25.181478</td>
          <td>0.403755</td>
          <td>21.009917</td>
          <td>0.010015</td>
          <td>26.325604</td>
          <td>0.845320</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.330063</td>
          <td>22.719615</td>
          <td>20.301839</td>
          <td>23.923643</td>
          <td>26.409430</td>
          <td>19.619684</td>
          <td>16.388100</td>
          <td>0.005002</td>
          <td>19.100745</td>
          <td>0.005220</td>
          <td>21.627026</td>
          <td>0.017474</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.085473</td>
          <td>20.286310</td>
          <td>19.182225</td>
          <td>23.957908</td>
          <td>23.939193</td>
          <td>21.983657</td>
          <td>21.693896</td>
          <td>0.020133</td>
          <td>27.188580</td>
          <td>1.322091</td>
          <td>19.624525</td>
          <td>0.005664</td>
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


