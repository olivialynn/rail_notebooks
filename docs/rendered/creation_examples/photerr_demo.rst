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
          <td>22.601010</td>
          <td>0.013927</td>
          <td>22.959980</td>
          <td>0.007771</td>
          <td>25.850436</td>
          <td>0.069080</td>
          <td>18.269848</td>
          <td>0.005006</td>
          <td>23.024590</td>
          <td>0.017956</td>
          <td>23.625865</td>
          <td>0.067945</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.474259</td>
          <td>0.068336</td>
          <td>20.928406</td>
          <td>0.005131</td>
          <td>24.333413</td>
          <td>0.018286</td>
          <td>22.923185</td>
          <td>0.009541</td>
          <td>26.270286</td>
          <td>0.297146</td>
          <td>23.261546</td>
          <td>0.049181</td>
          <td>24.175778</td>
          <td>24.107270</td>
          <td>22.120187</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.024959</td>
          <td>0.009374</td>
          <td>25.379651</td>
          <td>0.051800</td>
          <td>22.957386</td>
          <td>0.007155</td>
          <td>23.449288</td>
          <td>0.013971</td>
          <td>24.798334</td>
          <td>0.084977</td>
          <td>25.121935</td>
          <td>0.246578</td>
          <td>24.365248</td>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.618972</td>
          <td>0.005171</td>
          <td>18.627895</td>
          <td>0.005008</td>
          <td>22.589914</td>
          <td>0.006223</td>
          <td>25.137739</td>
          <td>0.059990</td>
          <td>22.228019</td>
          <td>0.009791</td>
          <td>23.035807</td>
          <td>0.040256</td>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.562855</td>
          <td>0.807565</td>
          <td>22.859790</td>
          <td>0.007398</td>
          <td>23.058871</td>
          <td>0.007507</td>
          <td>22.873340</td>
          <td>0.009240</td>
          <td>16.975438</td>
          <td>0.005003</td>
          <td>25.387360</td>
          <td>0.305947</td>
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
          <td>22.499403</td>
          <td>0.012909</td>
          <td>19.349510</td>
          <td>0.005018</td>
          <td>29.649925</td>
          <td>1.250199</td>
          <td>17.296801</td>
          <td>0.005002</td>
          <td>25.415039</td>
          <td>0.145460</td>
          <td>21.293185</td>
          <td>0.009692</td>
          <td>23.401882</td>
          <td>20.084787</td>
          <td>17.316507</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.139511</td>
          <td>0.021438</td>
          <td>23.242490</td>
          <td>0.009113</td>
          <td>23.961182</td>
          <td>0.013526</td>
          <td>25.871255</td>
          <td>0.114469</td>
          <td>24.619571</td>
          <td>0.072571</td>
          <td>23.995580</td>
          <td>0.094146</td>
          <td>21.691973</td>
          <td>25.383890</td>
          <td>22.249757</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262490</td>
          <td>0.005013</td>
          <td>20.073593</td>
          <td>0.005042</td>
          <td>23.784156</td>
          <td>0.011824</td>
          <td>19.035200</td>
          <td>0.005014</td>
          <td>23.111686</td>
          <td>0.019315</td>
          <td>25.556837</td>
          <td>0.350045</td>
          <td>20.307481</td>
          <td>25.088338</td>
          <td>25.547408</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.516224</td>
          <td>0.005151</td>
          <td>18.696338</td>
          <td>0.005009</td>
          <td>19.695467</td>
          <td>0.005016</td>
          <td>23.640520</td>
          <td>0.016291</td>
          <td>22.721392</td>
          <td>0.014037</td>
          <td>22.691888</td>
          <td>0.029720</td>
          <td>23.722045</td>
          <td>21.819881</td>
          <td>20.238721</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.834513</td>
          <td>0.005224</td>
          <td>24.470256</td>
          <td>0.023331</td>
          <td>24.483447</td>
          <td>0.020755</td>
          <td>23.459068</td>
          <td>0.014079</td>
          <td>21.241012</td>
          <td>0.006114</td>
          <td>16.854731</td>
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
          <td>24.949406</td>
          <td>0.032730</td>
          <td>17.299702</td>
          <td>0.005000</td>
          <td>25.728716</td>
          <td>0.071588</td>
          <td>22.689314</td>
          <td>0.009547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.169099</td>
          <td>0.016648</td>
          <td>24.085328</td>
          <td>0.016188</td>
          <td>22.119910</td>
          <td>0.005658</td>
          <td>23.400646</td>
          <td>0.016394</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.337095</td>
          <td>0.019179</td>
          <td>22.996809</td>
          <td>0.007560</td>
          <td>22.921933</td>
          <td>0.007464</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
          <td>19.547547</td>
          <td>0.005005</td>
          <td>22.309841</td>
          <td>0.005838</td>
          <td>24.284165</td>
          <td>0.019966</td>
          <td>12.209135</td>
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
          <td>24.524527</td>
          <td>0.022534</td>
          <td>26.501129</td>
          <td>0.134958</td>
          <td>27.846872</td>
          <td>0.424486</td>
          <td>24.140382</td>
          <td>0.031063</td>
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
          <td>23.399390</td>
          <td>0.009298</td>
          <td>20.085506</td>
          <td>0.005015</td>
          <td>17.319646</td>
          <td>0.005000</td>
          <td>20.848216</td>
          <td>0.005219</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.686446</td>
          <td>0.005256</td>
          <td>25.365397</td>
          <td>0.049556</td>
          <td>22.258909</td>
          <td>0.005837</td>
          <td>20.794542</td>
          <td>0.005198</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262651</td>
          <td>20.081021</td>
          <td>23.758420</td>
          <td>19.030635</td>
          <td>23.078506</td>
          <td>25.476684</td>
          <td>20.315516</td>
          <td>0.005021</td>
          <td>25.030711</td>
          <td>0.036779</td>
          <td>25.491446</td>
          <td>0.057974</td>
          <td>18.308012</td>
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
          <td>23.735676</td>
          <td>0.011788</td>
          <td>21.824371</td>
          <td>0.005359</td>
          <td>20.242128</td>
          <td>0.005022</td>
          <td>25.164580</td>
          <td>0.077250</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>25.744528</td>
          <td>0.066432</td>
          <td>20.237528</td>
          <td>0.005020</td>
          <td>24.340557</td>
          <td>0.020956</td>
          <td>22.900235</td>
          <td>0.011064</td>
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
          <td>24.970916</td>
          <td>0.432324</td>
          <td>17.294850</td>
          <td>0.005014</td>
          <td>25.404126</td>
          <td>0.637734</td>
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
          <td>24.237396</td>
          <td>0.241338</td>
          <td>24.133325</td>
          <td>0.187173</td>
          <td>22.119530</td>
          <td>0.041617</td>
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
          <td>24.694024</td>
          <td>0.348933</td>
          <td>22.924076</td>
          <td>0.065235</td>
          <td>22.784871</td>
          <td>0.075243</td>
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
          <td>19.542788</td>
          <td>0.006141</td>
          <td>22.333747</td>
          <td>0.038556</td>
          <td>24.271070</td>
          <td>0.269317</td>
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
          <td>24.336820</td>
          <td>0.261887</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.851457</td>
          <td>0.859381</td>
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
          <td>23.396866</td>
          <td>0.118040</td>
          <td>20.085270</td>
          <td>0.006992</td>
          <td>17.319595</td>
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
          <td>21.655184</td>
          <td>0.025252</td>
          <td>25.050729</td>
          <td>0.394299</td>
          <td>22.209493</td>
          <td>0.045091</td>
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
          <td>20.300947</td>
          <td>0.008734</td>
          <td>24.720851</td>
          <td>0.304029</td>
          <td>24.815443</td>
          <td>0.414416</td>
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
          <td>23.899337</td>
          <td>0.181865</td>
          <td>21.819653</td>
          <td>0.024479</td>
          <td>20.231277</td>
          <td>0.008901</td>
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
          <td>20.231574</td>
          <td>0.007501</td>
          <td>24.315885</td>
          <td>0.279319</td>
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


