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
          <td>22.595272</td>
          <td>0.013866</td>
          <td>22.974050</td>
          <td>0.007827</td>
          <td>25.791520</td>
          <td>0.065567</td>
          <td>18.271440</td>
          <td>0.005006</td>
          <td>23.017147</td>
          <td>0.017845</td>
          <td>23.694173</td>
          <td>0.072179</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.392889</td>
          <td>0.063616</td>
          <td>20.939220</td>
          <td>0.005133</td>
          <td>24.369897</td>
          <td>0.018853</td>
          <td>22.927068</td>
          <td>0.009565</td>
          <td>26.564460</td>
          <td>0.375111</td>
          <td>23.146242</td>
          <td>0.044397</td>
          <td>24.175778</td>
          <td>24.107270</td>
          <td>22.120187</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.006608</td>
          <td>0.009271</td>
          <td>25.317225</td>
          <td>0.049014</td>
          <td>22.959189</td>
          <td>0.007161</td>
          <td>23.448650</td>
          <td>0.013964</td>
          <td>24.795987</td>
          <td>0.084801</td>
          <td>26.100551</td>
          <td>0.528851</td>
          <td>24.365248</td>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.618448</td>
          <td>0.005171</td>
          <td>18.617854</td>
          <td>0.005008</td>
          <td>22.601072</td>
          <td>0.006245</td>
          <td>25.199008</td>
          <td>0.063341</td>
          <td>22.203132</td>
          <td>0.009631</td>
          <td>23.071147</td>
          <td>0.041537</td>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.253326</td>
          <td>1.222954</td>
          <td>22.875577</td>
          <td>0.007454</td>
          <td>23.070605</td>
          <td>0.007551</td>
          <td>22.877660</td>
          <td>0.009266</td>
          <td>16.971608</td>
          <td>0.005003</td>
          <td>25.324070</td>
          <td>0.290756</td>
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
          <td>22.489940</td>
          <td>0.012819</td>
          <td>19.342156</td>
          <td>0.005017</td>
          <td>31.222573</td>
          <td>2.522109</td>
          <td>17.306576</td>
          <td>0.005002</td>
          <td>25.379839</td>
          <td>0.141119</td>
          <td>21.294662</td>
          <td>0.009701</td>
          <td>23.401882</td>
          <td>20.084787</td>
          <td>17.316507</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.111752</td>
          <td>0.020948</td>
          <td>23.238214</td>
          <td>0.009089</td>
          <td>23.956855</td>
          <td>0.013480</td>
          <td>25.851494</td>
          <td>0.112515</td>
          <td>24.782489</td>
          <td>0.083798</td>
          <td>24.128661</td>
          <td>0.105789</td>
          <td>21.691973</td>
          <td>25.383890</td>
          <td>22.249757</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.264425</td>
          <td>0.005013</td>
          <td>20.076155</td>
          <td>0.005042</td>
          <td>23.776157</td>
          <td>0.011754</td>
          <td>19.031934</td>
          <td>0.005014</td>
          <td>23.083402</td>
          <td>0.018861</td>
          <td>25.487004</td>
          <td>0.331257</td>
          <td>20.307481</td>
          <td>25.088338</td>
          <td>25.547408</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.514052</td>
          <td>0.005150</td>
          <td>18.695799</td>
          <td>0.005009</td>
          <td>19.684690</td>
          <td>0.005016</td>
          <td>23.683136</td>
          <td>0.016871</td>
          <td>22.704854</td>
          <td>0.013856</td>
          <td>22.625296</td>
          <td>0.028037</td>
          <td>23.722045</td>
          <td>21.819881</td>
          <td>20.238721</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.832488</td>
          <td>0.005223</td>
          <td>24.446370</td>
          <td>0.022858</td>
          <td>24.432959</td>
          <td>0.019884</td>
          <td>23.470738</td>
          <td>0.014209</td>
          <td>21.236894</td>
          <td>0.006106</td>
          <td>16.862000</td>
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
          <td>24.957100</td>
          <td>0.032954</td>
          <td>17.287836</td>
          <td>0.005000</td>
          <td>25.680446</td>
          <td>0.068586</td>
          <td>22.691413</td>
          <td>0.009560</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.163515</td>
          <td>0.016571</td>
          <td>24.096691</td>
          <td>0.016341</td>
          <td>22.123094</td>
          <td>0.005662</td>
          <td>23.387020</td>
          <td>0.016210</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.354230</td>
          <td>0.019461</td>
          <td>22.990069</td>
          <td>0.007533</td>
          <td>22.911356</td>
          <td>0.007424</td>
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
          <td>19.537127</td>
          <td>0.005005</td>
          <td>22.310624</td>
          <td>0.005839</td>
          <td>24.255024</td>
          <td>0.019475</td>
          <td>12.204748</td>
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
          <td>24.494417</td>
          <td>0.021954</td>
          <td>26.724810</td>
          <td>0.163579</td>
          <td>27.980285</td>
          <td>0.469474</td>
          <td>24.211557</td>
          <td>0.033085</td>
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
          <td>23.401663</td>
          <td>0.009312</td>
          <td>20.079206</td>
          <td>0.005015</td>
          <td>17.326177</td>
          <td>0.005000</td>
          <td>20.858887</td>
          <td>0.005223</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.685130</td>
          <td>0.005256</td>
          <td>25.348275</td>
          <td>0.048805</td>
          <td>22.247904</td>
          <td>0.005821</td>
          <td>20.796918</td>
          <td>0.005199</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.262651</td>
          <td>20.081021</td>
          <td>23.758420</td>
          <td>19.030635</td>
          <td>23.078506</td>
          <td>25.476684</td>
          <td>20.304319</td>
          <td>0.005021</td>
          <td>25.131609</td>
          <td>0.040234</td>
          <td>25.509657</td>
          <td>0.058921</td>
          <td>18.299778</td>
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
          <td>23.743440</td>
          <td>0.011857</td>
          <td>21.820824</td>
          <td>0.005356</td>
          <td>20.239930</td>
          <td>0.005022</td>
          <td>25.304990</td>
          <td>0.087458</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>25.746425</td>
          <td>0.066544</td>
          <td>20.227648</td>
          <td>0.005020</td>
          <td>24.317341</td>
          <td>0.020542</td>
          <td>22.895771</td>
          <td>0.011028</td>
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
          <td>24.506384</td>
          <td>0.300514</td>
          <td>17.307615</td>
          <td>0.005014</td>
          <td>25.668529</td>
          <td>0.763184</td>
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
          <td>23.969608</td>
          <td>0.192996</td>
          <td>24.553201</td>
          <td>0.265418</td>
          <td>22.185730</td>
          <td>0.044146</td>
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
          <td>23.995162</td>
          <td>0.197197</td>
          <td>22.984328</td>
          <td>0.068823</td>
          <td>22.852031</td>
          <td>0.079853</td>
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
          <td>19.551128</td>
          <td>0.006157</td>
          <td>22.335310</td>
          <td>0.038609</td>
          <td>24.487742</td>
          <td>0.320747</td>
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
          <td>25.953296</td>
          <td>0.860387</td>
          <td>28.329530</td>
          <td>2.496850</td>
          <td>28.094090</td>
          <td>2.555105</td>
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
          <td>23.417977</td>
          <td>0.120231</td>
          <td>20.078844</td>
          <td>0.006972</td>
          <td>17.311817</td>
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
          <td>21.671255</td>
          <td>0.025610</td>
          <td>26.640672</td>
          <td>1.153615</td>
          <td>22.235332</td>
          <td>0.046142</td>
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
          <td>20.302488</td>
          <td>0.008743</td>
          <td>25.807409</td>
          <td>0.684829</td>
          <td>25.185444</td>
          <td>0.545984</td>
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
          <td>23.731657</td>
          <td>0.157650</td>
          <td>21.838799</td>
          <td>0.024892</td>
          <td>20.236263</td>
          <td>0.008929</td>
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
          <td>26.826632</td>
          <td>1.421088</td>
          <td>20.245379</td>
          <td>0.007554</td>
          <td>23.853604</td>
          <td>0.190407</td>
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


