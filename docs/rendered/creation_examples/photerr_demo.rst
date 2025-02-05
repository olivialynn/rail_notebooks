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
          <td>22.589093</td>
          <td>0.013802</td>
          <td>22.953337</td>
          <td>0.007745</td>
          <td>25.831553</td>
          <td>0.067934</td>
          <td>18.279792</td>
          <td>0.005006</td>
          <td>23.014210</td>
          <td>0.017801</td>
          <td>23.721883</td>
          <td>0.073969</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
          <td>22.682690</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.426049</td>
          <td>0.065499</td>
          <td>20.937879</td>
          <td>0.005133</td>
          <td>24.394002</td>
          <td>0.019240</td>
          <td>22.934351</td>
          <td>0.009611</td>
          <td>26.049398</td>
          <td>0.248250</td>
          <td>23.163574</td>
          <td>0.045085</td>
          <td>24.175778</td>
          <td>24.107270</td>
          <td>22.120187</td>
          <td>23.411418</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.002070</td>
          <td>0.009246</td>
          <td>25.426242</td>
          <td>0.053983</td>
          <td>22.970262</td>
          <td>0.007197</td>
          <td>23.441384</td>
          <td>0.013885</td>
          <td>24.871237</td>
          <td>0.090607</td>
          <td>26.715600</td>
          <td>0.808464</td>
          <td>24.365248</td>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.629288</td>
          <td>0.005173</td>
          <td>18.621249</td>
          <td>0.005008</td>
          <td>22.592122</td>
          <td>0.006228</td>
          <td>25.213670</td>
          <td>0.064170</td>
          <td>22.209607</td>
          <td>0.009672</td>
          <td>23.004631</td>
          <td>0.039160</td>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.860372</td>
          <td>0.007400</td>
          <td>23.062960</td>
          <td>0.007522</td>
          <td>22.878721</td>
          <td>0.009272</td>
          <td>16.978320</td>
          <td>0.005003</td>
          <td>25.089843</td>
          <td>0.240141</td>
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
          <td>22.500092</td>
          <td>0.012915</td>
          <td>19.345536</td>
          <td>0.005017</td>
          <td>30.740363</td>
          <td>2.097820</td>
          <td>17.291699</td>
          <td>0.005002</td>
          <td>25.364804</td>
          <td>0.139302</td>
          <td>21.282152</td>
          <td>0.009622</td>
          <td>23.401882</td>
          <td>20.084787</td>
          <td>17.316507</td>
          <td>20.853307</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.175577</td>
          <td>0.022095</td>
          <td>23.243160</td>
          <td>0.009117</td>
          <td>23.952259</td>
          <td>0.013432</td>
          <td>26.056753</td>
          <td>0.134460</td>
          <td>24.461570</td>
          <td>0.063095</td>
          <td>23.993256</td>
          <td>0.093954</td>
          <td>21.691973</td>
          <td>25.383890</td>
          <td>22.249757</td>
          <td>20.794577</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.258854</td>
          <td>0.005013</td>
          <td>20.077670</td>
          <td>0.005042</td>
          <td>23.754995</td>
          <td>0.011573</td>
          <td>19.029297</td>
          <td>0.005014</td>
          <td>23.061218</td>
          <td>0.018513</td>
          <td>25.330257</td>
          <td>0.292211</td>
          <td>20.307481</td>
          <td>25.088338</td>
          <td>25.547408</td>
          <td>18.303647</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.514722</td>
          <td>0.005150</td>
          <td>18.693593</td>
          <td>0.005009</td>
          <td>19.693659</td>
          <td>0.005016</td>
          <td>23.648444</td>
          <td>0.016397</td>
          <td>22.717012</td>
          <td>0.013989</td>
          <td>22.653225</td>
          <td>0.028730</td>
          <td>23.722045</td>
          <td>21.819881</td>
          <td>20.238721</td>
          <td>25.211479</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.826247</td>
          <td>0.005221</td>
          <td>24.470417</td>
          <td>0.023334</td>
          <td>24.488399</td>
          <td>0.020842</td>
          <td>23.454073</td>
          <td>0.014024</td>
          <td>21.236786</td>
          <td>0.006106</td>
          <td>16.846925</td>
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
          <td>24.991819</td>
          <td>0.033985</td>
          <td>17.302566</td>
          <td>0.005000</td>
          <td>25.810535</td>
          <td>0.076974</td>
          <td>22.689297</td>
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
          <td>24.162735</td>
          <td>0.016560</td>
          <td>24.122744</td>
          <td>0.016698</td>
          <td>22.111704</td>
          <td>0.005649</td>
          <td>23.419380</td>
          <td>0.016652</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.397678</td>
          <td>0.020198</td>
          <td>22.972001</td>
          <td>0.007464</td>
          <td>22.923611</td>
          <td>0.007470</td>
          <td>27.002486</td>
          <td>0.365315</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
          <td>19.548507</td>
          <td>0.005005</td>
          <td>22.310503</td>
          <td>0.005839</td>
          <td>24.255028</td>
          <td>0.019475</td>
          <td>12.195832</td>
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
          <td>24.498990</td>
          <td>0.022041</td>
          <td>26.740380</td>
          <td>0.165768</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.165783</td>
          <td>0.031769</td>
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
          <td>23.395047</td>
          <td>0.009272</td>
          <td>20.079781</td>
          <td>0.005015</td>
          <td>17.306222</td>
          <td>0.005000</td>
          <td>20.852512</td>
          <td>0.005220</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.135983</td>
          <td>23.237559</td>
          <td>23.945569</td>
          <td>25.918791</td>
          <td>24.586021</td>
          <td>24.115220</td>
          <td>21.690210</td>
          <td>0.005258</td>
          <td>25.301230</td>
          <td>0.046800</td>
          <td>22.248348</td>
          <td>0.005822</td>
          <td>20.797399</td>
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
          <td>20.320755</td>
          <td>0.005021</td>
          <td>25.068749</td>
          <td>0.038044</td>
          <td>25.508231</td>
          <td>0.058847</td>
          <td>18.296172</td>
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
          <td>23.730647</td>
          <td>0.011743</td>
          <td>21.821121</td>
          <td>0.005357</td>
          <td>20.246599</td>
          <td>0.005022</td>
          <td>25.216749</td>
          <td>0.080900</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.828188</td>
          <td>24.465429</td>
          <td>24.486282</td>
          <td>23.454620</td>
          <td>21.239060</td>
          <td>16.850985</td>
          <td>25.777947</td>
          <td>0.068434</td>
          <td>20.233271</td>
          <td>0.005020</td>
          <td>24.330932</td>
          <td>0.020783</td>
          <td>22.895487</td>
          <td>0.011026</td>
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
          <td>24.868573</td>
          <td>0.399764</td>
          <td>17.303511</td>
          <td>0.005014</td>
          <td>26.399496</td>
          <td>1.192469</td>
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
          <td>23.871993</td>
          <td>0.177695</td>
          <td>24.497977</td>
          <td>0.253681</td>
          <td>22.061921</td>
          <td>0.039535</td>
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
          <td>25.081097</td>
          <td>0.469759</td>
          <td>22.924407</td>
          <td>0.065254</td>
          <td>22.791235</td>
          <td>0.075669</td>
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
          <td>19.528218</td>
          <td>0.006114</td>
          <td>22.356862</td>
          <td>0.039357</td>
          <td>24.680871</td>
          <td>0.373500</td>
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
          <td>24.535478</td>
          <td>0.307619</td>
          <td>25.884491</td>
          <td>0.721570</td>
          <td>26.735361</td>
          <td>1.427466</td>
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
          <td>23.296578</td>
          <td>0.108143</td>
          <td>20.079970</td>
          <td>0.006975</td>
          <td>17.313750</td>
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
          <td>21.667748</td>
          <td>0.025531</td>
          <td>26.117936</td>
          <td>0.841179</td>
          <td>22.228680</td>
          <td>0.045869</td>
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
          <td>20.307422</td>
          <td>0.008770</td>
          <td>25.015744</td>
          <td>0.383766</td>
          <td>25.999369</td>
          <td>0.942749</td>
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
          <td>23.760794</td>
          <td>0.161631</td>
          <td>21.868047</td>
          <td>0.025538</td>
          <td>20.235334</td>
          <td>0.008924</td>
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
          <td>26.876380</td>
          <td>1.457620</td>
          <td>20.235713</td>
          <td>0.007516</td>
          <td>24.104994</td>
          <td>0.234959</td>
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


