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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.593975</td>
          <td>0.013853</td>
          <td>22.961697</td>
          <td>0.007778</td>
          <td>25.840797</td>
          <td>0.068493</td>
          <td>18.273973</td>
          <td>0.005006</td>
          <td>23.031637</td>
          <td>0.018061</td>
          <td>23.701224</td>
          <td>0.072630</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.696522</td>
          <td>0.014985</td>
          <td>24.383098</td>
          <td>0.021657</td>
          <td>20.931922</td>
          <td>0.005088</td>
          <td>24.357006</td>
          <td>0.030048</td>
          <td>22.931415</td>
          <td>0.016623</td>
          <td>26.316866</td>
          <td>0.617404</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.116007</td>
          <td>0.009917</td>
          <td>23.394969</td>
          <td>0.010049</td>
          <td>21.998752</td>
          <td>0.005475</td>
          <td>25.328741</td>
          <td>0.071055</td>
          <td>22.964220</td>
          <td>0.017078</td>
          <td>23.425208</td>
          <td>0.056871</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986487</td>
          <td>0.018889</td>
          <td>22.928578</td>
          <td>0.007649</td>
          <td>27.785301</td>
          <td>0.356357</td>
          <td>19.621053</td>
          <td>0.005029</td>
          <td>18.620673</td>
          <td>0.005021</td>
          <td>22.605555</td>
          <td>0.027558</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.547045</td>
          <td>0.005156</td>
          <td>22.295424</td>
          <td>0.006029</td>
          <td>24.293579</td>
          <td>0.017688</td>
          <td>12.213353</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.948250</td>
          <td>0.037253</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.288881</td>
          <td>0.005038</td>
          <td>18.224713</td>
          <td>0.005005</td>
          <td>14.355722</td>
          <td>0.005000</td>
          <td>22.220940</td>
          <td>0.006629</td>
          <td>23.059686</td>
          <td>0.018490</td>
          <td>23.363872</td>
          <td>0.053858</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.032927</td>
          <td>0.561876</td>
          <td>19.204239</td>
          <td>0.005015</td>
          <td>23.719947</td>
          <td>0.011282</td>
          <td>23.080158</td>
          <td>0.010608</td>
          <td>25.110466</td>
          <td>0.111726</td>
          <td>21.815432</td>
          <td>0.014187</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.792852</td>
          <td>0.005808</td>
          <td>24.933984</td>
          <td>0.034935</td>
          <td>20.551507</td>
          <td>0.005050</td>
          <td>20.930927</td>
          <td>0.005207</td>
          <td>24.400313</td>
          <td>0.059759</td>
          <td>23.526232</td>
          <td>0.062203</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.623812</td>
          <td>0.077932</td>
          <td>24.837024</td>
          <td>0.032082</td>
          <td>26.198606</td>
          <td>0.093916</td>
          <td>25.958464</td>
          <td>0.123488</td>
          <td>17.076152</td>
          <td>0.005004</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159403</td>
          <td>0.010196</td>
          <td>17.799004</td>
          <td>0.005003</td>
          <td>20.729339</td>
          <td>0.005065</td>
          <td>20.684833</td>
          <td>0.005141</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.832996</td>
          <td>0.005510</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.962271</td>
          <td>0.033106</td>
          <td>17.295337</td>
          <td>0.005000</td>
          <td>25.720328</td>
          <td>0.110413</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.223497</td>
          <td>0.008336</td>
          <td>24.195552</td>
          <td>0.028567</td>
          <td>24.072368</td>
          <td>0.025635</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.857150</td>
          <td>0.030163</td>
          <td>25.576789</td>
          <td>0.097360</td>
          <td>24.398781</td>
          <td>0.034196</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.328280</td>
          <td>0.045853</td>
          <td>22.214915</td>
          <td>0.006781</td>
          <td>23.035464</td>
          <td>0.010946</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.070828</td>
          <td>0.007655</td>
          <td>22.874599</td>
          <td>0.009776</td>
          <td>16.975394</td>
          <td>0.005000</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.734962</td>
          <td>0.027082</td>
          <td>21.640400</td>
          <td>0.005682</td>
          <td>19.868088</td>
          <td>0.005028</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.313914</td>
          <td>0.005131</td>
          <td>22.322244</td>
          <td>0.007111</td>
          <td>22.755483</td>
          <td>0.009039</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.396127</td>
          <td>0.020171</td>
          <td>26.283311</td>
          <td>0.179410</td>
          <td>24.427557</td>
          <td>0.035081</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.032356</td>
          <td>0.014876</td>
          <td>26.082682</td>
          <td>0.151165</td>
          <td>19.985552</td>
          <td>0.005034</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>27.163881</td>
          <td>0.227083</td>
          <td>22.620655</td>
          <td>0.008322</td>
          <td>24.735331</td>
          <td>0.046142</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>25.101830</td>
          <td>0.379642</td>
          <td>17.289490</td>
          <td>0.005008</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.232615</td>
          <td>0.078493</td>
          <td>24.132404</td>
          <td>0.144767</td>
          <td>24.073452</td>
          <td>0.149971</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>25.050428</td>
          <td>0.364728</td>
          <td>26.229830</td>
          <td>0.743819</td>
          <td>24.306619</td>
          <td>0.182990</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>28.097315</td>
          <td>2.202929</td>
          <td>22.249618</td>
          <td>0.027434</td>
          <td>23.094964</td>
          <td>0.063568</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>22.961260</td>
          <td>0.061691</td>
          <td>22.898959</td>
          <td>0.048835</td>
          <td>16.974412</td>
          <td>0.005005</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.495086</td>
          <td>0.233039</td>
          <td>21.614313</td>
          <td>0.015909</td>
          <td>19.867738</td>
          <td>0.006006</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.314215</td>
          <td>0.014659</td>
          <td>22.320590</td>
          <td>0.029205</td>
          <td>22.801159</td>
          <td>0.048931</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.337187</td>
          <td>0.204286</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.270531</td>
          <td>0.177474</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.020435</td>
          <td>0.156142</td>
          <td>26.706675</td>
          <td>1.006306</td>
          <td>19.993276</td>
          <td>0.006241</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>24.793363</td>
          <td>0.297381</td>
          <td>22.647670</td>
          <td>0.039037</td>
          <td>24.615575</td>
          <td>0.237025</td>
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


