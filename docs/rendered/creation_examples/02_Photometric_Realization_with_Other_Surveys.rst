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
          <td>22.594172</td>
          <td>0.013855</td>
          <td>22.976268</td>
          <td>0.007836</td>
          <td>25.853902</td>
          <td>0.069292</td>
          <td>18.279596</td>
          <td>0.005006</td>
          <td>23.028862</td>
          <td>0.018020</td>
          <td>23.498007</td>
          <td>0.060666</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.683136</td>
          <td>0.014830</td>
          <td>24.375038</td>
          <td>0.021509</td>
          <td>20.920321</td>
          <td>0.005087</td>
          <td>24.366303</td>
          <td>0.030294</td>
          <td>22.896072</td>
          <td>0.016149</td>
          <td>25.590149</td>
          <td>0.359322</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120389</td>
          <td>0.009945</td>
          <td>23.404462</td>
          <td>0.010113</td>
          <td>22.002748</td>
          <td>0.005478</td>
          <td>25.340762</td>
          <td>0.071815</td>
          <td>22.961012</td>
          <td>0.017033</td>
          <td>23.535473</td>
          <td>0.062715</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.008510</td>
          <td>0.019233</td>
          <td>22.924005</td>
          <td>0.007632</td>
          <td>29.057025</td>
          <td>0.881828</td>
          <td>19.625096</td>
          <td>0.005030</td>
          <td>18.622565</td>
          <td>0.005021</td>
          <td>22.588934</td>
          <td>0.027161</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.549395</td>
          <td>0.005157</td>
          <td>22.299107</td>
          <td>0.006035</td>
          <td>24.294625</td>
          <td>0.017704</td>
          <td>12.202376</td>
          <td>0.005000</td>
          <td>27.670844</td>
          <td>0.827512</td>
          <td>22.902936</td>
          <td>0.035791</td>
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
          <td>18.290161</td>
          <td>0.005038</td>
          <td>18.221469</td>
          <td>0.005005</td>
          <td>14.348846</td>
          <td>0.005000</td>
          <td>22.221887</td>
          <td>0.006631</td>
          <td>23.051191</td>
          <td>0.018359</td>
          <td>23.339947</td>
          <td>0.052726</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.669310</td>
          <td>0.864687</td>
          <td>19.197663</td>
          <td>0.005015</td>
          <td>23.731506</td>
          <td>0.011376</td>
          <td>23.109019</td>
          <td>0.010825</td>
          <td>24.947121</td>
          <td>0.096851</td>
          <td>21.778503</td>
          <td>0.013781</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.776468</td>
          <td>0.005790</td>
          <td>24.943053</td>
          <td>0.035215</td>
          <td>20.556473</td>
          <td>0.005051</td>
          <td>20.930976</td>
          <td>0.005207</td>
          <td>24.440345</td>
          <td>0.061919</td>
          <td>23.555394</td>
          <td>0.063832</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.790818</td>
          <td>0.090215</td>
          <td>24.851593</td>
          <td>0.032495</td>
          <td>26.100766</td>
          <td>0.086173</td>
          <td>25.939544</td>
          <td>0.121476</td>
          <td>17.084794</td>
          <td>0.005004</td>
          <td>26.633713</td>
          <td>0.766246</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.149339</td>
          <td>0.010130</td>
          <td>17.809322</td>
          <td>0.005003</td>
          <td>20.727161</td>
          <td>0.005065</td>
          <td>20.687949</td>
          <td>0.005141</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.849187</td>
          <td>0.005524</td>
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
          <td>24.951243</td>
          <td>0.032784</td>
          <td>17.294277</td>
          <td>0.005000</td>
          <td>25.770876</td>
          <td>0.115396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.251794</td>
          <td>0.008477</td>
          <td>24.154726</td>
          <td>0.027557</td>
          <td>24.057919</td>
          <td>0.025312</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.740030</td>
          <td>0.027203</td>
          <td>25.654792</td>
          <td>0.104258</td>
          <td>24.343201</td>
          <td>0.032551</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.219224</td>
          <td>0.041605</td>
          <td>22.211343</td>
          <td>0.006771</td>
          <td>23.046464</td>
          <td>0.011034</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.052144</td>
          <td>0.007580</td>
          <td>22.874242</td>
          <td>0.009773</td>
          <td>16.990400</td>
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
          <td>24.700877</td>
          <td>0.026284</td>
          <td>21.639009</td>
          <td>0.005681</td>
          <td>19.858082</td>
          <td>0.005027</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.315861</td>
          <td>0.005131</td>
          <td>22.310903</td>
          <td>0.007074</td>
          <td>22.759011</td>
          <td>0.009059</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.390669</td>
          <td>0.020077</td>
          <td>26.460572</td>
          <td>0.208331</td>
          <td>24.390406</td>
          <td>0.033943</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.010087</td>
          <td>0.014610</td>
          <td>25.889576</td>
          <td>0.127950</td>
          <td>19.987999</td>
          <td>0.005035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>27.291827</td>
          <td>0.252403</td>
          <td>22.622718</td>
          <td>0.008332</td>
          <td>24.652354</td>
          <td>0.042852</td>
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
          <td>24.967766</td>
          <td>0.341780</td>
          <td>17.291995</td>
          <td>0.005008</td>
          <td>26.192742</td>
          <td>0.775475</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.178088</td>
          <td>0.074792</td>
          <td>24.231991</td>
          <td>0.157695</td>
          <td>23.985189</td>
          <td>0.138990</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.806545</td>
          <td>0.300553</td>
          <td>25.947549</td>
          <td>0.612966</td>
          <td>24.462127</td>
          <td>0.208603</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.206330</td>
          <td>0.411533</td>
          <td>22.212422</td>
          <td>0.026551</td>
          <td>23.077785</td>
          <td>0.062604</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.025732</td>
          <td>0.065331</td>
          <td>22.868707</td>
          <td>0.047535</td>
          <td>16.972655</td>
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
          <td>24.653204</td>
          <td>0.265419</td>
          <td>21.668479</td>
          <td>0.016639</td>
          <td>19.876590</td>
          <td>0.006021</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.313313</td>
          <td>0.014648</td>
          <td>22.317544</td>
          <td>0.029126</td>
          <td>22.801478</td>
          <td>0.048945</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.556082</td>
          <td>0.245087</td>
          <td>26.041747</td>
          <td>0.654607</td>
          <td>24.491360</td>
          <td>0.213767</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>23.965325</td>
          <td>0.148927</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.988809</td>
          <td>0.006232</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.632789</td>
          <td>0.038523</td>
          <td>24.650036</td>
          <td>0.243868</td>
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


