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
          <td>22.596570</td>
          <td>0.013880</td>
          <td>22.970325</td>
          <td>0.007812</td>
          <td>25.800498</td>
          <td>0.066091</td>
          <td>18.266035</td>
          <td>0.005006</td>
          <td>23.054769</td>
          <td>0.018414</td>
          <td>23.640084</td>
          <td>0.068805</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.693690</td>
          <td>0.014952</td>
          <td>24.450561</td>
          <td>0.022940</td>
          <td>20.919247</td>
          <td>0.005087</td>
          <td>24.330123</td>
          <td>0.029348</td>
          <td>22.903917</td>
          <td>0.016253</td>
          <td>27.037492</td>
          <td>0.989251</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.127803</td>
          <td>0.009992</td>
          <td>23.415060</td>
          <td>0.010185</td>
          <td>22.000143</td>
          <td>0.005476</td>
          <td>25.362583</td>
          <td>0.073215</td>
          <td>22.959166</td>
          <td>0.017007</td>
          <td>23.507189</td>
          <td>0.061162</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.976230</td>
          <td>0.018731</td>
          <td>22.924334</td>
          <td>0.007633</td>
          <td>28.503304</td>
          <td>0.609092</td>
          <td>19.627090</td>
          <td>0.005030</td>
          <td>18.620458</td>
          <td>0.005021</td>
          <td>22.673992</td>
          <td>0.029258</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.543025</td>
          <td>0.005156</td>
          <td>22.307601</td>
          <td>0.006048</td>
          <td>24.333800</td>
          <td>0.018291</td>
          <td>12.208395</td>
          <td>0.005000</td>
          <td>26.668865</td>
          <td>0.406644</td>
          <td>22.903879</td>
          <td>0.035820</td>
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
          <td>18.285128</td>
          <td>0.005038</td>
          <td>18.225921</td>
          <td>0.005005</td>
          <td>14.350271</td>
          <td>0.005000</td>
          <td>22.218756</td>
          <td>0.006623</td>
          <td>23.073430</td>
          <td>0.018704</td>
          <td>23.301431</td>
          <td>0.050954</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.192706</td>
          <td>0.629216</td>
          <td>19.204961</td>
          <td>0.005015</td>
          <td>23.748462</td>
          <td>0.011518</td>
          <td>23.101215</td>
          <td>0.010766</td>
          <td>24.885796</td>
          <td>0.091774</td>
          <td>21.777621</td>
          <td>0.013771</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.785162</td>
          <td>0.005799</td>
          <td>24.980348</td>
          <td>0.036391</td>
          <td>20.556182</td>
          <td>0.005051</td>
          <td>20.924128</td>
          <td>0.005205</td>
          <td>24.420228</td>
          <td>0.060824</td>
          <td>23.510303</td>
          <td>0.061331</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817529</td>
          <td>0.092347</td>
          <td>24.860531</td>
          <td>0.032751</td>
          <td>26.115206</td>
          <td>0.087276</td>
          <td>25.809729</td>
          <td>0.108488</td>
          <td>17.077598</td>
          <td>0.005004</td>
          <td>27.548364</td>
          <td>1.322469</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159984</td>
          <td>0.010200</td>
          <td>17.809247</td>
          <td>0.005003</td>
          <td>20.717789</td>
          <td>0.005064</td>
          <td>20.690508</td>
          <td>0.005142</td>
          <td>27.893991</td>
          <td>0.952176</td>
          <td>19.853430</td>
          <td>0.005528</td>
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
          <td>24.958646</td>
          <td>0.033000</td>
          <td>17.293333</td>
          <td>0.005000</td>
          <td>25.720732</td>
          <td>0.110452</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.243678</td>
          <td>0.008436</td>
          <td>24.161423</td>
          <td>0.027720</td>
          <td>24.123648</td>
          <td>0.026814</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.808793</td>
          <td>0.028902</td>
          <td>25.691411</td>
          <td>0.107655</td>
          <td>24.334547</td>
          <td>0.032302</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.168665</td>
          <td>0.039773</td>
          <td>22.224628</td>
          <td>0.006809</td>
          <td>23.063036</td>
          <td>0.011168</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.046474</td>
          <td>0.007558</td>
          <td>22.884770</td>
          <td>0.009844</td>
          <td>16.983871</td>
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
          <td>24.796271</td>
          <td>0.028585</td>
          <td>21.632755</td>
          <td>0.005673</td>
          <td>19.869779</td>
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
          <td>21.329303</td>
          <td>0.005134</td>
          <td>22.312347</td>
          <td>0.007078</td>
          <td>22.760901</td>
          <td>0.009070</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.366608</td>
          <td>0.019668</td>
          <td>26.320805</td>
          <td>0.185201</td>
          <td>24.481009</td>
          <td>0.036788</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.028546</td>
          <td>0.014830</td>
          <td>26.111664</td>
          <td>0.154972</td>
          <td>19.985253</td>
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
          <td>26.734991</td>
          <td>0.158101</td>
          <td>22.614788</td>
          <td>0.008293</td>
          <td>24.664795</td>
          <td>0.043330</td>
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
          <td>24.579371</td>
          <td>0.249832</td>
          <td>17.297637</td>
          <td>0.005008</td>
          <td>24.526724</td>
          <td>0.220170</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.123830</td>
          <td>0.071279</td>
          <td>24.202306</td>
          <td>0.153733</td>
          <td>24.037709</td>
          <td>0.145430</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.922179</td>
          <td>0.329658</td>
          <td>25.504283</td>
          <td>0.443397</td>
          <td>24.335739</td>
          <td>0.187555</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.895532</td>
          <td>0.679290</td>
          <td>22.221787</td>
          <td>0.026771</td>
          <td>23.068448</td>
          <td>0.062086</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.077276</td>
          <td>0.068393</td>
          <td>22.882242</td>
          <td>0.048113</td>
          <td>16.972530</td>
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
          <td>25.072930</td>
          <td>0.371195</td>
          <td>21.618631</td>
          <td>0.015966</td>
          <td>19.868105</td>
          <td>0.006007</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.332121</td>
          <td>0.014874</td>
          <td>22.318702</td>
          <td>0.029156</td>
          <td>22.761580</td>
          <td>0.047234</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>25.007157</td>
          <td>0.352558</td>
          <td>25.821039</td>
          <td>0.560190</td>
          <td>24.378268</td>
          <td>0.194411</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.394930</td>
          <td>0.214406</td>
          <td>26.679020</td>
          <td>0.989682</td>
          <td>19.990257</td>
          <td>0.006235</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>25.887843</td>
          <td>0.675721</td>
          <td>22.602784</td>
          <td>0.037508</td>
          <td>25.342516</td>
          <td>0.423079</td>
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


