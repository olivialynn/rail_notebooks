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
          <td>21.586207</td>
          <td>23.297535</td>
          <td>23.441593</td>
          <td>22.419557</td>
          <td>23.005064</td>
          <td>29.446922</td>
          <td>15.609062</td>
          <td>18.986505</td>
          <td>26.057583</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.671161</td>
          <td>23.914044</td>
          <td>21.206489</td>
          <td>29.531333</td>
          <td>30.347462</td>
          <td>24.906279</td>
          <td>23.020126</td>
          <td>26.339197</td>
          <td>28.164087</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.273295</td>
          <td>23.230127</td>
          <td>21.456354</td>
          <td>21.236216</td>
          <td>25.905136</td>
          <td>25.182123</td>
          <td>27.246097</td>
          <td>24.632826</td>
          <td>19.929786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.180074</td>
          <td>26.269725</td>
          <td>23.533456</td>
          <td>23.405905</td>
          <td>27.581815</td>
          <td>27.136128</td>
          <td>21.097186</td>
          <td>27.077365</td>
          <td>19.012760</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.636594</td>
          <td>20.741087</td>
          <td>19.903482</td>
          <td>21.413444</td>
          <td>25.418287</td>
          <td>22.539541</td>
          <td>22.611102</td>
          <td>22.260832</td>
          <td>22.079092</td>
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
          <td>23.960030</td>
          <td>24.533009</td>
          <td>18.984507</td>
          <td>20.034179</td>
          <td>24.281921</td>
          <td>24.843418</td>
          <td>19.646623</td>
          <td>21.695711</td>
          <td>24.512680</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.910400</td>
          <td>23.448391</td>
          <td>23.673251</td>
          <td>26.827373</td>
          <td>22.575469</td>
          <td>26.251524</td>
          <td>21.876254</td>
          <td>25.679021</td>
          <td>18.594462</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.749691</td>
          <td>18.680093</td>
          <td>19.287629</td>
          <td>21.632974</td>
          <td>22.821590</td>
          <td>24.037786</td>
          <td>21.548136</td>
          <td>26.805149</td>
          <td>18.061908</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.670669</td>
          <td>23.417347</td>
          <td>29.735443</td>
          <td>15.236667</td>
          <td>18.383042</td>
          <td>22.167652</td>
          <td>20.560500</td>
          <td>20.349420</td>
          <td>18.437758</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.258590</td>
          <td>23.885222</td>
          <td>25.074272</td>
          <td>20.093903</td>
          <td>27.583766</td>
          <td>21.461230</td>
          <td>22.355110</td>
          <td>17.569112</td>
          <td>19.593968</td>
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
          <td>21.599942</td>
          <td>0.007483</td>
          <td>23.312159</td>
          <td>0.009520</td>
          <td>23.425260</td>
          <td>0.009234</td>
          <td>22.415418</td>
          <td>0.007190</td>
          <td>22.988450</td>
          <td>0.017424</td>
          <td>inf</td>
          <td>inf</td>
          <td>15.609062</td>
          <td>18.986505</td>
          <td>26.057583</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.333136</td>
          <td>0.693176</td>
          <td>23.898725</td>
          <td>0.014534</td>
          <td>21.212924</td>
          <td>0.005136</td>
          <td>28.340666</td>
          <td>0.794356</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.047239</td>
          <td>0.231829</td>
          <td>23.020126</td>
          <td>26.339197</td>
          <td>28.164087</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.272018</td>
          <td>0.005396</td>
          <td>23.219152</td>
          <td>0.008984</td>
          <td>21.452104</td>
          <td>0.005198</td>
          <td>21.241701</td>
          <td>0.005341</td>
          <td>25.647054</td>
          <td>0.177346</td>
          <td>25.971608</td>
          <td>0.480947</td>
          <td>27.246097</td>
          <td>24.632826</td>
          <td>19.929786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.953532</td>
          <td>0.103982</td>
          <td>26.394024</td>
          <td>0.126374</td>
          <td>23.537352</td>
          <td>0.009933</td>
          <td>23.382214</td>
          <td>0.013259</td>
          <td>28.359616</td>
          <td>1.247563</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.097186</td>
          <td>27.077365</td>
          <td>19.012760</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.628494</td>
          <td>0.005643</td>
          <td>20.744170</td>
          <td>0.005101</td>
          <td>19.912303</td>
          <td>0.005021</td>
          <td>21.415070</td>
          <td>0.005451</td>
          <td>26.067628</td>
          <td>0.251996</td>
          <td>22.551950</td>
          <td>0.026299</td>
          <td>22.611102</td>
          <td>22.260832</td>
          <td>22.079092</td>
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
          <td>23.981117</td>
          <td>0.044288</td>
          <td>24.507040</td>
          <td>0.024080</td>
          <td>18.985597</td>
          <td>0.005007</td>
          <td>20.038891</td>
          <td>0.005053</td>
          <td>24.248319</td>
          <td>0.052218</td>
          <td>25.004660</td>
          <td>0.223781</td>
          <td>19.646623</td>
          <td>21.695711</td>
          <td>24.512680</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.876598</td>
          <td>0.097237</td>
          <td>23.435901</td>
          <td>0.010329</td>
          <td>23.655690</td>
          <td>0.010776</td>
          <td>27.438848</td>
          <td>0.418368</td>
          <td>22.560787</td>
          <td>0.012402</td>
          <td>26.429895</td>
          <td>0.667862</td>
          <td>21.876254</td>
          <td>25.679021</td>
          <td>18.594462</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.757399</td>
          <td>0.005203</td>
          <td>18.681651</td>
          <td>0.005008</td>
          <td>19.293040</td>
          <td>0.005010</td>
          <td>21.635120</td>
          <td>0.005643</td>
          <td>22.824762</td>
          <td>0.015242</td>
          <td>23.902928</td>
          <td>0.086781</td>
          <td>21.548136</td>
          <td>26.805149</td>
          <td>18.061908</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.663094</td>
          <td>0.014603</td>
          <td>23.404325</td>
          <td>0.010112</td>
          <td>inf</td>
          <td>inf</td>
          <td>15.234996</td>
          <td>0.005000</td>
          <td>18.376525</td>
          <td>0.005016</td>
          <td>22.176555</td>
          <td>0.019061</td>
          <td>20.560500</td>
          <td>20.349420</td>
          <td>18.437758</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.270943</td>
          <td>0.010979</td>
          <td>23.874977</td>
          <td>0.014265</td>
          <td>25.094117</td>
          <td>0.035324</td>
          <td>20.089181</td>
          <td>0.005057</td>
          <td>27.962678</td>
          <td>0.992807</td>
          <td>21.456037</td>
          <td>0.010831</td>
          <td>22.355110</td>
          <td>17.569112</td>
          <td>19.593968</td>
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
          <td>21.586207</td>
          <td>23.297535</td>
          <td>23.441593</td>
          <td>22.419557</td>
          <td>23.005064</td>
          <td>29.446922</td>
          <td>15.612878</td>
          <td>0.005000</td>
          <td>18.991379</td>
          <td>0.005006</td>
          <td>25.832624</td>
          <td>0.121773</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.671161</td>
          <td>23.914044</td>
          <td>21.206489</td>
          <td>29.531333</td>
          <td>30.347462</td>
          <td>24.906279</td>
          <td>23.023783</td>
          <td>0.007471</td>
          <td>26.462809</td>
          <td>0.208722</td>
          <td>27.564895</td>
          <td>0.499931</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.273295</td>
          <td>23.230127</td>
          <td>21.456354</td>
          <td>21.236216</td>
          <td>25.905136</td>
          <td>25.182123</td>
          <td>27.443837</td>
          <td>0.285722</td>
          <td>24.654721</td>
          <td>0.042942</td>
          <td>19.934093</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.180074</td>
          <td>26.269725</td>
          <td>23.533456</td>
          <td>23.405905</td>
          <td>27.581815</td>
          <td>27.136128</td>
          <td>21.097868</td>
          <td>0.005088</td>
          <td>27.162614</td>
          <td>0.368218</td>
          <td>19.015476</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.636594</td>
          <td>20.741087</td>
          <td>19.903482</td>
          <td>21.413444</td>
          <td>25.418287</td>
          <td>22.539541</td>
          <td>22.602964</td>
          <td>0.006262</td>
          <td>22.260872</td>
          <td>0.006916</td>
          <td>22.077050</td>
          <td>0.006425</td>
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
          <td>23.960030</td>
          <td>24.533009</td>
          <td>18.984507</td>
          <td>20.034179</td>
          <td>24.281921</td>
          <td>24.843418</td>
          <td>19.647354</td>
          <td>0.005006</td>
          <td>21.699301</td>
          <td>0.005755</td>
          <td>24.521535</td>
          <td>0.038139</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.910400</td>
          <td>23.448391</td>
          <td>23.673251</td>
          <td>26.827373</td>
          <td>22.575469</td>
          <td>26.251524</td>
          <td>21.873364</td>
          <td>0.005358</td>
          <td>25.856576</td>
          <td>0.124335</td>
          <td>18.593789</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.749691</td>
          <td>18.680093</td>
          <td>19.287629</td>
          <td>21.632974</td>
          <td>22.821590</td>
          <td>24.037786</td>
          <td>21.557589</td>
          <td>0.005203</td>
          <td>26.335402</td>
          <td>0.187502</td>
          <td>18.061201</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.670669</td>
          <td>23.417347</td>
          <td>29.735443</td>
          <td>15.236667</td>
          <td>18.383042</td>
          <td>22.167652</td>
          <td>20.560089</td>
          <td>0.005033</td>
          <td>20.355484</td>
          <td>0.005068</td>
          <td>18.443011</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.258590</td>
          <td>23.885222</td>
          <td>25.074272</td>
          <td>20.093903</td>
          <td>27.583766</td>
          <td>21.461230</td>
          <td>22.345404</td>
          <td>0.005817</td>
          <td>17.586717</td>
          <td>0.005000</td>
          <td>19.594704</td>
          <td>0.005017</td>
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
          <td>21.586207</td>
          <td>23.297535</td>
          <td>23.441593</td>
          <td>22.419557</td>
          <td>23.005064</td>
          <td>29.446922</td>
          <td>15.603515</td>
          <td>0.005001</td>
          <td>18.991036</td>
          <td>0.005180</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.671161</td>
          <td>23.914044</td>
          <td>21.206489</td>
          <td>29.531333</td>
          <td>30.347462</td>
          <td>24.906279</td>
          <td>22.983709</td>
          <td>0.062935</td>
          <td>25.276127</td>
          <td>0.372122</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.273295</td>
          <td>23.230127</td>
          <td>21.456354</td>
          <td>21.236216</td>
          <td>25.905136</td>
          <td>25.182123</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.463334</td>
          <td>0.191978</td>
          <td>19.923651</td>
          <td>0.006105</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.180074</td>
          <td>26.269725</td>
          <td>23.533456</td>
          <td>23.405905</td>
          <td>27.581815</td>
          <td>27.136128</td>
          <td>21.084385</td>
          <td>0.012231</td>
          <td>25.666344</td>
          <td>0.500466</td>
          <td>19.010566</td>
          <td>0.005224</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.636594</td>
          <td>20.741087</td>
          <td>19.903482</td>
          <td>21.413444</td>
          <td>25.418287</td>
          <td>22.539541</td>
          <td>22.595993</td>
          <td>0.044552</td>
          <td>22.280765</td>
          <td>0.028197</td>
          <td>22.098103</td>
          <td>0.026220</td>
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
          <td>23.960030</td>
          <td>24.533009</td>
          <td>18.984507</td>
          <td>20.034179</td>
          <td>24.281921</td>
          <td>24.843418</td>
          <td>19.647320</td>
          <td>0.005820</td>
          <td>21.692666</td>
          <td>0.016978</td>
          <td>24.525648</td>
          <td>0.219972</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.910400</td>
          <td>23.448391</td>
          <td>23.673251</td>
          <td>26.827373</td>
          <td>22.575469</td>
          <td>26.251524</td>
          <td>21.876190</td>
          <td>0.023569</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.592277</td>
          <td>0.005105</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.749691</td>
          <td>18.680093</td>
          <td>19.287629</td>
          <td>21.632974</td>
          <td>22.821590</td>
          <td>24.037786</td>
          <td>21.576217</td>
          <td>0.018213</td>
          <td>27.883152</td>
          <td>1.853114</td>
          <td>18.053309</td>
          <td>0.005039</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.670669</td>
          <td>23.417347</td>
          <td>29.735443</td>
          <td>15.236667</td>
          <td>18.383042</td>
          <td>22.167652</td>
          <td>20.546041</td>
          <td>0.008448</td>
          <td>20.342991</td>
          <td>0.006863</td>
          <td>18.434279</td>
          <td>0.005078</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.258590</td>
          <td>23.885222</td>
          <td>25.074272</td>
          <td>20.093903</td>
          <td>27.583766</td>
          <td>21.461230</td>
          <td>22.361806</td>
          <td>0.036165</td>
          <td>17.577836</td>
          <td>0.005014</td>
          <td>19.592730</td>
          <td>0.005628</td>
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


