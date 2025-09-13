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
          <td>19.026954</td>
          <td>25.829397</td>
          <td>16.715267</td>
          <td>21.665284</td>
          <td>26.399003</td>
          <td>26.597665</td>
          <td>28.094913</td>
          <td>23.403678</td>
          <td>23.693345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.366629</td>
          <td>24.039625</td>
          <td>19.791147</td>
          <td>28.708114</td>
          <td>23.528356</td>
          <td>21.266654</td>
          <td>19.751685</td>
          <td>24.696488</td>
          <td>22.518345</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.935219</td>
          <td>22.435471</td>
          <td>22.722660</td>
          <td>23.981500</td>
          <td>28.538375</td>
          <td>16.929203</td>
          <td>28.284574</td>
          <td>26.146072</td>
          <td>20.457892</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.681486</td>
          <td>24.022652</td>
          <td>19.840418</td>
          <td>24.615505</td>
          <td>22.447140</td>
          <td>27.257496</td>
          <td>22.445153</td>
          <td>22.609286</td>
          <td>23.535443</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.765528</td>
          <td>21.498559</td>
          <td>22.175922</td>
          <td>22.879324</td>
          <td>26.575045</td>
          <td>19.591031</td>
          <td>23.554278</td>
          <td>24.249611</td>
          <td>23.248854</td>
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
          <td>27.218123</td>
          <td>24.173920</td>
          <td>24.496385</td>
          <td>19.564515</td>
          <td>26.176967</td>
          <td>19.162790</td>
          <td>26.040693</td>
          <td>21.334591</td>
          <td>23.128631</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.871879</td>
          <td>24.816489</td>
          <td>22.103098</td>
          <td>21.541191</td>
          <td>26.591616</td>
          <td>23.803936</td>
          <td>29.438678</td>
          <td>26.304011</td>
          <td>26.456370</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.202065</td>
          <td>23.389941</td>
          <td>25.248379</td>
          <td>22.843671</td>
          <td>18.422329</td>
          <td>19.899997</td>
          <td>18.764430</td>
          <td>22.359746</td>
          <td>22.393311</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.241948</td>
          <td>21.865467</td>
          <td>25.565814</td>
          <td>22.701558</td>
          <td>29.535579</td>
          <td>22.420394</td>
          <td>23.551888</td>
          <td>24.263885</td>
          <td>20.450035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.073877</td>
          <td>20.429390</td>
          <td>21.795014</td>
          <td>24.432263</td>
          <td>22.751537</td>
          <td>23.240985</td>
          <td>24.936787</td>
          <td>21.577719</td>
          <td>22.627917</td>
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
          <td>19.022235</td>
          <td>0.005084</td>
          <td>25.807032</td>
          <td>0.075602</td>
          <td>16.718832</td>
          <td>0.005001</td>
          <td>21.659403</td>
          <td>0.005669</td>
          <td>26.787998</td>
          <td>0.445263</td>
          <td>28.529298</td>
          <td>2.093227</td>
          <td>28.094913</td>
          <td>23.403678</td>
          <td>23.693345</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.365060</td>
          <td>0.005015</td>
          <td>24.032252</td>
          <td>0.016172</td>
          <td>19.793622</td>
          <td>0.005018</td>
          <td>28.638203</td>
          <td>0.958824</td>
          <td>23.545022</td>
          <td>0.028047</td>
          <td>21.284132</td>
          <td>0.009635</td>
          <td>19.751685</td>
          <td>24.696488</td>
          <td>22.518345</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.121584</td>
          <td>0.279171</td>
          <td>22.435557</td>
          <td>0.006274</td>
          <td>22.714454</td>
          <td>0.006486</td>
          <td>23.958412</td>
          <td>0.021263</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.923077</td>
          <td>0.005008</td>
          <td>28.284574</td>
          <td>26.146072</td>
          <td>20.457892</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.684016</td>
          <td>0.005695</td>
          <td>24.068340</td>
          <td>0.016654</td>
          <td>19.845139</td>
          <td>0.005019</td>
          <td>24.619652</td>
          <td>0.037881</td>
          <td>22.424106</td>
          <td>0.011213</td>
          <td>26.656911</td>
          <td>0.778049</td>
          <td>22.445153</td>
          <td>22.609286</td>
          <td>23.535443</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.657246</td>
          <td>0.080252</td>
          <td>21.503022</td>
          <td>0.005305</td>
          <td>22.168841</td>
          <td>0.005625</td>
          <td>22.876957</td>
          <td>0.009262</td>
          <td>26.299823</td>
          <td>0.304286</td>
          <td>19.593599</td>
          <td>0.005348</td>
          <td>23.554278</td>
          <td>24.249611</td>
          <td>23.248854</td>
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
          <td>26.153276</td>
          <td>0.286420</td>
          <td>24.158291</td>
          <td>0.017931</td>
          <td>24.526660</td>
          <td>0.021534</td>
          <td>19.568050</td>
          <td>0.005027</td>
          <td>26.603492</td>
          <td>0.386653</td>
          <td>19.164755</td>
          <td>0.005176</td>
          <td>26.040693</td>
          <td>21.334591</td>
          <td>23.128631</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>24.820113</td>
          <td>0.031610</td>
          <td>22.107088</td>
          <td>0.005565</td>
          <td>21.543845</td>
          <td>0.005555</td>
          <td>25.968035</td>
          <td>0.232126</td>
          <td>23.769074</td>
          <td>0.077119</td>
          <td>29.438678</td>
          <td>26.304011</td>
          <td>26.456370</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.211381</td>
          <td>0.005365</td>
          <td>23.380430</td>
          <td>0.009952</td>
          <td>25.269706</td>
          <td>0.041265</td>
          <td>22.843388</td>
          <td>0.009068</td>
          <td>18.427729</td>
          <td>0.005017</td>
          <td>19.899726</td>
          <td>0.005568</td>
          <td>18.764430</td>
          <td>22.359746</td>
          <td>22.393311</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.261529</td>
          <td>0.056671</td>
          <td>21.865493</td>
          <td>0.005531</td>
          <td>25.483699</td>
          <td>0.049895</td>
          <td>22.696323</td>
          <td>0.008307</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.447323</td>
          <td>0.024017</td>
          <td>23.551888</td>
          <td>24.263885</td>
          <td>20.450035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.081044</td>
          <td>0.020420</td>
          <td>20.419319</td>
          <td>0.005065</td>
          <td>21.798566</td>
          <td>0.005344</td>
          <td>24.483525</td>
          <td>0.033588</td>
          <td>22.743526</td>
          <td>0.014285</td>
          <td>23.218974</td>
          <td>0.047357</td>
          <td>24.936787</td>
          <td>21.577719</td>
          <td>22.627917</td>
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
          <td>19.026954</td>
          <td>25.829397</td>
          <td>16.715267</td>
          <td>21.665284</td>
          <td>26.399003</td>
          <td>26.597665</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.380981</td>
          <td>0.014271</td>
          <td>23.680617</td>
          <td>0.018281</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.366629</td>
          <td>24.039625</td>
          <td>19.791147</td>
          <td>28.708114</td>
          <td>23.528356</td>
          <td>21.266654</td>
          <td>19.755116</td>
          <td>0.005007</td>
          <td>24.736205</td>
          <td>0.046178</td>
          <td>22.523576</td>
          <td>0.007875</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.935219</td>
          <td>22.435471</td>
          <td>22.722660</td>
          <td>23.981500</td>
          <td>28.538375</td>
          <td>16.929203</td>
          <td>29.663725</td>
          <td>1.304650</td>
          <td>25.977839</td>
          <td>0.138111</td>
          <td>20.467489</td>
          <td>0.005083</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.681486</td>
          <td>24.022652</td>
          <td>19.840418</td>
          <td>24.615505</td>
          <td>22.447140</td>
          <td>27.257496</td>
          <td>22.439578</td>
          <td>0.005960</td>
          <td>22.619123</td>
          <td>0.008314</td>
          <td>23.533830</td>
          <td>0.016168</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.765528</td>
          <td>21.498559</td>
          <td>22.175922</td>
          <td>22.879324</td>
          <td>26.575045</td>
          <td>19.591031</td>
          <td>23.542845</td>
          <td>0.010246</td>
          <td>24.233235</td>
          <td>0.029533</td>
          <td>23.264255</td>
          <td>0.013010</td>
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
          <td>27.218123</td>
          <td>24.173920</td>
          <td>24.496385</td>
          <td>19.564515</td>
          <td>26.176967</td>
          <td>19.162790</td>
          <td>26.001342</td>
          <td>0.083412</td>
          <td>21.332133</td>
          <td>0.005397</td>
          <td>23.117118</td>
          <td>0.011625</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.871879</td>
          <td>24.816489</td>
          <td>22.103098</td>
          <td>21.541191</td>
          <td>26.591616</td>
          <td>23.803936</td>
          <td>28.569952</td>
          <td>0.667468</td>
          <td>26.413536</td>
          <td>0.200268</td>
          <td>26.381621</td>
          <td>0.194961</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.202065</td>
          <td>23.389941</td>
          <td>25.248379</td>
          <td>22.843671</td>
          <td>18.422329</td>
          <td>19.899997</td>
          <td>18.762525</td>
          <td>0.005001</td>
          <td>22.355062</td>
          <td>0.007222</td>
          <td>22.386676</td>
          <td>0.007334</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.241948</td>
          <td>21.865467</td>
          <td>25.565814</td>
          <td>22.701558</td>
          <td>29.535579</td>
          <td>22.420394</td>
          <td>23.543288</td>
          <td>0.010250</td>
          <td>24.226823</td>
          <td>0.029366</td>
          <td>20.445868</td>
          <td>0.005080</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.073877</td>
          <td>20.429390</td>
          <td>21.795014</td>
          <td>24.432263</td>
          <td>22.751537</td>
          <td>23.240985</td>
          <td>24.919146</td>
          <td>0.031864</td>
          <td>21.587333</td>
          <td>0.005622</td>
          <td>22.623987</td>
          <td>0.008338</td>
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
          <td>19.026954</td>
          <td>25.829397</td>
          <td>16.715267</td>
          <td>21.665284</td>
          <td>26.399003</td>
          <td>26.597665</td>
          <td>25.530145</td>
          <td>0.524466</td>
          <td>23.469105</td>
          <td>0.081069</td>
          <td>23.651867</td>
          <td>0.103991</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.366629</td>
          <td>24.039625</td>
          <td>19.791147</td>
          <td>28.708114</td>
          <td>23.528356</td>
          <td>21.266654</td>
          <td>19.755340</td>
          <td>0.005985</td>
          <td>24.460451</td>
          <td>0.191511</td>
          <td>22.497682</td>
          <td>0.037338</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.935219</td>
          <td>22.435471</td>
          <td>22.722660</td>
          <td>23.981500</td>
          <td>28.538375</td>
          <td>16.929203</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.136265</td>
          <td>0.333364</td>
          <td>20.455267</td>
          <td>0.007593</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.681486</td>
          <td>24.022652</td>
          <td>19.840418</td>
          <td>24.615505</td>
          <td>22.447140</td>
          <td>27.257496</td>
          <td>22.455267</td>
          <td>0.039301</td>
          <td>22.602306</td>
          <td>0.037492</td>
          <td>23.484034</td>
          <td>0.089727</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.765528</td>
          <td>21.498559</td>
          <td>22.175922</td>
          <td>22.879324</td>
          <td>26.575045</td>
          <td>19.591031</td>
          <td>23.587985</td>
          <td>0.107333</td>
          <td>24.251798</td>
          <td>0.160392</td>
          <td>23.233125</td>
          <td>0.071869</td>
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
          <td>27.218123</td>
          <td>24.173920</td>
          <td>24.496385</td>
          <td>19.564515</td>
          <td>26.176967</td>
          <td>19.162790</td>
          <td>26.149661</td>
          <td>0.804901</td>
          <td>21.332845</td>
          <td>0.012696</td>
          <td>22.999691</td>
          <td>0.058401</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.871879</td>
          <td>24.816489</td>
          <td>22.103098</td>
          <td>21.541191</td>
          <td>26.591616</td>
          <td>23.803936</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.119392</td>
          <td>1.273840</td>
          <td>25.793447</td>
          <td>0.589957</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.202065</td>
          <td>23.389941</td>
          <td>25.248379</td>
          <td>22.843671</td>
          <td>18.422329</td>
          <td>19.899997</td>
          <td>18.756690</td>
          <td>0.005169</td>
          <td>22.320111</td>
          <td>0.029192</td>
          <td>22.375146</td>
          <td>0.033486</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.241948</td>
          <td>21.865467</td>
          <td>25.565814</td>
          <td>22.701558</td>
          <td>29.535579</td>
          <td>22.420394</td>
          <td>23.451412</td>
          <td>0.095211</td>
          <td>24.088310</td>
          <td>0.139365</td>
          <td>20.454378</td>
          <td>0.007589</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.073877</td>
          <td>20.429390</td>
          <td>21.795014</td>
          <td>24.432263</td>
          <td>22.751537</td>
          <td>23.240985</td>
          <td>25.237043</td>
          <td>0.421316</td>
          <td>21.605137</td>
          <td>0.015789</td>
          <td>22.625446</td>
          <td>0.041837</td>
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


