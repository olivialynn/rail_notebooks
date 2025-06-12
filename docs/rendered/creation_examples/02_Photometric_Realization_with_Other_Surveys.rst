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
          <td>22.591899</td>
          <td>19.849955</td>
          <td>24.799013</td>
          <td>19.526217</td>
          <td>20.779886</td>
          <td>26.851862</td>
          <td>28.534170</td>
          <td>23.875700</td>
          <td>23.376236</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.390071</td>
          <td>24.319855</td>
          <td>27.131431</td>
          <td>18.040803</td>
          <td>25.857742</td>
          <td>23.296357</td>
          <td>19.716164</td>
          <td>21.716588</td>
          <td>23.182884</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.646651</td>
          <td>21.829462</td>
          <td>23.275269</td>
          <td>26.364313</td>
          <td>23.982106</td>
          <td>24.582761</td>
          <td>24.557973</td>
          <td>25.388087</td>
          <td>23.465455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.796144</td>
          <td>20.335093</td>
          <td>22.094968</td>
          <td>26.675308</td>
          <td>25.556725</td>
          <td>24.432445</td>
          <td>23.473087</td>
          <td>23.516596</td>
          <td>17.088101</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.012949</td>
          <td>22.445185</td>
          <td>28.988005</td>
          <td>27.974919</td>
          <td>22.707931</td>
          <td>23.344579</td>
          <td>24.546463</td>
          <td>25.187526</td>
          <td>19.232508</td>
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
          <td>24.375265</td>
          <td>21.597926</td>
          <td>25.736851</td>
          <td>24.427826</td>
          <td>22.729918</td>
          <td>27.698510</td>
          <td>16.630134</td>
          <td>21.602034</td>
          <td>25.778904</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.952069</td>
          <td>25.590815</td>
          <td>16.561371</td>
          <td>26.249507</td>
          <td>22.425872</td>
          <td>25.430710</td>
          <td>16.264165</td>
          <td>24.341714</td>
          <td>22.004377</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.529334</td>
          <td>23.463186</td>
          <td>23.565323</td>
          <td>15.658476</td>
          <td>18.668932</td>
          <td>21.679863</td>
          <td>19.805623</td>
          <td>22.713130</td>
          <td>19.075957</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.082079</td>
          <td>22.810399</td>
          <td>29.543632</td>
          <td>21.147522</td>
          <td>23.406953</td>
          <td>20.226270</td>
          <td>21.589699</td>
          <td>23.702835</td>
          <td>23.678302</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.931727</td>
          <td>14.449855</td>
          <td>30.106211</td>
          <td>20.501922</td>
          <td>21.308764</td>
          <td>20.044776</td>
          <td>22.415970</td>
          <td>22.138238</td>
          <td>23.525495</td>
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
          <td>22.638130</td>
          <td>0.014326</td>
          <td>19.851376</td>
          <td>0.005032</td>
          <td>24.761883</td>
          <td>0.026386</td>
          <td>19.524197</td>
          <td>0.005026</td>
          <td>20.775209</td>
          <td>0.005529</td>
          <td>26.548435</td>
          <td>0.723920</td>
          <td>28.534170</td>
          <td>23.875700</td>
          <td>23.376236</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.493336</td>
          <td>0.069492</td>
          <td>24.309750</td>
          <td>0.020352</td>
          <td>27.285646</td>
          <td>0.238180</td>
          <td>18.038223</td>
          <td>0.005004</td>
          <td>25.767858</td>
          <td>0.196400</td>
          <td>23.232771</td>
          <td>0.047941</td>
          <td>19.716164</td>
          <td>21.716588</td>
          <td>23.182884</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.428196</td>
          <td>0.156616</td>
          <td>21.819940</td>
          <td>0.005495</td>
          <td>23.269197</td>
          <td>0.008402</td>
          <td>26.465528</td>
          <td>0.190656</td>
          <td>23.974630</td>
          <td>0.040959</td>
          <td>24.403931</td>
          <td>0.134387</td>
          <td>24.557973</td>
          <td>25.388087</td>
          <td>23.465455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.797220</td>
          <td>0.016216</td>
          <td>20.337996</td>
          <td>0.005058</td>
          <td>22.103070</td>
          <td>0.005562</td>
          <td>26.711314</td>
          <td>0.234130</td>
          <td>26.000820</td>
          <td>0.238506</td>
          <td>24.394770</td>
          <td>0.133328</td>
          <td>23.473087</td>
          <td>23.516596</td>
          <td>17.088101</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.019670</td>
          <td>0.019410</td>
          <td>22.446516</td>
          <td>0.006296</td>
          <td>28.442881</td>
          <td>0.583568</td>
          <td>28.316900</td>
          <td>0.782092</td>
          <td>22.718780</td>
          <td>0.014008</td>
          <td>23.373444</td>
          <td>0.054317</td>
          <td>24.546463</td>
          <td>25.187526</td>
          <td>19.232508</td>
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
          <td>24.490781</td>
          <td>0.069336</td>
          <td>21.593713</td>
          <td>0.005350</td>
          <td>25.763289</td>
          <td>0.063947</td>
          <td>24.424020</td>
          <td>0.031872</td>
          <td>22.738025</td>
          <td>0.014223</td>
          <td>28.429736</td>
          <td>2.008729</td>
          <td>16.630134</td>
          <td>21.602034</td>
          <td>25.778904</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.954938</td>
          <td>0.018408</td>
          <td>25.567442</td>
          <td>0.061173</td>
          <td>16.574968</td>
          <td>0.005001</td>
          <td>26.120415</td>
          <td>0.142050</td>
          <td>22.446449</td>
          <td>0.011396</td>
          <td>25.578921</td>
          <td>0.356172</td>
          <td>16.264165</td>
          <td>24.341714</td>
          <td>22.004377</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.528746</td>
          <td>0.005049</td>
          <td>23.449870</td>
          <td>0.010428</td>
          <td>23.555775</td>
          <td>0.010057</td>
          <td>15.654955</td>
          <td>0.005000</td>
          <td>18.665055</td>
          <td>0.005023</td>
          <td>21.686710</td>
          <td>0.012836</td>
          <td>19.805623</td>
          <td>22.713130</td>
          <td>19.075957</td>
        </tr>
        <tr>
          <th>998</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.816635</td>
          <td>0.007252</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.154102</td>
          <td>0.005296</td>
          <td>23.387713</td>
          <td>0.024459</td>
          <td>20.224841</td>
          <td>0.005956</td>
          <td>21.589699</td>
          <td>23.702835</td>
          <td>23.678302</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.935620</td>
          <td>0.005255</td>
          <td>14.444890</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.507540</td>
          <td>0.005107</td>
          <td>21.313267</td>
          <td>0.006248</td>
          <td>20.051030</td>
          <td>0.005725</td>
          <td>22.415970</td>
          <td>22.138238</td>
          <td>23.525495</td>
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
          <td>22.591899</td>
          <td>19.849955</td>
          <td>24.799013</td>
          <td>19.526217</td>
          <td>20.779886</td>
          <td>26.851862</td>
          <td>29.177729</td>
          <td>0.988910</td>
          <td>23.886283</td>
          <td>0.021800</td>
          <td>23.391172</td>
          <td>0.014389</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.390071</td>
          <td>24.319855</td>
          <td>27.131431</td>
          <td>18.040803</td>
          <td>25.857742</td>
          <td>23.296357</td>
          <td>19.718199</td>
          <td>0.005007</td>
          <td>21.715161</td>
          <td>0.005776</td>
          <td>23.178542</td>
          <td>0.012177</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.646651</td>
          <td>21.829462</td>
          <td>23.275269</td>
          <td>26.364313</td>
          <td>23.982106</td>
          <td>24.582761</td>
          <td>24.555428</td>
          <td>0.023147</td>
          <td>25.345708</td>
          <td>0.079408</td>
          <td>23.450298</td>
          <td>0.015095</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.796144</td>
          <td>20.335093</td>
          <td>22.094968</td>
          <td>26.675308</td>
          <td>25.556725</td>
          <td>24.432445</td>
          <td>23.482784</td>
          <td>0.009830</td>
          <td>23.488624</td>
          <td>0.015576</td>
          <td>17.092880</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.012949</td>
          <td>22.445185</td>
          <td>28.988005</td>
          <td>27.974919</td>
          <td>22.707931</td>
          <td>23.344579</td>
          <td>24.562619</td>
          <td>0.023292</td>
          <td>25.303715</td>
          <td>0.076510</td>
          <td>19.235577</td>
          <td>0.005009</td>
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
          <td>24.375265</td>
          <td>21.597926</td>
          <td>25.736851</td>
          <td>24.427826</td>
          <td>22.729918</td>
          <td>27.698510</td>
          <td>16.637303</td>
          <td>0.005000</td>
          <td>21.601741</td>
          <td>0.005638</td>
          <td>25.842238</td>
          <td>0.122795</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.952069</td>
          <td>25.590815</td>
          <td>16.561371</td>
          <td>26.249507</td>
          <td>22.425872</td>
          <td>25.430710</td>
          <td>16.263206</td>
          <td>0.005000</td>
          <td>24.336798</td>
          <td>0.032367</td>
          <td>22.008968</td>
          <td>0.006274</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.529334</td>
          <td>23.463186</td>
          <td>23.565323</td>
          <td>15.658476</td>
          <td>18.668932</td>
          <td>21.679863</td>
          <td>19.800326</td>
          <td>0.005008</td>
          <td>22.716254</td>
          <td>0.008818</td>
          <td>19.077736</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.082079</td>
          <td>22.810399</td>
          <td>29.543632</td>
          <td>21.147522</td>
          <td>23.406953</td>
          <td>20.226270</td>
          <td>21.590217</td>
          <td>0.005215</td>
          <td>23.714910</td>
          <td>0.018821</td>
          <td>23.690173</td>
          <td>0.018430</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.931727</td>
          <td>14.449855</td>
          <td>30.106211</td>
          <td>20.501922</td>
          <td>21.308764</td>
          <td>20.044776</td>
          <td>22.410598</td>
          <td>0.005914</td>
          <td>22.135326</td>
          <td>0.006567</td>
          <td>23.559324</td>
          <td>0.016513</td>
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
          <td>22.591899</td>
          <td>19.849955</td>
          <td>24.799013</td>
          <td>19.526217</td>
          <td>20.779886</td>
          <td>26.851862</td>
          <td>28.349545</td>
          <td>2.425158</td>
          <td>23.868986</td>
          <td>0.115206</td>
          <td>23.355317</td>
          <td>0.080086</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.390071</td>
          <td>24.319855</td>
          <td>27.131431</td>
          <td>18.040803</td>
          <td>25.857742</td>
          <td>23.296357</td>
          <td>19.718003</td>
          <td>0.005925</td>
          <td>21.728812</td>
          <td>0.017500</td>
          <td>23.153366</td>
          <td>0.066956</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.646651</td>
          <td>21.829462</td>
          <td>23.275269</td>
          <td>26.364313</td>
          <td>23.982106</td>
          <td>24.582761</td>
          <td>24.623458</td>
          <td>0.259038</td>
          <td>25.990323</td>
          <td>0.631625</td>
          <td>23.538531</td>
          <td>0.094139</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.796144</td>
          <td>20.335093</td>
          <td>22.094968</td>
          <td>26.675308</td>
          <td>25.556725</td>
          <td>24.432445</td>
          <td>23.369883</td>
          <td>0.088615</td>
          <td>23.462000</td>
          <td>0.080561</td>
          <td>17.090972</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.012949</td>
          <td>22.445185</td>
          <td>28.988005</td>
          <td>27.974919</td>
          <td>22.707931</td>
          <td>23.344579</td>
          <td>24.394913</td>
          <td>0.214403</td>
          <td>24.982429</td>
          <td>0.294772</td>
          <td>19.225369</td>
          <td>0.005329</td>
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
          <td>24.375265</td>
          <td>21.597926</td>
          <td>25.736851</td>
          <td>24.427826</td>
          <td>22.729918</td>
          <td>27.698510</td>
          <td>16.626912</td>
          <td>0.005003</td>
          <td>21.592033</td>
          <td>0.015620</td>
          <td>25.680051</td>
          <td>0.543856</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.952069</td>
          <td>25.590815</td>
          <td>16.561371</td>
          <td>26.249507</td>
          <td>22.425872</td>
          <td>25.430710</td>
          <td>16.271316</td>
          <td>0.005002</td>
          <td>24.352396</td>
          <td>0.174761</td>
          <td>21.989821</td>
          <td>0.023850</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.529334</td>
          <td>23.463186</td>
          <td>23.565323</td>
          <td>15.658476</td>
          <td>18.668932</td>
          <td>21.679863</td>
          <td>19.807737</td>
          <td>0.006076</td>
          <td>22.748744</td>
          <td>0.042714</td>
          <td>19.076181</td>
          <td>0.005252</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.082079</td>
          <td>22.810399</td>
          <td>29.543632</td>
          <td>21.147522</td>
          <td>23.406953</td>
          <td>20.226270</td>
          <td>21.578126</td>
          <td>0.018243</td>
          <td>23.788706</td>
          <td>0.107401</td>
          <td>23.811585</td>
          <td>0.119563</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.931727</td>
          <td>14.449855</td>
          <td>30.106211</td>
          <td>20.501922</td>
          <td>21.308764</td>
          <td>20.044776</td>
          <td>22.440382</td>
          <td>0.038784</td>
          <td>22.118269</td>
          <td>0.024450</td>
          <td>23.546437</td>
          <td>0.094796</td>
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


