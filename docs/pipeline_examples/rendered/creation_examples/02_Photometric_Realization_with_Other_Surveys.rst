Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


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
          <td>20.935975</td>
          <td>26.589256</td>
          <td>21.916374</td>
          <td>24.557194</td>
          <td>20.295741</td>
          <td>25.725951</td>
          <td>25.178057</td>
          <td>22.168790</td>
          <td>23.689540</td>
        </tr>
        <tr>
          <th>1</th>
          <td>15.438706</td>
          <td>21.661395</td>
          <td>19.368127</td>
          <td>25.870778</td>
          <td>26.670071</td>
          <td>20.574275</td>
          <td>23.451664</td>
          <td>26.420671</td>
          <td>19.620160</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.306725</td>
          <td>20.764530</td>
          <td>22.547389</td>
          <td>28.115088</td>
          <td>24.783695</td>
          <td>21.316483</td>
          <td>20.254560</td>
          <td>20.640615</td>
          <td>22.958047</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.970092</td>
          <td>22.440011</td>
          <td>26.939880</td>
          <td>25.399090</td>
          <td>23.693283</td>
          <td>18.287292</td>
          <td>25.146011</td>
          <td>25.823853</td>
          <td>21.902673</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.501909</td>
          <td>25.946372</td>
          <td>21.714462</td>
          <td>24.213176</td>
          <td>21.227433</td>
          <td>27.273172</td>
          <td>21.763581</td>
          <td>18.274677</td>
          <td>23.623509</td>
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
          <td>24.999206</td>
          <td>27.748462</td>
          <td>23.292906</td>
          <td>25.391481</td>
          <td>25.739932</td>
          <td>20.064377</td>
          <td>25.218886</td>
          <td>21.760672</td>
          <td>26.793696</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.563615</td>
          <td>22.169453</td>
          <td>22.880895</td>
          <td>23.036186</td>
          <td>27.818139</td>
          <td>25.482409</td>
          <td>22.818514</td>
          <td>20.763417</td>
          <td>20.464376</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.658487</td>
          <td>20.520089</td>
          <td>22.713383</td>
          <td>21.739692</td>
          <td>22.120447</td>
          <td>27.555182</td>
          <td>23.972961</td>
          <td>14.160476</td>
          <td>21.623867</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.932424</td>
          <td>19.714576</td>
          <td>19.978818</td>
          <td>21.162381</td>
          <td>17.151503</td>
          <td>22.064316</td>
          <td>23.011314</td>
          <td>20.830055</td>
          <td>21.599361</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.999100</td>
          <td>22.967581</td>
          <td>23.295336</td>
          <td>23.353017</td>
          <td>22.202272</td>
          <td>20.678222</td>
          <td>24.213246</td>
          <td>22.842812</td>
          <td>23.907240</td>
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
          <td>20.942203</td>
          <td>0.005995</td>
          <td>26.407071</td>
          <td>0.127810</td>
          <td>21.919385</td>
          <td>0.005418</td>
          <td>24.583184</td>
          <td>0.036679</td>
          <td>20.305765</td>
          <td>0.005249</td>
          <td>25.796228</td>
          <td>0.421413</td>
          <td>25.178057</td>
          <td>22.168790</td>
          <td>23.689540</td>
        </tr>
        <tr>
          <th>1</th>
          <td>15.442213</td>
          <td>0.005002</td>
          <td>21.663178</td>
          <td>0.005389</td>
          <td>19.374311</td>
          <td>0.005011</td>
          <td>25.854032</td>
          <td>0.112764</td>
          <td>26.279650</td>
          <td>0.299394</td>
          <td>20.584021</td>
          <td>0.006677</td>
          <td>23.451664</td>
          <td>26.420671</td>
          <td>19.620160</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.302149</td>
          <td>0.011216</td>
          <td>20.771969</td>
          <td>0.005105</td>
          <td>22.536197</td>
          <td>0.006124</td>
          <td>28.694691</td>
          <td>0.992305</td>
          <td>24.680348</td>
          <td>0.076576</td>
          <td>21.316355</td>
          <td>0.009841</td>
          <td>20.254560</td>
          <td>20.640615</td>
          <td>22.958047</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.974428</td>
          <td>0.018704</td>
          <td>22.441824</td>
          <td>0.006287</td>
          <td>27.106030</td>
          <td>0.205111</td>
          <td>25.452575</td>
          <td>0.079274</td>
          <td>23.723451</td>
          <td>0.032803</td>
          <td>18.292453</td>
          <td>0.005048</td>
          <td>25.146011</td>
          <td>25.823853</td>
          <td>21.902673</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.183026</td>
          <td>0.293375</td>
          <td>25.989220</td>
          <td>0.088758</td>
          <td>21.719358</td>
          <td>0.005303</td>
          <td>24.252308</td>
          <td>0.027415</td>
          <td>21.234875</td>
          <td>0.006103</td>
          <td>26.201480</td>
          <td>0.568873</td>
          <td>21.763581</td>
          <td>18.274677</td>
          <td>23.623509</td>
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
          <td>25.112930</td>
          <td>0.119422</td>
          <td>28.723741</td>
          <td>0.780536</td>
          <td>23.299194</td>
          <td>0.008550</td>
          <td>25.372832</td>
          <td>0.073881</td>
          <td>25.470770</td>
          <td>0.152589</td>
          <td>20.059222</td>
          <td>0.005734</td>
          <td>25.218886</td>
          <td>21.760672</td>
          <td>26.793696</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.560994</td>
          <td>0.007355</td>
          <td>22.166576</td>
          <td>0.005844</td>
          <td>22.885594</td>
          <td>0.006934</td>
          <td>23.004842</td>
          <td>0.010072</td>
          <td>27.157852</td>
          <td>0.584163</td>
          <td>25.808452</td>
          <td>0.425358</td>
          <td>22.818514</td>
          <td>20.763417</td>
          <td>20.464376</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.660654</td>
          <td>0.080493</td>
          <td>20.523928</td>
          <td>0.005075</td>
          <td>22.707756</td>
          <td>0.006471</td>
          <td>21.747671</td>
          <td>0.005771</td>
          <td>22.120068</td>
          <td>0.009133</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.972961</td>
          <td>14.160476</td>
          <td>21.623867</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.921488</td>
          <td>0.237034</td>
          <td>19.715535</td>
          <td>0.005027</td>
          <td>19.976943</td>
          <td>0.005023</td>
          <td>21.168154</td>
          <td>0.005303</td>
          <td>17.152950</td>
          <td>0.005004</td>
          <td>22.091629</td>
          <td>0.017753</td>
          <td>23.011314</td>
          <td>20.830055</td>
          <td>21.599361</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.996803</td>
          <td>0.005082</td>
          <td>22.966226</td>
          <td>0.007796</td>
          <td>23.289725</td>
          <td>0.008503</td>
          <td>23.343613</td>
          <td>0.012870</td>
          <td>22.197652</td>
          <td>0.009597</td>
          <td>20.684257</td>
          <td>0.006954</td>
          <td>24.213246</td>
          <td>22.842812</td>
          <td>23.907240</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


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
          <td>20.935975</td>
          <td>26.589256</td>
          <td>21.916374</td>
          <td>24.557194</td>
          <td>20.295741</td>
          <td>25.725951</td>
          <td>25.201220</td>
          <td>0.040943</td>
          <td>22.183272</td>
          <td>0.006693</td>
          <td>23.696303</td>
          <td>0.018526</td>
        </tr>
        <tr>
          <th>1</th>
          <td>15.438706</td>
          <td>21.661395</td>
          <td>19.368127</td>
          <td>25.870778</td>
          <td>26.670071</td>
          <td>20.574275</td>
          <td>23.455764</td>
          <td>0.009652</td>
          <td>26.394723</td>
          <td>0.197124</td>
          <td>19.625634</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.306725</td>
          <td>20.764530</td>
          <td>22.547389</td>
          <td>28.115088</td>
          <td>24.783695</td>
          <td>21.316483</td>
          <td>20.260319</td>
          <td>0.005019</td>
          <td>20.639253</td>
          <td>0.005114</td>
          <td>22.962963</td>
          <td>0.010392</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.970092</td>
          <td>22.440011</td>
          <td>26.939880</td>
          <td>25.399090</td>
          <td>23.693283</td>
          <td>18.287292</td>
          <td>25.096786</td>
          <td>0.037308</td>
          <td>25.699625</td>
          <td>0.108432</td>
          <td>21.902392</td>
          <td>0.006067</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.501909</td>
          <td>25.946372</td>
          <td>21.714462</td>
          <td>24.213176</td>
          <td>21.227433</td>
          <td>27.273172</td>
          <td>21.763636</td>
          <td>0.005294</td>
          <td>18.274973</td>
          <td>0.005001</td>
          <td>23.623449</td>
          <td>0.017422</td>
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
          <td>24.999206</td>
          <td>27.748462</td>
          <td>23.292906</td>
          <td>25.391481</td>
          <td>25.739932</td>
          <td>20.064377</td>
          <td>25.235277</td>
          <td>0.042205</td>
          <td>21.764776</td>
          <td>0.005845</td>
          <td>26.530958</td>
          <td>0.220948</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.563615</td>
          <td>22.169453</td>
          <td>22.880895</td>
          <td>23.036186</td>
          <td>27.818139</td>
          <td>25.482409</td>
          <td>22.823619</td>
          <td>0.006806</td>
          <td>20.759422</td>
          <td>0.005142</td>
          <td>20.461565</td>
          <td>0.005082</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.658487</td>
          <td>20.520089</td>
          <td>22.713383</td>
          <td>21.739692</td>
          <td>22.120447</td>
          <td>27.555182</td>
          <td>23.985059</td>
          <td>0.014318</td>
          <td>14.165822</td>
          <td>0.005000</td>
          <td>21.623347</td>
          <td>0.005662</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.932424</td>
          <td>19.714576</td>
          <td>19.978818</td>
          <td>21.162381</td>
          <td>17.151503</td>
          <td>22.064316</td>
          <td>23.006276</td>
          <td>0.007405</td>
          <td>20.825883</td>
          <td>0.005160</td>
          <td>21.607178</td>
          <td>0.005644</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.999100</td>
          <td>22.967581</td>
          <td>23.295336</td>
          <td>23.353017</td>
          <td>22.202272</td>
          <td>20.678222</td>
          <td>24.213448</td>
          <td>0.017276</td>
          <td>22.833695</td>
          <td>0.009511</td>
          <td>23.892156</td>
          <td>0.021911</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


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
          <td>20.935975</td>
          <td>26.589256</td>
          <td>21.916374</td>
          <td>24.557194</td>
          <td>20.295741</td>
          <td>25.725951</td>
          <td>26.022318</td>
          <td>0.740100</td>
          <td>22.133372</td>
          <td>0.024774</td>
          <td>23.682209</td>
          <td>0.106792</td>
        </tr>
        <tr>
          <th>1</th>
          <td>15.438706</td>
          <td>21.661395</td>
          <td>19.368127</td>
          <td>25.870778</td>
          <td>26.670071</td>
          <td>20.574275</td>
          <td>23.594627</td>
          <td>0.107959</td>
          <td>25.740205</td>
          <td>0.528330</td>
          <td>19.613688</td>
          <td>0.005651</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.306725</td>
          <td>20.764530</td>
          <td>22.547389</td>
          <td>28.115088</td>
          <td>24.783695</td>
          <td>21.316483</td>
          <td>20.252623</td>
          <td>0.007213</td>
          <td>20.641808</td>
          <td>0.007955</td>
          <td>22.955924</td>
          <td>0.056168</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.970092</td>
          <td>22.440011</td>
          <td>26.939880</td>
          <td>25.399090</td>
          <td>23.693283</td>
          <td>18.287292</td>
          <td>24.783229</td>
          <td>0.294962</td>
          <td>26.257555</td>
          <td>0.757657</td>
          <td>21.922335</td>
          <td>0.022491</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.501909</td>
          <td>25.946372</td>
          <td>21.714462</td>
          <td>24.213176</td>
          <td>21.227433</td>
          <td>27.273172</td>
          <td>21.765831</td>
          <td>0.021418</td>
          <td>18.276430</td>
          <td>0.005049</td>
          <td>23.445305</td>
          <td>0.086714</td>
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
          <td>24.999206</td>
          <td>27.748462</td>
          <td>23.292906</td>
          <td>25.391481</td>
          <td>25.739932</td>
          <td>20.064377</td>
          <td>25.849236</td>
          <td>0.658004</td>
          <td>21.774878</td>
          <td>0.018193</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.563615</td>
          <td>22.169453</td>
          <td>22.880895</td>
          <td>23.036186</td>
          <td>27.818139</td>
          <td>25.482409</td>
          <td>22.839594</td>
          <td>0.055357</td>
          <td>20.778661</td>
          <td>0.008616</td>
          <td>20.461345</td>
          <td>0.007617</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.658487</td>
          <td>20.520089</td>
          <td>22.713383</td>
          <td>21.739692</td>
          <td>22.120447</td>
          <td>27.555182</td>
          <td>24.018509</td>
          <td>0.155884</td>
          <td>14.162204</td>
          <td>0.005000</td>
          <td>21.641084</td>
          <td>0.017682</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.932424</td>
          <td>19.714576</td>
          <td>19.978818</td>
          <td>21.162381</td>
          <td>17.151503</td>
          <td>22.064316</td>
          <td>22.881672</td>
          <td>0.057471</td>
          <td>20.827657</td>
          <td>0.008881</td>
          <td>21.607297</td>
          <td>0.017187</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.999100</td>
          <td>22.967581</td>
          <td>23.295336</td>
          <td>23.353017</td>
          <td>22.202272</td>
          <td>20.678222</td>
          <td>24.306101</td>
          <td>0.199020</td>
          <td>22.881904</td>
          <td>0.048098</td>
          <td>23.856473</td>
          <td>0.124324</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


