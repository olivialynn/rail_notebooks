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
          <td>20.307771</td>
          <td>26.571516</td>
          <td>21.776998</td>
          <td>22.795955</td>
          <td>27.954374</td>
          <td>27.076047</td>
          <td>25.030525</td>
          <td>23.088219</td>
          <td>19.060450</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.215800</td>
          <td>26.531478</td>
          <td>24.290233</td>
          <td>23.873880</td>
          <td>27.852661</td>
          <td>26.882575</td>
          <td>28.715946</td>
          <td>26.457785</td>
          <td>21.747149</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.048029</td>
          <td>28.970898</td>
          <td>26.799743</td>
          <td>29.307607</td>
          <td>17.484309</td>
          <td>25.157580</td>
          <td>22.071059</td>
          <td>19.372609</td>
          <td>25.956791</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.737882</td>
          <td>24.167506</td>
          <td>22.396917</td>
          <td>22.627620</td>
          <td>24.566196</td>
          <td>23.854800</td>
          <td>22.287821</td>
          <td>21.947197</td>
          <td>28.812923</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.514698</td>
          <td>25.367791</td>
          <td>22.997900</td>
          <td>24.142810</td>
          <td>21.211320</td>
          <td>24.944115</td>
          <td>14.411205</td>
          <td>27.050683</td>
          <td>19.576489</td>
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
          <td>23.702896</td>
          <td>28.042930</td>
          <td>20.613329</td>
          <td>21.645528</td>
          <td>17.771709</td>
          <td>25.146339</td>
          <td>25.343271</td>
          <td>22.857758</td>
          <td>23.201616</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.028190</td>
          <td>24.987648</td>
          <td>15.057559</td>
          <td>28.553012</td>
          <td>22.403627</td>
          <td>26.866953</td>
          <td>25.811514</td>
          <td>21.066795</td>
          <td>21.216273</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.028523</td>
          <td>22.861042</td>
          <td>16.051766</td>
          <td>23.185336</td>
          <td>23.308318</td>
          <td>20.551972</td>
          <td>19.479990</td>
          <td>22.298639</td>
          <td>25.549019</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.012086</td>
          <td>21.215741</td>
          <td>22.623299</td>
          <td>25.130073</td>
          <td>24.738685</td>
          <td>26.270691</td>
          <td>24.082012</td>
          <td>22.126880</td>
          <td>20.618954</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.729510</td>
          <td>22.524696</td>
          <td>22.947697</td>
          <td>22.571140</td>
          <td>23.093957</td>
          <td>23.900048</td>
          <td>19.475304</td>
          <td>24.608333</td>
          <td>22.726973</td>
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
          <td>20.314907</td>
          <td>0.005420</td>
          <td>26.635724</td>
          <td>0.155611</td>
          <td>21.779873</td>
          <td>0.005334</td>
          <td>22.787077</td>
          <td>0.008761</td>
          <td>27.252927</td>
          <td>0.624727</td>
          <td>26.175617</td>
          <td>0.558403</td>
          <td>25.030525</td>
          <td>23.088219</td>
          <td>19.060450</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.753036</td>
          <td>1.584280</td>
          <td>26.514390</td>
          <td>0.140219</td>
          <td>24.293199</td>
          <td>0.017683</td>
          <td>23.872243</td>
          <td>0.019761</td>
          <td>27.193972</td>
          <td>0.599334</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.715946</td>
          <td>26.457785</td>
          <td>21.747149</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.070043</td>
          <td>0.047886</td>
          <td>29.407602</td>
          <td>1.183722</td>
          <td>26.812960</td>
          <td>0.160039</td>
          <td>28.165618</td>
          <td>0.707071</td>
          <td>17.488791</td>
          <td>0.005006</td>
          <td>25.025447</td>
          <td>0.227678</td>
          <td>22.071059</td>
          <td>19.372609</td>
          <td>25.956791</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.678343</td>
          <td>0.081751</td>
          <td>24.172841</td>
          <td>0.018148</td>
          <td>22.406601</td>
          <td>0.005915</td>
          <td>22.629409</td>
          <td>0.008003</td>
          <td>24.459789</td>
          <td>0.062996</td>
          <td>23.880923</td>
          <td>0.085116</td>
          <td>22.287821</td>
          <td>21.947197</td>
          <td>28.812923</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.515030</td>
          <td>0.007211</td>
          <td>25.357868</td>
          <td>0.050810</td>
          <td>22.995757</td>
          <td>0.007283</td>
          <td>24.159427</td>
          <td>0.025285</td>
          <td>21.213950</td>
          <td>0.006067</td>
          <td>24.773781</td>
          <td>0.184388</td>
          <td>14.411205</td>
          <td>27.050683</td>
          <td>19.576489</td>
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
          <td>23.641055</td>
          <td>0.032904</td>
          <td>28.722356</td>
          <td>0.779826</td>
          <td>20.618823</td>
          <td>0.005056</td>
          <td>21.644486</td>
          <td>0.005653</td>
          <td>17.776125</td>
          <td>0.005008</td>
          <td>24.877123</td>
          <td>0.201164</td>
          <td>25.343271</td>
          <td>22.857758</td>
          <td>23.201616</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.103164</td>
          <td>0.590772</td>
          <td>24.994831</td>
          <td>0.036858</td>
          <td>15.054703</td>
          <td>0.005000</td>
          <td>28.718861</td>
          <td>1.006846</td>
          <td>22.418923</td>
          <td>0.011171</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.811514</td>
          <td>21.066795</td>
          <td>21.216273</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.033190</td>
          <td>0.009421</td>
          <td>22.852358</td>
          <td>0.007373</td>
          <td>16.057150</td>
          <td>0.005000</td>
          <td>23.188605</td>
          <td>0.011462</td>
          <td>23.316493</td>
          <td>0.023000</td>
          <td>20.554623</td>
          <td>0.006603</td>
          <td>19.479990</td>
          <td>22.298639</td>
          <td>25.549019</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.025832</td>
          <td>0.110732</td>
          <td>21.212397</td>
          <td>0.005198</td>
          <td>22.628612</td>
          <td>0.006300</td>
          <td>25.067686</td>
          <td>0.056374</td>
          <td>24.733084</td>
          <td>0.080226</td>
          <td>25.763434</td>
          <td>0.410976</td>
          <td>24.082012</td>
          <td>22.126880</td>
          <td>20.618954</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.726922</td>
          <td>0.005008</td>
          <td>22.522196</td>
          <td>0.006453</td>
          <td>22.945521</td>
          <td>0.007117</td>
          <td>22.575362</td>
          <td>0.007776</td>
          <td>23.133434</td>
          <td>0.019673</td>
          <td>23.759836</td>
          <td>0.076492</td>
          <td>19.475304</td>
          <td>24.608333</td>
          <td>22.726973</td>
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
          <td>20.307771</td>
          <td>26.571516</td>
          <td>21.776998</td>
          <td>22.795955</td>
          <td>27.954374</td>
          <td>27.076047</td>
          <td>25.042440</td>
          <td>0.035548</td>
          <td>23.096012</td>
          <td>0.011444</td>
          <td>19.060814</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.215800</td>
          <td>26.531478</td>
          <td>24.290233</td>
          <td>23.873880</td>
          <td>27.852661</td>
          <td>26.882575</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.669930</td>
          <td>0.247899</td>
          <td>21.748503</td>
          <td>0.005822</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.048029</td>
          <td>28.970898</td>
          <td>26.799743</td>
          <td>29.307607</td>
          <td>17.484309</td>
          <td>25.157580</td>
          <td>22.063998</td>
          <td>0.005502</td>
          <td>19.371953</td>
          <td>0.005011</td>
          <td>25.867742</td>
          <td>0.125547</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.737882</td>
          <td>24.167506</td>
          <td>22.396917</td>
          <td>22.627620</td>
          <td>24.566196</td>
          <td>23.854800</td>
          <td>22.286453</td>
          <td>0.005739</td>
          <td>21.955980</td>
          <td>0.006167</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.514698</td>
          <td>25.367791</td>
          <td>22.997900</td>
          <td>24.142810</td>
          <td>21.211320</td>
          <td>24.944115</td>
          <td>14.417296</td>
          <td>0.005000</td>
          <td>27.153181</td>
          <td>0.365514</td>
          <td>19.584860</td>
          <td>0.005017</td>
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
          <td>23.702896</td>
          <td>28.042930</td>
          <td>20.613329</td>
          <td>21.645528</td>
          <td>17.771709</td>
          <td>25.146339</td>
          <td>25.376873</td>
          <td>0.047883</td>
          <td>22.855337</td>
          <td>0.009650</td>
          <td>23.196399</td>
          <td>0.012344</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.028190</td>
          <td>24.987648</td>
          <td>15.057559</td>
          <td>28.553012</td>
          <td>22.403627</td>
          <td>26.866953</td>
          <td>25.904399</td>
          <td>0.076556</td>
          <td>21.058885</td>
          <td>0.005244</td>
          <td>21.213299</td>
          <td>0.005322</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.028523</td>
          <td>22.861042</td>
          <td>16.051766</td>
          <td>23.185336</td>
          <td>23.308318</td>
          <td>20.551972</td>
          <td>19.479604</td>
          <td>0.005005</td>
          <td>22.308821</td>
          <td>0.007067</td>
          <td>25.645434</td>
          <td>0.103406</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.012086</td>
          <td>21.215741</td>
          <td>22.623299</td>
          <td>25.130073</td>
          <td>24.738685</td>
          <td>26.270691</td>
          <td>24.083855</td>
          <td>0.015515</td>
          <td>22.118250</td>
          <td>0.006524</td>
          <td>20.621437</td>
          <td>0.005110</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.729510</td>
          <td>22.524696</td>
          <td>22.947697</td>
          <td>22.571140</td>
          <td>23.093957</td>
          <td>23.900048</td>
          <td>19.486798</td>
          <td>0.005005</td>
          <td>24.531378</td>
          <td>0.038474</td>
          <td>22.722418</td>
          <td>0.008852</td>
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
          <td>20.307771</td>
          <td>26.571516</td>
          <td>21.776998</td>
          <td>22.795955</td>
          <td>27.954374</td>
          <td>27.076047</td>
          <td>26.885145</td>
          <td>1.250304</td>
          <td>22.939790</td>
          <td>0.050645</td>
          <td>19.061485</td>
          <td>0.005245</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.215800</td>
          <td>26.531478</td>
          <td>24.290233</td>
          <td>23.873880</td>
          <td>27.852661</td>
          <td>26.882575</td>
          <td>26.142219</td>
          <td>0.801011</td>
          <td>24.888561</td>
          <td>0.273183</td>
          <td>21.761168</td>
          <td>0.019577</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.048029</td>
          <td>28.970898</td>
          <td>26.799743</td>
          <td>29.307607</td>
          <td>17.484309</td>
          <td>25.157580</td>
          <td>22.060955</td>
          <td>0.027709</td>
          <td>19.368701</td>
          <td>0.005355</td>
          <td>26.007131</td>
          <td>0.684699</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.737882</td>
          <td>24.167506</td>
          <td>22.396917</td>
          <td>22.627620</td>
          <td>24.566196</td>
          <td>23.854800</td>
          <td>22.338061</td>
          <td>0.035410</td>
          <td>21.936662</td>
          <td>0.020886</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.514698</td>
          <td>25.367791</td>
          <td>22.997900</td>
          <td>24.142810</td>
          <td>21.211320</td>
          <td>24.944115</td>
          <td>14.417321</td>
          <td>0.005000</td>
          <td>25.876802</td>
          <td>0.583011</td>
          <td>19.585887</td>
          <td>0.005621</td>
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
          <td>23.702896</td>
          <td>28.042930</td>
          <td>20.613329</td>
          <td>21.645528</td>
          <td>17.771709</td>
          <td>25.146339</td>
          <td>24.982448</td>
          <td>0.345764</td>
          <td>22.970106</td>
          <td>0.052033</td>
          <td>23.188147</td>
          <td>0.069056</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.028190</td>
          <td>24.987648</td>
          <td>15.057559</td>
          <td>28.553012</td>
          <td>22.403627</td>
          <td>26.866953</td>
          <td>25.355855</td>
          <td>0.460961</td>
          <td>21.053928</td>
          <td>0.010326</td>
          <td>21.216293</td>
          <td>0.012535</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.028523</td>
          <td>22.861042</td>
          <td>16.051766</td>
          <td>23.185336</td>
          <td>23.308318</td>
          <td>20.551972</td>
          <td>19.480854</td>
          <td>0.005615</td>
          <td>22.279316</td>
          <td>0.028161</td>
          <td>25.335578</td>
          <td>0.420845</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.012086</td>
          <td>21.215741</td>
          <td>22.623299</td>
          <td>25.130073</td>
          <td>24.738685</td>
          <td>26.270691</td>
          <td>24.007682</td>
          <td>0.154444</td>
          <td>22.139953</td>
          <td>0.024917</td>
          <td>20.614331</td>
          <td>0.008291</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.729510</td>
          <td>22.524696</td>
          <td>22.947697</td>
          <td>22.571140</td>
          <td>23.093957</td>
          <td>23.900048</td>
          <td>19.461970</td>
          <td>0.005595</td>
          <td>24.633322</td>
          <td>0.221383</td>
          <td>22.629027</td>
          <td>0.041970</td>
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


