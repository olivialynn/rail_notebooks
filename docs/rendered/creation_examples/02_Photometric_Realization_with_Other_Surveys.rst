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
          <td>21.290040</td>
          <td>23.677341</td>
          <td>23.967023</td>
          <td>27.680204</td>
          <td>23.938183</td>
          <td>27.014557</td>
          <td>23.070746</td>
          <td>20.856026</td>
          <td>24.299959</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.236994</td>
          <td>23.744869</td>
          <td>24.946315</td>
          <td>27.861586</td>
          <td>22.535859</td>
          <td>22.572523</td>
          <td>20.554211</td>
          <td>22.117133</td>
          <td>21.199942</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.100546</td>
          <td>23.652807</td>
          <td>19.450196</td>
          <td>17.303240</td>
          <td>24.947324</td>
          <td>18.603740</td>
          <td>25.642760</td>
          <td>24.282009</td>
          <td>24.740207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.927890</td>
          <td>27.649349</td>
          <td>19.189790</td>
          <td>24.674458</td>
          <td>17.732564</td>
          <td>19.229785</td>
          <td>25.106977</td>
          <td>21.485836</td>
          <td>24.936305</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.793277</td>
          <td>22.508119</td>
          <td>16.019917</td>
          <td>24.561775</td>
          <td>23.468834</td>
          <td>25.245645</td>
          <td>17.968740</td>
          <td>25.830591</td>
          <td>21.862312</td>
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
          <td>27.037143</td>
          <td>15.583640</td>
          <td>24.779372</td>
          <td>24.653258</td>
          <td>25.850140</td>
          <td>23.664216</td>
          <td>21.186496</td>
          <td>22.287142</td>
          <td>21.604349</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.119075</td>
          <td>23.055816</td>
          <td>23.997800</td>
          <td>22.451906</td>
          <td>21.965051</td>
          <td>26.289947</td>
          <td>19.010933</td>
          <td>19.269332</td>
          <td>17.075385</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.499418</td>
          <td>24.729690</td>
          <td>22.640023</td>
          <td>23.741946</td>
          <td>21.832841</td>
          <td>22.843288</td>
          <td>22.358261</td>
          <td>23.961268</td>
          <td>24.297149</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.730508</td>
          <td>21.607891</td>
          <td>23.714864</td>
          <td>26.653756</td>
          <td>20.667129</td>
          <td>21.572472</td>
          <td>28.954445</td>
          <td>25.783732</td>
          <td>16.724587</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.901001</td>
          <td>22.408525</td>
          <td>22.845746</td>
          <td>19.516255</td>
          <td>21.213827</td>
          <td>29.074503</td>
          <td>21.970337</td>
          <td>24.548896</td>
          <td>26.625352</td>
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
          <td>21.288818</td>
          <td>0.006616</td>
          <td>23.674418</td>
          <td>0.012239</td>
          <td>23.985426</td>
          <td>0.013784</td>
          <td>28.369870</td>
          <td>0.809604</td>
          <td>23.909873</td>
          <td>0.038675</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.070746</td>
          <td>20.856026</td>
          <td>24.299959</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.296101</td>
          <td>0.058422</td>
          <td>23.733599</td>
          <td>0.012793</td>
          <td>24.931650</td>
          <td>0.030612</td>
          <td>27.918463</td>
          <td>0.595792</td>
          <td>22.504935</td>
          <td>0.011895</td>
          <td>22.571633</td>
          <td>0.026754</td>
          <td>20.554211</td>
          <td>22.117133</td>
          <td>21.199942</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.112186</td>
          <td>0.005093</td>
          <td>23.657181</td>
          <td>0.012084</td>
          <td>19.456238</td>
          <td>0.005012</td>
          <td>17.308615</td>
          <td>0.005002</td>
          <td>25.086378</td>
          <td>0.109403</td>
          <td>18.604710</td>
          <td>0.005075</td>
          <td>25.642760</td>
          <td>24.282009</td>
          <td>24.740207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.929204</td>
          <td>0.005977</td>
          <td>28.155726</td>
          <td>0.526265</td>
          <td>19.191457</td>
          <td>0.005009</td>
          <td>24.645026</td>
          <td>0.038742</td>
          <td>17.734615</td>
          <td>0.005007</td>
          <td>19.239151</td>
          <td>0.005198</td>
          <td>25.106977</td>
          <td>21.485836</td>
          <td>24.936305</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.790144</td>
          <td>0.005805</td>
          <td>22.511766</td>
          <td>0.006431</td>
          <td>16.014493</td>
          <td>0.005000</td>
          <td>24.580057</td>
          <td>0.036577</td>
          <td>23.447845</td>
          <td>0.025769</td>
          <td>25.441092</td>
          <td>0.319379</td>
          <td>17.968740</td>
          <td>25.830591</td>
          <td>21.862312</td>
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
          <td>26.514242</td>
          <td>0.381218</td>
          <td>15.581069</td>
          <td>0.005000</td>
          <td>24.810354</td>
          <td>0.027526</td>
          <td>24.684592</td>
          <td>0.040124</td>
          <td>25.823147</td>
          <td>0.205731</td>
          <td>23.703104</td>
          <td>0.072751</td>
          <td>21.186496</td>
          <td>22.287142</td>
          <td>21.604349</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.122683</td>
          <td>0.021139</td>
          <td>23.051623</td>
          <td>0.008157</td>
          <td>23.981772</td>
          <td>0.013745</td>
          <td>22.446927</td>
          <td>0.007296</td>
          <td>21.968875</td>
          <td>0.008342</td>
          <td>26.954250</td>
          <td>0.940256</td>
          <td>19.010933</td>
          <td>19.269332</td>
          <td>17.075385</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.149179</td>
          <td>0.610301</td>
          <td>24.715829</td>
          <td>0.028856</td>
          <td>22.634104</td>
          <td>0.006311</td>
          <td>23.726776</td>
          <td>0.017492</td>
          <td>21.814503</td>
          <td>0.007674</td>
          <td>22.860266</td>
          <td>0.034467</td>
          <td>22.358261</td>
          <td>23.961268</td>
          <td>24.297149</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.732766</td>
          <td>0.015414</td>
          <td>21.614911</td>
          <td>0.005362</td>
          <td>23.701412</td>
          <td>0.011132</td>
          <td>26.793676</td>
          <td>0.250584</td>
          <td>20.671907</td>
          <td>0.005448</td>
          <td>21.588529</td>
          <td>0.011922</td>
          <td>28.954445</td>
          <td>25.783732</td>
          <td>16.724587</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.890703</td>
          <td>0.017474</td>
          <td>22.403146</td>
          <td>0.006213</td>
          <td>22.844507</td>
          <td>0.006816</td>
          <td>19.516714</td>
          <td>0.005026</td>
          <td>21.216982</td>
          <td>0.006072</td>
          <td>27.371537</td>
          <td>1.201025</td>
          <td>21.970337</td>
          <td>24.548896</td>
          <td>26.625352</td>
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
          <td>21.290040</td>
          <td>23.677341</td>
          <td>23.967023</td>
          <td>27.680204</td>
          <td>23.938183</td>
          <td>27.014557</td>
          <td>23.070075</td>
          <td>0.007652</td>
          <td>20.853589</td>
          <td>0.005168</td>
          <td>24.269449</td>
          <td>0.030493</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.236994</td>
          <td>23.744869</td>
          <td>24.946315</td>
          <td>27.861586</td>
          <td>22.535859</td>
          <td>22.572523</td>
          <td>20.555088</td>
          <td>0.005033</td>
          <td>22.117424</td>
          <td>0.006522</td>
          <td>21.208006</td>
          <td>0.005319</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.100546</td>
          <td>23.652807</td>
          <td>19.450196</td>
          <td>17.303240</td>
          <td>24.947324</td>
          <td>18.603740</td>
          <td>25.511425</td>
          <td>0.053985</td>
          <td>24.277137</td>
          <td>0.030701</td>
          <td>24.737228</td>
          <td>0.046220</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.927890</td>
          <td>27.649349</td>
          <td>19.189790</td>
          <td>24.674458</td>
          <td>17.732564</td>
          <td>19.229785</td>
          <td>25.144945</td>
          <td>0.038942</td>
          <td>21.470020</td>
          <td>0.005507</td>
          <td>24.936822</td>
          <td>0.055220</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.793277</td>
          <td>22.508119</td>
          <td>16.019917</td>
          <td>24.561775</td>
          <td>23.468834</td>
          <td>25.245645</td>
          <td>17.971576</td>
          <td>0.005000</td>
          <td>26.029740</td>
          <td>0.144435</td>
          <td>21.866139</td>
          <td>0.006004</td>
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
          <td>27.037143</td>
          <td>15.583640</td>
          <td>24.779372</td>
          <td>24.653258</td>
          <td>25.850140</td>
          <td>23.664216</td>
          <td>21.191275</td>
          <td>0.005104</td>
          <td>22.281776</td>
          <td>0.006981</td>
          <td>21.601589</td>
          <td>0.005638</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.119075</td>
          <td>23.055816</td>
          <td>23.997800</td>
          <td>22.451906</td>
          <td>21.965051</td>
          <td>26.289947</td>
          <td>19.006822</td>
          <td>0.005002</td>
          <td>19.271536</td>
          <td>0.005009</td>
          <td>17.073321</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.499418</td>
          <td>24.729690</td>
          <td>22.640023</td>
          <td>23.741946</td>
          <td>21.832841</td>
          <td>22.843288</td>
          <td>22.360753</td>
          <td>0.005839</td>
          <td>23.906161</td>
          <td>0.022178</td>
          <td>24.260935</td>
          <td>0.030264</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.730508</td>
          <td>21.607891</td>
          <td>23.714864</td>
          <td>26.653756</td>
          <td>20.667129</td>
          <td>21.572472</td>
          <td>28.864481</td>
          <td>0.812684</td>
          <td>25.766021</td>
          <td>0.114908</td>
          <td>16.730727</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.901001</td>
          <td>22.408525</td>
          <td>22.845746</td>
          <td>19.516255</td>
          <td>21.213827</td>
          <td>29.074503</td>
          <td>21.966218</td>
          <td>0.005422</td>
          <td>24.501140</td>
          <td>0.037453</td>
          <td>26.419575</td>
          <td>0.201287</td>
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
          <td>21.290040</td>
          <td>23.677341</td>
          <td>23.967023</td>
          <td>27.680204</td>
          <td>23.938183</td>
          <td>27.014557</td>
          <td>23.064369</td>
          <td>0.067613</td>
          <td>20.859462</td>
          <td>0.009062</td>
          <td>24.444094</td>
          <td>0.205473</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.236994</td>
          <td>23.744869</td>
          <td>24.946315</td>
          <td>27.861586</td>
          <td>22.535859</td>
          <td>22.572523</td>
          <td>20.558147</td>
          <td>0.008509</td>
          <td>22.090729</td>
          <td>0.023869</td>
          <td>21.210382</td>
          <td>0.012478</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.100546</td>
          <td>23.652807</td>
          <td>19.450196</td>
          <td>17.303240</td>
          <td>24.947324</td>
          <td>18.603740</td>
          <td>25.416487</td>
          <td>0.482319</td>
          <td>24.523620</td>
          <td>0.201973</td>
          <td>24.565848</td>
          <td>0.227454</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.927890</td>
          <td>27.649349</td>
          <td>19.189790</td>
          <td>24.674458</td>
          <td>17.732564</td>
          <td>19.229785</td>
          <td>24.637400</td>
          <td>0.262012</td>
          <td>21.472286</td>
          <td>0.014172</td>
          <td>24.990072</td>
          <td>0.321344</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.793277</td>
          <td>22.508119</td>
          <td>16.019917</td>
          <td>24.561775</td>
          <td>23.468834</td>
          <td>25.245645</td>
          <td>17.970193</td>
          <td>0.005040</td>
          <td>25.623478</td>
          <td>0.484832</td>
          <td>21.831826</td>
          <td>0.020799</td>
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
          <td>27.037143</td>
          <td>15.583640</td>
          <td>24.779372</td>
          <td>24.653258</td>
          <td>25.850140</td>
          <td>23.664216</td>
          <td>21.174346</td>
          <td>0.013113</td>
          <td>22.218163</td>
          <td>0.026685</td>
          <td>21.608673</td>
          <td>0.017207</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.119075</td>
          <td>23.055816</td>
          <td>23.997800</td>
          <td>22.451906</td>
          <td>21.965051</td>
          <td>26.289947</td>
          <td>19.014004</td>
          <td>0.005269</td>
          <td>19.268386</td>
          <td>0.005297</td>
          <td>17.066680</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.499418</td>
          <td>24.729690</td>
          <td>22.640023</td>
          <td>23.741946</td>
          <td>21.832841</td>
          <td>22.843288</td>
          <td>22.353849</td>
          <td>0.035911</td>
          <td>24.208120</td>
          <td>0.154502</td>
          <td>24.310987</td>
          <td>0.183668</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.730508</td>
          <td>21.607891</td>
          <td>23.714864</td>
          <td>26.653756</td>
          <td>20.667129</td>
          <td>21.572472</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.134489</td>
          <td>0.332895</td>
          <td>16.727310</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.901001</td>
          <td>22.408525</td>
          <td>22.845746</td>
          <td>19.516255</td>
          <td>21.213827</td>
          <td>29.074503</td>
          <td>21.959150</td>
          <td>0.025339</td>
          <td>24.247298</td>
          <td>0.159776</td>
          <td>26.821300</td>
          <td>1.140978</td>
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


