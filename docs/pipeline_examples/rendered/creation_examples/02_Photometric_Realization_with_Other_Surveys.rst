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
          <td>22.896168</td>
          <td>21.605764</td>
          <td>21.332876</td>
          <td>27.666596</td>
          <td>21.673207</td>
          <td>24.570380</td>
          <td>23.346782</td>
          <td>25.280115</td>
          <td>26.016149</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725573</td>
          <td>25.699767</td>
          <td>21.281968</td>
          <td>21.679808</td>
          <td>24.765238</td>
          <td>22.515272</td>
          <td>19.612994</td>
          <td>22.031989</td>
          <td>20.881490</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.344439</td>
          <td>22.992132</td>
          <td>27.278673</td>
          <td>20.740282</td>
          <td>24.647291</td>
          <td>22.023698</td>
          <td>23.228228</td>
          <td>20.927605</td>
          <td>25.840284</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.143196</td>
          <td>25.448503</td>
          <td>21.812759</td>
          <td>20.505342</td>
          <td>25.216961</td>
          <td>24.827834</td>
          <td>22.853308</td>
          <td>21.545186</td>
          <td>19.734181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.410040</td>
          <td>19.841781</td>
          <td>29.763576</td>
          <td>18.683438</td>
          <td>23.746985</td>
          <td>18.102131</td>
          <td>23.633620</td>
          <td>25.088118</td>
          <td>26.634364</td>
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
          <td>22.781675</td>
          <td>18.263275</td>
          <td>22.367964</td>
          <td>20.544849</td>
          <td>21.201182</td>
          <td>18.375910</td>
          <td>26.459694</td>
          <td>20.816369</td>
          <td>18.231530</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.132770</td>
          <td>21.477191</td>
          <td>24.763975</td>
          <td>19.154737</td>
          <td>27.432169</td>
          <td>21.152955</td>
          <td>20.850076</td>
          <td>21.603113</td>
          <td>22.929295</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.599822</td>
          <td>17.207475</td>
          <td>26.563086</td>
          <td>19.576532</td>
          <td>25.619653</td>
          <td>21.672765</td>
          <td>28.833697</td>
          <td>24.302363</td>
          <td>20.766440</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.264155</td>
          <td>22.526826</td>
          <td>20.495583</td>
          <td>25.212171</td>
          <td>22.114507</td>
          <td>33.632924</td>
          <td>22.849863</td>
          <td>27.553229</td>
          <td>15.847177</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.341897</td>
          <td>19.952091</td>
          <td>19.920929</td>
          <td>23.262230</td>
          <td>26.844252</td>
          <td>19.582621</td>
          <td>24.630353</td>
          <td>18.347768</td>
          <td>16.083454</td>
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
          <td>22.868884</td>
          <td>0.017170</td>
          <td>21.612544</td>
          <td>0.005360</td>
          <td>21.336272</td>
          <td>0.005165</td>
          <td>27.580489</td>
          <td>0.465679</td>
          <td>21.689570</td>
          <td>0.007222</td>
          <td>24.621018</td>
          <td>0.161939</td>
          <td>23.346782</td>
          <td>25.280115</td>
          <td>26.016149</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.718033</td>
          <td>0.015238</td>
          <td>25.785333</td>
          <td>0.074168</td>
          <td>21.280687</td>
          <td>0.005151</td>
          <td>21.686316</td>
          <td>0.005699</td>
          <td>24.846665</td>
          <td>0.088670</td>
          <td>22.556803</td>
          <td>0.026410</td>
          <td>19.612994</td>
          <td>22.031989</td>
          <td>20.881490</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.134358</td>
          <td>1.144028</td>
          <td>22.991407</td>
          <td>0.007898</td>
          <td>27.381649</td>
          <td>0.257752</td>
          <td>20.736686</td>
          <td>0.005152</td>
          <td>24.587744</td>
          <td>0.070556</td>
          <td>22.013825</td>
          <td>0.016648</td>
          <td>23.228228</td>
          <td>20.927605</td>
          <td>25.840284</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.145062</td>
          <td>0.005097</td>
          <td>25.494109</td>
          <td>0.057327</td>
          <td>21.820763</td>
          <td>0.005356</td>
          <td>20.496456</td>
          <td>0.005105</td>
          <td>25.133702</td>
          <td>0.114012</td>
          <td>24.960291</td>
          <td>0.215664</td>
          <td>22.853308</td>
          <td>21.545186</td>
          <td>19.734181</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.409824</td>
          <td>0.012096</td>
          <td>19.841795</td>
          <td>0.005031</td>
          <td>29.109424</td>
          <td>0.911275</td>
          <td>18.682082</td>
          <td>0.005009</td>
          <td>23.763999</td>
          <td>0.033997</td>
          <td>18.099479</td>
          <td>0.005037</td>
          <td>23.633620</td>
          <td>25.088118</td>
          <td>26.634364</td>
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
          <td>22.784296</td>
          <td>0.016051</td>
          <td>18.263901</td>
          <td>0.005006</td>
          <td>22.368834</td>
          <td>0.005861</td>
          <td>20.542655</td>
          <td>0.005113</td>
          <td>21.206607</td>
          <td>0.006055</td>
          <td>18.374709</td>
          <td>0.005054</td>
          <td>26.459694</td>
          <td>20.816369</td>
          <td>18.231530</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.115116</td>
          <td>0.021006</td>
          <td>21.474410</td>
          <td>0.005292</td>
          <td>24.713282</td>
          <td>0.025294</td>
          <td>19.157084</td>
          <td>0.005016</td>
          <td>27.605159</td>
          <td>0.792981</td>
          <td>21.157937</td>
          <td>0.008895</td>
          <td>20.850076</td>
          <td>21.603113</td>
          <td>22.929295</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.595818</td>
          <td>0.013872</td>
          <td>17.202024</td>
          <td>0.005002</td>
          <td>26.443395</td>
          <td>0.116335</td>
          <td>19.573013</td>
          <td>0.005028</td>
          <td>25.348682</td>
          <td>0.137378</td>
          <td>21.680725</td>
          <td>0.012778</td>
          <td>28.833697</td>
          <td>24.302363</td>
          <td>20.766440</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.234824</td>
          <td>0.647926</td>
          <td>22.517156</td>
          <td>0.006442</td>
          <td>20.504251</td>
          <td>0.005047</td>
          <td>25.262744</td>
          <td>0.067022</td>
          <td>22.122906</td>
          <td>0.009149</td>
          <td>27.487964</td>
          <td>1.280292</td>
          <td>22.849863</td>
          <td>27.553229</td>
          <td>15.847177</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.339878</td>
          <td>0.005122</td>
          <td>19.944948</td>
          <td>0.005035</td>
          <td>19.916125</td>
          <td>0.005021</td>
          <td>23.260856</td>
          <td>0.012090</td>
          <td>29.363093</td>
          <td>2.020636</td>
          <td>19.576960</td>
          <td>0.005338</td>
          <td>24.630353</td>
          <td>18.347768</td>
          <td>16.083454</td>
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
          <td>22.896168</td>
          <td>21.605764</td>
          <td>21.332876</td>
          <td>27.666596</td>
          <td>21.673207</td>
          <td>24.570380</td>
          <td>23.348642</td>
          <td>0.008999</td>
          <td>25.250056</td>
          <td>0.072956</td>
          <td>26.017950</td>
          <td>0.142975</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725573</td>
          <td>25.699767</td>
          <td>21.281968</td>
          <td>21.679808</td>
          <td>24.765238</td>
          <td>22.515272</td>
          <td>19.614027</td>
          <td>0.005006</td>
          <td>22.039957</td>
          <td>0.006341</td>
          <td>20.890048</td>
          <td>0.005180</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.344439</td>
          <td>22.992132</td>
          <td>27.278673</td>
          <td>20.740282</td>
          <td>24.647291</td>
          <td>22.023698</td>
          <td>23.246476</td>
          <td>0.008450</td>
          <td>20.927821</td>
          <td>0.005192</td>
          <td>25.936402</td>
          <td>0.133249</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.143196</td>
          <td>25.448503</td>
          <td>21.812759</td>
          <td>20.505342</td>
          <td>25.216961</td>
          <td>24.827834</td>
          <td>22.860487</td>
          <td>0.006915</td>
          <td>21.545986</td>
          <td>0.005579</td>
          <td>19.736665</td>
          <td>0.005022</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.410040</td>
          <td>19.841781</td>
          <td>29.763576</td>
          <td>18.683438</td>
          <td>23.746985</td>
          <td>18.102131</td>
          <td>23.625880</td>
          <td>0.010870</td>
          <td>25.060695</td>
          <td>0.061660</td>
          <td>26.391497</td>
          <td>0.196589</td>
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
          <td>22.781675</td>
          <td>18.263275</td>
          <td>22.367964</td>
          <td>20.544849</td>
          <td>21.201182</td>
          <td>18.375910</td>
          <td>26.414825</td>
          <td>0.119901</td>
          <td>20.808077</td>
          <td>0.005155</td>
          <td>18.236177</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.132770</td>
          <td>21.477191</td>
          <td>24.763975</td>
          <td>19.154737</td>
          <td>27.432169</td>
          <td>21.152955</td>
          <td>20.840924</td>
          <td>0.005055</td>
          <td>21.600650</td>
          <td>0.005637</td>
          <td>22.916027</td>
          <td>0.010057</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.599822</td>
          <td>17.207475</td>
          <td>26.563086</td>
          <td>19.576532</td>
          <td>25.619653</td>
          <td>21.672765</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.278369</td>
          <td>0.030735</td>
          <td>20.765140</td>
          <td>0.005143</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.264155</td>
          <td>22.526826</td>
          <td>20.495583</td>
          <td>25.212171</td>
          <td>22.114507</td>
          <td>33.632924</td>
          <td>22.843269</td>
          <td>0.006863</td>
          <td>27.703818</td>
          <td>0.553282</td>
          <td>15.849361</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.341897</td>
          <td>19.952091</td>
          <td>19.920929</td>
          <td>23.262230</td>
          <td>26.844252</td>
          <td>19.582621</td>
          <td>24.667619</td>
          <td>0.025528</td>
          <td>18.346271</td>
          <td>0.005002</td>
          <td>16.078846</td>
          <td>0.005000</td>
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
          <td>22.896168</td>
          <td>21.605764</td>
          <td>21.332876</td>
          <td>27.666596</td>
          <td>21.673207</td>
          <td>24.570380</td>
          <td>23.381395</td>
          <td>0.089518</td>
          <td>25.777045</td>
          <td>0.542672</td>
          <td>24.941242</td>
          <td>0.309044</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.725573</td>
          <td>25.699767</td>
          <td>21.281968</td>
          <td>21.679808</td>
          <td>24.765238</td>
          <td>22.515272</td>
          <td>19.612611</td>
          <td>0.005773</td>
          <td>22.020408</td>
          <td>0.022453</td>
          <td>20.877989</td>
          <td>0.009798</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.344439</td>
          <td>22.992132</td>
          <td>27.278673</td>
          <td>20.740282</td>
          <td>24.647291</td>
          <td>22.023698</td>
          <td>23.122363</td>
          <td>0.071186</td>
          <td>20.913679</td>
          <td>0.009386</td>
          <td>26.137330</td>
          <td>0.747545</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.143196</td>
          <td>25.448503</td>
          <td>21.812759</td>
          <td>20.505342</td>
          <td>25.216961</td>
          <td>24.827834</td>
          <td>22.881909</td>
          <td>0.057483</td>
          <td>21.558737</td>
          <td>0.015199</td>
          <td>19.743622</td>
          <td>0.005815</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.410040</td>
          <td>19.841781</td>
          <td>29.763576</td>
          <td>18.683438</td>
          <td>23.746985</td>
          <td>18.102131</td>
          <td>23.708621</td>
          <td>0.119255</td>
          <td>24.963016</td>
          <td>0.290189</td>
          <td>26.243840</td>
          <td>0.801857</td>
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
          <td>22.781675</td>
          <td>18.263275</td>
          <td>22.367964</td>
          <td>20.544849</td>
          <td>21.201182</td>
          <td>18.375910</td>
          <td>27.783444</td>
          <td>1.935882</td>
          <td>20.810627</td>
          <td>0.008787</td>
          <td>18.234766</td>
          <td>0.005054</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.132770</td>
          <td>21.477191</td>
          <td>24.763975</td>
          <td>19.154737</td>
          <td>27.432169</td>
          <td>21.152955</td>
          <td>20.858198</td>
          <td>0.010357</td>
          <td>21.628744</td>
          <td>0.016100</td>
          <td>22.878823</td>
          <td>0.052439</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.599822</td>
          <td>17.207475</td>
          <td>26.563086</td>
          <td>19.576532</td>
          <td>25.619653</td>
          <td>21.672765</td>
          <td>26.832912</td>
          <td>1.214857</td>
          <td>24.620309</td>
          <td>0.218996</td>
          <td>20.771812</td>
          <td>0.009134</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.264155</td>
          <td>22.526826</td>
          <td>20.495583</td>
          <td>25.212171</td>
          <td>22.114507</td>
          <td>33.632924</td>
          <td>22.918200</td>
          <td>0.059371</td>
          <td>25.464132</td>
          <td>0.430101</td>
          <td>15.846835</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.341897</td>
          <td>19.952091</td>
          <td>19.920929</td>
          <td>23.262230</td>
          <td>26.844252</td>
          <td>19.582621</td>
          <td>24.417583</td>
          <td>0.218498</td>
          <td>18.337271</td>
          <td>0.005055</td>
          <td>16.083251</td>
          <td>0.005001</td>
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


