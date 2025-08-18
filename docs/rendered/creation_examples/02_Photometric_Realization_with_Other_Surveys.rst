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
          <td>22.338257</td>
          <td>23.039479</td>
          <td>19.622550</td>
          <td>24.138721</td>
          <td>25.797287</td>
          <td>21.780927</td>
          <td>23.070337</td>
          <td>23.547402</td>
          <td>22.758138</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.096918</td>
          <td>24.970589</td>
          <td>23.160498</td>
          <td>27.411095</td>
          <td>22.840202</td>
          <td>24.178590</td>
          <td>19.268666</td>
          <td>21.181258</td>
          <td>21.323262</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.290439</td>
          <td>20.668905</td>
          <td>19.335284</td>
          <td>23.389995</td>
          <td>22.777920</td>
          <td>23.438591</td>
          <td>26.783541</td>
          <td>24.615392</td>
          <td>27.051363</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.547013</td>
          <td>22.856900</td>
          <td>20.342810</td>
          <td>28.920462</td>
          <td>28.163290</td>
          <td>18.356169</td>
          <td>30.058344</td>
          <td>23.839475</td>
          <td>22.338619</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.434513</td>
          <td>19.476568</td>
          <td>23.718990</td>
          <td>20.131579</td>
          <td>22.749239</td>
          <td>25.228740</td>
          <td>24.711304</td>
          <td>20.151372</td>
          <td>20.452884</td>
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
          <td>21.147467</td>
          <td>17.802474</td>
          <td>23.230747</td>
          <td>21.233959</td>
          <td>26.951834</td>
          <td>19.756040</td>
          <td>27.742316</td>
          <td>21.556696</td>
          <td>26.284629</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.498452</td>
          <td>26.185058</td>
          <td>24.086379</td>
          <td>23.764178</td>
          <td>23.753367</td>
          <td>28.869063</td>
          <td>23.237927</td>
          <td>21.953431</td>
          <td>25.137493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.023543</td>
          <td>24.792781</td>
          <td>22.623609</td>
          <td>22.232072</td>
          <td>24.529862</td>
          <td>23.179637</td>
          <td>21.427390</td>
          <td>24.632894</td>
          <td>23.601599</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.122119</td>
          <td>24.999557</td>
          <td>20.134525</td>
          <td>26.270308</td>
          <td>23.883935</td>
          <td>22.019895</td>
          <td>23.724528</td>
          <td>26.005370</td>
          <td>22.343892</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.324382</td>
          <td>22.804368</td>
          <td>22.011104</td>
          <td>27.543658</td>
          <td>24.217176</td>
          <td>19.490478</td>
          <td>21.000792</td>
          <td>22.585370</td>
          <td>21.045665</td>
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
          <td>22.333721</td>
          <td>0.011463</td>
          <td>23.048068</td>
          <td>0.008141</td>
          <td>19.626108</td>
          <td>0.005015</td>
          <td>24.152015</td>
          <td>0.025122</td>
          <td>26.322100</td>
          <td>0.309769</td>
          <td>21.778333</td>
          <td>0.013779</td>
          <td>23.070337</td>
          <td>23.547402</td>
          <td>22.758138</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.094626</td>
          <td>0.273132</td>
          <td>24.922408</td>
          <td>0.034581</td>
          <td>23.163087</td>
          <td>0.007921</td>
          <td>29.568758</td>
          <td>1.594190</td>
          <td>22.876244</td>
          <td>0.015890</td>
          <td>24.220701</td>
          <td>0.114635</td>
          <td>19.268666</td>
          <td>21.181258</td>
          <td>21.323262</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.265257</td>
          <td>0.136205</td>
          <td>20.664486</td>
          <td>0.005091</td>
          <td>19.334352</td>
          <td>0.005010</td>
          <td>23.417628</td>
          <td>0.013629</td>
          <td>22.766479</td>
          <td>0.014547</td>
          <td>23.463135</td>
          <td>0.058818</td>
          <td>26.783541</td>
          <td>24.615392</td>
          <td>27.051363</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.549158</td>
          <td>0.072987</td>
          <td>22.869011</td>
          <td>0.007431</td>
          <td>20.352630</td>
          <td>0.005038</td>
          <td>27.403788</td>
          <td>0.407285</td>
          <td>27.039402</td>
          <td>0.536450</td>
          <td>18.344122</td>
          <td>0.005051</td>
          <td>30.058344</td>
          <td>23.839475</td>
          <td>22.338619</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.235321</td>
          <td>0.132739</td>
          <td>19.479422</td>
          <td>0.005020</td>
          <td>23.730854</td>
          <td>0.011371</td>
          <td>20.137827</td>
          <td>0.005061</td>
          <td>22.771421</td>
          <td>0.014605</td>
          <td>25.911620</td>
          <td>0.459867</td>
          <td>24.711304</td>
          <td>20.151372</td>
          <td>20.452884</td>
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
          <td>21.141955</td>
          <td>0.006316</td>
          <td>17.796963</td>
          <td>0.005003</td>
          <td>23.228155</td>
          <td>0.008208</td>
          <td>21.234632</td>
          <td>0.005337</td>
          <td>28.758564</td>
          <td>1.535155</td>
          <td>19.759255</td>
          <td>0.005453</td>
          <td>27.742316</td>
          <td>21.556696</td>
          <td>26.284629</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.428582</td>
          <td>0.065645</td>
          <td>26.272017</td>
          <td>0.113671</td>
          <td>24.099211</td>
          <td>0.015087</td>
          <td>23.757008</td>
          <td>0.017937</td>
          <td>23.746660</td>
          <td>0.033481</td>
          <td>27.188944</td>
          <td>1.082311</td>
          <td>23.237927</td>
          <td>21.953431</td>
          <td>25.137493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.030928</td>
          <td>0.111223</td>
          <td>24.804411</td>
          <td>0.031178</td>
          <td>22.626376</td>
          <td>0.006296</td>
          <td>22.227175</td>
          <td>0.006645</td>
          <td>24.561848</td>
          <td>0.068957</td>
          <td>23.170313</td>
          <td>0.045356</td>
          <td>21.427390</td>
          <td>24.632894</td>
          <td>23.601599</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.804146</td>
          <td>0.215065</td>
          <td>24.977585</td>
          <td>0.036302</td>
          <td>20.133781</td>
          <td>0.005028</td>
          <td>26.124133</td>
          <td>0.142506</td>
          <td>23.884780</td>
          <td>0.037826</td>
          <td>21.976246</td>
          <td>0.016143</td>
          <td>23.724528</td>
          <td>26.005370</td>
          <td>22.343892</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.326939</td>
          <td>0.005426</td>
          <td>22.811518</td>
          <td>0.007235</td>
          <td>22.008031</td>
          <td>0.005482</td>
          <td>27.743935</td>
          <td>0.525496</td>
          <td>24.188886</td>
          <td>0.049534</td>
          <td>19.494169</td>
          <td>0.005296</td>
          <td>21.000792</td>
          <td>22.585370</td>
          <td>21.045665</td>
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
          <td>22.338257</td>
          <td>23.039479</td>
          <td>19.622550</td>
          <td>24.138721</td>
          <td>25.797287</td>
          <td>21.780927</td>
          <td>23.071052</td>
          <td>0.007656</td>
          <td>23.551442</td>
          <td>0.016405</td>
          <td>22.773539</td>
          <td>0.009144</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.096918</td>
          <td>24.970589</td>
          <td>23.160498</td>
          <td>27.411095</td>
          <td>22.840202</td>
          <td>24.178590</td>
          <td>19.269272</td>
          <td>0.005003</td>
          <td>21.176362</td>
          <td>0.005301</td>
          <td>21.327667</td>
          <td>0.005394</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.290439</td>
          <td>20.668905</td>
          <td>19.335284</td>
          <td>23.389995</td>
          <td>22.777920</td>
          <td>23.438591</td>
          <td>26.995021</td>
          <td>0.197173</td>
          <td>24.687385</td>
          <td>0.044211</td>
          <td>27.150823</td>
          <td>0.364841</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.547013</td>
          <td>22.856900</td>
          <td>20.342810</td>
          <td>28.920462</td>
          <td>28.163290</td>
          <td>18.356169</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.845627</td>
          <td>0.021048</td>
          <td>22.345645</td>
          <td>0.007189</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.434513</td>
          <td>19.476568</td>
          <td>23.718990</td>
          <td>20.131579</td>
          <td>22.749239</td>
          <td>25.228740</td>
          <td>24.724072</td>
          <td>0.026824</td>
          <td>20.151144</td>
          <td>0.005047</td>
          <td>20.457094</td>
          <td>0.005082</td>
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
          <td>21.147467</td>
          <td>17.802474</td>
          <td>23.230747</td>
          <td>21.233959</td>
          <td>26.951834</td>
          <td>19.756040</td>
          <td>27.854254</td>
          <td>0.395374</td>
          <td>21.562872</td>
          <td>0.005596</td>
          <td>26.140963</td>
          <td>0.158912</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.498452</td>
          <td>26.185058</td>
          <td>24.086379</td>
          <td>23.764178</td>
          <td>23.753367</td>
          <td>28.869063</td>
          <td>23.245896</td>
          <td>0.008447</td>
          <td>21.945733</td>
          <td>0.006147</td>
          <td>25.178067</td>
          <td>0.068441</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.023543</td>
          <td>24.792781</td>
          <td>22.623609</td>
          <td>22.232072</td>
          <td>24.529862</td>
          <td>23.179637</td>
          <td>21.423715</td>
          <td>0.005159</td>
          <td>24.611677</td>
          <td>0.041327</td>
          <td>23.609812</td>
          <td>0.017224</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.122119</td>
          <td>24.999557</td>
          <td>20.134525</td>
          <td>26.270308</td>
          <td>23.883935</td>
          <td>22.019895</td>
          <td>23.728896</td>
          <td>0.011728</td>
          <td>26.024685</td>
          <td>0.143807</td>
          <td>22.352066</td>
          <td>0.007212</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.324382</td>
          <td>22.804368</td>
          <td>22.011104</td>
          <td>27.543658</td>
          <td>24.217176</td>
          <td>19.490478</td>
          <td>21.002214</td>
          <td>0.005074</td>
          <td>22.594545</td>
          <td>0.008196</td>
          <td>21.038526</td>
          <td>0.005235</td>
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
          <td>22.338257</td>
          <td>23.039479</td>
          <td>19.622550</td>
          <td>24.138721</td>
          <td>25.797287</td>
          <td>21.780927</td>
          <td>22.934120</td>
          <td>0.060219</td>
          <td>23.282159</td>
          <td>0.068690</td>
          <td>22.806928</td>
          <td>0.049183</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.096918</td>
          <td>24.970589</td>
          <td>23.160498</td>
          <td>27.411095</td>
          <td>22.840202</td>
          <td>24.178590</td>
          <td>19.264874</td>
          <td>0.005421</td>
          <td>21.195549</td>
          <td>0.011440</td>
          <td>21.332257</td>
          <td>0.013726</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.290439</td>
          <td>20.668905</td>
          <td>19.335284</td>
          <td>23.389995</td>
          <td>22.777920</td>
          <td>23.438591</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.711846</td>
          <td>0.236295</td>
          <td>25.074993</td>
          <td>0.343736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.547013</td>
          <td>22.856900</td>
          <td>20.342810</td>
          <td>28.920462</td>
          <td>28.163290</td>
          <td>18.356169</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.884844</td>
          <td>0.116810</td>
          <td>22.317573</td>
          <td>0.031820</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.434513</td>
          <td>19.476568</td>
          <td>23.718990</td>
          <td>20.131579</td>
          <td>22.749239</td>
          <td>25.228740</td>
          <td>24.583868</td>
          <td>0.250757</td>
          <td>20.159441</td>
          <td>0.006385</td>
          <td>20.458165</td>
          <td>0.007604</td>
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
          <td>21.147467</td>
          <td>17.802474</td>
          <td>23.230747</td>
          <td>21.233959</td>
          <td>26.951834</td>
          <td>19.756040</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.550015</td>
          <td>0.015092</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.498452</td>
          <td>26.185058</td>
          <td>24.086379</td>
          <td>23.764178</td>
          <td>23.753367</td>
          <td>28.869063</td>
          <td>23.178585</td>
          <td>0.074825</td>
          <td>21.970255</td>
          <td>0.021500</td>
          <td>24.885460</td>
          <td>0.295493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.023543</td>
          <td>24.792781</td>
          <td>22.623609</td>
          <td>22.232072</td>
          <td>24.529862</td>
          <td>23.179637</td>
          <td>21.433704</td>
          <td>0.016166</td>
          <td>24.491510</td>
          <td>0.196591</td>
          <td>23.493292</td>
          <td>0.090462</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.122119</td>
          <td>24.999557</td>
          <td>20.134525</td>
          <td>26.270308</td>
          <td>23.883935</td>
          <td>22.019895</td>
          <td>23.714808</td>
          <td>0.119899</td>
          <td>26.009739</td>
          <td>0.640231</td>
          <td>22.334614</td>
          <td>0.032304</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.324382</td>
          <td>22.804368</td>
          <td>22.011104</td>
          <td>27.543658</td>
          <td>24.217176</td>
          <td>19.490478</td>
          <td>20.999592</td>
          <td>0.011474</td>
          <td>22.631949</td>
          <td>0.038494</td>
          <td>21.049225</td>
          <td>0.011056</td>
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


