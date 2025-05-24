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
          <td>25.427499</td>
          <td>18.170753</td>
          <td>26.027706</td>
          <td>22.525884</td>
          <td>24.286477</td>
          <td>22.217567</td>
          <td>24.513669</td>
          <td>22.604437</td>
          <td>22.783533</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.686447</td>
          <td>21.718389</td>
          <td>20.720181</td>
          <td>27.492281</td>
          <td>22.381837</td>
          <td>21.293333</td>
          <td>22.210626</td>
          <td>23.963023</td>
          <td>22.922309</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.746799</td>
          <td>20.765867</td>
          <td>24.551051</td>
          <td>25.080601</td>
          <td>21.723325</td>
          <td>16.276637</td>
          <td>26.965179</td>
          <td>26.248550</td>
          <td>24.743800</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.389734</td>
          <td>27.176726</td>
          <td>23.537098</td>
          <td>20.673545</td>
          <td>23.224358</td>
          <td>17.843756</td>
          <td>25.785463</td>
          <td>22.685018</td>
          <td>27.778952</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.814024</td>
          <td>26.457802</td>
          <td>28.108527</td>
          <td>27.138130</td>
          <td>25.319035</td>
          <td>24.045351</td>
          <td>22.694471</td>
          <td>22.225955</td>
          <td>17.939159</td>
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
          <td>26.143699</td>
          <td>22.513160</td>
          <td>23.174613</td>
          <td>24.334205</td>
          <td>29.926667</td>
          <td>29.738569</td>
          <td>21.265734</td>
          <td>20.792007</td>
          <td>19.926023</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.526273</td>
          <td>22.269552</td>
          <td>24.109493</td>
          <td>26.081711</td>
          <td>23.634453</td>
          <td>23.240734</td>
          <td>21.965339</td>
          <td>23.137354</td>
          <td>22.569358</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.806919</td>
          <td>27.587054</td>
          <td>24.381513</td>
          <td>26.044659</td>
          <td>19.813632</td>
          <td>22.295858</td>
          <td>17.499404</td>
          <td>22.691392</td>
          <td>21.539168</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.622068</td>
          <td>21.935734</td>
          <td>23.178882</td>
          <td>23.761915</td>
          <td>25.090661</td>
          <td>20.440707</td>
          <td>21.221381</td>
          <td>25.117010</td>
          <td>17.628568</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.478270</td>
          <td>22.187314</td>
          <td>24.036869</td>
          <td>24.393990</td>
          <td>26.841091</td>
          <td>20.092573</td>
          <td>24.718078</td>
          <td>22.322924</td>
          <td>22.496912</td>
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
          <td>25.369675</td>
          <td>0.148975</td>
          <td>18.175484</td>
          <td>0.005005</td>
          <td>26.037479</td>
          <td>0.081498</td>
          <td>22.526588</td>
          <td>0.007584</td>
          <td>24.174002</td>
          <td>0.048884</td>
          <td>22.241921</td>
          <td>0.020142</td>
          <td>24.513669</td>
          <td>22.604437</td>
          <td>22.783533</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.686719</td>
          <td>0.034236</td>
          <td>21.716027</td>
          <td>0.005422</td>
          <td>20.717958</td>
          <td>0.005064</td>
          <td>30.198949</td>
          <td>2.108253</td>
          <td>22.374331</td>
          <td>0.010822</td>
          <td>21.281489</td>
          <td>0.009618</td>
          <td>22.210626</td>
          <td>23.963023</td>
          <td>22.922309</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.728202</td>
          <td>0.007955</td>
          <td>20.758163</td>
          <td>0.005103</td>
          <td>24.540412</td>
          <td>0.021789</td>
          <td>24.958912</td>
          <td>0.051184</td>
          <td>21.713233</td>
          <td>0.007302</td>
          <td>16.271965</td>
          <td>0.005004</td>
          <td>26.965179</td>
          <td>26.248550</td>
          <td>24.743800</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.393755</td>
          <td>0.005130</td>
          <td>27.471147</td>
          <td>0.311401</td>
          <td>23.545932</td>
          <td>0.009991</td>
          <td>20.671804</td>
          <td>0.005138</td>
          <td>23.214599</td>
          <td>0.021077</td>
          <td>17.840927</td>
          <td>0.005026</td>
          <td>25.785463</td>
          <td>22.685018</td>
          <td>27.778952</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.847815</td>
          <td>0.094823</td>
          <td>26.608664</td>
          <td>0.152047</td>
          <td>28.500054</td>
          <td>0.607699</td>
          <td>26.860742</td>
          <td>0.264735</td>
          <td>25.375163</td>
          <td>0.140551</td>
          <td>24.031311</td>
          <td>0.097145</td>
          <td>22.694471</td>
          <td>22.225955</td>
          <td>17.939159</td>
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
          <td>25.995969</td>
          <td>0.252005</td>
          <td>22.517316</td>
          <td>0.006443</td>
          <td>23.182145</td>
          <td>0.008002</td>
          <td>24.328548</td>
          <td>0.029307</td>
          <td>28.950592</td>
          <td>1.683506</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.265734</td>
          <td>20.792007</td>
          <td>19.926023</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.554363</td>
          <td>0.073321</td>
          <td>22.276775</td>
          <td>0.006000</td>
          <td>24.085450</td>
          <td>0.014922</td>
          <td>26.094808</td>
          <td>0.138950</td>
          <td>23.648288</td>
          <td>0.030704</td>
          <td>23.251755</td>
          <td>0.048755</td>
          <td>21.965339</td>
          <td>23.137354</td>
          <td>22.569358</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.829333</td>
          <td>0.016635</td>
          <td>27.228960</td>
          <td>0.255943</td>
          <td>24.384452</td>
          <td>0.019086</td>
          <td>25.925812</td>
          <td>0.120035</td>
          <td>19.815035</td>
          <td>0.005116</td>
          <td>22.281053</td>
          <td>0.020823</td>
          <td>17.499404</td>
          <td>22.691392</td>
          <td>21.539168</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.665063</td>
          <td>0.033598</td>
          <td>21.936222</td>
          <td>0.005592</td>
          <td>23.186083</td>
          <td>0.008020</td>
          <td>23.753183</td>
          <td>0.017880</td>
          <td>25.172963</td>
          <td>0.117976</td>
          <td>20.446384</td>
          <td>0.006355</td>
          <td>21.221381</td>
          <td>25.117010</td>
          <td>17.628568</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.484735</td>
          <td>0.005145</td>
          <td>22.198484</td>
          <td>0.005886</td>
          <td>24.033615</td>
          <td>0.014318</td>
          <td>24.373487</td>
          <td>0.030486</td>
          <td>27.640639</td>
          <td>0.811510</td>
          <td>20.097609</td>
          <td>0.005781</td>
          <td>24.718078</td>
          <td>22.322924</td>
          <td>22.496912</td>
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
          <td>25.427499</td>
          <td>18.170753</td>
          <td>26.027706</td>
          <td>22.525884</td>
          <td>24.286477</td>
          <td>22.217567</td>
          <td>24.522768</td>
          <td>0.022499</td>
          <td>22.616641</td>
          <td>0.008302</td>
          <td>22.784436</td>
          <td>0.009208</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.686447</td>
          <td>21.718389</td>
          <td>20.720181</td>
          <td>27.492281</td>
          <td>22.381837</td>
          <td>21.293333</td>
          <td>22.213526</td>
          <td>0.005651</td>
          <td>23.994306</td>
          <td>0.023944</td>
          <td>22.892390</td>
          <td>0.009895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.746799</td>
          <td>20.765867</td>
          <td>24.551051</td>
          <td>25.080601</td>
          <td>21.723325</td>
          <td>16.276637</td>
          <td>27.392418</td>
          <td>0.274042</td>
          <td>26.468295</td>
          <td>0.209683</td>
          <td>24.767081</td>
          <td>0.047467</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.389734</td>
          <td>27.176726</td>
          <td>23.537098</td>
          <td>20.673545</td>
          <td>23.224358</td>
          <td>17.843756</td>
          <td>25.778177</td>
          <td>0.068448</td>
          <td>22.678955</td>
          <td>0.008617</td>
          <td>27.014096</td>
          <td>0.327548</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.814024</td>
          <td>26.457802</td>
          <td>28.108527</td>
          <td>27.138130</td>
          <td>25.319035</td>
          <td>24.045351</td>
          <td>22.706610</td>
          <td>0.006496</td>
          <td>22.228364</td>
          <td>0.006820</td>
          <td>17.940760</td>
          <td>0.005001</td>
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
          <td>26.143699</td>
          <td>22.513160</td>
          <td>23.174613</td>
          <td>24.334205</td>
          <td>29.926667</td>
          <td>29.738569</td>
          <td>21.256211</td>
          <td>0.005118</td>
          <td>20.791476</td>
          <td>0.005150</td>
          <td>19.926758</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.526273</td>
          <td>22.269552</td>
          <td>24.109493</td>
          <td>26.081711</td>
          <td>23.634453</td>
          <td>23.240734</td>
          <td>21.963893</td>
          <td>0.005420</td>
          <td>23.140679</td>
          <td>0.011832</td>
          <td>22.582262</td>
          <td>0.008139</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.806919</td>
          <td>27.587054</td>
          <td>24.381513</td>
          <td>26.044659</td>
          <td>19.813632</td>
          <td>22.295858</td>
          <td>17.494151</td>
          <td>0.005000</td>
          <td>22.692241</td>
          <td>0.008688</td>
          <td>21.541338</td>
          <td>0.005574</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.622068</td>
          <td>21.935734</td>
          <td>23.178882</td>
          <td>23.761915</td>
          <td>25.090661</td>
          <td>20.440707</td>
          <td>21.218441</td>
          <td>0.005110</td>
          <td>25.057287</td>
          <td>0.061473</td>
          <td>17.629034</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.478270</td>
          <td>22.187314</td>
          <td>24.036869</td>
          <td>24.393990</td>
          <td>26.841091</td>
          <td>20.092573</td>
          <td>24.721579</td>
          <td>0.026766</td>
          <td>22.311659</td>
          <td>0.007076</td>
          <td>22.485661</td>
          <td>0.007715</td>
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
          <td>25.427499</td>
          <td>18.170753</td>
          <td>26.027706</td>
          <td>22.525884</td>
          <td>24.286477</td>
          <td>22.217567</td>
          <td>24.401396</td>
          <td>0.215567</td>
          <td>22.610217</td>
          <td>0.037757</td>
          <td>22.868150</td>
          <td>0.051942</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.686447</td>
          <td>21.718389</td>
          <td>20.720181</td>
          <td>27.492281</td>
          <td>22.381837</td>
          <td>21.293333</td>
          <td>22.173292</td>
          <td>0.030597</td>
          <td>23.819804</td>
          <td>0.110362</td>
          <td>23.075461</td>
          <td>0.062475</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.746799</td>
          <td>20.765867</td>
          <td>24.551051</td>
          <td>25.080601</td>
          <td>21.723325</td>
          <td>16.276637</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.996001</td>
          <td>0.298014</td>
          <td>24.488626</td>
          <td>0.213279</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.389734</td>
          <td>27.176726</td>
          <td>23.537098</td>
          <td>20.673545</td>
          <td>23.224358</td>
          <td>17.843756</td>
          <td>24.915349</td>
          <td>0.327874</td>
          <td>22.667605</td>
          <td>0.039736</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.814024</td>
          <td>26.457802</td>
          <td>28.108527</td>
          <td>27.138130</td>
          <td>25.319035</td>
          <td>24.045351</td>
          <td>22.685659</td>
          <td>0.048259</td>
          <td>22.214344</td>
          <td>0.026596</td>
          <td>17.935809</td>
          <td>0.005031</td>
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
          <td>26.143699</td>
          <td>22.513160</td>
          <td>23.174613</td>
          <td>24.334205</td>
          <td>29.926667</td>
          <td>29.738569</td>
          <td>21.256671</td>
          <td>0.013996</td>
          <td>20.785124</td>
          <td>0.008650</td>
          <td>19.936694</td>
          <td>0.006130</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.526273</td>
          <td>22.269552</td>
          <td>24.109493</td>
          <td>26.081711</td>
          <td>23.634453</td>
          <td>23.240734</td>
          <td>21.985649</td>
          <td>0.025935</td>
          <td>23.164780</td>
          <td>0.061884</td>
          <td>22.482669</td>
          <td>0.036843</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.806919</td>
          <td>27.587054</td>
          <td>24.381513</td>
          <td>26.044659</td>
          <td>19.813632</td>
          <td>22.295858</td>
          <td>17.497483</td>
          <td>0.005017</td>
          <td>22.673709</td>
          <td>0.039952</td>
          <td>21.551973</td>
          <td>0.016413</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.622068</td>
          <td>21.935734</td>
          <td>23.178882</td>
          <td>23.761915</td>
          <td>25.090661</td>
          <td>20.440707</td>
          <td>21.237976</td>
          <td>0.013789</td>
          <td>25.547295</td>
          <td>0.458007</td>
          <td>17.624493</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.478270</td>
          <td>22.187314</td>
          <td>24.036869</td>
          <td>24.393990</td>
          <td>26.841091</td>
          <td>20.092573</td>
          <td>24.565246</td>
          <td>0.246944</td>
          <td>22.320215</td>
          <td>0.029195</td>
          <td>22.447239</td>
          <td>0.035700</td>
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


