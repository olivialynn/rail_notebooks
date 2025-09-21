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
          <td>27.117194</td>
          <td>23.596183</td>
          <td>26.926223</td>
          <td>28.148389</td>
          <td>19.928134</td>
          <td>20.411181</td>
          <td>25.699791</td>
          <td>25.238514</td>
          <td>17.864137</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.250580</td>
          <td>23.584614</td>
          <td>22.764905</td>
          <td>20.985088</td>
          <td>30.411366</td>
          <td>20.568428</td>
          <td>23.352964</td>
          <td>24.951398</td>
          <td>22.181192</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.072970</td>
          <td>25.499564</td>
          <td>23.253525</td>
          <td>21.105217</td>
          <td>22.243325</td>
          <td>21.188192</td>
          <td>23.060089</td>
          <td>27.974724</td>
          <td>23.458467</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.653015</td>
          <td>23.807871</td>
          <td>22.302071</td>
          <td>21.489838</td>
          <td>28.332367</td>
          <td>23.159480</td>
          <td>19.776648</td>
          <td>21.801345</td>
          <td>24.711250</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.412536</td>
          <td>24.628151</td>
          <td>19.791507</td>
          <td>26.885229</td>
          <td>23.825927</td>
          <td>16.136082</td>
          <td>26.630147</td>
          <td>20.816874</td>
          <td>23.204242</td>
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
          <td>20.128553</td>
          <td>22.627072</td>
          <td>25.471321</td>
          <td>23.589373</td>
          <td>20.483112</td>
          <td>19.878566</td>
          <td>19.049558</td>
          <td>23.078195</td>
          <td>19.068514</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.692132</td>
          <td>27.790354</td>
          <td>26.802240</td>
          <td>20.413273</td>
          <td>17.912021</td>
          <td>25.785145</td>
          <td>23.950197</td>
          <td>25.207112</td>
          <td>22.076943</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.239365</td>
          <td>21.242666</td>
          <td>25.302148</td>
          <td>21.380275</td>
          <td>22.028213</td>
          <td>25.460178</td>
          <td>27.903417</td>
          <td>23.278518</td>
          <td>21.502251</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.921551</td>
          <td>26.895248</td>
          <td>25.882931</td>
          <td>23.609322</td>
          <td>24.187779</td>
          <td>24.195230</td>
          <td>26.051143</td>
          <td>19.882628</td>
          <td>27.101913</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.816556</td>
          <td>22.051822</td>
          <td>21.590319</td>
          <td>25.707265</td>
          <td>22.905520</td>
          <td>21.465334</td>
          <td>20.779782</td>
          <td>23.822303</td>
          <td>21.943225</td>
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
          <td>28.162857</td>
          <td>1.162669</td>
          <td>23.608226</td>
          <td>0.011658</td>
          <td>26.708657</td>
          <td>0.146351</td>
          <td>33.095419</td>
          <td>4.849122</td>
          <td>19.930574</td>
          <td>0.005138</td>
          <td>20.414312</td>
          <td>0.006289</td>
          <td>25.699791</td>
          <td>25.238514</td>
          <td>17.864137</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.372791</td>
          <td>0.341288</td>
          <td>23.585723</td>
          <td>0.011470</td>
          <td>22.760301</td>
          <td>0.006596</td>
          <td>20.982334</td>
          <td>0.005225</td>
          <td>27.987698</td>
          <td>1.007867</td>
          <td>20.563944</td>
          <td>0.006626</td>
          <td>23.352964</td>
          <td>24.951398</td>
          <td>22.181192</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.056681</td>
          <td>0.020012</td>
          <td>25.487974</td>
          <td>0.057017</td>
          <td>23.268895</td>
          <td>0.008400</td>
          <td>21.103837</td>
          <td>0.005273</td>
          <td>22.251216</td>
          <td>0.009943</td>
          <td>21.196966</td>
          <td>0.009112</td>
          <td>23.060089</td>
          <td>27.974724</td>
          <td>23.458467</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.743555</td>
          <td>0.086559</td>
          <td>23.811313</td>
          <td>0.013576</td>
          <td>22.298482</td>
          <td>0.005770</td>
          <td>21.488396</td>
          <td>0.005508</td>
          <td>27.951251</td>
          <td>0.985975</td>
          <td>23.182398</td>
          <td>0.045845</td>
          <td>19.776648</td>
          <td>21.801345</td>
          <td>24.711250</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.394034</td>
          <td>0.026590</td>
          <td>24.637770</td>
          <td>0.026961</td>
          <td>19.791043</td>
          <td>0.005018</td>
          <td>26.895160</td>
          <td>0.272268</td>
          <td>23.830180</td>
          <td>0.036043</td>
          <td>16.134923</td>
          <td>0.005003</td>
          <td>26.630147</td>
          <td>20.816874</td>
          <td>23.204242</td>
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
          <td>20.118382</td>
          <td>0.005323</td>
          <td>22.635245</td>
          <td>0.006722</td>
          <td>25.507229</td>
          <td>0.050948</td>
          <td>23.590667</td>
          <td>0.015643</td>
          <td>20.485587</td>
          <td>0.005332</td>
          <td>19.882478</td>
          <td>0.005553</td>
          <td>19.049558</td>
          <td>23.078195</td>
          <td>19.068514</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.689865</td>
          <td>0.005187</td>
          <td>27.314067</td>
          <td>0.274351</td>
          <td>26.634728</td>
          <td>0.137323</td>
          <td>20.415102</td>
          <td>0.005093</td>
          <td>17.906916</td>
          <td>0.005009</td>
          <td>25.053438</td>
          <td>0.233022</td>
          <td>23.950197</td>
          <td>25.207112</td>
          <td>22.076943</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.260628</td>
          <td>0.135664</td>
          <td>21.247955</td>
          <td>0.005208</td>
          <td>25.307205</td>
          <td>0.042660</td>
          <td>21.382603</td>
          <td>0.005428</td>
          <td>22.030420</td>
          <td>0.008647</td>
          <td>26.329356</td>
          <td>0.622837</td>
          <td>27.903417</td>
          <td>23.278518</td>
          <td>21.502251</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.689948</td>
          <td>0.436183</td>
          <td>26.719598</td>
          <td>0.167160</td>
          <td>25.824035</td>
          <td>0.067483</td>
          <td>23.608357</td>
          <td>0.015869</td>
          <td>24.234869</td>
          <td>0.051598</td>
          <td>24.135831</td>
          <td>0.106454</td>
          <td>26.051143</td>
          <td>19.882628</td>
          <td>27.101913</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.860950</td>
          <td>0.095917</td>
          <td>22.053424</td>
          <td>0.005709</td>
          <td>21.591407</td>
          <td>0.005247</td>
          <td>25.668159</td>
          <td>0.095841</td>
          <td>22.931567</td>
          <td>0.016625</td>
          <td>21.463794</td>
          <td>0.010891</td>
          <td>20.779782</td>
          <td>23.822303</td>
          <td>21.943225</td>
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
          <td>27.117194</td>
          <td>23.596183</td>
          <td>26.926223</td>
          <td>28.148389</td>
          <td>19.928134</td>
          <td>20.411181</td>
          <td>25.729636</td>
          <td>0.065558</td>
          <td>25.231403</td>
          <td>0.071759</td>
          <td>17.865789</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.250580</td>
          <td>23.584614</td>
          <td>22.764905</td>
          <td>20.985088</td>
          <td>30.411366</td>
          <td>20.568428</td>
          <td>23.352966</td>
          <td>0.009024</td>
          <td>24.808053</td>
          <td>0.049233</td>
          <td>22.178392</td>
          <td>0.006680</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.072970</td>
          <td>25.499564</td>
          <td>23.253525</td>
          <td>21.105217</td>
          <td>22.243325</td>
          <td>21.188192</td>
          <td>23.052386</td>
          <td>0.007581</td>
          <td>28.948112</td>
          <td>1.225115</td>
          <td>23.443467</td>
          <td>0.015011</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.653015</td>
          <td>23.807871</td>
          <td>22.302071</td>
          <td>21.489838</td>
          <td>28.332367</td>
          <td>23.159480</td>
          <td>19.785277</td>
          <td>0.005008</td>
          <td>21.796001</td>
          <td>0.005891</td>
          <td>24.727412</td>
          <td>0.045817</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.412536</td>
          <td>24.628151</td>
          <td>19.791507</td>
          <td>26.885229</td>
          <td>23.825927</td>
          <td>16.136082</td>
          <td>26.614613</td>
          <td>0.142564</td>
          <td>20.817473</td>
          <td>0.005158</td>
          <td>23.202950</td>
          <td>0.012407</td>
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
          <td>20.128553</td>
          <td>22.627072</td>
          <td>25.471321</td>
          <td>23.589373</td>
          <td>20.483112</td>
          <td>19.878566</td>
          <td>19.042771</td>
          <td>0.005002</td>
          <td>23.065212</td>
          <td>0.011186</td>
          <td>19.071291</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.692132</td>
          <td>27.790354</td>
          <td>26.802240</td>
          <td>20.413273</td>
          <td>17.912021</td>
          <td>25.785145</td>
          <td>23.947689</td>
          <td>0.013896</td>
          <td>25.326092</td>
          <td>0.078041</td>
          <td>22.076012</td>
          <td>0.006423</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.239365</td>
          <td>21.242666</td>
          <td>25.302148</td>
          <td>21.380275</td>
          <td>22.028213</td>
          <td>25.460178</td>
          <td>27.566079</td>
          <td>0.315248</td>
          <td>23.278585</td>
          <td>0.013157</td>
          <td>21.496251</td>
          <td>0.005531</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.921551</td>
          <td>26.895248</td>
          <td>25.882931</td>
          <td>23.609322</td>
          <td>24.187779</td>
          <td>24.195230</td>
          <td>26.082153</td>
          <td>0.089578</td>
          <td>19.883025</td>
          <td>0.005029</td>
          <td>26.851482</td>
          <td>0.287495</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.816556</td>
          <td>22.051822</td>
          <td>21.590319</td>
          <td>25.707265</td>
          <td>22.905520</td>
          <td>21.465334</td>
          <td>20.772920</td>
          <td>0.005049</td>
          <td>23.815625</td>
          <td>0.020512</td>
          <td>21.946760</td>
          <td>0.006149</td>
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
          <td>27.117194</td>
          <td>23.596183</td>
          <td>26.926223</td>
          <td>28.148389</td>
          <td>19.928134</td>
          <td>20.411181</td>
          <td>25.179663</td>
          <td>0.403192</td>
          <td>25.139890</td>
          <td>0.334323</td>
          <td>17.858423</td>
          <td>0.005027</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.250580</td>
          <td>23.584614</td>
          <td>22.764905</td>
          <td>20.985088</td>
          <td>30.411366</td>
          <td>20.568428</td>
          <td>23.514161</td>
          <td>0.100608</td>
          <td>24.643741</td>
          <td>0.223312</td>
          <td>22.219336</td>
          <td>0.029172</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.072970</td>
          <td>25.499564</td>
          <td>23.253525</td>
          <td>21.105217</td>
          <td>22.243325</td>
          <td>21.188192</td>
          <td>23.142757</td>
          <td>0.072486</td>
          <td>27.051360</td>
          <td>1.227313</td>
          <td>23.527442</td>
          <td>0.093224</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.653015</td>
          <td>23.807871</td>
          <td>22.302071</td>
          <td>21.489838</td>
          <td>28.332367</td>
          <td>23.159480</td>
          <td>19.775213</td>
          <td>0.006019</td>
          <td>21.762262</td>
          <td>0.018000</td>
          <td>24.568431</td>
          <td>0.227943</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.412536</td>
          <td>24.628151</td>
          <td>19.791507</td>
          <td>26.885229</td>
          <td>23.825927</td>
          <td>16.136082</td>
          <td>26.849967</td>
          <td>1.226371</td>
          <td>20.819768</td>
          <td>0.008837</td>
          <td>23.258290</td>
          <td>0.073491</td>
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
          <td>20.128553</td>
          <td>22.627072</td>
          <td>25.471321</td>
          <td>23.589373</td>
          <td>20.483112</td>
          <td>19.878566</td>
          <td>19.047795</td>
          <td>0.005286</td>
          <td>23.045856</td>
          <td>0.055667</td>
          <td>19.069837</td>
          <td>0.005249</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.692132</td>
          <td>27.790354</td>
          <td>26.802240</td>
          <td>20.413273</td>
          <td>17.912021</td>
          <td>25.785145</td>
          <td>23.785238</td>
          <td>0.127469</td>
          <td>24.651791</td>
          <td>0.224812</td>
          <td>22.048946</td>
          <td>0.025114</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.239365</td>
          <td>21.242666</td>
          <td>25.302148</td>
          <td>21.380275</td>
          <td>22.028213</td>
          <td>25.460178</td>
          <td>27.482582</td>
          <td>1.691797</td>
          <td>23.464731</td>
          <td>0.080756</td>
          <td>21.498192</td>
          <td>0.015699</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.921551</td>
          <td>26.895248</td>
          <td>25.882931</td>
          <td>23.609322</td>
          <td>24.187779</td>
          <td>24.195230</td>
          <td>24.960687</td>
          <td>0.339873</td>
          <td>19.876846</td>
          <td>0.005863</td>
          <td>26.318044</td>
          <td>0.841237</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.816556</td>
          <td>22.051822</td>
          <td>21.590319</td>
          <td>25.707265</td>
          <td>22.905520</td>
          <td>21.465334</td>
          <td>20.768407</td>
          <td>0.009735</td>
          <td>23.889230</td>
          <td>0.117257</td>
          <td>21.941806</td>
          <td>0.022874</td>
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


