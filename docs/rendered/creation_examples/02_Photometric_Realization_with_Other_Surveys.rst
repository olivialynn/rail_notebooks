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
          <td>20.965787</td>
          <td>20.770729</td>
          <td>19.636326</td>
          <td>21.497215</td>
          <td>25.235135</td>
          <td>24.797086</td>
          <td>20.500964</td>
          <td>24.351428</td>
          <td>24.822541</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.545624</td>
          <td>21.441032</td>
          <td>22.915292</td>
          <td>17.263542</td>
          <td>23.093379</td>
          <td>26.376300</td>
          <td>16.489279</td>
          <td>21.587115</td>
          <td>21.424553</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.676427</td>
          <td>24.664388</td>
          <td>23.351581</td>
          <td>21.505605</td>
          <td>24.784128</td>
          <td>23.041624</td>
          <td>18.132321</td>
          <td>19.307621</td>
          <td>23.191432</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.966910</td>
          <td>27.845586</td>
          <td>19.038138</td>
          <td>22.792160</td>
          <td>21.722435</td>
          <td>24.696200</td>
          <td>21.676894</td>
          <td>25.971538</td>
          <td>28.595749</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.652017</td>
          <td>26.247020</td>
          <td>16.662090</td>
          <td>25.249614</td>
          <td>22.509089</td>
          <td>22.866157</td>
          <td>22.577899</td>
          <td>23.742199</td>
          <td>18.731635</td>
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
          <td>22.302772</td>
          <td>26.091854</td>
          <td>21.810418</td>
          <td>27.975014</td>
          <td>27.665486</td>
          <td>22.814104</td>
          <td>23.458449</td>
          <td>21.629411</td>
          <td>26.874986</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.262275</td>
          <td>25.710144</td>
          <td>26.271369</td>
          <td>22.349460</td>
          <td>25.861609</td>
          <td>17.779220</td>
          <td>25.579804</td>
          <td>25.117955</td>
          <td>28.483057</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.986964</td>
          <td>19.946509</td>
          <td>18.184466</td>
          <td>20.872983</td>
          <td>23.679070</td>
          <td>20.943718</td>
          <td>18.096021</td>
          <td>20.034984</td>
          <td>19.696766</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.969796</td>
          <td>23.518846</td>
          <td>15.567984</td>
          <td>23.918579</td>
          <td>19.813008</td>
          <td>22.505433</td>
          <td>20.474076</td>
          <td>18.787239</td>
          <td>29.439315</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.197686</td>
          <td>27.867977</td>
          <td>22.049455</td>
          <td>17.614417</td>
          <td>20.393672</td>
          <td>21.078744</td>
          <td>27.662035</td>
          <td>24.181003</td>
          <td>21.312102</td>
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
          <td>20.957338</td>
          <td>0.006017</td>
          <td>20.758297</td>
          <td>0.005103</td>
          <td>19.639338</td>
          <td>0.005015</td>
          <td>21.502088</td>
          <td>0.005519</td>
          <td>25.338966</td>
          <td>0.136231</td>
          <td>25.092016</td>
          <td>0.240572</td>
          <td>20.500964</td>
          <td>24.351428</td>
          <td>24.822541</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.504257</td>
          <td>0.070163</td>
          <td>21.435885</td>
          <td>0.005276</td>
          <td>22.913034</td>
          <td>0.007016</td>
          <td>17.256896</td>
          <td>0.005002</td>
          <td>23.089753</td>
          <td>0.018962</td>
          <td>27.074423</td>
          <td>1.011480</td>
          <td>16.489279</td>
          <td>21.587115</td>
          <td>21.424553</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.675988</td>
          <td>0.014749</td>
          <td>24.620736</td>
          <td>0.026565</td>
          <td>23.358551</td>
          <td>0.008859</td>
          <td>21.515357</td>
          <td>0.005530</td>
          <td>24.861856</td>
          <td>0.089863</td>
          <td>23.128754</td>
          <td>0.043714</td>
          <td>18.132321</td>
          <td>19.307621</td>
          <td>23.191432</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.967731</td>
          <td>0.018601</td>
          <td>28.588149</td>
          <td>0.713154</td>
          <td>19.040120</td>
          <td>0.005007</td>
          <td>22.792139</td>
          <td>0.008787</td>
          <td>21.727715</td>
          <td>0.007353</td>
          <td>24.375968</td>
          <td>0.131177</td>
          <td>21.676894</td>
          <td>25.971538</td>
          <td>28.595749</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.647002</td>
          <td>0.005660</td>
          <td>26.183473</td>
          <td>0.105225</td>
          <td>16.663286</td>
          <td>0.005001</td>
          <td>25.384295</td>
          <td>0.074634</td>
          <td>22.473525</td>
          <td>0.011623</td>
          <td>22.809979</td>
          <td>0.032972</td>
          <td>22.577899</td>
          <td>23.742199</td>
          <td>18.731635</td>
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
          <td>22.275254</td>
          <td>0.011011</td>
          <td>26.054284</td>
          <td>0.093975</td>
          <td>21.807329</td>
          <td>0.005349</td>
          <td>27.240198</td>
          <td>0.358738</td>
          <td>28.640721</td>
          <td>1.447175</td>
          <td>22.851987</td>
          <td>0.034216</td>
          <td>23.458449</td>
          <td>21.629411</td>
          <td>26.874986</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.261488</td>
          <td>0.005391</td>
          <td>25.827925</td>
          <td>0.077008</td>
          <td>26.224632</td>
          <td>0.096087</td>
          <td>22.347670</td>
          <td>0.006978</td>
          <td>25.690362</td>
          <td>0.183972</td>
          <td>17.776300</td>
          <td>0.005024</td>
          <td>25.579804</td>
          <td>25.117955</td>
          <td>28.483057</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.940566</td>
          <td>0.042740</td>
          <td>19.947599</td>
          <td>0.005036</td>
          <td>18.199297</td>
          <td>0.005003</td>
          <td>20.871005</td>
          <td>0.005188</td>
          <td>23.628400</td>
          <td>0.030172</td>
          <td>20.931108</td>
          <td>0.007819</td>
          <td>18.096021</td>
          <td>20.034984</td>
          <td>19.696766</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.972447</td>
          <td>0.005027</td>
          <td>23.513574</td>
          <td>0.010897</td>
          <td>15.559063</td>
          <td>0.005000</td>
          <td>23.928673</td>
          <td>0.020730</td>
          <td>19.811547</td>
          <td>0.005115</td>
          <td>22.484135</td>
          <td>0.024795</td>
          <td>20.474076</td>
          <td>18.787239</td>
          <td>29.439315</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.203885</td>
          <td>0.022627</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.053655</td>
          <td>0.005519</td>
          <td>17.614436</td>
          <td>0.005003</td>
          <td>20.393030</td>
          <td>0.005286</td>
          <td>21.081621</td>
          <td>0.008499</td>
          <td>27.662035</td>
          <td>24.181003</td>
          <td>21.312102</td>
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
          <td>20.965787</td>
          <td>20.770729</td>
          <td>19.636326</td>
          <td>21.497215</td>
          <td>25.235135</td>
          <td>24.797086</td>
          <td>20.495172</td>
          <td>0.005029</td>
          <td>24.384605</td>
          <td>0.033768</td>
          <td>24.847771</td>
          <td>0.051007</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.545624</td>
          <td>21.441032</td>
          <td>22.915292</td>
          <td>17.263542</td>
          <td>23.093379</td>
          <td>26.376300</td>
          <td>16.488187</td>
          <td>0.005000</td>
          <td>21.585720</td>
          <td>0.005621</td>
          <td>21.421812</td>
          <td>0.005466</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.676427</td>
          <td>24.664388</td>
          <td>23.351581</td>
          <td>21.505605</td>
          <td>24.784128</td>
          <td>23.041624</td>
          <td>18.139472</td>
          <td>0.005000</td>
          <td>19.300988</td>
          <td>0.005010</td>
          <td>23.218484</td>
          <td>0.012556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.966910</td>
          <td>27.845586</td>
          <td>19.038138</td>
          <td>22.792160</td>
          <td>21.722435</td>
          <td>24.696200</td>
          <td>21.678279</td>
          <td>0.005252</td>
          <td>25.961783</td>
          <td>0.136207</td>
          <td>28.322360</td>
          <td>0.843566</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.652017</td>
          <td>26.247020</td>
          <td>16.662090</td>
          <td>25.249614</td>
          <td>22.509089</td>
          <td>22.866157</td>
          <td>22.580090</td>
          <td>0.006215</td>
          <td>23.735456</td>
          <td>0.019152</td>
          <td>18.734484</td>
          <td>0.005003</td>
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
          <td>22.302772</td>
          <td>26.091854</td>
          <td>21.810418</td>
          <td>27.975014</td>
          <td>27.665486</td>
          <td>22.814104</td>
          <td>23.444576</td>
          <td>0.009580</td>
          <td>21.620451</td>
          <td>0.005659</td>
          <td>26.617068</td>
          <td>0.237318</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.262275</td>
          <td>25.710144</td>
          <td>26.271369</td>
          <td>22.349460</td>
          <td>25.861609</td>
          <td>17.779220</td>
          <td>25.578296</td>
          <td>0.057299</td>
          <td>25.100882</td>
          <td>0.063904</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.986964</td>
          <td>19.946509</td>
          <td>18.184466</td>
          <td>20.872983</td>
          <td>23.679070</td>
          <td>20.943718</td>
          <td>18.102976</td>
          <td>0.005000</td>
          <td>20.040701</td>
          <td>0.005038</td>
          <td>19.695648</td>
          <td>0.005020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.969796</td>
          <td>23.518846</td>
          <td>15.567984</td>
          <td>23.918579</td>
          <td>19.813008</td>
          <td>22.505433</td>
          <td>20.465759</td>
          <td>0.005028</td>
          <td>18.785774</td>
          <td>0.005004</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.197686</td>
          <td>27.867977</td>
          <td>22.049455</td>
          <td>17.614417</td>
          <td>20.393672</td>
          <td>21.078744</td>
          <td>27.322338</td>
          <td>0.258801</td>
          <td>24.118408</td>
          <td>0.026691</td>
          <td>21.311642</td>
          <td>0.005383</td>
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
          <td>20.965787</td>
          <td>20.770729</td>
          <td>19.636326</td>
          <td>21.497215</td>
          <td>25.235135</td>
          <td>24.797086</td>
          <td>20.492636</td>
          <td>0.008187</td>
          <td>24.457568</td>
          <td>0.191046</td>
          <td>25.524241</td>
          <td>0.485107</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.545624</td>
          <td>21.441032</td>
          <td>22.915292</td>
          <td>17.263542</td>
          <td>23.093379</td>
          <td>26.376300</td>
          <td>16.496797</td>
          <td>0.005003</td>
          <td>21.547302</td>
          <td>0.015058</td>
          <td>21.441119</td>
          <td>0.014983</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.676427</td>
          <td>24.664388</td>
          <td>23.351581</td>
          <td>21.505605</td>
          <td>24.784128</td>
          <td>23.041624</td>
          <td>18.124626</td>
          <td>0.005053</td>
          <td>19.306317</td>
          <td>0.005318</td>
          <td>23.141158</td>
          <td>0.066233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.966910</td>
          <td>27.845586</td>
          <td>19.038138</td>
          <td>22.792160</td>
          <td>21.722435</td>
          <td>24.696200</td>
          <td>21.650838</td>
          <td>0.019405</td>
          <td>26.804782</td>
          <td>1.066635</td>
          <td>28.497601</td>
          <td>2.468165</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.652017</td>
          <td>26.247020</td>
          <td>16.662090</td>
          <td>25.249614</td>
          <td>22.509089</td>
          <td>22.866157</td>
          <td>22.577884</td>
          <td>0.043838</td>
          <td>23.613246</td>
          <td>0.092066</td>
          <td>18.737199</td>
          <td>0.005136</td>
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
          <td>22.302772</td>
          <td>26.091854</td>
          <td>21.810418</td>
          <td>27.975014</td>
          <td>27.665486</td>
          <td>22.814104</td>
          <td>23.434564</td>
          <td>0.093810</td>
          <td>21.621956</td>
          <td>0.016010</td>
          <td>26.568900</td>
          <td>0.983640</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.262275</td>
          <td>25.710144</td>
          <td>26.271369</td>
          <td>22.349460</td>
          <td>25.861609</td>
          <td>17.779220</td>
          <td>25.745447</td>
          <td>0.612060</td>
          <td>25.234473</td>
          <td>0.360200</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.986964</td>
          <td>19.946509</td>
          <td>18.184466</td>
          <td>20.872983</td>
          <td>23.679070</td>
          <td>20.943718</td>
          <td>18.099704</td>
          <td>0.005051</td>
          <td>20.032587</td>
          <td>0.006122</td>
          <td>19.695925</td>
          <td>0.005751</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.969796</td>
          <td>23.518846</td>
          <td>15.567984</td>
          <td>23.918579</td>
          <td>19.813008</td>
          <td>22.505433</td>
          <td>20.465499</td>
          <td>0.008061</td>
          <td>18.786843</td>
          <td>0.005124</td>
          <td>25.678740</td>
          <td>0.543339</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.197686</td>
          <td>27.867977</td>
          <td>22.049455</td>
          <td>17.614417</td>
          <td>20.393672</td>
          <td>21.078744</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.187942</td>
          <td>0.151849</td>
          <td>21.327702</td>
          <td>0.013677</td>
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


