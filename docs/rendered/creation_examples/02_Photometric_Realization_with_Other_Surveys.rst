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
          <td>24.296161</td>
          <td>22.796395</td>
          <td>20.438505</td>
          <td>21.453562</td>
          <td>23.412373</td>
          <td>20.151929</td>
          <td>24.568135</td>
          <td>20.703040</td>
          <td>19.487356</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.945281</td>
          <td>24.801610</td>
          <td>26.818200</td>
          <td>30.101590</td>
          <td>25.042201</td>
          <td>22.636804</td>
          <td>23.299194</td>
          <td>17.291005</td>
          <td>22.957098</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.093650</td>
          <td>18.015355</td>
          <td>32.144352</td>
          <td>28.022945</td>
          <td>24.609114</td>
          <td>25.154763</td>
          <td>18.789573</td>
          <td>28.847508</td>
          <td>23.485770</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.852296</td>
          <td>27.590085</td>
          <td>17.721021</td>
          <td>28.944710</td>
          <td>24.605515</td>
          <td>15.595787</td>
          <td>20.698592</td>
          <td>19.486569</td>
          <td>26.414337</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.858272</td>
          <td>26.930236</td>
          <td>26.026261</td>
          <td>21.135521</td>
          <td>25.379709</td>
          <td>21.637395</td>
          <td>21.413760</td>
          <td>23.980803</td>
          <td>27.478247</td>
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
          <td>19.937440</td>
          <td>21.993413</td>
          <td>21.078964</td>
          <td>31.521719</td>
          <td>25.548067</td>
          <td>20.943778</td>
          <td>24.521770</td>
          <td>29.435170</td>
          <td>25.399443</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.851182</td>
          <td>23.746812</td>
          <td>17.779852</td>
          <td>25.305112</td>
          <td>26.929446</td>
          <td>17.643538</td>
          <td>23.029290</td>
          <td>18.985414</td>
          <td>22.329246</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.001802</td>
          <td>22.952523</td>
          <td>25.630817</td>
          <td>20.866088</td>
          <td>27.546427</td>
          <td>19.075056</td>
          <td>18.007210</td>
          <td>25.226254</td>
          <td>18.582206</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.574139</td>
          <td>19.775441</td>
          <td>20.015291</td>
          <td>20.190617</td>
          <td>19.728640</td>
          <td>20.315061</td>
          <td>18.956921</td>
          <td>23.103133</td>
          <td>20.432832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.170405</td>
          <td>21.860701</td>
          <td>18.079773</td>
          <td>20.173074</td>
          <td>24.961169</td>
          <td>16.328771</td>
          <td>23.423661</td>
          <td>25.104710</td>
          <td>23.565086</td>
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
          <td>24.277323</td>
          <td>0.057464</td>
          <td>22.803656</td>
          <td>0.007210</td>
          <td>20.441598</td>
          <td>0.005043</td>
          <td>21.460387</td>
          <td>0.005485</td>
          <td>23.404374</td>
          <td>0.024814</td>
          <td>20.156658</td>
          <td>0.005858</td>
          <td>24.568135</td>
          <td>20.703040</td>
          <td>19.487356</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.940577</td>
          <td>0.008918</td>
          <td>24.815008</td>
          <td>0.031469</td>
          <td>26.982634</td>
          <td>0.184873</td>
          <td>29.835447</td>
          <td>1.804994</td>
          <td>24.821972</td>
          <td>0.086764</td>
          <td>22.606794</td>
          <td>0.027587</td>
          <td>23.299194</td>
          <td>17.291005</td>
          <td>22.957098</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.952797</td>
          <td>0.103916</td>
          <td>18.014139</td>
          <td>0.005004</td>
          <td>28.239100</td>
          <td>0.503455</td>
          <td>27.470787</td>
          <td>0.428679</td>
          <td>24.621485</td>
          <td>0.072694</td>
          <td>24.909417</td>
          <td>0.206685</td>
          <td>18.789573</td>
          <td>28.847508</td>
          <td>23.485770</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.849416</td>
          <td>0.005228</td>
          <td>27.396726</td>
          <td>0.293336</td>
          <td>17.721448</td>
          <td>0.005002</td>
          <td>30.210382</td>
          <td>2.118052</td>
          <td>24.580457</td>
          <td>0.070103</td>
          <td>15.591723</td>
          <td>0.005002</td>
          <td>20.698592</td>
          <td>19.486569</td>
          <td>26.414337</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.191374</td>
          <td>0.295353</td>
          <td>27.002024</td>
          <td>0.212127</td>
          <td>26.049012</td>
          <td>0.082331</td>
          <td>21.134550</td>
          <td>0.005287</td>
          <td>25.309880</td>
          <td>0.132850</td>
          <td>21.642301</td>
          <td>0.012411</td>
          <td>21.413760</td>
          <td>23.980803</td>
          <td>27.478247</td>
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
          <td>19.931192</td>
          <td>0.005253</td>
          <td>21.987099</td>
          <td>0.005640</td>
          <td>21.077759</td>
          <td>0.005110</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.440861</td>
          <td>0.148723</td>
          <td>20.944040</td>
          <td>0.007873</td>
          <td>24.521770</td>
          <td>29.435170</td>
          <td>25.399443</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.852311</td>
          <td>0.005024</td>
          <td>23.771333</td>
          <td>0.013165</td>
          <td>17.778803</td>
          <td>0.005002</td>
          <td>25.247266</td>
          <td>0.066109</td>
          <td>27.447430</td>
          <td>0.714115</td>
          <td>17.639217</td>
          <td>0.005020</td>
          <td>23.029290</td>
          <td>18.985414</td>
          <td>22.329246</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.002292</td>
          <td>0.006083</td>
          <td>22.947868</td>
          <td>0.007723</td>
          <td>25.619879</td>
          <td>0.056307</td>
          <td>20.859520</td>
          <td>0.005185</td>
          <td>26.537728</td>
          <td>0.367375</td>
          <td>19.073606</td>
          <td>0.005153</td>
          <td>18.007210</td>
          <td>25.226254</td>
          <td>18.582206</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.569839</td>
          <td>0.005593</td>
          <td>19.775601</td>
          <td>0.005029</td>
          <td>20.023168</td>
          <td>0.005024</td>
          <td>20.191030</td>
          <td>0.005066</td>
          <td>19.728047</td>
          <td>0.005102</td>
          <td>20.325187</td>
          <td>0.006121</td>
          <td>18.956921</td>
          <td>23.103133</td>
          <td>20.432832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.703719</td>
          <td>0.440752</td>
          <td>21.868276</td>
          <td>0.005533</td>
          <td>18.079081</td>
          <td>0.005003</td>
          <td>20.179544</td>
          <td>0.005065</td>
          <td>24.850787</td>
          <td>0.088992</td>
          <td>16.330289</td>
          <td>0.005004</td>
          <td>23.423661</td>
          <td>25.104710</td>
          <td>23.565086</td>
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
          <td>24.296161</td>
          <td>22.796395</td>
          <td>20.438505</td>
          <td>21.453562</td>
          <td>23.412373</td>
          <td>20.151929</td>
          <td>24.565895</td>
          <td>0.023358</td>
          <td>20.699168</td>
          <td>0.005127</td>
          <td>19.496519</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.945281</td>
          <td>24.801610</td>
          <td>26.818200</td>
          <td>30.101590</td>
          <td>25.042201</td>
          <td>22.636804</td>
          <td>23.286315</td>
          <td>0.008656</td>
          <td>17.278397</td>
          <td>0.005000</td>
          <td>22.962185</td>
          <td>0.010387</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.093650</td>
          <td>18.015355</td>
          <td>32.144352</td>
          <td>28.022945</td>
          <td>24.609114</td>
          <td>25.154763</td>
          <td>18.788082</td>
          <td>0.005001</td>
          <td>30.302687</td>
          <td>2.295018</td>
          <td>23.498739</td>
          <td>0.015706</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.852296</td>
          <td>27.590085</td>
          <td>17.721021</td>
          <td>28.944710</td>
          <td>24.605515</td>
          <td>15.595787</td>
          <td>20.692299</td>
          <td>0.005042</td>
          <td>19.492914</td>
          <td>0.005014</td>
          <td>26.676510</td>
          <td>0.249245</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.858272</td>
          <td>26.930236</td>
          <td>26.026261</td>
          <td>21.135521</td>
          <td>25.379709</td>
          <td>21.637395</td>
          <td>21.421341</td>
          <td>0.005159</td>
          <td>24.015551</td>
          <td>0.024392</td>
          <td>27.670391</td>
          <td>0.540060</td>
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
          <td>19.937440</td>
          <td>21.993413</td>
          <td>21.078964</td>
          <td>31.521719</td>
          <td>25.548067</td>
          <td>20.943778</td>
          <td>24.528422</td>
          <td>0.022610</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.497771</td>
          <td>0.090820</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.851182</td>
          <td>23.746812</td>
          <td>17.779852</td>
          <td>25.305112</td>
          <td>26.929446</td>
          <td>17.643538</td>
          <td>23.027427</td>
          <td>0.007485</td>
          <td>18.978831</td>
          <td>0.005005</td>
          <td>22.332149</td>
          <td>0.007144</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.001802</td>
          <td>22.952523</td>
          <td>25.630817</td>
          <td>20.866088</td>
          <td>27.546427</td>
          <td>19.075056</td>
          <td>18.002880</td>
          <td>0.005000</td>
          <td>25.197060</td>
          <td>0.069605</td>
          <td>18.581882</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.574139</td>
          <td>19.775441</td>
          <td>20.015291</td>
          <td>20.190617</td>
          <td>19.728640</td>
          <td>20.315061</td>
          <td>18.961119</td>
          <td>0.005002</td>
          <td>23.087152</td>
          <td>0.011369</td>
          <td>20.430375</td>
          <td>0.005078</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.170405</td>
          <td>21.860701</td>
          <td>18.079773</td>
          <td>20.173074</td>
          <td>24.961169</td>
          <td>16.328771</td>
          <td>23.427159</td>
          <td>0.009470</td>
          <td>25.086015</td>
          <td>0.063064</td>
          <td>23.568307</td>
          <td>0.016637</td>
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
          <td>24.296161</td>
          <td>22.796395</td>
          <td>20.438505</td>
          <td>21.453562</td>
          <td>23.412373</td>
          <td>20.151929</td>
          <td>24.669836</td>
          <td>0.269046</td>
          <td>20.692859</td>
          <td>0.008188</td>
          <td>19.496174</td>
          <td>0.005531</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.945281</td>
          <td>24.801610</td>
          <td>26.818200</td>
          <td>30.101590</td>
          <td>25.042201</td>
          <td>22.636804</td>
          <td>23.308657</td>
          <td>0.083953</td>
          <td>17.287589</td>
          <td>0.005008</td>
          <td>22.989617</td>
          <td>0.057879</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.093650</td>
          <td>18.015355</td>
          <td>32.144352</td>
          <td>28.022945</td>
          <td>24.609114</td>
          <td>25.154763</td>
          <td>18.789640</td>
          <td>0.005180</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.543470</td>
          <td>0.094549</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.852296</td>
          <td>27.590085</td>
          <td>17.721021</td>
          <td>28.944710</td>
          <td>24.605515</td>
          <td>15.595787</td>
          <td>20.686423</td>
          <td>0.009220</td>
          <td>19.488258</td>
          <td>0.005439</td>
          <td>26.961699</td>
          <td>1.234324</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.858272</td>
          <td>26.930236</td>
          <td>26.026261</td>
          <td>21.135521</td>
          <td>25.379709</td>
          <td>21.637395</td>
          <td>21.410965</td>
          <td>0.015865</td>
          <td>24.159624</td>
          <td>0.148198</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>19.937440</td>
          <td>21.993413</td>
          <td>21.078964</td>
          <td>31.521719</td>
          <td>25.548067</td>
          <td>20.943778</td>
          <td>24.661491</td>
          <td>0.267221</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.941239</td>
          <td>0.309043</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.851182</td>
          <td>23.746812</td>
          <td>17.779852</td>
          <td>25.305112</td>
          <td>26.929446</td>
          <td>17.643538</td>
          <td>22.949080</td>
          <td>0.061026</td>
          <td>18.988139</td>
          <td>0.005179</td>
          <td>22.278723</td>
          <td>0.030744</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.001802</td>
          <td>22.952523</td>
          <td>25.630817</td>
          <td>20.866088</td>
          <td>27.546427</td>
          <td>19.075056</td>
          <td>18.007221</td>
          <td>0.005043</td>
          <td>26.218923</td>
          <td>0.738424</td>
          <td>18.586675</td>
          <td>0.005104</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.574139</td>
          <td>19.775441</td>
          <td>20.015291</td>
          <td>20.190617</td>
          <td>19.728640</td>
          <td>20.315061</td>
          <td>18.954629</td>
          <td>0.005242</td>
          <td>23.137461</td>
          <td>0.060398</td>
          <td>20.442192</td>
          <td>0.007542</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.170405</td>
          <td>21.860701</td>
          <td>18.079773</td>
          <td>20.173074</td>
          <td>24.961169</td>
          <td>16.328771</td>
          <td>23.387480</td>
          <td>0.090000</td>
          <td>25.743032</td>
          <td>0.529420</td>
          <td>23.543246</td>
          <td>0.094530</td>
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


