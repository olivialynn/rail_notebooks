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
          <td>19.303497</td>
          <td>26.037791</td>
          <td>23.405359</td>
          <td>20.585742</td>
          <td>19.897440</td>
          <td>29.744292</td>
          <td>28.029565</td>
          <td>20.265778</td>
          <td>21.375458</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.501649</td>
          <td>26.030368</td>
          <td>23.939567</td>
          <td>26.663237</td>
          <td>19.692737</td>
          <td>23.438167</td>
          <td>18.869121</td>
          <td>22.463977</td>
          <td>23.614660</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.346668</td>
          <td>20.177550</td>
          <td>26.605790</td>
          <td>24.049017</td>
          <td>25.312579</td>
          <td>25.690257</td>
          <td>20.751201</td>
          <td>22.390828</td>
          <td>24.813227</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.528333</td>
          <td>28.633465</td>
          <td>22.635314</td>
          <td>24.565055</td>
          <td>20.701124</td>
          <td>22.466772</td>
          <td>23.445337</td>
          <td>24.740239</td>
          <td>25.283771</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.045023</td>
          <td>25.965051</td>
          <td>18.203844</td>
          <td>25.497002</td>
          <td>30.435450</td>
          <td>23.407554</td>
          <td>20.769146</td>
          <td>17.358827</td>
          <td>26.840933</td>
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
          <td>29.173292</td>
          <td>21.221532</td>
          <td>23.547727</td>
          <td>19.783396</td>
          <td>15.419120</td>
          <td>31.001947</td>
          <td>19.474306</td>
          <td>19.677827</td>
          <td>17.957267</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.579481</td>
          <td>21.912601</td>
          <td>23.187114</td>
          <td>21.772095</td>
          <td>26.409160</td>
          <td>25.922162</td>
          <td>21.020946</td>
          <td>31.096169</td>
          <td>23.128705</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.553683</td>
          <td>22.073980</td>
          <td>29.951498</td>
          <td>22.831657</td>
          <td>22.387455</td>
          <td>23.482755</td>
          <td>27.474421</td>
          <td>16.061921</td>
          <td>21.585061</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.592579</td>
          <td>21.917747</td>
          <td>22.741694</td>
          <td>24.125922</td>
          <td>19.656845</td>
          <td>20.212138</td>
          <td>25.949883</td>
          <td>24.307318</td>
          <td>22.806643</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.312095</td>
          <td>26.526793</td>
          <td>23.803626</td>
          <td>27.974664</td>
          <td>21.986691</td>
          <td>25.791555</td>
          <td>27.630016</td>
          <td>24.623625</td>
          <td>22.731298</td>
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
          <td>19.298213</td>
          <td>0.005116</td>
          <td>25.862019</td>
          <td>0.079358</td>
          <td>23.402335</td>
          <td>0.009102</td>
          <td>20.591728</td>
          <td>0.005122</td>
          <td>19.898342</td>
          <td>0.005132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.029565</td>
          <td>20.265778</td>
          <td>21.375458</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.061666</td>
          <td>0.573566</td>
          <td>26.019763</td>
          <td>0.091171</td>
          <td>23.936163</td>
          <td>0.013266</td>
          <td>26.289349</td>
          <td>0.164187</td>
          <td>19.689337</td>
          <td>0.005096</td>
          <td>23.448223</td>
          <td>0.058045</td>
          <td>18.869121</td>
          <td>22.463977</td>
          <td>23.614660</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.342743</td>
          <td>0.011536</td>
          <td>20.176909</td>
          <td>0.005047</td>
          <td>26.730603</td>
          <td>0.149137</td>
          <td>24.081471</td>
          <td>0.023634</td>
          <td>25.089303</td>
          <td>0.109682</td>
          <td>25.248915</td>
          <td>0.273577</td>
          <td>20.751201</td>
          <td>22.390828</td>
          <td>24.813227</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.646287</td>
          <td>0.079484</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.623540</td>
          <td>0.006290</td>
          <td>24.534766</td>
          <td>0.035142</td>
          <td>20.699284</td>
          <td>0.005468</td>
          <td>22.485651</td>
          <td>0.024827</td>
          <td>23.445337</td>
          <td>24.740239</td>
          <td>25.283771</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.045340</td>
          <td>0.006150</td>
          <td>25.851963</td>
          <td>0.078658</td>
          <td>18.197378</td>
          <td>0.005003</td>
          <td>25.559918</td>
          <td>0.087143</td>
          <td>27.674274</td>
          <td>0.829343</td>
          <td>23.335109</td>
          <td>0.052500</td>
          <td>20.769146</td>
          <td>17.358827</td>
          <td>26.840933</td>
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
          <td>28.271044</td>
          <td>1.234956</td>
          <td>21.225950</td>
          <td>0.005202</td>
          <td>23.572724</td>
          <td>0.010173</td>
          <td>19.782675</td>
          <td>0.005037</td>
          <td>15.416220</td>
          <td>0.005001</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.474306</td>
          <td>19.677827</td>
          <td>17.957267</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.587451</td>
          <td>0.005608</td>
          <td>21.910807</td>
          <td>0.005569</td>
          <td>23.192139</td>
          <td>0.008046</td>
          <td>21.772698</td>
          <td>0.005803</td>
          <td>26.430084</td>
          <td>0.337570</td>
          <td>25.441288</td>
          <td>0.319429</td>
          <td>21.020946</td>
          <td>31.096169</td>
          <td>23.128705</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.983382</td>
          <td>0.542151</td>
          <td>22.070820</td>
          <td>0.005728</td>
          <td>27.994243</td>
          <td>0.418943</td>
          <td>22.833507</td>
          <td>0.009013</td>
          <td>22.395162</td>
          <td>0.010983</td>
          <td>23.414325</td>
          <td>0.056324</td>
          <td>27.474421</td>
          <td>16.061921</td>
          <td>21.585061</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.615225</td>
          <td>0.183560</td>
          <td>21.923013</td>
          <td>0.005580</td>
          <td>22.758721</td>
          <td>0.006592</td>
          <td>24.128458</td>
          <td>0.024614</td>
          <td>19.657451</td>
          <td>0.005091</td>
          <td>20.224080</td>
          <td>0.005955</td>
          <td>25.949883</td>
          <td>24.307318</td>
          <td>22.806643</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.301724</td>
          <td>0.058711</td>
          <td>26.627174</td>
          <td>0.154477</td>
          <td>23.794860</td>
          <td>0.011918</td>
          <td>27.853228</td>
          <td>0.568727</td>
          <td>22.007694</td>
          <td>0.008532</td>
          <td>26.137344</td>
          <td>0.543181</td>
          <td>27.630016</td>
          <td>24.623625</td>
          <td>22.731298</td>
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
          <td>19.303497</td>
          <td>26.037791</td>
          <td>23.405359</td>
          <td>20.585742</td>
          <td>19.897440</td>
          <td>29.744292</td>
          <td>28.134110</td>
          <td>0.488673</td>
          <td>20.258776</td>
          <td>0.005057</td>
          <td>21.375367</td>
          <td>0.005429</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.501649</td>
          <td>26.030368</td>
          <td>23.939567</td>
          <td>26.663237</td>
          <td>19.692737</td>
          <td>23.438167</td>
          <td>18.877491</td>
          <td>0.005001</td>
          <td>22.476043</td>
          <td>0.007676</td>
          <td>23.594952</td>
          <td>0.017011</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.346668</td>
          <td>20.177550</td>
          <td>26.605790</td>
          <td>24.049017</td>
          <td>25.312579</td>
          <td>25.690257</td>
          <td>20.746095</td>
          <td>0.005046</td>
          <td>22.393492</td>
          <td>0.007358</td>
          <td>24.837966</td>
          <td>0.050563</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.528333</td>
          <td>28.633465</td>
          <td>22.635314</td>
          <td>24.565055</td>
          <td>20.701124</td>
          <td>22.466772</td>
          <td>23.453996</td>
          <td>0.009641</td>
          <td>24.803959</td>
          <td>0.049053</td>
          <td>25.343953</td>
          <td>0.079285</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.045023</td>
          <td>25.965051</td>
          <td>18.203844</td>
          <td>25.497002</td>
          <td>30.435450</td>
          <td>23.407554</td>
          <td>20.762036</td>
          <td>0.005048</td>
          <td>17.363764</td>
          <td>0.005000</td>
          <td>26.908837</td>
          <td>0.301108</td>
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
          <td>29.173292</td>
          <td>21.221532</td>
          <td>23.547727</td>
          <td>19.783396</td>
          <td>15.419120</td>
          <td>31.001947</td>
          <td>19.468742</td>
          <td>0.005004</td>
          <td>19.688861</td>
          <td>0.005020</td>
          <td>17.959246</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.579481</td>
          <td>21.912601</td>
          <td>23.187114</td>
          <td>21.772095</td>
          <td>26.409160</td>
          <td>25.922162</td>
          <td>21.016208</td>
          <td>0.005076</td>
          <td>28.289589</td>
          <td>0.825987</td>
          <td>23.118299</td>
          <td>0.011635</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.553683</td>
          <td>22.073980</td>
          <td>29.951498</td>
          <td>22.831657</td>
          <td>22.387455</td>
          <td>23.482755</td>
          <td>27.163990</td>
          <td>0.227103</td>
          <td>16.059154</td>
          <td>0.005000</td>
          <td>21.590443</td>
          <td>0.005626</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.592579</td>
          <td>21.917747</td>
          <td>22.741694</td>
          <td>24.125922</td>
          <td>19.656845</td>
          <td>20.212138</td>
          <td>25.930130</td>
          <td>0.078321</td>
          <td>24.315483</td>
          <td>0.031761</td>
          <td>22.807347</td>
          <td>0.009347</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.312095</td>
          <td>26.526793</td>
          <td>23.803626</td>
          <td>27.974664</td>
          <td>21.986691</td>
          <td>25.791555</td>
          <td>27.359697</td>
          <td>0.266830</td>
          <td>24.589680</td>
          <td>0.040525</td>
          <td>22.743793</td>
          <td>0.008972</td>
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
          <td>19.303497</td>
          <td>26.037791</td>
          <td>23.405359</td>
          <td>20.585742</td>
          <td>19.897440</td>
          <td>29.744292</td>
          <td>25.424108</td>
          <td>0.485059</td>
          <td>20.265051</td>
          <td>0.006644</td>
          <td>21.368937</td>
          <td>0.014134</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.501649</td>
          <td>26.030368</td>
          <td>23.939567</td>
          <td>26.663237</td>
          <td>19.692737</td>
          <td>23.438167</td>
          <td>18.870150</td>
          <td>0.005208</td>
          <td>22.414803</td>
          <td>0.031742</td>
          <td>23.490358</td>
          <td>0.090229</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.346668</td>
          <td>20.177550</td>
          <td>26.605790</td>
          <td>24.049017</td>
          <td>25.312579</td>
          <td>25.690257</td>
          <td>20.746229</td>
          <td>0.009591</td>
          <td>22.406491</td>
          <td>0.031509</td>
          <td>24.634748</td>
          <td>0.240811</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.528333</td>
          <td>28.633465</td>
          <td>22.635314</td>
          <td>24.565055</td>
          <td>20.701124</td>
          <td>22.466772</td>
          <td>23.461794</td>
          <td>0.096085</td>
          <td>24.776409</td>
          <td>0.249224</td>
          <td>25.696222</td>
          <td>0.550256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.045023</td>
          <td>25.965051</td>
          <td>18.203844</td>
          <td>25.497002</td>
          <td>30.435450</td>
          <td>23.407554</td>
          <td>20.790658</td>
          <td>0.009883</td>
          <td>17.364975</td>
          <td>0.005009</td>
          <td>25.959511</td>
          <td>0.662686</td>
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
          <td>29.173292</td>
          <td>21.221532</td>
          <td>23.547727</td>
          <td>19.783396</td>
          <td>15.419120</td>
          <td>31.001947</td>
          <td>19.470507</td>
          <td>0.005604</td>
          <td>19.679019</td>
          <td>0.005613</td>
          <td>17.951089</td>
          <td>0.005032</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.579481</td>
          <td>21.912601</td>
          <td>23.187114</td>
          <td>21.772095</td>
          <td>26.409160</td>
          <td>25.922162</td>
          <td>21.023178</td>
          <td>0.011678</td>
          <td>25.877681</td>
          <td>0.583376</td>
          <td>23.091966</td>
          <td>0.063399</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.553683</td>
          <td>22.073980</td>
          <td>29.951498</td>
          <td>22.831657</td>
          <td>22.387455</td>
          <td>23.482755</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.068966</td>
          <td>0.005001</td>
          <td>21.575175</td>
          <td>0.016732</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.592579</td>
          <td>21.917747</td>
          <td>22.741694</td>
          <td>24.125922</td>
          <td>19.656845</td>
          <td>20.212138</td>
          <td>25.422889</td>
          <td>0.484620</td>
          <td>24.233258</td>
          <td>0.157867</td>
          <td>22.895530</td>
          <td>0.053225</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.312095</td>
          <td>26.526793</td>
          <td>23.803626</td>
          <td>27.974664</td>
          <td>21.986691</td>
          <td>25.791555</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.852953</td>
          <td>0.265364</td>
          <td>22.729201</td>
          <td>0.045890</td>
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


