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
          <td>27.547647</td>
          <td>26.457336</td>
          <td>15.613492</td>
          <td>29.510550</td>
          <td>19.095656</td>
          <td>20.497465</td>
          <td>25.223310</td>
          <td>21.117342</td>
          <td>21.150837</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.305775</td>
          <td>23.041267</td>
          <td>25.990408</td>
          <td>24.618167</td>
          <td>25.284138</td>
          <td>29.247732</td>
          <td>23.546155</td>
          <td>20.985121</td>
          <td>22.220628</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.991156</td>
          <td>25.152169</td>
          <td>23.749354</td>
          <td>26.576041</td>
          <td>24.008468</td>
          <td>22.920724</td>
          <td>17.900042</td>
          <td>21.643494</td>
          <td>21.406168</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.419502</td>
          <td>24.117114</td>
          <td>22.873059</td>
          <td>22.632126</td>
          <td>22.203495</td>
          <td>24.063951</td>
          <td>20.438888</td>
          <td>28.118096</td>
          <td>26.427598</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.704598</td>
          <td>20.823273</td>
          <td>22.964299</td>
          <td>26.662606</td>
          <td>24.139122</td>
          <td>24.715015</td>
          <td>27.599504</td>
          <td>23.080151</td>
          <td>25.837126</td>
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
          <td>20.481049</td>
          <td>21.654202</td>
          <td>26.067108</td>
          <td>21.408189</td>
          <td>28.162295</td>
          <td>20.852647</td>
          <td>24.530210</td>
          <td>21.544710</td>
          <td>15.697759</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.816989</td>
          <td>26.580129</td>
          <td>27.353258</td>
          <td>19.393035</td>
          <td>23.962527</td>
          <td>26.777695</td>
          <td>26.546489</td>
          <td>28.641516</td>
          <td>26.208603</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.363175</td>
          <td>19.095871</td>
          <td>28.808795</td>
          <td>27.523676</td>
          <td>24.534043</td>
          <td>25.376797</td>
          <td>25.501162</td>
          <td>24.393413</td>
          <td>22.632809</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.340559</td>
          <td>23.304124</td>
          <td>24.249846</td>
          <td>22.822134</td>
          <td>24.047602</td>
          <td>20.020505</td>
          <td>18.975818</td>
          <td>18.822989</td>
          <td>22.791063</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.384186</td>
          <td>25.546131</td>
          <td>20.170894</td>
          <td>18.839859</td>
          <td>23.954141</td>
          <td>24.164872</td>
          <td>18.181164</td>
          <td>23.704723</td>
          <td>22.065589</td>
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
          <td>27.457353</td>
          <td>0.753522</td>
          <td>26.607955</td>
          <td>0.151955</td>
          <td>15.616148</td>
          <td>0.005000</td>
          <td>27.740815</td>
          <td>0.524301</td>
          <td>19.100875</td>
          <td>0.005041</td>
          <td>20.495691</td>
          <td>0.006463</td>
          <td>25.223310</td>
          <td>21.117342</td>
          <td>21.150837</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.290444</td>
          <td>0.006620</td>
          <td>23.042921</td>
          <td>0.008118</td>
          <td>25.965985</td>
          <td>0.076513</td>
          <td>24.635993</td>
          <td>0.038433</td>
          <td>25.357616</td>
          <td>0.138441</td>
          <td>27.558748</td>
          <td>1.329791</td>
          <td>23.546155</td>
          <td>20.985121</td>
          <td>22.220628</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.545558</td>
          <td>0.173056</td>
          <td>25.229974</td>
          <td>0.045370</td>
          <td>23.765010</td>
          <td>0.011658</td>
          <td>27.052550</td>
          <td>0.309166</td>
          <td>24.058255</td>
          <td>0.044111</td>
          <td>22.950878</td>
          <td>0.037340</td>
          <td>17.900042</td>
          <td>21.643494</td>
          <td>21.406168</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.399130</td>
          <td>0.063966</td>
          <td>24.118902</td>
          <td>0.017358</td>
          <td>22.870181</td>
          <td>0.006889</td>
          <td>22.627893</td>
          <td>0.007997</td>
          <td>22.208921</td>
          <td>0.009668</td>
          <td>24.125977</td>
          <td>0.105541</td>
          <td>20.438888</td>
          <td>28.118096</td>
          <td>26.427598</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.703957</td>
          <td>0.007860</td>
          <td>20.820889</td>
          <td>0.005113</td>
          <td>22.967142</td>
          <td>0.007187</td>
          <td>26.483229</td>
          <td>0.193522</td>
          <td>24.101633</td>
          <td>0.045842</td>
          <td>24.941491</td>
          <td>0.212305</td>
          <td>27.599504</td>
          <td>23.080151</td>
          <td>25.837126</td>
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
          <td>20.487285</td>
          <td>0.005530</td>
          <td>21.653863</td>
          <td>0.005384</td>
          <td>25.894355</td>
          <td>0.071818</td>
          <td>21.409378</td>
          <td>0.005447</td>
          <td>27.156555</td>
          <td>0.583623</td>
          <td>20.860264</td>
          <td>0.007541</td>
          <td>24.530210</td>
          <td>21.544710</td>
          <td>15.697759</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.902384</td>
          <td>0.041333</td>
          <td>26.522958</td>
          <td>0.141257</td>
          <td>27.323760</td>
          <td>0.245786</td>
          <td>19.392440</td>
          <td>0.005022</td>
          <td>23.924738</td>
          <td>0.039188</td>
          <td>25.947032</td>
          <td>0.472220</td>
          <td>26.546489</td>
          <td>28.641516</td>
          <td>26.208603</td>
        </tr>
        <tr>
          <th>997</th>
          <td>31.788664</td>
          <td>4.352400</td>
          <td>19.089652</td>
          <td>0.005013</td>
          <td>28.146561</td>
          <td>0.470045</td>
          <td>33.193548</td>
          <td>4.946171</td>
          <td>24.484135</td>
          <td>0.064370</td>
          <td>26.168788</td>
          <td>0.555663</td>
          <td>25.501162</td>
          <td>24.393413</td>
          <td>22.632809</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.342747</td>
          <td>0.005436</td>
          <td>23.292976</td>
          <td>0.009405</td>
          <td>24.242684</td>
          <td>0.016958</td>
          <td>22.827644</td>
          <td>0.008980</td>
          <td>24.021705</td>
          <td>0.042704</td>
          <td>20.020288</td>
          <td>0.005690</td>
          <td>18.975818</td>
          <td>18.822989</td>
          <td>22.791063</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.385579</td>
          <td>0.005129</td>
          <td>25.510394</td>
          <td>0.058160</td>
          <td>20.171872</td>
          <td>0.005030</td>
          <td>18.840126</td>
          <td>0.005011</td>
          <td>24.009607</td>
          <td>0.042249</td>
          <td>24.105091</td>
          <td>0.103630</td>
          <td>18.181164</td>
          <td>23.704723</td>
          <td>22.065589</td>
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
          <td>27.547647</td>
          <td>26.457336</td>
          <td>15.613492</td>
          <td>29.510550</td>
          <td>19.095656</td>
          <td>20.497465</td>
          <td>25.205815</td>
          <td>0.041111</td>
          <td>21.121288</td>
          <td>0.005273</td>
          <td>21.149297</td>
          <td>0.005287</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.305775</td>
          <td>23.041267</td>
          <td>25.990408</td>
          <td>24.618167</td>
          <td>25.284138</td>
          <td>29.247732</td>
          <td>23.554904</td>
          <td>0.010333</td>
          <td>20.978798</td>
          <td>0.005211</td>
          <td>22.228913</td>
          <td>0.006821</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.991156</td>
          <td>25.152169</td>
          <td>23.749354</td>
          <td>26.576041</td>
          <td>24.008468</td>
          <td>22.920724</td>
          <td>17.903066</td>
          <td>0.005000</td>
          <td>21.644525</td>
          <td>0.005687</td>
          <td>21.408889</td>
          <td>0.005455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.419502</td>
          <td>24.117114</td>
          <td>22.873059</td>
          <td>22.632126</td>
          <td>22.203495</td>
          <td>24.063951</td>
          <td>20.438345</td>
          <td>0.005026</td>
          <td>27.661779</td>
          <td>0.536693</td>
          <td>26.454611</td>
          <td>0.207293</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.704598</td>
          <td>20.823273</td>
          <td>22.964299</td>
          <td>26.662606</td>
          <td>24.139122</td>
          <td>24.715015</td>
          <td>27.443969</td>
          <td>0.285752</td>
          <td>23.085995</td>
          <td>0.011359</td>
          <td>25.764125</td>
          <td>0.114718</td>
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
          <td>20.481049</td>
          <td>21.654202</td>
          <td>26.067108</td>
          <td>21.408189</td>
          <td>28.162295</td>
          <td>20.852647</td>
          <td>24.532290</td>
          <td>0.022686</td>
          <td>21.549931</td>
          <td>0.005583</td>
          <td>15.713866</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.816989</td>
          <td>26.580129</td>
          <td>27.353258</td>
          <td>19.393035</td>
          <td>23.962527</td>
          <td>26.777695</td>
          <td>26.469745</td>
          <td>0.125766</td>
          <td>28.139358</td>
          <td>0.748555</td>
          <td>26.614378</td>
          <td>0.236791</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.363175</td>
          <td>19.095871</td>
          <td>28.808795</td>
          <td>27.523676</td>
          <td>24.534043</td>
          <td>25.376797</td>
          <td>25.479430</td>
          <td>0.052467</td>
          <td>24.435394</td>
          <td>0.035326</td>
          <td>22.623210</td>
          <td>0.008334</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.340559</td>
          <td>23.304124</td>
          <td>24.249846</td>
          <td>22.822134</td>
          <td>24.047602</td>
          <td>20.020505</td>
          <td>18.966281</td>
          <td>0.005002</td>
          <td>18.831104</td>
          <td>0.005004</td>
          <td>22.781705</td>
          <td>0.009192</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.384186</td>
          <td>25.546131</td>
          <td>20.170894</td>
          <td>18.839859</td>
          <td>23.954141</td>
          <td>24.164872</td>
          <td>18.185522</td>
          <td>0.005000</td>
          <td>23.749903</td>
          <td>0.019390</td>
          <td>22.064883</td>
          <td>0.006397</td>
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
          <td>27.547647</td>
          <td>26.457336</td>
          <td>15.613492</td>
          <td>29.510550</td>
          <td>19.095656</td>
          <td>20.497465</td>
          <td>25.040787</td>
          <td>0.361986</td>
          <td>21.113418</td>
          <td>0.010773</td>
          <td>21.156863</td>
          <td>0.011978</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.305775</td>
          <td>23.041267</td>
          <td>25.990408</td>
          <td>24.618167</td>
          <td>25.284138</td>
          <td>29.247732</td>
          <td>23.404743</td>
          <td>0.091379</td>
          <td>20.975599</td>
          <td>0.009782</td>
          <td>22.220231</td>
          <td>0.029195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.991156</td>
          <td>25.152169</td>
          <td>23.749354</td>
          <td>26.576041</td>
          <td>24.008468</td>
          <td>22.920724</td>
          <td>17.892034</td>
          <td>0.005035</td>
          <td>21.636574</td>
          <td>0.016204</td>
          <td>21.397290</td>
          <td>0.014460</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.419502</td>
          <td>24.117114</td>
          <td>22.873059</td>
          <td>22.632126</td>
          <td>22.203495</td>
          <td>24.063951</td>
          <td>20.434797</td>
          <td>0.007924</td>
          <td>28.946509</td>
          <td>2.785902</td>
          <td>25.762055</td>
          <td>0.576909</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.704598</td>
          <td>20.823273</td>
          <td>22.964299</td>
          <td>26.662606</td>
          <td>24.139122</td>
          <td>24.715015</td>
          <td>27.202930</td>
          <td>1.477298</td>
          <td>22.995122</td>
          <td>0.053206</td>
          <td>27.254200</td>
          <td>1.441277</td>
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
          <td>20.481049</td>
          <td>21.654202</td>
          <td>26.067108</td>
          <td>21.408189</td>
          <td>28.162295</td>
          <td>20.852647</td>
          <td>24.536027</td>
          <td>0.241066</td>
          <td>21.560547</td>
          <td>0.015222</td>
          <td>15.689767</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.816989</td>
          <td>26.580129</td>
          <td>27.353258</td>
          <td>19.393035</td>
          <td>23.962527</td>
          <td>26.777695</td>
          <td>26.795029</td>
          <td>1.189493</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.650390</td>
          <td>1.745678</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.363175</td>
          <td>19.095871</td>
          <td>28.808795</td>
          <td>27.523676</td>
          <td>24.534043</td>
          <td>25.376797</td>
          <td>25.341542</td>
          <td>0.456031</td>
          <td>24.397656</td>
          <td>0.181606</td>
          <td>22.658994</td>
          <td>0.043106</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.340559</td>
          <td>23.304124</td>
          <td>24.249846</td>
          <td>22.822134</td>
          <td>24.047602</td>
          <td>20.020505</td>
          <td>18.975056</td>
          <td>0.005251</td>
          <td>18.812167</td>
          <td>0.005130</td>
          <td>22.777112</td>
          <td>0.047893</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.384186</td>
          <td>25.546131</td>
          <td>20.170894</td>
          <td>18.839859</td>
          <td>23.954141</td>
          <td>24.164872</td>
          <td>18.183749</td>
          <td>0.005060</td>
          <td>23.825925</td>
          <td>0.110954</td>
          <td>22.068597</td>
          <td>0.025550</td>
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


