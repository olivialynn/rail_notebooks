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
          <td>15.828718</td>
          <td>26.937498</td>
          <td>20.356781</td>
          <td>23.035761</td>
          <td>19.793265</td>
          <td>23.547577</td>
          <td>23.857612</td>
          <td>21.897468</td>
          <td>21.333744</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.866149</td>
          <td>23.072206</td>
          <td>25.645317</td>
          <td>25.096998</td>
          <td>27.258857</td>
          <td>20.190337</td>
          <td>18.925667</td>
          <td>26.471137</td>
          <td>23.812812</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.186855</td>
          <td>21.075022</td>
          <td>21.980926</td>
          <td>17.676033</td>
          <td>24.380341</td>
          <td>22.155918</td>
          <td>23.402115</td>
          <td>25.112681</td>
          <td>22.955985</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.061068</td>
          <td>23.635544</td>
          <td>19.252532</td>
          <td>23.125975</td>
          <td>19.085292</td>
          <td>23.515773</td>
          <td>20.261810</td>
          <td>21.344745</td>
          <td>27.227907</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.261467</td>
          <td>25.838093</td>
          <td>26.672334</td>
          <td>30.191577</td>
          <td>25.647550</td>
          <td>18.267589</td>
          <td>21.084278</td>
          <td>22.651295</td>
          <td>20.188402</td>
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
          <td>22.255220</td>
          <td>22.966945</td>
          <td>22.207685</td>
          <td>21.190595</td>
          <td>25.913156</td>
          <td>23.704516</td>
          <td>26.354961</td>
          <td>25.360378</td>
          <td>24.074610</td>
        </tr>
        <tr>
          <th>996</th>
          <td>30.336892</td>
          <td>19.979927</td>
          <td>27.362947</td>
          <td>22.268618</td>
          <td>22.812647</td>
          <td>21.160419</td>
          <td>28.419097</td>
          <td>22.042462</td>
          <td>24.161638</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214806</td>
          <td>23.617373</td>
          <td>18.297741</td>
          <td>26.647906</td>
          <td>29.736650</td>
          <td>18.775186</td>
          <td>27.665953</td>
          <td>21.949571</td>
          <td>23.389197</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.735434</td>
          <td>25.804512</td>
          <td>22.771385</td>
          <td>23.675669</td>
          <td>18.936043</td>
          <td>23.897918</td>
          <td>25.043672</td>
          <td>25.247979</td>
          <td>24.479135</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708757</td>
          <td>24.776714</td>
          <td>20.993915</td>
          <td>24.448191</td>
          <td>19.219788</td>
          <td>16.114722</td>
          <td>25.681374</td>
          <td>26.426505</td>
          <td>26.225161</td>
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
          <td>15.837208</td>
          <td>0.005003</td>
          <td>27.178905</td>
          <td>0.245635</td>
          <td>20.351179</td>
          <td>0.005038</td>
          <td>23.050971</td>
          <td>0.010395</td>
          <td>19.794769</td>
          <td>0.005112</td>
          <td>23.441391</td>
          <td>0.057694</td>
          <td>23.857612</td>
          <td>21.897468</td>
          <td>21.333744</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.865879</td>
          <td>0.005009</td>
          <td>23.061903</td>
          <td>0.008203</td>
          <td>25.665143</td>
          <td>0.058615</td>
          <td>25.066843</td>
          <td>0.056332</td>
          <td>27.424280</td>
          <td>0.703022</td>
          <td>20.193686</td>
          <td>0.005910</td>
          <td>18.925667</td>
          <td>26.471137</td>
          <td>23.812812</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.193426</td>
          <td>0.005013</td>
          <td>21.076664</td>
          <td>0.005162</td>
          <td>21.988661</td>
          <td>0.005467</td>
          <td>17.673584</td>
          <td>0.005003</td>
          <td>24.393523</td>
          <td>0.059400</td>
          <td>22.149373</td>
          <td>0.018630</td>
          <td>23.402115</td>
          <td>25.112681</td>
          <td>22.955985</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.064845</td>
          <td>0.005301</td>
          <td>23.636832</td>
          <td>0.011904</td>
          <td>19.254509</td>
          <td>0.005009</td>
          <td>23.119063</td>
          <td>0.010902</td>
          <td>19.086345</td>
          <td>0.005040</td>
          <td>23.462260</td>
          <td>0.058772</td>
          <td>20.261810</td>
          <td>21.344745</td>
          <td>27.227907</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.276864</td>
          <td>0.024065</td>
          <td>25.801634</td>
          <td>0.075243</td>
          <td>26.503519</td>
          <td>0.122578</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.021860</td>
          <td>0.242684</td>
          <td>18.264285</td>
          <td>0.005046</td>
          <td>21.084278</td>
          <td>22.651295</td>
          <td>20.188402</td>
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
          <td>22.268236</td>
          <td>0.010959</td>
          <td>22.963366</td>
          <td>0.007785</td>
          <td>22.195069</td>
          <td>0.005652</td>
          <td>21.189036</td>
          <td>0.005313</td>
          <td>25.630856</td>
          <td>0.174924</td>
          <td>23.692394</td>
          <td>0.072065</td>
          <td>26.354961</td>
          <td>25.360378</td>
          <td>24.074610</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>19.987627</td>
          <td>0.005037</td>
          <td>26.917320</td>
          <td>0.174919</td>
          <td>22.274318</td>
          <td>0.006768</td>
          <td>22.787941</td>
          <td>0.014798</td>
          <td>21.171804</td>
          <td>0.008971</td>
          <td>28.419097</td>
          <td>22.042462</td>
          <td>24.161638</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.221067</td>
          <td>0.022957</td>
          <td>23.619684</td>
          <td>0.011756</td>
          <td>18.302783</td>
          <td>0.005003</td>
          <td>27.470070</td>
          <td>0.428445</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.780276</td>
          <td>0.005098</td>
          <td>27.665953</td>
          <td>21.949571</td>
          <td>23.389197</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.336025</td>
          <td>0.694540</td>
          <td>25.825410</td>
          <td>0.076838</td>
          <td>22.768143</td>
          <td>0.006615</td>
          <td>23.697956</td>
          <td>0.017079</td>
          <td>18.944067</td>
          <td>0.005033</td>
          <td>23.919640</td>
          <td>0.088067</td>
          <td>25.043672</td>
          <td>25.247979</td>
          <td>24.479135</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708714</td>
          <td>0.007878</td>
          <td>24.828632</td>
          <td>0.031847</td>
          <td>21.003731</td>
          <td>0.005098</td>
          <td>24.487064</td>
          <td>0.033693</td>
          <td>19.213272</td>
          <td>0.005048</td>
          <td>16.115727</td>
          <td>0.005003</td>
          <td>25.681374</td>
          <td>26.426505</td>
          <td>26.225161</td>
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
          <td>15.828718</td>
          <td>26.937498</td>
          <td>20.356781</td>
          <td>23.035761</td>
          <td>19.793265</td>
          <td>23.547577</td>
          <td>23.865126</td>
          <td>0.013019</td>
          <td>21.896284</td>
          <td>0.006056</td>
          <td>21.328167</td>
          <td>0.005395</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.866149</td>
          <td>23.072206</td>
          <td>25.645317</td>
          <td>25.096998</td>
          <td>27.258857</td>
          <td>20.190337</td>
          <td>18.918834</td>
          <td>0.005002</td>
          <td>26.385961</td>
          <td>0.195675</td>
          <td>23.815121</td>
          <td>0.020503</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.186855</td>
          <td>21.075022</td>
          <td>21.980926</td>
          <td>17.676033</td>
          <td>24.380341</td>
          <td>22.155918</td>
          <td>23.410472</td>
          <td>0.009366</td>
          <td>25.222592</td>
          <td>0.071200</td>
          <td>22.924839</td>
          <td>0.010119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.061068</td>
          <td>23.635544</td>
          <td>19.252532</td>
          <td>23.125975</td>
          <td>19.085292</td>
          <td>23.515773</td>
          <td>20.258419</td>
          <td>0.005019</td>
          <td>21.340571</td>
          <td>0.005403</td>
          <td>26.986269</td>
          <td>0.320370</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.261467</td>
          <td>25.838093</td>
          <td>26.672334</td>
          <td>30.191577</td>
          <td>25.647550</td>
          <td>18.267589</td>
          <td>21.077205</td>
          <td>0.005085</td>
          <td>22.656217</td>
          <td>0.008500</td>
          <td>20.196499</td>
          <td>0.005051</td>
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
          <td>22.255220</td>
          <td>22.966945</td>
          <td>22.207685</td>
          <td>21.190595</td>
          <td>25.913156</td>
          <td>23.704516</td>
          <td>26.528174</td>
          <td>0.132303</td>
          <td>25.291509</td>
          <td>0.075687</td>
          <td>24.052153</td>
          <td>0.025185</td>
        </tr>
        <tr>
          <th>996</th>
          <td>30.336892</td>
          <td>19.979927</td>
          <td>27.362947</td>
          <td>22.268618</td>
          <td>22.812647</td>
          <td>21.160419</td>
          <td>28.671778</td>
          <td>0.715417</td>
          <td>22.041322</td>
          <td>0.006344</td>
          <td>24.149931</td>
          <td>0.027441</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214806</td>
          <td>23.617373</td>
          <td>18.297741</td>
          <td>26.647906</td>
          <td>29.736650</td>
          <td>18.775186</td>
          <td>27.560966</td>
          <td>0.313962</td>
          <td>21.956573</td>
          <td>0.006168</td>
          <td>23.361351</td>
          <td>0.014049</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.735434</td>
          <td>25.804512</td>
          <td>22.771385</td>
          <td>23.675669</td>
          <td>18.936043</td>
          <td>23.897918</td>
          <td>25.035678</td>
          <td>0.035335</td>
          <td>25.124075</td>
          <td>0.065235</td>
          <td>24.492936</td>
          <td>0.037181</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708757</td>
          <td>24.776714</td>
          <td>20.993915</td>
          <td>24.448191</td>
          <td>19.219788</td>
          <td>16.114722</td>
          <td>25.607121</td>
          <td>0.058788</td>
          <td>26.202075</td>
          <td>0.167430</td>
          <td>26.065091</td>
          <td>0.148897</td>
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
          <td>15.828718</td>
          <td>26.937498</td>
          <td>20.356781</td>
          <td>23.035761</td>
          <td>19.793265</td>
          <td>23.547577</td>
          <td>23.929129</td>
          <td>0.144359</td>
          <td>21.893563</td>
          <td>0.020127</td>
          <td>21.320342</td>
          <td>0.013597</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.866149</td>
          <td>23.072206</td>
          <td>25.645317</td>
          <td>25.096998</td>
          <td>27.258857</td>
          <td>20.190337</td>
          <td>18.920194</td>
          <td>0.005227</td>
          <td>28.517294</td>
          <td>2.396407</td>
          <td>23.831966</td>
          <td>0.121703</td>
        </tr>
        <tr>
          <th>2</th>
          <td>17.186855</td>
          <td>21.075022</td>
          <td>21.980926</td>
          <td>17.676033</td>
          <td>24.380341</td>
          <td>22.155918</td>
          <td>23.496661</td>
          <td>0.099074</td>
          <td>24.910476</td>
          <td>0.278095</td>
          <td>22.965592</td>
          <td>0.056654</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.061068</td>
          <td>23.635544</td>
          <td>19.252532</td>
          <td>23.125975</td>
          <td>19.085292</td>
          <td>23.515773</td>
          <td>20.270627</td>
          <td>0.007276</td>
          <td>21.347540</td>
          <td>0.012842</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.261467</td>
          <td>25.838093</td>
          <td>26.672334</td>
          <td>30.191577</td>
          <td>25.647550</td>
          <td>18.267589</td>
          <td>21.099053</td>
          <td>0.012369</td>
          <td>22.740907</td>
          <td>0.042417</td>
          <td>20.195552</td>
          <td>0.006727</td>
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
          <td>22.255220</td>
          <td>22.966945</td>
          <td>22.207685</td>
          <td>21.190595</td>
          <td>25.913156</td>
          <td>23.704516</td>
          <td>29.014881</td>
          <td>3.035846</td>
          <td>25.450986</td>
          <td>0.425819</td>
          <td>24.089363</td>
          <td>0.152035</td>
        </tr>
        <tr>
          <th>996</th>
          <td>30.336892</td>
          <td>19.979927</td>
          <td>27.362947</td>
          <td>22.268618</td>
          <td>22.812647</td>
          <td>21.160419</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.020791</td>
          <td>0.022461</td>
          <td>24.131207</td>
          <td>0.157590</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.214806</td>
          <td>23.617373</td>
          <td>18.297741</td>
          <td>26.647906</td>
          <td>29.736650</td>
          <td>18.775186</td>
          <td>25.867910</td>
          <td>0.666531</td>
          <td>21.953495</td>
          <td>0.021191</td>
          <td>23.192195</td>
          <td>0.069305</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.735434</td>
          <td>25.804512</td>
          <td>22.771385</td>
          <td>23.675669</td>
          <td>18.936043</td>
          <td>23.897918</td>
          <td>25.848420</td>
          <td>0.657633</td>
          <td>25.302117</td>
          <td>0.379726</td>
          <td>24.418651</td>
          <td>0.201131</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708757</td>
          <td>24.776714</td>
          <td>20.993915</td>
          <td>24.448191</td>
          <td>19.219788</td>
          <td>16.114722</td>
          <td>27.291500</td>
          <td>1.543830</td>
          <td>25.697793</td>
          <td>0.512186</td>
          <td>26.522492</td>
          <td>0.956228</td>
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


