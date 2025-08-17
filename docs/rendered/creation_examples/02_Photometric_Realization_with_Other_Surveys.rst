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
          <td>16.331872</td>
          <td>22.737745</td>
          <td>20.332776</td>
          <td>23.703629</td>
          <td>26.134636</td>
          <td>21.250210</td>
          <td>13.458208</td>
          <td>27.331307</td>
          <td>23.913008</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.885518</td>
          <td>23.546313</td>
          <td>25.604442</td>
          <td>22.774709</td>
          <td>23.964746</td>
          <td>22.529641</td>
          <td>24.181521</td>
          <td>25.867829</td>
          <td>24.388782</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.210794</td>
          <td>25.819818</td>
          <td>23.041401</td>
          <td>27.207654</td>
          <td>26.552632</td>
          <td>26.824486</td>
          <td>20.318299</td>
          <td>22.983023</td>
          <td>19.737149</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.075785</td>
          <td>24.258233</td>
          <td>22.889168</td>
          <td>25.204411</td>
          <td>17.176925</td>
          <td>20.132026</td>
          <td>20.238484</td>
          <td>24.194724</td>
          <td>23.634968</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.034224</td>
          <td>19.078600</td>
          <td>21.447982</td>
          <td>21.508364</td>
          <td>26.202551</td>
          <td>27.668110</td>
          <td>23.311518</td>
          <td>22.854337</td>
          <td>19.098232</td>
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
          <td>25.374397</td>
          <td>24.450414</td>
          <td>25.129570</td>
          <td>22.485121</td>
          <td>20.835490</td>
          <td>30.172031</td>
          <td>26.567123</td>
          <td>28.577288</td>
          <td>24.179274</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.708065</td>
          <td>18.705272</td>
          <td>25.549860</td>
          <td>18.046504</td>
          <td>22.053591</td>
          <td>21.039369</td>
          <td>21.062535</td>
          <td>25.063511</td>
          <td>26.497396</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.056490</td>
          <td>22.288788</td>
          <td>19.840706</td>
          <td>18.049760</td>
          <td>24.873280</td>
          <td>24.824927</td>
          <td>21.065503</td>
          <td>26.159453</td>
          <td>27.632126</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.687860</td>
          <td>21.571629</td>
          <td>24.116515</td>
          <td>22.082798</td>
          <td>23.188512</td>
          <td>21.776898</td>
          <td>23.594423</td>
          <td>16.730542</td>
          <td>21.464321</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.133924</td>
          <td>18.841168</td>
          <td>25.405678</td>
          <td>21.657168</td>
          <td>21.499351</td>
          <td>27.049400</td>
          <td>31.346398</td>
          <td>23.720284</td>
          <td>21.780409</td>
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
          <td>16.326687</td>
          <td>0.005005</td>
          <td>22.733646</td>
          <td>0.006994</td>
          <td>20.332199</td>
          <td>0.005037</td>
          <td>23.657161</td>
          <td>0.016515</td>
          <td>25.903283</td>
          <td>0.219973</td>
          <td>21.244047</td>
          <td>0.009388</td>
          <td>13.458208</td>
          <td>27.331307</td>
          <td>23.913008</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.899708</td>
          <td>0.017602</td>
          <td>23.542166</td>
          <td>0.011119</td>
          <td>25.637736</td>
          <td>0.057207</td>
          <td>22.784418</td>
          <td>0.008747</td>
          <td>23.938432</td>
          <td>0.039666</td>
          <td>22.507717</td>
          <td>0.025307</td>
          <td>24.181521</td>
          <td>25.867829</td>
          <td>24.388782</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.292861</td>
          <td>0.139477</td>
          <td>25.772563</td>
          <td>0.073337</td>
          <td>23.047526</td>
          <td>0.007465</td>
          <td>27.548279</td>
          <td>0.454558</td>
          <td>26.682067</td>
          <td>0.410783</td>
          <td>26.454121</td>
          <td>0.679057</td>
          <td>20.318299</td>
          <td>22.983023</td>
          <td>19.737149</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.309476</td>
          <td>0.324603</td>
          <td>24.237266</td>
          <td>0.019149</td>
          <td>22.880020</td>
          <td>0.006917</td>
          <td>25.207789</td>
          <td>0.063836</td>
          <td>17.178852</td>
          <td>0.005004</td>
          <td>20.130919</td>
          <td>0.005823</td>
          <td>20.238484</td>
          <td>24.194724</td>
          <td>23.634968</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.041713</td>
          <td>0.005292</td>
          <td>19.076627</td>
          <td>0.005013</td>
          <td>21.450302</td>
          <td>0.005197</td>
          <td>21.502244</td>
          <td>0.005519</td>
          <td>26.040924</td>
          <td>0.246525</td>
          <td>27.666079</td>
          <td>1.406665</td>
          <td>23.311518</td>
          <td>22.854337</td>
          <td>19.098232</td>
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
          <td>25.386538</td>
          <td>0.151140</td>
          <td>24.394380</td>
          <td>0.021866</td>
          <td>25.133048</td>
          <td>0.036561</td>
          <td>22.485240</td>
          <td>0.007431</td>
          <td>20.833999</td>
          <td>0.005581</td>
          <td>28.132994</td>
          <td>1.764366</td>
          <td>26.567123</td>
          <td>28.577288</td>
          <td>24.179274</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.700607</td>
          <td>0.005021</td>
          <td>18.708234</td>
          <td>0.005009</td>
          <td>25.539376</td>
          <td>0.052423</td>
          <td>18.050856</td>
          <td>0.005004</td>
          <td>22.045279</td>
          <td>0.008724</td>
          <td>21.047754</td>
          <td>0.008335</td>
          <td>21.062535</td>
          <td>25.063511</td>
          <td>26.497396</td>
        </tr>
        <tr>
          <th>997</th>
          <td>33.148698</td>
          <td>5.698253</td>
          <td>22.287563</td>
          <td>0.006017</td>
          <td>19.839520</td>
          <td>0.005019</td>
          <td>18.055896</td>
          <td>0.005004</td>
          <td>24.937324</td>
          <td>0.096022</td>
          <td>24.879105</td>
          <td>0.201499</td>
          <td>21.065503</td>
          <td>26.159453</td>
          <td>27.632126</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.967620</td>
          <td>0.246210</td>
          <td>21.572939</td>
          <td>0.005339</td>
          <td>24.105027</td>
          <td>0.015158</td>
          <td>22.077351</td>
          <td>0.006303</td>
          <td>23.184869</td>
          <td>0.020550</td>
          <td>21.777330</td>
          <td>0.013768</td>
          <td>23.594423</td>
          <td>16.730542</td>
          <td>21.464321</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.133558</td>
          <td>0.005096</td>
          <td>18.841995</td>
          <td>0.005010</td>
          <td>25.386068</td>
          <td>0.045752</td>
          <td>21.657015</td>
          <td>0.005666</td>
          <td>21.502510</td>
          <td>0.006674</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.346398</td>
          <td>23.720284</td>
          <td>21.780409</td>
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
          <td>16.331872</td>
          <td>22.737745</td>
          <td>20.332776</td>
          <td>23.703629</td>
          <td>26.134636</td>
          <td>21.250210</td>
          <td>13.463644</td>
          <td>0.005000</td>
          <td>28.281733</td>
          <td>0.821809</td>
          <td>23.935427</td>
          <td>0.022748</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.885518</td>
          <td>23.546313</td>
          <td>25.604442</td>
          <td>22.774709</td>
          <td>23.964746</td>
          <td>22.529641</td>
          <td>24.192945</td>
          <td>0.016982</td>
          <td>25.771253</td>
          <td>0.115434</td>
          <td>24.406553</td>
          <td>0.034433</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.210794</td>
          <td>25.819818</td>
          <td>23.041401</td>
          <td>27.207654</td>
          <td>26.552632</td>
          <td>26.824486</td>
          <td>20.313968</td>
          <td>0.005021</td>
          <td>22.987946</td>
          <td>0.010578</td>
          <td>19.740485</td>
          <td>0.005022</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.075785</td>
          <td>24.258233</td>
          <td>22.889168</td>
          <td>25.204411</td>
          <td>17.176925</td>
          <td>20.132026</td>
          <td>20.230714</td>
          <td>0.005018</td>
          <td>24.246133</td>
          <td>0.029871</td>
          <td>23.647611</td>
          <td>0.017779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.034224</td>
          <td>19.078600</td>
          <td>21.447982</td>
          <td>21.508364</td>
          <td>26.202551</td>
          <td>27.668110</td>
          <td>23.293499</td>
          <td>0.008694</td>
          <td>22.849944</td>
          <td>0.009615</td>
          <td>19.099417</td>
          <td>0.005007</td>
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
          <td>25.374397</td>
          <td>24.450414</td>
          <td>25.129570</td>
          <td>22.485121</td>
          <td>20.835490</td>
          <td>30.172031</td>
          <td>26.648791</td>
          <td>0.146824</td>
          <td>27.543026</td>
          <td>0.491913</td>
          <td>24.172188</td>
          <td>0.027984</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.708065</td>
          <td>18.705272</td>
          <td>25.549860</td>
          <td>18.046504</td>
          <td>22.053591</td>
          <td>21.039369</td>
          <td>21.059921</td>
          <td>0.005082</td>
          <td>25.117881</td>
          <td>0.064877</td>
          <td>26.301685</td>
          <td>0.182227</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.056490</td>
          <td>22.288788</td>
          <td>19.840706</td>
          <td>18.049760</td>
          <td>24.873280</td>
          <td>24.824927</td>
          <td>21.072992</td>
          <td>0.005084</td>
          <td>26.372935</td>
          <td>0.193539</td>
          <td>27.462263</td>
          <td>0.463181</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.687860</td>
          <td>21.571629</td>
          <td>24.116515</td>
          <td>22.082798</td>
          <td>23.188512</td>
          <td>21.776898</td>
          <td>23.592063</td>
          <td>0.010609</td>
          <td>16.722372</td>
          <td>0.005000</td>
          <td>21.459539</td>
          <td>0.005498</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.133924</td>
          <td>18.841168</td>
          <td>25.405678</td>
          <td>21.657168</td>
          <td>21.499351</td>
          <td>27.049400</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.664597</td>
          <td>0.018036</td>
          <td>21.780192</td>
          <td>0.005867</td>
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
          <td>16.331872</td>
          <td>22.737745</td>
          <td>20.332776</td>
          <td>23.703629</td>
          <td>26.134636</td>
          <td>21.250210</td>
          <td>13.454498</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.159369</td>
          <td>0.161434</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.885518</td>
          <td>23.546313</td>
          <td>25.604442</td>
          <td>22.774709</td>
          <td>23.964746</td>
          <td>22.529641</td>
          <td>24.440386</td>
          <td>0.222689</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.339761</td>
          <td>0.188194</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.210794</td>
          <td>25.819818</td>
          <td>23.041401</td>
          <td>27.207654</td>
          <td>26.552632</td>
          <td>26.824486</td>
          <td>20.321220</td>
          <td>0.007461</td>
          <td>23.008361</td>
          <td>0.053837</td>
          <td>19.740549</td>
          <td>0.005811</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.075785</td>
          <td>24.258233</td>
          <td>22.889168</td>
          <td>25.204411</td>
          <td>17.176925</td>
          <td>20.132026</td>
          <td>20.236333</td>
          <td>0.007158</td>
          <td>24.225841</td>
          <td>0.156867</td>
          <td>23.469366</td>
          <td>0.088574</td>
        </tr>
        <tr>
          <th>4</th>
          <td>20.034224</td>
          <td>19.078600</td>
          <td>21.447982</td>
          <td>21.508364</td>
          <td>26.202551</td>
          <td>27.668110</td>
          <td>23.391976</td>
          <td>0.090357</td>
          <td>22.808924</td>
          <td>0.045068</td>
          <td>19.098507</td>
          <td>0.005262</td>
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
          <td>25.374397</td>
          <td>24.450414</td>
          <td>25.129570</td>
          <td>22.485121</td>
          <td>20.835490</td>
          <td>30.172031</td>
          <td>26.003536</td>
          <td>0.730859</td>
          <td>27.440687</td>
          <td>1.505495</td>
          <td>23.939156</td>
          <td>0.133567</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.708065</td>
          <td>18.705272</td>
          <td>25.549860</td>
          <td>18.046504</td>
          <td>22.053591</td>
          <td>21.039369</td>
          <td>21.036559</td>
          <td>0.011796</td>
          <td>24.927983</td>
          <td>0.282075</td>
          <td>27.318935</td>
          <td>1.489220</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.056490</td>
          <td>22.288788</td>
          <td>19.840706</td>
          <td>18.049760</td>
          <td>24.873280</td>
          <td>24.824927</td>
          <td>21.063016</td>
          <td>0.012034</td>
          <td>25.625920</td>
          <td>0.485712</td>
          <td>26.926550</td>
          <td>1.210577</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.687860</td>
          <td>21.571629</td>
          <td>24.116515</td>
          <td>22.082798</td>
          <td>23.188512</td>
          <td>21.776898</td>
          <td>23.547243</td>
          <td>0.103570</td>
          <td>16.721949</td>
          <td>0.005003</td>
          <td>21.468897</td>
          <td>0.015326</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.133924</td>
          <td>18.841168</td>
          <td>25.405678</td>
          <td>21.657168</td>
          <td>21.499351</td>
          <td>27.049400</td>
          <td>26.828139</td>
          <td>1.211645</td>
          <td>23.836920</td>
          <td>0.112025</td>
          <td>21.780989</td>
          <td>0.019912</td>
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


