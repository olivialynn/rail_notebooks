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
          <td>21.514487</td>
          <td>21.556280</td>
          <td>25.222850</td>
          <td>26.358269</td>
          <td>27.454294</td>
          <td>22.746177</td>
          <td>26.687892</td>
          <td>26.190754</td>
          <td>23.473139</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.189736</td>
          <td>20.545172</td>
          <td>20.922308</td>
          <td>19.267225</td>
          <td>17.224702</td>
          <td>23.359560</td>
          <td>26.919238</td>
          <td>25.360926</td>
          <td>18.123638</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.128794</td>
          <td>23.795081</td>
          <td>25.068989</td>
          <td>27.058125</td>
          <td>22.923835</td>
          <td>24.862132</td>
          <td>19.364865</td>
          <td>29.495324</td>
          <td>26.088634</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.686753</td>
          <td>24.938946</td>
          <td>24.136903</td>
          <td>24.848597</td>
          <td>20.642775</td>
          <td>15.730374</td>
          <td>29.003903</td>
          <td>22.257692</td>
          <td>28.008680</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.150856</td>
          <td>22.101981</td>
          <td>18.340790</td>
          <td>25.985771</td>
          <td>24.121540</td>
          <td>17.930346</td>
          <td>19.815200</td>
          <td>20.781908</td>
          <td>27.993555</td>
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
          <td>20.276037</td>
          <td>18.753478</td>
          <td>21.908055</td>
          <td>19.965382</td>
          <td>30.645588</td>
          <td>23.744937</td>
          <td>25.644852</td>
          <td>22.885427</td>
          <td>25.258916</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.939019</td>
          <td>22.360634</td>
          <td>24.963525</td>
          <td>25.297976</td>
          <td>20.985909</td>
          <td>30.464628</td>
          <td>25.765157</td>
          <td>26.825532</td>
          <td>21.713787</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.639628</td>
          <td>19.090261</td>
          <td>16.493646</td>
          <td>23.839472</td>
          <td>23.276279</td>
          <td>20.541054</td>
          <td>23.687566</td>
          <td>21.768445</td>
          <td>19.613449</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.479440</td>
          <td>22.862037</td>
          <td>19.485933</td>
          <td>26.962722</td>
          <td>25.148608</td>
          <td>21.870388</td>
          <td>18.246256</td>
          <td>24.216978</td>
          <td>23.681718</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.908791</td>
          <td>23.201696</td>
          <td>24.281920</td>
          <td>22.217844</td>
          <td>22.321678</td>
          <td>21.658389</td>
          <td>25.624211</td>
          <td>22.167685</td>
          <td>20.143583</td>
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
          <td>21.520385</td>
          <td>0.007227</td>
          <td>21.548104</td>
          <td>0.005327</td>
          <td>25.223312</td>
          <td>0.039602</td>
          <td>26.425875</td>
          <td>0.184377</td>
          <td>26.693877</td>
          <td>0.414516</td>
          <td>22.741277</td>
          <td>0.031038</td>
          <td>26.687892</td>
          <td>26.190754</td>
          <td>23.473139</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.186314</td>
          <td>0.005034</td>
          <td>20.549316</td>
          <td>0.005077</td>
          <td>20.921077</td>
          <td>0.005087</td>
          <td>19.265013</td>
          <td>0.005018</td>
          <td>17.229563</td>
          <td>0.005004</td>
          <td>23.273158</td>
          <td>0.049691</td>
          <td>26.919238</td>
          <td>25.360926</td>
          <td>18.123638</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.128489</td>
          <td>0.005327</td>
          <td>23.805791</td>
          <td>0.013518</td>
          <td>25.051058</td>
          <td>0.034007</td>
          <td>27.026991</td>
          <td>0.302893</td>
          <td>22.943938</td>
          <td>0.016795</td>
          <td>24.308032</td>
          <td>0.123678</td>
          <td>19.364865</td>
          <td>29.495324</td>
          <td>26.088634</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.688284</td>
          <td>0.007800</td>
          <td>24.890469</td>
          <td>0.033624</td>
          <td>24.136014</td>
          <td>0.015542</td>
          <td>24.784766</td>
          <td>0.043852</td>
          <td>20.649877</td>
          <td>0.005432</td>
          <td>15.732034</td>
          <td>0.005002</td>
          <td>29.003903</td>
          <td>22.257692</td>
          <td>28.008680</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.157575</td>
          <td>0.021764</td>
          <td>22.102563</td>
          <td>0.005765</td>
          <td>18.338277</td>
          <td>0.005003</td>
          <td>26.028947</td>
          <td>0.131266</td>
          <td>24.092582</td>
          <td>0.045476</td>
          <td>17.932610</td>
          <td>0.005029</td>
          <td>19.815200</td>
          <td>20.781908</td>
          <td>27.993555</td>
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
          <td>20.273884</td>
          <td>0.005397</td>
          <td>18.756470</td>
          <td>0.005009</td>
          <td>21.911832</td>
          <td>0.005413</td>
          <td>19.964816</td>
          <td>0.005048</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.718046</td>
          <td>0.073719</td>
          <td>25.644852</td>
          <td>22.885427</td>
          <td>25.258916</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.128077</td>
          <td>0.120998</td>
          <td>22.362044</td>
          <td>0.006139</td>
          <td>24.998366</td>
          <td>0.032463</td>
          <td>25.452238</td>
          <td>0.079251</td>
          <td>20.975081</td>
          <td>0.005729</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.765157</td>
          <td>26.825532</td>
          <td>21.713787</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.639385</td>
          <td>0.005020</td>
          <td>19.091136</td>
          <td>0.005013</td>
          <td>16.488290</td>
          <td>0.005001</td>
          <td>23.858289</td>
          <td>0.019529</td>
          <td>23.300158</td>
          <td>0.022679</td>
          <td>20.543234</td>
          <td>0.006575</td>
          <td>23.687566</td>
          <td>21.768445</td>
          <td>19.613449</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.478680</td>
          <td>0.012714</td>
          <td>22.844897</td>
          <td>0.007347</td>
          <td>19.484719</td>
          <td>0.005012</td>
          <td>26.592953</td>
          <td>0.212183</td>
          <td>25.070317</td>
          <td>0.107879</td>
          <td>21.876601</td>
          <td>0.014894</td>
          <td>18.246256</td>
          <td>24.216978</td>
          <td>23.681718</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.914153</td>
          <td>0.005248</td>
          <td>23.205221</td>
          <td>0.008909</td>
          <td>24.276996</td>
          <td>0.017446</td>
          <td>22.216821</td>
          <td>0.006619</td>
          <td>22.327491</td>
          <td>0.010473</td>
          <td>21.656726</td>
          <td>0.012547</td>
          <td>25.624211</td>
          <td>22.167685</td>
          <td>20.143583</td>
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
          <td>21.514487</td>
          <td>21.556280</td>
          <td>25.222850</td>
          <td>26.358269</td>
          <td>27.454294</td>
          <td>22.746177</td>
          <td>26.626213</td>
          <td>0.143997</td>
          <td>26.064743</td>
          <td>0.148852</td>
          <td>23.487499</td>
          <td>0.015562</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.189736</td>
          <td>20.545172</td>
          <td>20.922308</td>
          <td>19.267225</td>
          <td>17.224702</td>
          <td>23.359560</td>
          <td>26.648241</td>
          <td>0.146754</td>
          <td>25.350607</td>
          <td>0.079753</td>
          <td>18.126041</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.128794</td>
          <td>23.795081</td>
          <td>25.068989</td>
          <td>27.058125</td>
          <td>22.923835</td>
          <td>24.862132</td>
          <td>19.362877</td>
          <td>0.005004</td>
          <td>32.834766</td>
          <td>4.701726</td>
          <td>25.899441</td>
          <td>0.129049</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.686753</td>
          <td>24.938946</td>
          <td>24.136903</td>
          <td>24.848597</td>
          <td>20.642775</td>
          <td>15.730374</td>
          <td>29.515430</td>
          <td>1.203116</td>
          <td>22.256015</td>
          <td>0.006902</td>
          <td>27.388162</td>
          <td>0.438019</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.150856</td>
          <td>22.101981</td>
          <td>18.340790</td>
          <td>25.985771</td>
          <td>24.121540</td>
          <td>17.930346</td>
          <td>19.799136</td>
          <td>0.005008</td>
          <td>20.787322</td>
          <td>0.005149</td>
          <td>27.438709</td>
          <td>0.455060</td>
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
          <td>20.276037</td>
          <td>18.753478</td>
          <td>21.908055</td>
          <td>19.965382</td>
          <td>30.645588</td>
          <td>23.744937</td>
          <td>25.647464</td>
          <td>0.060938</td>
          <td>22.881869</td>
          <td>0.009824</td>
          <td>25.265140</td>
          <td>0.073939</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.939019</td>
          <td>22.360634</td>
          <td>24.963525</td>
          <td>25.297976</td>
          <td>20.985909</td>
          <td>30.464628</td>
          <td>25.718808</td>
          <td>0.064930</td>
          <td>27.121675</td>
          <td>0.356602</td>
          <td>21.716330</td>
          <td>0.005778</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.639628</td>
          <td>19.090261</td>
          <td>16.493646</td>
          <td>23.839472</td>
          <td>23.276279</td>
          <td>20.541054</td>
          <td>23.685205</td>
          <td>0.011352</td>
          <td>21.775115</td>
          <td>0.005860</td>
          <td>19.608047</td>
          <td>0.005017</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.479440</td>
          <td>22.862037</td>
          <td>19.485933</td>
          <td>26.962722</td>
          <td>25.148608</td>
          <td>21.870388</td>
          <td>18.253295</td>
          <td>0.005000</td>
          <td>24.223113</td>
          <td>0.029270</td>
          <td>23.712734</td>
          <td>0.018786</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.908791</td>
          <td>23.201696</td>
          <td>24.281920</td>
          <td>22.217844</td>
          <td>22.321678</td>
          <td>21.658389</td>
          <td>25.552310</td>
          <td>0.055988</td>
          <td>22.160019</td>
          <td>0.006631</td>
          <td>20.142798</td>
          <td>0.005046</td>
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
          <td>21.514487</td>
          <td>21.556280</td>
          <td>25.222850</td>
          <td>26.358269</td>
          <td>27.454294</td>
          <td>22.746177</td>
          <td>26.449992</td>
          <td>0.972414</td>
          <td>25.840189</td>
          <td>0.567949</td>
          <td>23.494208</td>
          <td>0.090535</td>
        </tr>
        <tr>
          <th>1</th>
          <td>18.189736</td>
          <td>20.545172</td>
          <td>20.922308</td>
          <td>19.267225</td>
          <td>17.224702</td>
          <td>23.359560</td>
          <td>29.857807</td>
          <td>3.842382</td>
          <td>25.608555</td>
          <td>0.479481</td>
          <td>18.126391</td>
          <td>0.005045</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.128794</td>
          <td>23.795081</td>
          <td>25.068989</td>
          <td>27.058125</td>
          <td>22.923835</td>
          <td>24.862132</td>
          <td>19.367451</td>
          <td>0.005505</td>
          <td>26.676342</td>
          <td>0.988080</td>
          <td>27.024472</td>
          <td>1.277351</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.686753</td>
          <td>24.938946</td>
          <td>24.136903</td>
          <td>24.848597</td>
          <td>20.642775</td>
          <td>15.730374</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.316315</td>
          <td>0.029095</td>
          <td>26.563794</td>
          <td>0.980601</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.150856</td>
          <td>22.101981</td>
          <td>18.340790</td>
          <td>25.985771</td>
          <td>24.121540</td>
          <td>17.930346</td>
          <td>19.811185</td>
          <td>0.006083</td>
          <td>20.792173</td>
          <td>0.008687</td>
          <td>26.998222</td>
          <td>1.259263</td>
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
          <td>20.276037</td>
          <td>18.753478</td>
          <td>21.908055</td>
          <td>19.965382</td>
          <td>30.645588</td>
          <td>23.744937</td>
          <td>26.855774</td>
          <td>1.230304</td>
          <td>22.899939</td>
          <td>0.048878</td>
          <td>25.539654</td>
          <td>0.490685</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.939019</td>
          <td>22.360634</td>
          <td>24.963525</td>
          <td>25.297976</td>
          <td>20.985909</td>
          <td>30.464628</td>
          <td>26.071362</td>
          <td>0.764615</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.700006</td>
          <td>0.018584</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.639628</td>
          <td>19.090261</td>
          <td>16.493646</td>
          <td>23.839472</td>
          <td>23.276279</td>
          <td>20.541054</td>
          <td>23.660135</td>
          <td>0.114319</td>
          <td>21.762961</td>
          <td>0.018011</td>
          <td>19.617813</td>
          <td>0.005656</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.479440</td>
          <td>22.862037</td>
          <td>19.485933</td>
          <td>26.962722</td>
          <td>25.148608</td>
          <td>21.870388</td>
          <td>18.258937</td>
          <td>0.005068</td>
          <td>24.325572</td>
          <td>0.170817</td>
          <td>23.894316</td>
          <td>0.128477</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.908791</td>
          <td>23.201696</td>
          <td>24.281920</td>
          <td>22.217844</td>
          <td>22.321678</td>
          <td>21.658389</td>
          <td>24.917876</td>
          <td>0.328533</td>
          <td>22.137238</td>
          <td>0.024858</td>
          <td>20.147448</td>
          <td>0.006598</td>
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


