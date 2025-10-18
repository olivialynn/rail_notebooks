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
          <td>23.312172</td>
          <td>24.033768</td>
          <td>23.175960</td>
          <td>21.078288</td>
          <td>26.533145</td>
          <td>24.279512</td>
          <td>21.982284</td>
          <td>21.321837</td>
          <td>16.553222</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.242852</td>
          <td>15.697660</td>
          <td>22.367128</td>
          <td>21.522410</td>
          <td>17.714344</td>
          <td>25.490646</td>
          <td>23.633152</td>
          <td>25.065857</td>
          <td>16.863805</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.091090</td>
          <td>22.872018</td>
          <td>22.334597</td>
          <td>23.474047</td>
          <td>22.483730</td>
          <td>23.685264</td>
          <td>23.673342</td>
          <td>19.112856</td>
          <td>25.871508</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759544</td>
          <td>23.195705</td>
          <td>24.305993</td>
          <td>21.918731</td>
          <td>25.482999</td>
          <td>18.193326</td>
          <td>26.719290</td>
          <td>19.353918</td>
          <td>29.469043</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.466462</td>
          <td>20.939735</td>
          <td>24.226951</td>
          <td>26.167589</td>
          <td>17.829389</td>
          <td>21.534681</td>
          <td>23.492107</td>
          <td>21.028769</td>
          <td>19.139433</td>
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
          <td>26.382362</td>
          <td>23.942146</td>
          <td>23.007739</td>
          <td>23.004419</td>
          <td>25.177428</td>
          <td>21.395948</td>
          <td>19.264254</td>
          <td>22.098368</td>
          <td>22.236878</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.703860</td>
          <td>18.186866</td>
          <td>22.144580</td>
          <td>21.333144</td>
          <td>21.247101</td>
          <td>25.445221</td>
          <td>22.055043</td>
          <td>13.454347</td>
          <td>18.671201</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.988151</td>
          <td>22.556139</td>
          <td>15.852135</td>
          <td>22.391501</td>
          <td>27.662741</td>
          <td>25.806251</td>
          <td>26.935798</td>
          <td>25.248562</td>
          <td>21.776834</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.008794</td>
          <td>20.354513</td>
          <td>23.537590</td>
          <td>25.474680</td>
          <td>21.206230</td>
          <td>31.117343</td>
          <td>29.543402</td>
          <td>21.001850</td>
          <td>22.091616</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.764395</td>
          <td>22.110564</td>
          <td>21.419048</td>
          <td>23.205773</td>
          <td>26.533064</td>
          <td>19.892330</td>
          <td>26.279475</td>
          <td>24.000191</td>
          <td>25.405642</td>
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
          <td>23.311743</td>
          <td>0.024788</td>
          <td>24.037564</td>
          <td>0.016242</td>
          <td>23.177068</td>
          <td>0.007980</td>
          <td>21.087667</td>
          <td>0.005266</td>
          <td>26.185628</td>
          <td>0.277487</td>
          <td>24.230392</td>
          <td>0.115607</td>
          <td>21.982284</td>
          <td>21.321837</td>
          <td>16.553222</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.317810</td>
          <td>0.059549</td>
          <td>15.698942</td>
          <td>0.005000</td>
          <td>22.362776</td>
          <td>0.005853</td>
          <td>21.528948</td>
          <td>0.005542</td>
          <td>17.717585</td>
          <td>0.005007</td>
          <td>25.384553</td>
          <td>0.305259</td>
          <td>23.633152</td>
          <td>25.065857</td>
          <td>16.863805</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.035444</td>
          <td>0.111660</td>
          <td>22.875287</td>
          <td>0.007453</td>
          <td>22.326469</td>
          <td>0.005805</td>
          <td>23.476074</td>
          <td>0.014270</td>
          <td>22.488672</td>
          <td>0.011753</td>
          <td>23.791639</td>
          <td>0.078671</td>
          <td>23.673342</td>
          <td>19.112856</td>
          <td>25.871508</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.760284</td>
          <td>0.005772</td>
          <td>23.206773</td>
          <td>0.008917</td>
          <td>24.287932</td>
          <td>0.017605</td>
          <td>21.932000</td>
          <td>0.006035</td>
          <td>25.299468</td>
          <td>0.131660</td>
          <td>18.195160</td>
          <td>0.005042</td>
          <td>26.719290</td>
          <td>19.353918</td>
          <td>29.469043</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.480569</td>
          <td>0.028640</td>
          <td>20.932297</td>
          <td>0.005132</td>
          <td>24.219392</td>
          <td>0.016636</td>
          <td>26.129970</td>
          <td>0.143224</td>
          <td>17.828017</td>
          <td>0.005008</td>
          <td>21.530825</td>
          <td>0.011427</td>
          <td>23.492107</td>
          <td>21.028769</td>
          <td>19.139433</td>
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
          <td>26.035801</td>
          <td>0.260354</td>
          <td>23.941468</td>
          <td>0.015035</td>
          <td>23.003954</td>
          <td>0.007311</td>
          <td>23.008257</td>
          <td>0.010095</td>
          <td>25.053613</td>
          <td>0.106316</td>
          <td>21.387545</td>
          <td>0.010326</td>
          <td>19.264254</td>
          <td>22.098368</td>
          <td>22.236878</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.699697</td>
          <td>0.007843</td>
          <td>18.176583</td>
          <td>0.005005</td>
          <td>22.137510</td>
          <td>0.005594</td>
          <td>21.332214</td>
          <td>0.005394</td>
          <td>21.247382</td>
          <td>0.006125</td>
          <td>26.096694</td>
          <td>0.527366</td>
          <td>22.055043</td>
          <td>13.454347</td>
          <td>18.671201</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.840582</td>
          <td>0.221681</td>
          <td>22.562646</td>
          <td>0.006545</td>
          <td>15.851832</td>
          <td>0.005000</td>
          <td>22.385894</td>
          <td>0.007095</td>
          <td>27.982694</td>
          <td>1.004844</td>
          <td>25.816463</td>
          <td>0.427960</td>
          <td>26.935798</td>
          <td>25.248562</td>
          <td>21.776834</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.129915</td>
          <td>0.121191</td>
          <td>20.348364</td>
          <td>0.005059</td>
          <td>23.532546</td>
          <td>0.009901</td>
          <td>25.579829</td>
          <td>0.088683</td>
          <td>21.207984</td>
          <td>0.006057</td>
          <td>26.443301</td>
          <td>0.674041</td>
          <td>29.543402</td>
          <td>21.001850</td>
          <td>22.091616</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.760429</td>
          <td>0.036507</td>
          <td>22.106828</td>
          <td>0.005770</td>
          <td>21.410530</td>
          <td>0.005185</td>
          <td>23.184929</td>
          <td>0.011431</td>
          <td>26.012679</td>
          <td>0.240853</td>
          <td>19.895038</td>
          <td>0.005564</td>
          <td>26.279475</td>
          <td>24.000191</td>
          <td>25.405642</td>
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
          <td>23.312172</td>
          <td>24.033768</td>
          <td>23.175960</td>
          <td>21.078288</td>
          <td>26.533145</td>
          <td>24.279512</td>
          <td>21.984420</td>
          <td>0.005436</td>
          <td>21.329938</td>
          <td>0.005396</td>
          <td>16.544017</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.242852</td>
          <td>15.697660</td>
          <td>22.367128</td>
          <td>21.522410</td>
          <td>17.714344</td>
          <td>25.490646</td>
          <td>23.627844</td>
          <td>0.010886</td>
          <td>25.140890</td>
          <td>0.066217</td>
          <td>16.862832</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.091090</td>
          <td>22.872018</td>
          <td>22.334597</td>
          <td>23.474047</td>
          <td>22.483730</td>
          <td>23.685264</td>
          <td>23.668621</td>
          <td>0.011214</td>
          <td>19.107669</td>
          <td>0.005007</td>
          <td>25.815308</td>
          <td>0.119951</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759544</td>
          <td>23.195705</td>
          <td>24.305993</td>
          <td>21.918731</td>
          <td>25.482999</td>
          <td>18.193326</td>
          <td>26.521336</td>
          <td>0.131522</td>
          <td>19.356590</td>
          <td>0.005011</td>
          <td>28.184718</td>
          <td>0.771387</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.466462</td>
          <td>20.939735</td>
          <td>24.226951</td>
          <td>26.167589</td>
          <td>17.829389</td>
          <td>21.534681</td>
          <td>23.492049</td>
          <td>0.009893</td>
          <td>21.032689</td>
          <td>0.005233</td>
          <td>19.148856</td>
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
          <td>26.382362</td>
          <td>23.942146</td>
          <td>23.007739</td>
          <td>23.004419</td>
          <td>25.177428</td>
          <td>21.395948</td>
          <td>19.263563</td>
          <td>0.005003</td>
          <td>22.100466</td>
          <td>0.006481</td>
          <td>22.238373</td>
          <td>0.006849</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.703860</td>
          <td>18.186866</td>
          <td>22.144580</td>
          <td>21.333144</td>
          <td>21.247101</td>
          <td>25.445221</td>
          <td>22.058063</td>
          <td>0.005496</td>
          <td>13.454893</td>
          <td>0.005000</td>
          <td>18.683587</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.988151</td>
          <td>22.556139</td>
          <td>15.852135</td>
          <td>22.391501</td>
          <td>27.662741</td>
          <td>25.806251</td>
          <td>26.918857</td>
          <td>0.184896</td>
          <td>25.262701</td>
          <td>0.073779</td>
          <td>21.773749</td>
          <td>0.005858</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.008794</td>
          <td>20.354513</td>
          <td>23.537590</td>
          <td>25.474680</td>
          <td>21.206230</td>
          <td>31.117343</td>
          <td>28.542159</td>
          <td>0.654793</td>
          <td>20.998358</td>
          <td>0.005219</td>
          <td>22.089759</td>
          <td>0.006455</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.764395</td>
          <td>22.110564</td>
          <td>21.419048</td>
          <td>23.205773</td>
          <td>26.533064</td>
          <td>19.892330</td>
          <td>26.184145</td>
          <td>0.097991</td>
          <td>24.020256</td>
          <td>0.024492</td>
          <td>25.515241</td>
          <td>0.092228</td>
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
          <td>23.312172</td>
          <td>24.033768</td>
          <td>23.175960</td>
          <td>21.078288</td>
          <td>26.533145</td>
          <td>24.279512</td>
          <td>21.957303</td>
          <td>0.025298</td>
          <td>21.322361</td>
          <td>0.012593</td>
          <td>16.551182</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.242852</td>
          <td>15.697660</td>
          <td>22.367128</td>
          <td>21.522410</td>
          <td>17.714344</td>
          <td>25.490646</td>
          <td>23.661750</td>
          <td>0.114481</td>
          <td>24.871931</td>
          <td>0.269507</td>
          <td>16.864990</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.091090</td>
          <td>22.872018</td>
          <td>22.334597</td>
          <td>23.474047</td>
          <td>22.483730</td>
          <td>23.685264</td>
          <td>23.784742</td>
          <td>0.127414</td>
          <td>19.119882</td>
          <td>0.005227</td>
          <td>25.692875</td>
          <td>0.548926</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.759544</td>
          <td>23.195705</td>
          <td>24.305993</td>
          <td>21.918731</td>
          <td>25.482999</td>
          <td>18.193326</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.351486</td>
          <td>0.005344</td>
          <td>26.514301</td>
          <td>0.951439</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.466462</td>
          <td>20.939735</td>
          <td>24.226951</td>
          <td>26.167589</td>
          <td>17.829389</td>
          <td>21.534681</td>
          <td>23.635245</td>
          <td>0.111862</td>
          <td>21.020809</td>
          <td>0.010090</td>
          <td>19.133357</td>
          <td>0.005279</td>
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
          <td>26.382362</td>
          <td>23.942146</td>
          <td>23.007739</td>
          <td>23.004419</td>
          <td>25.177428</td>
          <td>21.395948</td>
          <td>19.263455</td>
          <td>0.005420</td>
          <td>22.173429</td>
          <td>0.025658</td>
          <td>22.203987</td>
          <td>0.028780</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.703860</td>
          <td>18.186866</td>
          <td>22.144580</td>
          <td>21.333144</td>
          <td>21.247101</td>
          <td>25.445221</td>
          <td>22.051435</td>
          <td>0.027478</td>
          <td>13.450885</td>
          <td>0.005000</td>
          <td>18.669474</td>
          <td>0.005120</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.988151</td>
          <td>22.556139</td>
          <td>15.852135</td>
          <td>22.391501</td>
          <td>27.662741</td>
          <td>25.806251</td>
          <td>26.794124</td>
          <td>1.188891</td>
          <td>25.122823</td>
          <td>0.329827</td>
          <td>21.759727</td>
          <td>0.019553</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.008794</td>
          <td>20.354513</td>
          <td>23.537590</td>
          <td>25.474680</td>
          <td>21.206230</td>
          <td>31.117343</td>
          <td>26.842148</td>
          <td>1.221085</td>
          <td>21.017304</td>
          <td>0.010066</td>
          <td>22.098602</td>
          <td>0.026231</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.764395</td>
          <td>22.110564</td>
          <td>21.419048</td>
          <td>23.205773</td>
          <td>26.533064</td>
          <td>19.892330</td>
          <td>26.539326</td>
          <td>1.026151</td>
          <td>23.957436</td>
          <td>0.124428</td>
          <td>25.875091</td>
          <td>0.624933</td>
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


