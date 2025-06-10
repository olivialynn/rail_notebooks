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
          <td>26.027650</td>
          <td>18.512423</td>
          <td>23.474003</td>
          <td>27.485217</td>
          <td>26.250425</td>
          <td>23.115404</td>
          <td>26.524588</td>
          <td>23.652369</td>
          <td>22.222369</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.799174</td>
          <td>25.746819</td>
          <td>20.603155</td>
          <td>19.122311</td>
          <td>28.534630</td>
          <td>20.414113</td>
          <td>18.672812</td>
          <td>19.294216</td>
          <td>23.818453</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.583409</td>
          <td>21.419209</td>
          <td>29.355647</td>
          <td>23.613353</td>
          <td>19.149610</td>
          <td>23.588517</td>
          <td>26.243635</td>
          <td>29.733927</td>
          <td>24.097550</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.488375</td>
          <td>26.791786</td>
          <td>23.608960</td>
          <td>16.455398</td>
          <td>23.716869</td>
          <td>20.819882</td>
          <td>18.985065</td>
          <td>23.561421</td>
          <td>30.319769</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.246167</td>
          <td>26.413562</td>
          <td>21.511985</td>
          <td>24.020122</td>
          <td>21.398487</td>
          <td>25.574973</td>
          <td>22.125923</td>
          <td>25.996866</td>
          <td>27.747161</td>
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
          <td>18.439798</td>
          <td>23.348599</td>
          <td>19.958228</td>
          <td>19.795914</td>
          <td>20.629200</td>
          <td>22.544269</td>
          <td>27.322550</td>
          <td>22.166998</td>
          <td>27.442275</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.435105</td>
          <td>24.043065</td>
          <td>19.385724</td>
          <td>23.686373</td>
          <td>22.550468</td>
          <td>25.121949</td>
          <td>18.965820</td>
          <td>20.924806</td>
          <td>26.964854</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.434191</td>
          <td>22.353402</td>
          <td>18.602685</td>
          <td>20.571149</td>
          <td>25.837981</td>
          <td>23.423075</td>
          <td>23.720564</td>
          <td>25.519654</td>
          <td>24.914591</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.267148</td>
          <td>26.257122</td>
          <td>22.895287</td>
          <td>21.555854</td>
          <td>26.497467</td>
          <td>20.161459</td>
          <td>18.461812</td>
          <td>23.775301</td>
          <td>21.549593</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.609634</td>
          <td>22.689313</td>
          <td>21.518382</td>
          <td>14.648085</td>
          <td>19.211369</td>
          <td>19.466439</td>
          <td>25.014591</td>
          <td>27.376445</td>
          <td>18.743943</td>
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
          <td>25.927332</td>
          <td>0.238179</td>
          <td>18.511885</td>
          <td>0.005007</td>
          <td>23.471887</td>
          <td>0.009514</td>
          <td>27.147004</td>
          <td>0.333331</td>
          <td>26.145878</td>
          <td>0.268657</td>
          <td>23.111816</td>
          <td>0.043062</td>
          <td>26.524588</td>
          <td>23.652369</td>
          <td>22.222369</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.886078</td>
          <td>0.230200</td>
          <td>25.648094</td>
          <td>0.065698</td>
          <td>20.609701</td>
          <td>0.005055</td>
          <td>19.129599</td>
          <td>0.005015</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.412632</td>
          <td>0.006286</td>
          <td>18.672812</td>
          <td>19.294216</td>
          <td>23.818453</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.589417</td>
          <td>0.005610</td>
          <td>21.419303</td>
          <td>0.005269</td>
          <td>28.320335</td>
          <td>0.534299</td>
          <td>23.634946</td>
          <td>0.016217</td>
          <td>19.146722</td>
          <td>0.005043</td>
          <td>23.559493</td>
          <td>0.064065</td>
          <td>26.243635</td>
          <td>29.733927</td>
          <td>24.097550</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.341871</td>
          <td>0.145469</td>
          <td>26.761124</td>
          <td>0.173168</td>
          <td>23.632160</td>
          <td>0.010599</td>
          <td>16.453607</td>
          <td>0.005001</td>
          <td>23.700049</td>
          <td>0.032134</td>
          <td>20.822168</td>
          <td>0.007402</td>
          <td>18.985065</td>
          <td>23.561421</td>
          <td>30.319769</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.242658</td>
          <td>0.006515</td>
          <td>26.500940</td>
          <td>0.138604</td>
          <td>21.507852</td>
          <td>0.005216</td>
          <td>24.019661</td>
          <td>0.022408</td>
          <td>21.394931</td>
          <td>0.006417</td>
          <td>24.963790</td>
          <td>0.216294</td>
          <td>22.125923</td>
          <td>25.996866</td>
          <td>27.747161</td>
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
          <td>18.447147</td>
          <td>0.005045</td>
          <td>23.332524</td>
          <td>0.009646</td>
          <td>19.966856</td>
          <td>0.005023</td>
          <td>19.794772</td>
          <td>0.005037</td>
          <td>20.624403</td>
          <td>0.005415</td>
          <td>22.533353</td>
          <td>0.025877</td>
          <td>27.322550</td>
          <td>22.166998</td>
          <td>27.442275</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.442133</td>
          <td>0.005498</td>
          <td>24.065735</td>
          <td>0.016618</td>
          <td>19.392374</td>
          <td>0.005011</td>
          <td>23.707586</td>
          <td>0.017216</td>
          <td>22.538939</td>
          <td>0.012200</td>
          <td>25.548074</td>
          <td>0.347639</td>
          <td>18.965820</td>
          <td>20.924806</td>
          <td>26.964854</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.408604</td>
          <td>0.026924</td>
          <td>22.352908</td>
          <td>0.006124</td>
          <td>18.598384</td>
          <td>0.005004</td>
          <td>20.573708</td>
          <td>0.005118</td>
          <td>25.811463</td>
          <td>0.203726</td>
          <td>23.384619</td>
          <td>0.054859</td>
          <td>23.720564</td>
          <td>25.519654</td>
          <td>24.914591</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.271279</td>
          <td>0.010982</td>
          <td>26.095139</td>
          <td>0.097401</td>
          <td>22.888508</td>
          <td>0.006942</td>
          <td>21.543443</td>
          <td>0.005555</td>
          <td>26.815058</td>
          <td>0.454437</td>
          <td>20.165641</td>
          <td>0.005870</td>
          <td>18.461812</td>
          <td>23.775301</td>
          <td>21.549593</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.608416</td>
          <td>0.005007</td>
          <td>22.686576</td>
          <td>0.006859</td>
          <td>21.522827</td>
          <td>0.005221</td>
          <td>14.653792</td>
          <td>0.005000</td>
          <td>19.207655</td>
          <td>0.005047</td>
          <td>19.466444</td>
          <td>0.005284</td>
          <td>25.014591</td>
          <td>27.376445</td>
          <td>18.743943</td>
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
          <td>26.027650</td>
          <td>18.512423</td>
          <td>23.474003</td>
          <td>27.485217</td>
          <td>26.250425</td>
          <td>23.115404</td>
          <td>26.337295</td>
          <td>0.112062</td>
          <td>23.621125</td>
          <td>0.017388</td>
          <td>22.214847</td>
          <td>0.006781</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.799174</td>
          <td>25.746819</td>
          <td>20.603155</td>
          <td>19.122311</td>
          <td>28.534630</td>
          <td>20.414113</td>
          <td>18.680010</td>
          <td>0.005001</td>
          <td>19.299455</td>
          <td>0.005010</td>
          <td>23.809484</td>
          <td>0.020404</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.583409</td>
          <td>21.419209</td>
          <td>29.355647</td>
          <td>23.613353</td>
          <td>19.149610</td>
          <td>23.588517</td>
          <td>26.269440</td>
          <td>0.105604</td>
          <td>28.838516</td>
          <td>1.152205</td>
          <td>24.101053</td>
          <td>0.026288</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.488375</td>
          <td>26.791786</td>
          <td>23.608960</td>
          <td>16.455398</td>
          <td>23.716869</td>
          <td>20.819882</td>
          <td>18.990330</td>
          <td>0.005002</td>
          <td>23.534143</td>
          <td>0.016172</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.246167</td>
          <td>26.413562</td>
          <td>21.511985</td>
          <td>24.020122</td>
          <td>21.398487</td>
          <td>25.574973</td>
          <td>22.129543</td>
          <td>0.005563</td>
          <td>26.239502</td>
          <td>0.172855</td>
          <td>28.768524</td>
          <td>1.106947</td>
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
          <td>18.439798</td>
          <td>23.348599</td>
          <td>19.958228</td>
          <td>19.795914</td>
          <td>20.629200</td>
          <td>22.544269</td>
          <td>27.312294</td>
          <td>0.256679</td>
          <td>22.171201</td>
          <td>0.006660</td>
          <td>27.639333</td>
          <td>0.527995</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.435105</td>
          <td>24.043065</td>
          <td>19.385724</td>
          <td>23.686373</td>
          <td>22.550468</td>
          <td>25.121949</td>
          <td>18.952123</td>
          <td>0.005002</td>
          <td>20.922271</td>
          <td>0.005191</td>
          <td>27.079976</td>
          <td>0.345091</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.434191</td>
          <td>22.353402</td>
          <td>18.602685</td>
          <td>20.571149</td>
          <td>25.837981</td>
          <td>23.423075</td>
          <td>23.711709</td>
          <td>0.011578</td>
          <td>25.477396</td>
          <td>0.089203</td>
          <td>24.967893</td>
          <td>0.056770</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.267148</td>
          <td>26.257122</td>
          <td>22.895287</td>
          <td>21.555854</td>
          <td>26.497467</td>
          <td>20.161459</td>
          <td>18.463834</td>
          <td>0.005001</td>
          <td>23.730454</td>
          <td>0.019071</td>
          <td>21.541237</td>
          <td>0.005574</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.609634</td>
          <td>22.689313</td>
          <td>21.518382</td>
          <td>14.648085</td>
          <td>19.211369</td>
          <td>19.466439</td>
          <td>25.029597</td>
          <td>0.035145</td>
          <td>28.301787</td>
          <td>0.832501</td>
          <td>18.737451</td>
          <td>0.005003</td>
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
          <td>26.027650</td>
          <td>18.512423</td>
          <td>23.474003</td>
          <td>27.485217</td>
          <td>26.250425</td>
          <td>23.115404</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.471575</td>
          <td>0.081246</td>
          <td>22.205235</td>
          <td>0.028812</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.799174</td>
          <td>25.746819</td>
          <td>20.603155</td>
          <td>19.122311</td>
          <td>28.534630</td>
          <td>20.414113</td>
          <td>18.672827</td>
          <td>0.005145</td>
          <td>19.280633</td>
          <td>0.005303</td>
          <td>23.683620</td>
          <td>0.106924</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.583409</td>
          <td>21.419209</td>
          <td>29.355647</td>
          <td>23.613353</td>
          <td>19.149610</td>
          <td>23.588517</td>
          <td>25.444479</td>
          <td>0.492442</td>
          <td>26.403064</td>
          <td>0.833185</td>
          <td>24.028794</td>
          <td>0.144317</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.488375</td>
          <td>26.791786</td>
          <td>23.608960</td>
          <td>16.455398</td>
          <td>23.716869</td>
          <td>20.819882</td>
          <td>18.984937</td>
          <td>0.005256</td>
          <td>23.468618</td>
          <td>0.081034</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.246167</td>
          <td>26.413562</td>
          <td>21.511985</td>
          <td>24.020122</td>
          <td>21.398487</td>
          <td>25.574973</td>
          <td>22.121469</td>
          <td>0.029227</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.229571</td>
          <td>0.387901</td>
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
          <td>18.439798</td>
          <td>23.348599</td>
          <td>19.958228</td>
          <td>19.795914</td>
          <td>20.629200</td>
          <td>22.544269</td>
          <td>26.486534</td>
          <td>0.994182</td>
          <td>22.202385</td>
          <td>0.026318</td>
          <td>29.980753</td>
          <td>3.864669</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.435105</td>
          <td>24.043065</td>
          <td>19.385724</td>
          <td>23.686373</td>
          <td>22.550468</td>
          <td>25.121949</td>
          <td>18.971080</td>
          <td>0.005249</td>
          <td>20.914752</td>
          <td>0.009393</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.434191</td>
          <td>22.353402</td>
          <td>18.602685</td>
          <td>20.571149</td>
          <td>25.837981</td>
          <td>23.423075</td>
          <td>23.762794</td>
          <td>0.125009</td>
          <td>26.045109</td>
          <td>0.656130</td>
          <td>24.331318</td>
          <td>0.186856</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.267148</td>
          <td>26.257122</td>
          <td>22.895287</td>
          <td>21.555854</td>
          <td>26.497467</td>
          <td>20.161459</td>
          <td>18.469280</td>
          <td>0.005100</td>
          <td>23.775584</td>
          <td>0.106174</td>
          <td>21.530797</td>
          <td>0.016127</td>
        </tr>
        <tr>
          <th>999</th>
          <td>16.609634</td>
          <td>22.689313</td>
          <td>21.518382</td>
          <td>14.648085</td>
          <td>19.211369</td>
          <td>19.466439</td>
          <td>24.756586</td>
          <td>0.288684</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.742464</td>
          <td>0.005138</td>
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


