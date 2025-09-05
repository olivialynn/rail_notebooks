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
          <td>23.229952</td>
          <td>22.293321</td>
          <td>20.880760</td>
          <td>20.282063</td>
          <td>19.893109</td>
          <td>26.477166</td>
          <td>17.170484</td>
          <td>23.954747</td>
          <td>22.242156</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.086882</td>
          <td>20.979072</td>
          <td>23.029571</td>
          <td>30.516456</td>
          <td>23.700311</td>
          <td>22.799908</td>
          <td>27.092565</td>
          <td>19.562870</td>
          <td>24.776366</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.932517</td>
          <td>20.041837</td>
          <td>23.975842</td>
          <td>24.056091</td>
          <td>25.171198</td>
          <td>23.431775</td>
          <td>25.691103</td>
          <td>18.489245</td>
          <td>21.313593</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.873353</td>
          <td>22.998659</td>
          <td>26.133910</td>
          <td>22.412382</td>
          <td>17.149949</td>
          <td>22.499218</td>
          <td>24.703339</td>
          <td>22.874951</td>
          <td>19.838969</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.441660</td>
          <td>25.294474</td>
          <td>28.498835</td>
          <td>22.429355</td>
          <td>22.063526</td>
          <td>25.487064</td>
          <td>22.152308</td>
          <td>25.835075</td>
          <td>23.148409</td>
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
          <td>23.918389</td>
          <td>25.434774</td>
          <td>27.160022</td>
          <td>22.924646</td>
          <td>22.480792</td>
          <td>19.203373</td>
          <td>19.309107</td>
          <td>21.387795</td>
          <td>23.236506</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.185088</td>
          <td>20.953809</td>
          <td>23.412233</td>
          <td>21.549330</td>
          <td>24.484951</td>
          <td>21.968976</td>
          <td>23.717297</td>
          <td>26.766496</td>
          <td>17.922478</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786492</td>
          <td>25.558316</td>
          <td>25.337319</td>
          <td>16.925068</td>
          <td>23.413426</td>
          <td>21.263594</td>
          <td>23.970600</td>
          <td>20.402312</td>
          <td>22.330254</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.716316</td>
          <td>26.414680</td>
          <td>22.220178</td>
          <td>21.765307</td>
          <td>24.995670</td>
          <td>27.836794</td>
          <td>20.908319</td>
          <td>21.340097</td>
          <td>17.963588</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.607455</td>
          <td>15.920178</td>
          <td>19.787114</td>
          <td>19.249244</td>
          <td>25.869390</td>
          <td>23.156100</td>
          <td>19.359015</td>
          <td>18.096531</td>
          <td>19.399728</td>
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
          <td>23.245548</td>
          <td>0.023436</td>
          <td>22.289533</td>
          <td>0.006020</td>
          <td>20.875833</td>
          <td>0.005081</td>
          <td>20.283709</td>
          <td>0.005076</td>
          <td>19.883944</td>
          <td>0.005129</td>
          <td>26.351269</td>
          <td>0.632455</td>
          <td>17.170484</td>
          <td>23.954747</td>
          <td>22.242156</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.081927</td>
          <td>0.006210</td>
          <td>20.981205</td>
          <td>0.005141</td>
          <td>23.031319</td>
          <td>0.007407</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.664510</td>
          <td>0.031145</td>
          <td>22.828363</td>
          <td>0.033511</td>
          <td>27.092565</td>
          <td>19.562870</td>
          <td>24.776366</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.932531</td>
          <td>0.008877</td>
          <td>20.037803</td>
          <td>0.005040</td>
          <td>23.977852</td>
          <td>0.013703</td>
          <td>24.104239</td>
          <td>0.024104</td>
          <td>25.205292</td>
          <td>0.121339</td>
          <td>23.425152</td>
          <td>0.056868</td>
          <td>25.691103</td>
          <td>18.489245</td>
          <td>21.313593</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.881104</td>
          <td>0.017340</td>
          <td>22.997730</td>
          <td>0.007925</td>
          <td>26.167002</td>
          <td>0.091344</td>
          <td>22.414255</td>
          <td>0.007187</td>
          <td>17.147416</td>
          <td>0.005004</td>
          <td>22.516236</td>
          <td>0.025495</td>
          <td>24.703339</td>
          <td>22.874951</td>
          <td>19.838969</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.453175</td>
          <td>0.012480</td>
          <td>25.304477</td>
          <td>0.048464</td>
          <td>27.305098</td>
          <td>0.242035</td>
          <td>22.430825</td>
          <td>0.007242</td>
          <td>22.058632</td>
          <td>0.008794</td>
          <td>24.878673</td>
          <td>0.201426</td>
          <td>22.152308</td>
          <td>25.835075</td>
          <td>23.148409</td>
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
          <td>23.917459</td>
          <td>0.041883</td>
          <td>25.449915</td>
          <td>0.055127</td>
          <td>27.070820</td>
          <td>0.199139</td>
          <td>22.919466</td>
          <td>0.009518</td>
          <td>22.479795</td>
          <td>0.011677</td>
          <td>19.205889</td>
          <td>0.005188</td>
          <td>19.309107</td>
          <td>21.387795</td>
          <td>23.236506</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.187781</td>
          <td>0.010386</td>
          <td>20.947752</td>
          <td>0.005135</td>
          <td>23.387806</td>
          <td>0.009020</td>
          <td>21.547622</td>
          <td>0.005559</td>
          <td>24.476593</td>
          <td>0.063941</td>
          <td>21.950543</td>
          <td>0.015809</td>
          <td>23.717297</td>
          <td>26.766496</td>
          <td>17.922478</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.784944</td>
          <td>0.008189</td>
          <td>25.619016</td>
          <td>0.064030</td>
          <td>25.402021</td>
          <td>0.046405</td>
          <td>16.934640</td>
          <td>0.005001</td>
          <td>23.433443</td>
          <td>0.025448</td>
          <td>21.292776</td>
          <td>0.009689</td>
          <td>23.970600</td>
          <td>20.402312</td>
          <td>22.330254</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.712271</td>
          <td>0.007892</td>
          <td>26.483679</td>
          <td>0.136557</td>
          <td>22.226016</td>
          <td>0.005685</td>
          <td>21.768643</td>
          <td>0.005798</td>
          <td>24.962535</td>
          <td>0.098169</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.908319</td>
          <td>21.340097</td>
          <td>17.963588</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.607914</td>
          <td>0.007511</td>
          <td>15.925251</td>
          <td>0.005001</td>
          <td>19.797738</td>
          <td>0.005018</td>
          <td>19.244995</td>
          <td>0.005018</td>
          <td>25.476724</td>
          <td>0.153370</td>
          <td>23.174678</td>
          <td>0.045532</td>
          <td>19.359015</td>
          <td>18.096531</td>
          <td>19.399728</td>
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
          <td>23.229952</td>
          <td>22.293321</td>
          <td>20.880760</td>
          <td>20.282063</td>
          <td>19.893109</td>
          <td>26.477166</td>
          <td>17.175955</td>
          <td>0.005000</td>
          <td>23.918568</td>
          <td>0.022418</td>
          <td>22.227124</td>
          <td>0.006816</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.086882</td>
          <td>20.979072</td>
          <td>23.029571</td>
          <td>30.516456</td>
          <td>23.700311</td>
          <td>22.799908</td>
          <td>26.916980</td>
          <td>0.184603</td>
          <td>19.556616</td>
          <td>0.005016</td>
          <td>24.717017</td>
          <td>0.045395</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.932517</td>
          <td>20.041837</td>
          <td>23.975842</td>
          <td>24.056091</td>
          <td>25.171198</td>
          <td>23.431775</td>
          <td>25.678780</td>
          <td>0.062660</td>
          <td>18.491568</td>
          <td>0.005002</td>
          <td>21.311633</td>
          <td>0.005383</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.873353</td>
          <td>22.998659</td>
          <td>26.133910</td>
          <td>22.412382</td>
          <td>17.149949</td>
          <td>22.499218</td>
          <td>24.674836</td>
          <td>0.025690</td>
          <td>22.864006</td>
          <td>0.009706</td>
          <td>19.834600</td>
          <td>0.005026</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.441660</td>
          <td>25.294474</td>
          <td>28.498835</td>
          <td>22.429355</td>
          <td>22.063526</td>
          <td>25.487064</td>
          <td>22.142318</td>
          <td>0.005575</td>
          <td>25.746192</td>
          <td>0.112936</td>
          <td>23.166562</td>
          <td>0.012066</td>
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
          <td>23.918389</td>
          <td>25.434774</td>
          <td>27.160022</td>
          <td>22.924646</td>
          <td>22.480792</td>
          <td>19.203373</td>
          <td>19.300926</td>
          <td>0.005003</td>
          <td>21.392648</td>
          <td>0.005442</td>
          <td>23.213580</td>
          <td>0.012508</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.185088</td>
          <td>20.953809</td>
          <td>23.412233</td>
          <td>21.549330</td>
          <td>24.484951</td>
          <td>21.968976</td>
          <td>23.691185</td>
          <td>0.011403</td>
          <td>26.438659</td>
          <td>0.204539</td>
          <td>17.924786</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786492</td>
          <td>25.558316</td>
          <td>25.337319</td>
          <td>16.925068</td>
          <td>23.413426</td>
          <td>21.263594</td>
          <td>23.957645</td>
          <td>0.014007</td>
          <td>20.401600</td>
          <td>0.005074</td>
          <td>22.326188</td>
          <td>0.007124</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.716316</td>
          <td>26.414680</td>
          <td>22.220178</td>
          <td>21.765307</td>
          <td>24.995670</td>
          <td>27.836794</td>
          <td>20.912011</td>
          <td>0.005063</td>
          <td>21.337156</td>
          <td>0.005401</td>
          <td>17.964758</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.607455</td>
          <td>15.920178</td>
          <td>19.787114</td>
          <td>19.249244</td>
          <td>25.869390</td>
          <td>23.156100</td>
          <td>19.357109</td>
          <td>0.005004</td>
          <td>18.096563</td>
          <td>0.005001</td>
          <td>19.399906</td>
          <td>0.005012</td>
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
          <td>23.229952</td>
          <td>22.293321</td>
          <td>20.880760</td>
          <td>20.282063</td>
          <td>19.893109</td>
          <td>26.477166</td>
          <td>17.172435</td>
          <td>0.005009</td>
          <td>23.840296</td>
          <td>0.112356</td>
          <td>22.271019</td>
          <td>0.030535</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.086882</td>
          <td>20.979072</td>
          <td>23.029571</td>
          <td>30.516456</td>
          <td>23.700311</td>
          <td>22.799908</td>
          <td>25.678287</td>
          <td>0.583628</td>
          <td>19.560242</td>
          <td>0.005498</td>
          <td>24.612034</td>
          <td>0.236332</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.932517</td>
          <td>20.041837</td>
          <td>23.975842</td>
          <td>24.056091</td>
          <td>25.171198</td>
          <td>23.431775</td>
          <td>26.174755</td>
          <td>0.818110</td>
          <td>18.489694</td>
          <td>0.005072</td>
          <td>21.346545</td>
          <td>0.013883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.873353</td>
          <td>22.998659</td>
          <td>26.133910</td>
          <td>22.412382</td>
          <td>17.149949</td>
          <td>22.499218</td>
          <td>24.090511</td>
          <td>0.165787</td>
          <td>22.776190</td>
          <td>0.043772</td>
          <td>19.828540</td>
          <td>0.005942</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.441660</td>
          <td>25.294474</td>
          <td>28.498835</td>
          <td>22.429355</td>
          <td>22.063526</td>
          <td>25.487064</td>
          <td>22.183471</td>
          <td>0.030874</td>
          <td>25.729702</td>
          <td>0.524296</td>
          <td>23.061811</td>
          <td>0.061721</td>
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
          <td>23.918389</td>
          <td>25.434774</td>
          <td>27.160022</td>
          <td>22.924646</td>
          <td>22.480792</td>
          <td>19.203373</td>
          <td>19.321066</td>
          <td>0.005465</td>
          <td>21.361365</td>
          <td>0.012981</td>
          <td>23.291057</td>
          <td>0.075657</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.185088</td>
          <td>20.953809</td>
          <td>23.412233</td>
          <td>21.549330</td>
          <td>24.484951</td>
          <td>21.968976</td>
          <td>23.639238</td>
          <td>0.112252</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.929518</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.786492</td>
          <td>25.558316</td>
          <td>25.337319</td>
          <td>16.925068</td>
          <td>23.413426</td>
          <td>21.263594</td>
          <td>24.054291</td>
          <td>0.160734</td>
          <td>20.400715</td>
          <td>0.007041</td>
          <td>22.362743</td>
          <td>0.033120</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.716316</td>
          <td>26.414680</td>
          <td>22.220178</td>
          <td>21.765307</td>
          <td>24.995670</td>
          <td>27.836794</td>
          <td>20.881605</td>
          <td>0.010530</td>
          <td>21.323755</td>
          <td>0.012607</td>
          <td>17.965838</td>
          <td>0.005033</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.607455</td>
          <td>15.920178</td>
          <td>19.787114</td>
          <td>19.249244</td>
          <td>25.869390</td>
          <td>23.156100</td>
          <td>19.364409</td>
          <td>0.005502</td>
          <td>18.088047</td>
          <td>0.005035</td>
          <td>19.396447</td>
          <td>0.005445</td>
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


