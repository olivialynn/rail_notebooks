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
          <td>24.392490</td>
          <td>21.613012</td>
          <td>20.560863</td>
          <td>27.190799</td>
          <td>21.772280</td>
          <td>30.539720</td>
          <td>19.990878</td>
          <td>26.068633</td>
          <td>26.918827</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.745548</td>
          <td>20.463015</td>
          <td>27.023260</td>
          <td>25.416461</td>
          <td>26.182869</td>
          <td>24.302694</td>
          <td>22.289837</td>
          <td>25.038026</td>
          <td>26.463556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.109820</td>
          <td>24.184139</td>
          <td>22.569994</td>
          <td>20.768359</td>
          <td>23.995138</td>
          <td>23.757046</td>
          <td>17.517610</td>
          <td>20.575069</td>
          <td>21.955996</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.089624</td>
          <td>23.413742</td>
          <td>21.412845</td>
          <td>29.447011</td>
          <td>26.063401</td>
          <td>21.015426</td>
          <td>27.925834</td>
          <td>20.754557</td>
          <td>21.219030</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.128935</td>
          <td>22.926052</td>
          <td>23.215326</td>
          <td>24.643012</td>
          <td>19.123105</td>
          <td>21.267024</td>
          <td>26.215293</td>
          <td>21.499342</td>
          <td>22.273603</td>
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
          <td>22.716140</td>
          <td>23.252076</td>
          <td>26.347139</td>
          <td>26.448424</td>
          <td>21.115907</td>
          <td>24.460000</td>
          <td>19.412810</td>
          <td>19.801440</td>
          <td>21.473138</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.906899</td>
          <td>20.171698</td>
          <td>19.289413</td>
          <td>24.770808</td>
          <td>21.514817</td>
          <td>20.883484</td>
          <td>21.192633</td>
          <td>21.187110</td>
          <td>30.944210</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.748672</td>
          <td>18.558847</td>
          <td>17.230123</td>
          <td>18.236109</td>
          <td>21.866923</td>
          <td>24.413599</td>
          <td>17.097523</td>
          <td>26.544800</td>
          <td>25.206659</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.742941</td>
          <td>24.366385</td>
          <td>23.096849</td>
          <td>22.740311</td>
          <td>22.985133</td>
          <td>24.847017</td>
          <td>23.293377</td>
          <td>21.128420</td>
          <td>20.881884</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.415805</td>
          <td>19.920985</td>
          <td>22.517882</td>
          <td>18.328422</td>
          <td>27.493713</td>
          <td>16.386478</td>
          <td>19.438950</td>
          <td>22.624027</td>
          <td>22.696908</td>
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
          <td>24.359825</td>
          <td>0.061792</td>
          <td>21.604137</td>
          <td>0.005356</td>
          <td>20.572361</td>
          <td>0.005052</td>
          <td>27.401632</td>
          <td>0.406612</td>
          <td>21.785289</td>
          <td>0.007562</td>
          <td>27.101055</td>
          <td>1.027697</td>
          <td>19.990878</td>
          <td>26.068633</td>
          <td>26.918827</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.742054</td>
          <td>0.005199</td>
          <td>20.463420</td>
          <td>0.005069</td>
          <td>26.755234</td>
          <td>0.152322</td>
          <td>25.464812</td>
          <td>0.080135</td>
          <td>26.187596</td>
          <td>0.277931</td>
          <td>24.158078</td>
          <td>0.108543</td>
          <td>22.289837</td>
          <td>25.038026</td>
          <td>26.463556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.640361</td>
          <td>0.420049</td>
          <td>24.203739</td>
          <td>0.018620</td>
          <td>22.571443</td>
          <td>0.006188</td>
          <td>20.770464</td>
          <td>0.005161</td>
          <td>23.928891</td>
          <td>0.039332</td>
          <td>23.731030</td>
          <td>0.074570</td>
          <td>17.517610</td>
          <td>20.575069</td>
          <td>21.955996</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.086146</td>
          <td>0.005091</td>
          <td>23.417854</td>
          <td>0.010204</td>
          <td>21.421555</td>
          <td>0.005188</td>
          <td>27.694142</td>
          <td>0.506667</td>
          <td>25.846683</td>
          <td>0.209825</td>
          <td>21.011715</td>
          <td>0.008167</td>
          <td>27.925834</td>
          <td>20.754557</td>
          <td>21.219030</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.145657</td>
          <td>0.021549</td>
          <td>22.915934</td>
          <td>0.007601</td>
          <td>23.215587</td>
          <td>0.008151</td>
          <td>24.658568</td>
          <td>0.039210</td>
          <td>19.117373</td>
          <td>0.005042</td>
          <td>21.256869</td>
          <td>0.009465</td>
          <td>26.215293</td>
          <td>21.499342</td>
          <td>22.273603</td>
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
          <td>22.722259</td>
          <td>0.015288</td>
          <td>23.257922</td>
          <td>0.009201</td>
          <td>26.196555</td>
          <td>0.093747</td>
          <td>26.706363</td>
          <td>0.233172</td>
          <td>21.114703</td>
          <td>0.005911</td>
          <td>24.450802</td>
          <td>0.139934</td>
          <td>19.412810</td>
          <td>19.801440</td>
          <td>21.473138</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.910142</td>
          <td>0.017751</td>
          <td>20.172961</td>
          <td>0.005047</td>
          <td>19.293753</td>
          <td>0.005010</td>
          <td>24.769558</td>
          <td>0.043264</td>
          <td>21.518097</td>
          <td>0.006714</td>
          <td>20.883840</td>
          <td>0.007631</td>
          <td>21.192633</td>
          <td>21.187110</td>
          <td>30.944210</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.737511</td>
          <td>0.007992</td>
          <td>18.552511</td>
          <td>0.005007</td>
          <td>17.228237</td>
          <td>0.005001</td>
          <td>18.239944</td>
          <td>0.005005</td>
          <td>21.858669</td>
          <td>0.007852</td>
          <td>24.431974</td>
          <td>0.137681</td>
          <td>17.097523</td>
          <td>26.544800</td>
          <td>25.206659</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.343792</td>
          <td>0.698214</td>
          <td>24.381902</td>
          <td>0.021635</td>
          <td>23.100993</td>
          <td>0.007668</td>
          <td>22.743195</td>
          <td>0.008535</td>
          <td>22.996298</td>
          <td>0.017538</td>
          <td>25.010551</td>
          <td>0.224879</td>
          <td>23.293377</td>
          <td>21.128420</td>
          <td>20.881884</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.417196</td>
          <td>0.005134</td>
          <td>19.910656</td>
          <td>0.005034</td>
          <td>22.511726</td>
          <td>0.006081</td>
          <td>18.329408</td>
          <td>0.005006</td>
          <td>27.323496</td>
          <td>0.656160</td>
          <td>16.391570</td>
          <td>0.005005</td>
          <td>19.438950</td>
          <td>22.624027</td>
          <td>22.696908</td>
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
          <td>24.392490</td>
          <td>21.613012</td>
          <td>20.560863</td>
          <td>27.190799</td>
          <td>21.772280</td>
          <td>30.539720</td>
          <td>19.988718</td>
          <td>0.005012</td>
          <td>26.050047</td>
          <td>0.146982</td>
          <td>26.748221</td>
          <td>0.264340</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.745548</td>
          <td>20.463015</td>
          <td>27.023260</td>
          <td>25.416461</td>
          <td>26.182869</td>
          <td>24.302694</td>
          <td>22.298237</td>
          <td>0.005754</td>
          <td>25.088085</td>
          <td>0.063181</td>
          <td>26.357323</td>
          <td>0.191006</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.109820</td>
          <td>24.184139</td>
          <td>22.569994</td>
          <td>20.768359</td>
          <td>23.995138</td>
          <td>23.757046</td>
          <td>17.513082</td>
          <td>0.005000</td>
          <td>20.568852</td>
          <td>0.005100</td>
          <td>21.967999</td>
          <td>0.006190</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.089624</td>
          <td>23.413742</td>
          <td>21.412845</td>
          <td>29.447011</td>
          <td>26.063401</td>
          <td>21.015426</td>
          <td>27.728907</td>
          <td>0.358632</td>
          <td>20.762431</td>
          <td>0.005143</td>
          <td>21.223529</td>
          <td>0.005328</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.128935</td>
          <td>22.926052</td>
          <td>23.215326</td>
          <td>24.643012</td>
          <td>19.123105</td>
          <td>21.267024</td>
          <td>26.046673</td>
          <td>0.086819</td>
          <td>21.509744</td>
          <td>0.005543</td>
          <td>22.271817</td>
          <td>0.006950</td>
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
          <td>22.716140</td>
          <td>23.252076</td>
          <td>26.347139</td>
          <td>26.448424</td>
          <td>21.115907</td>
          <td>24.460000</td>
          <td>19.409348</td>
          <td>0.005004</td>
          <td>19.800938</td>
          <td>0.005025</td>
          <td>21.471721</td>
          <td>0.005508</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.906899</td>
          <td>20.171698</td>
          <td>19.289413</td>
          <td>24.770808</td>
          <td>21.514817</td>
          <td>20.883484</td>
          <td>21.188046</td>
          <td>0.005104</td>
          <td>21.192457</td>
          <td>0.005310</td>
          <td>28.090375</td>
          <td>0.724431</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.748672</td>
          <td>18.558847</td>
          <td>17.230123</td>
          <td>18.236109</td>
          <td>21.866923</td>
          <td>24.413599</td>
          <td>17.102780</td>
          <td>0.005000</td>
          <td>26.877725</td>
          <td>0.293656</td>
          <td>25.167286</td>
          <td>0.067789</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.742941</td>
          <td>24.366385</td>
          <td>23.096849</td>
          <td>22.740311</td>
          <td>22.985133</td>
          <td>24.847017</td>
          <td>23.286402</td>
          <td>0.008657</td>
          <td>21.123596</td>
          <td>0.005274</td>
          <td>20.887358</td>
          <td>0.005179</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.415805</td>
          <td>19.920985</td>
          <td>22.517882</td>
          <td>18.328422</td>
          <td>27.493713</td>
          <td>16.386478</td>
          <td>19.441773</td>
          <td>0.005004</td>
          <td>22.629800</td>
          <td>0.008367</td>
          <td>22.695317</td>
          <td>0.008704</td>
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
          <td>24.392490</td>
          <td>21.613012</td>
          <td>20.560863</td>
          <td>27.190799</td>
          <td>21.772280</td>
          <td>30.539720</td>
          <td>19.994845</td>
          <td>0.006467</td>
          <td>25.943566</td>
          <td>0.611250</td>
          <td>25.707240</td>
          <td>0.554649</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.745548</td>
          <td>20.463015</td>
          <td>27.023260</td>
          <td>25.416461</td>
          <td>26.182869</td>
          <td>24.302694</td>
          <td>22.305101</td>
          <td>0.034388</td>
          <td>24.850561</td>
          <td>0.264846</td>
          <td>25.601431</td>
          <td>0.513555</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.109820</td>
          <td>24.184139</td>
          <td>22.569994</td>
          <td>20.768359</td>
          <td>23.995138</td>
          <td>23.757046</td>
          <td>17.517772</td>
          <td>0.005018</td>
          <td>20.579538</td>
          <td>0.007690</td>
          <td>21.951670</td>
          <td>0.023071</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.089624</td>
          <td>23.413742</td>
          <td>21.412845</td>
          <td>29.447011</td>
          <td>26.063401</td>
          <td>21.015426</td>
          <td>26.953876</td>
          <td>1.297772</td>
          <td>20.738066</td>
          <td>0.008408</td>
          <td>21.220391</td>
          <td>0.012574</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.128935</td>
          <td>22.926052</td>
          <td>23.215326</td>
          <td>24.643012</td>
          <td>19.123105</td>
          <td>21.267024</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.488416</td>
          <td>0.014357</td>
          <td>22.252869</td>
          <td>0.030049</td>
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
          <td>22.716140</td>
          <td>23.252076</td>
          <td>26.347139</td>
          <td>26.448424</td>
          <td>21.115907</td>
          <td>24.460000</td>
          <td>19.409575</td>
          <td>0.005543</td>
          <td>19.802414</td>
          <td>0.005759</td>
          <td>21.454036</td>
          <td>0.015141</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.906899</td>
          <td>20.171698</td>
          <td>19.289413</td>
          <td>24.770808</td>
          <td>21.514817</td>
          <td>20.883484</td>
          <td>21.195835</td>
          <td>0.013336</td>
          <td>21.181221</td>
          <td>0.011319</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.748672</td>
          <td>18.558847</td>
          <td>17.230123</td>
          <td>18.236109</td>
          <td>21.866923</td>
          <td>24.413599</td>
          <td>17.102548</td>
          <td>0.005008</td>
          <td>25.874016</td>
          <td>0.581854</td>
          <td>24.897289</td>
          <td>0.298323</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.742941</td>
          <td>24.366385</td>
          <td>23.096849</td>
          <td>22.740311</td>
          <td>22.985133</td>
          <td>24.847017</td>
          <td>23.254300</td>
          <td>0.080014</td>
          <td>21.140761</td>
          <td>0.010988</td>
          <td>20.878763</td>
          <td>0.009803</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.415805</td>
          <td>19.920985</td>
          <td>22.517882</td>
          <td>18.328422</td>
          <td>27.493713</td>
          <td>16.386478</td>
          <td>19.442480</td>
          <td>0.005575</td>
          <td>22.545365</td>
          <td>0.035641</td>
          <td>22.686408</td>
          <td>0.044173</td>
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


