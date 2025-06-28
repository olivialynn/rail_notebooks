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
          <td>23.496009</td>
          <td>22.411799</td>
          <td>22.576798</td>
          <td>19.736382</td>
          <td>20.674284</td>
          <td>25.847297</td>
          <td>25.725537</td>
          <td>24.708401</td>
          <td>26.000352</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.968367</td>
          <td>28.366019</td>
          <td>26.333230</td>
          <td>25.887547</td>
          <td>26.212815</td>
          <td>24.833187</td>
          <td>25.395636</td>
          <td>22.926513</td>
          <td>21.117620</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.572865</td>
          <td>23.287931</td>
          <td>20.755393</td>
          <td>18.614098</td>
          <td>22.371151</td>
          <td>20.206273</td>
          <td>24.408162</td>
          <td>21.136419</td>
          <td>26.684074</td>
        </tr>
        <tr>
          <th>3</th>
          <td>13.911739</td>
          <td>26.790876</td>
          <td>22.412121</td>
          <td>21.620345</td>
          <td>15.188332</td>
          <td>27.121461</td>
          <td>20.607696</td>
          <td>21.710717</td>
          <td>24.801559</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.272075</td>
          <td>21.305338</td>
          <td>27.087670</td>
          <td>22.014624</td>
          <td>23.843197</td>
          <td>22.504886</td>
          <td>23.204662</td>
          <td>24.473833</td>
          <td>24.534977</td>
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
          <td>24.131537</td>
          <td>25.134279</td>
          <td>23.683193</td>
          <td>15.153306</td>
          <td>18.166292</td>
          <td>24.840559</td>
          <td>26.587401</td>
          <td>25.497551</td>
          <td>22.739733</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.057906</td>
          <td>19.648913</td>
          <td>20.309685</td>
          <td>23.622129</td>
          <td>23.920381</td>
          <td>26.105178</td>
          <td>25.295436</td>
          <td>21.755304</td>
          <td>19.359754</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.097058</td>
          <td>22.080542</td>
          <td>20.353394</td>
          <td>22.633032</td>
          <td>18.953648</td>
          <td>21.527304</td>
          <td>19.032905</td>
          <td>21.439785</td>
          <td>22.784357</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.275686</td>
          <td>23.225726</td>
          <td>22.941204</td>
          <td>24.133092</td>
          <td>19.269339</td>
          <td>20.991979</td>
          <td>22.886735</td>
          <td>26.770668</td>
          <td>22.631953</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.411594</td>
          <td>26.896699</td>
          <td>23.524691</td>
          <td>23.986656</td>
          <td>22.793423</td>
          <td>24.719673</td>
          <td>21.218828</td>
          <td>22.902595</td>
          <td>22.772250</td>
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
          <td>23.439647</td>
          <td>0.027650</td>
          <td>22.413478</td>
          <td>0.006232</td>
          <td>22.580387</td>
          <td>0.006205</td>
          <td>19.736640</td>
          <td>0.005035</td>
          <td>20.681355</td>
          <td>0.005455</td>
          <td>25.968007</td>
          <td>0.479661</td>
          <td>25.725537</td>
          <td>24.708401</td>
          <td>26.000352</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.970344</td>
          <td>0.105516</td>
          <td>27.768754</td>
          <td>0.393498</td>
          <td>26.123055</td>
          <td>0.087881</td>
          <td>25.815910</td>
          <td>0.109075</td>
          <td>26.011363</td>
          <td>0.240592</td>
          <td>25.099332</td>
          <td>0.242029</td>
          <td>25.395636</td>
          <td>22.926513</td>
          <td>21.117620</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.690278</td>
          <td>0.082611</td>
          <td>23.281680</td>
          <td>0.009338</td>
          <td>20.750083</td>
          <td>0.005067</td>
          <td>18.618171</td>
          <td>0.005008</td>
          <td>22.376741</td>
          <td>0.010841</td>
          <td>20.209935</td>
          <td>0.005934</td>
          <td>24.408162</td>
          <td>21.136419</td>
          <td>26.684074</td>
        </tr>
        <tr>
          <th>3</th>
          <td>13.911167</td>
          <td>0.005001</td>
          <td>26.712882</td>
          <td>0.166207</td>
          <td>22.413950</td>
          <td>0.005926</td>
          <td>21.615278</td>
          <td>0.005623</td>
          <td>15.199085</td>
          <td>0.005001</td>
          <td>26.335607</td>
          <td>0.625570</td>
          <td>20.607696</td>
          <td>21.710717</td>
          <td>24.801559</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.032691</td>
          <td>0.259693</td>
          <td>21.315342</td>
          <td>0.005230</td>
          <td>27.386115</td>
          <td>0.258697</td>
          <td>22.007840</td>
          <td>0.006168</td>
          <td>23.753658</td>
          <td>0.033688</td>
          <td>22.520408</td>
          <td>0.025588</td>
          <td>23.204662</td>
          <td>24.473833</td>
          <td>24.534977</td>
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
          <td>24.090226</td>
          <td>0.048743</td>
          <td>25.102250</td>
          <td>0.040524</td>
          <td>23.676035</td>
          <td>0.010932</td>
          <td>15.142912</td>
          <td>0.005000</td>
          <td>18.172059</td>
          <td>0.005012</td>
          <td>24.970512</td>
          <td>0.217510</td>
          <td>26.587401</td>
          <td>25.497551</td>
          <td>22.739733</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>19.657699</td>
          <td>0.005025</td>
          <td>20.301083</td>
          <td>0.005035</td>
          <td>23.611397</td>
          <td>0.015908</td>
          <td>23.890288</td>
          <td>0.038011</td>
          <td>26.117773</td>
          <td>0.535522</td>
          <td>25.295436</td>
          <td>21.755304</td>
          <td>19.359754</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.108306</td>
          <td>0.009869</td>
          <td>22.078572</td>
          <td>0.005737</td>
          <td>20.358663</td>
          <td>0.005038</td>
          <td>22.638831</td>
          <td>0.008045</td>
          <td>18.956009</td>
          <td>0.005033</td>
          <td>21.521538</td>
          <td>0.011351</td>
          <td>19.032905</td>
          <td>21.439785</td>
          <td>22.784357</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.272200</td>
          <td>0.010989</td>
          <td>23.220038</td>
          <td>0.008989</td>
          <td>22.951442</td>
          <td>0.007136</td>
          <td>24.117070</td>
          <td>0.024373</td>
          <td>19.269499</td>
          <td>0.005052</td>
          <td>20.988611</td>
          <td>0.008064</td>
          <td>22.886735</td>
          <td>26.770668</td>
          <td>22.631953</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.271581</td>
          <td>0.057174</td>
          <td>26.593347</td>
          <td>0.150064</td>
          <td>23.535376</td>
          <td>0.009920</td>
          <td>23.970685</td>
          <td>0.021487</td>
          <td>22.781216</td>
          <td>0.014719</td>
          <td>24.554384</td>
          <td>0.152965</td>
          <td>21.218828</td>
          <td>22.902595</td>
          <td>22.772250</td>
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
          <td>23.496009</td>
          <td>22.411799</td>
          <td>22.576798</td>
          <td>19.736382</td>
          <td>20.674284</td>
          <td>25.847297</td>
          <td>25.859528</td>
          <td>0.073572</td>
          <td>24.684124</td>
          <td>0.044083</td>
          <td>26.094573</td>
          <td>0.152716</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.968367</td>
          <td>28.366019</td>
          <td>26.333230</td>
          <td>25.887547</td>
          <td>26.212815</td>
          <td>24.833187</td>
          <td>25.407400</td>
          <td>0.049204</td>
          <td>22.916169</td>
          <td>0.010058</td>
          <td>21.119433</td>
          <td>0.005272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.572865</td>
          <td>23.287931</td>
          <td>20.755393</td>
          <td>18.614098</td>
          <td>22.371151</td>
          <td>20.206273</td>
          <td>24.394961</td>
          <td>0.020151</td>
          <td>21.134682</td>
          <td>0.005279</td>
          <td>26.626970</td>
          <td>0.239269</td>
        </tr>
        <tr>
          <th>3</th>
          <td>13.911739</td>
          <td>26.790876</td>
          <td>22.412121</td>
          <td>21.620345</td>
          <td>15.188332</td>
          <td>27.121461</td>
          <td>20.610101</td>
          <td>0.005036</td>
          <td>21.712489</td>
          <td>0.005773</td>
          <td>24.889023</td>
          <td>0.052918</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.272075</td>
          <td>21.305338</td>
          <td>27.087670</td>
          <td>22.014624</td>
          <td>23.843197</td>
          <td>22.504886</td>
          <td>23.213052</td>
          <td>0.008285</td>
          <td>24.413128</td>
          <td>0.034635</td>
          <td>24.557776</td>
          <td>0.039389</td>
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
          <td>24.131537</td>
          <td>25.134279</td>
          <td>23.683193</td>
          <td>15.153306</td>
          <td>18.166292</td>
          <td>24.840559</td>
          <td>26.317243</td>
          <td>0.110116</td>
          <td>25.593715</td>
          <td>0.098818</td>
          <td>22.727371</td>
          <td>0.008879</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.057906</td>
          <td>19.648913</td>
          <td>20.309685</td>
          <td>23.622129</td>
          <td>23.920381</td>
          <td>26.105178</td>
          <td>25.192399</td>
          <td>0.040623</td>
          <td>21.757819</td>
          <td>0.005835</td>
          <td>19.357798</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.097058</td>
          <td>22.080542</td>
          <td>20.353394</td>
          <td>22.633032</td>
          <td>18.953648</td>
          <td>21.527304</td>
          <td>19.031154</td>
          <td>0.005002</td>
          <td>21.434911</td>
          <td>0.005477</td>
          <td>22.791325</td>
          <td>0.009250</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.275686</td>
          <td>23.225726</td>
          <td>22.941204</td>
          <td>24.133092</td>
          <td>19.269339</td>
          <td>20.991979</td>
          <td>22.880352</td>
          <td>0.006976</td>
          <td>26.831834</td>
          <td>0.282957</td>
          <td>22.640620</td>
          <td>0.008421</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.411594</td>
          <td>26.896699</td>
          <td>23.524691</td>
          <td>23.986656</td>
          <td>22.793423</td>
          <td>24.719673</td>
          <td>21.222016</td>
          <td>0.005110</td>
          <td>22.907577</td>
          <td>0.009999</td>
          <td>22.778221</td>
          <td>0.009172</td>
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
          <td>23.496009</td>
          <td>22.411799</td>
          <td>22.576798</td>
          <td>19.736382</td>
          <td>20.674284</td>
          <td>25.847297</td>
          <td>26.993857</td>
          <td>1.325809</td>
          <td>24.707698</td>
          <td>0.235486</td>
          <td>25.974606</td>
          <td>0.669608</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.968367</td>
          <td>28.366019</td>
          <td>26.333230</td>
          <td>25.887547</td>
          <td>26.212815</td>
          <td>24.833187</td>
          <td>26.427958</td>
          <td>0.959432</td>
          <td>22.859151</td>
          <td>0.047132</td>
          <td>21.123833</td>
          <td>0.011684</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.572865</td>
          <td>23.287931</td>
          <td>20.755393</td>
          <td>18.614098</td>
          <td>22.371151</td>
          <td>20.206273</td>
          <td>24.794013</td>
          <td>0.297537</td>
          <td>21.132541</td>
          <td>0.010923</td>
          <td>25.630249</td>
          <td>0.524505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>13.911739</td>
          <td>26.790876</td>
          <td>22.412121</td>
          <td>21.620345</td>
          <td>15.188332</td>
          <td>27.121461</td>
          <td>20.603917</td>
          <td>0.008751</td>
          <td>21.721962</td>
          <td>0.017400</td>
          <td>25.010003</td>
          <td>0.326484</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.272075</td>
          <td>21.305338</td>
          <td>27.087670</td>
          <td>22.014624</td>
          <td>23.843197</td>
          <td>22.504886</td>
          <td>23.205126</td>
          <td>0.076606</td>
          <td>24.338405</td>
          <td>0.172693</td>
          <td>24.506870</td>
          <td>0.216554</td>
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
          <td>24.131537</td>
          <td>25.134279</td>
          <td>23.683193</td>
          <td>15.153306</td>
          <td>18.166292</td>
          <td>24.840559</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.237388</td>
          <td>0.361024</td>
          <td>22.710026</td>
          <td>0.045112</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.057906</td>
          <td>19.648913</td>
          <td>20.309685</td>
          <td>23.622129</td>
          <td>23.920381</td>
          <td>26.105178</td>
          <td>24.999971</td>
          <td>0.350571</td>
          <td>21.752558</td>
          <td>0.017853</td>
          <td>19.350484</td>
          <td>0.005410</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.097058</td>
          <td>22.080542</td>
          <td>20.353394</td>
          <td>22.633032</td>
          <td>18.953648</td>
          <td>21.527304</td>
          <td>19.033444</td>
          <td>0.005279</td>
          <td>21.447807</td>
          <td>0.013897</td>
          <td>22.750428</td>
          <td>0.046767</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.275686</td>
          <td>23.225726</td>
          <td>22.941204</td>
          <td>24.133092</td>
          <td>19.269339</td>
          <td>20.991979</td>
          <td>22.931836</td>
          <td>0.060096</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.648436</td>
          <td>0.042703</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.411594</td>
          <td>26.896699</td>
          <td>23.524691</td>
          <td>23.986656</td>
          <td>22.793423</td>
          <td>24.719673</td>
          <td>21.206937</td>
          <td>0.013454</td>
          <td>22.893000</td>
          <td>0.048576</td>
          <td>22.739010</td>
          <td>0.046293</td>
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


