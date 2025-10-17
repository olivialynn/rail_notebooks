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
          <td>17.187081</td>
          <td>21.372303</td>
          <td>20.088055</td>
          <td>16.869162</td>
          <td>22.040326</td>
          <td>24.476453</td>
          <td>22.519422</td>
          <td>30.078806</td>
          <td>25.304716</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.749895</td>
          <td>19.949325</td>
          <td>16.467612</td>
          <td>18.538683</td>
          <td>26.117555</td>
          <td>22.781052</td>
          <td>23.483522</td>
          <td>20.873993</td>
          <td>23.487049</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.012945</td>
          <td>23.931437</td>
          <td>27.103586</td>
          <td>23.909349</td>
          <td>14.814159</td>
          <td>17.265538</td>
          <td>23.096325</td>
          <td>25.911785</td>
          <td>25.165212</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.408767</td>
          <td>21.922027</td>
          <td>21.954870</td>
          <td>26.520232</td>
          <td>17.412688</td>
          <td>24.035418</td>
          <td>22.345603</td>
          <td>17.318853</td>
          <td>25.087905</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.892050</td>
          <td>23.301868</td>
          <td>21.627544</td>
          <td>22.719422</td>
          <td>21.615870</td>
          <td>21.295917</td>
          <td>24.514869</td>
          <td>21.492174</td>
          <td>18.006787</td>
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
          <td>27.244991</td>
          <td>26.577452</td>
          <td>19.329303</td>
          <td>23.792181</td>
          <td>25.560809</td>
          <td>20.786265</td>
          <td>22.570749</td>
          <td>24.107056</td>
          <td>23.997081</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.098078</td>
          <td>26.725787</td>
          <td>20.356145</td>
          <td>23.906246</td>
          <td>20.550596</td>
          <td>25.680864</td>
          <td>24.258601</td>
          <td>18.583893</td>
          <td>21.478493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.775992</td>
          <td>23.025796</td>
          <td>25.341866</td>
          <td>22.266321</td>
          <td>20.116203</td>
          <td>22.836620</td>
          <td>24.280392</td>
          <td>19.640723</td>
          <td>20.757365</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.738601</td>
          <td>24.385330</td>
          <td>22.598742</td>
          <td>24.298623</td>
          <td>21.884342</td>
          <td>25.679570</td>
          <td>22.318269</td>
          <td>23.830689</td>
          <td>20.113630</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708821</td>
          <td>21.580445</td>
          <td>23.004474</td>
          <td>26.243589</td>
          <td>20.736445</td>
          <td>18.707741</td>
          <td>22.335161</td>
          <td>27.774015</td>
          <td>24.599260</td>
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
          <td>17.180408</td>
          <td>0.005012</td>
          <td>21.372498</td>
          <td>0.005251</td>
          <td>20.084629</td>
          <td>0.005026</td>
          <td>16.865520</td>
          <td>0.005001</td>
          <td>22.027084</td>
          <td>0.008630</td>
          <td>24.698912</td>
          <td>0.173049</td>
          <td>22.519422</td>
          <td>30.078806</td>
          <td>25.304716</td>
        </tr>
        <tr>
          <th>1</th>
          <td>inf</td>
          <td>inf</td>
          <td>19.948387</td>
          <td>0.005036</td>
          <td>16.473773</td>
          <td>0.005001</td>
          <td>18.533954</td>
          <td>0.005008</td>
          <td>26.385932</td>
          <td>0.325954</td>
          <td>22.813301</td>
          <td>0.033069</td>
          <td>23.483522</td>
          <td>20.873993</td>
          <td>23.487049</td>
        </tr>
        <tr>
          <th>2</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.927544</td>
          <td>0.014869</td>
          <td>26.918301</td>
          <td>0.175065</td>
          <td>23.909345</td>
          <td>0.020392</td>
          <td>14.799522</td>
          <td>0.005000</td>
          <td>17.267522</td>
          <td>0.005012</td>
          <td>23.096325</td>
          <td>25.911785</td>
          <td>25.165212</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.414786</td>
          <td>0.006925</td>
          <td>21.915181</td>
          <td>0.005573</td>
          <td>21.945840</td>
          <td>0.005436</td>
          <td>26.417134</td>
          <td>0.183019</td>
          <td>17.419468</td>
          <td>0.005005</td>
          <td>24.148540</td>
          <td>0.107642</td>
          <td>22.345603</td>
          <td>17.318853</td>
          <td>25.087905</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.045889</td>
          <td>0.112677</td>
          <td>23.294740</td>
          <td>0.009415</td>
          <td>21.635622</td>
          <td>0.005265</td>
          <td>22.725570</td>
          <td>0.008448</td>
          <td>21.598229</td>
          <td>0.006937</td>
          <td>21.286048</td>
          <td>0.009647</td>
          <td>24.514869</td>
          <td>21.492174</td>
          <td>18.006787</td>
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
          <td>26.650807</td>
          <td>0.423407</td>
          <td>26.418137</td>
          <td>0.129040</td>
          <td>19.332501</td>
          <td>0.005010</td>
          <td>23.796718</td>
          <td>0.018543</td>
          <td>25.486464</td>
          <td>0.154655</td>
          <td>20.775296</td>
          <td>0.007240</td>
          <td>22.570749</td>
          <td>24.107056</td>
          <td>23.997081</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.100117</td>
          <td>0.005011</td>
          <td>26.556298</td>
          <td>0.145366</td>
          <td>20.342099</td>
          <td>0.005037</td>
          <td>23.881275</td>
          <td>0.019912</td>
          <td>20.555250</td>
          <td>0.005371</td>
          <td>25.676383</td>
          <td>0.384305</td>
          <td>24.258601</td>
          <td>18.583893</td>
          <td>21.478493</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.777600</td>
          <td>0.015966</td>
          <td>23.028181</td>
          <td>0.008054</td>
          <td>25.374519</td>
          <td>0.045285</td>
          <td>22.270574</td>
          <td>0.006758</td>
          <td>20.114312</td>
          <td>0.005184</td>
          <td>22.854346</td>
          <td>0.034288</td>
          <td>24.280392</td>
          <td>19.640723</td>
          <td>20.757365</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.853723</td>
          <td>0.493069</td>
          <td>24.384083</td>
          <td>0.021675</td>
          <td>22.596604</td>
          <td>0.006236</td>
          <td>24.295041</td>
          <td>0.028459</td>
          <td>21.876949</td>
          <td>0.007929</td>
          <td>26.259556</td>
          <td>0.592926</td>
          <td>22.318269</td>
          <td>23.830689</td>
          <td>20.113630</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.701484</td>
          <td>0.007850</td>
          <td>21.570201</td>
          <td>0.005338</td>
          <td>23.005168</td>
          <td>0.007315</td>
          <td>26.306519</td>
          <td>0.166609</td>
          <td>20.736119</td>
          <td>0.005497</td>
          <td>18.722540</td>
          <td>0.005089</td>
          <td>22.335161</td>
          <td>27.774015</td>
          <td>24.599260</td>
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
          <td>17.187081</td>
          <td>21.372303</td>
          <td>20.088055</td>
          <td>16.869162</td>
          <td>22.040326</td>
          <td>24.476453</td>
          <td>22.520882</td>
          <td>0.006100</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.456361</td>
          <td>0.087564</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.749895</td>
          <td>19.949325</td>
          <td>16.467612</td>
          <td>18.538683</td>
          <td>26.117555</td>
          <td>22.781052</td>
          <td>23.488217</td>
          <td>0.009867</td>
          <td>20.879884</td>
          <td>0.005177</td>
          <td>23.447663</td>
          <td>0.015063</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.012945</td>
          <td>23.931437</td>
          <td>27.103586</td>
          <td>23.909349</td>
          <td>14.814159</td>
          <td>17.265538</td>
          <td>23.084845</td>
          <td>0.007712</td>
          <td>25.811912</td>
          <td>0.119597</td>
          <td>25.144297</td>
          <td>0.066418</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.408767</td>
          <td>21.922027</td>
          <td>21.954870</td>
          <td>26.520232</td>
          <td>17.412688</td>
          <td>24.035418</td>
          <td>22.344033</td>
          <td>0.005816</td>
          <td>17.318761</td>
          <td>0.005000</td>
          <td>25.008013</td>
          <td>0.058835</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.892050</td>
          <td>23.301868</td>
          <td>21.627544</td>
          <td>22.719422</td>
          <td>21.615870</td>
          <td>21.295917</td>
          <td>24.491252</td>
          <td>0.021894</td>
          <td>21.501179</td>
          <td>0.005535</td>
          <td>18.004092</td>
          <td>0.005001</td>
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
          <td>27.244991</td>
          <td>26.577452</td>
          <td>19.329303</td>
          <td>23.792181</td>
          <td>25.560809</td>
          <td>20.786265</td>
          <td>22.578724</td>
          <td>0.006212</td>
          <td>24.084294</td>
          <td>0.025904</td>
          <td>23.983748</td>
          <td>0.023724</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.098078</td>
          <td>26.725787</td>
          <td>20.356145</td>
          <td>23.906246</td>
          <td>20.550596</td>
          <td>25.680864</td>
          <td>24.272368</td>
          <td>0.018154</td>
          <td>18.583588</td>
          <td>0.005003</td>
          <td>21.471475</td>
          <td>0.005508</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.775992</td>
          <td>23.025796</td>
          <td>25.341866</td>
          <td>22.266321</td>
          <td>20.116203</td>
          <td>22.836620</td>
          <td>24.295916</td>
          <td>0.018520</td>
          <td>19.645195</td>
          <td>0.005018</td>
          <td>20.755993</td>
          <td>0.005141</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.738601</td>
          <td>24.385330</td>
          <td>22.598742</td>
          <td>24.298623</td>
          <td>21.884342</td>
          <td>25.679570</td>
          <td>22.327124</td>
          <td>0.005792</td>
          <td>23.873004</td>
          <td>0.021551</td>
          <td>20.112842</td>
          <td>0.005044</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708821</td>
          <td>21.580445</td>
          <td>23.004474</td>
          <td>26.243589</td>
          <td>20.736445</td>
          <td>18.707741</td>
          <td>22.340811</td>
          <td>0.005811</td>
          <td>27.386905</td>
          <td>0.437602</td>
          <td>24.692921</td>
          <td>0.044430</td>
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
          <td>17.187081</td>
          <td>21.372303</td>
          <td>20.088055</td>
          <td>16.869162</td>
          <td>22.040326</td>
          <td>24.476453</td>
          <td>22.570422</td>
          <td>0.043548</td>
          <td>26.197999</td>
          <td>0.728150</td>
          <td>24.879670</td>
          <td>0.294117</td>
        </tr>
        <tr>
          <th>1</th>
          <td>29.749895</td>
          <td>19.949325</td>
          <td>16.467612</td>
          <td>18.538683</td>
          <td>26.117555</td>
          <td>22.781052</td>
          <td>23.639657</td>
          <td>0.112294</td>
          <td>20.865316</td>
          <td>0.009096</td>
          <td>23.455849</td>
          <td>0.087524</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30.012945</td>
          <td>23.931437</td>
          <td>27.103586</td>
          <td>23.909349</td>
          <td>14.814159</td>
          <td>17.265538</td>
          <td>23.039431</td>
          <td>0.066132</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.450610</td>
          <td>0.459149</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.408767</td>
          <td>21.922027</td>
          <td>21.954870</td>
          <td>26.520232</td>
          <td>17.412688</td>
          <td>24.035418</td>
          <td>22.340768</td>
          <td>0.035495</td>
          <td>17.315461</td>
          <td>0.005008</td>
          <td>24.506061</td>
          <td>0.216408</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.892050</td>
          <td>23.301868</td>
          <td>21.627544</td>
          <td>22.719422</td>
          <td>21.615870</td>
          <td>21.295917</td>
          <td>24.754121</td>
          <td>0.288109</td>
          <td>21.471715</td>
          <td>0.014166</td>
          <td>18.005848</td>
          <td>0.005036</td>
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
          <td>27.244991</td>
          <td>26.577452</td>
          <td>19.329303</td>
          <td>23.792181</td>
          <td>25.560809</td>
          <td>20.786265</td>
          <td>22.600533</td>
          <td>0.044732</td>
          <td>24.119551</td>
          <td>0.143172</td>
          <td>23.991628</td>
          <td>0.139765</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.098078</td>
          <td>26.725787</td>
          <td>20.356145</td>
          <td>23.906246</td>
          <td>20.550596</td>
          <td>25.680864</td>
          <td>24.701876</td>
          <td>0.276158</td>
          <td>18.587264</td>
          <td>0.005086</td>
          <td>21.473809</td>
          <td>0.015388</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.775992</td>
          <td>23.025796</td>
          <td>25.341866</td>
          <td>22.266321</td>
          <td>20.116203</td>
          <td>22.836620</td>
          <td>24.311134</td>
          <td>0.199864</td>
          <td>19.643786</td>
          <td>0.005577</td>
          <td>20.764844</td>
          <td>0.009093</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.738601</td>
          <td>24.385330</td>
          <td>22.598742</td>
          <td>24.298623</td>
          <td>21.884342</td>
          <td>25.679570</td>
          <td>22.283074</td>
          <td>0.033723</td>
          <td>23.766503</td>
          <td>0.105333</td>
          <td>20.105867</td>
          <td>0.006494</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.708821</td>
          <td>21.580445</td>
          <td>23.004474</td>
          <td>26.243589</td>
          <td>20.736445</td>
          <td>18.707741</td>
          <td>22.333137</td>
          <td>0.035256</td>
          <td>28.783529</td>
          <td>2.636353</td>
          <td>24.586125</td>
          <td>0.231314</td>
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


