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
          <td>17.468857</td>
          <td>21.633918</td>
          <td>22.580739</td>
          <td>23.755262</td>
          <td>31.436270</td>
          <td>18.240851</td>
          <td>22.167478</td>
          <td>24.981389</td>
          <td>17.227493</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.955056</td>
          <td>22.286449</td>
          <td>21.420846</td>
          <td>24.534375</td>
          <td>21.382819</td>
          <td>24.197269</td>
          <td>24.724132</td>
          <td>22.165070</td>
          <td>20.880646</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.305213</td>
          <td>21.237085</td>
          <td>21.832577</td>
          <td>22.239741</td>
          <td>24.261959</td>
          <td>21.372624</td>
          <td>25.892586</td>
          <td>16.143722</td>
          <td>21.518461</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.025482</td>
          <td>22.593903</td>
          <td>19.626689</td>
          <td>19.595713</td>
          <td>27.740744</td>
          <td>21.462105</td>
          <td>24.427249</td>
          <td>23.722094</td>
          <td>25.711631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.660264</td>
          <td>25.326700</td>
          <td>21.188936</td>
          <td>24.161917</td>
          <td>28.809585</td>
          <td>21.272530</td>
          <td>22.269826</td>
          <td>25.923578</td>
          <td>19.360812</td>
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
          <td>24.858683</td>
          <td>24.087509</td>
          <td>26.272832</td>
          <td>26.718693</td>
          <td>21.180850</td>
          <td>26.154648</td>
          <td>25.150193</td>
          <td>24.451736</td>
          <td>24.238248</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.616338</td>
          <td>20.557255</td>
          <td>19.868136</td>
          <td>24.697927</td>
          <td>23.152629</td>
          <td>22.167321</td>
          <td>25.133225</td>
          <td>22.980546</td>
          <td>23.620978</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.066793</td>
          <td>23.965453</td>
          <td>19.349379</td>
          <td>24.337447</td>
          <td>23.209035</td>
          <td>27.180475</td>
          <td>20.541935</td>
          <td>30.459551</td>
          <td>27.852517</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.976956</td>
          <td>30.449671</td>
          <td>21.825688</td>
          <td>19.721759</td>
          <td>17.062542</td>
          <td>27.034394</td>
          <td>24.818700</td>
          <td>18.486926</td>
          <td>27.557462</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.053141</td>
          <td>25.757690</td>
          <td>24.503522</td>
          <td>26.583855</td>
          <td>18.206570</td>
          <td>19.795288</td>
          <td>21.997862</td>
          <td>20.486271</td>
          <td>17.489431</td>
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
          <td>17.472573</td>
          <td>0.005017</td>
          <td>21.634572</td>
          <td>0.005372</td>
          <td>22.574245</td>
          <td>0.006194</td>
          <td>23.779691</td>
          <td>0.018280</td>
          <td>27.782617</td>
          <td>0.888543</td>
          <td>18.246243</td>
          <td>0.005045</td>
          <td>22.167478</td>
          <td>24.981389</td>
          <td>17.227493</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.955169</td>
          <td>0.005261</td>
          <td>22.286382</td>
          <td>0.006015</td>
          <td>21.420096</td>
          <td>0.005188</td>
          <td>24.546564</td>
          <td>0.035510</td>
          <td>21.369834</td>
          <td>0.006363</td>
          <td>24.280285</td>
          <td>0.120734</td>
          <td>24.724132</td>
          <td>22.165070</td>
          <td>20.880646</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.515823</td>
          <td>0.168745</td>
          <td>21.234803</td>
          <td>0.005204</td>
          <td>21.830142</td>
          <td>0.005362</td>
          <td>22.237239</td>
          <td>0.006670</td>
          <td>24.326213</td>
          <td>0.055956</td>
          <td>21.364191</td>
          <td>0.010163</td>
          <td>25.892586</td>
          <td>16.143722</td>
          <td>21.518461</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.029406</td>
          <td>0.005085</td>
          <td>22.591606</td>
          <td>0.006613</td>
          <td>19.633525</td>
          <td>0.005015</td>
          <td>19.598271</td>
          <td>0.005029</td>
          <td>30.053462</td>
          <td>2.628491</td>
          <td>21.438602</td>
          <td>0.010699</td>
          <td>24.427249</td>
          <td>23.722094</td>
          <td>25.711631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.655914</td>
          <td>0.005020</td>
          <td>25.304444</td>
          <td>0.048462</td>
          <td>21.190337</td>
          <td>0.005131</td>
          <td>24.169833</td>
          <td>0.025514</td>
          <td>27.518618</td>
          <td>0.749001</td>
          <td>21.272917</td>
          <td>0.009564</td>
          <td>22.269826</td>
          <td>25.923578</td>
          <td>19.360812</td>
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
          <td>24.850646</td>
          <td>0.095058</td>
          <td>24.086768</td>
          <td>0.016906</td>
          <td>26.432072</td>
          <td>0.115194</td>
          <td>26.761542</td>
          <td>0.244045</td>
          <td>21.182198</td>
          <td>0.006015</td>
          <td>25.232700</td>
          <td>0.269989</td>
          <td>25.150193</td>
          <td>24.451736</td>
          <td>24.238248</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.596990</td>
          <td>0.013884</td>
          <td>20.551266</td>
          <td>0.005078</td>
          <td>19.865994</td>
          <td>0.005020</td>
          <td>24.730496</td>
          <td>0.041791</td>
          <td>23.150219</td>
          <td>0.019955</td>
          <td>22.162156</td>
          <td>0.018831</td>
          <td>25.133225</td>
          <td>22.980546</td>
          <td>23.620978</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.072127</td>
          <td>0.005030</td>
          <td>23.985126</td>
          <td>0.015569</td>
          <td>19.342693</td>
          <td>0.005010</td>
          <td>24.359937</td>
          <td>0.030126</td>
          <td>23.193934</td>
          <td>0.020709</td>
          <td>26.260279</td>
          <td>0.593230</td>
          <td>20.541935</td>
          <td>30.459551</td>
          <td>27.852517</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.974512</td>
          <td>0.005010</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.822420</td>
          <td>0.005357</td>
          <td>19.724213</td>
          <td>0.005034</td>
          <td>17.070382</td>
          <td>0.005004</td>
          <td>25.883209</td>
          <td>0.450145</td>
          <td>24.818700</td>
          <td>18.486926</td>
          <td>27.557462</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.984567</td>
          <td>0.106830</td>
          <td>25.756996</td>
          <td>0.072336</td>
          <td>24.531617</td>
          <td>0.021626</td>
          <td>26.525399</td>
          <td>0.200509</td>
          <td>18.200723</td>
          <td>0.005013</td>
          <td>19.793315</td>
          <td>0.005479</td>
          <td>21.997862</td>
          <td>20.486271</td>
          <td>17.489431</td>
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
          <td>17.468857</td>
          <td>21.633918</td>
          <td>22.580739</td>
          <td>23.755262</td>
          <td>31.436270</td>
          <td>18.240851</td>
          <td>22.170479</td>
          <td>0.005604</td>
          <td>24.983490</td>
          <td>0.057564</td>
          <td>17.233744</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.955056</td>
          <td>22.286449</td>
          <td>21.420846</td>
          <td>24.534375</td>
          <td>21.382819</td>
          <td>24.197269</td>
          <td>24.717249</td>
          <td>0.026664</td>
          <td>22.162884</td>
          <td>0.006638</td>
          <td>20.877928</td>
          <td>0.005176</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.305213</td>
          <td>21.237085</td>
          <td>21.832577</td>
          <td>22.239741</td>
          <td>24.261959</td>
          <td>21.372624</td>
          <td>25.890050</td>
          <td>0.075589</td>
          <td>16.136791</td>
          <td>0.005000</td>
          <td>21.511573</td>
          <td>0.005545</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.025482</td>
          <td>22.593903</td>
          <td>19.626689</td>
          <td>19.595713</td>
          <td>27.740744</td>
          <td>21.462105</td>
          <td>24.453831</td>
          <td>0.021197</td>
          <td>23.710879</td>
          <td>0.018756</td>
          <td>26.058512</td>
          <td>0.148057</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.660264</td>
          <td>25.326700</td>
          <td>21.188936</td>
          <td>24.161917</td>
          <td>28.809585</td>
          <td>21.272530</td>
          <td>22.270184</td>
          <td>0.005718</td>
          <td>25.965550</td>
          <td>0.136652</td>
          <td>19.360990</td>
          <td>0.005011</td>
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
          <td>24.858683</td>
          <td>24.087509</td>
          <td>26.272832</td>
          <td>26.718693</td>
          <td>21.180850</td>
          <td>26.154648</td>
          <td>25.127892</td>
          <td>0.038355</td>
          <td>24.449092</td>
          <td>0.035759</td>
          <td>24.250071</td>
          <td>0.029975</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.616338</td>
          <td>20.557255</td>
          <td>19.868136</td>
          <td>24.697927</td>
          <td>23.152629</td>
          <td>22.167321</td>
          <td>25.092215</td>
          <td>0.037157</td>
          <td>22.971550</td>
          <td>0.010455</td>
          <td>23.629841</td>
          <td>0.017515</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.066793</td>
          <td>23.965453</td>
          <td>19.349379</td>
          <td>24.337447</td>
          <td>23.209035</td>
          <td>27.180475</td>
          <td>20.537318</td>
          <td>0.005032</td>
          <td>30.148310</td>
          <td>2.160494</td>
          <td>28.675811</td>
          <td>1.048602</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.976956</td>
          <td>30.449671</td>
          <td>21.825688</td>
          <td>19.721759</td>
          <td>17.062542</td>
          <td>27.034394</td>
          <td>24.797681</td>
          <td>0.028620</td>
          <td>18.480330</td>
          <td>0.005002</td>
          <td>27.008497</td>
          <td>0.326093</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.053141</td>
          <td>25.757690</td>
          <td>24.503522</td>
          <td>26.583855</td>
          <td>18.206570</td>
          <td>19.795288</td>
          <td>21.996557</td>
          <td>0.005445</td>
          <td>20.487098</td>
          <td>0.005086</td>
          <td>17.480486</td>
          <td>0.005000</td>
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
          <td>17.468857</td>
          <td>21.633918</td>
          <td>22.580739</td>
          <td>23.755262</td>
          <td>31.436270</td>
          <td>18.240851</td>
          <td>22.199174</td>
          <td>0.031306</td>
          <td>24.711261</td>
          <td>0.236181</td>
          <td>17.226945</td>
          <td>0.005009</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.955056</td>
          <td>22.286449</td>
          <td>21.420846</td>
          <td>24.534375</td>
          <td>21.382819</td>
          <td>24.197269</td>
          <td>24.680163</td>
          <td>0.271321</td>
          <td>22.156738</td>
          <td>0.025286</td>
          <td>20.864229</td>
          <td>0.009707</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.305213</td>
          <td>21.237085</td>
          <td>21.832577</td>
          <td>22.239741</td>
          <td>24.261959</td>
          <td>21.372624</td>
          <td>25.793914</td>
          <td>0.633210</td>
          <td>16.147114</td>
          <td>0.005001</td>
          <td>21.505903</td>
          <td>0.015799</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.025482</td>
          <td>22.593903</td>
          <td>19.626689</td>
          <td>19.595713</td>
          <td>27.740744</td>
          <td>21.462105</td>
          <td>24.433668</td>
          <td>0.221447</td>
          <td>23.916008</td>
          <td>0.120025</td>
          <td>25.591016</td>
          <td>0.509642</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.660264</td>
          <td>25.326700</td>
          <td>21.188936</td>
          <td>24.161917</td>
          <td>28.809585</td>
          <td>21.272530</td>
          <td>22.311139</td>
          <td>0.034573</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.361301</td>
          <td>0.005418</td>
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
          <td>24.858683</td>
          <td>24.087509</td>
          <td>26.272832</td>
          <td>26.718693</td>
          <td>21.180850</td>
          <td>26.154648</td>
          <td>24.908017</td>
          <td>0.325968</td>
          <td>24.500505</td>
          <td>0.198085</td>
          <td>24.066982</td>
          <td>0.149139</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.616338</td>
          <td>20.557255</td>
          <td>19.868136</td>
          <td>24.697927</td>
          <td>23.152629</td>
          <td>22.167321</td>
          <td>26.227964</td>
          <td>0.846597</td>
          <td>22.933723</td>
          <td>0.050372</td>
          <td>23.468815</td>
          <td>0.088531</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.066793</td>
          <td>23.965453</td>
          <td>19.349379</td>
          <td>24.337447</td>
          <td>23.209035</td>
          <td>27.180475</td>
          <td>20.541061</td>
          <td>0.008423</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.622909</td>
          <td>0.521699</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.976956</td>
          <td>30.449671</td>
          <td>21.825688</td>
          <td>19.721759</td>
          <td>17.062542</td>
          <td>27.034394</td>
          <td>25.036288</td>
          <td>0.360713</td>
          <td>18.482401</td>
          <td>0.005071</td>
          <td>26.929047</td>
          <td>1.212256</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.053141</td>
          <td>25.757690</td>
          <td>24.503522</td>
          <td>26.583855</td>
          <td>18.206570</td>
          <td>19.795288</td>
          <td>21.988529</td>
          <td>0.026000</td>
          <td>20.491829</td>
          <td>0.007352</td>
          <td>17.489334</td>
          <td>0.005014</td>
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


