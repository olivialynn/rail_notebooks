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
          <td>29.872757</td>
          <td>29.514717</td>
          <td>22.006534</td>
          <td>23.832455</td>
          <td>21.116290</td>
          <td>21.920919</td>
          <td>17.592406</td>
          <td>21.729849</td>
          <td>30.977695</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.917720</td>
          <td>22.758461</td>
          <td>18.034445</td>
          <td>22.034108</td>
          <td>24.477586</td>
          <td>22.436871</td>
          <td>23.225129</td>
          <td>20.854677</td>
          <td>20.937871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.198994</td>
          <td>20.790335</td>
          <td>23.884056</td>
          <td>25.271273</td>
          <td>23.988082</td>
          <td>24.418967</td>
          <td>26.858737</td>
          <td>20.466145</td>
          <td>23.504723</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.478169</td>
          <td>23.222825</td>
          <td>22.248945</td>
          <td>24.176434</td>
          <td>24.287418</td>
          <td>26.009369</td>
          <td>20.608162</td>
          <td>17.293061</td>
          <td>21.865049</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.202028</td>
          <td>31.888368</td>
          <td>21.021721</td>
          <td>25.762218</td>
          <td>20.047908</td>
          <td>18.315391</td>
          <td>25.065600</td>
          <td>27.291584</td>
          <td>25.568164</td>
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
          <td>24.179715</td>
          <td>25.582359</td>
          <td>20.984558</td>
          <td>23.083639</td>
          <td>18.885631</td>
          <td>22.222352</td>
          <td>15.908268</td>
          <td>22.808453</td>
          <td>23.258166</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.462884</td>
          <td>21.207371</td>
          <td>22.662072</td>
          <td>19.417803</td>
          <td>24.903312</td>
          <td>19.207609</td>
          <td>19.567583</td>
          <td>24.222751</td>
          <td>19.659550</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.345136</td>
          <td>20.949138</td>
          <td>23.111157</td>
          <td>29.213346</td>
          <td>25.864161</td>
          <td>18.747325</td>
          <td>25.548954</td>
          <td>19.581755</td>
          <td>24.187096</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.232152</td>
          <td>22.507495</td>
          <td>25.361153</td>
          <td>21.273335</td>
          <td>23.748147</td>
          <td>23.445438</td>
          <td>27.084669</td>
          <td>24.510659</td>
          <td>18.761255</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.344503</td>
          <td>23.351990</td>
          <td>19.456487</td>
          <td>25.004445</td>
          <td>27.020651</td>
          <td>20.917093</td>
          <td>24.797083</td>
          <td>26.951291</td>
          <td>21.482582</td>
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
          <td>26.676350</td>
          <td>0.431709</td>
          <td>28.318776</td>
          <td>0.591790</td>
          <td>22.005177</td>
          <td>0.005480</td>
          <td>23.839739</td>
          <td>0.019226</td>
          <td>21.122679</td>
          <td>0.005923</td>
          <td>21.918230</td>
          <td>0.015401</td>
          <td>17.592406</td>
          <td>21.729849</td>
          <td>30.977695</td>
        </tr>
        <tr>
          <th>1</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.749783</td>
          <td>0.007042</td>
          <td>18.037849</td>
          <td>0.005002</td>
          <td>22.027513</td>
          <td>0.006205</td>
          <td>24.439607</td>
          <td>0.061878</td>
          <td>22.470924</td>
          <td>0.024513</td>
          <td>23.225129</td>
          <td>20.854677</td>
          <td>20.937871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.124259</td>
          <td>0.279777</td>
          <td>20.789149</td>
          <td>0.005108</td>
          <td>23.875306</td>
          <td>0.012660</td>
          <td>25.360847</td>
          <td>0.073102</td>
          <td>23.972199</td>
          <td>0.040871</td>
          <td>24.381546</td>
          <td>0.131812</td>
          <td>26.858737</td>
          <td>20.466145</td>
          <td>23.504723</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.919711</td>
          <td>0.517593</td>
          <td>23.222572</td>
          <td>0.009003</td>
          <td>22.247373</td>
          <td>0.005709</td>
          <td>24.210059</td>
          <td>0.026423</td>
          <td>24.298140</td>
          <td>0.054579</td>
          <td>28.038820</td>
          <td>1.689396</td>
          <td>20.608162</td>
          <td>17.293061</td>
          <td>21.865049</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.199089</td>
          <td>0.005005</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.023202</td>
          <td>0.005101</td>
          <td>25.858162</td>
          <td>0.113171</td>
          <td>20.043365</td>
          <td>0.005165</td>
          <td>18.319770</td>
          <td>0.005050</td>
          <td>25.065600</td>
          <td>27.291584</td>
          <td>25.568164</td>
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
          <td>24.204728</td>
          <td>0.053908</td>
          <td>25.587084</td>
          <td>0.062246</td>
          <td>20.992352</td>
          <td>0.005097</td>
          <td>23.084788</td>
          <td>0.010642</td>
          <td>18.881780</td>
          <td>0.005030</td>
          <td>22.245781</td>
          <td>0.020208</td>
          <td>15.908268</td>
          <td>22.808453</td>
          <td>23.258166</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.472440</td>
          <td>0.012656</td>
          <td>21.205243</td>
          <td>0.005196</td>
          <td>22.666788</td>
          <td>0.006380</td>
          <td>19.414123</td>
          <td>0.005022</td>
          <td>24.850261</td>
          <td>0.088951</td>
          <td>19.207039</td>
          <td>0.005188</td>
          <td>19.567583</td>
          <td>24.222751</td>
          <td>19.659550</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.385880</td>
          <td>0.063225</td>
          <td>20.946493</td>
          <td>0.005134</td>
          <td>23.100437</td>
          <td>0.007665</td>
          <td>27.869721</td>
          <td>0.575479</td>
          <td>25.841335</td>
          <td>0.208888</td>
          <td>18.751179</td>
          <td>0.005093</td>
          <td>25.548954</td>
          <td>19.581755</td>
          <td>24.187096</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.879678</td>
          <td>2.534237</td>
          <td>22.515699</td>
          <td>0.006439</td>
          <td>25.340025</td>
          <td>0.043920</td>
          <td>21.270323</td>
          <td>0.005357</td>
          <td>23.758121</td>
          <td>0.033821</td>
          <td>23.385139</td>
          <td>0.054884</td>
          <td>27.084669</td>
          <td>24.510659</td>
          <td>18.761255</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.337329</td>
          <td>0.011492</td>
          <td>23.339932</td>
          <td>0.009692</td>
          <td>19.460738</td>
          <td>0.005012</td>
          <td>25.089261</td>
          <td>0.057464</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.909224</td>
          <td>0.007731</td>
          <td>24.797083</td>
          <td>26.951291</td>
          <td>21.482582</td>
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
          <td>29.872757</td>
          <td>29.514717</td>
          <td>22.006534</td>
          <td>23.832455</td>
          <td>21.116290</td>
          <td>21.920919</td>
          <td>17.593036</td>
          <td>0.005000</td>
          <td>21.725252</td>
          <td>0.005790</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.917720</td>
          <td>22.758461</td>
          <td>18.034445</td>
          <td>22.034108</td>
          <td>24.477586</td>
          <td>22.436871</td>
          <td>23.218767</td>
          <td>0.008313</td>
          <td>20.845193</td>
          <td>0.005166</td>
          <td>20.937073</td>
          <td>0.005196</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.198994</td>
          <td>20.790335</td>
          <td>23.884056</td>
          <td>25.271273</td>
          <td>23.988082</td>
          <td>24.418967</td>
          <td>27.067130</td>
          <td>0.209478</td>
          <td>20.465689</td>
          <td>0.005083</td>
          <td>23.499948</td>
          <td>0.015722</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.478169</td>
          <td>23.222825</td>
          <td>22.248945</td>
          <td>24.176434</td>
          <td>24.287418</td>
          <td>26.009369</td>
          <td>20.615346</td>
          <td>0.005036</td>
          <td>17.289032</td>
          <td>0.005000</td>
          <td>21.865409</td>
          <td>0.006002</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.202028</td>
          <td>31.888368</td>
          <td>21.021721</td>
          <td>25.762218</td>
          <td>20.047908</td>
          <td>18.315391</td>
          <td>25.047095</td>
          <td>0.035696</td>
          <td>26.913841</td>
          <td>0.302322</td>
          <td>25.713454</td>
          <td>0.109751</td>
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
          <td>24.179715</td>
          <td>25.582359</td>
          <td>20.984558</td>
          <td>23.083639</td>
          <td>18.885631</td>
          <td>22.222352</td>
          <td>15.907053</td>
          <td>0.005000</td>
          <td>22.816597</td>
          <td>0.009404</td>
          <td>23.250579</td>
          <td>0.012872</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.462884</td>
          <td>21.207371</td>
          <td>22.662072</td>
          <td>19.417803</td>
          <td>24.903312</td>
          <td>19.207609</td>
          <td>19.560817</td>
          <td>0.005005</td>
          <td>24.251055</td>
          <td>0.030001</td>
          <td>19.664465</td>
          <td>0.005019</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.345136</td>
          <td>20.949138</td>
          <td>23.111157</td>
          <td>29.213346</td>
          <td>25.864161</td>
          <td>18.747325</td>
          <td>25.468802</td>
          <td>0.051972</td>
          <td>19.578303</td>
          <td>0.005016</td>
          <td>24.240698</td>
          <td>0.029728</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.232152</td>
          <td>22.507495</td>
          <td>25.361153</td>
          <td>21.273335</td>
          <td>23.748147</td>
          <td>23.445438</td>
          <td>27.301453</td>
          <td>0.254406</td>
          <td>24.514968</td>
          <td>0.037917</td>
          <td>18.762432</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.344503</td>
          <td>23.351990</td>
          <td>19.456487</td>
          <td>25.004445</td>
          <td>27.020651</td>
          <td>20.917093</td>
          <td>24.806910</td>
          <td>0.028854</td>
          <td>27.297751</td>
          <td>0.408834</td>
          <td>21.490892</td>
          <td>0.005526</td>
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
          <td>29.872757</td>
          <td>29.514717</td>
          <td>22.006534</td>
          <td>23.832455</td>
          <td>21.116290</td>
          <td>21.920919</td>
          <td>17.585530</td>
          <td>0.005020</td>
          <td>21.745059</td>
          <td>0.017741</td>
          <td>25.544813</td>
          <td>0.492564</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.917720</td>
          <td>22.758461</td>
          <td>18.034445</td>
          <td>22.034108</td>
          <td>24.477586</td>
          <td>22.436871</td>
          <td>23.209910</td>
          <td>0.076931</td>
          <td>20.863173</td>
          <td>0.009083</td>
          <td>20.914214</td>
          <td>0.010044</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.198994</td>
          <td>20.790335</td>
          <td>23.884056</td>
          <td>25.271273</td>
          <td>23.988082</td>
          <td>24.418967</td>
          <td>26.368824</td>
          <td>0.925128</td>
          <td>20.455011</td>
          <td>0.007222</td>
          <td>23.637485</td>
          <td>0.102688</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.478169</td>
          <td>23.222825</td>
          <td>22.248945</td>
          <td>24.176434</td>
          <td>24.287418</td>
          <td>26.009369</td>
          <td>20.596167</td>
          <td>0.008709</td>
          <td>17.290679</td>
          <td>0.005008</td>
          <td>21.849383</td>
          <td>0.021116</td>
        </tr>
        <tr>
          <th>4</th>
          <td>16.202028</td>
          <td>31.888368</td>
          <td>21.021721</td>
          <td>25.762218</td>
          <td>20.047908</td>
          <td>18.315391</td>
          <td>25.371580</td>
          <td>0.466426</td>
          <td>26.726732</td>
          <td>1.018469</td>
          <td>25.221121</td>
          <td>0.385369</td>
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
          <td>24.179715</td>
          <td>25.582359</td>
          <td>20.984558</td>
          <td>23.083639</td>
          <td>18.885631</td>
          <td>22.222352</td>
          <td>15.904418</td>
          <td>0.005001</td>
          <td>22.829232</td>
          <td>0.045892</td>
          <td>23.275412</td>
          <td>0.074615</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.462884</td>
          <td>21.207371</td>
          <td>22.662072</td>
          <td>19.417803</td>
          <td>24.903312</td>
          <td>19.207609</td>
          <td>19.570828</td>
          <td>0.005719</td>
          <td>24.288566</td>
          <td>0.165512</td>
          <td>19.657720</td>
          <td>0.005703</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.345136</td>
          <td>20.949138</td>
          <td>23.111157</td>
          <td>29.213346</td>
          <td>25.864161</td>
          <td>18.747325</td>
          <td>24.745893</td>
          <td>0.286198</td>
          <td>19.577300</td>
          <td>0.005513</td>
          <td>23.759832</td>
          <td>0.114289</td>
        </tr>
        <tr>
          <th>998</th>
          <td>29.232152</td>
          <td>22.507495</td>
          <td>25.361153</td>
          <td>21.273335</td>
          <td>23.748147</td>
          <td>23.445438</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.576543</td>
          <td>0.211135</td>
          <td>18.767051</td>
          <td>0.005144</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.344503</td>
          <td>23.351990</td>
          <td>19.456487</td>
          <td>25.004445</td>
          <td>27.020651</td>
          <td>20.917093</td>
          <td>24.741898</td>
          <td>0.285274</td>
          <td>25.898189</td>
          <td>0.591948</td>
          <td>21.486323</td>
          <td>0.015547</td>
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


