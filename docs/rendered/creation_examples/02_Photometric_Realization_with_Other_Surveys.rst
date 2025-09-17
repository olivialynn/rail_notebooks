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
          <td>27.502204</td>
          <td>20.515375</td>
          <td>21.233295</td>
          <td>21.469222</td>
          <td>26.363163</td>
          <td>19.723845</td>
          <td>23.776374</td>
          <td>23.728365</td>
          <td>20.507974</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.107741</td>
          <td>21.393745</td>
          <td>19.926748</td>
          <td>24.614444</td>
          <td>23.367640</td>
          <td>21.245834</td>
          <td>21.617501</td>
          <td>24.732376</td>
          <td>23.319040</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.715956</td>
          <td>25.211735</td>
          <td>22.058143</td>
          <td>16.216041</td>
          <td>20.770211</td>
          <td>18.600850</td>
          <td>15.681967</td>
          <td>24.201785</td>
          <td>21.405803</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.865421</td>
          <td>31.891417</td>
          <td>23.785106</td>
          <td>24.434645</td>
          <td>25.268466</td>
          <td>20.654434</td>
          <td>21.253757</td>
          <td>22.272399</td>
          <td>31.283089</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.205588</td>
          <td>19.575232</td>
          <td>21.768218</td>
          <td>24.432272</td>
          <td>26.530639</td>
          <td>24.855361</td>
          <td>21.281184</td>
          <td>24.456915</td>
          <td>26.331141</td>
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
          <td>27.269184</td>
          <td>22.319325</td>
          <td>21.942505</td>
          <td>24.488766</td>
          <td>23.004408</td>
          <td>22.580600</td>
          <td>26.035229</td>
          <td>22.137192</td>
          <td>26.909212</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.460863</td>
          <td>19.789888</td>
          <td>26.148149</td>
          <td>23.354909</td>
          <td>19.438038</td>
          <td>22.696357</td>
          <td>22.142038</td>
          <td>23.458743</td>
          <td>23.771391</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.010158</td>
          <td>22.988405</td>
          <td>23.761486</td>
          <td>18.974411</td>
          <td>19.571412</td>
          <td>15.718122</td>
          <td>23.171880</td>
          <td>21.693171</td>
          <td>26.952801</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.582051</td>
          <td>27.196763</td>
          <td>24.519874</td>
          <td>24.601780</td>
          <td>19.127611</td>
          <td>22.712172</td>
          <td>22.896784</td>
          <td>26.912485</td>
          <td>24.939832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.329067</td>
          <td>25.574019</td>
          <td>22.413387</td>
          <td>28.889897</td>
          <td>19.977925</td>
          <td>21.690983</td>
          <td>19.473962</td>
          <td>20.866046</td>
          <td>25.793567</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>20.521261</td>
          <td>0.005075</td>
          <td>21.229793</td>
          <td>0.005139</td>
          <td>21.461916</td>
          <td>0.005486</td>
          <td>26.645773</td>
          <td>0.399487</td>
          <td>19.727475</td>
          <td>0.005431</td>
          <td>23.776374</td>
          <td>23.728365</td>
          <td>20.507974</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.022114</td>
          <td>0.257458</td>
          <td>21.394777</td>
          <td>0.005259</td>
          <td>19.927577</td>
          <td>0.005021</td>
          <td>24.623576</td>
          <td>0.038013</td>
          <td>23.376283</td>
          <td>0.024218</td>
          <td>21.256306</td>
          <td>0.009462</td>
          <td>21.617501</td>
          <td>24.732376</td>
          <td>23.319040</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.416844</td>
          <td>0.155106</td>
          <td>25.165325</td>
          <td>0.042848</td>
          <td>22.060507</td>
          <td>0.005524</td>
          <td>16.217097</td>
          <td>0.005001</td>
          <td>20.771147</td>
          <td>0.005525</td>
          <td>18.600736</td>
          <td>0.005075</td>
          <td>15.681967</td>
          <td>24.201785</td>
          <td>21.405803</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.878054</td>
          <td>0.017297</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.802096</td>
          <td>0.011982</td>
          <td>24.431865</td>
          <td>0.032093</td>
          <td>25.146045</td>
          <td>0.115244</td>
          <td>20.655450</td>
          <td>0.006870</td>
          <td>21.253757</td>
          <td>22.272399</td>
          <td>31.283089</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.202441</td>
          <td>0.005035</td>
          <td>19.574590</td>
          <td>0.005023</td>
          <td>21.765055</td>
          <td>0.005326</td>
          <td>24.434922</td>
          <td>0.032179</td>
          <td>26.360101</td>
          <td>0.319319</td>
          <td>24.585930</td>
          <td>0.157154</td>
          <td>21.281184</td>
          <td>24.456915</td>
          <td>26.331141</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>22.315786</td>
          <td>0.006061</td>
          <td>21.940628</td>
          <td>0.005432</td>
          <td>24.477511</td>
          <td>0.033410</td>
          <td>22.991166</td>
          <td>0.017464</td>
          <td>22.569233</td>
          <td>0.026698</td>
          <td>26.035229</td>
          <td>22.137192</td>
          <td>26.909212</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.459867</td>
          <td>0.005141</td>
          <td>19.795082</td>
          <td>0.005030</td>
          <td>26.197474</td>
          <td>0.093823</td>
          <td>23.354447</td>
          <td>0.012978</td>
          <td>19.447427</td>
          <td>0.005067</td>
          <td>22.719816</td>
          <td>0.030458</td>
          <td>22.142038</td>
          <td>23.458743</td>
          <td>23.771391</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.006267</td>
          <td>0.009269</td>
          <td>22.992678</td>
          <td>0.007904</td>
          <td>23.756166</td>
          <td>0.011583</td>
          <td>18.973808</td>
          <td>0.005013</td>
          <td>19.583983</td>
          <td>0.005082</td>
          <td>15.718976</td>
          <td>0.005002</td>
          <td>23.171880</td>
          <td>21.693171</td>
          <td>26.952801</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.579427</td>
          <td>0.005163</td>
          <td>27.373629</td>
          <td>0.287919</td>
          <td>24.486948</td>
          <td>0.020817</td>
          <td>24.634865</td>
          <td>0.038395</td>
          <td>19.130048</td>
          <td>0.005042</td>
          <td>22.714947</td>
          <td>0.030328</td>
          <td>22.896784</td>
          <td>26.912485</td>
          <td>24.939832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.325376</td>
          <td>0.006701</td>
          <td>25.632255</td>
          <td>0.064784</td>
          <td>22.400695</td>
          <td>0.005906</td>
          <td>28.424505</td>
          <td>0.838657</td>
          <td>19.982917</td>
          <td>0.005150</td>
          <td>21.676451</td>
          <td>0.012736</td>
          <td>19.473962</td>
          <td>20.866046</td>
          <td>25.793567</td>
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
          <td>27.502204</td>
          <td>20.515375</td>
          <td>21.233295</td>
          <td>21.469222</td>
          <td>26.363163</td>
          <td>19.723845</td>
          <td>23.768922</td>
          <td>0.012088</td>
          <td>23.722332</td>
          <td>0.018940</td>
          <td>20.502517</td>
          <td>0.005089</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.107741</td>
          <td>21.393745</td>
          <td>19.926748</td>
          <td>24.614444</td>
          <td>23.367640</td>
          <td>21.245834</td>
          <td>21.619298</td>
          <td>0.005227</td>
          <td>24.679228</td>
          <td>0.043891</td>
          <td>23.318455</td>
          <td>0.013577</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.715956</td>
          <td>25.211735</td>
          <td>22.058143</td>
          <td>16.216041</td>
          <td>20.770211</td>
          <td>18.600850</td>
          <td>15.684293</td>
          <td>0.005000</td>
          <td>24.204938</td>
          <td>0.028804</td>
          <td>21.413137</td>
          <td>0.005459</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.865421</td>
          <td>31.891417</td>
          <td>23.785106</td>
          <td>24.434645</td>
          <td>25.268466</td>
          <td>20.654434</td>
          <td>21.262019</td>
          <td>0.005119</td>
          <td>22.266564</td>
          <td>0.006934</td>
          <td>28.989075</td>
          <td>1.252993</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.205588</td>
          <td>19.575232</td>
          <td>21.768218</td>
          <td>24.432272</td>
          <td>26.530639</td>
          <td>24.855361</td>
          <td>21.274524</td>
          <td>0.005122</td>
          <td>24.498916</td>
          <td>0.037379</td>
          <td>26.805603</td>
          <td>0.276996</td>
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
          <td>27.269184</td>
          <td>22.319325</td>
          <td>21.942505</td>
          <td>24.488766</td>
          <td>23.004408</td>
          <td>22.580600</td>
          <td>25.917471</td>
          <td>0.077448</td>
          <td>22.140759</td>
          <td>0.006581</td>
          <td>26.778658</td>
          <td>0.270989</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.460863</td>
          <td>19.789888</td>
          <td>26.148149</td>
          <td>23.354909</td>
          <td>19.438038</td>
          <td>22.696357</td>
          <td>22.137642</td>
          <td>0.005571</td>
          <td>23.479827</td>
          <td>0.015464</td>
          <td>23.768811</td>
          <td>0.019705</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.010158</td>
          <td>22.988405</td>
          <td>23.761486</td>
          <td>18.974411</td>
          <td>19.571412</td>
          <td>15.718122</td>
          <td>23.190245</td>
          <td>0.008176</td>
          <td>21.700525</td>
          <td>0.005757</td>
          <td>27.087444</td>
          <td>0.347129</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.582051</td>
          <td>27.196763</td>
          <td>24.519874</td>
          <td>24.601780</td>
          <td>19.127611</td>
          <td>22.712172</td>
          <td>22.905205</td>
          <td>0.007055</td>
          <td>27.511392</td>
          <td>0.480495</td>
          <td>24.933714</td>
          <td>0.055068</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.329067</td>
          <td>25.574019</td>
          <td>22.413387</td>
          <td>28.889897</td>
          <td>19.977925</td>
          <td>21.690983</td>
          <td>19.476787</td>
          <td>0.005004</td>
          <td>20.865912</td>
          <td>0.005172</td>
          <td>25.869946</td>
          <td>0.125788</td>
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
          <td>27.502204</td>
          <td>20.515375</td>
          <td>21.233295</td>
          <td>21.469222</td>
          <td>26.363163</td>
          <td>19.723845</td>
          <td>23.593234</td>
          <td>0.107827</td>
          <td>23.752749</td>
          <td>0.104071</td>
          <td>20.512483</td>
          <td>0.007828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.107741</td>
          <td>21.393745</td>
          <td>19.926748</td>
          <td>24.614444</td>
          <td>23.367640</td>
          <td>21.245834</td>
          <td>21.645922</td>
          <td>0.019324</td>
          <td>24.810818</td>
          <td>0.256368</td>
          <td>23.224178</td>
          <td>0.071301</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.715956</td>
          <td>25.211735</td>
          <td>22.058143</td>
          <td>16.216041</td>
          <td>20.770211</td>
          <td>18.600850</td>
          <td>15.678646</td>
          <td>0.005001</td>
          <td>23.868779</td>
          <td>0.115185</td>
          <td>21.400166</td>
          <td>0.014494</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.865421</td>
          <td>31.891417</td>
          <td>23.785106</td>
          <td>24.434645</td>
          <td>25.268466</td>
          <td>20.654434</td>
          <td>21.261081</td>
          <td>0.014046</td>
          <td>22.245715</td>
          <td>0.027340</td>
          <td>26.394536</td>
          <td>0.883150</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.205588</td>
          <td>19.575232</td>
          <td>21.768218</td>
          <td>24.432272</td>
          <td>26.530639</td>
          <td>24.855361</td>
          <td>21.286892</td>
          <td>0.014339</td>
          <td>24.696951</td>
          <td>0.233399</td>
          <td>26.230990</td>
          <td>0.795166</td>
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
          <td>27.269184</td>
          <td>22.319325</td>
          <td>21.942505</td>
          <td>24.488766</td>
          <td>23.004408</td>
          <td>22.580600</td>
          <td>25.346305</td>
          <td>0.457667</td>
          <td>22.121278</td>
          <td>0.024514</td>
          <td>26.211041</td>
          <td>0.784854</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.460863</td>
          <td>19.789888</td>
          <td>26.148149</td>
          <td>23.354909</td>
          <td>19.438038</td>
          <td>22.696357</td>
          <td>22.098361</td>
          <td>0.028637</td>
          <td>23.457695</td>
          <td>0.080255</td>
          <td>23.768138</td>
          <td>0.115120</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.010158</td>
          <td>22.988405</td>
          <td>23.761486</td>
          <td>18.974411</td>
          <td>19.571412</td>
          <td>15.718122</td>
          <td>23.249735</td>
          <td>0.079691</td>
          <td>21.703343</td>
          <td>0.017131</td>
          <td>25.664876</td>
          <td>0.537902</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.582051</td>
          <td>27.196763</td>
          <td>24.519874</td>
          <td>24.601780</td>
          <td>19.127611</td>
          <td>22.712172</td>
          <td>22.891087</td>
          <td>0.057955</td>
          <td>26.622043</td>
          <td>0.955965</td>
          <td>24.625154</td>
          <td>0.238910</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.329067</td>
          <td>25.574019</td>
          <td>22.413387</td>
          <td>28.889897</td>
          <td>19.977925</td>
          <td>21.690983</td>
          <td>19.478278</td>
          <td>0.005613</td>
          <td>20.868296</td>
          <td>0.009113</td>
          <td>25.293401</td>
          <td>0.407471</td>
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


