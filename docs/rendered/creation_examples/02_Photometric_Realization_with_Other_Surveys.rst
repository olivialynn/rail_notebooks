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
          <td>26.303727</td>
          <td>19.713560</td>
          <td>25.818159</td>
          <td>21.903723</td>
          <td>27.736378</td>
          <td>15.636510</td>
          <td>24.642950</td>
          <td>23.629187</td>
          <td>23.063155</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.958817</td>
          <td>25.166599</td>
          <td>18.848050</td>
          <td>20.857719</td>
          <td>21.122466</td>
          <td>19.904906</td>
          <td>21.077652</td>
          <td>26.769465</td>
          <td>21.133407</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.972232</td>
          <td>21.794529</td>
          <td>23.183319</td>
          <td>20.875095</td>
          <td>23.859394</td>
          <td>20.372000</td>
          <td>21.516138</td>
          <td>25.995094</td>
          <td>25.208172</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.097264</td>
          <td>23.965329</td>
          <td>19.928208</td>
          <td>18.489689</td>
          <td>22.773740</td>
          <td>23.657727</td>
          <td>22.680038</td>
          <td>26.338050</td>
          <td>23.529866</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.132003</td>
          <td>22.994291</td>
          <td>16.694799</td>
          <td>22.784186</td>
          <td>30.143666</td>
          <td>23.120945</td>
          <td>27.642598</td>
          <td>20.441305</td>
          <td>19.692761</td>
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
          <td>20.044138</td>
          <td>18.969800</td>
          <td>25.298442</td>
          <td>17.957175</td>
          <td>27.078439</td>
          <td>19.698857</td>
          <td>24.836306</td>
          <td>22.075668</td>
          <td>23.551726</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.062215</td>
          <td>21.764077</td>
          <td>32.156134</td>
          <td>21.449544</td>
          <td>27.122786</td>
          <td>28.612431</td>
          <td>24.394471</td>
          <td>28.473044</td>
          <td>21.081489</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.183739</td>
          <td>23.580039</td>
          <td>22.392753</td>
          <td>20.622788</td>
          <td>26.154679</td>
          <td>22.972151</td>
          <td>18.186881</td>
          <td>28.061170</td>
          <td>20.870233</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.684790</td>
          <td>22.605746</td>
          <td>24.589587</td>
          <td>24.190703</td>
          <td>19.861108</td>
          <td>21.647182</td>
          <td>19.946107</td>
          <td>25.709143</td>
          <td>19.928969</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.991100</td>
          <td>17.270168</td>
          <td>23.646907</td>
          <td>25.550084</td>
          <td>22.227173</td>
          <td>22.103191</td>
          <td>28.331576</td>
          <td>24.297010</td>
          <td>20.458091</td>
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
          <td>27.167846</td>
          <td>0.618361</td>
          <td>19.708191</td>
          <td>0.005027</td>
          <td>25.740622</td>
          <td>0.062675</td>
          <td>21.899807</td>
          <td>0.005984</td>
          <td>26.732926</td>
          <td>0.427054</td>
          <td>15.632063</td>
          <td>0.005002</td>
          <td>24.642950</td>
          <td>23.629187</td>
          <td>23.063155</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.918361</td>
          <td>0.100844</td>
          <td>25.121841</td>
          <td>0.041231</td>
          <td>18.839661</td>
          <td>0.005006</td>
          <td>20.861313</td>
          <td>0.005185</td>
          <td>21.118491</td>
          <td>0.005917</td>
          <td>19.892883</td>
          <td>0.005562</td>
          <td>21.077652</td>
          <td>26.769465</td>
          <td>21.133407</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.969894</td>
          <td>0.005079</td>
          <td>21.798534</td>
          <td>0.005479</td>
          <td>23.183225</td>
          <td>0.008007</td>
          <td>20.873629</td>
          <td>0.005189</td>
          <td>23.836655</td>
          <td>0.036250</td>
          <td>20.370452</td>
          <td>0.006203</td>
          <td>21.516138</td>
          <td>25.995094</td>
          <td>25.208172</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.094231</td>
          <td>0.005091</td>
          <td>23.958910</td>
          <td>0.015245</td>
          <td>19.922451</td>
          <td>0.005021</td>
          <td>18.487646</td>
          <td>0.005007</td>
          <td>22.764531</td>
          <td>0.014525</td>
          <td>23.587436</td>
          <td>0.065671</td>
          <td>22.680038</td>
          <td>26.338050</td>
          <td>23.529866</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.156119</td>
          <td>0.021738</td>
          <td>23.007399</td>
          <td>0.007965</td>
          <td>16.696703</td>
          <td>0.005001</td>
          <td>22.767022</td>
          <td>0.008656</td>
          <td>26.512639</td>
          <td>0.360236</td>
          <td>23.049947</td>
          <td>0.040764</td>
          <td>27.642598</td>
          <td>20.441305</td>
          <td>19.692761</td>
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
          <td>20.048962</td>
          <td>0.005295</td>
          <td>18.973779</td>
          <td>0.005012</td>
          <td>25.255171</td>
          <td>0.040736</td>
          <td>17.954022</td>
          <td>0.005004</td>
          <td>26.007451</td>
          <td>0.239816</td>
          <td>19.693485</td>
          <td>0.005408</td>
          <td>24.836306</td>
          <td>22.075668</td>
          <td>23.551726</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.143549</td>
          <td>0.284178</td>
          <td>21.767552</td>
          <td>0.005456</td>
          <td>29.227973</td>
          <td>0.980180</td>
          <td>21.447747</td>
          <td>0.005475</td>
          <td>28.282895</td>
          <td>1.195755</td>
          <td>26.134883</td>
          <td>0.542213</td>
          <td>24.394471</td>
          <td>28.473044</td>
          <td>21.081489</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.177199</td>
          <td>0.022125</td>
          <td>23.573576</td>
          <td>0.011371</td>
          <td>22.391460</td>
          <td>0.005893</td>
          <td>20.624828</td>
          <td>0.005128</td>
          <td>26.242170</td>
          <td>0.290487</td>
          <td>23.014484</td>
          <td>0.039503</td>
          <td>18.186881</td>
          <td>28.061170</td>
          <td>20.870233</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.705275</td>
          <td>0.034794</td>
          <td>22.601506</td>
          <td>0.006638</td>
          <td>24.592842</td>
          <td>0.022792</td>
          <td>24.177717</td>
          <td>0.025690</td>
          <td>19.861630</td>
          <td>0.005124</td>
          <td>21.643964</td>
          <td>0.012427</td>
          <td>19.946107</td>
          <td>25.709143</td>
          <td>19.928969</td>
        </tr>
        <tr>
          <th>999</th>
          <td>inf</td>
          <td>inf</td>
          <td>17.265349</td>
          <td>0.005002</td>
          <td>23.649475</td>
          <td>0.010728</td>
          <td>25.535134</td>
          <td>0.085261</td>
          <td>22.230820</td>
          <td>0.009809</td>
          <td>22.112818</td>
          <td>0.018070</td>
          <td>28.331576</td>
          <td>24.297010</td>
          <td>20.458091</td>
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
          <td>26.303727</td>
          <td>19.713560</td>
          <td>25.818159</td>
          <td>21.903723</td>
          <td>27.736378</td>
          <td>15.636510</td>
          <td>24.619915</td>
          <td>0.024485</td>
          <td>23.660898</td>
          <td>0.017979</td>
          <td>23.062815</td>
          <td>0.011167</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.958817</td>
          <td>25.166599</td>
          <td>18.848050</td>
          <td>20.857719</td>
          <td>21.122466</td>
          <td>19.904906</td>
          <td>21.073294</td>
          <td>0.005084</td>
          <td>27.034610</td>
          <td>0.332927</td>
          <td>21.133327</td>
          <td>0.005279</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.972232</td>
          <td>21.794529</td>
          <td>23.183319</td>
          <td>20.875095</td>
          <td>23.859394</td>
          <td>20.372000</td>
          <td>21.524335</td>
          <td>0.005191</td>
          <td>26.358017</td>
          <td>0.191118</td>
          <td>25.167837</td>
          <td>0.067822</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.097264</td>
          <td>23.965329</td>
          <td>19.928208</td>
          <td>18.489689</td>
          <td>22.773740</td>
          <td>23.657727</td>
          <td>22.687523</td>
          <td>0.006450</td>
          <td>26.272972</td>
          <td>0.177842</td>
          <td>23.552805</td>
          <td>0.016424</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.132003</td>
          <td>22.994291</td>
          <td>16.694799</td>
          <td>22.784186</td>
          <td>30.143666</td>
          <td>23.120945</td>
          <td>29.107583</td>
          <td>0.947524</td>
          <td>20.440650</td>
          <td>0.005079</td>
          <td>19.684212</td>
          <td>0.005020</td>
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
          <td>20.044138</td>
          <td>18.969800</td>
          <td>25.298442</td>
          <td>17.957175</td>
          <td>27.078439</td>
          <td>19.698857</td>
          <td>24.856153</td>
          <td>0.030137</td>
          <td>22.080666</td>
          <td>0.006434</td>
          <td>23.566629</td>
          <td>0.016614</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.062215</td>
          <td>21.764077</td>
          <td>32.156134</td>
          <td>21.449544</td>
          <td>27.122786</td>
          <td>28.612431</td>
          <td>24.417171</td>
          <td>0.020539</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.083602</td>
          <td>0.005255</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.183739</td>
          <td>23.580039</td>
          <td>22.392753</td>
          <td>20.622788</td>
          <td>26.154679</td>
          <td>22.972151</td>
          <td>18.183769</td>
          <td>0.005000</td>
          <td>28.487242</td>
          <td>0.935728</td>
          <td>20.864300</td>
          <td>0.005172</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.684790</td>
          <td>22.605746</td>
          <td>24.589587</td>
          <td>24.190703</td>
          <td>19.861108</td>
          <td>21.647182</td>
          <td>19.943702</td>
          <td>0.005011</td>
          <td>25.713958</td>
          <td>0.109800</td>
          <td>19.921756</td>
          <td>0.005031</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.991100</td>
          <td>17.270168</td>
          <td>23.646907</td>
          <td>25.550084</td>
          <td>22.227173</td>
          <td>22.103191</td>
          <td>28.265930</td>
          <td>0.538314</td>
          <td>24.264806</td>
          <td>0.030368</td>
          <td>20.458716</td>
          <td>0.005082</td>
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
          <td>26.303727</td>
          <td>19.713560</td>
          <td>25.818159</td>
          <td>21.903723</td>
          <td>27.736378</td>
          <td>15.636510</td>
          <td>24.743196</td>
          <td>0.285573</td>
          <td>23.460670</td>
          <td>0.080466</td>
          <td>23.064749</td>
          <td>0.061882</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.958817</td>
          <td>25.166599</td>
          <td>18.848050</td>
          <td>20.857719</td>
          <td>21.122466</td>
          <td>19.904906</td>
          <td>21.069064</td>
          <td>0.012089</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.153000</td>
          <td>0.011943</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.972232</td>
          <td>21.794529</td>
          <td>23.183319</td>
          <td>20.875095</td>
          <td>23.859394</td>
          <td>20.372000</td>
          <td>21.529930</td>
          <td>0.017517</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.683178</td>
          <td>0.545089</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.097264</td>
          <td>23.965329</td>
          <td>19.928208</td>
          <td>18.489689</td>
          <td>22.773740</td>
          <td>23.657727</td>
          <td>22.676583</td>
          <td>0.047870</td>
          <td>25.771085</td>
          <td>0.540332</td>
          <td>23.532492</td>
          <td>0.093640</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.132003</td>
          <td>22.994291</td>
          <td>16.694799</td>
          <td>22.784186</td>
          <td>30.143666</td>
          <td>23.120945</td>
          <td>25.502755</td>
          <td>0.514055</td>
          <td>20.437482</td>
          <td>0.007162</td>
          <td>19.696725</td>
          <td>0.005752</td>
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
          <td>20.044138</td>
          <td>18.969800</td>
          <td>25.298442</td>
          <td>17.957175</td>
          <td>27.078439</td>
          <td>19.698857</td>
          <td>24.740886</td>
          <td>0.285040</td>
          <td>22.058653</td>
          <td>0.023212</td>
          <td>23.596378</td>
          <td>0.099050</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.062215</td>
          <td>21.764077</td>
          <td>32.156134</td>
          <td>21.449544</td>
          <td>27.122786</td>
          <td>28.612431</td>
          <td>24.465346</td>
          <td>0.227359</td>
          <td>27.556417</td>
          <td>1.593437</td>
          <td>21.085738</td>
          <td>0.011357</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.183739</td>
          <td>23.580039</td>
          <td>22.392753</td>
          <td>20.622788</td>
          <td>26.154679</td>
          <td>22.972151</td>
          <td>18.193877</td>
          <td>0.005061</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.865150</td>
          <td>0.009714</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.684790</td>
          <td>22.605746</td>
          <td>24.589587</td>
          <td>24.190703</td>
          <td>19.861108</td>
          <td>21.647182</td>
          <td>19.948411</td>
          <td>0.006360</td>
          <td>24.563784</td>
          <td>0.208892</td>
          <td>19.931371</td>
          <td>0.006120</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.991100</td>
          <td>17.270168</td>
          <td>23.646907</td>
          <td>25.550084</td>
          <td>22.227173</td>
          <td>22.103191</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.518770</td>
          <td>0.201151</td>
          <td>20.459579</td>
          <td>0.007610</td>
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


