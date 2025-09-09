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
          <td>22.697965</td>
          <td>21.473287</td>
          <td>24.946329</td>
          <td>24.871449</td>
          <td>22.745891</td>
          <td>22.171287</td>
          <td>26.732564</td>
          <td>23.285564</td>
          <td>24.557181</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.823690</td>
          <td>21.778838</td>
          <td>21.019660</td>
          <td>21.226602</td>
          <td>22.227001</td>
          <td>25.523504</td>
          <td>22.620488</td>
          <td>22.537527</td>
          <td>24.068415</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.587360</td>
          <td>21.645043</td>
          <td>17.392642</td>
          <td>24.818535</td>
          <td>24.207072</td>
          <td>23.279539</td>
          <td>24.929337</td>
          <td>23.396991</td>
          <td>27.180463</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.154948</td>
          <td>20.149180</td>
          <td>21.962824</td>
          <td>17.939759</td>
          <td>21.755407</td>
          <td>18.381449</td>
          <td>27.849715</td>
          <td>27.589182</td>
          <td>23.759257</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.968283</td>
          <td>20.663533</td>
          <td>17.122388</td>
          <td>24.416756</td>
          <td>24.481340</td>
          <td>20.741778</td>
          <td>25.423717</td>
          <td>26.042209</td>
          <td>21.883035</td>
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
          <td>28.650400</td>
          <td>24.008232</td>
          <td>25.326699</td>
          <td>27.058588</td>
          <td>19.154560</td>
          <td>25.150805</td>
          <td>21.774320</td>
          <td>16.289922</td>
          <td>22.067742</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.189872</td>
          <td>21.371725</td>
          <td>23.551021</td>
          <td>22.709877</td>
          <td>18.603016</td>
          <td>18.659662</td>
          <td>20.786211</td>
          <td>26.851853</td>
          <td>22.603200</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.656131</td>
          <td>24.394183</td>
          <td>29.881802</td>
          <td>19.279610</td>
          <td>21.480869</td>
          <td>22.128059</td>
          <td>22.800014</td>
          <td>26.613234</td>
          <td>24.247485</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.012755</td>
          <td>25.696731</td>
          <td>21.813689</td>
          <td>28.764432</td>
          <td>23.267950</td>
          <td>23.547949</td>
          <td>23.303780</td>
          <td>21.851068</td>
          <td>21.949383</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.959416</td>
          <td>25.022293</td>
          <td>26.862832</td>
          <td>26.383287</td>
          <td>22.008104</td>
          <td>20.637705</td>
          <td>26.641108</td>
          <td>22.922093</td>
          <td>25.048119</td>
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
          <td>22.688422</td>
          <td>0.014891</td>
          <td>21.466809</td>
          <td>0.005289</td>
          <td>24.929988</td>
          <td>0.030568</td>
          <td>24.794802</td>
          <td>0.044244</td>
          <td>22.761152</td>
          <td>0.014486</td>
          <td>22.167989</td>
          <td>0.018924</td>
          <td>26.732564</td>
          <td>23.285564</td>
          <td>24.557181</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.881827</td>
          <td>0.503399</td>
          <td>21.773668</td>
          <td>0.005461</td>
          <td>21.015829</td>
          <td>0.005100</td>
          <td>21.220181</td>
          <td>0.005329</td>
          <td>22.227054</td>
          <td>0.009784</td>
          <td>25.106462</td>
          <td>0.243456</td>
          <td>22.620488</td>
          <td>22.537527</td>
          <td>24.068415</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.594529</td>
          <td>0.005614</td>
          <td>21.640116</td>
          <td>0.005376</td>
          <td>17.397130</td>
          <td>0.005001</td>
          <td>24.804545</td>
          <td>0.044629</td>
          <td>24.195405</td>
          <td>0.049821</td>
          <td>23.262064</td>
          <td>0.049204</td>
          <td>24.929337</td>
          <td>23.396991</td>
          <td>27.180463</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.154220</td>
          <td>0.006339</td>
          <td>20.140025</td>
          <td>0.005045</td>
          <td>21.969124</td>
          <td>0.005453</td>
          <td>17.931676</td>
          <td>0.005004</td>
          <td>21.760109</td>
          <td>0.007468</td>
          <td>18.379769</td>
          <td>0.005054</td>
          <td>27.849715</td>
          <td>27.589182</td>
          <td>23.759257</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.981770</td>
          <td>0.018816</td>
          <td>20.659824</td>
          <td>0.005090</td>
          <td>17.125697</td>
          <td>0.005001</td>
          <td>24.438561</td>
          <td>0.032282</td>
          <td>24.571163</td>
          <td>0.069528</td>
          <td>20.746327</td>
          <td>0.007145</td>
          <td>25.423717</td>
          <td>26.042209</td>
          <td>21.883035</td>
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
          <td>27.052934</td>
          <td>0.569995</td>
          <td>24.019854</td>
          <td>0.016011</td>
          <td>25.319387</td>
          <td>0.043123</td>
          <td>27.375940</td>
          <td>0.398655</td>
          <td>19.152645</td>
          <td>0.005044</td>
          <td>25.087559</td>
          <td>0.239689</td>
          <td>21.774320</td>
          <td>16.289922</td>
          <td>22.067742</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.189217</td>
          <td>0.022350</td>
          <td>21.376804</td>
          <td>0.005252</td>
          <td>23.547547</td>
          <td>0.010001</td>
          <td>22.721013</td>
          <td>0.008425</td>
          <td>18.598197</td>
          <td>0.005021</td>
          <td>18.658231</td>
          <td>0.005081</td>
          <td>20.786211</td>
          <td>26.851853</td>
          <td>22.603200</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.650705</td>
          <td>0.005020</td>
          <td>24.402544</td>
          <td>0.022018</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.296049</td>
          <td>0.005019</td>
          <td>21.481110</td>
          <td>0.006619</td>
          <td>22.144851</td>
          <td>0.018560</td>
          <td>22.800014</td>
          <td>26.613234</td>
          <td>24.247485</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.025752</td>
          <td>0.019508</td>
          <td>25.847923</td>
          <td>0.078378</td>
          <td>21.819415</td>
          <td>0.005355</td>
          <td>28.695065</td>
          <td>0.992529</td>
          <td>23.277203</td>
          <td>0.022236</td>
          <td>23.488465</td>
          <td>0.060154</td>
          <td>23.303780</td>
          <td>21.851068</td>
          <td>21.949383</td>
        </tr>
        <tr>
          <th>999</th>
          <td>28.664929</td>
          <td>1.517312</td>
          <td>24.999780</td>
          <td>0.037020</td>
          <td>26.843824</td>
          <td>0.164312</td>
          <td>26.414040</td>
          <td>0.182540</td>
          <td>22.023105</td>
          <td>0.008609</td>
          <td>20.637996</td>
          <td>0.006821</td>
          <td>26.641108</td>
          <td>22.922093</td>
          <td>25.048119</td>
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
          <td>22.697965</td>
          <td>21.473287</td>
          <td>24.946329</td>
          <td>24.871449</td>
          <td>22.745891</td>
          <td>22.171287</td>
          <td>26.808847</td>
          <td>0.168400</td>
          <td>23.271475</td>
          <td>0.013084</td>
          <td>24.560838</td>
          <td>0.039497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.823690</td>
          <td>21.778838</td>
          <td>21.019660</td>
          <td>21.226602</td>
          <td>22.227001</td>
          <td>25.523504</td>
          <td>22.619975</td>
          <td>0.006297</td>
          <td>22.536575</td>
          <td>0.007932</td>
          <td>24.099002</td>
          <td>0.026240</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.587360</td>
          <td>21.645043</td>
          <td>17.392642</td>
          <td>24.818535</td>
          <td>24.207072</td>
          <td>23.279539</td>
          <td>24.902863</td>
          <td>0.031408</td>
          <td>23.372432</td>
          <td>0.014174</td>
          <td>27.534744</td>
          <td>0.488903</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.154948</td>
          <td>20.149180</td>
          <td>21.962824</td>
          <td>17.939759</td>
          <td>21.755407</td>
          <td>18.381449</td>
          <td>27.431470</td>
          <td>0.282873</td>
          <td>28.678497</td>
          <td>1.050266</td>
          <td>23.718684</td>
          <td>0.018881</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.968283</td>
          <td>20.663533</td>
          <td>17.122388</td>
          <td>24.416756</td>
          <td>24.481340</td>
          <td>20.741778</td>
          <td>25.433617</td>
          <td>0.050367</td>
          <td>26.330778</td>
          <td>0.186770</td>
          <td>21.879799</td>
          <td>0.006027</td>
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
          <td>28.650400</td>
          <td>24.008232</td>
          <td>25.326699</td>
          <td>27.058588</td>
          <td>19.154560</td>
          <td>25.150805</td>
          <td>21.776946</td>
          <td>0.005301</td>
          <td>16.298422</td>
          <td>0.005000</td>
          <td>22.077659</td>
          <td>0.006427</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.189872</td>
          <td>21.371725</td>
          <td>23.551021</td>
          <td>22.709877</td>
          <td>18.603016</td>
          <td>18.659662</td>
          <td>20.790317</td>
          <td>0.005050</td>
          <td>26.788385</td>
          <td>0.273144</td>
          <td>22.604352</td>
          <td>0.008243</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.656131</td>
          <td>24.394183</td>
          <td>29.881802</td>
          <td>19.279610</td>
          <td>21.480869</td>
          <td>22.128059</td>
          <td>22.805802</td>
          <td>0.006755</td>
          <td>26.370921</td>
          <td>0.193210</td>
          <td>24.283350</td>
          <td>0.030870</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.012755</td>
          <td>25.696731</td>
          <td>21.813689</td>
          <td>28.764432</td>
          <td>23.267950</td>
          <td>23.547949</td>
          <td>23.310586</td>
          <td>0.008787</td>
          <td>21.851115</td>
          <td>0.005978</td>
          <td>21.953100</td>
          <td>0.006161</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.959416</td>
          <td>25.022293</td>
          <td>26.862832</td>
          <td>26.383287</td>
          <td>22.008104</td>
          <td>20.637705</td>
          <td>26.758896</td>
          <td>0.161369</td>
          <td>22.925448</td>
          <td>0.010123</td>
          <td>25.066117</td>
          <td>0.061958</td>
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
          <td>22.697965</td>
          <td>21.473287</td>
          <td>24.946329</td>
          <td>24.871449</td>
          <td>22.745891</td>
          <td>22.171287</td>
          <td>27.936427</td>
          <td>2.064604</td>
          <td>23.214599</td>
          <td>0.064688</td>
          <td>24.843922</td>
          <td>0.285741</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.823690</td>
          <td>21.778838</td>
          <td>21.019660</td>
          <td>21.226602</td>
          <td>22.227001</td>
          <td>25.523504</td>
          <td>22.590726</td>
          <td>0.044343</td>
          <td>22.527584</td>
          <td>0.035082</td>
          <td>24.488518</td>
          <td>0.213260</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.587360</td>
          <td>21.645043</td>
          <td>17.392642</td>
          <td>24.818535</td>
          <td>24.207072</td>
          <td>23.279539</td>
          <td>27.485461</td>
          <td>1.694071</td>
          <td>23.374112</td>
          <td>0.074529</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.154948</td>
          <td>20.149180</td>
          <td>21.962824</td>
          <td>17.939759</td>
          <td>21.755407</td>
          <td>18.381449</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.679976</td>
          <td>0.106583</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.968283</td>
          <td>20.663533</td>
          <td>17.122388</td>
          <td>24.416756</td>
          <td>24.481340</td>
          <td>20.741778</td>
          <td>25.482119</td>
          <td>0.506318</td>
          <td>25.221944</td>
          <td>0.356678</td>
          <td>21.914999</td>
          <td>0.022348</td>
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
          <td>28.650400</td>
          <td>24.008232</td>
          <td>25.326699</td>
          <td>27.058588</td>
          <td>19.154560</td>
          <td>25.150805</td>
          <td>21.794328</td>
          <td>0.021952</td>
          <td>16.288985</td>
          <td>0.005001</td>
          <td>22.043857</td>
          <td>0.025003</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.189872</td>
          <td>21.371725</td>
          <td>23.551021</td>
          <td>22.709877</td>
          <td>18.603016</td>
          <td>18.659662</td>
          <td>20.793184</td>
          <td>0.009900</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.605919</td>
          <td>0.041115</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.656131</td>
          <td>24.394183</td>
          <td>29.881802</td>
          <td>19.279610</td>
          <td>21.480869</td>
          <td>22.128059</td>
          <td>22.888866</td>
          <td>0.057840</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.205376</td>
          <td>0.167902</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.012755</td>
          <td>25.696731</td>
          <td>21.813689</td>
          <td>28.764432</td>
          <td>23.267950</td>
          <td>23.547949</td>
          <td>23.273581</td>
          <td>0.081390</td>
          <td>21.898746</td>
          <td>0.020217</td>
          <td>21.980493</td>
          <td>0.023657</td>
        </tr>
        <tr>
          <th>999</th>
          <td>27.959416</td>
          <td>25.022293</td>
          <td>26.862832</td>
          <td>26.383287</td>
          <td>22.008104</td>
          <td>20.637705</td>
          <td>25.510703</td>
          <td>0.517059</td>
          <td>22.976351</td>
          <td>0.052323</td>
          <td>25.352061</td>
          <td>0.426168</td>
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


