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
          <td>27.627007</td>
          <td>21.417165</td>
          <td>21.356735</td>
          <td>26.319741</td>
          <td>22.997447</td>
          <td>26.792633</td>
          <td>24.365562</td>
          <td>20.828041</td>
          <td>24.149584</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.792600</td>
          <td>30.121404</td>
          <td>26.723142</td>
          <td>19.833481</td>
          <td>21.069211</td>
          <td>21.759735</td>
          <td>21.543753</td>
          <td>26.214006</td>
          <td>19.008641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.092918</td>
          <td>25.202024</td>
          <td>20.700190</td>
          <td>17.928803</td>
          <td>24.485646</td>
          <td>21.437772</td>
          <td>26.694657</td>
          <td>22.829836</td>
          <td>20.889957</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.972737</td>
          <td>24.344353</td>
          <td>27.012531</td>
          <td>21.951360</td>
          <td>22.810209</td>
          <td>23.824376</td>
          <td>26.151802</td>
          <td>22.648044</td>
          <td>23.449388</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.129879</td>
          <td>27.186182</td>
          <td>19.578435</td>
          <td>22.426960</td>
          <td>19.114104</td>
          <td>27.104605</td>
          <td>24.690947</td>
          <td>21.158004</td>
          <td>21.963019</td>
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
          <td>22.731029</td>
          <td>21.263385</td>
          <td>24.307600</td>
          <td>20.195966</td>
          <td>25.049024</td>
          <td>17.078163</td>
          <td>23.146609</td>
          <td>16.730409</td>
          <td>25.555645</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.383172</td>
          <td>19.377542</td>
          <td>24.129939</td>
          <td>21.862563</td>
          <td>23.682862</td>
          <td>20.942201</td>
          <td>18.874107</td>
          <td>25.379576</td>
          <td>20.125356</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.524072</td>
          <td>23.448332</td>
          <td>28.224900</td>
          <td>21.785689</td>
          <td>23.703352</td>
          <td>27.644350</td>
          <td>25.767457</td>
          <td>25.477292</td>
          <td>21.242115</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.386311</td>
          <td>22.258534</td>
          <td>20.429555</td>
          <td>24.872637</td>
          <td>22.638585</td>
          <td>23.110822</td>
          <td>21.611016</td>
          <td>24.226918</td>
          <td>23.912537</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.943640</td>
          <td>23.713548</td>
          <td>18.121158</td>
          <td>24.098013</td>
          <td>26.511160</td>
          <td>22.754504</td>
          <td>25.602653</td>
          <td>26.808127</td>
          <td>23.216777</td>
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
          <td>21.415318</td>
          <td>0.005267</td>
          <td>21.359311</td>
          <td>0.005171</td>
          <td>26.493728</td>
          <td>0.195240</td>
          <td>22.995140</td>
          <td>0.017521</td>
          <td>28.532765</td>
          <td>2.096190</td>
          <td>24.365562</td>
          <td>20.828041</td>
          <td>24.149584</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.759288</td>
          <td>0.036471</td>
          <td>29.730792</td>
          <td>1.408606</td>
          <td>26.583123</td>
          <td>0.131334</td>
          <td>19.830802</td>
          <td>0.005039</td>
          <td>21.077895</td>
          <td>0.005860</td>
          <td>21.762414</td>
          <td>0.013609</td>
          <td>21.543753</td>
          <td>26.214006</td>
          <td>19.008641</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.149974</td>
          <td>0.051372</td>
          <td>25.215620</td>
          <td>0.044797</td>
          <td>20.696220</td>
          <td>0.005062</td>
          <td>17.942279</td>
          <td>0.005004</td>
          <td>24.405547</td>
          <td>0.060037</td>
          <td>21.423072</td>
          <td>0.010583</td>
          <td>26.694657</td>
          <td>22.829836</td>
          <td>20.889957</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.974747</td>
          <td>0.006042</td>
          <td>24.360163</td>
          <td>0.021239</td>
          <td>27.120998</td>
          <td>0.207698</td>
          <td>21.952306</td>
          <td>0.006069</td>
          <td>22.811758</td>
          <td>0.015083</td>
          <td>23.820647</td>
          <td>0.080711</td>
          <td>26.151802</td>
          <td>22.648044</td>
          <td>23.449388</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.152448</td>
          <td>0.051484</td>
          <td>27.216816</td>
          <td>0.253407</td>
          <td>19.574781</td>
          <td>0.005014</td>
          <td>22.423948</td>
          <td>0.007219</td>
          <td>19.119337</td>
          <td>0.005042</td>
          <td>26.847514</td>
          <td>0.879710</td>
          <td>24.690947</td>
          <td>21.158004</td>
          <td>21.963019</td>
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
          <td>22.738820</td>
          <td>0.015487</td>
          <td>21.262462</td>
          <td>0.005213</td>
          <td>24.314224</td>
          <td>0.017995</td>
          <td>20.201104</td>
          <td>0.005067</td>
          <td>25.054904</td>
          <td>0.106436</td>
          <td>17.084592</td>
          <td>0.005010</td>
          <td>23.146609</td>
          <td>16.730409</td>
          <td>25.555645</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.389777</td>
          <td>0.005015</td>
          <td>19.376578</td>
          <td>0.005018</td>
          <td>24.116644</td>
          <td>0.015300</td>
          <td>21.858985</td>
          <td>0.005922</td>
          <td>23.613075</td>
          <td>0.029769</td>
          <td>20.936712</td>
          <td>0.007842</td>
          <td>18.874107</td>
          <td>25.379576</td>
          <td>20.125356</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.518073</td>
          <td>0.005553</td>
          <td>23.443821</td>
          <td>0.010385</td>
          <td>27.681557</td>
          <td>0.328333</td>
          <td>21.791509</td>
          <td>0.005827</td>
          <td>23.661324</td>
          <td>0.031058</td>
          <td>26.970812</td>
          <td>0.949881</td>
          <td>25.767457</td>
          <td>25.477292</td>
          <td>21.242115</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.382281</td>
          <td>0.011861</td>
          <td>22.251634</td>
          <td>0.005962</td>
          <td>20.420722</td>
          <td>0.005042</td>
          <td>24.764572</td>
          <td>0.043073</td>
          <td>22.635163</td>
          <td>0.013125</td>
          <td>23.097388</td>
          <td>0.042514</td>
          <td>21.611016</td>
          <td>24.226918</td>
          <td>23.912537</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.938729</td>
          <td>0.005077</td>
          <td>23.739058</td>
          <td>0.012846</td>
          <td>18.116510</td>
          <td>0.005003</td>
          <td>24.034088</td>
          <td>0.022688</td>
          <td>28.475701</td>
          <td>1.328175</td>
          <td>22.761942</td>
          <td>0.031607</td>
          <td>25.602653</td>
          <td>26.808127</td>
          <td>23.216777</td>
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
          <td>27.627007</td>
          <td>21.417165</td>
          <td>21.356735</td>
          <td>26.319741</td>
          <td>22.997447</td>
          <td>26.792633</td>
          <td>24.360195</td>
          <td>0.019561</td>
          <td>20.829638</td>
          <td>0.005161</td>
          <td>24.184170</td>
          <td>0.028281</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.792600</td>
          <td>30.121404</td>
          <td>26.723142</td>
          <td>19.833481</td>
          <td>21.069211</td>
          <td>21.759735</td>
          <td>21.547593</td>
          <td>0.005200</td>
          <td>26.285663</td>
          <td>0.179768</td>
          <td>19.009507</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.092918</td>
          <td>25.202024</td>
          <td>20.700190</td>
          <td>17.928803</td>
          <td>24.485646</td>
          <td>21.437772</td>
          <td>26.523140</td>
          <td>0.131728</td>
          <td>22.835547</td>
          <td>0.009523</td>
          <td>20.882190</td>
          <td>0.005177</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.972737</td>
          <td>24.344353</td>
          <td>27.012531</td>
          <td>21.951360</td>
          <td>22.810209</td>
          <td>23.824376</td>
          <td>26.277436</td>
          <td>0.106346</td>
          <td>22.654922</td>
          <td>0.008493</td>
          <td>23.472190</td>
          <td>0.015368</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.129879</td>
          <td>27.186182</td>
          <td>19.578435</td>
          <td>22.426960</td>
          <td>19.114104</td>
          <td>27.104605</td>
          <td>24.724159</td>
          <td>0.026826</td>
          <td>21.161601</td>
          <td>0.005293</td>
          <td>21.973613</td>
          <td>0.006202</td>
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
          <td>22.731029</td>
          <td>21.263385</td>
          <td>24.307600</td>
          <td>20.195966</td>
          <td>25.049024</td>
          <td>17.078163</td>
          <td>23.151768</td>
          <td>0.007999</td>
          <td>16.740117</td>
          <td>0.005000</td>
          <td>25.610786</td>
          <td>0.100311</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.383172</td>
          <td>19.377542</td>
          <td>24.129939</td>
          <td>21.862563</td>
          <td>23.682862</td>
          <td>20.942201</td>
          <td>18.876562</td>
          <td>0.005001</td>
          <td>25.408025</td>
          <td>0.083906</td>
          <td>20.136921</td>
          <td>0.005046</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.524072</td>
          <td>23.448332</td>
          <td>28.224900</td>
          <td>21.785689</td>
          <td>23.703352</td>
          <td>27.644350</td>
          <td>25.791048</td>
          <td>0.069235</td>
          <td>25.487115</td>
          <td>0.089971</td>
          <td>21.235783</td>
          <td>0.005335</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.386311</td>
          <td>22.258534</td>
          <td>20.429555</td>
          <td>24.872637</td>
          <td>22.638585</td>
          <td>23.110822</td>
          <td>21.615761</td>
          <td>0.005226</td>
          <td>24.271212</td>
          <td>0.030541</td>
          <td>23.946368</td>
          <td>0.022965</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.943640</td>
          <td>23.713548</td>
          <td>18.121158</td>
          <td>24.098013</td>
          <td>26.511160</td>
          <td>22.754504</td>
          <td>25.526597</td>
          <td>0.054719</td>
          <td>26.608449</td>
          <td>0.235632</td>
          <td>23.229244</td>
          <td>0.012661</td>
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
          <td>27.627007</td>
          <td>21.417165</td>
          <td>21.356735</td>
          <td>26.319741</td>
          <td>22.997447</td>
          <td>26.792633</td>
          <td>24.199080</td>
          <td>0.181825</td>
          <td>20.818547</td>
          <td>0.008830</td>
          <td>24.183636</td>
          <td>0.164816</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.792600</td>
          <td>30.121404</td>
          <td>26.723142</td>
          <td>19.833481</td>
          <td>21.069211</td>
          <td>21.759735</td>
          <td>21.545425</td>
          <td>0.017746</td>
          <td>26.271266</td>
          <td>0.764566</td>
          <td>19.006113</td>
          <td>0.005222</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.092918</td>
          <td>25.202024</td>
          <td>20.700190</td>
          <td>17.928803</td>
          <td>24.485646</td>
          <td>21.437772</td>
          <td>26.711742</td>
          <td>1.134771</td>
          <td>22.862629</td>
          <td>0.047279</td>
          <td>20.896041</td>
          <td>0.009920</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.972737</td>
          <td>24.344353</td>
          <td>27.012531</td>
          <td>21.951360</td>
          <td>22.810209</td>
          <td>23.824376</td>
          <td>25.830594</td>
          <td>0.649571</td>
          <td>22.569605</td>
          <td>0.036417</td>
          <td>23.465223</td>
          <td>0.088251</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.129879</td>
          <td>27.186182</td>
          <td>19.578435</td>
          <td>22.426960</td>
          <td>19.114104</td>
          <td>27.104605</td>
          <td>24.603596</td>
          <td>0.254854</td>
          <td>21.164385</td>
          <td>0.011180</td>
          <td>21.981095</td>
          <td>0.023670</td>
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
          <td>22.731029</td>
          <td>21.263385</td>
          <td>24.307600</td>
          <td>20.195966</td>
          <td>25.049024</td>
          <td>17.078163</td>
          <td>23.122272</td>
          <td>0.071180</td>
          <td>16.737381</td>
          <td>0.005003</td>
          <td>25.760029</td>
          <td>0.576074</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.383172</td>
          <td>19.377542</td>
          <td>24.129939</td>
          <td>21.862563</td>
          <td>23.682862</td>
          <td>20.942201</td>
          <td>18.879650</td>
          <td>0.005211</td>
          <td>25.301304</td>
          <td>0.379487</td>
          <td>20.126640</td>
          <td>0.006545</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.524072</td>
          <td>23.448332</td>
          <td>28.224900</td>
          <td>21.785689</td>
          <td>23.703352</td>
          <td>27.644350</td>
          <td>25.124739</td>
          <td>0.386451</td>
          <td>26.239018</td>
          <td>0.748386</td>
          <td>21.230300</td>
          <td>0.012671</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.386311</td>
          <td>22.258534</td>
          <td>20.429555</td>
          <td>24.872637</td>
          <td>22.638585</td>
          <td>23.110822</td>
          <td>21.615649</td>
          <td>0.018832</td>
          <td>24.152159</td>
          <td>0.147250</td>
          <td>23.895978</td>
          <td>0.128662</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.943640</td>
          <td>23.713548</td>
          <td>18.121158</td>
          <td>24.098013</td>
          <td>26.511160</td>
          <td>22.754504</td>
          <td>24.536875</td>
          <td>0.241234</td>
          <td>26.225066</td>
          <td>0.741459</td>
          <td>23.227856</td>
          <td>0.071534</td>
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


