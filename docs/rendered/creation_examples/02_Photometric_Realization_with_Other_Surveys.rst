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
          <td>16.170490</td>
          <td>19.495027</td>
          <td>26.543577</td>
          <td>18.744691</td>
          <td>21.884808</td>
          <td>23.200052</td>
          <td>24.825488</td>
          <td>20.434695</td>
          <td>21.943730</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.756465</td>
          <td>21.075726</td>
          <td>23.284064</td>
          <td>21.754085</td>
          <td>24.760491</td>
          <td>22.925225</td>
          <td>18.917680</td>
          <td>25.883875</td>
          <td>24.078144</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.738709</td>
          <td>18.273582</td>
          <td>18.679224</td>
          <td>26.177051</td>
          <td>24.279108</td>
          <td>20.166080</td>
          <td>21.032236</td>
          <td>30.211982</td>
          <td>18.982422</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.777915</td>
          <td>24.051914</td>
          <td>20.681607</td>
          <td>26.151080</td>
          <td>23.667326</td>
          <td>22.134876</td>
          <td>19.656592</td>
          <td>19.748254</td>
          <td>22.927670</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.633448</td>
          <td>21.311897</td>
          <td>20.780894</td>
          <td>20.521543</td>
          <td>26.890780</td>
          <td>24.061688</td>
          <td>24.694923</td>
          <td>17.646894</td>
          <td>22.616498</td>
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
          <td>25.844432</td>
          <td>21.641780</td>
          <td>24.264926</td>
          <td>27.205050</td>
          <td>22.936660</td>
          <td>23.849925</td>
          <td>24.902073</td>
          <td>22.281419</td>
          <td>19.436301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.353913</td>
          <td>23.344318</td>
          <td>21.594489</td>
          <td>23.566514</td>
          <td>21.571965</td>
          <td>26.333888</td>
          <td>20.183763</td>
          <td>25.723953</td>
          <td>29.619634</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.758667</td>
          <td>17.858769</td>
          <td>18.912956</td>
          <td>22.673854</td>
          <td>22.626912</td>
          <td>25.239083</td>
          <td>21.209717</td>
          <td>22.137215</td>
          <td>27.303446</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.674147</td>
          <td>21.192936</td>
          <td>22.287384</td>
          <td>25.592987</td>
          <td>19.804111</td>
          <td>22.460785</td>
          <td>22.503007</td>
          <td>24.715585</td>
          <td>22.135832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.605118</td>
          <td>20.833110</td>
          <td>25.641287</td>
          <td>24.129281</td>
          <td>24.032803</td>
          <td>20.388470</td>
          <td>23.917378</td>
          <td>20.837774</td>
          <td>22.192705</td>
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
          <td>16.170735</td>
          <td>0.005005</td>
          <td>19.495974</td>
          <td>0.005021</td>
          <td>26.478290</td>
          <td>0.119921</td>
          <td>18.739748</td>
          <td>0.005010</td>
          <td>21.881678</td>
          <td>0.007949</td>
          <td>23.221633</td>
          <td>0.047469</td>
          <td>24.825488</td>
          <td>20.434695</td>
          <td>21.943730</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.757391</td>
          <td>0.008073</td>
          <td>21.076853</td>
          <td>0.005162</td>
          <td>23.258359</td>
          <td>0.008350</td>
          <td>21.757761</td>
          <td>0.005784</td>
          <td>24.773813</td>
          <td>0.083160</td>
          <td>22.979841</td>
          <td>0.038309</td>
          <td>18.917680</td>
          <td>25.883875</td>
          <td>24.078144</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.781361</td>
          <td>0.037181</td>
          <td>18.285272</td>
          <td>0.005006</td>
          <td>18.673808</td>
          <td>0.005005</td>
          <td>26.277466</td>
          <td>0.162530</td>
          <td>24.301510</td>
          <td>0.054743</td>
          <td>20.170904</td>
          <td>0.005877</td>
          <td>21.032236</td>
          <td>30.211982</td>
          <td>18.982422</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.779718</td>
          <td>0.008166</td>
          <td>24.073131</td>
          <td>0.016719</td>
          <td>20.682822</td>
          <td>0.005061</td>
          <td>26.033371</td>
          <td>0.131769</td>
          <td>23.662771</td>
          <td>0.031097</td>
          <td>22.136503</td>
          <td>0.018431</td>
          <td>19.656592</td>
          <td>19.748254</td>
          <td>22.927670</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.641899</td>
          <td>0.032928</td>
          <td>21.313542</td>
          <td>0.005230</td>
          <td>20.779055</td>
          <td>0.005070</td>
          <td>20.517548</td>
          <td>0.005108</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.959200</td>
          <td>0.091185</td>
          <td>24.694923</td>
          <td>17.646894</td>
          <td>22.616498</td>
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
          <td>25.810297</td>
          <td>0.216169</td>
          <td>21.630865</td>
          <td>0.005370</td>
          <td>24.244470</td>
          <td>0.016983</td>
          <td>27.058459</td>
          <td>0.310632</td>
          <td>22.962777</td>
          <td>0.017058</td>
          <td>23.956334</td>
          <td>0.090955</td>
          <td>24.902073</td>
          <td>22.281419</td>
          <td>19.436301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.390594</td>
          <td>0.011931</td>
          <td>23.357617</td>
          <td>0.009804</td>
          <td>21.595891</td>
          <td>0.005248</td>
          <td>23.585484</td>
          <td>0.015577</td>
          <td>21.570727</td>
          <td>0.006858</td>
          <td>26.259257</td>
          <td>0.592801</td>
          <td>20.183763</td>
          <td>25.723953</td>
          <td>29.619634</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.762756</td>
          <td>0.005775</td>
          <td>17.855675</td>
          <td>0.005004</td>
          <td>18.919154</td>
          <td>0.005006</td>
          <td>22.670612</td>
          <td>0.008187</td>
          <td>22.607278</td>
          <td>0.012848</td>
          <td>25.423571</td>
          <td>0.314945</td>
          <td>21.209717</td>
          <td>22.137215</td>
          <td>27.303446</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.659591</td>
          <td>0.033438</td>
          <td>21.189378</td>
          <td>0.005191</td>
          <td>22.288746</td>
          <td>0.005758</td>
          <td>25.560445</td>
          <td>0.087183</td>
          <td>19.797450</td>
          <td>0.005113</td>
          <td>22.437155</td>
          <td>0.023807</td>
          <td>22.503007</td>
          <td>24.715585</td>
          <td>22.135832</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.609309</td>
          <td>0.032010</td>
          <td>20.836646</td>
          <td>0.005115</td>
          <td>25.614676</td>
          <td>0.056048</td>
          <td>24.157535</td>
          <td>0.025243</td>
          <td>23.945180</td>
          <td>0.039904</td>
          <td>20.383549</td>
          <td>0.006228</td>
          <td>23.917378</td>
          <td>20.837774</td>
          <td>22.192705</td>
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
          <td>16.170490</td>
          <td>19.495027</td>
          <td>26.543577</td>
          <td>18.744691</td>
          <td>21.884808</td>
          <td>23.200052</td>
          <td>24.829983</td>
          <td>0.029448</td>
          <td>20.438590</td>
          <td>0.005079</td>
          <td>21.950288</td>
          <td>0.006156</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.756465</td>
          <td>21.075726</td>
          <td>23.284064</td>
          <td>21.754085</td>
          <td>24.760491</td>
          <td>22.925225</td>
          <td>18.904803</td>
          <td>0.005002</td>
          <td>25.816393</td>
          <td>0.120065</td>
          <td>24.074997</td>
          <td>0.025694</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.738709</td>
          <td>18.273582</td>
          <td>18.679224</td>
          <td>26.177051</td>
          <td>24.279108</td>
          <td>20.166080</td>
          <td>21.026758</td>
          <td>0.005077</td>
          <td>30.187314</td>
          <td>2.194248</td>
          <td>18.976949</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.777915</td>
          <td>24.051914</td>
          <td>20.681607</td>
          <td>26.151080</td>
          <td>23.667326</td>
          <td>22.134876</td>
          <td>19.654276</td>
          <td>0.005006</td>
          <td>19.740230</td>
          <td>0.005022</td>
          <td>22.909265</td>
          <td>0.010010</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.633448</td>
          <td>21.311897</td>
          <td>20.780894</td>
          <td>20.521543</td>
          <td>26.890780</td>
          <td>24.061688</td>
          <td>24.708039</td>
          <td>0.026449</td>
          <td>17.652039</td>
          <td>0.005000</td>
          <td>22.618044</td>
          <td>0.008309</td>
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
          <td>25.844432</td>
          <td>21.641780</td>
          <td>24.264926</td>
          <td>27.205050</td>
          <td>22.936660</td>
          <td>23.849925</td>
          <td>24.859588</td>
          <td>0.030228</td>
          <td>22.282420</td>
          <td>0.006983</td>
          <td>19.430093</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.353913</td>
          <td>23.344318</td>
          <td>21.594489</td>
          <td>23.566514</td>
          <td>21.571965</td>
          <td>26.333888</td>
          <td>20.185360</td>
          <td>0.005017</td>
          <td>25.805318</td>
          <td>0.118913</td>
          <td>28.017597</td>
          <td>0.689607</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.758667</td>
          <td>17.858769</td>
          <td>18.912956</td>
          <td>22.673854</td>
          <td>22.626912</td>
          <td>25.239083</td>
          <td>21.207008</td>
          <td>0.005108</td>
          <td>22.131258</td>
          <td>0.006557</td>
          <td>27.714640</td>
          <td>0.557616</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.674147</td>
          <td>21.192936</td>
          <td>22.287384</td>
          <td>25.592987</td>
          <td>19.804111</td>
          <td>22.460785</td>
          <td>22.509455</td>
          <td>0.006079</td>
          <td>24.648820</td>
          <td>0.042717</td>
          <td>22.140214</td>
          <td>0.006579</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.605118</td>
          <td>20.833110</td>
          <td>25.641287</td>
          <td>24.129281</td>
          <td>24.032803</td>
          <td>20.388470</td>
          <td>23.907075</td>
          <td>0.013455</td>
          <td>20.845158</td>
          <td>0.005166</td>
          <td>22.195078</td>
          <td>0.006725</td>
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
          <td>16.170490</td>
          <td>19.495027</td>
          <td>26.543577</td>
          <td>18.744691</td>
          <td>21.884808</td>
          <td>23.200052</td>
          <td>25.128197</td>
          <td>0.387488</td>
          <td>20.436497</td>
          <td>0.007158</td>
          <td>21.961375</td>
          <td>0.023267</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.756465</td>
          <td>21.075726</td>
          <td>23.284064</td>
          <td>21.754085</td>
          <td>24.760491</td>
          <td>22.925225</td>
          <td>18.920904</td>
          <td>0.005228</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.157505</td>
          <td>0.161177</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.738709</td>
          <td>18.273582</td>
          <td>18.679224</td>
          <td>26.177051</td>
          <td>24.279108</td>
          <td>20.166080</td>
          <td>21.055536</td>
          <td>0.011966</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.986601</td>
          <td>0.005214</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.777915</td>
          <td>24.051914</td>
          <td>20.681607</td>
          <td>26.151080</td>
          <td>23.667326</td>
          <td>22.134876</td>
          <td>19.654379</td>
          <td>0.005830</td>
          <td>19.745547</td>
          <td>0.005688</td>
          <td>22.963930</td>
          <td>0.056570</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.633448</td>
          <td>21.311897</td>
          <td>20.780894</td>
          <td>20.521543</td>
          <td>26.890780</td>
          <td>24.061688</td>
          <td>24.766247</td>
          <td>0.290947</td>
          <td>17.650001</td>
          <td>0.005015</td>
          <td>22.636606</td>
          <td>0.042255</td>
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
          <td>25.844432</td>
          <td>21.641780</td>
          <td>24.264926</td>
          <td>27.205050</td>
          <td>22.936660</td>
          <td>23.849925</td>
          <td>25.010052</td>
          <td>0.353361</td>
          <td>22.252007</td>
          <td>0.027492</td>
          <td>19.433921</td>
          <td>0.005476</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.353913</td>
          <td>23.344318</td>
          <td>21.594489</td>
          <td>23.566514</td>
          <td>21.571965</td>
          <td>26.333888</td>
          <td>20.180808</td>
          <td>0.006978</td>
          <td>26.708343</td>
          <td>1.007314</td>
          <td>25.029440</td>
          <td>0.331564</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.758667</td>
          <td>17.858769</td>
          <td>18.912956</td>
          <td>22.673854</td>
          <td>22.626912</td>
          <td>25.239083</td>
          <td>21.207352</td>
          <td>0.013458</td>
          <td>22.117147</td>
          <td>0.024426</td>
          <td>28.534387</td>
          <td>2.501221</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.674147</td>
          <td>21.192936</td>
          <td>22.287384</td>
          <td>25.592987</td>
          <td>19.804111</td>
          <td>22.460785</td>
          <td>22.536824</td>
          <td>0.042263</td>
          <td>24.929040</td>
          <td>0.282317</td>
          <td>22.116665</td>
          <td>0.026650</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.605118</td>
          <td>20.833110</td>
          <td>25.641287</td>
          <td>24.129281</td>
          <td>24.032803</td>
          <td>20.388470</td>
          <td>24.049882</td>
          <td>0.160129</td>
          <td>20.847666</td>
          <td>0.008994</td>
          <td>22.147838</td>
          <td>0.027391</td>
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


