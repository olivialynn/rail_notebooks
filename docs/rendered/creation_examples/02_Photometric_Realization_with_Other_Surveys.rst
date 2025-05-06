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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.611163</td>
          <td>0.014034</td>
          <td>22.955963</td>
          <td>0.007755</td>
          <td>25.755323</td>
          <td>0.063497</td>
          <td>18.275605</td>
          <td>0.005006</td>
          <td>23.032755</td>
          <td>0.018078</td>
          <td>23.673928</td>
          <td>0.070898</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.652732</td>
          <td>0.014487</td>
          <td>24.442809</td>
          <td>0.022789</td>
          <td>20.935519</td>
          <td>0.005089</td>
          <td>24.400892</td>
          <td>0.031230</td>
          <td>22.944914</td>
          <td>0.016809</td>
          <td>27.577153</td>
          <td>1.342819</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.123325</td>
          <td>0.009963</td>
          <td>23.419121</td>
          <td>0.010213</td>
          <td>22.002061</td>
          <td>0.005477</td>
          <td>25.264792</td>
          <td>0.067144</td>
          <td>22.981272</td>
          <td>0.017321</td>
          <td>23.443623</td>
          <td>0.057808</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.024307</td>
          <td>0.019485</td>
          <td>22.934550</td>
          <td>0.007672</td>
          <td>28.258620</td>
          <td>0.510736</td>
          <td>19.621428</td>
          <td>0.005029</td>
          <td>18.627480</td>
          <td>0.005022</td>
          <td>22.616083</td>
          <td>0.027812</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.538513</td>
          <td>0.005155</td>
          <td>22.302334</td>
          <td>0.006040</td>
          <td>24.241882</td>
          <td>0.016947</td>
          <td>12.206120</td>
          <td>0.005000</td>
          <td>27.633217</td>
          <td>0.807610</td>
          <td>22.873164</td>
          <td>0.034862</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.288155</td>
          <td>0.005038</td>
          <td>18.226047</td>
          <td>0.005005</td>
          <td>14.348363</td>
          <td>0.005000</td>
          <td>22.212222</td>
          <td>0.006607</td>
          <td>23.074028</td>
          <td>0.018713</td>
          <td>23.357725</td>
          <td>0.053565</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.176568</td>
          <td>0.622153</td>
          <td>19.208382</td>
          <td>0.005015</td>
          <td>23.751492</td>
          <td>0.011543</td>
          <td>23.090813</td>
          <td>0.010687</td>
          <td>24.885936</td>
          <td>0.091786</td>
          <td>21.781084</td>
          <td>0.013809</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.787971</td>
          <td>0.005803</td>
          <td>24.983914</td>
          <td>0.036505</td>
          <td>20.552241</td>
          <td>0.005050</td>
          <td>20.931456</td>
          <td>0.005207</td>
          <td>24.422864</td>
          <td>0.060966</td>
          <td>23.625255</td>
          <td>0.067908</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.767297</td>
          <td>0.088377</td>
          <td>24.846173</td>
          <td>0.032341</td>
          <td>26.243103</td>
          <td>0.097656</td>
          <td>26.391801</td>
          <td>0.179134</td>
          <td>17.085081</td>
          <td>0.005004</td>
          <td>26.509863</td>
          <td>0.705325</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.165286</td>
          <td>0.010235</td>
          <td>17.798152</td>
          <td>0.005003</td>
          <td>20.731699</td>
          <td>0.005066</td>
          <td>20.699614</td>
          <td>0.005144</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.849755</td>
          <td>0.005524</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>25.048711</td>
          <td>0.035747</td>
          <td>17.298727</td>
          <td>0.005000</td>
          <td>25.781259</td>
          <td>0.116445</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.256579</td>
          <td>0.008501</td>
          <td>24.128414</td>
          <td>0.026927</td>
          <td>24.073397</td>
          <td>0.025658</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.790874</td>
          <td>0.028449</td>
          <td>25.997557</td>
          <td>0.140482</td>
          <td>24.331166</td>
          <td>0.032205</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.177151</td>
          <td>0.040075</td>
          <td>22.211934</td>
          <td>0.006773</td>
          <td>23.024283</td>
          <td>0.010858</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.050272</td>
          <td>0.007573</td>
          <td>22.873256</td>
          <td>0.009767</td>
          <td>16.981573</td>
          <td>0.005000</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.708551</td>
          <td>0.026461</td>
          <td>21.632681</td>
          <td>0.005673</td>
          <td>19.868481</td>
          <td>0.005028</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.323952</td>
          <td>0.005133</td>
          <td>22.328861</td>
          <td>0.007133</td>
          <td>22.775448</td>
          <td>0.009155</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.396415</td>
          <td>0.020176</td>
          <td>26.642422</td>
          <td>0.242342</td>
          <td>24.441495</td>
          <td>0.035518</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.010260</td>
          <td>0.014612</td>
          <td>26.299008</td>
          <td>0.181814</td>
          <td>19.986934</td>
          <td>0.005035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>26.573031</td>
          <td>0.137538</td>
          <td>22.621581</td>
          <td>0.008326</td>
          <td>24.651692</td>
          <td>0.042827</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>25.120593</td>
          <td>0.385212</td>
          <td>17.291139</td>
          <td>0.005008</td>
          <td>25.217236</td>
          <td>0.384210</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.321069</td>
          <td>0.084878</td>
          <td>24.063898</td>
          <td>0.136457</td>
          <td>24.233081</td>
          <td>0.171913</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.963593</td>
          <td>0.340655</td>
          <td>26.979748</td>
          <td>1.179346</td>
          <td>24.714066</td>
          <td>0.257052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.109393</td>
          <td>0.381879</td>
          <td>22.195807</td>
          <td>0.026167</td>
          <td>23.084589</td>
          <td>0.062984</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.000034</td>
          <td>0.063856</td>
          <td>22.832531</td>
          <td>0.046027</td>
          <td>16.969491</td>
          <td>0.005005</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.329187</td>
          <td>0.202919</td>
          <td>21.639373</td>
          <td>0.016242</td>
          <td>19.867732</td>
          <td>0.006006</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.318421</td>
          <td>0.014709</td>
          <td>22.311275</td>
          <td>0.028966</td>
          <td>22.842410</td>
          <td>0.050764</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.558064</td>
          <td>0.245488</td>
          <td>25.927319</td>
          <td>0.604285</td>
          <td>24.366849</td>
          <td>0.192548</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.069634</td>
          <td>0.162857</td>
          <td>25.985893</td>
          <td>0.629673</td>
          <td>19.988855</td>
          <td>0.006232</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>27.514490</td>
          <td>1.717065</td>
          <td>22.637970</td>
          <td>0.038701</td>
          <td>24.963914</td>
          <td>0.314703</td>
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


