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
          <td>22.568720</td>
          <td>28.697463</td>
          <td>24.671482</td>
          <td>24.900815</td>
          <td>19.322802</td>
          <td>23.049772</td>
          <td>25.491310</td>
          <td>18.975578</td>
          <td>23.231497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.585075</td>
          <td>28.085153</td>
          <td>22.109686</td>
          <td>22.652616</td>
          <td>20.863719</td>
          <td>30.849204</td>
          <td>14.748341</td>
          <td>24.720811</td>
          <td>21.958286</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.919628</td>
          <td>25.166095</td>
          <td>22.163333</td>
          <td>25.806631</td>
          <td>18.463239</td>
          <td>23.674873</td>
          <td>28.833354</td>
          <td>23.504402</td>
          <td>18.926896</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.403576</td>
          <td>26.437323</td>
          <td>21.991262</td>
          <td>26.993321</td>
          <td>29.473102</td>
          <td>23.456578</td>
          <td>21.801856</td>
          <td>20.447217</td>
          <td>23.510050</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.648665</td>
          <td>16.374126</td>
          <td>25.420486</td>
          <td>19.314718</td>
          <td>18.847720</td>
          <td>18.612176</td>
          <td>25.886727</td>
          <td>21.763207</td>
          <td>23.103899</td>
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
          <td>24.325075</td>
          <td>23.048136</td>
          <td>18.768943</td>
          <td>25.491447</td>
          <td>23.186627</td>
          <td>15.270800</td>
          <td>21.385841</td>
          <td>27.509148</td>
          <td>18.418203</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.066965</td>
          <td>22.164079</td>
          <td>22.884243</td>
          <td>26.048576</td>
          <td>19.684864</td>
          <td>27.493275</td>
          <td>22.079627</td>
          <td>18.208366</td>
          <td>20.479922</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.673804</td>
          <td>24.430655</td>
          <td>29.780900</td>
          <td>24.035717</td>
          <td>21.905437</td>
          <td>27.832137</td>
          <td>27.386056</td>
          <td>26.502380</td>
          <td>20.317746</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.779700</td>
          <td>13.181691</td>
          <td>21.304515</td>
          <td>27.644632</td>
          <td>22.757566</td>
          <td>22.762933</td>
          <td>24.097511</td>
          <td>23.763399</td>
          <td>21.032268</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.272808</td>
          <td>25.312642</td>
          <td>26.069178</td>
          <td>25.641692</td>
          <td>19.845626</td>
          <td>25.878979</td>
          <td>24.726337</td>
          <td>21.229859</td>
          <td>21.569976</td>
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
          <td>22.575083</td>
          <td>0.013657</td>
          <td>28.322704</td>
          <td>0.593441</td>
          <td>24.708407</td>
          <td>0.025187</td>
          <td>24.914977</td>
          <td>0.049226</td>
          <td>19.321692</td>
          <td>0.005056</td>
          <td>23.074695</td>
          <td>0.041668</td>
          <td>25.491310</td>
          <td>18.975578</td>
          <td>23.231497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.634820</td>
          <td>0.032726</td>
          <td>28.272779</td>
          <td>0.572708</td>
          <td>22.107414</td>
          <td>0.005566</td>
          <td>22.651342</td>
          <td>0.008100</td>
          <td>20.868694</td>
          <td>0.005615</td>
          <td>28.476462</td>
          <td>2.048239</td>
          <td>14.748341</td>
          <td>24.720811</td>
          <td>21.958286</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.914240</td>
          <td>0.008785</td>
          <td>25.170943</td>
          <td>0.043061</td>
          <td>22.155702</td>
          <td>0.005612</td>
          <td>25.972519</td>
          <td>0.125003</td>
          <td>18.463245</td>
          <td>0.005017</td>
          <td>23.594078</td>
          <td>0.066058</td>
          <td>28.833354</td>
          <td>23.504402</td>
          <td>18.926896</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.923069</td>
          <td>1.717077</td>
          <td>26.338545</td>
          <td>0.120439</td>
          <td>21.997123</td>
          <td>0.005473</td>
          <td>28.080984</td>
          <td>0.667397</td>
          <td>27.854303</td>
          <td>0.929180</td>
          <td>23.383892</td>
          <td>0.054823</td>
          <td>21.801856</td>
          <td>20.447217</td>
          <td>23.510050</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.079668</td>
          <td>0.580983</td>
          <td>16.372826</td>
          <td>0.005001</td>
          <td>25.425211</td>
          <td>0.047370</td>
          <td>19.318214</td>
          <td>0.005020</td>
          <td>18.850963</td>
          <td>0.005029</td>
          <td>18.609207</td>
          <td>0.005076</td>
          <td>25.886727</td>
          <td>21.763207</td>
          <td>23.103899</td>
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
          <td>24.447107</td>
          <td>0.066724</td>
          <td>23.057456</td>
          <td>0.008183</td>
          <td>18.775892</td>
          <td>0.005005</td>
          <td>25.379188</td>
          <td>0.074298</td>
          <td>23.164720</td>
          <td>0.020201</td>
          <td>15.272141</td>
          <td>0.005001</td>
          <td>21.385841</td>
          <td>27.509148</td>
          <td>18.418203</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.054707</td>
          <td>0.006165</td>
          <td>22.163470</td>
          <td>0.005840</td>
          <td>22.898842</td>
          <td>0.006973</td>
          <td>25.885978</td>
          <td>0.115946</td>
          <td>19.684172</td>
          <td>0.005095</td>
          <td>26.477250</td>
          <td>0.689870</td>
          <td>22.079627</td>
          <td>18.208366</td>
          <td>20.479922</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.805175</td>
          <td>0.215249</td>
          <td>24.441726</td>
          <td>0.022768</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.030350</td>
          <td>0.022615</td>
          <td>21.906836</td>
          <td>0.008058</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.386056</td>
          <td>26.502380</td>
          <td>20.317746</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.765189</td>
          <td>0.015811</td>
          <td>13.175248</td>
          <td>0.005000</td>
          <td>21.300370</td>
          <td>0.005156</td>
          <td>27.805004</td>
          <td>0.549327</td>
          <td>22.765761</td>
          <td>0.014539</td>
          <td>22.785339</td>
          <td>0.032264</td>
          <td>24.097511</td>
          <td>23.763399</td>
          <td>21.032268</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.150331</td>
          <td>0.051388</td>
          <td>25.313480</td>
          <td>0.048851</td>
          <td>26.143478</td>
          <td>0.089474</td>
          <td>25.646934</td>
          <td>0.094072</td>
          <td>19.849957</td>
          <td>0.005122</td>
          <td>25.945450</td>
          <td>0.471662</td>
          <td>24.726337</td>
          <td>21.229859</td>
          <td>21.569976</td>
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
          <td>22.568720</td>
          <td>28.697463</td>
          <td>24.671482</td>
          <td>24.900815</td>
          <td>19.322802</td>
          <td>23.049772</td>
          <td>25.401586</td>
          <td>0.048950</td>
          <td>18.971486</td>
          <td>0.005005</td>
          <td>23.216224</td>
          <td>0.012534</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.585075</td>
          <td>28.085153</td>
          <td>22.109686</td>
          <td>22.652616</td>
          <td>20.863719</td>
          <td>30.849204</td>
          <td>14.754277</td>
          <td>0.005000</td>
          <td>24.659762</td>
          <td>0.043136</td>
          <td>21.954810</td>
          <td>0.006165</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.919628</td>
          <td>25.166095</td>
          <td>22.163333</td>
          <td>25.806631</td>
          <td>18.463239</td>
          <td>23.674873</td>
          <td>29.039039</td>
          <td>0.908147</td>
          <td>23.495526</td>
          <td>0.015665</td>
          <td>18.926870</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.403576</td>
          <td>26.437323</td>
          <td>21.991262</td>
          <td>26.993321</td>
          <td>29.473102</td>
          <td>23.456578</td>
          <td>21.795229</td>
          <td>0.005311</td>
          <td>20.436789</td>
          <td>0.005079</td>
          <td>23.512251</td>
          <td>0.015882</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.648665</td>
          <td>16.374126</td>
          <td>25.420486</td>
          <td>19.314718</td>
          <td>18.847720</td>
          <td>18.612176</td>
          <td>25.775304</td>
          <td>0.068273</td>
          <td>21.770009</td>
          <td>0.005853</td>
          <td>23.100365</td>
          <td>0.011481</td>
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
          <td>24.325075</td>
          <td>23.048136</td>
          <td>18.768943</td>
          <td>25.491447</td>
          <td>23.186627</td>
          <td>15.270800</td>
          <td>21.377093</td>
          <td>0.005146</td>
          <td>28.380378</td>
          <td>0.875291</td>
          <td>18.410894</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.066965</td>
          <td>22.164079</td>
          <td>22.884243</td>
          <td>26.048576</td>
          <td>19.684864</td>
          <td>27.493275</td>
          <td>22.085478</td>
          <td>0.005521</td>
          <td>18.207358</td>
          <td>0.005001</td>
          <td>20.481928</td>
          <td>0.005086</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.673804</td>
          <td>24.430655</td>
          <td>29.780900</td>
          <td>24.035717</td>
          <td>21.905437</td>
          <td>27.832137</td>
          <td>27.290879</td>
          <td>0.252206</td>
          <td>26.344250</td>
          <td>0.188909</td>
          <td>20.315460</td>
          <td>0.005063</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.779700</td>
          <td>13.181691</td>
          <td>21.304515</td>
          <td>27.644632</td>
          <td>22.757566</td>
          <td>22.762933</td>
          <td>24.092131</td>
          <td>0.015621</td>
          <td>23.755452</td>
          <td>0.019482</td>
          <td>21.026184</td>
          <td>0.005230</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.272808</td>
          <td>25.312642</td>
          <td>26.069178</td>
          <td>25.641692</td>
          <td>19.845626</td>
          <td>25.878979</td>
          <td>24.741214</td>
          <td>0.027232</td>
          <td>21.232758</td>
          <td>0.005333</td>
          <td>21.564502</td>
          <td>0.005598</td>
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
          <td>22.568720</td>
          <td>28.697463</td>
          <td>24.671482</td>
          <td>24.900815</td>
          <td>19.322802</td>
          <td>23.049772</td>
          <td>25.521182</td>
          <td>0.521041</td>
          <td>18.975035</td>
          <td>0.005175</td>
          <td>23.299666</td>
          <td>0.076236</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.585075</td>
          <td>28.085153</td>
          <td>22.109686</td>
          <td>22.652616</td>
          <td>20.863719</td>
          <td>30.849204</td>
          <td>14.742577</td>
          <td>0.005000</td>
          <td>24.700919</td>
          <td>0.234168</td>
          <td>21.946604</td>
          <td>0.022970</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.919628</td>
          <td>25.166095</td>
          <td>22.163333</td>
          <td>25.806631</td>
          <td>18.463239</td>
          <td>23.674873</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.641858</td>
          <td>0.094415</td>
          <td>18.925378</td>
          <td>0.005192</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.403576</td>
          <td>26.437323</td>
          <td>21.991262</td>
          <td>26.993321</td>
          <td>29.473102</td>
          <td>23.456578</td>
          <td>21.777737</td>
          <td>0.021639</td>
          <td>20.439258</td>
          <td>0.007168</td>
          <td>23.660998</td>
          <td>0.104826</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.648665</td>
          <td>16.374126</td>
          <td>25.420486</td>
          <td>19.314718</td>
          <td>18.847720</td>
          <td>18.612176</td>
          <td>26.235358</td>
          <td>0.850607</td>
          <td>21.761628</td>
          <td>0.017990</td>
          <td>23.137985</td>
          <td>0.066047</td>
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
          <td>24.325075</td>
          <td>23.048136</td>
          <td>18.768943</td>
          <td>25.491447</td>
          <td>23.186627</td>
          <td>15.270800</td>
          <td>21.395856</td>
          <td>0.015669</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.421979</td>
          <td>0.005077</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.066965</td>
          <td>22.164079</td>
          <td>22.884243</td>
          <td>26.048576</td>
          <td>19.684864</td>
          <td>27.493275</td>
          <td>22.066345</td>
          <td>0.027841</td>
          <td>18.210717</td>
          <td>0.005043</td>
          <td>20.485721</td>
          <td>0.007716</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.673804</td>
          <td>24.430655</td>
          <td>29.780900</td>
          <td>24.035717</td>
          <td>21.905437</td>
          <td>27.832137</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.747684</td>
          <td>0.531218</td>
          <td>20.319761</td>
          <td>0.007103</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.779700</td>
          <td>13.181691</td>
          <td>21.304515</td>
          <td>27.644632</td>
          <td>22.757566</td>
          <td>22.762933</td>
          <td>24.364025</td>
          <td>0.208935</td>
          <td>23.819294</td>
          <td>0.110313</td>
          <td>21.044326</td>
          <td>0.011017</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.272808</td>
          <td>25.312642</td>
          <td>26.069178</td>
          <td>25.641692</td>
          <td>19.845626</td>
          <td>25.878979</td>
          <td>24.460459</td>
          <td>0.226438</td>
          <td>21.220594</td>
          <td>0.011655</td>
          <td>21.582099</td>
          <td>0.016829</td>
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


