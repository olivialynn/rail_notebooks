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
          <td>20.985428</td>
          <td>19.030377</td>
          <td>25.629463</td>
          <td>20.239588</td>
          <td>24.406545</td>
          <td>21.986185</td>
          <td>23.660570</td>
          <td>23.089326</td>
          <td>24.411725</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.187547</td>
          <td>25.639685</td>
          <td>20.256495</td>
          <td>22.590937</td>
          <td>25.980478</td>
          <td>17.893167</td>
          <td>25.261739</td>
          <td>23.521273</td>
          <td>19.884756</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.550897</td>
          <td>25.165897</td>
          <td>20.744101</td>
          <td>21.099819</td>
          <td>27.932242</td>
          <td>17.107983</td>
          <td>25.164975</td>
          <td>25.600567</td>
          <td>21.049071</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.167868</td>
          <td>19.285785</td>
          <td>17.098968</td>
          <td>28.182577</td>
          <td>22.590248</td>
          <td>26.852676</td>
          <td>27.527088</td>
          <td>23.036614</td>
          <td>24.316671</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.361261</td>
          <td>23.560334</td>
          <td>24.995683</td>
          <td>28.554460</td>
          <td>26.107826</td>
          <td>23.963521</td>
          <td>22.983540</td>
          <td>20.609184</td>
          <td>24.431544</td>
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
          <td>26.834177</td>
          <td>16.944374</td>
          <td>19.271985</td>
          <td>17.382830</td>
          <td>19.716834</td>
          <td>22.424523</td>
          <td>26.915366</td>
          <td>22.522088</td>
          <td>23.878191</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.340512</td>
          <td>25.661744</td>
          <td>25.327973</td>
          <td>24.143299</td>
          <td>19.733180</td>
          <td>20.674939</td>
          <td>30.453619</td>
          <td>22.376856</td>
          <td>23.532832</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.035290</td>
          <td>20.576153</td>
          <td>22.674045</td>
          <td>24.079420</td>
          <td>23.325125</td>
          <td>23.210427</td>
          <td>24.607967</td>
          <td>15.651192</td>
          <td>21.383038</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.278444</td>
          <td>17.101780</td>
          <td>21.888512</td>
          <td>24.675073</td>
          <td>28.624174</td>
          <td>22.961223</td>
          <td>20.245742</td>
          <td>28.119484</td>
          <td>19.491090</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.140035</td>
          <td>24.009815</td>
          <td>27.277054</td>
          <td>25.059658</td>
          <td>24.681767</td>
          <td>25.738929</td>
          <td>23.407276</td>
          <td>20.341970</td>
          <td>20.718720</td>
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
          <td>20.974104</td>
          <td>0.006041</td>
          <td>19.035568</td>
          <td>0.005012</td>
          <td>25.593598</td>
          <td>0.055009</td>
          <td>20.236149</td>
          <td>0.005071</td>
          <td>24.434516</td>
          <td>0.061600</td>
          <td>21.978161</td>
          <td>0.016169</td>
          <td>23.660570</td>
          <td>23.089326</td>
          <td>24.411725</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.181394</td>
          <td>0.005351</td>
          <td>25.696480</td>
          <td>0.068570</td>
          <td>20.259249</td>
          <td>0.005033</td>
          <td>22.583503</td>
          <td>0.007809</td>
          <td>26.169169</td>
          <td>0.273800</td>
          <td>17.890465</td>
          <td>0.005028</td>
          <td>25.261739</td>
          <td>23.521273</td>
          <td>19.884756</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.558086</td>
          <td>0.005050</td>
          <td>25.163288</td>
          <td>0.042770</td>
          <td>20.744264</td>
          <td>0.005067</td>
          <td>21.087843</td>
          <td>0.005266</td>
          <td>27.217860</td>
          <td>0.609528</td>
          <td>17.105177</td>
          <td>0.005010</td>
          <td>25.164975</td>
          <td>25.600567</td>
          <td>21.049071</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.168543</td>
          <td>0.005005</td>
          <td>19.283486</td>
          <td>0.005016</td>
          <td>17.096292</td>
          <td>0.005001</td>
          <td>28.727084</td>
          <td>1.011823</td>
          <td>22.579008</td>
          <td>0.012574</td>
          <td>26.827315</td>
          <td>0.868543</td>
          <td>27.527088</td>
          <td>23.036614</td>
          <td>24.316671</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.358639</td>
          <td>0.006781</td>
          <td>23.561371</td>
          <td>0.011272</td>
          <td>24.989381</td>
          <td>0.032207</td>
          <td>29.613263</td>
          <td>1.628603</td>
          <td>25.970436</td>
          <td>0.232588</td>
          <td>23.886413</td>
          <td>0.085528</td>
          <td>22.983540</td>
          <td>20.609184</td>
          <td>24.431544</td>
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
          <td>26.384582</td>
          <td>0.344476</td>
          <td>16.946716</td>
          <td>0.005002</td>
          <td>19.269164</td>
          <td>0.005009</td>
          <td>17.381223</td>
          <td>0.005002</td>
          <td>19.714688</td>
          <td>0.005100</td>
          <td>22.442305</td>
          <td>0.023913</td>
          <td>26.915366</td>
          <td>22.522088</td>
          <td>23.878191</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.355998</td>
          <td>0.011643</td>
          <td>25.652169</td>
          <td>0.065935</td>
          <td>25.337483</td>
          <td>0.043821</td>
          <td>24.103747</td>
          <td>0.024093</td>
          <td>19.735042</td>
          <td>0.005103</td>
          <td>20.677720</td>
          <td>0.006934</td>
          <td>30.453619</td>
          <td>22.376856</td>
          <td>23.532832</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.034966</td>
          <td>0.005029</td>
          <td>20.572207</td>
          <td>0.005080</td>
          <td>22.669228</td>
          <td>0.006385</td>
          <td>24.060581</td>
          <td>0.023212</td>
          <td>23.327518</td>
          <td>0.023219</td>
          <td>23.166346</td>
          <td>0.045196</td>
          <td>24.607967</td>
          <td>15.651192</td>
          <td>21.383038</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.279850</td>
          <td>0.011046</td>
          <td>17.092348</td>
          <td>0.005002</td>
          <td>21.894946</td>
          <td>0.005401</td>
          <td>24.620600</td>
          <td>0.037913</td>
          <td>27.038668</td>
          <td>0.536164</td>
          <td>22.999759</td>
          <td>0.038991</td>
          <td>20.245742</td>
          <td>28.119484</td>
          <td>19.491090</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.123543</td>
          <td>0.021154</td>
          <td>23.987342</td>
          <td>0.015597</td>
          <td>26.911369</td>
          <td>0.174037</td>
          <td>24.923055</td>
          <td>0.049580</td>
          <td>24.589523</td>
          <td>0.070667</td>
          <td>25.596297</td>
          <td>0.361057</td>
          <td>23.407276</td>
          <td>20.341970</td>
          <td>20.718720</td>
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
          <td>20.985428</td>
          <td>19.030377</td>
          <td>25.629463</td>
          <td>20.239588</td>
          <td>24.406545</td>
          <td>21.986185</td>
          <td>23.656724</td>
          <td>0.011117</td>
          <td>23.080852</td>
          <td>0.011316</td>
          <td>24.407471</td>
          <td>0.034461</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.187547</td>
          <td>25.639685</td>
          <td>20.256495</td>
          <td>22.590937</td>
          <td>25.980478</td>
          <td>17.893167</td>
          <td>25.242600</td>
          <td>0.042481</td>
          <td>23.535849</td>
          <td>0.016195</td>
          <td>19.890413</td>
          <td>0.005029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.550897</td>
          <td>25.165897</td>
          <td>20.744101</td>
          <td>21.099819</td>
          <td>27.932242</td>
          <td>17.107983</td>
          <td>25.144262</td>
          <td>0.038918</td>
          <td>25.384869</td>
          <td>0.082206</td>
          <td>21.039676</td>
          <td>0.005236</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.167868</td>
          <td>19.285785</td>
          <td>17.098968</td>
          <td>28.182577</td>
          <td>22.590248</td>
          <td>26.852676</td>
          <td>27.361742</td>
          <td>0.267276</td>
          <td>23.042254</td>
          <td>0.011000</td>
          <td>24.319402</td>
          <td>0.031871</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.361261</td>
          <td>23.560334</td>
          <td>24.995683</td>
          <td>28.554460</td>
          <td>26.107826</td>
          <td>23.963521</td>
          <td>22.983554</td>
          <td>0.007322</td>
          <td>20.603487</td>
          <td>0.005107</td>
          <td>24.468949</td>
          <td>0.036396</td>
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
          <td>26.834177</td>
          <td>16.944374</td>
          <td>19.271985</td>
          <td>17.382830</td>
          <td>19.716834</td>
          <td>22.424523</td>
          <td>27.310346</td>
          <td>0.256269</td>
          <td>22.517812</td>
          <td>0.007850</td>
          <td>23.851202</td>
          <td>0.021149</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.340512</td>
          <td>25.661744</td>
          <td>25.327973</td>
          <td>24.143299</td>
          <td>19.733180</td>
          <td>20.674939</td>
          <td>29.076946</td>
          <td>0.929792</td>
          <td>22.379097</td>
          <td>0.007306</td>
          <td>23.524827</td>
          <td>0.016048</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.035290</td>
          <td>20.576153</td>
          <td>22.674045</td>
          <td>24.079420</td>
          <td>23.325125</td>
          <td>23.210427</td>
          <td>24.583600</td>
          <td>0.023721</td>
          <td>15.656037</td>
          <td>0.005000</td>
          <td>21.390371</td>
          <td>0.005441</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.278444</td>
          <td>17.101780</td>
          <td>21.888512</td>
          <td>24.675073</td>
          <td>28.624174</td>
          <td>22.961223</td>
          <td>20.240301</td>
          <td>0.005018</td>
          <td>27.071426</td>
          <td>0.342770</td>
          <td>19.486707</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.140035</td>
          <td>24.009815</td>
          <td>27.277054</td>
          <td>25.059658</td>
          <td>24.681767</td>
          <td>25.738929</td>
          <td>23.409964</td>
          <td>0.009363</td>
          <td>20.326685</td>
          <td>0.005064</td>
          <td>20.715540</td>
          <td>0.005131</td>
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
          <td>20.985428</td>
          <td>19.030377</td>
          <td>25.629463</td>
          <td>20.239588</td>
          <td>24.406545</td>
          <td>21.986185</td>
          <td>23.422657</td>
          <td>0.092832</td>
          <td>23.129075</td>
          <td>0.059949</td>
          <td>24.328819</td>
          <td>0.186461</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.187547</td>
          <td>25.639685</td>
          <td>20.256495</td>
          <td>22.590937</td>
          <td>25.980478</td>
          <td>17.893167</td>
          <td>24.848656</td>
          <td>0.310885</td>
          <td>23.612437</td>
          <td>0.092001</td>
          <td>19.895309</td>
          <td>0.006054</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.550897</td>
          <td>25.165897</td>
          <td>20.744101</td>
          <td>21.099819</td>
          <td>27.932242</td>
          <td>17.107983</td>
          <td>25.530013</td>
          <td>0.524415</td>
          <td>24.976143</td>
          <td>0.293281</td>
          <td>21.033289</td>
          <td>0.010929</td>
        </tr>
        <tr>
          <th>3</th>
          <td>16.167868</td>
          <td>19.285785</td>
          <td>17.098968</td>
          <td>28.182577</td>
          <td>22.590248</td>
          <td>26.852676</td>
          <td>27.402016</td>
          <td>1.628696</td>
          <td>23.184572</td>
          <td>0.062983</td>
          <td>23.972116</td>
          <td>0.137429</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.361261</td>
          <td>23.560334</td>
          <td>24.995683</td>
          <td>28.554460</td>
          <td>26.107826</td>
          <td>23.963521</td>
          <td>23.119668</td>
          <td>0.071016</td>
          <td>20.610599</td>
          <td>0.007820</td>
          <td>24.479469</td>
          <td>0.211653</td>
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
          <td>26.834177</td>
          <td>16.944374</td>
          <td>19.271985</td>
          <td>17.382830</td>
          <td>19.716834</td>
          <td>22.424523</td>
          <td>25.790108</td>
          <td>0.631530</td>
          <td>22.529074</td>
          <td>0.035129</td>
          <td>23.817560</td>
          <td>0.120187</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.340512</td>
          <td>25.661744</td>
          <td>25.327973</td>
          <td>24.143299</td>
          <td>19.733180</td>
          <td>20.674939</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.377301</td>
          <td>0.030705</td>
          <td>23.467591</td>
          <td>0.088436</td>
        </tr>
        <tr>
          <th>997</th>
          <td>18.035290</td>
          <td>20.576153</td>
          <td>22.674045</td>
          <td>24.079420</td>
          <td>23.325125</td>
          <td>23.210427</td>
          <td>24.627407</td>
          <td>0.259877</td>
          <td>15.656365</td>
          <td>0.005000</td>
          <td>21.383534</td>
          <td>0.014301</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.278444</td>
          <td>17.101780</td>
          <td>21.888512</td>
          <td>24.675073</td>
          <td>28.624174</td>
          <td>22.961223</td>
          <td>20.253792</td>
          <td>0.007217</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.488193</td>
          <td>0.005523</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.140035</td>
          <td>24.009815</td>
          <td>27.277054</td>
          <td>25.059658</td>
          <td>24.681767</td>
          <td>25.738929</td>
          <td>23.283778</td>
          <td>0.082127</td>
          <td>20.352298</td>
          <td>0.006890</td>
          <td>20.738178</td>
          <td>0.008940</td>
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


