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
          <td>23.877382</td>
          <td>19.792399</td>
          <td>23.226065</td>
          <td>17.213891</td>
          <td>29.798057</td>
          <td>26.165595</td>
          <td>19.790818</td>
          <td>23.648728</td>
          <td>19.643259</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.970202</td>
          <td>24.895669</td>
          <td>27.802887</td>
          <td>17.436137</td>
          <td>22.827616</td>
          <td>18.737439</td>
          <td>24.511410</td>
          <td>19.866245</td>
          <td>21.846074</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.664625</td>
          <td>30.180300</td>
          <td>19.937123</td>
          <td>19.784795</td>
          <td>21.399917</td>
          <td>24.569573</td>
          <td>25.357411</td>
          <td>22.823784</td>
          <td>22.456610</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.799993</td>
          <td>26.526448</td>
          <td>28.283818</td>
          <td>23.996395</td>
          <td>23.688279</td>
          <td>25.028363</td>
          <td>24.431313</td>
          <td>25.391947</td>
          <td>26.539654</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.376034</td>
          <td>20.908225</td>
          <td>19.183103</td>
          <td>23.400151</td>
          <td>26.467292</td>
          <td>18.941621</td>
          <td>17.363272</td>
          <td>23.199137</td>
          <td>22.785803</td>
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
          <td>23.750704</td>
          <td>24.898197</td>
          <td>26.535751</td>
          <td>23.327217</td>
          <td>24.265969</td>
          <td>23.054766</td>
          <td>23.198137</td>
          <td>23.437692</td>
          <td>21.905110</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.917304</td>
          <td>18.327282</td>
          <td>23.853631</td>
          <td>21.780209</td>
          <td>21.195567</td>
          <td>21.345696</td>
          <td>21.955963</td>
          <td>20.027180</td>
          <td>22.251165</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.540995</td>
          <td>17.614668</td>
          <td>23.898199</td>
          <td>21.164636</td>
          <td>22.787340</td>
          <td>23.117558</td>
          <td>23.862092</td>
          <td>22.493276</td>
          <td>22.691785</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.457739</td>
          <td>21.938892</td>
          <td>21.579978</td>
          <td>25.287445</td>
          <td>22.740577</td>
          <td>20.631980</td>
          <td>21.354907</td>
          <td>27.370566</td>
          <td>20.506920</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.096413</td>
          <td>19.520792</td>
          <td>20.212736</td>
          <td>23.023496</td>
          <td>22.714584</td>
          <td>25.568496</td>
          <td>21.532498</td>
          <td>30.145800</td>
          <td>18.864711</td>
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
          <td>23.818482</td>
          <td>0.038406</td>
          <td>19.790580</td>
          <td>0.005029</td>
          <td>23.218985</td>
          <td>0.008166</td>
          <td>17.220337</td>
          <td>0.005002</td>
          <td>28.349948</td>
          <td>1.240969</td>
          <td>25.577375</td>
          <td>0.355741</td>
          <td>19.790818</td>
          <td>23.648728</td>
          <td>19.643259</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.961179</td>
          <td>0.006022</td>
          <td>24.924553</td>
          <td>0.034646</td>
          <td>28.098413</td>
          <td>0.453375</td>
          <td>17.430130</td>
          <td>0.005002</td>
          <td>22.857157</td>
          <td>0.015646</td>
          <td>18.739599</td>
          <td>0.005092</td>
          <td>24.511410</td>
          <td>19.866245</td>
          <td>21.846074</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.657769</td>
          <td>0.014543</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.942507</td>
          <td>0.005022</td>
          <td>19.795349</td>
          <td>0.005038</td>
          <td>21.399575</td>
          <td>0.006428</td>
          <td>24.499602</td>
          <td>0.145938</td>
          <td>25.357411</td>
          <td>22.823784</td>
          <td>22.456610</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.815227</td>
          <td>0.016450</td>
          <td>26.598053</td>
          <td>0.150671</td>
          <td>28.549320</td>
          <td>0.629084</td>
          <td>23.966996</td>
          <td>0.021419</td>
          <td>23.635181</td>
          <td>0.030353</td>
          <td>25.653112</td>
          <td>0.377424</td>
          <td>24.431313</td>
          <td>25.391947</td>
          <td>26.539654</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.376405</td>
          <td>0.006826</td>
          <td>20.911738</td>
          <td>0.005128</td>
          <td>19.190094</td>
          <td>0.005009</td>
          <td>23.407427</td>
          <td>0.013521</td>
          <td>26.855527</td>
          <td>0.468442</td>
          <td>18.942661</td>
          <td>0.005125</td>
          <td>17.363272</td>
          <td>23.199137</td>
          <td>22.785803</td>
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
          <td>23.772040</td>
          <td>0.036879</td>
          <td>24.896726</td>
          <td>0.033809</td>
          <td>26.560227</td>
          <td>0.128757</td>
          <td>23.343358</td>
          <td>0.012868</td>
          <td>24.404099</td>
          <td>0.059960</td>
          <td>23.067864</td>
          <td>0.041416</td>
          <td>23.198137</td>
          <td>23.437692</td>
          <td>21.905110</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.882900</td>
          <td>0.017365</td>
          <td>18.322480</td>
          <td>0.005006</td>
          <td>23.827664</td>
          <td>0.012213</td>
          <td>21.783264</td>
          <td>0.005817</td>
          <td>21.200166</td>
          <td>0.006044</td>
          <td>21.348374</td>
          <td>0.010055</td>
          <td>21.955963</td>
          <td>20.027180</td>
          <td>22.251165</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.585740</td>
          <td>0.031363</td>
          <td>17.604345</td>
          <td>0.005003</td>
          <td>23.885648</td>
          <td>0.012761</td>
          <td>21.160002</td>
          <td>0.005299</td>
          <td>22.770234</td>
          <td>0.014591</td>
          <td>23.004233</td>
          <td>0.039146</td>
          <td>23.862092</td>
          <td>22.493276</td>
          <td>22.691785</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.453335</td>
          <td>0.005045</td>
          <td>21.944353</td>
          <td>0.005599</td>
          <td>21.568800</td>
          <td>0.005238</td>
          <td>25.282144</td>
          <td>0.068183</td>
          <td>22.718162</td>
          <td>0.014002</td>
          <td>20.635686</td>
          <td>0.006815</td>
          <td>21.354907</td>
          <td>27.370566</td>
          <td>20.506920</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.100128</td>
          <td>0.005315</td>
          <td>19.519881</td>
          <td>0.005021</td>
          <td>20.212575</td>
          <td>0.005031</td>
          <td>23.029832</td>
          <td>0.010245</td>
          <td>22.696052</td>
          <td>0.013761</td>
          <td>25.761446</td>
          <td>0.410351</td>
          <td>21.532498</td>
          <td>30.145800</td>
          <td>18.864711</td>
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
          <td>23.877382</td>
          <td>19.792399</td>
          <td>23.226065</td>
          <td>17.213891</td>
          <td>29.798057</td>
          <td>26.165595</td>
          <td>19.796071</td>
          <td>0.005008</td>
          <td>23.661799</td>
          <td>0.017993</td>
          <td>19.644670</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.970202</td>
          <td>24.895669</td>
          <td>27.802887</td>
          <td>17.436137</td>
          <td>22.827616</td>
          <td>18.737439</td>
          <td>24.499739</td>
          <td>0.022055</td>
          <td>19.866507</td>
          <td>0.005028</td>
          <td>21.850056</td>
          <td>0.005977</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.664625</td>
          <td>30.180300</td>
          <td>19.937123</td>
          <td>19.784795</td>
          <td>21.399917</td>
          <td>24.569573</td>
          <td>25.389445</td>
          <td>0.048423</td>
          <td>22.821572</td>
          <td>0.009435</td>
          <td>22.456959</td>
          <td>0.007599</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.799993</td>
          <td>26.526448</td>
          <td>28.283818</td>
          <td>23.996395</td>
          <td>23.688279</td>
          <td>25.028363</td>
          <td>24.433320</td>
          <td>0.020826</td>
          <td>25.421687</td>
          <td>0.084925</td>
          <td>26.342718</td>
          <td>0.188665</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.376034</td>
          <td>20.908225</td>
          <td>19.183103</td>
          <td>23.400151</td>
          <td>26.467292</td>
          <td>18.941621</td>
          <td>17.361923</td>
          <td>0.005000</td>
          <td>23.211287</td>
          <td>0.012486</td>
          <td>22.781875</td>
          <td>0.009193</td>
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
          <td>23.750704</td>
          <td>24.898197</td>
          <td>26.535751</td>
          <td>23.327217</td>
          <td>24.265969</td>
          <td>23.054766</td>
          <td>23.204880</td>
          <td>0.008246</td>
          <td>23.411489</td>
          <td>0.014627</td>
          <td>21.907747</td>
          <td>0.006076</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.917304</td>
          <td>18.327282</td>
          <td>23.853631</td>
          <td>21.780209</td>
          <td>21.195567</td>
          <td>21.345696</td>
          <td>21.953512</td>
          <td>0.005413</td>
          <td>20.028060</td>
          <td>0.005037</td>
          <td>22.253119</td>
          <td>0.006893</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.540995</td>
          <td>17.614668</td>
          <td>23.898199</td>
          <td>21.164636</td>
          <td>22.787340</td>
          <td>23.117558</td>
          <td>23.855820</td>
          <td>0.012925</td>
          <td>22.500661</td>
          <td>0.007778</td>
          <td>22.695877</td>
          <td>0.008707</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.457739</td>
          <td>21.938892</td>
          <td>21.579978</td>
          <td>25.287445</td>
          <td>22.740577</td>
          <td>20.631980</td>
          <td>21.353409</td>
          <td>0.005140</td>
          <td>27.101737</td>
          <td>0.351058</td>
          <td>20.509307</td>
          <td>0.005090</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.096413</td>
          <td>19.520792</td>
          <td>20.212736</td>
          <td>23.023496</td>
          <td>22.714584</td>
          <td>25.568496</td>
          <td>21.538876</td>
          <td>0.005196</td>
          <td>28.694576</td>
          <td>1.060261</td>
          <td>18.858422</td>
          <td>0.005004</td>
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
          <td>23.877382</td>
          <td>19.792399</td>
          <td>23.226065</td>
          <td>17.213891</td>
          <td>29.798057</td>
          <td>26.165595</td>
          <td>19.788797</td>
          <td>0.006043</td>
          <td>23.638060</td>
          <td>0.094100</td>
          <td>19.651800</td>
          <td>0.005696</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.970202</td>
          <td>24.895669</td>
          <td>27.802887</td>
          <td>17.436137</td>
          <td>22.827616</td>
          <td>18.737439</td>
          <td>24.676083</td>
          <td>0.270420</td>
          <td>19.855438</td>
          <td>0.005832</td>
          <td>21.877259</td>
          <td>0.021630</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.664625</td>
          <td>30.180300</td>
          <td>19.937123</td>
          <td>19.784795</td>
          <td>21.399917</td>
          <td>24.569573</td>
          <td>25.007679</td>
          <td>0.352703</td>
          <td>22.786932</td>
          <td>0.044193</td>
          <td>22.446668</td>
          <td>0.035682</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.799993</td>
          <td>26.526448</td>
          <td>28.283818</td>
          <td>23.996395</td>
          <td>23.688279</td>
          <td>25.028363</td>
          <td>24.530262</td>
          <td>0.239921</td>
          <td>24.889609</td>
          <td>0.273416</td>
          <td>25.220415</td>
          <td>0.385158</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.376034</td>
          <td>20.908225</td>
          <td>19.183103</td>
          <td>23.400151</td>
          <td>26.467292</td>
          <td>18.941621</td>
          <td>17.364836</td>
          <td>0.005013</td>
          <td>23.156305</td>
          <td>0.061419</td>
          <td>22.779831</td>
          <td>0.048009</td>
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
          <td>23.750704</td>
          <td>24.898197</td>
          <td>26.535751</td>
          <td>23.327217</td>
          <td>24.265969</td>
          <td>23.054766</td>
          <td>23.102175</td>
          <td>0.069922</td>
          <td>23.411114</td>
          <td>0.077013</td>
          <td>21.929600</td>
          <td>0.022633</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.917304</td>
          <td>18.327282</td>
          <td>23.853631</td>
          <td>21.780209</td>
          <td>21.195567</td>
          <td>21.345696</td>
          <td>21.950112</td>
          <td>0.025140</td>
          <td>20.024850</td>
          <td>0.006108</td>
          <td>22.238260</td>
          <td>0.029664</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.540995</td>
          <td>17.614668</td>
          <td>23.898199</td>
          <td>21.164636</td>
          <td>22.787340</td>
          <td>23.117558</td>
          <td>23.718170</td>
          <td>0.120251</td>
          <td>22.494295</td>
          <td>0.034060</td>
          <td>22.710892</td>
          <td>0.045147</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.457739</td>
          <td>21.938892</td>
          <td>21.579978</td>
          <td>25.287445</td>
          <td>22.740577</td>
          <td>20.631980</td>
          <td>21.326662</td>
          <td>0.014808</td>
          <td>26.818377</td>
          <td>1.075159</td>
          <td>20.507114</td>
          <td>0.007805</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.096413</td>
          <td>19.520792</td>
          <td>20.212736</td>
          <td>23.023496</td>
          <td>22.714584</td>
          <td>25.568496</td>
          <td>21.522538</td>
          <td>0.017408</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.871065</td>
          <td>0.005174</td>
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


