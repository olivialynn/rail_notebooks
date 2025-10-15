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
          <td>22.614416</td>
          <td>18.826890</td>
          <td>26.592165</td>
          <td>21.206230</td>
          <td>17.490870</td>
          <td>23.282744</td>
          <td>21.952907</td>
          <td>22.511836</td>
          <td>21.830256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.737047</td>
          <td>22.833850</td>
          <td>17.192817</td>
          <td>23.472554</td>
          <td>17.761677</td>
          <td>22.309995</td>
          <td>23.467833</td>
          <td>20.984046</td>
          <td>20.857111</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.141078</td>
          <td>21.446297</td>
          <td>21.493298</td>
          <td>23.620398</td>
          <td>24.098115</td>
          <td>26.962721</td>
          <td>24.269661</td>
          <td>30.078054</td>
          <td>21.692852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.963079</td>
          <td>20.745423</td>
          <td>26.400639</td>
          <td>18.310814</td>
          <td>25.103137</td>
          <td>20.887099</td>
          <td>20.376781</td>
          <td>19.311500</td>
          <td>25.304974</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.980567</td>
          <td>19.517035</td>
          <td>16.445996</td>
          <td>15.333849</td>
          <td>24.793942</td>
          <td>26.149078</td>
          <td>17.210618</td>
          <td>24.244331</td>
          <td>20.778352</td>
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
          <td>25.400691</td>
          <td>18.800056</td>
          <td>23.684293</td>
          <td>25.225143</td>
          <td>17.880575</td>
          <td>25.831158</td>
          <td>24.442821</td>
          <td>18.153983</td>
          <td>24.355597</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.541731</td>
          <td>19.186046</td>
          <td>25.079840</td>
          <td>26.254442</td>
          <td>24.485454</td>
          <td>22.524136</td>
          <td>15.095111</td>
          <td>22.983077</td>
          <td>20.747952</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.145662</td>
          <td>19.411030</td>
          <td>17.985078</td>
          <td>23.949181</td>
          <td>22.473688</td>
          <td>24.527935</td>
          <td>21.440082</td>
          <td>27.550391</td>
          <td>22.981483</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.714964</td>
          <td>18.552711</td>
          <td>24.617402</td>
          <td>19.008878</td>
          <td>25.111338</td>
          <td>22.323092</td>
          <td>23.594186</td>
          <td>24.740915</td>
          <td>21.912667</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.760151</td>
          <td>19.668979</td>
          <td>26.939160</td>
          <td>22.631577</td>
          <td>25.076192</td>
          <td>25.221830</td>
          <td>17.531661</td>
          <td>19.324495</td>
          <td>16.040670</td>
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
          <td>22.626946</td>
          <td>0.014204</td>
          <td>18.831119</td>
          <td>0.005010</td>
          <td>26.810663</td>
          <td>0.159725</td>
          <td>21.205726</td>
          <td>0.005321</td>
          <td>17.490946</td>
          <td>0.005006</td>
          <td>23.248814</td>
          <td>0.048628</td>
          <td>21.952907</td>
          <td>22.511836</td>
          <td>21.830256</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.916793</td>
          <td>0.100706</td>
          <td>22.842340</td>
          <td>0.007338</td>
          <td>17.191542</td>
          <td>0.005001</td>
          <td>23.473484</td>
          <td>0.014240</td>
          <td>17.759027</td>
          <td>0.005007</td>
          <td>22.291881</td>
          <td>0.021016</td>
          <td>23.467833</td>
          <td>20.984046</td>
          <td>20.857111</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.142384</td>
          <td>0.006317</td>
          <td>21.440251</td>
          <td>0.005278</td>
          <td>21.501857</td>
          <td>0.005214</td>
          <td>23.663029</td>
          <td>0.016595</td>
          <td>24.051145</td>
          <td>0.043834</td>
          <td>28.961344</td>
          <td>2.472140</td>
          <td>24.269661</td>
          <td>30.078054</td>
          <td>21.692852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.965982</td>
          <td>0.018575</td>
          <td>20.752720</td>
          <td>0.005102</td>
          <td>26.281013</td>
          <td>0.100955</td>
          <td>18.314957</td>
          <td>0.005006</td>
          <td>25.134226</td>
          <td>0.114064</td>
          <td>20.889969</td>
          <td>0.007655</td>
          <td>20.376781</td>
          <td>19.311500</td>
          <td>25.304974</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.981881</td>
          <td>0.005028</td>
          <td>19.516652</td>
          <td>0.005021</td>
          <td>16.449538</td>
          <td>0.005001</td>
          <td>15.321392</td>
          <td>0.005000</td>
          <td>24.715878</td>
          <td>0.079017</td>
          <td>26.290161</td>
          <td>0.605906</td>
          <td>17.210618</td>
          <td>24.244331</td>
          <td>20.778352</td>
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
          <td>25.142173</td>
          <td>0.122483</td>
          <td>18.808949</td>
          <td>0.005010</td>
          <td>23.687356</td>
          <td>0.011020</td>
          <td>25.243419</td>
          <td>0.065884</td>
          <td>17.875501</td>
          <td>0.005009</td>
          <td>26.022922</td>
          <td>0.499583</td>
          <td>24.442821</td>
          <td>18.153983</td>
          <td>24.355597</td>
        </tr>
        <tr>
          <th>996</th>
          <td>28.367169</td>
          <td>1.301156</td>
          <td>19.187427</td>
          <td>0.005015</td>
          <td>25.065219</td>
          <td>0.034434</td>
          <td>25.950650</td>
          <td>0.122653</td>
          <td>24.491411</td>
          <td>0.064786</td>
          <td>22.550270</td>
          <td>0.026261</td>
          <td>15.095111</td>
          <td>22.983077</td>
          <td>20.747952</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.147702</td>
          <td>0.010120</td>
          <td>19.418357</td>
          <td>0.005019</td>
          <td>17.979650</td>
          <td>0.005002</td>
          <td>23.934371</td>
          <td>0.020831</td>
          <td>22.486000</td>
          <td>0.011730</td>
          <td>24.498262</td>
          <td>0.145770</td>
          <td>21.440082</td>
          <td>27.550391</td>
          <td>22.981483</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.511973</td>
          <td>0.168195</td>
          <td>18.555707</td>
          <td>0.005007</td>
          <td>24.623897</td>
          <td>0.023411</td>
          <td>19.007743</td>
          <td>0.005013</td>
          <td>25.167160</td>
          <td>0.117382</td>
          <td>22.296692</td>
          <td>0.021103</td>
          <td>23.594186</td>
          <td>24.740915</td>
          <td>21.912667</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.619165</td>
          <td>0.077615</td>
          <td>19.666453</td>
          <td>0.005025</td>
          <td>26.879069</td>
          <td>0.169322</td>
          <td>22.630489</td>
          <td>0.008008</td>
          <td>24.862845</td>
          <td>0.089941</td>
          <td>25.314927</td>
          <td>0.288617</td>
          <td>17.531661</td>
          <td>19.324495</td>
          <td>16.040670</td>
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
          <td>22.614416</td>
          <td>18.826890</td>
          <td>26.592165</td>
          <td>21.206230</td>
          <td>17.490870</td>
          <td>23.282744</td>
          <td>21.951275</td>
          <td>0.005411</td>
          <td>22.502664</td>
          <td>0.007786</td>
          <td>21.842564</td>
          <td>0.005964</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.737047</td>
          <td>22.833850</td>
          <td>17.192817</td>
          <td>23.472554</td>
          <td>17.761677</td>
          <td>22.309995</td>
          <td>23.476970</td>
          <td>0.009792</td>
          <td>20.980413</td>
          <td>0.005212</td>
          <td>20.854941</td>
          <td>0.005169</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.141078</td>
          <td>21.446297</td>
          <td>21.493298</td>
          <td>23.620398</td>
          <td>24.098115</td>
          <td>26.962721</td>
          <td>24.271965</td>
          <td>0.018148</td>
          <td>28.892430</td>
          <td>1.187765</td>
          <td>21.683983</td>
          <td>0.005736</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.963079</td>
          <td>20.745423</td>
          <td>26.400639</td>
          <td>18.310814</td>
          <td>25.103137</td>
          <td>20.887099</td>
          <td>20.376332</td>
          <td>0.005023</td>
          <td>19.311204</td>
          <td>0.005010</td>
          <td>25.275954</td>
          <td>0.074651</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.980567</td>
          <td>19.517035</td>
          <td>16.445996</td>
          <td>15.333849</td>
          <td>24.793942</td>
          <td>26.149078</td>
          <td>17.210234</td>
          <td>0.005000</td>
          <td>24.264274</td>
          <td>0.030354</td>
          <td>20.784473</td>
          <td>0.005148</td>
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
          <td>25.400691</td>
          <td>18.800056</td>
          <td>23.684293</td>
          <td>25.225143</td>
          <td>17.880575</td>
          <td>25.831158</td>
          <td>24.390357</td>
          <td>0.020072</td>
          <td>18.151282</td>
          <td>0.005001</td>
          <td>24.323006</td>
          <td>0.031973</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.541731</td>
          <td>19.186046</td>
          <td>25.079840</td>
          <td>26.254442</td>
          <td>24.485454</td>
          <td>22.524136</td>
          <td>15.098811</td>
          <td>0.005000</td>
          <td>22.983349</td>
          <td>0.010543</td>
          <td>20.741645</td>
          <td>0.005137</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.145662</td>
          <td>19.411030</td>
          <td>17.985078</td>
          <td>23.949181</td>
          <td>22.473688</td>
          <td>24.527935</td>
          <td>21.437538</td>
          <td>0.005163</td>
          <td>27.091558</td>
          <td>0.348256</td>
          <td>23.000024</td>
          <td>0.010670</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.714964</td>
          <td>18.552711</td>
          <td>24.617402</td>
          <td>19.008878</td>
          <td>25.111338</td>
          <td>22.323092</td>
          <td>23.601460</td>
          <td>0.010681</td>
          <td>24.726216</td>
          <td>0.045768</td>
          <td>21.908840</td>
          <td>0.006078</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.760151</td>
          <td>19.668979</td>
          <td>26.939160</td>
          <td>22.631577</td>
          <td>25.076192</td>
          <td>25.221830</td>
          <td>17.544162</td>
          <td>0.005000</td>
          <td>19.329025</td>
          <td>0.005010</td>
          <td>16.036885</td>
          <td>0.005000</td>
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
          <td>22.614416</td>
          <td>18.826890</td>
          <td>26.592165</td>
          <td>21.206230</td>
          <td>17.490870</td>
          <td>23.282744</td>
          <td>21.934364</td>
          <td>0.024796</td>
          <td>22.526743</td>
          <td>0.035056</td>
          <td>21.806542</td>
          <td>0.020352</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.737047</td>
          <td>22.833850</td>
          <td>17.192817</td>
          <td>23.472554</td>
          <td>17.761677</td>
          <td>22.309995</td>
          <td>23.700832</td>
          <td>0.118449</td>
          <td>20.987700</td>
          <td>0.009863</td>
          <td>20.861658</td>
          <td>0.009691</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.141078</td>
          <td>21.446297</td>
          <td>21.493298</td>
          <td>23.620398</td>
          <td>24.098115</td>
          <td>26.962721</td>
          <td>23.784338</td>
          <td>0.127370</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.687596</td>
          <td>0.018390</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.963079</td>
          <td>20.745423</td>
          <td>26.400639</td>
          <td>18.310814</td>
          <td>25.103137</td>
          <td>20.887099</td>
          <td>20.373314</td>
          <td>0.007665</td>
          <td>19.305477</td>
          <td>0.005317</td>
          <td>25.204289</td>
          <td>0.380368</td>
        </tr>
        <tr>
          <th>4</th>
          <td>17.980567</td>
          <td>19.517035</td>
          <td>16.445996</td>
          <td>15.333849</td>
          <td>24.793942</td>
          <td>26.149078</td>
          <td>17.209573</td>
          <td>0.005010</td>
          <td>24.247922</td>
          <td>0.159861</td>
          <td>20.774178</td>
          <td>0.009148</td>
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
          <td>25.400691</td>
          <td>18.800056</td>
          <td>23.684293</td>
          <td>25.225143</td>
          <td>17.880575</td>
          <td>25.831158</td>
          <td>24.211726</td>
          <td>0.183783</td>
          <td>18.149421</td>
          <td>0.005039</td>
          <td>24.327363</td>
          <td>0.186232</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.541731</td>
          <td>19.186046</td>
          <td>25.079840</td>
          <td>26.254442</td>
          <td>24.485454</td>
          <td>22.524136</td>
          <td>15.100955</td>
          <td>0.005000</td>
          <td>22.930810</td>
          <td>0.050242</td>
          <td>20.749211</td>
          <td>0.009003</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.145662</td>
          <td>19.411030</td>
          <td>17.985078</td>
          <td>23.949181</td>
          <td>22.473688</td>
          <td>24.527935</td>
          <td>21.421551</td>
          <td>0.016004</td>
          <td>27.636302</td>
          <td>1.655425</td>
          <td>22.874943</td>
          <td>0.052258</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.714964</td>
          <td>18.552711</td>
          <td>24.617402</td>
          <td>19.008878</td>
          <td>25.111338</td>
          <td>22.323092</td>
          <td>23.513823</td>
          <td>0.100578</td>
          <td>24.626208</td>
          <td>0.220075</td>
          <td>21.923991</td>
          <td>0.022523</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.760151</td>
          <td>19.668979</td>
          <td>26.939160</td>
          <td>22.631577</td>
          <td>25.076192</td>
          <td>25.221830</td>
          <td>17.532580</td>
          <td>0.005018</td>
          <td>19.321393</td>
          <td>0.005326</td>
          <td>16.036522</td>
          <td>0.005001</td>
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


