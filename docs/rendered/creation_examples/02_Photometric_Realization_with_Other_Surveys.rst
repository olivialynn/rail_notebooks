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
          <td>31.813226</td>
          <td>23.152375</td>
          <td>22.937234</td>
          <td>25.527429</td>
          <td>21.600239</td>
          <td>25.144262</td>
          <td>28.310749</td>
          <td>26.365443</td>
          <td>16.049776</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.130364</td>
          <td>20.507583</td>
          <td>20.623872</td>
          <td>20.379343</td>
          <td>23.098461</td>
          <td>20.166050</td>
          <td>21.511059</td>
          <td>26.519508</td>
          <td>19.210435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.378073</td>
          <td>26.860255</td>
          <td>22.163983</td>
          <td>21.073619</td>
          <td>26.539161</td>
          <td>20.882043</td>
          <td>20.727709</td>
          <td>19.772838</td>
          <td>25.244209</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.254223</td>
          <td>22.885498</td>
          <td>28.835636</td>
          <td>25.332432</td>
          <td>26.570082</td>
          <td>21.080235</td>
          <td>21.783290</td>
          <td>25.246988</td>
          <td>21.369466</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.025134</td>
          <td>18.954663</td>
          <td>12.721625</td>
          <td>29.101327</td>
          <td>22.559854</td>
          <td>17.919910</td>
          <td>19.994708</td>
          <td>25.756843</td>
          <td>21.831179</td>
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
          <td>27.399571</td>
          <td>28.983448</td>
          <td>25.730283</td>
          <td>25.811354</td>
          <td>25.838950</td>
          <td>25.249170</td>
          <td>28.247106</td>
          <td>18.733034</td>
          <td>23.119203</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.564151</td>
          <td>23.062035</td>
          <td>27.089635</td>
          <td>22.823330</td>
          <td>22.246812</td>
          <td>25.574861</td>
          <td>19.752352</td>
          <td>22.580897</td>
          <td>20.256470</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.033099</td>
          <td>25.088493</td>
          <td>29.900230</td>
          <td>27.969394</td>
          <td>23.128210</td>
          <td>27.972877</td>
          <td>29.622850</td>
          <td>25.780574</td>
          <td>20.218044</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.885065</td>
          <td>16.522201</td>
          <td>23.985289</td>
          <td>29.772109</td>
          <td>30.423127</td>
          <td>26.570650</td>
          <td>24.643257</td>
          <td>22.885739</td>
          <td>23.358424</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.411669</td>
          <td>17.402272</td>
          <td>23.103247</td>
          <td>23.254508</td>
          <td>21.192160</td>
          <td>21.813342</td>
          <td>26.331417</td>
          <td>23.458665</td>
          <td>18.395915</td>
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
          <td>23.141202</td>
          <td>0.008578</td>
          <td>22.936134</td>
          <td>0.007087</td>
          <td>25.393891</td>
          <td>0.075270</td>
          <td>21.603113</td>
          <td>0.006951</td>
          <td>24.843223</td>
          <td>0.195513</td>
          <td>28.310749</td>
          <td>26.365443</td>
          <td>16.049776</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.354919</td>
          <td>0.147104</td>
          <td>20.510937</td>
          <td>0.005074</td>
          <td>20.628678</td>
          <td>0.005056</td>
          <td>20.380926</td>
          <td>0.005088</td>
          <td>23.083720</td>
          <td>0.018866</td>
          <td>20.165485</td>
          <td>0.005870</td>
          <td>21.511059</td>
          <td>26.519508</td>
          <td>19.210435</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.377789</td>
          <td>0.005457</td>
          <td>26.833456</td>
          <td>0.184112</td>
          <td>22.164787</td>
          <td>0.005621</td>
          <td>21.069549</td>
          <td>0.005258</td>
          <td>26.498152</td>
          <td>0.356168</td>
          <td>20.887492</td>
          <td>0.007645</td>
          <td>20.727709</td>
          <td>19.772838</td>
          <td>25.244209</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.244692</td>
          <td>0.133815</td>
          <td>22.880979</td>
          <td>0.007473</td>
          <td>29.481997</td>
          <td>1.138235</td>
          <td>25.354894</td>
          <td>0.072718</td>
          <td>26.380263</td>
          <td>0.324488</td>
          <td>21.078863</td>
          <td>0.008486</td>
          <td>21.783290</td>
          <td>25.246988</td>
          <td>21.369466</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.033538</td>
          <td>0.009422</td>
          <td>18.949946</td>
          <td>0.005011</td>
          <td>12.717192</td>
          <td>0.005000</td>
          <td>28.547707</td>
          <td>0.906674</td>
          <td>22.566257</td>
          <td>0.012453</td>
          <td>17.918859</td>
          <td>0.005029</td>
          <td>19.994708</td>
          <td>25.756843</td>
          <td>21.831179</td>
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
          <td>29.675250</td>
          <td>2.351403</td>
          <td>29.424280</td>
          <td>1.194821</td>
          <td>25.726936</td>
          <td>0.061918</td>
          <td>25.801360</td>
          <td>0.107698</td>
          <td>25.952797</td>
          <td>0.229213</td>
          <td>24.817144</td>
          <td>0.191265</td>
          <td>28.247106</td>
          <td>18.733034</td>
          <td>23.119203</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.559995</td>
          <td>0.013503</td>
          <td>23.056599</td>
          <td>0.008179</td>
          <td>27.405054</td>
          <td>0.262736</td>
          <td>22.845095</td>
          <td>0.009078</td>
          <td>22.246764</td>
          <td>0.009913</td>
          <td>25.077647</td>
          <td>0.237735</td>
          <td>19.752352</td>
          <td>22.580897</td>
          <td>20.256470</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.031358</td>
          <td>0.005288</td>
          <td>25.140388</td>
          <td>0.041913</td>
          <td>29.481038</td>
          <td>1.137612</td>
          <td>28.345702</td>
          <td>0.796971</td>
          <td>23.166742</td>
          <td>0.020236</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.622850</td>
          <td>25.780574</td>
          <td>20.218044</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.617346</td>
          <td>0.412729</td>
          <td>16.518705</td>
          <td>0.005001</td>
          <td>23.979993</td>
          <td>0.013726</td>
          <td>27.726559</td>
          <td>0.518865</td>
          <td>27.282789</td>
          <td>0.637890</td>
          <td>26.381427</td>
          <td>0.645869</td>
          <td>24.643257</td>
          <td>22.885739</td>
          <td>23.358424</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.407216</td>
          <td>0.005132</td>
          <td>17.409255</td>
          <td>0.005002</td>
          <td>23.099373</td>
          <td>0.007661</td>
          <td>23.232103</td>
          <td>0.011834</td>
          <td>21.183927</td>
          <td>0.006017</td>
          <td>21.808817</td>
          <td>0.014113</td>
          <td>26.331417</td>
          <td>23.458665</td>
          <td>18.395915</td>
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
          <td>31.813226</td>
          <td>23.152375</td>
          <td>22.937234</td>
          <td>25.527429</td>
          <td>21.600239</td>
          <td>25.144262</td>
          <td>28.555409</td>
          <td>0.660814</td>
          <td>26.368782</td>
          <td>0.192862</td>
          <td>16.047695</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.130364</td>
          <td>20.507583</td>
          <td>20.623872</td>
          <td>20.379343</td>
          <td>23.098461</td>
          <td>20.166050</td>
          <td>21.513118</td>
          <td>0.005187</td>
          <td>26.427967</td>
          <td>0.202711</td>
          <td>19.209582</td>
          <td>0.005008</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.378073</td>
          <td>26.860255</td>
          <td>22.163983</td>
          <td>21.073619</td>
          <td>26.539161</td>
          <td>20.882043</td>
          <td>20.727509</td>
          <td>0.005045</td>
          <td>19.779662</td>
          <td>0.005024</td>
          <td>25.234786</td>
          <td>0.071975</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.254223</td>
          <td>22.885498</td>
          <td>28.835636</td>
          <td>25.332432</td>
          <td>26.570082</td>
          <td>21.080235</td>
          <td>21.791907</td>
          <td>0.005310</td>
          <td>25.424673</td>
          <td>0.085149</td>
          <td>21.363422</td>
          <td>0.005420</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.025134</td>
          <td>18.954663</td>
          <td>12.721625</td>
          <td>29.101327</td>
          <td>22.559854</td>
          <td>17.919910</td>
          <td>19.994664</td>
          <td>0.005012</td>
          <td>25.655058</td>
          <td>0.104282</td>
          <td>21.823930</td>
          <td>0.005934</td>
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
          <td>27.399571</td>
          <td>28.983448</td>
          <td>25.730283</td>
          <td>25.811354</td>
          <td>25.838950</td>
          <td>25.249170</td>
          <td>28.786102</td>
          <td>0.772091</td>
          <td>18.733026</td>
          <td>0.005003</td>
          <td>23.148751</td>
          <td>0.011905</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.564151</td>
          <td>23.062035</td>
          <td>27.089635</td>
          <td>22.823330</td>
          <td>22.246812</td>
          <td>25.574861</td>
          <td>19.762382</td>
          <td>0.005008</td>
          <td>22.576917</td>
          <td>0.008114</td>
          <td>20.264417</td>
          <td>0.005057</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.033099</td>
          <td>25.088493</td>
          <td>29.900230</td>
          <td>27.969394</td>
          <td>23.128210</td>
          <td>27.972877</td>
          <td>27.608342</td>
          <td>0.326052</td>
          <td>25.491613</td>
          <td>0.090328</td>
          <td>20.222266</td>
          <td>0.005053</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.885065</td>
          <td>16.522201</td>
          <td>23.985289</td>
          <td>29.772109</td>
          <td>30.423127</td>
          <td>26.570650</td>
          <td>24.669863</td>
          <td>0.025578</td>
          <td>22.885276</td>
          <td>0.009847</td>
          <td>23.352239</td>
          <td>0.013947</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.411669</td>
          <td>17.402272</td>
          <td>23.103247</td>
          <td>23.254508</td>
          <td>21.192160</td>
          <td>21.813342</td>
          <td>26.528016</td>
          <td>0.132285</td>
          <td>23.470025</td>
          <td>0.015340</td>
          <td>18.398250</td>
          <td>0.005002</td>
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
          <td>31.813226</td>
          <td>23.152375</td>
          <td>22.937234</td>
          <td>25.527429</td>
          <td>21.600239</td>
          <td>25.144262</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.004570</td>
          <td>1.195854</td>
          <td>16.042753</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.130364</td>
          <td>20.507583</td>
          <td>20.623872</td>
          <td>20.379343</td>
          <td>23.098461</td>
          <td>20.166050</td>
          <td>21.532269</td>
          <td>0.017551</td>
          <td>26.357306</td>
          <td>0.808910</td>
          <td>19.210639</td>
          <td>0.005320</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.378073</td>
          <td>26.860255</td>
          <td>22.163983</td>
          <td>21.073619</td>
          <td>26.539161</td>
          <td>20.882043</td>
          <td>20.740170</td>
          <td>0.009552</td>
          <td>19.772501</td>
          <td>0.005721</td>
          <td>25.308591</td>
          <td>0.412247</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.254223</td>
          <td>22.885498</td>
          <td>28.835636</td>
          <td>25.332432</td>
          <td>26.570082</td>
          <td>21.080235</td>
          <td>21.770618</td>
          <td>0.021507</td>
          <td>25.367182</td>
          <td>0.399336</td>
          <td>21.348778</td>
          <td>0.013908</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.025134</td>
          <td>18.954663</td>
          <td>12.721625</td>
          <td>29.101327</td>
          <td>22.559854</td>
          <td>17.919910</td>
          <td>19.998684</td>
          <td>0.006476</td>
          <td>25.528780</td>
          <td>0.451671</td>
          <td>21.833871</td>
          <td>0.020836</td>
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
          <td>27.399571</td>
          <td>28.983448</td>
          <td>25.730283</td>
          <td>25.811354</td>
          <td>25.838950</td>
          <td>25.249170</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.732384</td>
          <td>0.005113</td>
          <td>23.175020</td>
          <td>0.068256</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.564151</td>
          <td>23.062035</td>
          <td>27.089635</td>
          <td>22.823330</td>
          <td>22.246812</td>
          <td>25.574861</td>
          <td>19.754247</td>
          <td>0.005984</td>
          <td>22.577288</td>
          <td>0.036667</td>
          <td>20.267499</td>
          <td>0.006937</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.033099</td>
          <td>25.088493</td>
          <td>29.900230</td>
          <td>27.969394</td>
          <td>23.128210</td>
          <td>27.972877</td>
          <td>26.104219</td>
          <td>0.781349</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.227682</td>
          <td>0.006818</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.885065</td>
          <td>16.522201</td>
          <td>23.985289</td>
          <td>29.772109</td>
          <td>30.423127</td>
          <td>26.570650</td>
          <td>24.289378</td>
          <td>0.196239</td>
          <td>22.868734</td>
          <td>0.047537</td>
          <td>23.445567</td>
          <td>0.086734</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.411669</td>
          <td>17.402272</td>
          <td>23.103247</td>
          <td>23.254508</td>
          <td>21.192160</td>
          <td>21.813342</td>
          <td>27.521638</td>
          <td>1.722747</td>
          <td>23.516202</td>
          <td>0.084514</td>
          <td>18.393415</td>
          <td>0.005073</td>
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


