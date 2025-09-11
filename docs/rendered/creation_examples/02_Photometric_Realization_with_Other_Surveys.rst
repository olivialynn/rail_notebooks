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
          <td>18.613817</td>
          <td>20.302193</td>
          <td>26.468342</td>
          <td>21.589165</td>
          <td>22.142546</td>
          <td>27.537869</td>
          <td>20.408834</td>
          <td>21.231463</td>
          <td>22.401578</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.065312</td>
          <td>27.905767</td>
          <td>26.454396</td>
          <td>22.949271</td>
          <td>22.287529</td>
          <td>25.864854</td>
          <td>23.948470</td>
          <td>22.064442</td>
          <td>23.318529</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.240402</td>
          <td>23.694273</td>
          <td>19.837737</td>
          <td>23.922312</td>
          <td>28.528831</td>
          <td>23.596148</td>
          <td>23.209253</td>
          <td>24.228420</td>
          <td>21.604453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.556388</td>
          <td>19.651837</td>
          <td>22.899755</td>
          <td>22.766546</td>
          <td>32.297729</td>
          <td>20.656157</td>
          <td>20.935379</td>
          <td>25.070125</td>
          <td>27.930531</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.064831</td>
          <td>24.918750</td>
          <td>27.047703</td>
          <td>20.524345</td>
          <td>26.538727</td>
          <td>27.395244</td>
          <td>25.336607</td>
          <td>22.378670</td>
          <td>21.426005</td>
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
          <td>26.288087</td>
          <td>25.770230</td>
          <td>24.177021</td>
          <td>22.048512</td>
          <td>20.830878</td>
          <td>14.278012</td>
          <td>19.866823</td>
          <td>25.280302</td>
          <td>22.494929</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.722594</td>
          <td>23.424550</td>
          <td>20.434098</td>
          <td>20.804492</td>
          <td>23.070999</td>
          <td>27.281512</td>
          <td>29.171135</td>
          <td>18.452979</td>
          <td>24.681777</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.825124</td>
          <td>19.131364</td>
          <td>17.835537</td>
          <td>23.750945</td>
          <td>26.839706</td>
          <td>24.413093</td>
          <td>21.119441</td>
          <td>26.043956</td>
          <td>27.815431</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.492599</td>
          <td>24.992350</td>
          <td>25.407802</td>
          <td>25.422464</td>
          <td>18.668900</td>
          <td>24.670891</td>
          <td>22.488210</td>
          <td>25.109076</td>
          <td>24.990094</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.715990</td>
          <td>23.954201</td>
          <td>24.862240</td>
          <td>19.840170</td>
          <td>25.834966</td>
          <td>21.038619</td>
          <td>33.230911</td>
          <td>18.066318</td>
          <td>22.583690</td>
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
          <td>18.613188</td>
          <td>0.005054</td>
          <td>20.307289</td>
          <td>0.005056</td>
          <td>26.660325</td>
          <td>0.140388</td>
          <td>21.590555</td>
          <td>0.005599</td>
          <td>22.141818</td>
          <td>0.009259</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.408834</td>
          <td>21.231463</td>
          <td>22.401578</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.119949</td>
          <td>0.050033</td>
          <td>29.353176</td>
          <td>1.147903</td>
          <td>26.506523</td>
          <td>0.122898</td>
          <td>22.944553</td>
          <td>0.009675</td>
          <td>22.289469</td>
          <td>0.010203</td>
          <td>26.057154</td>
          <td>0.512329</td>
          <td>23.948470</td>
          <td>22.064442</td>
          <td>23.318529</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.245895</td>
          <td>0.005383</td>
          <td>23.700884</td>
          <td>0.012483</td>
          <td>19.837413</td>
          <td>0.005019</td>
          <td>23.928412</td>
          <td>0.020726</td>
          <td>27.043420</td>
          <td>0.538018</td>
          <td>23.672178</td>
          <td>0.070788</td>
          <td>23.209253</td>
          <td>24.228420</td>
          <td>21.604453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.790894</td>
          <td>0.212703</td>
          <td>19.651405</td>
          <td>0.005025</td>
          <td>22.905783</td>
          <td>0.006994</td>
          <td>22.755729</td>
          <td>0.008598</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.666413</td>
          <td>0.006902</td>
          <td>20.935379</td>
          <td>25.070125</td>
          <td>27.930531</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.079731</td>
          <td>0.116035</td>
          <td>24.895062</td>
          <td>0.033760</td>
          <td>27.135620</td>
          <td>0.210255</td>
          <td>20.521994</td>
          <td>0.005109</td>
          <td>26.783000</td>
          <td>0.443585</td>
          <td>27.681536</td>
          <td>1.417912</td>
          <td>25.336607</td>
          <td>22.378670</td>
          <td>21.426005</td>
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
          <td>25.926592</td>
          <td>0.238034</td>
          <td>25.809659</td>
          <td>0.075778</td>
          <td>24.196563</td>
          <td>0.016327</td>
          <td>22.054035</td>
          <td>0.006256</td>
          <td>20.823281</td>
          <td>0.005571</td>
          <td>14.273960</td>
          <td>0.005001</td>
          <td>19.866823</td>
          <td>25.280302</td>
          <td>22.494929</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.717958</td>
          <td>0.005060</td>
          <td>23.432875</td>
          <td>0.010308</td>
          <td>20.434667</td>
          <td>0.005043</td>
          <td>20.814219</td>
          <td>0.005172</td>
          <td>23.083115</td>
          <td>0.018857</td>
          <td>25.993607</td>
          <td>0.488868</td>
          <td>29.171135</td>
          <td>18.452979</td>
          <td>24.681777</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.824697</td>
          <td>0.005221</td>
          <td>19.134944</td>
          <td>0.005014</td>
          <td>17.834604</td>
          <td>0.005002</td>
          <td>23.727303</td>
          <td>0.017499</td>
          <td>26.433119</td>
          <td>0.338381</td>
          <td>24.178695</td>
          <td>0.110513</td>
          <td>21.119441</td>
          <td>26.043956</td>
          <td>27.815431</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.492150</td>
          <td>0.012840</td>
          <td>24.965561</td>
          <td>0.035920</td>
          <td>25.367226</td>
          <td>0.044993</td>
          <td>25.412863</td>
          <td>0.076542</td>
          <td>18.670213</td>
          <td>0.005023</td>
          <td>24.500372</td>
          <td>0.146035</td>
          <td>22.488210</td>
          <td>25.109076</td>
          <td>24.990094</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.706977</td>
          <td>0.015107</td>
          <td>23.953189</td>
          <td>0.015176</td>
          <td>24.857504</td>
          <td>0.028685</td>
          <td>19.829167</td>
          <td>0.005039</td>
          <td>25.789861</td>
          <td>0.200066</td>
          <td>21.040252</td>
          <td>0.008299</td>
          <td>33.230911</td>
          <td>18.066318</td>
          <td>22.583690</td>
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
          <td>18.613817</td>
          <td>20.302193</td>
          <td>26.468342</td>
          <td>21.589165</td>
          <td>22.142546</td>
          <td>27.537869</td>
          <td>20.412873</td>
          <td>0.005025</td>
          <td>21.240426</td>
          <td>0.005338</td>
          <td>22.404323</td>
          <td>0.007398</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.065312</td>
          <td>27.905767</td>
          <td>26.454396</td>
          <td>22.949271</td>
          <td>22.287529</td>
          <td>25.864854</td>
          <td>23.954842</td>
          <td>0.013976</td>
          <td>22.061116</td>
          <td>0.006388</td>
          <td>23.308991</td>
          <td>0.013476</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.240402</td>
          <td>23.694273</td>
          <td>19.837737</td>
          <td>23.922312</td>
          <td>28.528831</td>
          <td>23.596148</td>
          <td>23.206070</td>
          <td>0.008251</td>
          <td>24.174304</td>
          <td>0.028037</td>
          <td>21.610166</td>
          <td>0.005647</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.556388</td>
          <td>19.651837</td>
          <td>22.899755</td>
          <td>22.766546</td>
          <td>32.297729</td>
          <td>20.656157</td>
          <td>20.927062</td>
          <td>0.005064</td>
          <td>25.071857</td>
          <td>0.062275</td>
          <td>28.754842</td>
          <td>1.098221</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.064831</td>
          <td>24.918750</td>
          <td>27.047703</td>
          <td>20.524345</td>
          <td>26.538727</td>
          <td>27.395244</td>
          <td>25.364494</td>
          <td>0.047357</td>
          <td>22.374305</td>
          <td>0.007289</td>
          <td>21.427191</td>
          <td>0.005470</td>
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
          <td>26.288087</td>
          <td>25.770230</td>
          <td>24.177021</td>
          <td>22.048512</td>
          <td>20.830878</td>
          <td>14.278012</td>
          <td>19.869411</td>
          <td>0.005009</td>
          <td>25.290323</td>
          <td>0.075608</td>
          <td>22.484578</td>
          <td>0.007711</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.722594</td>
          <td>23.424550</td>
          <td>20.434098</td>
          <td>20.804492</td>
          <td>23.070999</td>
          <td>27.281512</td>
          <td>28.816648</td>
          <td>0.787743</td>
          <td>18.462832</td>
          <td>0.005002</td>
          <td>24.689513</td>
          <td>0.044295</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.825124</td>
          <td>19.131364</td>
          <td>17.835537</td>
          <td>23.750945</td>
          <td>26.839706</td>
          <td>24.413093</td>
          <td>21.116381</td>
          <td>0.005091</td>
          <td>26.253610</td>
          <td>0.174941</td>
          <td>27.703645</td>
          <td>0.553213</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.492599</td>
          <td>24.992350</td>
          <td>25.407802</td>
          <td>25.422464</td>
          <td>18.668900</td>
          <td>24.670891</td>
          <td>22.495897</td>
          <td>0.006055</td>
          <td>25.155297</td>
          <td>0.067071</td>
          <td>24.967302</td>
          <td>0.056740</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.715990</td>
          <td>23.954201</td>
          <td>24.862240</td>
          <td>19.840170</td>
          <td>25.834966</td>
          <td>21.038619</td>
          <td>30.923347</td>
          <td>2.313203</td>
          <td>18.070230</td>
          <td>0.005001</td>
          <td>22.581964</td>
          <td>0.008137</td>
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
          <td>18.613817</td>
          <td>20.302193</td>
          <td>26.468342</td>
          <td>21.589165</td>
          <td>22.142546</td>
          <td>27.537869</td>
          <td>20.411967</td>
          <td>0.007826</td>
          <td>21.213376</td>
          <td>0.011593</td>
          <td>22.424264</td>
          <td>0.034979</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.065312</td>
          <td>27.905767</td>
          <td>26.454396</td>
          <td>22.949271</td>
          <td>22.287529</td>
          <td>25.864854</td>
          <td>23.955352</td>
          <td>0.147655</td>
          <td>22.050404</td>
          <td>0.023046</td>
          <td>23.335357</td>
          <td>0.078684</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.240402</td>
          <td>23.694273</td>
          <td>19.837737</td>
          <td>23.922312</td>
          <td>28.528831</td>
          <td>23.596148</td>
          <td>23.187338</td>
          <td>0.075408</td>
          <td>24.108420</td>
          <td>0.141805</td>
          <td>21.589363</td>
          <td>0.016932</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.556388</td>
          <td>19.651837</td>
          <td>22.899755</td>
          <td>22.766546</td>
          <td>32.297729</td>
          <td>20.656157</td>
          <td>20.940724</td>
          <td>0.010988</td>
          <td>25.020054</td>
          <td>0.303835</td>
          <td>28.877549</td>
          <td>2.814589</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.064831</td>
          <td>24.918750</td>
          <td>27.047703</td>
          <td>20.524345</td>
          <td>26.538727</td>
          <td>27.395244</td>
          <td>25.072145</td>
          <td>0.370968</td>
          <td>22.396970</td>
          <td>0.031245</td>
          <td>21.421431</td>
          <td>0.014745</td>
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
          <td>26.288087</td>
          <td>25.770230</td>
          <td>24.177021</td>
          <td>22.048512</td>
          <td>20.830878</td>
          <td>14.278012</td>
          <td>19.869546</td>
          <td>0.006193</td>
          <td>24.995979</td>
          <td>0.298008</td>
          <td>22.564812</td>
          <td>0.039637</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.722594</td>
          <td>23.424550</td>
          <td>20.434098</td>
          <td>20.804492</td>
          <td>23.070999</td>
          <td>27.281512</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.448384</td>
          <td>0.005067</td>
          <td>24.873640</td>
          <td>0.292689</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.825124</td>
          <td>19.131364</td>
          <td>17.835537</td>
          <td>23.750945</td>
          <td>26.839706</td>
          <td>24.413093</td>
          <td>21.139810</td>
          <td>0.012765</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.044880</td>
          <td>2.071799</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.492599</td>
          <td>24.992350</td>
          <td>25.407802</td>
          <td>25.422464</td>
          <td>18.668900</td>
          <td>24.670891</td>
          <td>22.566375</td>
          <td>0.043391</td>
          <td>25.565817</td>
          <td>0.464417</td>
          <td>25.035590</td>
          <td>0.333186</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.715990</td>
          <td>23.954201</td>
          <td>24.862240</td>
          <td>19.840170</td>
          <td>25.834966</td>
          <td>21.038619</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.067938</td>
          <td>0.005033</td>
          <td>22.622885</td>
          <td>0.041741</td>
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


