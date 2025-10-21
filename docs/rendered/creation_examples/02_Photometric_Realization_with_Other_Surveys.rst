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
          <td>22.870890</td>
          <td>30.897423</td>
          <td>24.375868</td>
          <td>22.069420</td>
          <td>22.532511</td>
          <td>22.271248</td>
          <td>22.195603</td>
          <td>23.571477</td>
          <td>18.310709</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413360</td>
          <td>26.720518</td>
          <td>20.193691</td>
          <td>23.987627</td>
          <td>25.511974</td>
          <td>20.934109</td>
          <td>27.129234</td>
          <td>28.142491</td>
          <td>21.436317</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.792121</td>
          <td>23.242754</td>
          <td>23.353094</td>
          <td>21.749605</td>
          <td>21.832945</td>
          <td>26.586608</td>
          <td>23.868248</td>
          <td>29.111009</td>
          <td>24.728366</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.774510</td>
          <td>23.368290</td>
          <td>21.944373</td>
          <td>21.381344</td>
          <td>26.036900</td>
          <td>23.838932</td>
          <td>19.692005</td>
          <td>26.491170</td>
          <td>23.906621</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.655292</td>
          <td>21.576378</td>
          <td>21.770368</td>
          <td>18.138373</td>
          <td>22.577279</td>
          <td>20.942897</td>
          <td>24.241311</td>
          <td>31.911945</td>
          <td>26.161118</td>
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
          <td>16.392166</td>
          <td>26.060456</td>
          <td>22.597943</td>
          <td>21.596334</td>
          <td>25.081507</td>
          <td>23.926489</td>
          <td>18.143080</td>
          <td>25.324908</td>
          <td>23.703858</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.389488</td>
          <td>24.102609</td>
          <td>22.602260</td>
          <td>24.835941</td>
          <td>20.022503</td>
          <td>22.186872</td>
          <td>25.998594</td>
          <td>20.600800</td>
          <td>23.387081</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.946477</td>
          <td>28.249018</td>
          <td>18.373481</td>
          <td>24.087811</td>
          <td>31.494367</td>
          <td>20.621929</td>
          <td>24.229333</td>
          <td>22.181866</td>
          <td>30.377600</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.042821</td>
          <td>19.384594</td>
          <td>24.661361</td>
          <td>22.356445</td>
          <td>26.638396</td>
          <td>24.686913</td>
          <td>25.047794</td>
          <td>20.768317</td>
          <td>18.075336</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.448941</td>
          <td>22.302186</td>
          <td>24.257093</td>
          <td>25.420176</td>
          <td>24.693761</td>
          <td>21.303649</td>
          <td>19.318690</td>
          <td>22.299030</td>
          <td>28.920471</td>
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
          <td>22.849481</td>
          <td>0.016905</td>
          <td>29.127424</td>
          <td>1.006031</td>
          <td>24.407318</td>
          <td>0.019457</td>
          <td>22.063801</td>
          <td>0.006275</td>
          <td>22.549476</td>
          <td>0.012297</td>
          <td>22.309087</td>
          <td>0.021327</td>
          <td>22.195603</td>
          <td>23.571477</td>
          <td>18.310709</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.366597</td>
          <td>0.062161</td>
          <td>26.949573</td>
          <td>0.203019</td>
          <td>20.194586</td>
          <td>0.005031</td>
          <td>23.965154</td>
          <td>0.021386</td>
          <td>25.433261</td>
          <td>0.147756</td>
          <td>20.928457</td>
          <td>0.007808</td>
          <td>27.129234</td>
          <td>28.142491</td>
          <td>21.436317</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.837817</td>
          <td>0.039061</td>
          <td>23.248926</td>
          <td>0.009149</td>
          <td>23.357278</td>
          <td>0.008852</td>
          <td>21.753927</td>
          <td>0.005779</td>
          <td>21.836408</td>
          <td>0.007761</td>
          <td>25.861461</td>
          <td>0.442817</td>
          <td>23.868248</td>
          <td>29.111009</td>
          <td>24.728366</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.771580</td>
          <td>0.036864</td>
          <td>23.354904</td>
          <td>0.009787</td>
          <td>21.932396</td>
          <td>0.005426</td>
          <td>21.382716</td>
          <td>0.005428</td>
          <td>26.088717</td>
          <td>0.256394</td>
          <td>23.964556</td>
          <td>0.091615</td>
          <td>19.692005</td>
          <td>26.491170</td>
          <td>23.906621</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.652218</td>
          <td>0.005056</td>
          <td>21.571463</td>
          <td>0.005338</td>
          <td>21.765027</td>
          <td>0.005326</td>
          <td>18.134210</td>
          <td>0.005005</td>
          <td>22.559656</td>
          <td>0.012392</td>
          <td>20.946104</td>
          <td>0.007881</td>
          <td>24.241311</td>
          <td>31.911945</td>
          <td>26.161118</td>
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
          <td>16.389912</td>
          <td>0.005006</td>
          <td>26.040427</td>
          <td>0.092839</td>
          <td>22.599380</td>
          <td>0.006242</td>
          <td>21.588074</td>
          <td>0.005596</td>
          <td>25.099496</td>
          <td>0.110662</td>
          <td>23.948393</td>
          <td>0.090323</td>
          <td>18.143080</td>
          <td>25.324908</td>
          <td>23.703858</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.466002</td>
          <td>0.161746</td>
          <td>24.074765</td>
          <td>0.016741</td>
          <td>22.607753</td>
          <td>0.006258</td>
          <td>24.746293</td>
          <td>0.042381</td>
          <td>20.024599</td>
          <td>0.005160</td>
          <td>22.157383</td>
          <td>0.018756</td>
          <td>25.998594</td>
          <td>20.600800</td>
          <td>23.387081</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.949241</td>
          <td>0.005027</td>
          <td>28.665549</td>
          <td>0.751099</td>
          <td>18.381887</td>
          <td>0.005003</td>
          <td>24.130886</td>
          <td>0.024666</td>
          <td>26.732912</td>
          <td>0.427050</td>
          <td>20.620537</td>
          <td>0.006773</td>
          <td>24.229333</td>
          <td>22.181866</td>
          <td>30.377600</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.063168</td>
          <td>0.009595</td>
          <td>19.380867</td>
          <td>0.005018</td>
          <td>24.637365</td>
          <td>0.023684</td>
          <td>22.346063</td>
          <td>0.006973</td>
          <td>27.205766</td>
          <td>0.604351</td>
          <td>24.883313</td>
          <td>0.202212</td>
          <td>25.047794</td>
          <td>20.768317</td>
          <td>18.075336</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.450263</td>
          <td>0.005139</td>
          <td>22.306463</td>
          <td>0.006046</td>
          <td>24.269660</td>
          <td>0.017340</td>
          <td>25.485541</td>
          <td>0.081614</td>
          <td>24.789713</td>
          <td>0.084334</td>
          <td>21.312829</td>
          <td>0.009818</td>
          <td>19.318690</td>
          <td>22.299030</td>
          <td>28.920471</td>
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
          <td>22.870890</td>
          <td>30.897423</td>
          <td>24.375868</td>
          <td>22.069420</td>
          <td>22.532511</td>
          <td>22.271248</td>
          <td>22.200096</td>
          <td>0.005636</td>
          <td>23.584842</td>
          <td>0.016868</td>
          <td>18.311836</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413360</td>
          <td>26.720518</td>
          <td>20.193691</td>
          <td>23.987627</td>
          <td>25.511974</td>
          <td>20.934109</td>
          <td>26.805029</td>
          <td>0.167853</td>
          <td>27.852889</td>
          <td>0.615273</td>
          <td>21.432363</td>
          <td>0.005474</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.792121</td>
          <td>23.242754</td>
          <td>23.353094</td>
          <td>21.749605</td>
          <td>21.832945</td>
          <td>26.586608</td>
          <td>23.880249</td>
          <td>0.013174</td>
          <td>27.816512</td>
          <td>0.599686</td>
          <td>24.685986</td>
          <td>0.044156</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.774510</td>
          <td>23.368290</td>
          <td>21.944373</td>
          <td>21.381344</td>
          <td>26.036900</td>
          <td>23.838932</td>
          <td>19.694038</td>
          <td>0.005007</td>
          <td>26.381619</td>
          <td>0.194960</td>
          <td>23.905836</td>
          <td>0.022172</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.655292</td>
          <td>21.576378</td>
          <td>21.770368</td>
          <td>18.138373</td>
          <td>22.577279</td>
          <td>20.942897</td>
          <td>24.243620</td>
          <td>0.017720</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.968347</td>
          <td>0.136982</td>
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
          <td>16.392166</td>
          <td>26.060456</td>
          <td>22.597943</td>
          <td>21.596334</td>
          <td>25.081507</td>
          <td>23.926489</td>
          <td>18.148989</td>
          <td>0.005000</td>
          <td>25.236305</td>
          <td>0.072072</td>
          <td>23.685592</td>
          <td>0.018359</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.389488</td>
          <td>24.102609</td>
          <td>22.602260</td>
          <td>24.835941</td>
          <td>20.022503</td>
          <td>22.186872</td>
          <td>25.930381</td>
          <td>0.078338</td>
          <td>20.597551</td>
          <td>0.005106</td>
          <td>23.388590</td>
          <td>0.014359</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.946477</td>
          <td>28.249018</td>
          <td>18.373481</td>
          <td>24.087811</td>
          <td>31.494367</td>
          <td>20.621929</td>
          <td>24.216497</td>
          <td>0.017320</td>
          <td>22.181300</td>
          <td>0.006688</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.042821</td>
          <td>19.384594</td>
          <td>24.661361</td>
          <td>22.356445</td>
          <td>26.638396</td>
          <td>24.686913</td>
          <td>25.054386</td>
          <td>0.035928</td>
          <td>20.775544</td>
          <td>0.005146</td>
          <td>18.069671</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.448941</td>
          <td>22.302186</td>
          <td>24.257093</td>
          <td>25.420176</td>
          <td>24.693761</td>
          <td>21.303649</td>
          <td>19.320314</td>
          <td>0.005003</td>
          <td>22.292197</td>
          <td>0.007014</td>
          <td>27.289921</td>
          <td>0.406384</td>
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
          <td>22.870890</td>
          <td>30.897423</td>
          <td>24.375868</td>
          <td>22.069420</td>
          <td>22.532511</td>
          <td>22.271248</td>
          <td>22.200390</td>
          <td>0.031339</td>
          <td>23.633076</td>
          <td>0.093688</td>
          <td>18.304468</td>
          <td>0.005062</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.413360</td>
          <td>26.720518</td>
          <td>20.193691</td>
          <td>23.987627</td>
          <td>25.511974</td>
          <td>20.934109</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.426794</td>
          <td>0.014809</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.792121</td>
          <td>23.242754</td>
          <td>23.353094</td>
          <td>21.749605</td>
          <td>21.832945</td>
          <td>26.586608</td>
          <td>23.866715</td>
          <td>0.136789</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.372585</td>
          <td>0.432873</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.774510</td>
          <td>23.368290</td>
          <td>21.944373</td>
          <td>21.381344</td>
          <td>26.036900</td>
          <td>23.838932</td>
          <td>19.687825</td>
          <td>0.005879</td>
          <td>26.560909</td>
          <td>0.920595</td>
          <td>23.726590</td>
          <td>0.111019</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.655292</td>
          <td>21.576378</td>
          <td>21.770368</td>
          <td>18.138373</td>
          <td>22.577279</td>
          <td>20.942897</td>
          <td>24.273457</td>
          <td>0.193624</td>
          <td>28.494581</td>
          <td>2.376217</td>
          <td>25.451747</td>
          <td>0.459541</td>
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
          <td>16.392166</td>
          <td>26.060456</td>
          <td>22.597943</td>
          <td>21.596334</td>
          <td>25.081507</td>
          <td>23.926489</td>
          <td>18.139194</td>
          <td>0.005055</td>
          <td>25.050794</td>
          <td>0.311417</td>
          <td>23.705585</td>
          <td>0.108999</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.389488</td>
          <td>24.102609</td>
          <td>22.602260</td>
          <td>24.835941</td>
          <td>20.022503</td>
          <td>22.186872</td>
          <td>26.333463</td>
          <td>0.904990</td>
          <td>20.595559</td>
          <td>0.007756</td>
          <td>23.447414</td>
          <td>0.086875</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.946477</td>
          <td>28.249018</td>
          <td>18.373481</td>
          <td>24.087811</td>
          <td>31.494367</td>
          <td>20.621929</td>
          <td>24.061068</td>
          <td>0.161669</td>
          <td>22.159742</td>
          <td>0.025353</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.042821</td>
          <td>19.384594</td>
          <td>24.661361</td>
          <td>22.356445</td>
          <td>26.638396</td>
          <td>24.686913</td>
          <td>25.108294</td>
          <td>0.381553</td>
          <td>20.760469</td>
          <td>0.008521</td>
          <td>18.076999</td>
          <td>0.005041</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.448941</td>
          <td>22.302186</td>
          <td>24.257093</td>
          <td>25.420176</td>
          <td>24.693761</td>
          <td>21.303649</td>
          <td>19.318491</td>
          <td>0.005463</td>
          <td>22.267141</td>
          <td>0.027860</td>
          <td>25.435152</td>
          <td>0.453844</td>
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


