Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


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
          <td>27.482661</td>
          <td>22.097914</td>
          <td>24.966595</td>
          <td>19.310211</td>
          <td>23.608600</td>
          <td>23.057128</td>
          <td>30.079309</td>
          <td>16.192879</td>
          <td>21.706311</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.325536</td>
          <td>19.558801</td>
          <td>22.329463</td>
          <td>23.206683</td>
          <td>20.228769</td>
          <td>22.727834</td>
          <td>23.333468</td>
          <td>18.491400</td>
          <td>21.850260</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.317284</td>
          <td>28.145567</td>
          <td>19.747574</td>
          <td>26.240025</td>
          <td>25.639399</td>
          <td>24.244728</td>
          <td>23.039579</td>
          <td>24.858212</td>
          <td>23.548941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.295558</td>
          <td>19.404690</td>
          <td>24.278785</td>
          <td>17.150958</td>
          <td>22.677502</td>
          <td>27.744644</td>
          <td>29.112214</td>
          <td>21.114003</td>
          <td>24.804115</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.689713</td>
          <td>19.842892</td>
          <td>23.111172</td>
          <td>21.790306</td>
          <td>22.142899</td>
          <td>21.796613</td>
          <td>27.453740</td>
          <td>21.859898</td>
          <td>23.916581</td>
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
          <td>23.992798</td>
          <td>24.882550</td>
          <td>15.864801</td>
          <td>19.557527</td>
          <td>22.043534</td>
          <td>20.418401</td>
          <td>25.053811</td>
          <td>28.339640</td>
          <td>22.308848</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.194702</td>
          <td>21.784577</td>
          <td>26.156489</td>
          <td>20.445770</td>
          <td>24.383252</td>
          <td>21.902993</td>
          <td>23.723350</td>
          <td>20.991641</td>
          <td>22.021040</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.441489</td>
          <td>21.678468</td>
          <td>23.449058</td>
          <td>24.629565</td>
          <td>25.603747</td>
          <td>19.411335</td>
          <td>26.241105</td>
          <td>23.653227</td>
          <td>23.107317</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.782678</td>
          <td>20.032707</td>
          <td>22.016296</td>
          <td>18.685025</td>
          <td>21.279153</td>
          <td>24.526742</td>
          <td>23.165522</td>
          <td>29.495037</td>
          <td>18.328367</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.923612</td>
          <td>21.379566</td>
          <td>22.166380</td>
          <td>24.612957</td>
          <td>26.060272</td>
          <td>19.216619</td>
          <td>18.977949</td>
          <td>24.501105</td>
          <td>25.190882</td>
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
          <td>22.093761</td>
          <td>0.005754</td>
          <td>25.018016</td>
          <td>0.033030</td>
          <td>19.306692</td>
          <td>0.005019</td>
          <td>23.617287</td>
          <td>0.029880</td>
          <td>23.057557</td>
          <td>0.041039</td>
          <td>30.079309</td>
          <td>16.192879</td>
          <td>21.706311</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.406396</td>
          <td>0.728337</td>
          <td>19.552703</td>
          <td>0.005022</td>
          <td>22.328550</td>
          <td>0.005808</td>
          <td>23.203218</td>
          <td>0.011585</td>
          <td>20.223434</td>
          <td>0.005219</td>
          <td>22.798824</td>
          <td>0.032650</td>
          <td>23.333468</td>
          <td>18.491400</td>
          <td>21.850260</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.332971</td>
          <td>0.060348</td>
          <td>28.133338</td>
          <td>0.517725</td>
          <td>19.747608</td>
          <td>0.005017</td>
          <td>26.028079</td>
          <td>0.131167</td>
          <td>25.750794</td>
          <td>0.193599</td>
          <td>24.278277</td>
          <td>0.120524</td>
          <td>23.039579</td>
          <td>24.858212</td>
          <td>23.548941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.289272</td>
          <td>0.005405</td>
          <td>19.400106</td>
          <td>0.005019</td>
          <td>24.271503</td>
          <td>0.017367</td>
          <td>17.148956</td>
          <td>0.005002</td>
          <td>22.677680</td>
          <td>0.013565</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.112214</td>
          <td>21.114003</td>
          <td>24.804115</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.691778</td>
          <td>0.005187</td>
          <td>19.848581</td>
          <td>0.005031</td>
          <td>23.109359</td>
          <td>0.007701</td>
          <td>21.781657</td>
          <td>0.005814</td>
          <td>22.144920</td>
          <td>0.009277</td>
          <td>21.809300</td>
          <td>0.014118</td>
          <td>27.453740</td>
          <td>21.859898</td>
          <td>23.916581</td>
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
          <td>23.994126</td>
          <td>0.044796</td>
          <td>24.795406</td>
          <td>0.030934</td>
          <td>15.870518</td>
          <td>0.005000</td>
          <td>19.551513</td>
          <td>0.005027</td>
          <td>22.052894</td>
          <td>0.008764</td>
          <td>20.422902</td>
          <td>0.006306</td>
          <td>25.053811</td>
          <td>28.339640</td>
          <td>22.308848</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.195541</td>
          <td>0.006419</td>
          <td>21.784365</td>
          <td>0.005468</td>
          <td>26.199975</td>
          <td>0.094029</td>
          <td>20.441894</td>
          <td>0.005097</td>
          <td>24.369603</td>
          <td>0.058153</td>
          <td>21.914769</td>
          <td>0.015358</td>
          <td>23.723350</td>
          <td>20.991641</td>
          <td>22.021040</td>
        </tr>
        <tr>
          <th>997</th>
          <td>28.258024</td>
          <td>1.226131</td>
          <td>21.680763</td>
          <td>0.005400</td>
          <td>23.438320</td>
          <td>0.009311</td>
          <td>24.654626</td>
          <td>0.039073</td>
          <td>25.380604</td>
          <td>0.141212</td>
          <td>19.411241</td>
          <td>0.005260</td>
          <td>26.241105</td>
          <td>23.653227</td>
          <td>23.107317</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.779422</td>
          <td>0.008165</td>
          <td>20.033759</td>
          <td>0.005040</td>
          <td>22.010210</td>
          <td>0.005484</td>
          <td>18.688934</td>
          <td>0.005009</td>
          <td>21.278358</td>
          <td>0.006181</td>
          <td>24.465300</td>
          <td>0.141693</td>
          <td>23.165522</td>
          <td>29.495037</td>
          <td>18.328367</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.921736</td>
          <td>0.005967</td>
          <td>21.378193</td>
          <td>0.005253</td>
          <td>22.175920</td>
          <td>0.005632</td>
          <td>24.579878</td>
          <td>0.036571</td>
          <td>26.103819</td>
          <td>0.259584</td>
          <td>19.219643</td>
          <td>0.005192</td>
          <td>18.977949</td>
          <td>24.501105</td>
          <td>25.190882</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


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
          <td>27.482661</td>
          <td>22.097914</td>
          <td>24.966595</td>
          <td>19.310211</td>
          <td>23.608600</td>
          <td>23.057128</td>
          <td>29.537769</td>
          <td>1.218130</td>
          <td>16.191082</td>
          <td>0.005000</td>
          <td>21.705795</td>
          <td>0.005764</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.325536</td>
          <td>19.558801</td>
          <td>22.329463</td>
          <td>23.206683</td>
          <td>20.228769</td>
          <td>22.727834</td>
          <td>23.327845</td>
          <td>0.008882</td>
          <td>18.490050</td>
          <td>0.005002</td>
          <td>21.849944</td>
          <td>0.005977</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.317284</td>
          <td>28.145567</td>
          <td>19.747574</td>
          <td>26.240025</td>
          <td>25.639399</td>
          <td>24.244728</td>
          <td>23.033513</td>
          <td>0.007508</td>
          <td>24.833648</td>
          <td>0.050369</td>
          <td>23.547775</td>
          <td>0.016355</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.295558</td>
          <td>19.404690</td>
          <td>24.278785</td>
          <td>17.150958</td>
          <td>22.677502</td>
          <td>27.744644</td>
          <td>29.219850</td>
          <td>1.014285</td>
          <td>21.112205</td>
          <td>0.005268</td>
          <td>24.846391</td>
          <td>0.050944</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.689713</td>
          <td>19.842892</td>
          <td>23.111172</td>
          <td>21.790306</td>
          <td>22.142899</td>
          <td>21.796613</td>
          <td>27.407268</td>
          <td>0.277371</td>
          <td>21.851033</td>
          <td>0.005978</td>
          <td>23.939657</td>
          <td>0.022832</td>
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
          <td>23.992798</td>
          <td>24.882550</td>
          <td>15.864801</td>
          <td>19.557527</td>
          <td>22.043534</td>
          <td>20.418401</td>
          <td>25.048773</td>
          <td>0.035749</td>
          <td>29.492303</td>
          <td>1.621157</td>
          <td>22.321240</td>
          <td>0.007108</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.194702</td>
          <td>21.784577</td>
          <td>26.156489</td>
          <td>20.445770</td>
          <td>24.383252</td>
          <td>21.902993</td>
          <td>23.732873</td>
          <td>0.011763</td>
          <td>20.998566</td>
          <td>0.005219</td>
          <td>22.024098</td>
          <td>0.006306</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.441489</td>
          <td>21.678468</td>
          <td>23.449058</td>
          <td>24.629565</td>
          <td>25.603747</td>
          <td>19.411335</td>
          <td>26.362895</td>
          <td>0.114595</td>
          <td>23.661711</td>
          <td>0.017992</td>
          <td>23.093875</td>
          <td>0.011426</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.782678</td>
          <td>20.032707</td>
          <td>22.016296</td>
          <td>18.685025</td>
          <td>21.279153</td>
          <td>24.526742</td>
          <td>23.170540</td>
          <td>0.008084</td>
          <td>27.799276</td>
          <td>0.592405</td>
          <td>18.330320</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.923612</td>
          <td>21.379566</td>
          <td>22.166380</td>
          <td>24.612957</td>
          <td>26.060272</td>
          <td>19.216619</td>
          <td>18.977574</td>
          <td>0.005002</td>
          <td>24.512510</td>
          <td>0.037834</td>
          <td>25.224078</td>
          <td>0.071294</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


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
          <td>27.482661</td>
          <td>22.097914</td>
          <td>24.966595</td>
          <td>19.310211</td>
          <td>23.608600</td>
          <td>23.057128</td>
          <td>inf</td>
          <td>inf</td>
          <td>16.187670</td>
          <td>0.005001</td>
          <td>21.723403</td>
          <td>0.018957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.325536</td>
          <td>19.558801</td>
          <td>22.329463</td>
          <td>23.206683</td>
          <td>20.228769</td>
          <td>22.727834</td>
          <td>23.250159</td>
          <td>0.079721</td>
          <td>18.483908</td>
          <td>0.005072</td>
          <td>21.852171</td>
          <td>0.021167</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.317284</td>
          <td>28.145567</td>
          <td>19.747574</td>
          <td>26.240025</td>
          <td>25.639399</td>
          <td>24.244728</td>
          <td>23.093393</td>
          <td>0.069379</td>
          <td>25.170954</td>
          <td>0.342642</td>
          <td>23.462588</td>
          <td>0.088046</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.295558</td>
          <td>19.404690</td>
          <td>24.278785</td>
          <td>17.150958</td>
          <td>22.677502</td>
          <td>27.744644</td>
          <td>25.645380</td>
          <td>0.570066</td>
          <td>21.108592</td>
          <td>0.010735</td>
          <td>24.758556</td>
          <td>0.266581</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.689713</td>
          <td>19.842892</td>
          <td>23.111172</td>
          <td>21.790306</td>
          <td>22.142899</td>
          <td>21.796613</td>
          <td>25.555451</td>
          <td>0.534229</td>
          <td>21.858778</td>
          <td>0.019537</td>
          <td>23.811398</td>
          <td>0.119544</td>
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
          <td>23.992798</td>
          <td>24.882550</td>
          <td>15.864801</td>
          <td>19.557527</td>
          <td>22.043534</td>
          <td>20.418401</td>
          <td>25.340283</td>
          <td>0.455599</td>
          <td>25.835000</td>
          <td>0.565839</td>
          <td>22.320197</td>
          <td>0.031894</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.194702</td>
          <td>21.784577</td>
          <td>26.156489</td>
          <td>20.445770</td>
          <td>24.383252</td>
          <td>21.902993</td>
          <td>23.569853</td>
          <td>0.105643</td>
          <td>20.999232</td>
          <td>0.009941</td>
          <td>22.020192</td>
          <td>0.024491</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.441489</td>
          <td>21.678468</td>
          <td>23.449058</td>
          <td>24.629565</td>
          <td>25.603747</td>
          <td>19.411335</td>
          <td>26.248028</td>
          <td>0.857507</td>
          <td>23.586378</td>
          <td>0.089913</td>
          <td>23.079768</td>
          <td>0.062715</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.782678</td>
          <td>20.032707</td>
          <td>22.016296</td>
          <td>18.685025</td>
          <td>21.279153</td>
          <td>24.526742</td>
          <td>23.291076</td>
          <td>0.082659</td>
          <td>25.506943</td>
          <td>0.444289</td>
          <td>18.329799</td>
          <td>0.005065</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.923612</td>
          <td>21.379566</td>
          <td>22.166380</td>
          <td>24.612957</td>
          <td>26.060272</td>
          <td>19.216619</td>
          <td>18.977676</td>
          <td>0.005252</td>
          <td>24.396321</td>
          <td>0.181400</td>
          <td>24.545411</td>
          <td>0.223622</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


