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
          <td>22.574148</td>
          <td>22.961464</td>
          <td>22.860837</td>
          <td>17.096406</td>
          <td>26.598348</td>
          <td>23.531409</td>
          <td>20.222116</td>
          <td>22.215863</td>
          <td>22.910032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.653726</td>
          <td>24.886808</td>
          <td>24.381236</td>
          <td>22.416764</td>
          <td>23.421666</td>
          <td>22.274154</td>
          <td>27.675274</td>
          <td>28.207124</td>
          <td>21.333551</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.407243</td>
          <td>23.502652</td>
          <td>22.690881</td>
          <td>23.325959</td>
          <td>21.707238</td>
          <td>18.481729</td>
          <td>18.374153</td>
          <td>21.880585</td>
          <td>25.070783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.490041</td>
          <td>27.109533</td>
          <td>23.538852</td>
          <td>19.575705</td>
          <td>20.569638</td>
          <td>21.488329</td>
          <td>21.401879</td>
          <td>15.906581</td>
          <td>21.109494</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.333192</td>
          <td>24.442697</td>
          <td>21.800615</td>
          <td>27.639239</td>
          <td>19.989515</td>
          <td>17.923648</td>
          <td>21.348647</td>
          <td>19.380930</td>
          <td>28.062476</td>
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
          <td>20.922410</td>
          <td>20.741286</td>
          <td>18.054376</td>
          <td>23.617085</td>
          <td>22.211575</td>
          <td>27.497711</td>
          <td>27.046360</td>
          <td>21.460503</td>
          <td>22.102713</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.422812</td>
          <td>23.946753</td>
          <td>18.570381</td>
          <td>24.514421</td>
          <td>25.411920</td>
          <td>19.419784</td>
          <td>23.914549</td>
          <td>22.092699</td>
          <td>28.004265</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.407802</td>
          <td>26.604643</td>
          <td>30.203883</td>
          <td>29.770562</td>
          <td>26.844451</td>
          <td>26.189000</td>
          <td>21.581264</td>
          <td>24.110028</td>
          <td>17.005911</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.389247</td>
          <td>25.277501</td>
          <td>26.839553</td>
          <td>22.236270</td>
          <td>17.335306</td>
          <td>21.666841</td>
          <td>23.673574</td>
          <td>25.271242</td>
          <td>24.171441</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.522847</td>
          <td>20.053239</td>
          <td>22.040772</td>
          <td>26.985275</td>
          <td>23.529901</td>
          <td>26.859055</td>
          <td>18.165966</td>
          <td>26.188770</td>
          <td>24.554887</td>
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
          <td>22.578192</td>
          <td>0.013689</td>
          <td>22.959843</td>
          <td>0.007771</td>
          <td>22.864201</td>
          <td>0.006872</td>
          <td>17.101798</td>
          <td>0.005002</td>
          <td>26.164791</td>
          <td>0.272827</td>
          <td>23.559870</td>
          <td>0.064086</td>
          <td>20.222116</td>
          <td>22.215863</td>
          <td>22.910032</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.657060</td>
          <td>0.005669</td>
          <td>24.929885</td>
          <td>0.034809</td>
          <td>24.375535</td>
          <td>0.018943</td>
          <td>22.406572</td>
          <td>0.007162</td>
          <td>23.410811</td>
          <td>0.024953</td>
          <td>22.299536</td>
          <td>0.021154</td>
          <td>27.675274</td>
          <td>28.207124</td>
          <td>21.333551</td>
        </tr>
        <tr>
          <th>2</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.504854</td>
          <td>0.010831</td>
          <td>22.691555</td>
          <td>0.006434</td>
          <td>23.359078</td>
          <td>0.013024</td>
          <td>21.704717</td>
          <td>0.007273</td>
          <td>18.486425</td>
          <td>0.005063</td>
          <td>18.374153</td>
          <td>21.880585</td>
          <td>25.070783</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.110456</td>
          <td>0.593835</td>
          <td>27.164119</td>
          <td>0.242663</td>
          <td>23.538756</td>
          <td>0.009943</td>
          <td>19.580582</td>
          <td>0.005028</td>
          <td>20.567506</td>
          <td>0.005379</td>
          <td>21.486893</td>
          <td>0.011071</td>
          <td>21.401879</td>
          <td>15.906581</td>
          <td>21.109494</td>
        </tr>
        <tr>
          <th>4</th>
          <td>inf</td>
          <td>inf</td>
          <td>24.438307</td>
          <td>0.022701</td>
          <td>21.805297</td>
          <td>0.005347</td>
          <td>27.396420</td>
          <td>0.404987</td>
          <td>19.991343</td>
          <td>0.005152</td>
          <td>17.920291</td>
          <td>0.005029</td>
          <td>21.348647</td>
          <td>19.380930</td>
          <td>28.062476</td>
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
          <td>20.916554</td>
          <td>0.005960</td>
          <td>20.737145</td>
          <td>0.005100</td>
          <td>18.050902</td>
          <td>0.005002</td>
          <td>23.595078</td>
          <td>0.015699</td>
          <td>22.219423</td>
          <td>0.009735</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.046360</td>
          <td>21.460503</td>
          <td>22.102713</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>23.956541</td>
          <td>0.015217</td>
          <td>18.574600</td>
          <td>0.005004</td>
          <td>24.517201</td>
          <td>0.034601</td>
          <td>25.199274</td>
          <td>0.120706</td>
          <td>19.412619</td>
          <td>0.005260</td>
          <td>23.914549</td>
          <td>22.092699</td>
          <td>28.004265</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.414703</td>
          <td>0.006925</td>
          <td>26.380240</td>
          <td>0.124874</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.943168</td>
          <td>0.499952</td>
          <td>25.366088</td>
          <td>0.300766</td>
          <td>21.581264</td>
          <td>24.110028</td>
          <td>17.005911</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.310647</td>
          <td>0.059174</td>
          <td>25.252570</td>
          <td>0.046287</td>
          <td>26.970835</td>
          <td>0.183037</td>
          <td>22.242780</td>
          <td>0.006685</td>
          <td>17.337324</td>
          <td>0.005005</td>
          <td>21.658391</td>
          <td>0.012563</td>
          <td>23.673574</td>
          <td>25.271242</td>
          <td>24.171441</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.537156</td>
          <td>0.013274</td>
          <td>20.057093</td>
          <td>0.005041</td>
          <td>22.050399</td>
          <td>0.005516</td>
          <td>27.146960</td>
          <td>0.333319</td>
          <td>23.530969</td>
          <td>0.027705</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.165966</td>
          <td>26.188770</td>
          <td>24.554887</td>
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
          <td>22.574148</td>
          <td>22.961464</td>
          <td>22.860837</td>
          <td>17.096406</td>
          <td>26.598348</td>
          <td>23.531409</td>
          <td>20.214362</td>
          <td>0.005017</td>
          <td>22.215610</td>
          <td>0.006783</td>
          <td>22.911657</td>
          <td>0.010027</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.653726</td>
          <td>24.886808</td>
          <td>24.381236</td>
          <td>22.416764</td>
          <td>23.421666</td>
          <td>22.274154</td>
          <td>28.461711</td>
          <td>0.619098</td>
          <td>27.292096</td>
          <td>0.407063</td>
          <td>21.339159</td>
          <td>0.005402</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.407243</td>
          <td>23.502652</td>
          <td>22.690881</td>
          <td>23.325959</td>
          <td>21.707238</td>
          <td>18.481729</td>
          <td>18.371473</td>
          <td>0.005001</td>
          <td>21.888006</td>
          <td>0.006041</td>
          <td>25.135505</td>
          <td>0.065901</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.490041</td>
          <td>27.109533</td>
          <td>23.538852</td>
          <td>19.575705</td>
          <td>20.569638</td>
          <td>21.488329</td>
          <td>21.403099</td>
          <td>0.005154</td>
          <td>15.901297</td>
          <td>0.005000</td>
          <td>21.114193</td>
          <td>0.005269</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.333192</td>
          <td>24.442697</td>
          <td>21.800615</td>
          <td>27.639239</td>
          <td>19.989515</td>
          <td>17.923648</td>
          <td>21.350632</td>
          <td>0.005140</td>
          <td>19.385971</td>
          <td>0.005011</td>
          <td>27.573550</td>
          <td>0.503132</td>
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
          <td>20.922410</td>
          <td>20.741286</td>
          <td>18.054376</td>
          <td>23.617085</td>
          <td>22.211575</td>
          <td>27.497711</td>
          <td>26.797848</td>
          <td>0.166828</td>
          <td>21.459256</td>
          <td>0.005497</td>
          <td>22.095254</td>
          <td>0.006468</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.422812</td>
          <td>23.946753</td>
          <td>18.570381</td>
          <td>24.514421</td>
          <td>25.411920</td>
          <td>19.419784</td>
          <td>23.925121</td>
          <td>0.013649</td>
          <td>22.087332</td>
          <td>0.006449</td>
          <td>29.327263</td>
          <td>1.495441</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.407802</td>
          <td>26.604643</td>
          <td>30.203883</td>
          <td>29.770562</td>
          <td>26.844451</td>
          <td>26.189000</td>
          <td>21.573893</td>
          <td>0.005209</td>
          <td>24.129169</td>
          <td>0.026945</td>
          <td>17.007316</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.389247</td>
          <td>25.277501</td>
          <td>26.839553</td>
          <td>22.236270</td>
          <td>17.335306</td>
          <td>21.666841</td>
          <td>23.681897</td>
          <td>0.011325</td>
          <td>25.288400</td>
          <td>0.075479</td>
          <td>24.168015</td>
          <td>0.027882</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.522847</td>
          <td>20.053239</td>
          <td>22.040772</td>
          <td>26.985275</td>
          <td>23.529901</td>
          <td>26.859055</td>
          <td>18.168293</td>
          <td>0.005000</td>
          <td>26.004176</td>
          <td>0.141286</td>
          <td>24.556795</td>
          <td>0.039355</td>
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
          <td>22.574148</td>
          <td>22.961464</td>
          <td>22.860837</td>
          <td>17.096406</td>
          <td>26.598348</td>
          <td>23.531409</td>
          <td>20.214943</td>
          <td>0.007087</td>
          <td>22.180006</td>
          <td>0.025807</td>
          <td>23.001930</td>
          <td>0.058517</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.653726</td>
          <td>24.886808</td>
          <td>24.381236</td>
          <td>22.416764</td>
          <td>23.421666</td>
          <td>22.274154</td>
          <td>26.081094</td>
          <td>0.769545</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.342902</td>
          <td>0.013843</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.407243</td>
          <td>23.502652</td>
          <td>22.690881</td>
          <td>23.325959</td>
          <td>21.707238</td>
          <td>18.481729</td>
          <td>18.372082</td>
          <td>0.005084</td>
          <td>21.870419</td>
          <td>0.019732</td>
          <td>26.095673</td>
          <td>0.727014</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.490041</td>
          <td>27.109533</td>
          <td>23.538852</td>
          <td>19.575705</td>
          <td>20.569638</td>
          <td>21.488329</td>
          <td>21.436920</td>
          <td>0.016209</td>
          <td>15.906776</td>
          <td>0.005001</td>
          <td>21.115291</td>
          <td>0.011609</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.333192</td>
          <td>24.442697</td>
          <td>21.800615</td>
          <td>27.639239</td>
          <td>19.989515</td>
          <td>17.923648</td>
          <td>21.369168</td>
          <td>0.015330</td>
          <td>19.388102</td>
          <td>0.005367</td>
          <td>26.139019</td>
          <td>0.748386</td>
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
          <td>20.922410</td>
          <td>20.741286</td>
          <td>18.054376</td>
          <td>23.617085</td>
          <td>22.211575</td>
          <td>27.497711</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.472473</td>
          <td>0.014174</td>
          <td>22.095289</td>
          <td>0.026155</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.422812</td>
          <td>23.946753</td>
          <td>18.570381</td>
          <td>24.514421</td>
          <td>25.411920</td>
          <td>19.419784</td>
          <td>23.838243</td>
          <td>0.133462</td>
          <td>22.076085</td>
          <td>0.023566</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.407802</td>
          <td>26.604643</td>
          <td>30.203883</td>
          <td>29.770562</td>
          <td>26.844451</td>
          <td>26.189000</td>
          <td>21.548418</td>
          <td>0.017791</td>
          <td>24.156846</td>
          <td>0.147845</td>
          <td>17.013537</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.389247</td>
          <td>25.277501</td>
          <td>26.839553</td>
          <td>22.236270</td>
          <td>17.335306</td>
          <td>21.666841</td>
          <td>23.751685</td>
          <td>0.123808</td>
          <td>24.910346</td>
          <td>0.278066</td>
          <td>24.162902</td>
          <td>0.161922</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.522847</td>
          <td>20.053239</td>
          <td>22.040772</td>
          <td>26.985275</td>
          <td>23.529901</td>
          <td>26.859055</td>
          <td>18.178040</td>
          <td>0.005059</td>
          <td>26.231793</td>
          <td>0.744793</td>
          <td>24.537612</td>
          <td>0.222176</td>
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


