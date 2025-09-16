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
          <td>18.816928</td>
          <td>23.937941</td>
          <td>27.762716</td>
          <td>24.753470</td>
          <td>27.021402</td>
          <td>20.003104</td>
          <td>26.475610</td>
          <td>20.484899</td>
          <td>26.706021</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.946608</td>
          <td>24.057056</td>
          <td>29.288648</td>
          <td>23.620421</td>
          <td>23.449943</td>
          <td>22.572930</td>
          <td>23.920751</td>
          <td>25.853611</td>
          <td>25.826902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.700777</td>
          <td>16.791320</td>
          <td>23.708875</td>
          <td>21.289438</td>
          <td>19.500485</td>
          <td>23.079898</td>
          <td>24.297420</td>
          <td>25.188957</td>
          <td>19.950233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.686578</td>
          <td>26.845336</td>
          <td>19.519255</td>
          <td>21.029335</td>
          <td>20.553678</td>
          <td>25.234310</td>
          <td>21.159103</td>
          <td>22.448487</td>
          <td>18.466617</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.136033</td>
          <td>25.230603</td>
          <td>16.971726</td>
          <td>26.869363</td>
          <td>21.097982</td>
          <td>18.655425</td>
          <td>27.405526</td>
          <td>20.754799</td>
          <td>23.551280</td>
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
          <td>23.457445</td>
          <td>23.191423</td>
          <td>24.219589</td>
          <td>21.439201</td>
          <td>27.369999</td>
          <td>23.289502</td>
          <td>21.517047</td>
          <td>19.892057</td>
          <td>22.794083</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.455848</td>
          <td>19.658725</td>
          <td>19.619551</td>
          <td>20.539947</td>
          <td>21.658547</td>
          <td>16.922332</td>
          <td>24.848095</td>
          <td>21.134958</td>
          <td>22.956255</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.595605</td>
          <td>18.654401</td>
          <td>24.088321</td>
          <td>20.848170</td>
          <td>18.475739</td>
          <td>23.240084</td>
          <td>17.940335</td>
          <td>24.609852</td>
          <td>27.632923</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.541485</td>
          <td>19.470007</td>
          <td>26.329820</td>
          <td>25.841424</td>
          <td>20.868805</td>
          <td>20.999280</td>
          <td>27.339892</td>
          <td>17.044706</td>
          <td>19.911704</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882330</td>
          <td>19.965107</td>
          <td>20.987674</td>
          <td>24.363374</td>
          <td>28.597331</td>
          <td>19.658986</td>
          <td>22.646754</td>
          <td>22.965082</td>
          <td>23.087933</td>
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
          <td>18.822362</td>
          <td>0.005067</td>
          <td>23.938654</td>
          <td>0.015001</td>
          <td>28.048275</td>
          <td>0.436527</td>
          <td>24.749733</td>
          <td>0.042510</td>
          <td>26.520832</td>
          <td>0.362554</td>
          <td>20.008158</td>
          <td>0.005676</td>
          <td>26.475610</td>
          <td>20.484899</td>
          <td>26.706021</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.934090</td>
          <td>0.008885</td>
          <td>24.069398</td>
          <td>0.016668</td>
          <td>27.915874</td>
          <td>0.394477</td>
          <td>23.621091</td>
          <td>0.016035</td>
          <td>23.448213</td>
          <td>0.025777</td>
          <td>22.566442</td>
          <td>0.026633</td>
          <td>23.920751</td>
          <td>25.853611</td>
          <td>25.826902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.699058</td>
          <td>0.005709</td>
          <td>16.792944</td>
          <td>0.005001</td>
          <td>23.707910</td>
          <td>0.011184</td>
          <td>21.293479</td>
          <td>0.005370</td>
          <td>19.493323</td>
          <td>0.005072</td>
          <td>23.065985</td>
          <td>0.041347</td>
          <td>24.297420</td>
          <td>25.188957</td>
          <td>19.950233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.386340</td>
          <td>0.718589</td>
          <td>26.972615</td>
          <td>0.206975</td>
          <td>19.515388</td>
          <td>0.005013</td>
          <td>21.032609</td>
          <td>0.005244</td>
          <td>20.551270</td>
          <td>0.005369</td>
          <td>25.121521</td>
          <td>0.246494</td>
          <td>21.159103</td>
          <td>22.448487</td>
          <td>18.466617</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.072235</td>
          <td>0.577911</td>
          <td>25.174121</td>
          <td>0.043182</td>
          <td>16.977763</td>
          <td>0.005001</td>
          <td>26.671901</td>
          <td>0.226604</td>
          <td>21.088294</td>
          <td>0.005874</td>
          <td>18.651155</td>
          <td>0.005080</td>
          <td>27.405526</td>
          <td>20.754799</td>
          <td>23.551280</td>
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
          <td>23.498434</td>
          <td>0.029084</td>
          <td>23.191563</td>
          <td>0.008836</td>
          <td>24.243039</td>
          <td>0.016963</td>
          <td>21.438467</td>
          <td>0.005468</td>
          <td>26.960585</td>
          <td>0.506409</td>
          <td>23.286042</td>
          <td>0.050262</td>
          <td>21.517047</td>
          <td>19.892057</td>
          <td>22.794083</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.318851</td>
          <td>0.059603</td>
          <td>19.662813</td>
          <td>0.005025</td>
          <td>19.608945</td>
          <td>0.005014</td>
          <td>20.547184</td>
          <td>0.005114</td>
          <td>21.654474</td>
          <td>0.007109</td>
          <td>16.922756</td>
          <td>0.005008</td>
          <td>24.848095</td>
          <td>21.134958</td>
          <td>22.956255</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.580067</td>
          <td>0.013708</td>
          <td>18.662756</td>
          <td>0.005008</td>
          <td>24.106496</td>
          <td>0.015176</td>
          <td>20.846001</td>
          <td>0.005181</td>
          <td>18.477424</td>
          <td>0.005018</td>
          <td>23.282215</td>
          <td>0.050092</td>
          <td>17.940335</td>
          <td>24.609852</td>
          <td>27.632923</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.546928</td>
          <td>0.007310</td>
          <td>19.472665</td>
          <td>0.005020</td>
          <td>26.739316</td>
          <td>0.150256</td>
          <td>25.918861</td>
          <td>0.119312</td>
          <td>20.872220</td>
          <td>0.005618</td>
          <td>21.009891</td>
          <td>0.008159</td>
          <td>27.339892</td>
          <td>17.044706</td>
          <td>19.911704</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.893299</td>
          <td>0.008682</td>
          <td>19.969390</td>
          <td>0.005037</td>
          <td>20.988687</td>
          <td>0.005096</td>
          <td>24.354416</td>
          <td>0.029980</td>
          <td>26.988552</td>
          <td>0.516915</td>
          <td>19.665081</td>
          <td>0.005390</td>
          <td>22.646754</td>
          <td>22.965082</td>
          <td>23.087933</td>
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
          <td>18.816928</td>
          <td>23.937941</td>
          <td>27.762716</td>
          <td>24.753470</td>
          <td>27.021402</td>
          <td>20.003104</td>
          <td>26.452902</td>
          <td>0.123939</td>
          <td>20.493546</td>
          <td>0.005087</td>
          <td>26.570294</td>
          <td>0.228296</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.946608</td>
          <td>24.057056</td>
          <td>29.288648</td>
          <td>23.620421</td>
          <td>23.449943</td>
          <td>22.572930</td>
          <td>23.912387</td>
          <td>0.013512</td>
          <td>25.848407</td>
          <td>0.123456</td>
          <td>25.805168</td>
          <td>0.118897</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.700777</td>
          <td>16.791320</td>
          <td>23.708875</td>
          <td>21.289438</td>
          <td>19.500485</td>
          <td>23.079898</td>
          <td>24.314862</td>
          <td>0.018820</td>
          <td>25.219331</td>
          <td>0.070995</td>
          <td>19.947900</td>
          <td>0.005032</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.686578</td>
          <td>26.845336</td>
          <td>19.519255</td>
          <td>21.029335</td>
          <td>20.553678</td>
          <td>25.234310</td>
          <td>21.154423</td>
          <td>0.005098</td>
          <td>22.442267</td>
          <td>0.007542</td>
          <td>18.465182</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.136033</td>
          <td>25.230603</td>
          <td>16.971726</td>
          <td>26.869363</td>
          <td>21.097982</td>
          <td>18.655425</td>
          <td>28.244222</td>
          <td>0.529880</td>
          <td>20.756666</td>
          <td>0.005141</td>
          <td>23.519705</td>
          <td>0.015980</td>
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
          <td>23.457445</td>
          <td>23.191423</td>
          <td>24.219589</td>
          <td>21.439201</td>
          <td>27.369999</td>
          <td>23.289502</td>
          <td>21.516938</td>
          <td>0.005189</td>
          <td>19.890149</td>
          <td>0.005029</td>
          <td>22.807009</td>
          <td>0.009345</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.455848</td>
          <td>19.658725</td>
          <td>19.619551</td>
          <td>20.539947</td>
          <td>21.658547</td>
          <td>16.922332</td>
          <td>24.834006</td>
          <td>0.029553</td>
          <td>21.136650</td>
          <td>0.005280</td>
          <td>22.948609</td>
          <td>0.010288</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.595605</td>
          <td>18.654401</td>
          <td>24.088321</td>
          <td>20.848170</td>
          <td>18.475739</td>
          <td>23.240084</td>
          <td>17.936192</td>
          <td>0.005000</td>
          <td>24.617373</td>
          <td>0.041537</td>
          <td>30.015078</td>
          <td>2.046471</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.541485</td>
          <td>19.470007</td>
          <td>26.329820</td>
          <td>25.841424</td>
          <td>20.868805</td>
          <td>20.999280</td>
          <td>27.039983</td>
          <td>0.204766</td>
          <td>17.055090</td>
          <td>0.005000</td>
          <td>19.908927</td>
          <td>0.005030</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882330</td>
          <td>19.965107</td>
          <td>20.987674</td>
          <td>24.363374</td>
          <td>28.597331</td>
          <td>19.658986</td>
          <td>22.649118</td>
          <td>0.006361</td>
          <td>22.978702</td>
          <td>0.010509</td>
          <td>23.080651</td>
          <td>0.011314</td>
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
          <td>18.816928</td>
          <td>23.937941</td>
          <td>27.762716</td>
          <td>24.753470</td>
          <td>27.021402</td>
          <td>20.003104</td>
          <td>26.582071</td>
          <td>1.052482</td>
          <td>20.484661</td>
          <td>0.007326</td>
          <td>26.207071</td>
          <td>0.782813</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.946608</td>
          <td>24.057056</td>
          <td>29.288648</td>
          <td>23.620421</td>
          <td>23.449943</td>
          <td>22.572930</td>
          <td>24.008421</td>
          <td>0.154542</td>
          <td>26.017402</td>
          <td>0.643652</td>
          <td>25.396793</td>
          <td>0.440892</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.700777</td>
          <td>16.791320</td>
          <td>23.708875</td>
          <td>21.289438</td>
          <td>19.500485</td>
          <td>23.079898</td>
          <td>24.214667</td>
          <td>0.184241</td>
          <td>25.123041</td>
          <td>0.329884</td>
          <td>19.954513</td>
          <td>0.006164</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.686578</td>
          <td>26.845336</td>
          <td>19.519255</td>
          <td>21.029335</td>
          <td>20.553678</td>
          <td>25.234310</td>
          <td>21.157887</td>
          <td>0.012946</td>
          <td>22.467559</td>
          <td>0.033262</td>
          <td>18.470625</td>
          <td>0.005084</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.136033</td>
          <td>25.230603</td>
          <td>16.971726</td>
          <td>26.869363</td>
          <td>21.097982</td>
          <td>18.655425</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.745405</td>
          <td>0.008445</td>
          <td>23.505001</td>
          <td>0.091400</td>
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
          <td>23.457445</td>
          <td>23.191423</td>
          <td>24.219589</td>
          <td>21.439201</td>
          <td>27.369999</td>
          <td>23.289502</td>
          <td>21.520151</td>
          <td>0.017373</td>
          <td>19.885592</td>
          <td>0.005876</td>
          <td>22.849502</td>
          <td>0.051086</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.455848</td>
          <td>19.658725</td>
          <td>19.619551</td>
          <td>20.539947</td>
          <td>21.658547</td>
          <td>16.922332</td>
          <td>24.645571</td>
          <td>0.263768</td>
          <td>21.140566</td>
          <td>0.010987</td>
          <td>22.936383</td>
          <td>0.055199</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.595605</td>
          <td>18.654401</td>
          <td>24.088321</td>
          <td>20.848170</td>
          <td>18.475739</td>
          <td>23.240084</td>
          <td>17.936661</td>
          <td>0.005038</td>
          <td>24.018632</td>
          <td>0.131214</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.541485</td>
          <td>19.470007</td>
          <td>26.329820</td>
          <td>25.841424</td>
          <td>20.868805</td>
          <td>20.999280</td>
          <td>29.693418</td>
          <td>3.683136</td>
          <td>17.046062</td>
          <td>0.005005</td>
          <td>19.908000</td>
          <td>0.006077</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882330</td>
          <td>19.965107</td>
          <td>20.987674</td>
          <td>24.363374</td>
          <td>28.597331</td>
          <td>19.658986</td>
          <td>22.592910</td>
          <td>0.044429</td>
          <td>22.988437</td>
          <td>0.052890</td>
          <td>23.196114</td>
          <td>0.069547</td>
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


