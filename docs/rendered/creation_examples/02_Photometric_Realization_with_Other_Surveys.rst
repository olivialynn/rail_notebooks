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
          <td>25.864495</td>
          <td>22.825097</td>
          <td>23.481321</td>
          <td>22.338529</td>
          <td>28.560314</td>
          <td>26.555811</td>
          <td>25.705134</td>
          <td>24.355666</td>
          <td>24.978662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.500274</td>
          <td>21.188627</td>
          <td>20.926649</td>
          <td>19.115199</td>
          <td>24.949448</td>
          <td>27.496791</td>
          <td>25.452905</td>
          <td>26.770774</td>
          <td>22.666354</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.052697</td>
          <td>23.286710</td>
          <td>21.464981</td>
          <td>23.795796</td>
          <td>21.949500</td>
          <td>25.098685</td>
          <td>24.976311</td>
          <td>17.103028</td>
          <td>24.436069</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.214034</td>
          <td>21.646913</td>
          <td>24.057536</td>
          <td>23.319123</td>
          <td>20.707222</td>
          <td>26.327545</td>
          <td>22.111660</td>
          <td>20.019056</td>
          <td>27.950765</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.708102</td>
          <td>22.181950</td>
          <td>22.963408</td>
          <td>22.177249</td>
          <td>19.656740</td>
          <td>27.255377</td>
          <td>17.312663</td>
          <td>21.336629</td>
          <td>21.058746</td>
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
          <td>22.627297</td>
          <td>25.254710</td>
          <td>19.054575</td>
          <td>18.681216</td>
          <td>25.787602</td>
          <td>26.188164</td>
          <td>18.109774</td>
          <td>26.085918</td>
          <td>22.932101</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.484296</td>
          <td>18.892812</td>
          <td>21.914096</td>
          <td>19.191251</td>
          <td>21.147010</td>
          <td>27.300092</td>
          <td>23.210109</td>
          <td>22.589707</td>
          <td>25.920172</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.731855</td>
          <td>20.111306</td>
          <td>24.630469</td>
          <td>17.802449</td>
          <td>23.346510</td>
          <td>21.978274</td>
          <td>21.002291</td>
          <td>28.018236</td>
          <td>19.261280</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.447586</td>
          <td>25.667707</td>
          <td>24.938249</td>
          <td>26.474372</td>
          <td>21.009740</td>
          <td>23.485851</td>
          <td>22.986156</td>
          <td>23.421234</td>
          <td>23.394250</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.451686</td>
          <td>25.382844</td>
          <td>22.237538</td>
          <td>23.784133</td>
          <td>22.351991</td>
          <td>21.953663</td>
          <td>18.921841</td>
          <td>27.149037</td>
          <td>20.110440</td>
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
          <td>25.939723</td>
          <td>0.240624</td>
          <td>22.818379</td>
          <td>0.007258</td>
          <td>23.481554</td>
          <td>0.009574</td>
          <td>22.342111</td>
          <td>0.006961</td>
          <td>27.736249</td>
          <td>0.862879</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.705134</td>
          <td>24.355666</td>
          <td>24.978662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.495600</td>
          <td>0.005147</td>
          <td>21.183945</td>
          <td>0.005190</td>
          <td>20.923054</td>
          <td>0.005087</td>
          <td>19.117313</td>
          <td>0.005015</td>
          <td>24.962736</td>
          <td>0.098186</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.452905</td>
          <td>26.770774</td>
          <td>22.666354</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.053439</td>
          <td>0.005297</td>
          <td>23.282042</td>
          <td>0.009340</td>
          <td>21.468445</td>
          <td>0.005203</td>
          <td>23.816183</td>
          <td>0.018848</td>
          <td>21.961878</td>
          <td>0.008309</td>
          <td>24.875679</td>
          <td>0.200920</td>
          <td>24.976311</td>
          <td>17.103028</td>
          <td>24.436069</td>
        </tr>
        <tr>
          <th>3</th>
          <td>inf</td>
          <td>inf</td>
          <td>21.639470</td>
          <td>0.005375</td>
          <td>24.052096</td>
          <td>0.014529</td>
          <td>23.307005</td>
          <td>0.012517</td>
          <td>20.706803</td>
          <td>0.005474</td>
          <td>26.235391</td>
          <td>0.582826</td>
          <td>22.111660</td>
          <td>20.019056</td>
          <td>27.950765</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.823266</td>
          <td>0.218514</td>
          <td>22.180325</td>
          <td>0.005862</td>
          <td>22.962177</td>
          <td>0.007171</td>
          <td>22.188096</td>
          <td>0.006548</td>
          <td>19.665631</td>
          <td>0.005092</td>
          <td>31.759431</td>
          <td>5.162063</td>
          <td>17.312663</td>
          <td>21.336629</td>
          <td>21.058746</td>
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
          <td>22.636688</td>
          <td>0.014310</td>
          <td>25.274890</td>
          <td>0.047210</td>
          <td>19.056232</td>
          <td>0.005007</td>
          <td>18.681634</td>
          <td>0.005009</td>
          <td>25.890273</td>
          <td>0.217602</td>
          <td>25.731247</td>
          <td>0.400941</td>
          <td>18.109774</td>
          <td>26.085918</td>
          <td>22.932101</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.498761</td>
          <td>0.007162</td>
          <td>18.898217</td>
          <td>0.005011</td>
          <td>21.919270</td>
          <td>0.005418</td>
          <td>19.186180</td>
          <td>0.005017</td>
          <td>21.149398</td>
          <td>0.005963</td>
          <td>25.898269</td>
          <td>0.455278</td>
          <td>23.210109</td>
          <td>22.589707</td>
          <td>25.920172</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.732217</td>
          <td>0.085704</td>
          <td>20.107302</td>
          <td>0.005043</td>
          <td>24.659682</td>
          <td>0.024146</td>
          <td>17.803821</td>
          <td>0.005003</td>
          <td>23.279707</td>
          <td>0.022284</td>
          <td>21.951529</td>
          <td>0.015822</td>
          <td>21.002291</td>
          <td>28.018236</td>
          <td>19.261280</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.061489</td>
          <td>0.573493</td>
          <td>25.644903</td>
          <td>0.065513</td>
          <td>24.934592</td>
          <td>0.030691</td>
          <td>26.853696</td>
          <td>0.263215</td>
          <td>21.006270</td>
          <td>0.005767</td>
          <td>23.422118</td>
          <td>0.056715</td>
          <td>22.986156</td>
          <td>23.421234</td>
          <td>23.394250</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.154783</td>
          <td>0.123826</td>
          <td>25.396195</td>
          <td>0.052565</td>
          <td>22.243235</td>
          <td>0.005704</td>
          <td>23.764391</td>
          <td>0.018048</td>
          <td>22.356268</td>
          <td>0.010685</td>
          <td>21.949288</td>
          <td>0.015793</td>
          <td>18.921841</td>
          <td>27.149037</td>
          <td>20.110440</td>
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
          <td>25.864495</td>
          <td>22.825097</td>
          <td>23.481321</td>
          <td>22.338529</td>
          <td>28.560314</td>
          <td>26.555811</td>
          <td>25.724937</td>
          <td>0.065285</td>
          <td>24.350570</td>
          <td>0.032764</td>
          <td>24.963318</td>
          <td>0.056539</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.500274</td>
          <td>21.188627</td>
          <td>20.926649</td>
          <td>19.115199</td>
          <td>24.949448</td>
          <td>27.496791</td>
          <td>25.447657</td>
          <td>0.051002</td>
          <td>26.606101</td>
          <td>0.235175</td>
          <td>22.666115</td>
          <td>0.008550</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.052697</td>
          <td>23.286710</td>
          <td>21.464981</td>
          <td>23.795796</td>
          <td>21.949500</td>
          <td>25.098685</td>
          <td>24.941829</td>
          <td>0.032511</td>
          <td>17.108530</td>
          <td>0.005000</td>
          <td>24.482486</td>
          <td>0.036837</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.214034</td>
          <td>21.646913</td>
          <td>24.057536</td>
          <td>23.319123</td>
          <td>20.707222</td>
          <td>26.327545</td>
          <td>22.103982</td>
          <td>0.005538</td>
          <td>20.031131</td>
          <td>0.005037</td>
          <td>28.172041</td>
          <td>0.764958</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.708102</td>
          <td>22.181950</td>
          <td>22.963408</td>
          <td>22.177249</td>
          <td>19.656740</td>
          <td>27.255377</td>
          <td>17.317487</td>
          <td>0.005000</td>
          <td>21.335370</td>
          <td>0.005400</td>
          <td>21.053477</td>
          <td>0.005241</td>
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
          <td>22.627297</td>
          <td>25.254710</td>
          <td>19.054575</td>
          <td>18.681216</td>
          <td>25.787602</td>
          <td>26.188164</td>
          <td>18.104921</td>
          <td>0.005000</td>
          <td>26.006424</td>
          <td>0.141561</td>
          <td>22.913320</td>
          <td>0.010038</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.484296</td>
          <td>18.892812</td>
          <td>21.914096</td>
          <td>19.191251</td>
          <td>21.147010</td>
          <td>27.300092</td>
          <td>23.210720</td>
          <td>0.008274</td>
          <td>22.592541</td>
          <td>0.008187</td>
          <td>25.864016</td>
          <td>0.125142</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.731855</td>
          <td>20.111306</td>
          <td>24.630469</td>
          <td>17.802449</td>
          <td>23.346510</td>
          <td>21.978274</td>
          <td>21.015097</td>
          <td>0.005076</td>
          <td>28.844265</td>
          <td>1.155968</td>
          <td>19.269786</td>
          <td>0.005009</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.447586</td>
          <td>25.667707</td>
          <td>24.938249</td>
          <td>26.474372</td>
          <td>21.009740</td>
          <td>23.485851</td>
          <td>22.989555</td>
          <td>0.007344</td>
          <td>23.396323</td>
          <td>0.014449</td>
          <td>23.407778</td>
          <td>0.014583</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.451686</td>
          <td>25.382844</td>
          <td>22.237538</td>
          <td>23.784133</td>
          <td>22.351991</td>
          <td>21.953663</td>
          <td>18.914961</td>
          <td>0.005002</td>
          <td>26.767598</td>
          <td>0.268556</td>
          <td>20.120129</td>
          <td>0.005044</td>
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
          <td>25.864495</td>
          <td>22.825097</td>
          <td>23.481321</td>
          <td>22.338529</td>
          <td>28.560314</td>
          <td>26.555811</td>
          <td>26.534376</td>
          <td>1.023127</td>
          <td>24.128826</td>
          <td>0.144321</td>
          <td>24.933556</td>
          <td>0.307145</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.500274</td>
          <td>21.188627</td>
          <td>20.926649</td>
          <td>19.115199</td>
          <td>24.949448</td>
          <td>27.496791</td>
          <td>25.222819</td>
          <td>0.416762</td>
          <td>27.958321</td>
          <td>1.915024</td>
          <td>22.657846</td>
          <td>0.043062</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.052697</td>
          <td>23.286710</td>
          <td>21.464981</td>
          <td>23.795796</td>
          <td>21.949500</td>
          <td>25.098685</td>
          <td>24.501341</td>
          <td>0.234250</td>
          <td>17.100364</td>
          <td>0.005006</td>
          <td>24.586955</td>
          <td>0.231474</td>
        </tr>
        <tr>
          <th>3</th>
          <td>29.214034</td>
          <td>21.646913</td>
          <td>24.057536</td>
          <td>23.319123</td>
          <td>20.707222</td>
          <td>26.327545</td>
          <td>22.118004</td>
          <td>0.029138</td>
          <td>20.011066</td>
          <td>0.006082</td>
          <td>26.596195</td>
          <td>0.999986</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.708102</td>
          <td>22.181950</td>
          <td>22.963408</td>
          <td>22.177249</td>
          <td>19.656740</td>
          <td>27.255377</td>
          <td>17.307959</td>
          <td>0.005012</td>
          <td>21.334908</td>
          <td>0.012716</td>
          <td>21.059499</td>
          <td>0.011140</td>
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
          <td>22.627297</td>
          <td>25.254710</td>
          <td>19.054575</td>
          <td>18.681216</td>
          <td>25.787602</td>
          <td>26.188164</td>
          <td>18.111020</td>
          <td>0.005052</td>
          <td>32.160359</td>
          <td>5.917608</td>
          <td>23.016881</td>
          <td>0.059301</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.484296</td>
          <td>18.892812</td>
          <td>21.914096</td>
          <td>19.191251</td>
          <td>21.147010</td>
          <td>27.300092</td>
          <td>23.254251</td>
          <td>0.080010</td>
          <td>22.590631</td>
          <td>0.037105</td>
          <td>25.230168</td>
          <td>0.388080</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.731855</td>
          <td>20.111306</td>
          <td>24.630469</td>
          <td>17.802449</td>
          <td>23.346510</td>
          <td>21.978274</td>
          <td>21.011750</td>
          <td>0.011578</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.259156</td>
          <td>0.005349</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.447586</td>
          <td>25.667707</td>
          <td>24.938249</td>
          <td>26.474372</td>
          <td>21.009740</td>
          <td>23.485851</td>
          <td>22.933185</td>
          <td>0.060168</td>
          <td>23.308785</td>
          <td>0.070333</td>
          <td>23.366150</td>
          <td>0.080857</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.451686</td>
          <td>25.382844</td>
          <td>22.237538</td>
          <td>23.784133</td>
          <td>22.351991</td>
          <td>21.953663</td>
          <td>18.924245</td>
          <td>0.005229</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.112793</td>
          <td>0.006511</td>
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


