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
          <td>17.644769</td>
          <td>23.066166</td>
          <td>20.634206</td>
          <td>19.852561</td>
          <td>18.960320</td>
          <td>22.572659</td>
          <td>23.097253</td>
          <td>25.323536</td>
          <td>20.676154</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.447469</td>
          <td>22.585922</td>
          <td>23.703510</td>
          <td>27.149251</td>
          <td>26.873669</td>
          <td>24.916484</td>
          <td>16.498289</td>
          <td>25.560684</td>
          <td>26.465587</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.203896</td>
          <td>23.226567</td>
          <td>26.685361</td>
          <td>19.678976</td>
          <td>16.542683</td>
          <td>21.052884</td>
          <td>24.068370</td>
          <td>23.969762</td>
          <td>19.021257</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866318</td>
          <td>24.445126</td>
          <td>21.244090</td>
          <td>28.281555</td>
          <td>28.270819</td>
          <td>21.882979</td>
          <td>19.633125</td>
          <td>23.487495</td>
          <td>24.160315</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.680719</td>
          <td>22.647230</td>
          <td>27.305204</td>
          <td>18.653688</td>
          <td>22.187116</td>
          <td>24.924100</td>
          <td>25.752669</td>
          <td>22.039224</td>
          <td>21.896500</td>
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
          <td>20.052752</td>
          <td>22.176929</td>
          <td>25.319088</td>
          <td>21.845246</td>
          <td>25.626223</td>
          <td>26.487612</td>
          <td>20.870126</td>
          <td>23.432292</td>
          <td>22.843205</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.588330</td>
          <td>22.769508</td>
          <td>21.662983</td>
          <td>24.670326</td>
          <td>21.183230</td>
          <td>23.726540</td>
          <td>29.488878</td>
          <td>20.417394</td>
          <td>23.603806</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614436</td>
          <td>19.538015</td>
          <td>31.237526</td>
          <td>24.436210</td>
          <td>19.523758</td>
          <td>20.427191</td>
          <td>22.409538</td>
          <td>21.271746</td>
          <td>19.464684</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.667172</td>
          <td>24.589881</td>
          <td>21.462934</td>
          <td>22.386142</td>
          <td>22.025964</td>
          <td>21.982355</td>
          <td>22.341599</td>
          <td>20.530422</td>
          <td>24.929367</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.933109</td>
          <td>23.960372</td>
          <td>22.691724</td>
          <td>23.190226</td>
          <td>20.411303</td>
          <td>19.071241</td>
          <td>25.400527</td>
          <td>16.239219</td>
          <td>28.083914</td>
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
          <td>17.645163</td>
          <td>0.005020</td>
          <td>23.075385</td>
          <td>0.008264</td>
          <td>20.632929</td>
          <td>0.005057</td>
          <td>19.858683</td>
          <td>0.005041</td>
          <td>18.963407</td>
          <td>0.005034</td>
          <td>22.610669</td>
          <td>0.027681</td>
          <td>23.097253</td>
          <td>25.323536</td>
          <td>20.676154</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.446618</td>
          <td>0.005139</td>
          <td>22.592461</td>
          <td>0.006615</td>
          <td>23.701931</td>
          <td>0.011136</td>
          <td>27.356271</td>
          <td>0.392652</td>
          <td>26.626874</td>
          <td>0.393707</td>
          <td>24.692278</td>
          <td>0.172076</td>
          <td>16.498289</td>
          <td>25.560684</td>
          <td>26.465587</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.192887</td>
          <td>0.010421</td>
          <td>23.228630</td>
          <td>0.009036</td>
          <td>26.772145</td>
          <td>0.154546</td>
          <td>19.675215</td>
          <td>0.005032</td>
          <td>16.540860</td>
          <td>0.005002</td>
          <td>21.062058</td>
          <td>0.008403</td>
          <td>24.068370</td>
          <td>23.969762</td>
          <td>19.021257</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.881084</td>
          <td>0.017340</td>
          <td>24.421768</td>
          <td>0.022382</td>
          <td>21.239043</td>
          <td>0.005141</td>
          <td>27.920394</td>
          <td>0.596608</td>
          <td>26.695233</td>
          <td>0.414946</td>
          <td>21.881117</td>
          <td>0.014948</td>
          <td>19.633125</td>
          <td>23.487495</td>
          <td>24.160315</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.674790</td>
          <td>0.014735</td>
          <td>22.658510</td>
          <td>0.006783</td>
          <td>27.549619</td>
          <td>0.295442</td>
          <td>18.649502</td>
          <td>0.005009</td>
          <td>22.194126</td>
          <td>0.009575</td>
          <td>25.147561</td>
          <td>0.251828</td>
          <td>25.752669</td>
          <td>22.039224</td>
          <td>21.896500</td>
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
          <td>20.058168</td>
          <td>0.005299</td>
          <td>22.178540</td>
          <td>0.005860</td>
          <td>25.269777</td>
          <td>0.041267</td>
          <td>21.839462</td>
          <td>0.005893</td>
          <td>25.558974</td>
          <td>0.164544</td>
          <td>26.044945</td>
          <td>0.507754</td>
          <td>20.870126</td>
          <td>23.432292</td>
          <td>22.843205</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.598493</td>
          <td>0.005617</td>
          <td>22.771081</td>
          <td>0.007107</td>
          <td>21.669982</td>
          <td>0.005280</td>
          <td>24.665635</td>
          <td>0.039456</td>
          <td>21.173092</td>
          <td>0.006000</td>
          <td>23.751166</td>
          <td>0.075909</td>
          <td>29.488878</td>
          <td>20.417394</td>
          <td>23.603806</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.613663</td>
          <td>0.005170</td>
          <td>19.537750</td>
          <td>0.005022</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.446652</td>
          <td>0.032513</td>
          <td>19.521930</td>
          <td>0.005075</td>
          <td>20.428338</td>
          <td>0.006318</td>
          <td>22.409538</td>
          <td>21.271746</td>
          <td>19.464684</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.273968</td>
          <td>0.315556</td>
          <td>24.579175</td>
          <td>0.025626</td>
          <td>21.470288</td>
          <td>0.005203</td>
          <td>22.385689</td>
          <td>0.007095</td>
          <td>22.017644</td>
          <td>0.008582</td>
          <td>21.944000</td>
          <td>0.015725</td>
          <td>22.341599</td>
          <td>20.530422</td>
          <td>24.929367</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.730591</td>
          <td>0.202253</td>
          <td>23.952085</td>
          <td>0.015163</td>
          <td>22.693083</td>
          <td>0.006438</td>
          <td>23.185331</td>
          <td>0.011435</td>
          <td>20.414163</td>
          <td>0.005296</td>
          <td>19.079672</td>
          <td>0.005154</td>
          <td>25.400527</td>
          <td>16.239219</td>
          <td>28.083914</td>
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
          <td>17.644769</td>
          <td>23.066166</td>
          <td>20.634206</td>
          <td>19.852561</td>
          <td>18.960320</td>
          <td>22.572659</td>
          <td>23.105753</td>
          <td>0.007799</td>
          <td>25.329133</td>
          <td>0.078252</td>
          <td>20.666813</td>
          <td>0.005120</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.447469</td>
          <td>22.585922</td>
          <td>23.703510</td>
          <td>27.149251</td>
          <td>26.873669</td>
          <td>24.916484</td>
          <td>16.502784</td>
          <td>0.005000</td>
          <td>25.655064</td>
          <td>0.104283</td>
          <td>26.489303</td>
          <td>0.213400</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.203896</td>
          <td>23.226567</td>
          <td>26.685361</td>
          <td>19.678976</td>
          <td>16.542683</td>
          <td>21.052884</td>
          <td>24.051430</td>
          <td>0.015109</td>
          <td>23.939439</td>
          <td>0.022827</td>
          <td>19.022856</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866318</td>
          <td>24.445126</td>
          <td>21.244090</td>
          <td>28.281555</td>
          <td>28.270819</td>
          <td>21.882979</td>
          <td>19.633095</td>
          <td>0.005006</td>
          <td>23.456679</td>
          <td>0.015174</td>
          <td>24.176421</td>
          <td>0.028089</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.680719</td>
          <td>22.647230</td>
          <td>27.305204</td>
          <td>18.653688</td>
          <td>22.187116</td>
          <td>24.924100</td>
          <td>25.701791</td>
          <td>0.063955</td>
          <td>22.042893</td>
          <td>0.006347</td>
          <td>21.897329</td>
          <td>0.006058</td>
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
          <td>20.052752</td>
          <td>22.176929</td>
          <td>25.319088</td>
          <td>21.845246</td>
          <td>25.626223</td>
          <td>26.487612</td>
          <td>20.875114</td>
          <td>0.005059</td>
          <td>23.415265</td>
          <td>0.014672</td>
          <td>22.840506</td>
          <td>0.009554</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.588330</td>
          <td>22.769508</td>
          <td>21.662983</td>
          <td>24.670326</td>
          <td>21.183230</td>
          <td>23.726540</td>
          <td>28.218075</td>
          <td>0.519857</td>
          <td>20.415576</td>
          <td>0.005076</td>
          <td>23.630351</td>
          <td>0.017523</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614436</td>
          <td>19.538015</td>
          <td>31.237526</td>
          <td>24.436210</td>
          <td>19.523758</td>
          <td>20.427191</td>
          <td>22.410348</td>
          <td>0.005913</td>
          <td>21.276719</td>
          <td>0.005360</td>
          <td>19.462215</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.667172</td>
          <td>24.589881</td>
          <td>21.462934</td>
          <td>22.386142</td>
          <td>22.025964</td>
          <td>21.982355</td>
          <td>22.341870</td>
          <td>0.005813</td>
          <td>20.525299</td>
          <td>0.005093</td>
          <td>24.922079</td>
          <td>0.054500</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.933109</td>
          <td>23.960372</td>
          <td>22.691724</td>
          <td>23.190226</td>
          <td>20.411303</td>
          <td>19.071241</td>
          <td>25.446445</td>
          <td>0.050947</td>
          <td>16.239037</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>17.644769</td>
          <td>23.066166</td>
          <td>20.634206</td>
          <td>19.852561</td>
          <td>18.960320</td>
          <td>22.572659</td>
          <td>23.194507</td>
          <td>0.075888</td>
          <td>25.075060</td>
          <td>0.317518</td>
          <td>20.673969</td>
          <td>0.008591</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.447469</td>
          <td>22.585922</td>
          <td>23.703510</td>
          <td>27.149251</td>
          <td>26.873669</td>
          <td>24.916484</td>
          <td>16.501305</td>
          <td>0.005003</td>
          <td>24.898692</td>
          <td>0.275444</td>
          <td>27.229713</td>
          <td>1.423337</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.203896</td>
          <td>23.226567</td>
          <td>26.685361</td>
          <td>19.678976</td>
          <td>16.542683</td>
          <td>21.052884</td>
          <td>24.052756</td>
          <td>0.160524</td>
          <td>23.984790</td>
          <td>0.127419</td>
          <td>19.023513</td>
          <td>0.005229</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.866318</td>
          <td>24.445126</td>
          <td>21.244090</td>
          <td>28.281555</td>
          <td>28.270819</td>
          <td>21.882979</td>
          <td>19.630610</td>
          <td>0.005797</td>
          <td>23.708850</td>
          <td>0.100140</td>
          <td>23.857800</td>
          <td>0.124468</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.680719</td>
          <td>22.647230</td>
          <td>27.305204</td>
          <td>18.653688</td>
          <td>22.187116</td>
          <td>24.924100</td>
          <td>25.610508</td>
          <td>0.555958</td>
          <td>22.018985</td>
          <td>0.022426</td>
          <td>21.873284</td>
          <td>0.021556</td>
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
          <td>20.052752</td>
          <td>22.176929</td>
          <td>25.319088</td>
          <td>21.845246</td>
          <td>25.626223</td>
          <td>26.487612</td>
          <td>20.872864</td>
          <td>0.010465</td>
          <td>23.370171</td>
          <td>0.074269</td>
          <td>22.816758</td>
          <td>0.049616</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.588330</td>
          <td>22.769508</td>
          <td>21.662983</td>
          <td>24.670326</td>
          <td>21.183230</td>
          <td>23.726540</td>
          <td>26.549844</td>
          <td>1.032593</td>
          <td>20.407431</td>
          <td>0.007062</td>
          <td>23.428868</td>
          <td>0.085465</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614436</td>
          <td>19.538015</td>
          <td>31.237526</td>
          <td>24.436210</td>
          <td>19.523758</td>
          <td>20.427191</td>
          <td>22.419197</td>
          <td>0.038060</td>
          <td>21.274517</td>
          <td>0.012140</td>
          <td>19.471230</td>
          <td>0.005508</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.667172</td>
          <td>24.589881</td>
          <td>21.462934</td>
          <td>22.386142</td>
          <td>22.025964</td>
          <td>21.982355</td>
          <td>22.318734</td>
          <td>0.034807</td>
          <td>20.529886</td>
          <td>0.007494</td>
          <td>24.601431</td>
          <td>0.234267</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.933109</td>
          <td>23.960372</td>
          <td>22.691724</td>
          <td>23.190226</td>
          <td>20.411303</td>
          <td>19.071241</td>
          <td>25.007676</td>
          <td>0.352702</td>
          <td>16.238195</td>
          <td>0.005001</td>
          <td>inf</td>
          <td>inf</td>
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


