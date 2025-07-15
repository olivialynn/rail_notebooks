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
          <td>27.433521</td>
          <td>27.044670</td>
          <td>25.984202</td>
          <td>25.198828</td>
          <td>22.995209</td>
          <td>20.012789</td>
          <td>23.964342</td>
          <td>27.342358</td>
          <td>23.779407</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.963827</td>
          <td>25.695087</td>
          <td>19.645182</td>
          <td>22.102009</td>
          <td>23.831680</td>
          <td>19.510954</td>
          <td>25.952206</td>
          <td>18.266626</td>
          <td>23.014546</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.607916</td>
          <td>20.943579</td>
          <td>18.431919</td>
          <td>17.031859</td>
          <td>22.088017</td>
          <td>25.023987</td>
          <td>24.587757</td>
          <td>21.138772</td>
          <td>24.520663</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.019782</td>
          <td>27.457085</td>
          <td>27.083430</td>
          <td>25.638170</td>
          <td>22.810262</td>
          <td>25.852571</td>
          <td>21.587041</td>
          <td>24.058018</td>
          <td>18.195438</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.867929</td>
          <td>27.596501</td>
          <td>23.729817</td>
          <td>17.518317</td>
          <td>20.281433</td>
          <td>18.146421</td>
          <td>23.112279</td>
          <td>27.757053</td>
          <td>22.727489</td>
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
          <td>21.155664</td>
          <td>26.087064</td>
          <td>20.214989</td>
          <td>22.879998</td>
          <td>18.120324</td>
          <td>25.030076</td>
          <td>16.864812</td>
          <td>26.579495</td>
          <td>27.687968</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.501453</td>
          <td>23.946520</td>
          <td>19.236138</td>
          <td>22.913552</td>
          <td>24.944796</td>
          <td>21.960215</td>
          <td>21.469563</td>
          <td>27.668838</td>
          <td>26.791490</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.790431</td>
          <td>20.049997</td>
          <td>24.234726</td>
          <td>27.182791</td>
          <td>25.824225</td>
          <td>23.667918</td>
          <td>19.551239</td>
          <td>21.863577</td>
          <td>20.772878</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.957052</td>
          <td>27.728127</td>
          <td>25.051989</td>
          <td>18.671677</td>
          <td>27.480352</td>
          <td>23.678487</td>
          <td>28.002452</td>
          <td>22.086048</td>
          <td>20.714742</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.243280</td>
          <td>16.692305</td>
          <td>13.592517</td>
          <td>19.184247</td>
          <td>18.314601</td>
          <td>24.324137</td>
          <td>33.840865</td>
          <td>24.734157</td>
          <td>27.985534</td>
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
          <td>26.337688</td>
          <td>0.331949</td>
          <td>27.213043</td>
          <td>0.252624</td>
          <td>26.023688</td>
          <td>0.080512</td>
          <td>25.234882</td>
          <td>0.065387</td>
          <td>22.949326</td>
          <td>0.016870</td>
          <td>20.012999</td>
          <td>0.005682</td>
          <td>23.964342</td>
          <td>27.342358</td>
          <td>23.779407</td>
        </tr>
        <tr>
          <th>1</th>
          <td>inf</td>
          <td>inf</td>
          <td>25.619579</td>
          <td>0.064062</td>
          <td>19.641190</td>
          <td>0.005015</td>
          <td>22.104376</td>
          <td>0.006359</td>
          <td>23.796474</td>
          <td>0.034985</td>
          <td>19.508719</td>
          <td>0.005303</td>
          <td>25.952206</td>
          <td>18.266626</td>
          <td>23.014546</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.629484</td>
          <td>0.014231</td>
          <td>20.937311</td>
          <td>0.005133</td>
          <td>18.431604</td>
          <td>0.005004</td>
          <td>17.036631</td>
          <td>0.005002</td>
          <td>22.097216</td>
          <td>0.009004</td>
          <td>25.029321</td>
          <td>0.228411</td>
          <td>24.587757</td>
          <td>21.138772</td>
          <td>24.520663</td>
        </tr>
        <tr>
          <th>3</th>
          <td>28.084861</td>
          <td>1.112057</td>
          <td>27.736446</td>
          <td>0.383785</td>
          <td>27.044047</td>
          <td>0.194705</td>
          <td>25.802327</td>
          <td>0.107789</td>
          <td>22.803083</td>
          <td>0.014979</td>
          <td>25.328830</td>
          <td>0.291875</td>
          <td>21.587041</td>
          <td>24.058018</td>
          <td>18.195438</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.861835</td>
          <td>0.005070</td>
          <td>27.934937</td>
          <td>0.446721</td>
          <td>23.734912</td>
          <td>0.011405</td>
          <td>17.515061</td>
          <td>0.005002</td>
          <td>20.289021</td>
          <td>0.005243</td>
          <td>18.155510</td>
          <td>0.005039</td>
          <td>23.112279</td>
          <td>27.757053</td>
          <td>22.727489</td>
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
          <td>21.155411</td>
          <td>0.006341</td>
          <td>26.083871</td>
          <td>0.096444</td>
          <td>20.209073</td>
          <td>0.005031</td>
          <td>22.874337</td>
          <td>0.009246</td>
          <td>18.123973</td>
          <td>0.005011</td>
          <td>24.865534</td>
          <td>0.199215</td>
          <td>16.864812</td>
          <td>26.579495</td>
          <td>27.687968</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.423992</td>
          <td>0.156055</td>
          <td>23.939784</td>
          <td>0.015015</td>
          <td>19.236076</td>
          <td>0.005009</td>
          <td>22.926744</td>
          <td>0.009563</td>
          <td>24.787486</td>
          <td>0.084168</td>
          <td>21.986801</td>
          <td>0.016283</td>
          <td>21.469563</td>
          <td>27.668838</td>
          <td>26.791490</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.852221</td>
          <td>0.039557</td>
          <td>20.057231</td>
          <td>0.005041</td>
          <td>24.224606</td>
          <td>0.016707</td>
          <td>27.532541</td>
          <td>0.449202</td>
          <td>25.803684</td>
          <td>0.202402</td>
          <td>23.631404</td>
          <td>0.068279</td>
          <td>19.551239</td>
          <td>21.863577</td>
          <td>20.772878</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.968019</td>
          <td>0.009062</td>
          <td>28.541465</td>
          <td>0.690934</td>
          <td>25.050291</td>
          <td>0.033984</td>
          <td>18.674875</td>
          <td>0.005009</td>
          <td>26.726412</td>
          <td>0.424941</td>
          <td>23.759845</td>
          <td>0.076493</td>
          <td>28.002452</td>
          <td>22.086048</td>
          <td>20.714742</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.453539</td>
          <td>0.160038</td>
          <td>16.694291</td>
          <td>0.005001</td>
          <td>13.588611</td>
          <td>0.005000</td>
          <td>19.184974</td>
          <td>0.005017</td>
          <td>18.317885</td>
          <td>0.005015</td>
          <td>24.273134</td>
          <td>0.119986</td>
          <td>33.840865</td>
          <td>24.734157</td>
          <td>27.985534</td>
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
          <td>27.433521</td>
          <td>27.044670</td>
          <td>25.984202</td>
          <td>25.198828</td>
          <td>22.995209</td>
          <td>20.012789</td>
          <td>23.992875</td>
          <td>0.014409</td>
          <td>27.118896</td>
          <td>0.355825</td>
          <td>23.772062</td>
          <td>0.019760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.963827</td>
          <td>25.695087</td>
          <td>19.645182</td>
          <td>22.102009</td>
          <td>23.831680</td>
          <td>19.510954</td>
          <td>26.007876</td>
          <td>0.083895</td>
          <td>18.266995</td>
          <td>0.005001</td>
          <td>23.012770</td>
          <td>0.010768</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.607916</td>
          <td>20.943579</td>
          <td>18.431919</td>
          <td>17.031859</td>
          <td>22.088017</td>
          <td>25.023987</td>
          <td>24.572655</td>
          <td>0.023496</td>
          <td>21.147003</td>
          <td>0.005286</td>
          <td>24.481304</td>
          <td>0.036798</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.019782</td>
          <td>27.457085</td>
          <td>27.083430</td>
          <td>25.638170</td>
          <td>22.810262</td>
          <td>25.852571</td>
          <td>21.586712</td>
          <td>0.005214</td>
          <td>24.050257</td>
          <td>0.025143</td>
          <td>18.205176</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.867929</td>
          <td>27.596501</td>
          <td>23.729817</td>
          <td>17.518317</td>
          <td>20.281433</td>
          <td>18.146421</td>
          <td>23.119214</td>
          <td>0.007857</td>
          <td>29.757140</td>
          <td>1.831868</td>
          <td>22.733979</td>
          <td>0.008916</td>
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
          <td>21.155664</td>
          <td>26.087064</td>
          <td>20.214989</td>
          <td>22.879998</td>
          <td>18.120324</td>
          <td>25.030076</td>
          <td>16.866147</td>
          <td>0.005000</td>
          <td>26.497455</td>
          <td>0.214859</td>
          <td>28.227968</td>
          <td>0.793598</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.501453</td>
          <td>23.946520</td>
          <td>19.236138</td>
          <td>22.913552</td>
          <td>24.944796</td>
          <td>21.960215</td>
          <td>21.480583</td>
          <td>0.005177</td>
          <td>28.733785</td>
          <td>1.084869</td>
          <td>27.261983</td>
          <td>0.397739</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.790431</td>
          <td>20.049997</td>
          <td>24.234726</td>
          <td>27.182791</td>
          <td>25.824225</td>
          <td>23.667918</td>
          <td>19.554786</td>
          <td>0.005005</td>
          <td>21.864460</td>
          <td>0.006001</td>
          <td>20.771871</td>
          <td>0.005145</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.957052</td>
          <td>27.728127</td>
          <td>25.051989</td>
          <td>18.671677</td>
          <td>27.480352</td>
          <td>23.678487</td>
          <td>27.768307</td>
          <td>0.369859</td>
          <td>22.084825</td>
          <td>0.006443</td>
          <td>20.712183</td>
          <td>0.005130</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.243280</td>
          <td>16.692305</td>
          <td>13.592517</td>
          <td>19.184247</td>
          <td>18.314601</td>
          <td>24.324137</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.801918</td>
          <td>0.048964</td>
          <td>27.658212</td>
          <td>0.535303</td>
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
          <td>27.433521</td>
          <td>27.044670</td>
          <td>25.984202</td>
          <td>25.198828</td>
          <td>22.995209</td>
          <td>20.012789</td>
          <td>23.944675</td>
          <td>0.146304</td>
          <td>28.007913</td>
          <td>1.956275</td>
          <td>23.903519</td>
          <td>0.129507</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.963827</td>
          <td>25.695087</td>
          <td>19.645182</td>
          <td>22.102009</td>
          <td>23.831680</td>
          <td>19.510954</td>
          <td>26.809069</td>
          <td>1.198860</td>
          <td>18.268724</td>
          <td>0.005048</td>
          <td>23.021863</td>
          <td>0.059565</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.607916</td>
          <td>20.943579</td>
          <td>18.431919</td>
          <td>17.031859</td>
          <td>22.088017</td>
          <td>25.023987</td>
          <td>24.280703</td>
          <td>0.194810</td>
          <td>21.141696</td>
          <td>0.010996</td>
          <td>24.333462</td>
          <td>0.187195</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.019782</td>
          <td>27.457085</td>
          <td>27.083430</td>
          <td>25.638170</td>
          <td>22.810262</td>
          <td>25.852571</td>
          <td>21.539020</td>
          <td>0.017651</td>
          <td>24.198807</td>
          <td>0.153272</td>
          <td>18.188801</td>
          <td>0.005050</td>
        </tr>
        <tr>
          <th>4</th>
          <td>18.867929</td>
          <td>27.596501</td>
          <td>23.729817</td>
          <td>17.518317</td>
          <td>20.281433</td>
          <td>18.146421</td>
          <td>23.071616</td>
          <td>0.068050</td>
          <td>26.053436</td>
          <td>0.659915</td>
          <td>22.733607</td>
          <td>0.046071</td>
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
          <td>21.155664</td>
          <td>26.087064</td>
          <td>20.214989</td>
          <td>22.879998</td>
          <td>18.120324</td>
          <td>25.030076</td>
          <td>16.866318</td>
          <td>0.005005</td>
          <td>28.127698</td>
          <td>2.057183</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.501453</td>
          <td>23.946520</td>
          <td>19.236138</td>
          <td>22.913552</td>
          <td>24.944796</td>
          <td>21.960215</td>
          <td>21.467083</td>
          <td>0.016620</td>
          <td>26.217945</td>
          <td>0.737941</td>
          <td>27.033524</td>
          <td>1.283620</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.790431</td>
          <td>20.049997</td>
          <td>24.234726</td>
          <td>27.182791</td>
          <td>25.824225</td>
          <td>23.667918</td>
          <td>19.552849</td>
          <td>0.005697</td>
          <td>21.855238</td>
          <td>0.019478</td>
          <td>20.783310</td>
          <td>0.009202</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.957052</td>
          <td>27.728127</td>
          <td>25.051989</td>
          <td>18.671677</td>
          <td>27.480352</td>
          <td>23.678487</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.081646</td>
          <td>0.023681</td>
          <td>20.709922</td>
          <td>0.008783</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.243280</td>
          <td>16.692305</td>
          <td>13.592517</td>
          <td>19.184247</td>
          <td>18.314601</td>
          <td>24.324137</td>
          <td>26.689028</td>
          <td>1.120099</td>
          <td>24.861865</td>
          <td>0.267302</td>
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


