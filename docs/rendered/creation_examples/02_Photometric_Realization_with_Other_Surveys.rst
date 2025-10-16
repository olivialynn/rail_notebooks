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
          <td>27.260087</td>
          <td>21.430253</td>
          <td>23.472723</td>
          <td>22.823055</td>
          <td>23.531249</td>
          <td>26.061359</td>
          <td>25.023380</td>
          <td>23.579519</td>
          <td>26.812145</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.106119</td>
          <td>22.978126</td>
          <td>19.657967</td>
          <td>28.864539</td>
          <td>20.435984</td>
          <td>22.354283</td>
          <td>20.515900</td>
          <td>26.443363</td>
          <td>20.820150</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.827525</td>
          <td>18.144140</td>
          <td>24.987059</td>
          <td>23.143656</td>
          <td>24.355676</td>
          <td>21.167066</td>
          <td>29.349245</td>
          <td>29.420944</td>
          <td>24.164704</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.352162</td>
          <td>25.263324</td>
          <td>20.356162</td>
          <td>24.727928</td>
          <td>22.610588</td>
          <td>23.386320</td>
          <td>22.449206</td>
          <td>19.900527</td>
          <td>21.920587</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.299346</td>
          <td>24.631481</td>
          <td>17.245915</td>
          <td>20.148690</td>
          <td>21.255377</td>
          <td>21.729577</td>
          <td>25.739130</td>
          <td>22.123527</td>
          <td>23.568466</td>
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
          <td>22.907031</td>
          <td>22.241972</td>
          <td>18.167072</td>
          <td>26.655503</td>
          <td>24.504749</td>
          <td>23.275854</td>
          <td>16.695987</td>
          <td>20.678875</td>
          <td>17.448840</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.841293</td>
          <td>22.043129</td>
          <td>23.251364</td>
          <td>25.786832</td>
          <td>22.462153</td>
          <td>27.893599</td>
          <td>25.238329</td>
          <td>27.738649</td>
          <td>14.527977</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.635500</td>
          <td>25.522787</td>
          <td>23.840486</td>
          <td>26.944838</td>
          <td>23.106478</td>
          <td>17.661577</td>
          <td>25.493197</td>
          <td>23.434811</td>
          <td>22.624806</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.408738</td>
          <td>22.224589</td>
          <td>24.840501</td>
          <td>28.006752</td>
          <td>25.738191</td>
          <td>20.910971</td>
          <td>20.257935</td>
          <td>26.409938</td>
          <td>25.219822</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.328929</td>
          <td>17.525573</td>
          <td>22.800023</td>
          <td>23.243899</td>
          <td>21.238203</td>
          <td>23.843211</td>
          <td>25.054107</td>
          <td>25.107831</td>
          <td>20.329061</td>
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
          <td>26.620891</td>
          <td>0.413850</td>
          <td>21.425321</td>
          <td>0.005271</td>
          <td>23.467074</td>
          <td>0.009484</td>
          <td>22.820330</td>
          <td>0.008940</td>
          <td>23.545688</td>
          <td>0.028063</td>
          <td>26.045954</td>
          <td>0.508131</td>
          <td>25.023380</td>
          <td>23.579519</td>
          <td>26.812145</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.102507</td>
          <td>0.006246</td>
          <td>22.975430</td>
          <td>0.007833</td>
          <td>19.658855</td>
          <td>0.005015</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.431468</td>
          <td>0.005305</td>
          <td>22.335183</td>
          <td>0.021808</td>
          <td>20.515900</td>
          <td>26.443363</td>
          <td>20.820150</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.839538</td>
          <td>0.039120</td>
          <td>18.141590</td>
          <td>0.005005</td>
          <td>25.020250</td>
          <td>0.033095</td>
          <td>23.144592</td>
          <td>0.011103</td>
          <td>24.406049</td>
          <td>0.060064</td>
          <td>21.167101</td>
          <td>0.008945</td>
          <td>29.349245</td>
          <td>29.420944</td>
          <td>24.164704</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.355524</td>
          <td>0.006773</td>
          <td>25.320499</td>
          <td>0.049156</td>
          <td>20.354976</td>
          <td>0.005038</td>
          <td>24.709240</td>
          <td>0.041011</td>
          <td>22.573622</td>
          <td>0.012523</td>
          <td>23.355418</td>
          <td>0.053455</td>
          <td>22.449206</td>
          <td>19.900527</td>
          <td>21.920587</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.304524</td>
          <td>0.006652</td>
          <td>24.628538</td>
          <td>0.026745</td>
          <td>17.248450</td>
          <td>0.005001</td>
          <td>20.149522</td>
          <td>0.005062</td>
          <td>21.259565</td>
          <td>0.006147</td>
          <td>21.715064</td>
          <td>0.013119</td>
          <td>25.739130</td>
          <td>22.123527</td>
          <td>23.568466</td>
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
          <td>22.911922</td>
          <td>0.017777</td>
          <td>22.249554</td>
          <td>0.005959</td>
          <td>18.170513</td>
          <td>0.005003</td>
          <td>26.363655</td>
          <td>0.174907</td>
          <td>24.518596</td>
          <td>0.066366</td>
          <td>23.267666</td>
          <td>0.049449</td>
          <td>16.695987</td>
          <td>20.678875</td>
          <td>17.448840</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.152415</td>
          <td>0.286221</td>
          <td>22.047395</td>
          <td>0.005702</td>
          <td>23.257434</td>
          <td>0.008345</td>
          <td>25.859126</td>
          <td>0.113266</td>
          <td>22.471914</td>
          <td>0.011609</td>
          <td>27.807684</td>
          <td>1.511299</td>
          <td>25.238329</td>
          <td>27.738649</td>
          <td>14.527977</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.631394</td>
          <td>0.007592</td>
          <td>25.519283</td>
          <td>0.058619</td>
          <td>23.828479</td>
          <td>0.012221</td>
          <td>26.629842</td>
          <td>0.218815</td>
          <td>23.115151</td>
          <td>0.019372</td>
          <td>17.656642</td>
          <td>0.005020</td>
          <td>25.493197</td>
          <td>23.434811</td>
          <td>22.624806</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.420596</td>
          <td>0.027202</td>
          <td>22.219555</td>
          <td>0.005916</td>
          <td>24.873443</td>
          <td>0.029088</td>
          <td>28.564992</td>
          <td>0.916492</td>
          <td>25.627381</td>
          <td>0.174409</td>
          <td>20.905229</td>
          <td>0.007715</td>
          <td>20.257935</td>
          <td>26.409938</td>
          <td>25.219822</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.928570</td>
          <td>0.238422</td>
          <td>17.520023</td>
          <td>0.005003</td>
          <td>22.794811</td>
          <td>0.006683</td>
          <td>23.254681</td>
          <td>0.012034</td>
          <td>21.245789</td>
          <td>0.006122</td>
          <td>23.921077</td>
          <td>0.088178</td>
          <td>25.054107</td>
          <td>25.107831</td>
          <td>20.329061</td>
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
          <td>27.260087</td>
          <td>21.430253</td>
          <td>23.472723</td>
          <td>22.823055</td>
          <td>23.531249</td>
          <td>26.061359</td>
          <td>25.053501</td>
          <td>0.035899</td>
          <td>23.571402</td>
          <td>0.016680</td>
          <td>26.820780</td>
          <td>0.280431</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.106119</td>
          <td>22.978126</td>
          <td>19.657967</td>
          <td>28.864539</td>
          <td>20.435984</td>
          <td>22.354283</td>
          <td>20.520188</td>
          <td>0.005031</td>
          <td>26.090546</td>
          <td>0.152189</td>
          <td>20.818337</td>
          <td>0.005158</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.827525</td>
          <td>18.144140</td>
          <td>24.987059</td>
          <td>23.143656</td>
          <td>24.355676</td>
          <td>21.167066</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.151963</td>
          <td>0.027490</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.352162</td>
          <td>25.263324</td>
          <td>20.356162</td>
          <td>24.727928</td>
          <td>22.610588</td>
          <td>23.386320</td>
          <td>22.445559</td>
          <td>0.005969</td>
          <td>19.898427</td>
          <td>0.005029</td>
          <td>21.925805</td>
          <td>0.006109</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.299346</td>
          <td>24.631481</td>
          <td>17.245915</td>
          <td>20.148690</td>
          <td>21.255377</td>
          <td>21.729577</td>
          <td>25.624663</td>
          <td>0.059714</td>
          <td>22.121257</td>
          <td>0.006532</td>
          <td>23.579777</td>
          <td>0.016797</td>
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
          <td>22.907031</td>
          <td>22.241972</td>
          <td>18.167072</td>
          <td>26.655503</td>
          <td>24.504749</td>
          <td>23.275854</td>
          <td>16.698601</td>
          <td>0.005000</td>
          <td>20.678871</td>
          <td>0.005123</td>
          <td>17.443159</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.841293</td>
          <td>22.043129</td>
          <td>23.251364</td>
          <td>25.786832</td>
          <td>22.462153</td>
          <td>27.893599</td>
          <td>25.194420</td>
          <td>0.040696</td>
          <td>28.023476</td>
          <td>0.692375</td>
          <td>14.528200</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.635500</td>
          <td>25.522787</td>
          <td>23.840486</td>
          <td>26.944838</td>
          <td>23.106478</td>
          <td>17.661577</td>
          <td>25.472638</td>
          <td>0.052150</td>
          <td>23.434949</td>
          <td>0.014908</td>
          <td>22.638343</td>
          <td>0.008409</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.408738</td>
          <td>22.224589</td>
          <td>24.840501</td>
          <td>28.006752</td>
          <td>25.738191</td>
          <td>20.910971</td>
          <td>20.265687</td>
          <td>0.005019</td>
          <td>26.087173</td>
          <td>0.151749</td>
          <td>25.344631</td>
          <td>0.079332</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.328929</td>
          <td>17.525573</td>
          <td>22.800023</td>
          <td>23.243899</td>
          <td>21.238203</td>
          <td>23.843211</td>
          <td>25.022231</td>
          <td>0.034916</td>
          <td>25.098197</td>
          <td>0.063751</td>
          <td>20.336281</td>
          <td>0.005066</td>
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
          <td>27.260087</td>
          <td>21.430253</td>
          <td>23.472723</td>
          <td>22.823055</td>
          <td>23.531249</td>
          <td>26.061359</td>
          <td>25.486959</td>
          <td>0.508124</td>
          <td>23.977106</td>
          <td>0.126572</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.106119</td>
          <td>22.978126</td>
          <td>19.657967</td>
          <td>28.864539</td>
          <td>20.435984</td>
          <td>22.354283</td>
          <td>20.517307</td>
          <td>0.008306</td>
          <td>27.054744</td>
          <td>1.229606</td>
          <td>20.824834</td>
          <td>0.009455</td>
        </tr>
        <tr>
          <th>2</th>
          <td>23.827525</td>
          <td>18.144140</td>
          <td>24.987059</td>
          <td>23.143656</td>
          <td>24.355676</td>
          <td>21.167066</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.942949</td>
          <td>0.610984</td>
          <td>24.182354</td>
          <td>0.164636</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.352162</td>
          <td>25.263324</td>
          <td>20.356162</td>
          <td>24.727928</td>
          <td>22.610588</td>
          <td>23.386320</td>
          <td>22.455077</td>
          <td>0.039295</td>
          <td>19.889957</td>
          <td>0.005882</td>
          <td>21.888382</td>
          <td>0.021839</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.299346</td>
          <td>24.631481</td>
          <td>17.245915</td>
          <td>20.148690</td>
          <td>21.255377</td>
          <td>21.729577</td>
          <td>27.380176</td>
          <td>1.611766</td>
          <td>22.162512</td>
          <td>0.025414</td>
          <td>23.822486</td>
          <td>0.120703</td>
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
          <td>22.907031</td>
          <td>22.241972</td>
          <td>18.167072</td>
          <td>26.655503</td>
          <td>24.504749</td>
          <td>23.275854</td>
          <td>16.697049</td>
          <td>0.005004</td>
          <td>20.685023</td>
          <td>0.008152</td>
          <td>17.442877</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.841293</td>
          <td>22.043129</td>
          <td>23.251364</td>
          <td>25.786832</td>
          <td>22.462153</td>
          <td>27.893599</td>
          <td>24.939651</td>
          <td>0.334260</td>
          <td>25.846519</td>
          <td>0.570532</td>
          <td>14.521154</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.635500</td>
          <td>25.522787</td>
          <td>23.840486</td>
          <td>26.944838</td>
          <td>23.106478</td>
          <td>17.661577</td>
          <td>25.093877</td>
          <td>0.377301</td>
          <td>23.407518</td>
          <td>0.076768</td>
          <td>22.620829</td>
          <td>0.041665</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.408738</td>
          <td>22.224589</td>
          <td>24.840501</td>
          <td>28.006752</td>
          <td>25.738191</td>
          <td>20.910971</td>
          <td>20.256124</td>
          <td>0.007226</td>
          <td>25.780193</td>
          <td>0.543911</td>
          <td>24.836578</td>
          <td>0.284047</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.328929</td>
          <td>17.525573</td>
          <td>22.800023</td>
          <td>23.243899</td>
          <td>21.238203</td>
          <td>23.843211</td>
          <td>25.537055</td>
          <td>0.527118</td>
          <td>25.161568</td>
          <td>0.340110</td>
          <td>20.324276</td>
          <td>0.007118</td>
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


