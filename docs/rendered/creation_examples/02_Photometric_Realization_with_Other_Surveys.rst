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
          <td>21.338098</td>
          <td>19.210569</td>
          <td>21.634000</td>
          <td>24.581962</td>
          <td>20.233008</td>
          <td>26.045048</td>
          <td>25.587644</td>
          <td>25.712488</td>
          <td>18.923211</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.471810</td>
          <td>23.389770</td>
          <td>18.291912</td>
          <td>15.866657</td>
          <td>25.093800</td>
          <td>23.563734</td>
          <td>20.979953</td>
          <td>23.687871</td>
          <td>26.718145</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.970739</td>
          <td>25.482338</td>
          <td>28.138736</td>
          <td>27.755073</td>
          <td>21.810059</td>
          <td>20.715591</td>
          <td>25.992218</td>
          <td>23.340335</td>
          <td>22.157503</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.457132</td>
          <td>18.608924</td>
          <td>22.727873</td>
          <td>22.151156</td>
          <td>17.409362</td>
          <td>21.170322</td>
          <td>24.216925</td>
          <td>23.983948</td>
          <td>22.012595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.410535</td>
          <td>23.065165</td>
          <td>25.887653</td>
          <td>25.597887</td>
          <td>26.297229</td>
          <td>25.716601</td>
          <td>20.875538</td>
          <td>26.473323</td>
          <td>26.701302</td>
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
          <td>21.175463</td>
          <td>20.578575</td>
          <td>30.036111</td>
          <td>22.276898</td>
          <td>22.430875</td>
          <td>20.917994</td>
          <td>27.834432</td>
          <td>25.643674</td>
          <td>20.105980</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.190684</td>
          <td>16.172963</td>
          <td>22.995862</td>
          <td>26.782784</td>
          <td>25.379950</td>
          <td>29.708323</td>
          <td>25.190714</td>
          <td>22.498359</td>
          <td>21.129410</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.270758</td>
          <td>20.448512</td>
          <td>19.665060</td>
          <td>16.503335</td>
          <td>16.040955</td>
          <td>25.942142</td>
          <td>25.906141</td>
          <td>24.750502</td>
          <td>25.452900</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574514</td>
          <td>22.958786</td>
          <td>25.356331</td>
          <td>25.271685</td>
          <td>23.827084</td>
          <td>23.552866</td>
          <td>17.908498</td>
          <td>24.205263</td>
          <td>23.394770</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.545555</td>
          <td>26.904702</td>
          <td>23.746061</td>
          <td>20.919204</td>
          <td>21.754939</td>
          <td>23.055262</td>
          <td>17.362573</td>
          <td>22.652907</td>
          <td>28.097751</td>
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
          <td>21.331476</td>
          <td>0.006715</td>
          <td>19.206808</td>
          <td>0.005015</td>
          <td>21.637265</td>
          <td>0.005265</td>
          <td>24.610197</td>
          <td>0.037566</td>
          <td>20.235916</td>
          <td>0.005223</td>
          <td>25.935424</td>
          <td>0.468142</td>
          <td>25.587644</td>
          <td>25.712488</td>
          <td>18.923211</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.471097</td>
          <td>0.007081</td>
          <td>23.376799</td>
          <td>0.009929</td>
          <td>18.294249</td>
          <td>0.005003</td>
          <td>15.861374</td>
          <td>0.005000</td>
          <td>25.311020</td>
          <td>0.132981</td>
          <td>23.471689</td>
          <td>0.059266</td>
          <td>20.979953</td>
          <td>23.687871</td>
          <td>26.718145</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.978996</td>
          <td>0.005004</td>
          <td>25.530269</td>
          <td>0.059193</td>
          <td>27.934244</td>
          <td>0.400104</td>
          <td>27.575737</td>
          <td>0.464025</td>
          <td>21.814236</td>
          <td>0.007673</td>
          <td>20.708874</td>
          <td>0.007028</td>
          <td>25.992218</td>
          <td>23.340335</td>
          <td>22.157503</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.756044</td>
          <td>0.206606</td>
          <td>18.600775</td>
          <td>0.005008</td>
          <td>22.728859</td>
          <td>0.006520</td>
          <td>22.163709</td>
          <td>0.006491</td>
          <td>17.405880</td>
          <td>0.005005</td>
          <td>21.174498</td>
          <td>0.008986</td>
          <td>24.216925</td>
          <td>23.983948</td>
          <td>22.012595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.510031</td>
          <td>0.379975</td>
          <td>23.063393</td>
          <td>0.008209</td>
          <td>25.930786</td>
          <td>0.074170</td>
          <td>25.409790</td>
          <td>0.076335</td>
          <td>25.836937</td>
          <td>0.208121</td>
          <td>26.300957</td>
          <td>0.610535</td>
          <td>20.875538</td>
          <td>26.473323</td>
          <td>26.701302</td>
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
          <td>21.178295</td>
          <td>0.006385</td>
          <td>20.577254</td>
          <td>0.005080</td>
          <td>29.564481</td>
          <td>1.192510</td>
          <td>22.277129</td>
          <td>0.006776</td>
          <td>22.425381</td>
          <td>0.011223</td>
          <td>20.916100</td>
          <td>0.007758</td>
          <td>27.834432</td>
          <td>25.643674</td>
          <td>20.105980</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.193470</td>
          <td>0.022430</td>
          <td>16.167010</td>
          <td>0.005001</td>
          <td>22.992250</td>
          <td>0.007271</td>
          <td>26.736126</td>
          <td>0.238982</td>
          <td>25.839832</td>
          <td>0.208626</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.190714</td>
          <td>22.498359</td>
          <td>21.129410</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.712432</td>
          <td>0.443662</td>
          <td>20.447163</td>
          <td>0.005068</td>
          <td>19.665999</td>
          <td>0.005015</td>
          <td>16.502690</td>
          <td>0.005001</td>
          <td>16.042517</td>
          <td>0.005001</td>
          <td>25.252278</td>
          <td>0.274326</td>
          <td>25.906141</td>
          <td>24.750502</td>
          <td>25.452900</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.578770</td>
          <td>0.007413</td>
          <td>22.973725</td>
          <td>0.007826</td>
          <td>25.343555</td>
          <td>0.044058</td>
          <td>25.142129</td>
          <td>0.060225</td>
          <td>23.846152</td>
          <td>0.036555</td>
          <td>23.670078</td>
          <td>0.070656</td>
          <td>17.908498</td>
          <td>24.205263</td>
          <td>23.394770</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.542182</td>
          <td>0.005571</td>
          <td>26.825791</td>
          <td>0.182923</td>
          <td>23.749259</td>
          <td>0.011524</td>
          <td>20.917630</td>
          <td>0.005203</td>
          <td>21.766526</td>
          <td>0.007492</td>
          <td>23.096020</td>
          <td>0.042463</td>
          <td>17.362573</td>
          <td>22.652907</td>
          <td>28.097751</td>
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
          <td>21.338098</td>
          <td>19.210569</td>
          <td>21.634000</td>
          <td>24.581962</td>
          <td>20.233008</td>
          <td>26.045048</td>
          <td>25.541656</td>
          <td>0.055459</td>
          <td>25.676733</td>
          <td>0.106281</td>
          <td>18.923250</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.471810</td>
          <td>23.389770</td>
          <td>18.291912</td>
          <td>15.866657</td>
          <td>25.093800</td>
          <td>23.563734</td>
          <td>20.982725</td>
          <td>0.005071</td>
          <td>23.683033</td>
          <td>0.018319</td>
          <td>27.412402</td>
          <td>0.446126</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.970739</td>
          <td>25.482338</td>
          <td>28.138736</td>
          <td>27.755073</td>
          <td>21.810059</td>
          <td>20.715591</td>
          <td>26.087952</td>
          <td>0.090037</td>
          <td>23.356708</td>
          <td>0.013996</td>
          <td>22.162888</td>
          <td>0.006638</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.457132</td>
          <td>18.608924</td>
          <td>22.727873</td>
          <td>22.151156</td>
          <td>17.409362</td>
          <td>21.170322</td>
          <td>24.237760</td>
          <td>0.017632</td>
          <td>23.969022</td>
          <td>0.023422</td>
          <td>22.004867</td>
          <td>0.006265</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.410535</td>
          <td>23.065165</td>
          <td>25.887653</td>
          <td>25.597887</td>
          <td>26.297229</td>
          <td>25.716601</td>
          <td>20.867757</td>
          <td>0.005058</td>
          <td>26.385193</td>
          <td>0.195548</td>
          <td>26.664068</td>
          <td>0.246705</td>
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
          <td>21.175463</td>
          <td>20.578575</td>
          <td>30.036111</td>
          <td>22.276898</td>
          <td>22.430875</td>
          <td>20.917994</td>
          <td>28.613013</td>
          <td>0.687454</td>
          <td>25.562883</td>
          <td>0.096177</td>
          <td>20.107257</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.190684</td>
          <td>16.172963</td>
          <td>22.995862</td>
          <td>26.782784</td>
          <td>25.379950</td>
          <td>29.708323</td>
          <td>25.154258</td>
          <td>0.039266</td>
          <td>22.501570</td>
          <td>0.007782</td>
          <td>21.136450</td>
          <td>0.005280</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.270758</td>
          <td>20.448512</td>
          <td>19.665060</td>
          <td>16.503335</td>
          <td>16.040955</td>
          <td>25.942142</td>
          <td>25.880450</td>
          <td>0.074949</td>
          <td>24.735251</td>
          <td>0.046138</td>
          <td>25.384477</td>
          <td>0.082178</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574514</td>
          <td>22.958786</td>
          <td>25.356331</td>
          <td>25.271685</td>
          <td>23.827084</td>
          <td>23.552866</td>
          <td>17.919744</td>
          <td>0.005000</td>
          <td>24.192336</td>
          <td>0.028486</td>
          <td>23.402123</td>
          <td>0.014517</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.545555</td>
          <td>26.904702</td>
          <td>23.746061</td>
          <td>20.919204</td>
          <td>21.754939</td>
          <td>23.055262</td>
          <td>17.368317</td>
          <td>0.005000</td>
          <td>22.665615</td>
          <td>0.008548</td>
          <td>30.358816</td>
          <td>2.344519</td>
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
          <td>21.338098</td>
          <td>19.210569</td>
          <td>21.634000</td>
          <td>24.581962</td>
          <td>20.233008</td>
          <td>26.045048</td>
          <td>25.244559</td>
          <td>0.423739</td>
          <td>25.977728</td>
          <td>0.626088</td>
          <td>18.911697</td>
          <td>0.005187</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.471810</td>
          <td>23.389770</td>
          <td>18.291912</td>
          <td>15.866657</td>
          <td>25.093800</td>
          <td>23.563734</td>
          <td>20.984900</td>
          <td>0.011350</td>
          <td>23.829503</td>
          <td>0.111302</td>
          <td>26.112620</td>
          <td>0.735318</td>
        </tr>
        <tr>
          <th>2</th>
          <td>15.970739</td>
          <td>25.482338</td>
          <td>28.138736</td>
          <td>27.755073</td>
          <td>21.810059</td>
          <td>20.715591</td>
          <td>25.219969</td>
          <td>0.415854</td>
          <td>23.241753</td>
          <td>0.066268</td>
          <td>22.166612</td>
          <td>0.027847</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.457132</td>
          <td>18.608924</td>
          <td>22.727873</td>
          <td>22.151156</td>
          <td>17.409362</td>
          <td>21.170322</td>
          <td>24.270758</td>
          <td>0.193183</td>
          <td>24.074730</td>
          <td>0.137740</td>
          <td>21.973395</td>
          <td>0.023511</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.410535</td>
          <td>23.065165</td>
          <td>25.887653</td>
          <td>25.597887</td>
          <td>26.297229</td>
          <td>25.716601</td>
          <td>20.884019</td>
          <td>0.010548</td>
          <td>26.285076</td>
          <td>0.771569</td>
          <td>26.246870</td>
          <td>0.803441</td>
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
          <td>21.175463</td>
          <td>20.578575</td>
          <td>30.036111</td>
          <td>22.276898</td>
          <td>22.430875</td>
          <td>20.917994</td>
          <td>27.636286</td>
          <td>1.814903</td>
          <td>26.030944</td>
          <td>0.649729</td>
          <td>20.109657</td>
          <td>0.006503</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.190684</td>
          <td>16.172963</td>
          <td>22.995862</td>
          <td>26.782784</td>
          <td>25.379950</td>
          <td>29.708323</td>
          <td>25.029397</td>
          <td>0.358769</td>
          <td>22.532872</td>
          <td>0.035247</td>
          <td>21.120049</td>
          <td>0.011651</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.270758</td>
          <td>20.448512</td>
          <td>19.665060</td>
          <td>16.503335</td>
          <td>16.040955</td>
          <td>25.942142</td>
          <td>26.286789</td>
          <td>0.878844</td>
          <td>25.423650</td>
          <td>0.417026</td>
          <td>25.428363</td>
          <td>0.451529</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.574514</td>
          <td>22.958786</td>
          <td>25.356331</td>
          <td>25.271685</td>
          <td>23.827084</td>
          <td>23.552866</td>
          <td>17.910654</td>
          <td>0.005036</td>
          <td>24.251050</td>
          <td>0.160290</td>
          <td>23.555346</td>
          <td>0.095541</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.545555</td>
          <td>26.904702</td>
          <td>23.746061</td>
          <td>20.919204</td>
          <td>21.754939</td>
          <td>23.055262</td>
          <td>17.358903</td>
          <td>0.005013</td>
          <td>22.618296</td>
          <td>0.038029</td>
          <td>27.633734</td>
          <td>1.732379</td>
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


