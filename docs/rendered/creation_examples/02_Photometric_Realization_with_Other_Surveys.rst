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
          <td>25.220219</td>
          <td>22.464885</td>
          <td>26.021963</td>
          <td>16.114012</td>
          <td>20.135388</td>
          <td>23.988138</td>
          <td>24.625653</td>
          <td>22.512954</td>
          <td>26.457851</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.270258</td>
          <td>23.518802</td>
          <td>21.392183</td>
          <td>23.270175</td>
          <td>27.179861</td>
          <td>22.186837</td>
          <td>26.589378</td>
          <td>26.119335</td>
          <td>24.362993</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.267122</td>
          <td>26.611478</td>
          <td>23.157401</td>
          <td>20.140331</td>
          <td>24.987312</td>
          <td>20.928980</td>
          <td>26.576648</td>
          <td>22.235041</td>
          <td>20.556947</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.482927</td>
          <td>25.044235</td>
          <td>26.366003</td>
          <td>26.251446</td>
          <td>18.267265</td>
          <td>23.681275</td>
          <td>23.728471</td>
          <td>26.263184</td>
          <td>23.924222</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.205110</td>
          <td>21.646582</td>
          <td>22.800862</td>
          <td>22.612008</td>
          <td>23.851433</td>
          <td>20.345923</td>
          <td>22.075585</td>
          <td>20.182903</td>
          <td>22.231100</td>
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
          <td>26.008774</td>
          <td>19.257698</td>
          <td>20.740718</td>
          <td>21.353441</td>
          <td>20.192708</td>
          <td>24.035598</td>
          <td>27.826445</td>
          <td>20.232271</td>
          <td>24.785164</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.799024</td>
          <td>22.901121</td>
          <td>27.070928</td>
          <td>27.034200</td>
          <td>22.285257</td>
          <td>16.511375</td>
          <td>23.284985</td>
          <td>21.211016</td>
          <td>22.278137</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.443846</td>
          <td>22.054800</td>
          <td>28.576017</td>
          <td>21.796758</td>
          <td>17.563436</td>
          <td>20.436939</td>
          <td>25.834023</td>
          <td>23.321911</td>
          <td>23.962649</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.853903</td>
          <td>22.458661</td>
          <td>25.170898</td>
          <td>26.414704</td>
          <td>28.201769</td>
          <td>23.528638</td>
          <td>22.260040</td>
          <td>19.661149</td>
          <td>27.960713</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.340237</td>
          <td>20.613625</td>
          <td>27.731920</td>
          <td>23.047258</td>
          <td>24.381025</td>
          <td>21.641973</td>
          <td>25.910578</td>
          <td>26.662914</td>
          <td>22.086939</td>
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
          <td>25.333570</td>
          <td>0.144437</td>
          <td>22.480496</td>
          <td>0.006364</td>
          <td>25.927159</td>
          <td>0.073932</td>
          <td>16.113081</td>
          <td>0.005001</td>
          <td>20.138825</td>
          <td>0.005191</td>
          <td>23.963519</td>
          <td>0.091532</td>
          <td>24.625653</td>
          <td>22.512954</td>
          <td>26.457851</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.782746</td>
          <td>0.211263</td>
          <td>23.533084</td>
          <td>0.011048</td>
          <td>21.393437</td>
          <td>0.005180</td>
          <td>23.262830</td>
          <td>0.012107</td>
          <td>27.115278</td>
          <td>0.566655</td>
          <td>22.169721</td>
          <td>0.018951</td>
          <td>26.589378</td>
          <td>26.119335</td>
          <td>24.362993</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.271399</td>
          <td>0.005396</td>
          <td>26.909567</td>
          <td>0.196313</td>
          <td>23.157446</td>
          <td>0.007897</td>
          <td>20.142166</td>
          <td>0.005062</td>
          <td>25.136773</td>
          <td>0.114317</td>
          <td>20.928674</td>
          <td>0.007809</td>
          <td>26.576648</td>
          <td>22.235041</td>
          <td>20.556947</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.495576</td>
          <td>0.165868</td>
          <td>24.974216</td>
          <td>0.036195</td>
          <td>26.597308</td>
          <td>0.132955</td>
          <td>26.373867</td>
          <td>0.176430</td>
          <td>18.270231</td>
          <td>0.005014</td>
          <td>23.613115</td>
          <td>0.067182</td>
          <td>23.728471</td>
          <td>26.263184</td>
          <td>23.924222</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.126178</td>
          <td>0.120799</td>
          <td>21.634301</td>
          <td>0.005372</td>
          <td>22.798822</td>
          <td>0.006693</td>
          <td>22.607265</td>
          <td>0.007908</td>
          <td>23.905496</td>
          <td>0.038526</td>
          <td>20.370449</td>
          <td>0.006203</td>
          <td>22.075585</td>
          <td>20.182903</td>
          <td>22.231100</td>
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
          <td>25.890400</td>
          <td>0.231025</td>
          <td>19.258343</td>
          <td>0.005016</td>
          <td>20.736773</td>
          <td>0.005066</td>
          <td>21.360928</td>
          <td>0.005413</td>
          <td>20.196031</td>
          <td>0.005210</td>
          <td>23.973504</td>
          <td>0.092338</td>
          <td>27.826445</td>
          <td>20.232271</td>
          <td>24.785164</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.838736</td>
          <td>0.039093</td>
          <td>22.897830</td>
          <td>0.007534</td>
          <td>27.222196</td>
          <td>0.225986</td>
          <td>27.531482</td>
          <td>0.448844</td>
          <td>22.295884</td>
          <td>0.010248</td>
          <td>16.515695</td>
          <td>0.005005</td>
          <td>23.284985</td>
          <td>21.211016</td>
          <td>22.278137</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.818469</td>
          <td>0.217644</td>
          <td>22.063004</td>
          <td>0.005719</td>
          <td>27.503481</td>
          <td>0.284636</td>
          <td>21.801614</td>
          <td>0.005841</td>
          <td>17.561473</td>
          <td>0.005006</td>
          <td>20.432130</td>
          <td>0.006325</td>
          <td>25.834023</td>
          <td>23.321911</td>
          <td>23.962649</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.596000</td>
          <td>0.180604</td>
          <td>22.460276</td>
          <td>0.006323</td>
          <td>25.173413</td>
          <td>0.037890</td>
          <td>26.645491</td>
          <td>0.221684</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.415418</td>
          <td>0.056379</td>
          <td>22.260040</td>
          <td>19.661149</td>
          <td>27.960713</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.334886</td>
          <td>0.005121</td>
          <td>20.617582</td>
          <td>0.005085</td>
          <td>27.406273</td>
          <td>0.262998</td>
          <td>23.042997</td>
          <td>0.010338</td>
          <td>24.314372</td>
          <td>0.055371</td>
          <td>21.655510</td>
          <td>0.012535</td>
          <td>25.910578</td>
          <td>26.662914</td>
          <td>22.086939</td>
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
          <td>25.220219</td>
          <td>22.464885</td>
          <td>26.021963</td>
          <td>16.114012</td>
          <td>20.135388</td>
          <td>23.988138</td>
          <td>24.627944</td>
          <td>0.024657</td>
          <td>22.508194</td>
          <td>0.007809</td>
          <td>26.493750</td>
          <td>0.214195</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.270258</td>
          <td>23.518802</td>
          <td>21.392183</td>
          <td>23.270175</td>
          <td>27.179861</td>
          <td>22.186837</td>
          <td>26.533167</td>
          <td>0.132876</td>
          <td>26.219574</td>
          <td>0.169947</td>
          <td>24.359840</td>
          <td>0.033035</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.267122</td>
          <td>26.611478</td>
          <td>23.157401</td>
          <td>20.140331</td>
          <td>24.987312</td>
          <td>20.928980</td>
          <td>26.614131</td>
          <td>0.142505</td>
          <td>22.235379</td>
          <td>0.006840</td>
          <td>20.559258</td>
          <td>0.005099</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.482927</td>
          <td>25.044235</td>
          <td>26.366003</td>
          <td>26.251446</td>
          <td>18.267265</td>
          <td>23.681275</td>
          <td>23.720994</td>
          <td>0.011659</td>
          <td>26.144506</td>
          <td>0.159394</td>
          <td>23.948994</td>
          <td>0.023018</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.205110</td>
          <td>21.646582</td>
          <td>22.800862</td>
          <td>22.612008</td>
          <td>23.851433</td>
          <td>20.345923</td>
          <td>22.079670</td>
          <td>0.005516</td>
          <td>20.184964</td>
          <td>0.005050</td>
          <td>22.233112</td>
          <td>0.006834</td>
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
          <td>26.008774</td>
          <td>19.257698</td>
          <td>20.740718</td>
          <td>21.353441</td>
          <td>20.192708</td>
          <td>24.035598</td>
          <td>27.652057</td>
          <td>0.337561</td>
          <td>20.227219</td>
          <td>0.005054</td>
          <td>24.739606</td>
          <td>0.046318</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.799024</td>
          <td>22.901121</td>
          <td>27.070928</td>
          <td>27.034200</td>
          <td>22.285257</td>
          <td>16.511375</td>
          <td>23.292999</td>
          <td>0.008692</td>
          <td>21.213886</td>
          <td>0.005322</td>
          <td>22.277318</td>
          <td>0.006967</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.443846</td>
          <td>22.054800</td>
          <td>28.576017</td>
          <td>21.796758</td>
          <td>17.563436</td>
          <td>20.436939</td>
          <td>25.873469</td>
          <td>0.074487</td>
          <td>23.309255</td>
          <td>0.013478</td>
          <td>23.914514</td>
          <td>0.022339</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.853903</td>
          <td>22.458661</td>
          <td>25.170898</td>
          <td>26.414704</td>
          <td>28.201769</td>
          <td>23.528638</td>
          <td>22.259186</td>
          <td>0.005705</td>
          <td>19.652755</td>
          <td>0.005019</td>
          <td>27.795097</td>
          <td>0.590649</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.340237</td>
          <td>20.613625</td>
          <td>27.731920</td>
          <td>23.047258</td>
          <td>24.381025</td>
          <td>21.641973</td>
          <td>25.906546</td>
          <td>0.076702</td>
          <td>26.930981</td>
          <td>0.306511</td>
          <td>22.082918</td>
          <td>0.006439</td>
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
          <td>25.220219</td>
          <td>22.464885</td>
          <td>26.021963</td>
          <td>16.114012</td>
          <td>20.135388</td>
          <td>23.988138</td>
          <td>24.748931</td>
          <td>0.286902</td>
          <td>22.496548</td>
          <td>0.034128</td>
          <td>25.854435</td>
          <td>0.615942</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.270258</td>
          <td>23.518802</td>
          <td>21.392183</td>
          <td>23.270175</td>
          <td>27.179861</td>
          <td>22.186837</td>
          <td>27.059686</td>
          <td>1.372636</td>
          <td>26.561562</td>
          <td>0.920969</td>
          <td>24.221648</td>
          <td>0.170247</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.267122</td>
          <td>26.611478</td>
          <td>23.157401</td>
          <td>20.140331</td>
          <td>24.987312</td>
          <td>20.928980</td>
          <td>26.166027</td>
          <td>0.813499</td>
          <td>22.205524</td>
          <td>0.026391</td>
          <td>20.541037</td>
          <td>0.007952</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.482927</td>
          <td>25.044235</td>
          <td>26.366003</td>
          <td>26.251446</td>
          <td>18.267265</td>
          <td>23.681275</td>
          <td>23.955938</td>
          <td>0.147729</td>
          <td>27.586205</td>
          <td>1.616432</td>
          <td>23.972373</td>
          <td>0.137460</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.205110</td>
          <td>21.646582</td>
          <td>22.800862</td>
          <td>22.612008</td>
          <td>23.851433</td>
          <td>20.345923</td>
          <td>22.102714</td>
          <td>0.028748</td>
          <td>20.197978</td>
          <td>0.006475</td>
          <td>22.259136</td>
          <td>0.030216</td>
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
          <td>26.008774</td>
          <td>19.257698</td>
          <td>20.740718</td>
          <td>21.353441</td>
          <td>20.192708</td>
          <td>24.035598</td>
          <td>25.292921</td>
          <td>0.439601</td>
          <td>20.229077</td>
          <td>0.006551</td>
          <td>24.756137</td>
          <td>0.266055</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.799024</td>
          <td>22.901121</td>
          <td>27.070928</td>
          <td>27.034200</td>
          <td>22.285257</td>
          <td>16.511375</td>
          <td>23.298975</td>
          <td>0.083238</td>
          <td>21.217229</td>
          <td>0.011626</td>
          <td>22.303603</td>
          <td>0.031429</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.443846</td>
          <td>22.054800</td>
          <td>28.576017</td>
          <td>21.796758</td>
          <td>17.563436</td>
          <td>20.436939</td>
          <td>25.387574</td>
          <td>0.472038</td>
          <td>23.423423</td>
          <td>0.077857</td>
          <td>23.920771</td>
          <td>0.131458</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.853903</td>
          <td>22.458661</td>
          <td>25.170898</td>
          <td>26.414704</td>
          <td>28.201769</td>
          <td>23.528638</td>
          <td>22.257829</td>
          <td>0.032976</td>
          <td>19.653996</td>
          <td>0.005587</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.340237</td>
          <td>20.613625</td>
          <td>27.731920</td>
          <td>23.047258</td>
          <td>24.381025</td>
          <td>21.641973</td>
          <td>24.693681</td>
          <td>0.274324</td>
          <td>25.667036</td>
          <td>0.500721</td>
          <td>22.069932</td>
          <td>0.025580</td>
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


