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
          <td>24.126346</td>
          <td>25.655789</td>
          <td>25.891787</td>
          <td>20.518395</td>
          <td>24.651138</td>
          <td>16.212049</td>
          <td>20.924660</td>
          <td>22.683717</td>
          <td>21.802258</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.159313</td>
          <td>22.349558</td>
          <td>22.079977</td>
          <td>20.446147</td>
          <td>19.470955</td>
          <td>23.215572</td>
          <td>26.795522</td>
          <td>24.438005</td>
          <td>23.907733</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.471701</td>
          <td>23.333268</td>
          <td>22.293331</td>
          <td>22.611656</td>
          <td>24.614514</td>
          <td>22.538395</td>
          <td>23.012323</td>
          <td>19.615020</td>
          <td>21.954052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.016181</td>
          <td>25.916472</td>
          <td>14.062312</td>
          <td>28.052240</td>
          <td>28.728567</td>
          <td>26.283577</td>
          <td>20.282771</td>
          <td>26.512883</td>
          <td>16.737983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.941355</td>
          <td>18.626695</td>
          <td>24.521902</td>
          <td>22.282730</td>
          <td>28.506378</td>
          <td>20.985512</td>
          <td>22.760887</td>
          <td>22.861386</td>
          <td>23.254170</td>
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
          <td>20.691529</td>
          <td>20.792564</td>
          <td>18.586436</td>
          <td>23.492937</td>
          <td>24.142454</td>
          <td>19.051571</td>
          <td>22.940026</td>
          <td>20.685614</td>
          <td>18.894085</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.172336</td>
          <td>25.476512</td>
          <td>17.868521</td>
          <td>21.828305</td>
          <td>25.449359</td>
          <td>23.468988</td>
          <td>23.062609</td>
          <td>22.529790</td>
          <td>16.697929</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.441015</td>
          <td>27.660265</td>
          <td>26.435999</td>
          <td>26.178271</td>
          <td>27.284031</td>
          <td>22.631388</td>
          <td>21.136350</td>
          <td>25.369884</td>
          <td>22.887973</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.130402</td>
          <td>23.880248</td>
          <td>29.476491</td>
          <td>21.866995</td>
          <td>21.486227</td>
          <td>25.215691</td>
          <td>21.158237</td>
          <td>22.268714</td>
          <td>19.660667</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.851748</td>
          <td>20.597279</td>
          <td>27.689254</td>
          <td>21.436688</td>
          <td>22.919432</td>
          <td>18.756663</td>
          <td>25.210561</td>
          <td>23.136839</td>
          <td>26.560628</td>
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
          <td>24.180456</td>
          <td>0.052769</td>
          <td>25.600716</td>
          <td>0.063001</td>
          <td>25.885795</td>
          <td>0.071276</td>
          <td>20.523574</td>
          <td>0.005110</td>
          <td>24.604388</td>
          <td>0.071603</td>
          <td>16.216498</td>
          <td>0.005004</td>
          <td>20.924660</td>
          <td>22.683717</td>
          <td>21.802258</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.511520</td>
          <td>0.380414</td>
          <td>22.357684</td>
          <td>0.006132</td>
          <td>22.091216</td>
          <td>0.005551</td>
          <td>20.445230</td>
          <td>0.005097</td>
          <td>19.466167</td>
          <td>0.005069</td>
          <td>23.174030</td>
          <td>0.045505</td>
          <td>26.795522</td>
          <td>24.438005</td>
          <td>23.907733</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.462853</td>
          <td>0.012568</td>
          <td>23.344479</td>
          <td>0.009721</td>
          <td>22.297372</td>
          <td>0.005768</td>
          <td>22.614645</td>
          <td>0.007940</td>
          <td>24.624214</td>
          <td>0.072870</td>
          <td>22.585200</td>
          <td>0.027072</td>
          <td>23.012323</td>
          <td>19.615020</td>
          <td>21.954052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.013614</td>
          <td>0.006100</td>
          <td>25.872366</td>
          <td>0.080085</td>
          <td>14.060502</td>
          <td>0.005000</td>
          <td>28.736699</td>
          <td>1.017661</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.759543</td>
          <td>3.210731</td>
          <td>20.282771</td>
          <td>26.512883</td>
          <td>16.737983</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.969638</td>
          <td>0.043844</td>
          <td>18.626282</td>
          <td>0.005008</td>
          <td>24.496382</td>
          <td>0.020985</td>
          <td>22.284370</td>
          <td>0.006796</td>
          <td>27.392102</td>
          <td>0.687807</td>
          <td>20.989891</td>
          <td>0.008070</td>
          <td>22.760887</td>
          <td>22.861386</td>
          <td>23.254170</td>
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
          <td>20.683102</td>
          <td>0.005694</td>
          <td>20.785087</td>
          <td>0.005107</td>
          <td>18.580225</td>
          <td>0.005004</td>
          <td>23.490172</td>
          <td>0.014430</td>
          <td>24.190536</td>
          <td>0.049607</td>
          <td>19.054930</td>
          <td>0.005148</td>
          <td>22.940026</td>
          <td>20.685614</td>
          <td>18.894085</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.167797</td>
          <td>0.005033</td>
          <td>25.445059</td>
          <td>0.054890</td>
          <td>17.878135</td>
          <td>0.005002</td>
          <td>21.829532</td>
          <td>0.005879</td>
          <td>26.008017</td>
          <td>0.239928</td>
          <td>23.451826</td>
          <td>0.058230</td>
          <td>23.062609</td>
          <td>22.529790</td>
          <td>16.697929</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.436816</td>
          <td>0.005137</td>
          <td>27.717766</td>
          <td>0.378261</td>
          <td>26.442159</td>
          <td>0.116210</td>
          <td>26.168752</td>
          <td>0.148081</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.646070</td>
          <td>0.028551</td>
          <td>21.136350</td>
          <td>25.369884</td>
          <td>22.887973</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.141016</td>
          <td>0.122360</td>
          <td>23.880420</td>
          <td>0.014326</td>
          <td>28.138070</td>
          <td>0.467070</td>
          <td>21.868042</td>
          <td>0.005935</td>
          <td>21.487671</td>
          <td>0.006636</td>
          <td>25.455185</td>
          <td>0.322986</td>
          <td>21.158237</td>
          <td>22.268714</td>
          <td>19.660667</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.837837</td>
          <td>0.016749</td>
          <td>20.595758</td>
          <td>0.005082</td>
          <td>27.077799</td>
          <td>0.200310</td>
          <td>21.431543</td>
          <td>0.005463</td>
          <td>22.921686</td>
          <td>0.016491</td>
          <td>18.755493</td>
          <td>0.005094</td>
          <td>25.210561</td>
          <td>23.136839</td>
          <td>26.560628</td>
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
          <td>24.126346</td>
          <td>25.655789</td>
          <td>25.891787</td>
          <td>20.518395</td>
          <td>24.651138</td>
          <td>16.212049</td>
          <td>20.932064</td>
          <td>0.005065</td>
          <td>22.684296</td>
          <td>0.008646</td>
          <td>21.800393</td>
          <td>0.005898</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.159313</td>
          <td>22.349558</td>
          <td>22.079977</td>
          <td>20.446147</td>
          <td>19.470955</td>
          <td>23.215572</td>
          <td>26.880769</td>
          <td>0.179023</td>
          <td>24.403124</td>
          <td>0.034328</td>
          <td>23.917753</td>
          <td>0.022402</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.471701</td>
          <td>23.333268</td>
          <td>22.293331</td>
          <td>22.611656</td>
          <td>24.614514</td>
          <td>22.538395</td>
          <td>23.004883</td>
          <td>0.007400</td>
          <td>19.618379</td>
          <td>0.005018</td>
          <td>21.958799</td>
          <td>0.006172</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.016181</td>
          <td>25.916472</td>
          <td>14.062312</td>
          <td>28.052240</td>
          <td>28.728567</td>
          <td>26.283577</td>
          <td>20.295754</td>
          <td>0.005020</td>
          <td>26.636530</td>
          <td>0.241166</td>
          <td>16.734317</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.941355</td>
          <td>18.626695</td>
          <td>24.521902</td>
          <td>22.282730</td>
          <td>28.506378</td>
          <td>20.985512</td>
          <td>22.769411</td>
          <td>0.006656</td>
          <td>22.864734</td>
          <td>0.009711</td>
          <td>23.266044</td>
          <td>0.013028</td>
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
          <td>20.691529</td>
          <td>20.792564</td>
          <td>18.586436</td>
          <td>23.492937</td>
          <td>24.142454</td>
          <td>19.051571</td>
          <td>22.939369</td>
          <td>0.007168</td>
          <td>20.687218</td>
          <td>0.005124</td>
          <td>18.896416</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.172336</td>
          <td>25.476512</td>
          <td>17.868521</td>
          <td>21.828305</td>
          <td>25.449359</td>
          <td>23.468988</td>
          <td>23.052704</td>
          <td>0.007583</td>
          <td>22.513557</td>
          <td>0.007832</td>
          <td>16.697841</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.441015</td>
          <td>27.660265</td>
          <td>26.435999</td>
          <td>26.178271</td>
          <td>27.284031</td>
          <td>22.631388</td>
          <td>21.136547</td>
          <td>0.005095</td>
          <td>25.141143</td>
          <td>0.066232</td>
          <td>22.887130</td>
          <td>0.009859</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.130402</td>
          <td>23.880248</td>
          <td>29.476491</td>
          <td>21.866995</td>
          <td>21.486227</td>
          <td>25.215691</td>
          <td>21.156022</td>
          <td>0.005098</td>
          <td>22.261792</td>
          <td>0.006919</td>
          <td>19.654883</td>
          <td>0.005019</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.851748</td>
          <td>20.597279</td>
          <td>27.689254</td>
          <td>21.436688</td>
          <td>22.919432</td>
          <td>18.756663</td>
          <td>25.125972</td>
          <td>0.038290</td>
          <td>23.150229</td>
          <td>0.011918</td>
          <td>26.721980</td>
          <td>0.258725</td>
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
          <td>24.126346</td>
          <td>25.655789</td>
          <td>25.891787</td>
          <td>20.518395</td>
          <td>24.651138</td>
          <td>16.212049</td>
          <td>20.928688</td>
          <td>0.010892</td>
          <td>22.620837</td>
          <td>0.038115</td>
          <td>21.794237</td>
          <td>0.020139</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.159313</td>
          <td>22.349558</td>
          <td>22.079977</td>
          <td>20.446147</td>
          <td>19.470955</td>
          <td>23.215572</td>
          <td>26.964957</td>
          <td>1.305512</td>
          <td>24.381662</td>
          <td>0.179159</td>
          <td>23.941094</td>
          <td>0.133791</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.471701</td>
          <td>23.333268</td>
          <td>22.293331</td>
          <td>22.611656</td>
          <td>24.614514</td>
          <td>22.538395</td>
          <td>23.027180</td>
          <td>0.065415</td>
          <td>19.624047</td>
          <td>0.005557</td>
          <td>21.925788</td>
          <td>0.022558</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.016181</td>
          <td>25.916472</td>
          <td>14.062312</td>
          <td>28.052240</td>
          <td>28.728567</td>
          <td>26.283577</td>
          <td>20.279962</td>
          <td>0.007309</td>
          <td>25.761587</td>
          <td>0.536618</td>
          <td>16.742282</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.941355</td>
          <td>18.626695</td>
          <td>24.521902</td>
          <td>22.282730</td>
          <td>28.506378</td>
          <td>20.985512</td>
          <td>22.760112</td>
          <td>0.051571</td>
          <td>22.848042</td>
          <td>0.046668</td>
          <td>23.307213</td>
          <td>0.076747</td>
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
          <td>20.691529</td>
          <td>20.792564</td>
          <td>18.586436</td>
          <td>23.492937</td>
          <td>24.142454</td>
          <td>19.051571</td>
          <td>22.842120</td>
          <td>0.055482</td>
          <td>20.693965</td>
          <td>0.008194</td>
          <td>18.903827</td>
          <td>0.005184</td>
        </tr>
        <tr>
          <th>996</th>
          <td>18.172336</td>
          <td>25.476512</td>
          <td>17.868521</td>
          <td>21.828305</td>
          <td>25.449359</td>
          <td>23.468988</td>
          <td>23.101629</td>
          <td>0.069888</td>
          <td>22.564201</td>
          <td>0.036243</td>
          <td>16.699335</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.441015</td>
          <td>27.660265</td>
          <td>26.435999</td>
          <td>26.178271</td>
          <td>27.284031</td>
          <td>22.631388</td>
          <td>21.126129</td>
          <td>0.012630</td>
          <td>25.100581</td>
          <td>0.324045</td>
          <td>22.887099</td>
          <td>0.052827</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.130402</td>
          <td>23.880248</td>
          <td>29.476491</td>
          <td>21.866995</td>
          <td>21.486227</td>
          <td>25.215691</td>
          <td>21.154629</td>
          <td>0.012913</td>
          <td>22.256373</td>
          <td>0.027597</td>
          <td>19.663755</td>
          <td>0.005710</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.851748</td>
          <td>20.597279</td>
          <td>27.689254</td>
          <td>21.436688</td>
          <td>22.919432</td>
          <td>18.756663</td>
          <td>25.089403</td>
          <td>0.375990</td>
          <td>23.094258</td>
          <td>0.058119</td>
          <td>26.852161</td>
          <td>1.161148</td>
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


