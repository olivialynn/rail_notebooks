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
```Photometric_Realization_with_Other_Surveys.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
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
          <td>25.419841</td>
          <td>24.666224</td>
          <td>21.553050</td>
          <td>16.294962</td>
          <td>26.991513</td>
          <td>18.211069</td>
          <td>24.618554</td>
          <td>23.322307</td>
          <td>18.662668</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.391657</td>
          <td>22.998684</td>
          <td>27.976088</td>
          <td>24.845287</td>
          <td>23.755343</td>
          <td>21.914646</td>
          <td>20.867816</td>
          <td>21.162975</td>
          <td>24.772481</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.623156</td>
          <td>18.767576</td>
          <td>24.419706</td>
          <td>15.421715</td>
          <td>22.779084</td>
          <td>20.989840</td>
          <td>23.332798</td>
          <td>21.369072</td>
          <td>21.883935</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.273743</td>
          <td>26.154370</td>
          <td>21.469264</td>
          <td>22.629823</td>
          <td>22.285213</td>
          <td>20.537738</td>
          <td>24.522300</td>
          <td>20.696366</td>
          <td>19.390134</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.805790</td>
          <td>21.346334</td>
          <td>22.591029</td>
          <td>14.441492</td>
          <td>23.200357</td>
          <td>20.687162</td>
          <td>22.279059</td>
          <td>17.116926</td>
          <td>27.789971</td>
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
          <td>22.174058</td>
          <td>21.458752</td>
          <td>24.433097</td>
          <td>25.167819</td>
          <td>24.003333</td>
          <td>19.908218</td>
          <td>16.361796</td>
          <td>19.093166</td>
          <td>20.090527</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.313887</td>
          <td>24.395679</td>
          <td>19.450278</td>
          <td>26.521361</td>
          <td>21.439028</td>
          <td>20.303513</td>
          <td>23.089908</td>
          <td>25.463571</td>
          <td>22.939065</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.754115</td>
          <td>20.647016</td>
          <td>16.877253</td>
          <td>21.966344</td>
          <td>24.053722</td>
          <td>21.757548</td>
          <td>23.573075</td>
          <td>23.613499</td>
          <td>21.342301</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.332927</td>
          <td>25.490199</td>
          <td>27.132583</td>
          <td>23.687093</td>
          <td>27.724105</td>
          <td>23.404066</td>
          <td>26.203347</td>
          <td>21.288233</td>
          <td>23.560364</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.853095</td>
          <td>20.671679</td>
          <td>22.993174</td>
          <td>26.814697</td>
          <td>26.335310</td>
          <td>23.581529</td>
          <td>21.003524</td>
          <td>23.212226</td>
          <td>24.542139</td>
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
          <td>25.540341</td>
          <td>0.172293</td>
          <td>24.675887</td>
          <td>0.027869</td>
          <td>21.563279</td>
          <td>0.005236</td>
          <td>16.293889</td>
          <td>0.005001</td>
          <td>27.740831</td>
          <td>0.865393</td>
          <td>18.209643</td>
          <td>0.005043</td>
          <td>24.618554</td>
          <td>23.322307</td>
          <td>18.662668</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.053949</td>
          <td>0.570409</td>
          <td>22.998045</td>
          <td>0.007926</td>
          <td>27.820009</td>
          <td>0.366175</td>
          <td>24.821653</td>
          <td>0.045311</td>
          <td>23.700593</td>
          <td>0.032149</td>
          <td>21.939008</td>
          <td>0.015662</td>
          <td>20.867816</td>
          <td>21.162975</td>
          <td>24.772481</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.620141</td>
          <td>0.005171</td>
          <td>18.766841</td>
          <td>0.005009</td>
          <td>24.429973</td>
          <td>0.019833</td>
          <td>15.420528</td>
          <td>0.005000</td>
          <td>22.801394</td>
          <td>0.014958</td>
          <td>20.997054</td>
          <td>0.008101</td>
          <td>23.332798</td>
          <td>21.369072</td>
          <td>21.883935</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.282743</td>
          <td>0.005402</td>
          <td>25.949122</td>
          <td>0.085684</td>
          <td>21.464369</td>
          <td>0.005201</td>
          <td>22.633799</td>
          <td>0.008022</td>
          <td>22.280405</td>
          <td>0.010140</td>
          <td>20.530884</td>
          <td>0.006545</td>
          <td>24.522300</td>
          <td>20.696366</td>
          <td>19.390134</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.846839</td>
          <td>0.016870</td>
          <td>21.344941</td>
          <td>0.005241</td>
          <td>22.586297</td>
          <td>0.006217</td>
          <td>14.441448</td>
          <td>0.005000</td>
          <td>23.228241</td>
          <td>0.021324</td>
          <td>20.686441</td>
          <td>0.006960</td>
          <td>22.279059</td>
          <td>17.116926</td>
          <td>27.789971</td>
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
          <td>22.183941</td>
          <td>0.010360</td>
          <td>21.462071</td>
          <td>0.005287</td>
          <td>24.426465</td>
          <td>0.019775</td>
          <td>25.339249</td>
          <td>0.071719</td>
          <td>24.017888</td>
          <td>0.042560</td>
          <td>19.906181</td>
          <td>0.005574</td>
          <td>16.361796</td>
          <td>19.093166</td>
          <td>20.090527</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.283880</td>
          <td>0.057797</td>
          <td>24.428437</td>
          <td>0.022510</td>
          <td>19.437360</td>
          <td>0.005012</td>
          <td>26.990263</td>
          <td>0.294073</td>
          <td>21.441459</td>
          <td>0.006523</td>
          <td>20.300872</td>
          <td>0.006078</td>
          <td>23.089908</td>
          <td>25.463571</td>
          <td>22.939065</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.748587</td>
          <td>0.005022</td>
          <td>20.647187</td>
          <td>0.005088</td>
          <td>16.873721</td>
          <td>0.005001</td>
          <td>21.959736</td>
          <td>0.006082</td>
          <td>24.133816</td>
          <td>0.047171</td>
          <td>21.738864</td>
          <td>0.013362</td>
          <td>23.573075</td>
          <td>23.613499</td>
          <td>21.342301</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.332842</td>
          <td>0.011456</td>
          <td>25.530654</td>
          <td>0.059213</td>
          <td>26.930789</td>
          <td>0.176930</td>
          <td>23.690096</td>
          <td>0.016969</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.461520</td>
          <td>0.058733</td>
          <td>26.203347</td>
          <td>21.288233</td>
          <td>23.560364</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.970919</td>
          <td>0.105568</td>
          <td>20.666377</td>
          <td>0.005091</td>
          <td>22.986700</td>
          <td>0.007252</td>
          <td>26.784357</td>
          <td>0.248672</td>
          <td>25.899935</td>
          <td>0.219361</td>
          <td>23.546314</td>
          <td>0.063320</td>
          <td>21.003524</td>
          <td>23.212226</td>
          <td>24.542139</td>
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
          <td>25.419841</td>
          <td>24.666224</td>
          <td>21.553050</td>
          <td>16.294962</td>
          <td>26.991513</td>
          <td>18.211069</td>
          <td>24.616722</td>
          <td>0.024417</td>
          <td>23.320094</td>
          <td>0.013594</td>
          <td>18.658936</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.391657</td>
          <td>22.998684</td>
          <td>27.976088</td>
          <td>24.845287</td>
          <td>23.755343</td>
          <td>21.914646</td>
          <td>20.865003</td>
          <td>0.005058</td>
          <td>21.164869</td>
          <td>0.005295</td>
          <td>24.728647</td>
          <td>0.045868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.623156</td>
          <td>18.767576</td>
          <td>24.419706</td>
          <td>15.421715</td>
          <td>22.779084</td>
          <td>20.989840</td>
          <td>23.324482</td>
          <td>0.008863</td>
          <td>21.376139</td>
          <td>0.005430</td>
          <td>21.879967</td>
          <td>0.006027</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.273743</td>
          <td>26.154370</td>
          <td>21.469264</td>
          <td>22.629823</td>
          <td>22.285213</td>
          <td>20.537738</td>
          <td>24.501314</td>
          <td>0.022085</td>
          <td>20.697821</td>
          <td>0.005127</td>
          <td>19.390841</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.805790</td>
          <td>21.346334</td>
          <td>22.591029</td>
          <td>14.441492</td>
          <td>23.200357</td>
          <td>20.687162</td>
          <td>22.282874</td>
          <td>0.005734</td>
          <td>17.123805</td>
          <td>0.005000</td>
          <td>27.450883</td>
          <td>0.459243</td>
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
          <td>22.174058</td>
          <td>21.458752</td>
          <td>24.433097</td>
          <td>25.167819</td>
          <td>24.003333</td>
          <td>19.908218</td>
          <td>16.353886</td>
          <td>0.005000</td>
          <td>19.092971</td>
          <td>0.005007</td>
          <td>20.079791</td>
          <td>0.005041</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.313887</td>
          <td>24.395679</td>
          <td>19.450278</td>
          <td>26.521361</td>
          <td>21.439028</td>
          <td>20.303513</td>
          <td>23.083552</td>
          <td>0.007707</td>
          <td>25.345538</td>
          <td>0.079396</td>
          <td>22.941080</td>
          <td>0.010234</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.754115</td>
          <td>20.647016</td>
          <td>16.877253</td>
          <td>21.966344</td>
          <td>24.053722</td>
          <td>21.757548</td>
          <td>23.574636</td>
          <td>0.010478</td>
          <td>23.603173</td>
          <td>0.017128</td>
          <td>21.336925</td>
          <td>0.005401</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.332927</td>
          <td>25.490199</td>
          <td>27.132583</td>
          <td>23.687093</td>
          <td>27.724105</td>
          <td>23.404066</td>
          <td>26.189821</td>
          <td>0.098481</td>
          <td>21.287530</td>
          <td>0.005367</td>
          <td>23.564893</td>
          <td>0.016590</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.853095</td>
          <td>20.671679</td>
          <td>22.993174</td>
          <td>26.814697</td>
          <td>26.335310</td>
          <td>23.581529</td>
          <td>21.003261</td>
          <td>0.005074</td>
          <td>23.222995</td>
          <td>0.012600</td>
          <td>24.510624</td>
          <td>0.037770</td>
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
          <td>25.419841</td>
          <td>24.666224</td>
          <td>21.553050</td>
          <td>16.294962</td>
          <td>26.991513</td>
          <td>18.211069</td>
          <td>24.258220</td>
          <td>0.191151</td>
          <td>23.334405</td>
          <td>0.071951</td>
          <td>18.662910</td>
          <td>0.005119</td>
        </tr>
        <tr>
          <th>1</th>
          <td>27.391657</td>
          <td>22.998684</td>
          <td>27.976088</td>
          <td>24.845287</td>
          <td>23.755343</td>
          <td>21.914646</td>
          <td>20.883450</td>
          <td>0.010544</td>
          <td>21.151116</td>
          <td>0.011071</td>
          <td>25.218945</td>
          <td>0.384720</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.623156</td>
          <td>18.767576</td>
          <td>24.419706</td>
          <td>15.421715</td>
          <td>22.779084</td>
          <td>20.989840</td>
          <td>23.385605</td>
          <td>0.089851</td>
          <td>21.370696</td>
          <td>0.013076</td>
          <td>21.867111</td>
          <td>0.021442</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.273743</td>
          <td>26.154370</td>
          <td>21.469264</td>
          <td>22.629823</td>
          <td>22.285213</td>
          <td>20.537738</td>
          <td>24.240852</td>
          <td>0.188368</td>
          <td>20.692704</td>
          <td>0.008188</td>
          <td>19.390345</td>
          <td>0.005440</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.805790</td>
          <td>21.346334</td>
          <td>22.591029</td>
          <td>14.441492</td>
          <td>23.200357</td>
          <td>20.687162</td>
          <td>22.350886</td>
          <td>0.035816</td>
          <td>17.113424</td>
          <td>0.005006</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>22.174058</td>
          <td>21.458752</td>
          <td>24.433097</td>
          <td>25.167819</td>
          <td>24.003333</td>
          <td>19.908218</td>
          <td>16.359993</td>
          <td>0.005002</td>
          <td>19.086532</td>
          <td>0.005214</td>
          <td>20.088412</td>
          <td>0.006452</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.313887</td>
          <td>24.395679</td>
          <td>19.450278</td>
          <td>26.521361</td>
          <td>21.439028</td>
          <td>20.303513</td>
          <td>23.108254</td>
          <td>0.070300</td>
          <td>26.112089</td>
          <td>0.687021</td>
          <td>22.960174</td>
          <td>0.056381</td>
        </tr>
        <tr>
          <th>997</th>
          <td>17.754115</td>
          <td>20.647016</td>
          <td>16.877253</td>
          <td>21.966344</td>
          <td>24.053722</td>
          <td>21.757548</td>
          <td>23.699313</td>
          <td>0.118292</td>
          <td>23.591714</td>
          <td>0.090336</td>
          <td>21.350926</td>
          <td>0.013932</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.332927</td>
          <td>25.490199</td>
          <td>27.132583</td>
          <td>23.687093</td>
          <td>27.724105</td>
          <td>23.404066</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.266847</td>
          <td>0.012069</td>
          <td>23.617629</td>
          <td>0.100915</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.853095</td>
          <td>20.671679</td>
          <td>22.993174</td>
          <td>26.814697</td>
          <td>26.335310</td>
          <td>23.581529</td>
          <td>21.019943</td>
          <td>0.011650</td>
          <td>23.297867</td>
          <td>0.069655</td>
          <td>24.646161</td>
          <td>0.243090</td>
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


