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
          <td>24.195002</td>
          <td>24.013165</td>
          <td>22.794386</td>
          <td>20.909082</td>
          <td>18.478031</td>
          <td>25.919858</td>
          <td>26.057625</td>
          <td>28.295276</td>
          <td>27.294831</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.842345</td>
          <td>21.120121</td>
          <td>18.877012</td>
          <td>17.565068</td>
          <td>17.815326</td>
          <td>29.947192</td>
          <td>19.859552</td>
          <td>22.029710</td>
          <td>22.739128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.668899</td>
          <td>20.220204</td>
          <td>21.026616</td>
          <td>25.627233</td>
          <td>22.991764</td>
          <td>28.168698</td>
          <td>21.701298</td>
          <td>23.985579</td>
          <td>19.860708</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.454497</td>
          <td>20.155820</td>
          <td>20.740059</td>
          <td>22.469224</td>
          <td>17.762545</td>
          <td>30.053555</td>
          <td>21.108209</td>
          <td>21.353098</td>
          <td>19.282182</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.002238</td>
          <td>21.746827</td>
          <td>24.314866</td>
          <td>15.040410</td>
          <td>24.076397</td>
          <td>21.090395</td>
          <td>26.444884</td>
          <td>19.689185</td>
          <td>21.372453</td>
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
          <td>20.459984</td>
          <td>23.165030</td>
          <td>24.818906</td>
          <td>21.651720</td>
          <td>19.854544</td>
          <td>21.200718</td>
          <td>22.731021</td>
          <td>17.939475</td>
          <td>22.008686</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.760495</td>
          <td>23.737047</td>
          <td>24.909816</td>
          <td>24.768005</td>
          <td>20.083935</td>
          <td>32.924908</td>
          <td>26.642959</td>
          <td>22.886496</td>
          <td>21.172369</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.472219</td>
          <td>19.988879</td>
          <td>24.549968</td>
          <td>20.955355</td>
          <td>18.407652</td>
          <td>19.241541</td>
          <td>23.110399</td>
          <td>24.122737</td>
          <td>19.701861</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.699512</td>
          <td>17.383599</td>
          <td>28.850153</td>
          <td>19.983647</td>
          <td>21.439425</td>
          <td>27.229086</td>
          <td>24.703031</td>
          <td>26.870576</td>
          <td>19.060949</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.836299</td>
          <td>21.137938</td>
          <td>21.728392</td>
          <td>19.515814</td>
          <td>22.226508</td>
          <td>20.051031</td>
          <td>23.396739</td>
          <td>18.825843</td>
          <td>25.428718</td>
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
          <td>24.185439</td>
          <td>0.053000</td>
          <td>23.991795</td>
          <td>0.015652</td>
          <td>22.797261</td>
          <td>0.006689</td>
          <td>20.916229</td>
          <td>0.005202</td>
          <td>18.480073</td>
          <td>0.005018</td>
          <td>25.687569</td>
          <td>0.387649</td>
          <td>26.057625</td>
          <td>28.295276</td>
          <td>27.294831</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.841042</td>
          <td>0.005864</td>
          <td>21.123603</td>
          <td>0.005174</td>
          <td>18.876057</td>
          <td>0.005006</td>
          <td>17.562416</td>
          <td>0.005003</td>
          <td>17.817123</td>
          <td>0.005008</td>
          <td>26.948541</td>
          <td>0.936952</td>
          <td>19.859552</td>
          <td>22.029710</td>
          <td>22.739128</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.672542</td>
          <td>0.007741</td>
          <td>20.217376</td>
          <td>0.005050</td>
          <td>21.024277</td>
          <td>0.005102</td>
          <td>25.552506</td>
          <td>0.086576</td>
          <td>23.011944</td>
          <td>0.017767</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.701298</td>
          <td>23.985579</td>
          <td>19.860708</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.440207</td>
          <td>0.012363</td>
          <td>20.148721</td>
          <td>0.005046</td>
          <td>20.740648</td>
          <td>0.005066</td>
          <td>22.474524</td>
          <td>0.007393</td>
          <td>17.755750</td>
          <td>0.005007</td>
          <td>29.832104</td>
          <td>3.279638</td>
          <td>21.108209</td>
          <td>21.353098</td>
          <td>19.282182</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.977751</td>
          <td>0.106198</td>
          <td>21.743207</td>
          <td>0.005440</td>
          <td>24.353013</td>
          <td>0.018588</td>
          <td>15.038639</td>
          <td>0.005000</td>
          <td>24.105609</td>
          <td>0.046004</td>
          <td>21.083175</td>
          <td>0.008507</td>
          <td>26.444884</td>
          <td>19.689185</td>
          <td>21.372453</td>
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
          <td>20.468546</td>
          <td>0.005517</td>
          <td>23.172371</td>
          <td>0.008736</td>
          <td>24.823930</td>
          <td>0.027854</td>
          <td>21.648888</td>
          <td>0.005658</td>
          <td>19.852869</td>
          <td>0.005123</td>
          <td>21.197334</td>
          <td>0.009114</td>
          <td>22.731021</td>
          <td>17.939475</td>
          <td>22.008686</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.759807</td>
          <td>0.008083</td>
          <td>23.702146</td>
          <td>0.012494</td>
          <td>24.892311</td>
          <td>0.029573</td>
          <td>24.734271</td>
          <td>0.041931</td>
          <td>20.080740</td>
          <td>0.005175</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.642959</td>
          <td>22.886496</td>
          <td>21.172369</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.477819</td>
          <td>0.005144</td>
          <td>19.991871</td>
          <td>0.005038</td>
          <td>24.543077</td>
          <td>0.021839</td>
          <td>20.962008</td>
          <td>0.005218</td>
          <td>18.398980</td>
          <td>0.005016</td>
          <td>19.233671</td>
          <td>0.005196</td>
          <td>23.110399</td>
          <td>24.122737</td>
          <td>19.701861</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.697608</td>
          <td>0.005059</td>
          <td>17.384029</td>
          <td>0.005002</td>
          <td>27.883208</td>
          <td>0.384635</td>
          <td>19.974139</td>
          <td>0.005048</td>
          <td>21.445275</td>
          <td>0.006532</td>
          <td>26.244777</td>
          <td>0.586734</td>
          <td>24.703031</td>
          <td>26.870576</td>
          <td>19.060949</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.757318</td>
          <td>0.036408</td>
          <td>21.137133</td>
          <td>0.005177</td>
          <td>21.736094</td>
          <td>0.005311</td>
          <td>19.519928</td>
          <td>0.005026</td>
          <td>22.246031</td>
          <td>0.009908</td>
          <td>20.048904</td>
          <td>0.005722</td>
          <td>23.396739</td>
          <td>18.825843</td>
          <td>25.428718</td>
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
          <td>24.195002</td>
          <td>24.013165</td>
          <td>22.794386</td>
          <td>20.909082</td>
          <td>18.478031</td>
          <td>25.919858</td>
          <td>26.065423</td>
          <td>0.088267</td>
          <td>27.520954</td>
          <td>0.483924</td>
          <td>27.565293</td>
          <td>0.500078</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.842345</td>
          <td>21.120121</td>
          <td>18.877012</td>
          <td>17.565068</td>
          <td>17.815326</td>
          <td>29.947192</td>
          <td>19.862266</td>
          <td>0.005009</td>
          <td>22.029750</td>
          <td>0.006319</td>
          <td>22.723880</td>
          <td>0.008860</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.668899</td>
          <td>20.220204</td>
          <td>21.026616</td>
          <td>25.627233</td>
          <td>22.991764</td>
          <td>28.168698</td>
          <td>21.693857</td>
          <td>0.005260</td>
          <td>24.017825</td>
          <td>0.024440</td>
          <td>19.860583</td>
          <td>0.005027</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.454497</td>
          <td>20.155820</td>
          <td>20.740059</td>
          <td>22.469224</td>
          <td>17.762545</td>
          <td>30.053555</td>
          <td>21.104942</td>
          <td>0.005089</td>
          <td>21.356521</td>
          <td>0.005415</td>
          <td>19.282573</td>
          <td>0.005009</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.002238</td>
          <td>21.746827</td>
          <td>24.314866</td>
          <td>15.040410</td>
          <td>24.076397</td>
          <td>21.090395</td>
          <td>26.306633</td>
          <td>0.109099</td>
          <td>19.689232</td>
          <td>0.005020</td>
          <td>21.373242</td>
          <td>0.005427</td>
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
          <td>20.459984</td>
          <td>23.165030</td>
          <td>24.818906</td>
          <td>21.651720</td>
          <td>19.854544</td>
          <td>21.200718</td>
          <td>22.729855</td>
          <td>0.006553</td>
          <td>17.944164</td>
          <td>0.005001</td>
          <td>22.013937</td>
          <td>0.006285</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.760495</td>
          <td>23.737047</td>
          <td>24.909816</td>
          <td>24.768005</td>
          <td>20.083935</td>
          <td>32.924908</td>
          <td>26.695686</td>
          <td>0.152862</td>
          <td>22.883426</td>
          <td>0.009835</td>
          <td>21.173866</td>
          <td>0.005300</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.472219</td>
          <td>19.988879</td>
          <td>24.549968</td>
          <td>20.955355</td>
          <td>18.407652</td>
          <td>19.241541</td>
          <td>23.112877</td>
          <td>0.007829</td>
          <td>24.157001</td>
          <td>0.027613</td>
          <td>19.702025</td>
          <td>0.005020</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.699512</td>
          <td>17.383599</td>
          <td>28.850153</td>
          <td>19.983647</td>
          <td>21.439425</td>
          <td>27.229086</td>
          <td>24.698822</td>
          <td>0.026236</td>
          <td>26.529198</td>
          <td>0.220624</td>
          <td>19.065904</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.836299</td>
          <td>21.137938</td>
          <td>21.728392</td>
          <td>19.515814</td>
          <td>22.226508</td>
          <td>20.051031</td>
          <td>23.396130</td>
          <td>0.009279</td>
          <td>18.831092</td>
          <td>0.005004</td>
          <td>25.388420</td>
          <td>0.082465</td>
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
          <td>24.195002</td>
          <td>24.013165</td>
          <td>22.794386</td>
          <td>20.909082</td>
          <td>18.478031</td>
          <td>25.919858</td>
          <td>25.534235</td>
          <td>0.526034</td>
          <td>27.078446</td>
          <td>1.245727</td>
          <td>27.664586</td>
          <td>1.757045</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.842345</td>
          <td>21.120121</td>
          <td>18.877012</td>
          <td>17.565068</td>
          <td>17.815326</td>
          <td>29.947192</td>
          <td>19.855191</td>
          <td>0.006165</td>
          <td>22.004561</td>
          <td>0.022147</td>
          <td>22.734447</td>
          <td>0.046105</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.668899</td>
          <td>20.220204</td>
          <td>21.026616</td>
          <td>25.627233</td>
          <td>22.991764</td>
          <td>28.168698</td>
          <td>21.718597</td>
          <td>0.020564</td>
          <td>23.915751</td>
          <td>0.119998</td>
          <td>19.874203</td>
          <td>0.006017</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.454497</td>
          <td>20.155820</td>
          <td>20.740059</td>
          <td>22.469224</td>
          <td>17.762545</td>
          <td>30.053555</td>
          <td>21.105952</td>
          <td>0.012435</td>
          <td>21.358506</td>
          <td>0.012952</td>
          <td>19.273463</td>
          <td>0.005358</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.002238</td>
          <td>21.746827</td>
          <td>24.314866</td>
          <td>15.040410</td>
          <td>24.076397</td>
          <td>21.090395</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.686053</td>
          <td>0.005621</td>
          <td>21.395241</td>
          <td>0.014436</td>
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
          <td>20.459984</td>
          <td>23.165030</td>
          <td>24.818906</td>
          <td>21.651720</td>
          <td>19.854544</td>
          <td>21.200718</td>
          <td>22.708997</td>
          <td>0.049274</td>
          <td>17.923189</td>
          <td>0.005026</td>
          <td>22.020772</td>
          <td>0.024503</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.760495</td>
          <td>23.737047</td>
          <td>24.909816</td>
          <td>24.768005</td>
          <td>20.083935</td>
          <td>32.924908</td>
          <td>25.580733</td>
          <td>0.544124</td>
          <td>22.888607</td>
          <td>0.048386</td>
          <td>21.171000</td>
          <td>0.012107</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.472219</td>
          <td>19.988879</td>
          <td>24.549968</td>
          <td>20.955355</td>
          <td>18.407652</td>
          <td>19.241541</td>
          <td>23.166529</td>
          <td>0.074030</td>
          <td>24.099893</td>
          <td>0.140765</td>
          <td>19.701415</td>
          <td>0.005758</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.699512</td>
          <td>17.383599</td>
          <td>28.850153</td>
          <td>19.983647</td>
          <td>21.439425</td>
          <td>27.229086</td>
          <td>24.576851</td>
          <td>0.249315</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.059764</td>
          <td>0.005244</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.836299</td>
          <td>21.137938</td>
          <td>21.728392</td>
          <td>19.515814</td>
          <td>22.226508</td>
          <td>20.051031</td>
          <td>23.328968</td>
          <td>0.085472</td>
          <td>18.832933</td>
          <td>0.005135</td>
          <td>25.092511</td>
          <td>0.348517</td>
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


