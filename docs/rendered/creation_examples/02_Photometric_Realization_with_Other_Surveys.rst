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
          <td>22.492229</td>
          <td>19.634940</td>
          <td>22.833583</td>
          <td>22.562030</td>
          <td>24.163095</td>
          <td>27.061375</td>
          <td>23.705392</td>
          <td>18.986174</td>
          <td>20.526531</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.006208</td>
          <td>19.909620</td>
          <td>25.887011</td>
          <td>24.225016</td>
          <td>20.575707</td>
          <td>25.508392</td>
          <td>22.332140</td>
          <td>25.215812</td>
          <td>15.267452</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536859</td>
          <td>21.149550</td>
          <td>19.069620</td>
          <td>22.669866</td>
          <td>26.105834</td>
          <td>27.012914</td>
          <td>24.543735</td>
          <td>28.310486</td>
          <td>23.310527</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.142281</td>
          <td>21.198196</td>
          <td>25.673269</td>
          <td>18.972050</td>
          <td>23.988603</td>
          <td>19.751967</td>
          <td>25.736626</td>
          <td>19.260922</td>
          <td>20.691310</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.563803</td>
          <td>14.957619</td>
          <td>20.703558</td>
          <td>18.578440</td>
          <td>24.367458</td>
          <td>23.304919</td>
          <td>23.847756</td>
          <td>22.141761</td>
          <td>17.386768</td>
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
          <td>28.607954</td>
          <td>19.757729</td>
          <td>17.357579</td>
          <td>17.411043</td>
          <td>20.336853</td>
          <td>20.116865</td>
          <td>22.648812</td>
          <td>25.084836</td>
          <td>23.682454</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.714277</td>
          <td>20.105174</td>
          <td>26.746382</td>
          <td>22.856628</td>
          <td>27.428612</td>
          <td>24.397995</td>
          <td>20.882651</td>
          <td>24.462285</td>
          <td>20.377909</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.315710</td>
          <td>22.166762</td>
          <td>21.941737</td>
          <td>28.307035</td>
          <td>25.075650</td>
          <td>25.367446</td>
          <td>22.942510</td>
          <td>24.426624</td>
          <td>23.592260</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.569138</td>
          <td>24.479101</td>
          <td>21.207797</td>
          <td>19.889643</td>
          <td>18.097657</td>
          <td>19.429645</td>
          <td>20.039196</td>
          <td>21.713832</td>
          <td>26.357182</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.878053</td>
          <td>21.836828</td>
          <td>21.065879</td>
          <td>25.813123</td>
          <td>19.624587</td>
          <td>22.169869</td>
          <td>22.394926</td>
          <td>22.843495</td>
          <td>24.374460</td>
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
          <td>22.475851</td>
          <td>0.012687</td>
          <td>19.640222</td>
          <td>0.005025</td>
          <td>22.832644</td>
          <td>0.006784</td>
          <td>22.561863</td>
          <td>0.007722</td>
          <td>24.154866</td>
          <td>0.048060</td>
          <td>26.668924</td>
          <td>0.784211</td>
          <td>23.705392</td>
          <td>18.986174</td>
          <td>20.526531</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.009015</td>
          <td>0.005280</td>
          <td>19.912689</td>
          <td>0.005034</td>
          <td>25.890352</td>
          <td>0.071564</td>
          <td>24.214529</td>
          <td>0.026526</td>
          <td>20.575435</td>
          <td>0.005384</td>
          <td>25.856474</td>
          <td>0.441150</td>
          <td>22.332140</td>
          <td>25.215812</td>
          <td>15.267452</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.547158</td>
          <td>0.005007</td>
          <td>21.159832</td>
          <td>0.005183</td>
          <td>19.059693</td>
          <td>0.005007</td>
          <td>22.666896</td>
          <td>0.008170</td>
          <td>26.262573</td>
          <td>0.295306</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.543735</td>
          <td>28.310486</td>
          <td>23.310527</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.167803</td>
          <td>0.052184</td>
          <td>21.199561</td>
          <td>0.005194</td>
          <td>25.692534</td>
          <td>0.060057</td>
          <td>18.970148</td>
          <td>0.005013</td>
          <td>24.023274</td>
          <td>0.042764</td>
          <td>19.752792</td>
          <td>0.005449</td>
          <td>25.736626</td>
          <td>19.260922</td>
          <td>20.691310</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.566572</td>
          <td>0.005160</td>
          <td>14.961122</td>
          <td>0.005000</td>
          <td>20.705627</td>
          <td>0.005063</td>
          <td>18.588097</td>
          <td>0.005008</td>
          <td>24.411398</td>
          <td>0.060350</td>
          <td>23.369555</td>
          <td>0.054130</td>
          <td>23.847756</td>
          <td>22.141761</td>
          <td>17.386768</td>
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
          <td>27.791368</td>
          <td>0.933358</td>
          <td>19.762557</td>
          <td>0.005028</td>
          <td>17.353136</td>
          <td>0.005001</td>
          <td>17.408116</td>
          <td>0.005002</td>
          <td>20.342906</td>
          <td>0.005264</td>
          <td>20.111898</td>
          <td>0.005799</td>
          <td>22.648812</td>
          <td>25.084836</td>
          <td>23.682454</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.767261</td>
          <td>0.036726</td>
          <td>20.100994</td>
          <td>0.005043</td>
          <td>26.690667</td>
          <td>0.144104</td>
          <td>22.853152</td>
          <td>0.009124</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.347519</td>
          <td>0.127986</td>
          <td>20.882651</td>
          <td>24.462285</td>
          <td>20.377909</td>
        </tr>
        <tr>
          <th>997</th>
          <td>inf</td>
          <td>inf</td>
          <td>22.167378</td>
          <td>0.005845</td>
          <td>21.944557</td>
          <td>0.005435</td>
          <td>30.252729</td>
          <td>2.154478</td>
          <td>25.096300</td>
          <td>0.110354</td>
          <td>25.041875</td>
          <td>0.230801</td>
          <td>22.942510</td>
          <td>24.426624</td>
          <td>23.592260</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.571039</td>
          <td>0.005161</td>
          <td>24.459978</td>
          <td>0.023126</td>
          <td>21.202515</td>
          <td>0.005134</td>
          <td>19.895875</td>
          <td>0.005043</td>
          <td>18.099301</td>
          <td>0.005011</td>
          <td>19.434659</td>
          <td>0.005270</td>
          <td>20.039196</td>
          <td>21.713832</td>
          <td>26.357182</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.882790</td>
          <td>0.005072</td>
          <td>21.825108</td>
          <td>0.005499</td>
          <td>21.057259</td>
          <td>0.005107</td>
          <td>25.923941</td>
          <td>0.119840</td>
          <td>19.625416</td>
          <td>0.005087</td>
          <td>22.178659</td>
          <td>0.019094</td>
          <td>22.394926</td>
          <td>22.843495</td>
          <td>24.374460</td>
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
          <td>22.492229</td>
          <td>19.634940</td>
          <td>22.833583</td>
          <td>22.562030</td>
          <td>24.163095</td>
          <td>27.061375</td>
          <td>23.712977</td>
          <td>0.011589</td>
          <td>18.986028</td>
          <td>0.005005</td>
          <td>20.529306</td>
          <td>0.005093</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.006208</td>
          <td>19.909620</td>
          <td>25.887011</td>
          <td>24.225016</td>
          <td>20.575707</td>
          <td>25.508392</td>
          <td>22.332457</td>
          <td>0.005800</td>
          <td>25.127123</td>
          <td>0.065412</td>
          <td>15.271878</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536859</td>
          <td>21.149550</td>
          <td>19.069620</td>
          <td>22.669866</td>
          <td>26.105834</td>
          <td>27.012914</td>
          <td>24.541459</td>
          <td>0.022867</td>
          <td>28.171007</td>
          <td>0.764435</td>
          <td>23.275595</td>
          <td>0.013126</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.142281</td>
          <td>21.198196</td>
          <td>25.673269</td>
          <td>18.972050</td>
          <td>23.988603</td>
          <td>19.751967</td>
          <td>25.669043</td>
          <td>0.062119</td>
          <td>19.257601</td>
          <td>0.005009</td>
          <td>20.686991</td>
          <td>0.005124</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.563803</td>
          <td>14.957619</td>
          <td>20.703558</td>
          <td>18.578440</td>
          <td>24.367458</td>
          <td>23.304919</td>
          <td>23.857071</td>
          <td>0.012937</td>
          <td>22.138174</td>
          <td>0.006574</td>
          <td>17.397653</td>
          <td>0.005000</td>
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
          <td>28.607954</td>
          <td>19.757729</td>
          <td>17.357579</td>
          <td>17.411043</td>
          <td>20.336853</td>
          <td>20.116865</td>
          <td>22.648992</td>
          <td>0.006361</td>
          <td>25.132442</td>
          <td>0.065722</td>
          <td>23.677663</td>
          <td>0.018236</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.714277</td>
          <td>20.105174</td>
          <td>26.746382</td>
          <td>22.856628</td>
          <td>27.428612</td>
          <td>24.397995</td>
          <td>20.882119</td>
          <td>0.005059</td>
          <td>24.419232</td>
          <td>0.034823</td>
          <td>20.372633</td>
          <td>0.005070</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.315710</td>
          <td>22.166762</td>
          <td>21.941737</td>
          <td>28.307035</td>
          <td>25.075650</td>
          <td>25.367446</td>
          <td>22.937728</td>
          <td>0.007163</td>
          <td>24.400163</td>
          <td>0.034238</td>
          <td>23.569862</td>
          <td>0.016658</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.569138</td>
          <td>24.479101</td>
          <td>21.207797</td>
          <td>19.889643</td>
          <td>18.097657</td>
          <td>19.429645</td>
          <td>20.038185</td>
          <td>0.005013</td>
          <td>21.715217</td>
          <td>0.005776</td>
          <td>26.023508</td>
          <td>0.143661</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.878053</td>
          <td>21.836828</td>
          <td>21.065879</td>
          <td>25.813123</td>
          <td>19.624587</td>
          <td>22.169869</td>
          <td>22.401224</td>
          <td>0.005899</td>
          <td>22.843562</td>
          <td>0.009574</td>
          <td>24.404854</td>
          <td>0.034381</td>
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
          <td>22.492229</td>
          <td>19.634940</td>
          <td>22.833583</td>
          <td>22.562030</td>
          <td>24.163095</td>
          <td>27.061375</td>
          <td>23.755434</td>
          <td>0.124212</td>
          <td>18.989024</td>
          <td>0.005179</td>
          <td>20.519379</td>
          <td>0.007857</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.006208</td>
          <td>19.909620</td>
          <td>25.887011</td>
          <td>24.225016</td>
          <td>20.575707</td>
          <td>25.508392</td>
          <td>22.360514</td>
          <td>0.036124</td>
          <td>26.914958</td>
          <td>1.136857</td>
          <td>15.273223</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.536859</td>
          <td>21.149550</td>
          <td>19.069620</td>
          <td>22.669866</td>
          <td>26.105834</td>
          <td>27.012914</td>
          <td>25.032057</td>
          <td>0.359518</td>
          <td>26.919475</td>
          <td>1.139791</td>
          <td>23.288619</td>
          <td>0.075494</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.142281</td>
          <td>21.198196</td>
          <td>25.673269</td>
          <td>18.972050</td>
          <td>23.988603</td>
          <td>19.751967</td>
          <td>25.346272</td>
          <td>0.457655</td>
          <td>19.262411</td>
          <td>0.005294</td>
          <td>20.701150</td>
          <td>0.008736</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.563803</td>
          <td>14.957619</td>
          <td>20.703558</td>
          <td>18.578440</td>
          <td>24.367458</td>
          <td>23.304919</td>
          <td>23.750745</td>
          <td>0.123707</td>
          <td>22.121927</td>
          <td>0.024528</td>
          <td>17.391579</td>
          <td>0.005012</td>
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
          <td>28.607954</td>
          <td>19.757729</td>
          <td>17.357579</td>
          <td>17.411043</td>
          <td>20.336853</td>
          <td>20.116865</td>
          <td>22.664219</td>
          <td>0.047346</td>
          <td>25.058463</td>
          <td>0.313334</td>
          <td>23.731113</td>
          <td>0.111459</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.714277</td>
          <td>20.105174</td>
          <td>26.746382</td>
          <td>22.856628</td>
          <td>27.428612</td>
          <td>24.397995</td>
          <td>20.891436</td>
          <td>0.010604</td>
          <td>24.481517</td>
          <td>0.194943</td>
          <td>20.370240</td>
          <td>0.007275</td>
        </tr>
        <tr>
          <th>997</th>
          <td>29.315710</td>
          <td>22.166762</td>
          <td>21.941737</td>
          <td>28.307035</td>
          <td>25.075650</td>
          <td>25.367446</td>
          <td>22.943862</td>
          <td>0.060743</td>
          <td>24.227724</td>
          <td>0.157120</td>
          <td>23.604509</td>
          <td>0.099759</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.569138</td>
          <td>24.479101</td>
          <td>21.207797</td>
          <td>19.889643</td>
          <td>18.097657</td>
          <td>19.429645</td>
          <td>20.033425</td>
          <td>0.006562</td>
          <td>21.739823</td>
          <td>0.017663</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.878053</td>
          <td>21.836828</td>
          <td>21.065879</td>
          <td>25.813123</td>
          <td>19.624587</td>
          <td>22.169869</td>
          <td>22.440904</td>
          <td>0.038802</td>
          <td>22.791153</td>
          <td>0.044360</td>
          <td>24.902069</td>
          <td>0.299473</td>
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


