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
          <td>24.285667</td>
          <td>25.590233</td>
          <td>16.617922</td>
          <td>24.562484</td>
          <td>21.465455</td>
          <td>28.544651</td>
          <td>24.309149</td>
          <td>18.939883</td>
          <td>26.857309</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.180008</td>
          <td>18.698606</td>
          <td>22.941177</td>
          <td>25.803891</td>
          <td>25.079153</td>
          <td>21.174363</td>
          <td>21.198175</td>
          <td>23.750913</td>
          <td>21.501203</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.044366</td>
          <td>22.915437</td>
          <td>17.967426</td>
          <td>22.540905</td>
          <td>24.316717</td>
          <td>24.830198</td>
          <td>16.462042</td>
          <td>24.928128</td>
          <td>23.398577</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.202065</td>
          <td>22.224725</td>
          <td>21.315880</td>
          <td>24.615585</td>
          <td>22.169187</td>
          <td>26.962035</td>
          <td>16.490912</td>
          <td>21.078514</td>
          <td>20.449170</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.177215</td>
          <td>20.698442</td>
          <td>28.144760</td>
          <td>20.571255</td>
          <td>22.594295</td>
          <td>27.601257</td>
          <td>21.724651</td>
          <td>20.292009</td>
          <td>24.899499</td>
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
          <td>25.620319</td>
          <td>22.666545</td>
          <td>27.253885</td>
          <td>29.431615</td>
          <td>22.007556</td>
          <td>24.197069</td>
          <td>21.963785</td>
          <td>20.315539</td>
          <td>26.127973</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.025056</td>
          <td>19.493729</td>
          <td>21.518186</td>
          <td>26.424909</td>
          <td>25.779826</td>
          <td>18.689472</td>
          <td>21.791130</td>
          <td>25.186951</td>
          <td>22.489087</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.040693</td>
          <td>21.350591</td>
          <td>27.786338</td>
          <td>25.012345</td>
          <td>23.379752</td>
          <td>24.002757</td>
          <td>24.158793</td>
          <td>26.916906</td>
          <td>23.705161</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.979771</td>
          <td>23.641341</td>
          <td>26.081023</td>
          <td>19.864231</td>
          <td>21.264162</td>
          <td>22.497700</td>
          <td>20.932647</td>
          <td>20.917969</td>
          <td>20.907867</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.387834</td>
          <td>25.285670</td>
          <td>16.802013</td>
          <td>20.761224</td>
          <td>20.218009</td>
          <td>22.460587</td>
          <td>20.145952</td>
          <td>25.330269</td>
          <td>23.616329</td>
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
          <td>24.329903</td>
          <td>0.060186</td>
          <td>25.550507</td>
          <td>0.060263</td>
          <td>16.622997</td>
          <td>0.005001</td>
          <td>24.582581</td>
          <td>0.036659</td>
          <td>21.458990</td>
          <td>0.006565</td>
          <td>25.306754</td>
          <td>0.286716</td>
          <td>24.309149</td>
          <td>18.939883</td>
          <td>26.857309</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.129700</td>
          <td>0.021263</td>
          <td>18.697197</td>
          <td>0.005009</td>
          <td>22.929052</td>
          <td>0.007065</td>
          <td>25.671191</td>
          <td>0.096097</td>
          <td>25.127382</td>
          <td>0.113386</td>
          <td>21.176726</td>
          <td>0.008998</td>
          <td>21.198175</td>
          <td>23.750913</td>
          <td>21.501203</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.061090</td>
          <td>0.006176</td>
          <td>22.904164</td>
          <td>0.007558</td>
          <td>17.962031</td>
          <td>0.005002</td>
          <td>22.547024</td>
          <td>0.007663</td>
          <td>24.318159</td>
          <td>0.055558</td>
          <td>24.863807</td>
          <td>0.198926</td>
          <td>16.462042</td>
          <td>24.928128</td>
          <td>23.398577</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.197482</td>
          <td>0.006423</td>
          <td>22.222808</td>
          <td>0.005920</td>
          <td>21.308168</td>
          <td>0.005158</td>
          <td>24.579206</td>
          <td>0.036550</td>
          <td>22.158684</td>
          <td>0.009359</td>
          <td>26.539225</td>
          <td>0.719449</td>
          <td>16.490912</td>
          <td>21.078514</td>
          <td>20.449170</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.391917</td>
          <td>0.346471</td>
          <td>20.683464</td>
          <td>0.005093</td>
          <td>28.613231</td>
          <td>0.657647</td>
          <td>20.578871</td>
          <td>0.005119</td>
          <td>22.580195</td>
          <td>0.012585</td>
          <td>30.412830</td>
          <td>3.838200</td>
          <td>21.724651</td>
          <td>20.292009</td>
          <td>24.899499</td>
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
          <td>25.435866</td>
          <td>0.157644</td>
          <td>22.664275</td>
          <td>0.006799</td>
          <td>27.287105</td>
          <td>0.238467</td>
          <td>28.845361</td>
          <td>1.085039</td>
          <td>22.004492</td>
          <td>0.008516</td>
          <td>24.373057</td>
          <td>0.130847</td>
          <td>21.963785</td>
          <td>20.315539</td>
          <td>26.127973</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.241623</td>
          <td>0.133462</td>
          <td>19.490002</td>
          <td>0.005021</td>
          <td>21.517453</td>
          <td>0.005219</td>
          <td>26.344791</td>
          <td>0.172126</td>
          <td>25.548376</td>
          <td>0.163063</td>
          <td>18.695246</td>
          <td>0.005086</td>
          <td>21.791130</td>
          <td>25.186951</td>
          <td>22.489087</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.034710</td>
          <td>0.005085</td>
          <td>21.360375</td>
          <td>0.005246</td>
          <td>28.781915</td>
          <td>0.737523</td>
          <td>25.078225</td>
          <td>0.056904</td>
          <td>23.394278</td>
          <td>0.024598</td>
          <td>23.890444</td>
          <td>0.085832</td>
          <td>24.158793</td>
          <td>26.916906</td>
          <td>23.705161</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.411352</td>
          <td>0.351805</td>
          <td>23.650599</td>
          <td>0.012025</td>
          <td>26.069324</td>
          <td>0.083819</td>
          <td>19.861585</td>
          <td>0.005041</td>
          <td>21.262594</td>
          <td>0.006152</td>
          <td>22.458512</td>
          <td>0.024251</td>
          <td>20.932647</td>
          <td>20.917969</td>
          <td>20.907867</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.382798</td>
          <td>0.005128</td>
          <td>25.260505</td>
          <td>0.046613</td>
          <td>16.806957</td>
          <td>0.005001</td>
          <td>20.767611</td>
          <td>0.005160</td>
          <td>20.214325</td>
          <td>0.005216</td>
          <td>22.458391</td>
          <td>0.024248</td>
          <td>20.145952</td>
          <td>25.330269</td>
          <td>23.616329</td>
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
          <td>24.285667</td>
          <td>25.590233</td>
          <td>16.617922</td>
          <td>24.562484</td>
          <td>21.465455</td>
          <td>28.544651</td>
          <td>24.287857</td>
          <td>0.018394</td>
          <td>18.946979</td>
          <td>0.005005</td>
          <td>26.714335</td>
          <td>0.257109</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.180008</td>
          <td>18.698606</td>
          <td>22.941177</td>
          <td>25.803891</td>
          <td>25.079153</td>
          <td>21.174363</td>
          <td>21.200513</td>
          <td>0.005106</td>
          <td>23.776514</td>
          <td>0.019835</td>
          <td>21.504143</td>
          <td>0.005538</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.044366</td>
          <td>22.915437</td>
          <td>17.967426</td>
          <td>22.540905</td>
          <td>24.316717</td>
          <td>24.830198</td>
          <td>16.461128</td>
          <td>0.005000</td>
          <td>24.803120</td>
          <td>0.049016</td>
          <td>23.365021</td>
          <td>0.014090</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.202065</td>
          <td>22.224725</td>
          <td>21.315880</td>
          <td>24.615585</td>
          <td>22.169187</td>
          <td>26.962035</td>
          <td>16.489450</td>
          <td>0.005000</td>
          <td>21.083147</td>
          <td>0.005255</td>
          <td>20.449883</td>
          <td>0.005081</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.177215</td>
          <td>20.698442</td>
          <td>28.144760</td>
          <td>20.571255</td>
          <td>22.594295</td>
          <td>27.601257</td>
          <td>21.726049</td>
          <td>0.005275</td>
          <td>20.293678</td>
          <td>0.005061</td>
          <td>24.923943</td>
          <td>0.054590</td>
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
          <td>25.620319</td>
          <td>22.666545</td>
          <td>27.253885</td>
          <td>29.431615</td>
          <td>22.007556</td>
          <td>24.197069</td>
          <td>21.968030</td>
          <td>0.005423</td>
          <td>20.322615</td>
          <td>0.005064</td>
          <td>26.104003</td>
          <td>0.153957</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.025056</td>
          <td>19.493729</td>
          <td>21.518186</td>
          <td>26.424909</td>
          <td>25.779826</td>
          <td>18.689472</td>
          <td>21.788127</td>
          <td>0.005307</td>
          <td>25.283873</td>
          <td>0.075177</td>
          <td>22.491426</td>
          <td>0.007739</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.040693</td>
          <td>21.350591</td>
          <td>27.786338</td>
          <td>25.012345</td>
          <td>23.379752</td>
          <td>24.002757</td>
          <td>24.156851</td>
          <td>0.016479</td>
          <td>26.868513</td>
          <td>0.291480</td>
          <td>23.702643</td>
          <td>0.018626</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.979771</td>
          <td>23.641341</td>
          <td>26.081023</td>
          <td>19.864231</td>
          <td>21.264162</td>
          <td>22.497700</td>
          <td>20.930196</td>
          <td>0.005065</td>
          <td>20.913476</td>
          <td>0.005188</td>
          <td>20.901379</td>
          <td>0.005184</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.387834</td>
          <td>25.285670</td>
          <td>16.802013</td>
          <td>20.761224</td>
          <td>20.218009</td>
          <td>22.460587</td>
          <td>20.147240</td>
          <td>0.005015</td>
          <td>25.351915</td>
          <td>0.079845</td>
          <td>23.595517</td>
          <td>0.017019</td>
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
          <td>24.285667</td>
          <td>25.590233</td>
          <td>16.617922</td>
          <td>24.562484</td>
          <td>21.465455</td>
          <td>28.544651</td>
          <td>24.141533</td>
          <td>0.173154</td>
          <td>18.938724</td>
          <td>0.005164</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.180008</td>
          <td>18.698606</td>
          <td>22.941177</td>
          <td>25.803891</td>
          <td>25.079153</td>
          <td>21.174363</td>
          <td>21.197878</td>
          <td>0.013358</td>
          <td>23.870823</td>
          <td>0.115390</td>
          <td>21.503671</td>
          <td>0.015770</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.044366</td>
          <td>22.915437</td>
          <td>17.967426</td>
          <td>22.540905</td>
          <td>24.316717</td>
          <td>24.830198</td>
          <td>16.457206</td>
          <td>0.005002</td>
          <td>24.430725</td>
          <td>0.186762</td>
          <td>23.379659</td>
          <td>0.081829</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.202065</td>
          <td>22.224725</td>
          <td>21.315880</td>
          <td>24.615585</td>
          <td>22.169187</td>
          <td>26.962035</td>
          <td>16.491748</td>
          <td>0.005003</td>
          <td>21.067324</td>
          <td>0.010424</td>
          <td>20.447003</td>
          <td>0.007560</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.177215</td>
          <td>20.698442</td>
          <td>28.144760</td>
          <td>20.571255</td>
          <td>22.594295</td>
          <td>27.601257</td>
          <td>21.697059</td>
          <td>0.020188</td>
          <td>20.296274</td>
          <td>0.006729</td>
          <td>24.504413</td>
          <td>0.216111</td>
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
          <td>25.620319</td>
          <td>22.666545</td>
          <td>27.253885</td>
          <td>29.431615</td>
          <td>22.007556</td>
          <td>24.197069</td>
          <td>21.999693</td>
          <td>0.026256</td>
          <td>20.312071</td>
          <td>0.006773</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.025056</td>
          <td>19.493729</td>
          <td>21.518186</td>
          <td>26.424909</td>
          <td>25.779826</td>
          <td>18.689472</td>
          <td>21.756750</td>
          <td>0.021251</td>
          <td>24.451935</td>
          <td>0.190139</td>
          <td>22.498693</td>
          <td>0.037372</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.040693</td>
          <td>21.350591</td>
          <td>27.786338</td>
          <td>25.012345</td>
          <td>23.379752</td>
          <td>24.002757</td>
          <td>24.436329</td>
          <td>0.221938</td>
          <td>26.372035</td>
          <td>0.816672</td>
          <td>23.789772</td>
          <td>0.117313</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.979771</td>
          <td>23.641341</td>
          <td>26.081023</td>
          <td>19.864231</td>
          <td>21.264162</td>
          <td>22.497700</td>
          <td>20.943548</td>
          <td>0.011011</td>
          <td>20.912835</td>
          <td>0.009381</td>
          <td>20.910796</td>
          <td>0.010021</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.387834</td>
          <td>25.285670</td>
          <td>16.802013</td>
          <td>20.761224</td>
          <td>20.218009</td>
          <td>22.460587</td>
          <td>20.141456</td>
          <td>0.006858</td>
          <td>24.976204</td>
          <td>0.293295</td>
          <td>23.490635</td>
          <td>0.090251</td>
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


