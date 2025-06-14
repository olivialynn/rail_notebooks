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
          <td>24.407918</td>
          <td>16.280076</td>
          <td>22.430054</td>
          <td>28.163583</td>
          <td>26.098439</td>
          <td>22.481627</td>
          <td>25.595697</td>
          <td>25.738540</td>
          <td>19.495024</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.618953</td>
          <td>18.830536</td>
          <td>22.399695</td>
          <td>26.437359</td>
          <td>24.921538</td>
          <td>22.463744</td>
          <td>21.035740</td>
          <td>29.358206</td>
          <td>26.034923</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.367499</td>
          <td>30.340965</td>
          <td>21.901191</td>
          <td>17.840097</td>
          <td>20.389559</td>
          <td>21.923725</td>
          <td>24.439527</td>
          <td>20.870003</td>
          <td>24.226609</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.503238</td>
          <td>20.767099</td>
          <td>24.555493</td>
          <td>21.484208</td>
          <td>24.640502</td>
          <td>20.981154</td>
          <td>24.296466</td>
          <td>23.539157</td>
          <td>22.420599</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.147892</td>
          <td>23.544250</td>
          <td>22.204861</td>
          <td>19.011106</td>
          <td>18.819451</td>
          <td>24.787456</td>
          <td>26.019188</td>
          <td>24.249570</td>
          <td>23.493226</td>
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
          <td>23.634906</td>
          <td>19.665473</td>
          <td>25.108371</td>
          <td>21.500785</td>
          <td>19.872153</td>
          <td>22.770295</td>
          <td>23.565112</td>
          <td>26.456462</td>
          <td>23.619302</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.478314</td>
          <td>28.158981</td>
          <td>23.838530</td>
          <td>19.745095</td>
          <td>25.029768</td>
          <td>20.373849</td>
          <td>23.738777</td>
          <td>24.253621</td>
          <td>20.687077</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.901730</td>
          <td>20.012738</td>
          <td>21.301849</td>
          <td>30.152750</td>
          <td>28.539931</td>
          <td>21.524679</td>
          <td>21.903830</td>
          <td>29.270408</td>
          <td>22.399568</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.741210</td>
          <td>19.205031</td>
          <td>23.340676</td>
          <td>22.944825</td>
          <td>22.752095</td>
          <td>27.128723</td>
          <td>18.925073</td>
          <td>20.740880</td>
          <td>28.255307</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.430190</td>
          <td>24.033600</td>
          <td>24.207704</td>
          <td>23.560237</td>
          <td>19.963333</td>
          <td>21.796707</td>
          <td>26.761603</td>
          <td>24.147414</td>
          <td>22.445189</td>
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
          <td>24.399300</td>
          <td>0.063976</td>
          <td>16.284291</td>
          <td>0.005001</td>
          <td>22.435346</td>
          <td>0.005958</td>
          <td>27.423426</td>
          <td>0.413463</td>
          <td>26.122100</td>
          <td>0.263494</td>
          <td>22.453170</td>
          <td>0.024139</td>
          <td>25.595697</td>
          <td>25.738540</td>
          <td>19.495024</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.623022</td>
          <td>0.005638</td>
          <td>18.829825</td>
          <td>0.005010</td>
          <td>22.397504</td>
          <td>0.005902</td>
          <td>26.297009</td>
          <td>0.165263</td>
          <td>24.887397</td>
          <td>0.091903</td>
          <td>22.494894</td>
          <td>0.025027</td>
          <td>21.035740</td>
          <td>29.358206</td>
          <td>26.034923</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.295581</td>
          <td>0.321037</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.907845</td>
          <td>0.005410</td>
          <td>17.843960</td>
          <td>0.005003</td>
          <td>20.397584</td>
          <td>0.005289</td>
          <td>21.920040</td>
          <td>0.015423</td>
          <td>24.439527</td>
          <td>20.870003</td>
          <td>24.226609</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.506099</td>
          <td>0.005017</td>
          <td>20.763985</td>
          <td>0.005104</td>
          <td>24.567542</td>
          <td>0.022302</td>
          <td>21.478630</td>
          <td>0.005500</td>
          <td>24.639335</td>
          <td>0.073850</td>
          <td>20.983892</td>
          <td>0.008043</td>
          <td>24.296466</td>
          <td>23.539157</td>
          <td>22.420599</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.147364</td>
          <td>0.010118</td>
          <td>23.540384</td>
          <td>0.011105</td>
          <td>22.217238</td>
          <td>0.005675</td>
          <td>19.009134</td>
          <td>0.005013</td>
          <td>18.818480</td>
          <td>0.005028</td>
          <td>24.641085</td>
          <td>0.164736</td>
          <td>26.019188</td>
          <td>24.249570</td>
          <td>23.493226</td>
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
          <td>23.668698</td>
          <td>0.033704</td>
          <td>19.672830</td>
          <td>0.005025</td>
          <td>25.131639</td>
          <td>0.036516</td>
          <td>21.497266</td>
          <td>0.005515</td>
          <td>19.872796</td>
          <td>0.005127</td>
          <td>22.780550</td>
          <td>0.032128</td>
          <td>23.565112</td>
          <td>26.456462</td>
          <td>23.619302</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.467772</td>
          <td>0.028327</td>
          <td>28.461310</td>
          <td>0.653947</td>
          <td>23.824273</td>
          <td>0.012182</td>
          <td>19.743014</td>
          <td>0.005035</td>
          <td>25.009745</td>
          <td>0.102314</td>
          <td>20.380320</td>
          <td>0.006222</td>
          <td>23.738777</td>
          <td>24.253621</td>
          <td>20.687077</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.890482</td>
          <td>0.231040</td>
          <td>20.015854</td>
          <td>0.005039</td>
          <td>21.299935</td>
          <td>0.005156</td>
          <td>28.498196</td>
          <td>0.878925</td>
          <td>29.698356</td>
          <td>2.310082</td>
          <td>21.529147</td>
          <td>0.011414</td>
          <td>21.903830</td>
          <td>29.270408</td>
          <td>22.399568</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.528520</td>
          <td>0.789696</td>
          <td>19.204586</td>
          <td>0.005015</td>
          <td>23.326343</td>
          <td>0.008689</td>
          <td>22.936482</td>
          <td>0.009624</td>
          <td>22.760737</td>
          <td>0.014481</td>
          <td>28.122785</td>
          <td>1.756175</td>
          <td>18.925073</td>
          <td>20.740880</td>
          <td>28.255307</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.502755</td>
          <td>0.070070</td>
          <td>24.043881</td>
          <td>0.016325</td>
          <td>24.173094</td>
          <td>0.016017</td>
          <td>23.558464</td>
          <td>0.015241</td>
          <td>19.961268</td>
          <td>0.005145</td>
          <td>21.819126</td>
          <td>0.014228</td>
          <td>26.761603</td>
          <td>24.147414</td>
          <td>22.445189</td>
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
          <td>24.407918</td>
          <td>16.280076</td>
          <td>22.430054</td>
          <td>28.163583</td>
          <td>26.098439</td>
          <td>22.481627</td>
          <td>25.498926</td>
          <td>0.053387</td>
          <td>25.718835</td>
          <td>0.110269</td>
          <td>19.495007</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.618953</td>
          <td>18.830536</td>
          <td>22.399695</td>
          <td>26.437359</td>
          <td>24.921538</td>
          <td>22.463744</td>
          <td>21.028384</td>
          <td>0.005078</td>
          <td>28.187896</td>
          <td>0.773004</td>
          <td>25.926834</td>
          <td>0.132150</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.367499</td>
          <td>30.340965</td>
          <td>21.901191</td>
          <td>17.840097</td>
          <td>20.389559</td>
          <td>21.923725</td>
          <td>24.448202</td>
          <td>0.021095</td>
          <td>20.871141</td>
          <td>0.005174</td>
          <td>24.240243</td>
          <td>0.029716</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.503238</td>
          <td>20.767099</td>
          <td>24.555493</td>
          <td>21.484208</td>
          <td>24.640502</td>
          <td>20.981154</td>
          <td>24.263746</td>
          <td>0.018023</td>
          <td>23.538595</td>
          <td>0.016231</td>
          <td>22.402082</td>
          <td>0.007390</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.147892</td>
          <td>23.544250</td>
          <td>22.204861</td>
          <td>19.011106</td>
          <td>18.819451</td>
          <td>24.787456</td>
          <td>26.130273</td>
          <td>0.093457</td>
          <td>24.231353</td>
          <td>0.029483</td>
          <td>23.514033</td>
          <td>0.015905</td>
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
          <td>23.634906</td>
          <td>19.665473</td>
          <td>25.108371</td>
          <td>21.500785</td>
          <td>19.872153</td>
          <td>22.770295</td>
          <td>23.565343</td>
          <td>0.010410</td>
          <td>27.405437</td>
          <td>0.443784</td>
          <td>23.624821</td>
          <td>0.017442</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.478314</td>
          <td>28.158981</td>
          <td>23.838530</td>
          <td>19.745095</td>
          <td>25.029768</td>
          <td>20.373849</td>
          <td>23.741648</td>
          <td>0.011841</td>
          <td>24.289966</td>
          <td>0.031052</td>
          <td>20.691442</td>
          <td>0.005125</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.901730</td>
          <td>20.012738</td>
          <td>21.301849</td>
          <td>30.152750</td>
          <td>28.539931</td>
          <td>21.524679</td>
          <td>21.898536</td>
          <td>0.005374</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.400509</td>
          <td>0.007384</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.741210</td>
          <td>19.205031</td>
          <td>23.340676</td>
          <td>22.944825</td>
          <td>22.752095</td>
          <td>27.128723</td>
          <td>18.931650</td>
          <td>0.005002</td>
          <td>20.740570</td>
          <td>0.005137</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.430190</td>
          <td>24.033600</td>
          <td>24.207704</td>
          <td>23.560237</td>
          <td>19.963333</td>
          <td>21.796707</td>
          <td>26.912090</td>
          <td>0.183840</td>
          <td>24.124969</td>
          <td>0.026846</td>
          <td>22.441947</td>
          <td>0.007541</td>
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
          <td>24.407918</td>
          <td>16.280076</td>
          <td>22.430054</td>
          <td>28.163583</td>
          <td>26.098439</td>
          <td>22.481627</td>
          <td>27.232019</td>
          <td>1.499000</td>
          <td>24.904390</td>
          <td>0.276723</td>
          <td>19.489050</td>
          <td>0.005524</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.618953</td>
          <td>18.830536</td>
          <td>22.399695</td>
          <td>26.437359</td>
          <td>24.921538</td>
          <td>22.463744</td>
          <td>21.041370</td>
          <td>0.011839</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.235193</td>
          <td>0.389592</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.367499</td>
          <td>30.340965</td>
          <td>21.901191</td>
          <td>17.840097</td>
          <td>20.389559</td>
          <td>21.923725</td>
          <td>24.391435</td>
          <td>0.213781</td>
          <td>20.875627</td>
          <td>0.009156</td>
          <td>24.409576</td>
          <td>0.199603</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.503238</td>
          <td>20.767099</td>
          <td>24.555493</td>
          <td>21.484208</td>
          <td>24.640502</td>
          <td>20.981154</td>
          <td>24.263175</td>
          <td>0.191952</td>
          <td>23.429033</td>
          <td>0.078245</td>
          <td>22.456214</td>
          <td>0.035986</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.147892</td>
          <td>23.544250</td>
          <td>22.204861</td>
          <td>19.011106</td>
          <td>18.819451</td>
          <td>24.787456</td>
          <td>26.077317</td>
          <td>0.767630</td>
          <td>24.497506</td>
          <td>0.197586</td>
          <td>23.581814</td>
          <td>0.097790</td>
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
          <td>23.634906</td>
          <td>19.665473</td>
          <td>25.108371</td>
          <td>21.500785</td>
          <td>19.872153</td>
          <td>22.770295</td>
          <td>23.604979</td>
          <td>0.108941</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.619044</td>
          <td>0.101040</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.478314</td>
          <td>28.158981</td>
          <td>23.838530</td>
          <td>19.745095</td>
          <td>25.029768</td>
          <td>20.373849</td>
          <td>23.940394</td>
          <td>0.145766</td>
          <td>24.296289</td>
          <td>0.166606</td>
          <td>20.690333</td>
          <td>0.008678</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.901730</td>
          <td>20.012738</td>
          <td>21.301849</td>
          <td>30.152750</td>
          <td>28.539931</td>
          <td>21.524679</td>
          <td>21.948795</td>
          <td>0.025111</td>
          <td>26.542965</td>
          <td>0.910374</td>
          <td>22.362694</td>
          <td>0.033118</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.741210</td>
          <td>19.205031</td>
          <td>23.340676</td>
          <td>22.944825</td>
          <td>22.752095</td>
          <td>27.128723</td>
          <td>18.922707</td>
          <td>0.005228</td>
          <td>20.755720</td>
          <td>0.008497</td>
          <td>27.533135</td>
          <td>1.652948</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.430190</td>
          <td>24.033600</td>
          <td>24.207704</td>
          <td>23.560237</td>
          <td>19.963333</td>
          <td>21.796707</td>
          <td>27.373549</td>
          <td>1.606644</td>
          <td>23.790004</td>
          <td>0.107523</td>
          <td>22.443384</td>
          <td>0.035578</td>
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


