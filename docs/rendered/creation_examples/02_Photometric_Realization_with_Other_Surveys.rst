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
          <td>22.968927</td>
          <td>25.099184</td>
          <td>25.011581</td>
          <td>19.573599</td>
          <td>27.315991</td>
          <td>18.682232</td>
          <td>20.129980</td>
          <td>21.640267</td>
          <td>19.960292</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.121039</td>
          <td>23.877781</td>
          <td>25.232722</td>
          <td>18.685755</td>
          <td>21.770592</td>
          <td>27.050688</td>
          <td>23.191900</td>
          <td>17.978102</td>
          <td>20.275233</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.615594</td>
          <td>22.972429</td>
          <td>21.885676</td>
          <td>21.753933</td>
          <td>23.388409</td>
          <td>23.914664</td>
          <td>25.818198</td>
          <td>24.507587</td>
          <td>21.583256</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.829580</td>
          <td>19.715872</td>
          <td>21.213575</td>
          <td>28.855681</td>
          <td>21.554037</td>
          <td>27.058728</td>
          <td>24.132117</td>
          <td>21.768388</td>
          <td>24.114654</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.972496</td>
          <td>19.811562</td>
          <td>30.019957</td>
          <td>19.413130</td>
          <td>31.201912</td>
          <td>18.360377</td>
          <td>24.208381</td>
          <td>24.584106</td>
          <td>25.856226</td>
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
          <td>23.936797</td>
          <td>22.679838</td>
          <td>26.051938</td>
          <td>22.517122</td>
          <td>26.163497</td>
          <td>22.753351</td>
          <td>25.216340</td>
          <td>21.193085</td>
          <td>22.597383</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.008710</td>
          <td>24.807481</td>
          <td>27.438051</td>
          <td>23.737093</td>
          <td>15.657242</td>
          <td>24.293869</td>
          <td>22.236064</td>
          <td>23.243459</td>
          <td>19.595266</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.753428</td>
          <td>22.683766</td>
          <td>18.380677</td>
          <td>21.221877</td>
          <td>32.224826</td>
          <td>23.138751</td>
          <td>17.625578</td>
          <td>20.224673</td>
          <td>26.521122</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.413495</td>
          <td>20.778907</td>
          <td>27.129064</td>
          <td>27.345989</td>
          <td>23.174688</td>
          <td>20.119061</td>
          <td>22.370140</td>
          <td>23.964708</td>
          <td>27.001541</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.699242</td>
          <td>24.007420</td>
          <td>26.578403</td>
          <td>21.326689</td>
          <td>23.337143</td>
          <td>17.842019</td>
          <td>25.653801</td>
          <td>24.716591</td>
          <td>21.744612</td>
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
          <td>22.966916</td>
          <td>0.018589</td>
          <td>25.097983</td>
          <td>0.040371</td>
          <td>25.055444</td>
          <td>0.034138</td>
          <td>19.578992</td>
          <td>0.005028</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.681933</td>
          <td>0.005084</td>
          <td>20.129980</td>
          <td>21.640267</td>
          <td>19.960292</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.120484</td>
          <td>0.050057</td>
          <td>23.891097</td>
          <td>0.014447</td>
          <td>25.219671</td>
          <td>0.039475</td>
          <td>18.685764</td>
          <td>0.005009</td>
          <td>21.770403</td>
          <td>0.007506</td>
          <td>26.400513</td>
          <td>0.654466</td>
          <td>23.191900</td>
          <td>17.978102</td>
          <td>20.275233</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.756077</td>
          <td>0.087513</td>
          <td>22.978201</td>
          <td>0.007844</td>
          <td>21.879170</td>
          <td>0.005391</td>
          <td>21.751467</td>
          <td>0.005776</td>
          <td>23.381717</td>
          <td>0.024332</td>
          <td>23.859159</td>
          <td>0.083499</td>
          <td>25.818198</td>
          <td>24.507587</td>
          <td>21.583256</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.666718</td>
          <td>0.080922</td>
          <td>19.720167</td>
          <td>0.005027</td>
          <td>21.223216</td>
          <td>0.005138</td>
          <td>28.751701</td>
          <td>1.026811</td>
          <td>21.560660</td>
          <td>0.006829</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.132117</td>
          <td>21.768388</td>
          <td>24.114654</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.999143</td>
          <td>0.019086</td>
          <td>19.807409</td>
          <td>0.005030</td>
          <td>29.154069</td>
          <td>0.936854</td>
          <td>19.409183</td>
          <td>0.005022</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.364359</td>
          <td>0.005053</td>
          <td>24.208381</td>
          <td>24.584106</td>
          <td>25.856226</td>
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
          <td>23.940800</td>
          <td>0.042749</td>
          <td>22.699690</td>
          <td>0.006896</td>
          <td>26.051145</td>
          <td>0.082486</td>
          <td>22.527873</td>
          <td>0.007589</td>
          <td>26.252709</td>
          <td>0.292968</td>
          <td>22.775219</td>
          <td>0.031978</td>
          <td>25.216340</td>
          <td>21.193085</td>
          <td>22.597383</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.056965</td>
          <td>0.020017</td>
          <td>24.847778</td>
          <td>0.032387</td>
          <td>27.299281</td>
          <td>0.240876</td>
          <td>23.719694</td>
          <td>0.017389</td>
          <td>15.655581</td>
          <td>0.005001</td>
          <td>24.343576</td>
          <td>0.127550</td>
          <td>22.236064</td>
          <td>23.243459</td>
          <td>19.595266</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.752005</td>
          <td>0.005202</td>
          <td>22.675473</td>
          <td>0.006829</td>
          <td>18.379386</td>
          <td>0.005003</td>
          <td>21.213974</td>
          <td>0.005326</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.173469</td>
          <td>0.045483</td>
          <td>17.625578</td>
          <td>20.224673</td>
          <td>26.521122</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.418200</td>
          <td>0.005006</td>
          <td>20.777880</td>
          <td>0.005106</td>
          <td>26.879180</td>
          <td>0.169338</td>
          <td>27.335432</td>
          <td>0.386373</td>
          <td>23.158458</td>
          <td>0.020094</td>
          <td>20.116197</td>
          <td>0.005804</td>
          <td>22.370140</td>
          <td>23.964708</td>
          <td>27.001541</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.711060</td>
          <td>0.084130</td>
          <td>23.985333</td>
          <td>0.015571</td>
          <td>26.515286</td>
          <td>0.123837</td>
          <td>21.331710</td>
          <td>0.005394</td>
          <td>23.321550</td>
          <td>0.023100</td>
          <td>17.838997</td>
          <td>0.005026</td>
          <td>25.653801</td>
          <td>24.716591</td>
          <td>21.744612</td>
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
          <td>22.968927</td>
          <td>25.099184</td>
          <td>25.011581</td>
          <td>19.573599</td>
          <td>27.315991</td>
          <td>18.682232</td>
          <td>20.137077</td>
          <td>0.005015</td>
          <td>21.635674</td>
          <td>0.005677</td>
          <td>19.968257</td>
          <td>0.005033</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.121039</td>
          <td>23.877781</td>
          <td>25.232722</td>
          <td>18.685755</td>
          <td>21.770592</td>
          <td>27.050688</td>
          <td>23.191310</td>
          <td>0.008181</td>
          <td>17.980269</td>
          <td>0.005001</td>
          <td>20.274389</td>
          <td>0.005059</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.615594</td>
          <td>22.972429</td>
          <td>21.885676</td>
          <td>21.753933</td>
          <td>23.388409</td>
          <td>23.914664</td>
          <td>25.752993</td>
          <td>0.066933</td>
          <td>24.482723</td>
          <td>0.036844</td>
          <td>21.588089</td>
          <td>0.005623</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.829580</td>
          <td>19.715872</td>
          <td>21.213575</td>
          <td>28.855681</td>
          <td>21.554037</td>
          <td>27.058728</td>
          <td>24.135240</td>
          <td>0.016186</td>
          <td>21.774068</td>
          <td>0.005858</td>
          <td>24.093334</td>
          <td>0.026110</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.972496</td>
          <td>19.811562</td>
          <td>30.019957</td>
          <td>19.413130</td>
          <td>31.201912</td>
          <td>18.360377</td>
          <td>24.175235</td>
          <td>0.016733</td>
          <td>24.579548</td>
          <td>0.040161</td>
          <td>25.737023</td>
          <td>0.112036</td>
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
          <td>23.936797</td>
          <td>22.679838</td>
          <td>26.051938</td>
          <td>22.517122</td>
          <td>26.163497</td>
          <td>22.753351</td>
          <td>25.210649</td>
          <td>0.041289</td>
          <td>21.193241</td>
          <td>0.005310</td>
          <td>22.618649</td>
          <td>0.008312</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.008710</td>
          <td>24.807481</td>
          <td>27.438051</td>
          <td>23.737093</td>
          <td>15.657242</td>
          <td>24.293869</td>
          <td>22.220202</td>
          <td>0.005659</td>
          <td>23.236244</td>
          <td>0.012729</td>
          <td>19.607653</td>
          <td>0.005017</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.753428</td>
          <td>22.683766</td>
          <td>18.380677</td>
          <td>21.221877</td>
          <td>32.224826</td>
          <td>23.138751</td>
          <td>17.618709</td>
          <td>0.005000</td>
          <td>20.227587</td>
          <td>0.005054</td>
          <td>26.830791</td>
          <td>0.282718</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.413495</td>
          <td>20.778907</td>
          <td>27.129064</td>
          <td>27.345989</td>
          <td>23.174688</td>
          <td>20.119061</td>
          <td>22.384380</td>
          <td>0.005874</td>
          <td>23.959898</td>
          <td>0.023237</td>
          <td>27.292522</td>
          <td>0.407196</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.699242</td>
          <td>24.007420</td>
          <td>26.578403</td>
          <td>21.326689</td>
          <td>23.337143</td>
          <td>17.842019</td>
          <td>25.750249</td>
          <td>0.066770</td>
          <td>24.746595</td>
          <td>0.046607</td>
          <td>21.748015</td>
          <td>0.005821</td>
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
          <td>22.968927</td>
          <td>25.099184</td>
          <td>25.011581</td>
          <td>19.573599</td>
          <td>27.315991</td>
          <td>18.682232</td>
          <td>20.130884</td>
          <td>0.006827</td>
          <td>21.641677</td>
          <td>0.016273</td>
          <td>19.954393</td>
          <td>0.006164</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.121039</td>
          <td>23.877781</td>
          <td>25.232722</td>
          <td>18.685755</td>
          <td>21.770592</td>
          <td>27.050688</td>
          <td>23.295139</td>
          <td>0.082956</td>
          <td>17.975816</td>
          <td>0.005028</td>
          <td>20.276870</td>
          <td>0.006965</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.615594</td>
          <td>22.972429</td>
          <td>21.885676</td>
          <td>21.753933</td>
          <td>23.388409</td>
          <td>23.914664</td>
          <td>27.674674</td>
          <td>1.846178</td>
          <td>24.517514</td>
          <td>0.200939</td>
          <td>21.584450</td>
          <td>0.016862</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.829580</td>
          <td>19.715872</td>
          <td>21.213575</td>
          <td>28.855681</td>
          <td>21.554037</td>
          <td>27.058728</td>
          <td>24.372800</td>
          <td>0.210475</td>
          <td>21.777771</td>
          <td>0.018237</td>
          <td>24.125584</td>
          <td>0.156832</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.972496</td>
          <td>19.811562</td>
          <td>30.019957</td>
          <td>19.413130</td>
          <td>31.201912</td>
          <td>18.360377</td>
          <td>24.103513</td>
          <td>0.167636</td>
          <td>24.779065</td>
          <td>0.249769</td>
          <td>25.712294</td>
          <td>0.556674</td>
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
          <td>23.936797</td>
          <td>22.679838</td>
          <td>26.051938</td>
          <td>22.517122</td>
          <td>26.163497</td>
          <td>22.753351</td>
          <td>24.793937</td>
          <td>0.297519</td>
          <td>21.204310</td>
          <td>0.011514</td>
          <td>22.508774</td>
          <td>0.037708</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.008710</td>
          <td>24.807481</td>
          <td>27.438051</td>
          <td>23.737093</td>
          <td>15.657242</td>
          <td>24.293869</td>
          <td>22.239024</td>
          <td>0.032430</td>
          <td>23.273819</td>
          <td>0.068183</td>
          <td>19.600536</td>
          <td>0.005637</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.753428</td>
          <td>22.683766</td>
          <td>18.380677</td>
          <td>21.221877</td>
          <td>32.224826</td>
          <td>23.138751</td>
          <td>17.617653</td>
          <td>0.005021</td>
          <td>20.230622</td>
          <td>0.006555</td>
          <td>25.741191</td>
          <td>0.568357</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.413495</td>
          <td>20.778907</td>
          <td>27.129064</td>
          <td>27.345989</td>
          <td>23.174688</td>
          <td>20.119061</td>
          <td>22.426611</td>
          <td>0.038312</td>
          <td>23.839152</td>
          <td>0.112244</td>
          <td>25.816786</td>
          <td>0.599802</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.699242</td>
          <td>24.007420</td>
          <td>26.578403</td>
          <td>21.326689</td>
          <td>23.337143</td>
          <td>17.842019</td>
          <td>27.625343</td>
          <td>1.806025</td>
          <td>24.548331</td>
          <td>0.206205</td>
          <td>21.739282</td>
          <td>0.019215</td>
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


