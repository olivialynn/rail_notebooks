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
          <td>29.436327</td>
          <td>20.300757</td>
          <td>27.500227</td>
          <td>23.654688</td>
          <td>22.467259</td>
          <td>22.397986</td>
          <td>19.276736</td>
          <td>20.524469</td>
          <td>25.522663</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.486542</td>
          <td>25.276606</td>
          <td>22.837700</td>
          <td>24.942912</td>
          <td>21.496929</td>
          <td>27.081015</td>
          <td>23.459174</td>
          <td>21.157648</td>
          <td>23.597065</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.790792</td>
          <td>22.489945</td>
          <td>22.438620</td>
          <td>20.131620</td>
          <td>26.527579</td>
          <td>22.672964</td>
          <td>23.620492</td>
          <td>24.186523</td>
          <td>24.187354</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.172383</td>
          <td>21.139712</td>
          <td>21.678747</td>
          <td>26.321189</td>
          <td>24.002049</td>
          <td>19.395540</td>
          <td>21.240463</td>
          <td>17.939640</td>
          <td>24.328493</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.327619</td>
          <td>23.187503</td>
          <td>23.733818</td>
          <td>26.434955</td>
          <td>26.232647</td>
          <td>21.855282</td>
          <td>23.047741</td>
          <td>20.100888</td>
          <td>27.906238</td>
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
          <td>22.491472</td>
          <td>18.730275</td>
          <td>24.787650</td>
          <td>22.066145</td>
          <td>28.484628</td>
          <td>24.395109</td>
          <td>20.514676</td>
          <td>22.444734</td>
          <td>29.867617</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.580953</td>
          <td>31.934623</td>
          <td>22.294469</td>
          <td>25.466625</td>
          <td>25.917865</td>
          <td>21.921613</td>
          <td>25.722092</td>
          <td>21.763863</td>
          <td>21.487617</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.419967</td>
          <td>23.420058</td>
          <td>22.397643</td>
          <td>26.488947</td>
          <td>23.178147</td>
          <td>23.556832</td>
          <td>23.789182</td>
          <td>21.086015</td>
          <td>23.539162</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.924201</td>
          <td>22.647176</td>
          <td>19.674060</td>
          <td>26.329439</td>
          <td>21.188798</td>
          <td>25.046782</td>
          <td>28.252654</td>
          <td>20.903310</td>
          <td>20.862488</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.858985</td>
          <td>20.867174</td>
          <td>26.757161</td>
          <td>26.927299</td>
          <td>20.479289</td>
          <td>22.145548</td>
          <td>22.713314</td>
          <td>26.288080</td>
          <td>25.843561</td>
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
          <td>27.700726</td>
          <td>0.882039</td>
          <td>20.301966</td>
          <td>0.005056</td>
          <td>27.393878</td>
          <td>0.260346</td>
          <td>23.672009</td>
          <td>0.016718</td>
          <td>22.484222</td>
          <td>0.011715</td>
          <td>22.329208</td>
          <td>0.021697</td>
          <td>19.276736</td>
          <td>20.524469</td>
          <td>25.522663</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.489218</td>
          <td>0.005006</td>
          <td>25.274206</td>
          <td>0.047182</td>
          <td>22.835210</td>
          <td>0.006791</td>
          <td>24.995087</td>
          <td>0.052855</td>
          <td>21.495174</td>
          <td>0.006655</td>
          <td>27.050395</td>
          <td>0.996983</td>
          <td>23.459174</td>
          <td>21.157648</td>
          <td>23.597065</td>
        </tr>
        <tr>
          <th>2</th>
          <td>28.931532</td>
          <td>1.723803</td>
          <td>22.493950</td>
          <td>0.006393</td>
          <td>22.442211</td>
          <td>0.005968</td>
          <td>20.130825</td>
          <td>0.005061</td>
          <td>26.221227</td>
          <td>0.285611</td>
          <td>22.705642</td>
          <td>0.030081</td>
          <td>23.620492</td>
          <td>24.186523</td>
          <td>24.187354</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.162711</td>
          <td>0.124677</td>
          <td>21.143864</td>
          <td>0.005179</td>
          <td>21.678565</td>
          <td>0.005283</td>
          <td>26.176521</td>
          <td>0.149072</td>
          <td>23.997374</td>
          <td>0.041793</td>
          <td>19.392138</td>
          <td>0.005252</td>
          <td>21.240463</td>
          <td>17.939640</td>
          <td>24.328493</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.439231</td>
          <td>0.158097</td>
          <td>23.212136</td>
          <td>0.008946</td>
          <td>23.731085</td>
          <td>0.011373</td>
          <td>26.467631</td>
          <td>0.190995</td>
          <td>25.786100</td>
          <td>0.199435</td>
          <td>21.863375</td>
          <td>0.014737</td>
          <td>23.047741</td>
          <td>20.100888</td>
          <td>27.906238</td>
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
          <td>22.497033</td>
          <td>0.012886</td>
          <td>18.726688</td>
          <td>0.005009</td>
          <td>24.725513</td>
          <td>0.025564</td>
          <td>22.074081</td>
          <td>0.006296</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.601513</td>
          <td>0.159263</td>
          <td>20.514676</td>
          <td>22.444734</td>
          <td>29.867617</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.575581</td>
          <td>0.005598</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.300504</td>
          <td>0.005772</td>
          <td>25.495708</td>
          <td>0.082349</td>
          <td>26.464492</td>
          <td>0.346865</td>
          <td>21.909089</td>
          <td>0.015288</td>
          <td>25.722092</td>
          <td>21.763863</td>
          <td>21.487617</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.407699</td>
          <td>0.064450</td>
          <td>23.413189</td>
          <td>0.010172</td>
          <td>22.393566</td>
          <td>0.005896</td>
          <td>26.925772</td>
          <td>0.279126</td>
          <td>23.181040</td>
          <td>0.020483</td>
          <td>23.476003</td>
          <td>0.059493</td>
          <td>23.789182</td>
          <td>21.086015</td>
          <td>23.539162</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.707090</td>
          <td>1.549194</td>
          <td>22.659718</td>
          <td>0.006786</td>
          <td>19.690854</td>
          <td>0.005016</td>
          <td>26.426088</td>
          <td>0.184410</td>
          <td>21.187286</td>
          <td>0.006023</td>
          <td>24.790477</td>
          <td>0.187008</td>
          <td>28.252654</td>
          <td>20.903310</td>
          <td>20.862488</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.859689</td>
          <td>0.005887</td>
          <td>20.863565</td>
          <td>0.005120</td>
          <td>26.802087</td>
          <td>0.158558</td>
          <td>26.936279</td>
          <td>0.281515</td>
          <td>20.475562</td>
          <td>0.005327</td>
          <td>22.138767</td>
          <td>0.018466</td>
          <td>22.713314</td>
          <td>26.288080</td>
          <td>25.843561</td>
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
          <td>29.436327</td>
          <td>20.300757</td>
          <td>27.500227</td>
          <td>23.654688</td>
          <td>22.467259</td>
          <td>22.397986</td>
          <td>19.279234</td>
          <td>0.005003</td>
          <td>20.532042</td>
          <td>0.005094</td>
          <td>25.613471</td>
          <td>0.100547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.486542</td>
          <td>25.276606</td>
          <td>22.837700</td>
          <td>24.942912</td>
          <td>21.496929</td>
          <td>27.081015</td>
          <td>23.472419</td>
          <td>0.009761</td>
          <td>21.155846</td>
          <td>0.005290</td>
          <td>23.587863</td>
          <td>0.016910</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.790792</td>
          <td>22.489945</td>
          <td>22.438620</td>
          <td>20.131620</td>
          <td>26.527579</td>
          <td>22.672964</td>
          <td>23.613832</td>
          <td>0.010776</td>
          <td>24.160490</td>
          <td>0.027698</td>
          <td>24.201690</td>
          <td>0.028722</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.172383</td>
          <td>21.139712</td>
          <td>21.678747</td>
          <td>26.321189</td>
          <td>24.002049</td>
          <td>19.395540</td>
          <td>21.249928</td>
          <td>0.005116</td>
          <td>17.936502</td>
          <td>0.005001</td>
          <td>24.366661</td>
          <td>0.033235</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.327619</td>
          <td>23.187503</td>
          <td>23.733818</td>
          <td>26.434955</td>
          <td>26.232647</td>
          <td>21.855282</td>
          <td>23.056379</td>
          <td>0.007597</td>
          <td>20.109450</td>
          <td>0.005043</td>
          <td>27.538397</td>
          <td>0.490229</td>
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
          <td>22.491472</td>
          <td>18.730275</td>
          <td>24.787650</td>
          <td>22.066145</td>
          <td>28.484628</td>
          <td>24.395109</td>
          <td>20.518242</td>
          <td>0.005030</td>
          <td>22.439212</td>
          <td>0.007530</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.580953</td>
          <td>31.934623</td>
          <td>22.294469</td>
          <td>25.466625</td>
          <td>25.917865</td>
          <td>21.921613</td>
          <td>25.755297</td>
          <td>0.067071</td>
          <td>21.763579</td>
          <td>0.005843</td>
          <td>21.509087</td>
          <td>0.005543</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.419967</td>
          <td>23.420058</td>
          <td>22.397643</td>
          <td>26.488947</td>
          <td>23.178147</td>
          <td>23.556832</td>
          <td>23.780055</td>
          <td>0.012191</td>
          <td>21.082284</td>
          <td>0.005254</td>
          <td>23.533131</td>
          <td>0.016158</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.924201</td>
          <td>22.647176</td>
          <td>19.674060</td>
          <td>26.329439</td>
          <td>21.188798</td>
          <td>25.046782</td>
          <td>28.041321</td>
          <td>0.455955</td>
          <td>20.895195</td>
          <td>0.005181</td>
          <td>20.861816</td>
          <td>0.005171</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.858985</td>
          <td>20.867174</td>
          <td>26.757161</td>
          <td>26.927299</td>
          <td>20.479289</td>
          <td>22.145548</td>
          <td>22.714823</td>
          <td>0.006516</td>
          <td>26.321472</td>
          <td>0.185306</td>
          <td>25.820553</td>
          <td>0.120500</td>
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
          <td>29.436327</td>
          <td>20.300757</td>
          <td>27.500227</td>
          <td>23.654688</td>
          <td>22.467259</td>
          <td>22.397986</td>
          <td>19.279711</td>
          <td>0.005432</td>
          <td>20.522883</td>
          <td>0.007467</td>
          <td>25.034417</td>
          <td>0.332876</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.486542</td>
          <td>25.276606</td>
          <td>22.837700</td>
          <td>24.942912</td>
          <td>21.496929</td>
          <td>27.081015</td>
          <td>23.563016</td>
          <td>0.105012</td>
          <td>21.156017</td>
          <td>0.011111</td>
          <td>23.660986</td>
          <td>0.104825</td>
        </tr>
        <tr>
          <th>2</th>
          <td>29.790792</td>
          <td>22.489945</td>
          <td>22.438620</td>
          <td>20.131620</td>
          <td>26.527579</td>
          <td>22.672964</td>
          <td>23.761285</td>
          <td>0.124845</td>
          <td>24.167954</td>
          <td>0.149264</td>
          <td>24.209933</td>
          <td>0.168556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.172383</td>
          <td>21.139712</td>
          <td>21.678747</td>
          <td>26.321189</td>
          <td>24.002049</td>
          <td>19.395540</td>
          <td>21.235373</td>
          <td>0.013760</td>
          <td>17.937504</td>
          <td>0.005026</td>
          <td>24.468061</td>
          <td>0.209642</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.327619</td>
          <td>23.187503</td>
          <td>23.733818</td>
          <td>26.434955</td>
          <td>26.232647</td>
          <td>21.855282</td>
          <td>23.108898</td>
          <td>0.070340</td>
          <td>20.101116</td>
          <td>0.006258</td>
          <td>28.091716</td>
          <td>2.111814</td>
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
          <td>22.491472</td>
          <td>18.730275</td>
          <td>24.787650</td>
          <td>22.066145</td>
          <td>28.484628</td>
          <td>24.395109</td>
          <td>20.495614</td>
          <td>0.008201</td>
          <td>22.430037</td>
          <td>0.032173</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.580953</td>
          <td>31.934623</td>
          <td>22.294469</td>
          <td>25.466625</td>
          <td>25.917865</td>
          <td>21.921613</td>
          <td>25.369149</td>
          <td>0.465578</td>
          <td>21.803182</td>
          <td>0.018634</td>
          <td>21.476061</td>
          <td>0.015416</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.419967</td>
          <td>23.420058</td>
          <td>22.397643</td>
          <td>26.488947</td>
          <td>23.178147</td>
          <td>23.556832</td>
          <td>23.835101</td>
          <td>0.133099</td>
          <td>21.100835</td>
          <td>0.010676</td>
          <td>23.726908</td>
          <td>0.111050</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.924201</td>
          <td>22.647176</td>
          <td>19.674060</td>
          <td>26.329439</td>
          <td>21.188798</td>
          <td>25.046782</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.901964</td>
          <td>0.009314</td>
          <td>20.855195</td>
          <td>0.009649</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.858985</td>
          <td>20.867174</td>
          <td>26.757161</td>
          <td>26.927299</td>
          <td>20.479289</td>
          <td>22.145548</td>
          <td>22.723733</td>
          <td>0.049926</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.447915</td>
          <td>0.913186</td>
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


