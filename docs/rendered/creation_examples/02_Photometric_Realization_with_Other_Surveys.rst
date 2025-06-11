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
          <td>22.845686</td>
          <td>18.213802</td>
          <td>24.929215</td>
          <td>26.195971</td>
          <td>26.266143</td>
          <td>22.577537</td>
          <td>21.961661</td>
          <td>24.623753</td>
          <td>23.435584</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.351262</td>
          <td>28.206291</td>
          <td>22.042529</td>
          <td>22.340806</td>
          <td>22.078797</td>
          <td>19.288634</td>
          <td>25.086212</td>
          <td>20.753420</td>
          <td>25.453813</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.875277</td>
          <td>24.363841</td>
          <td>25.873055</td>
          <td>19.575953</td>
          <td>20.767581</td>
          <td>21.209539</td>
          <td>19.609168</td>
          <td>23.319002</td>
          <td>24.106721</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.572512</td>
          <td>23.923339</td>
          <td>24.121915</td>
          <td>20.920106</td>
          <td>21.863959</td>
          <td>25.773455</td>
          <td>22.439411</td>
          <td>21.657827</td>
          <td>25.311138</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.375784</td>
          <td>20.671934</td>
          <td>20.832960</td>
          <td>21.067252</td>
          <td>25.513980</td>
          <td>21.474170</td>
          <td>28.938957</td>
          <td>21.299733</td>
          <td>23.480127</td>
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
          <td>22.138529</td>
          <td>21.936861</td>
          <td>24.969632</td>
          <td>24.857653</td>
          <td>23.566047</td>
          <td>22.727760</td>
          <td>24.068290</td>
          <td>22.020520</td>
          <td>21.214453</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.032439</td>
          <td>22.713686</td>
          <td>27.682541</td>
          <td>30.399108</td>
          <td>24.228285</td>
          <td>22.447438</td>
          <td>23.060839</td>
          <td>23.221855</td>
          <td>23.030428</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614694</td>
          <td>25.997743</td>
          <td>22.676699</td>
          <td>24.461403</td>
          <td>18.899588</td>
          <td>27.514478</td>
          <td>26.533111</td>
          <td>21.753299</td>
          <td>24.959588</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.523415</td>
          <td>20.727076</td>
          <td>25.946678</td>
          <td>27.242042</td>
          <td>22.967528</td>
          <td>23.732421</td>
          <td>23.628137</td>
          <td>22.228568</td>
          <td>22.309766</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.163313</td>
          <td>20.689572</td>
          <td>25.553451</td>
          <td>23.034494</td>
          <td>18.434426</td>
          <td>21.192973</td>
          <td>20.724230</td>
          <td>23.726782</td>
          <td>21.140572</td>
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
          <td>22.858357</td>
          <td>0.017026</td>
          <td>18.204977</td>
          <td>0.005005</td>
          <td>24.932471</td>
          <td>0.030634</td>
          <td>26.217206</td>
          <td>0.154366</td>
          <td>26.851900</td>
          <td>0.467173</td>
          <td>22.559928</td>
          <td>0.026482</td>
          <td>21.961661</td>
          <td>24.623753</td>
          <td>23.435584</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.362680</td>
          <td>0.025887</td>
          <td>27.418938</td>
          <td>0.298630</td>
          <td>22.040321</td>
          <td>0.005508</td>
          <td>22.342467</td>
          <td>0.006962</td>
          <td>22.087523</td>
          <td>0.008950</td>
          <td>19.284633</td>
          <td>0.005213</td>
          <td>25.086212</td>
          <td>20.753420</td>
          <td>25.453813</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.883607</td>
          <td>0.017375</td>
          <td>24.356160</td>
          <td>0.021167</td>
          <td>25.915277</td>
          <td>0.073159</td>
          <td>19.574774</td>
          <td>0.005028</td>
          <td>20.759871</td>
          <td>0.005516</td>
          <td>21.217682</td>
          <td>0.009232</td>
          <td>19.609168</td>
          <td>23.319002</td>
          <td>24.106721</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.270003</td>
          <td>0.663864</td>
          <td>23.950468</td>
          <td>0.015143</td>
          <td>24.110820</td>
          <td>0.015229</td>
          <td>20.916408</td>
          <td>0.005202</td>
          <td>21.855118</td>
          <td>0.007837</td>
          <td>25.120430</td>
          <td>0.246272</td>
          <td>22.439411</td>
          <td>21.657827</td>
          <td>25.311138</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.651513</td>
          <td>0.189261</td>
          <td>20.667362</td>
          <td>0.005091</td>
          <td>20.836253</td>
          <td>0.005077</td>
          <td>21.070354</td>
          <td>0.005259</td>
          <td>25.641270</td>
          <td>0.176478</td>
          <td>21.473539</td>
          <td>0.010966</td>
          <td>28.938957</td>
          <td>21.299733</td>
          <td>23.480127</td>
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
          <td>22.148628</td>
          <td>0.010126</td>
          <td>21.943257</td>
          <td>0.005598</td>
          <td>25.002915</td>
          <td>0.032593</td>
          <td>24.778572</td>
          <td>0.043612</td>
          <td>23.588097</td>
          <td>0.029125</td>
          <td>22.647063</td>
          <td>0.028576</td>
          <td>24.068290</td>
          <td>22.020520</td>
          <td>21.214453</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.041032</td>
          <td>0.019755</td>
          <td>22.713389</td>
          <td>0.006935</td>
          <td>27.675550</td>
          <td>0.326770</td>
          <td>27.796171</td>
          <td>0.545830</td>
          <td>24.126181</td>
          <td>0.046852</td>
          <td>22.468291</td>
          <td>0.024457</td>
          <td>23.060839</td>
          <td>23.221855</td>
          <td>23.030428</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.611482</td>
          <td>0.005169</td>
          <td>25.888250</td>
          <td>0.081214</td>
          <td>22.673145</td>
          <td>0.006394</td>
          <td>24.486288</td>
          <td>0.033670</td>
          <td>18.902099</td>
          <td>0.005031</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.533111</td>
          <td>21.753299</td>
          <td>24.959588</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.602883</td>
          <td>0.408183</td>
          <td>20.727630</td>
          <td>0.005099</td>
          <td>25.869245</td>
          <td>0.070240</td>
          <td>27.056864</td>
          <td>0.310236</td>
          <td>22.992263</td>
          <td>0.017479</td>
          <td>23.821375</td>
          <td>0.080763</td>
          <td>23.628137</td>
          <td>22.228568</td>
          <td>22.309766</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.153027</td>
          <td>0.021682</td>
          <td>20.697447</td>
          <td>0.005095</td>
          <td>25.690292</td>
          <td>0.059938</td>
          <td>23.043281</td>
          <td>0.010340</td>
          <td>18.433246</td>
          <td>0.005017</td>
          <td>21.204002</td>
          <td>0.009152</td>
          <td>20.724230</td>
          <td>23.726782</td>
          <td>21.140572</td>
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
          <td>22.845686</td>
          <td>18.213802</td>
          <td>24.929215</td>
          <td>26.195971</td>
          <td>26.266143</td>
          <td>22.577537</td>
          <td>21.958374</td>
          <td>0.005416</td>
          <td>24.668192</td>
          <td>0.043461</td>
          <td>23.474418</td>
          <td>0.015396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.351262</td>
          <td>28.206291</td>
          <td>22.042529</td>
          <td>22.340806</td>
          <td>22.078797</td>
          <td>19.288634</td>
          <td>25.133281</td>
          <td>0.038540</td>
          <td>20.745265</td>
          <td>0.005138</td>
          <td>25.504501</td>
          <td>0.091360</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.875277</td>
          <td>24.363841</td>
          <td>25.873055</td>
          <td>19.575953</td>
          <td>20.767581</td>
          <td>21.209539</td>
          <td>19.615025</td>
          <td>0.005006</td>
          <td>23.298544</td>
          <td>0.013365</td>
          <td>24.087246</td>
          <td>0.025971</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.572512</td>
          <td>23.923339</td>
          <td>24.121915</td>
          <td>20.920106</td>
          <td>21.863959</td>
          <td>25.773455</td>
          <td>22.439203</td>
          <td>0.005959</td>
          <td>21.660551</td>
          <td>0.005706</td>
          <td>25.308901</td>
          <td>0.076862</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.375784</td>
          <td>20.671934</td>
          <td>20.832960</td>
          <td>21.067252</td>
          <td>25.513980</td>
          <td>21.474170</td>
          <td>28.168183</td>
          <td>0.501145</td>
          <td>21.305053</td>
          <td>0.005379</td>
          <td>23.471944</td>
          <td>0.015364</td>
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
          <td>22.138529</td>
          <td>21.936861</td>
          <td>24.969632</td>
          <td>24.857653</td>
          <td>23.566047</td>
          <td>22.727760</td>
          <td>24.050071</td>
          <td>0.015092</td>
          <td>22.020289</td>
          <td>0.006298</td>
          <td>21.208115</td>
          <td>0.005319</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.032439</td>
          <td>22.713686</td>
          <td>27.682541</td>
          <td>30.399108</td>
          <td>24.228285</td>
          <td>22.447438</td>
          <td>23.057494</td>
          <td>0.007602</td>
          <td>23.232634</td>
          <td>0.012694</td>
          <td>23.032220</td>
          <td>0.010920</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614694</td>
          <td>25.997743</td>
          <td>22.676699</td>
          <td>24.461403</td>
          <td>18.899588</td>
          <td>27.514478</td>
          <td>26.731212</td>
          <td>0.157590</td>
          <td>21.755634</td>
          <td>0.005832</td>
          <td>25.025587</td>
          <td>0.059763</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.523415</td>
          <td>20.727076</td>
          <td>25.946678</td>
          <td>27.242042</td>
          <td>22.967528</td>
          <td>23.732421</td>
          <td>23.634344</td>
          <td>0.010937</td>
          <td>22.230335</td>
          <td>0.006825</td>
          <td>22.301718</td>
          <td>0.007044</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.163313</td>
          <td>20.689572</td>
          <td>25.553451</td>
          <td>23.034494</td>
          <td>18.434426</td>
          <td>21.192973</td>
          <td>20.736059</td>
          <td>0.005045</td>
          <td>23.717426</td>
          <td>0.018861</td>
          <td>21.139865</td>
          <td>0.005282</td>
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
          <td>22.845686</td>
          <td>18.213802</td>
          <td>24.929215</td>
          <td>26.195971</td>
          <td>26.266143</td>
          <td>22.577537</td>
          <td>22.014164</td>
          <td>0.026592</td>
          <td>24.503730</td>
          <td>0.198624</td>
          <td>23.380895</td>
          <td>0.081918</td>
        </tr>
        <tr>
          <th>1</th>
          <td>23.351262</td>
          <td>28.206291</td>
          <td>22.042529</td>
          <td>22.340806</td>
          <td>22.078797</td>
          <td>19.288634</td>
          <td>24.864329</td>
          <td>0.314807</td>
          <td>20.743782</td>
          <td>0.008437</td>
          <td>25.271531</td>
          <td>0.400676</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.875277</td>
          <td>24.363841</td>
          <td>25.873055</td>
          <td>19.575953</td>
          <td>20.767581</td>
          <td>21.209539</td>
          <td>19.608135</td>
          <td>0.005767</td>
          <td>23.374746</td>
          <td>0.074571</td>
          <td>24.064945</td>
          <td>0.148878</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.572512</td>
          <td>23.923339</td>
          <td>24.121915</td>
          <td>20.920106</td>
          <td>21.863959</td>
          <td>25.773455</td>
          <td>22.364387</td>
          <td>0.036249</td>
          <td>21.631011</td>
          <td>0.016130</td>
          <td>24.889584</td>
          <td>0.296477</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.375784</td>
          <td>20.671934</td>
          <td>20.832960</td>
          <td>21.067252</td>
          <td>25.513980</td>
          <td>21.474170</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.297146</td>
          <td>0.012351</td>
          <td>23.425652</td>
          <td>0.085222</td>
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
          <td>22.138529</td>
          <td>21.936861</td>
          <td>24.969632</td>
          <td>24.857653</td>
          <td>23.566047</td>
          <td>22.727760</td>
          <td>23.845919</td>
          <td>0.134351</td>
          <td>22.019909</td>
          <td>0.022444</td>
          <td>21.209624</td>
          <td>0.012470</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.032439</td>
          <td>22.713686</td>
          <td>27.682541</td>
          <td>30.399108</td>
          <td>24.228285</td>
          <td>22.447438</td>
          <td>22.928469</td>
          <td>0.059916</td>
          <td>23.243379</td>
          <td>0.066364</td>
          <td>23.119575</td>
          <td>0.064975</td>
        </tr>
        <tr>
          <th>997</th>
          <td>19.614694</td>
          <td>25.997743</td>
          <td>22.676699</td>
          <td>24.461403</td>
          <td>18.899588</td>
          <td>27.514478</td>
          <td>26.430036</td>
          <td>0.960651</td>
          <td>21.761188</td>
          <td>0.017984</td>
          <td>24.772041</td>
          <td>0.269531</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.523415</td>
          <td>20.727076</td>
          <td>25.946678</td>
          <td>27.242042</td>
          <td>22.967528</td>
          <td>23.732421</td>
          <td>23.688946</td>
          <td>0.117228</td>
          <td>22.255925</td>
          <td>0.027586</td>
          <td>22.282249</td>
          <td>0.030840</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.163313</td>
          <td>20.689572</td>
          <td>25.553451</td>
          <td>23.034494</td>
          <td>18.434426</td>
          <td>21.192973</td>
          <td>20.730011</td>
          <td>0.009488</td>
          <td>23.669656</td>
          <td>0.096751</td>
          <td>21.143833</td>
          <td>0.011861</td>
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


