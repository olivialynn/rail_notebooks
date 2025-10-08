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
          <td>17.474708</td>
          <td>19.690731</td>
          <td>17.190230</td>
          <td>23.359190</td>
          <td>20.293470</td>
          <td>19.636710</td>
          <td>21.786990</td>
          <td>24.244470</td>
          <td>29.983003</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.253032</td>
          <td>20.089829</td>
          <td>19.794113</td>
          <td>26.445079</td>
          <td>20.874359</td>
          <td>21.613576</td>
          <td>21.700225</td>
          <td>19.782600</td>
          <td>23.094038</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.668988</td>
          <td>24.927721</td>
          <td>22.161515</td>
          <td>18.450531</td>
          <td>25.729953</td>
          <td>17.699815</td>
          <td>24.079618</td>
          <td>21.131417</td>
          <td>21.665389</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.223352</td>
          <td>21.311994</td>
          <td>23.287924</td>
          <td>25.501063</td>
          <td>22.271854</td>
          <td>28.967068</td>
          <td>22.220833</td>
          <td>20.752538</td>
          <td>20.198838</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.826989</td>
          <td>23.467035</td>
          <td>23.098899</td>
          <td>20.724904</td>
          <td>16.786124</td>
          <td>23.798659</td>
          <td>24.946526</td>
          <td>19.704319</td>
          <td>22.932772</td>
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
          <td>20.649553</td>
          <td>23.453922</td>
          <td>21.672076</td>
          <td>25.764339</td>
          <td>19.885743</td>
          <td>21.538517</td>
          <td>21.864111</td>
          <td>21.473661</td>
          <td>24.881177</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.447011</td>
          <td>22.339311</td>
          <td>25.888948</td>
          <td>24.078751</td>
          <td>24.345890</td>
          <td>22.581505</td>
          <td>19.860603</td>
          <td>22.904191</td>
          <td>20.081222</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.323807</td>
          <td>21.382274</td>
          <td>25.490166</td>
          <td>22.343304</td>
          <td>16.309252</td>
          <td>28.630340</td>
          <td>21.121403</td>
          <td>23.694598</td>
          <td>25.694318</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.005744</td>
          <td>23.008908</td>
          <td>25.249924</td>
          <td>23.800483</td>
          <td>20.296720</td>
          <td>19.343442</td>
          <td>23.651572</td>
          <td>23.799870</td>
          <td>25.331995</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.774198</td>
          <td>25.765226</td>
          <td>24.977679</td>
          <td>21.242733</td>
          <td>18.010141</td>
          <td>23.902406</td>
          <td>26.080264</td>
          <td>23.413978</td>
          <td>19.946877</td>
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
          <td>17.474302</td>
          <td>0.005017</td>
          <td>19.690761</td>
          <td>0.005026</td>
          <td>17.184387</td>
          <td>0.005001</td>
          <td>23.371320</td>
          <td>0.013147</td>
          <td>20.296687</td>
          <td>0.005246</td>
          <td>19.634092</td>
          <td>0.005371</td>
          <td>21.786990</td>
          <td>24.244470</td>
          <td>29.983003</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.247685</td>
          <td>0.010808</td>
          <td>20.101357</td>
          <td>0.005043</td>
          <td>19.793909</td>
          <td>0.005018</td>
          <td>26.459154</td>
          <td>0.189634</td>
          <td>20.863694</td>
          <td>0.005610</td>
          <td>21.627136</td>
          <td>0.012270</td>
          <td>21.700225</td>
          <td>19.782600</td>
          <td>23.094038</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.663295</td>
          <td>0.005056</td>
          <td>24.866167</td>
          <td>0.032913</td>
          <td>22.156265</td>
          <td>0.005612</td>
          <td>18.450812</td>
          <td>0.005007</td>
          <td>25.828727</td>
          <td>0.206695</td>
          <td>17.704836</td>
          <td>0.005022</td>
          <td>24.079618</td>
          <td>21.131417</td>
          <td>21.665389</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.208200</td>
          <td>0.010527</td>
          <td>21.320784</td>
          <td>0.005232</td>
          <td>23.286486</td>
          <td>0.008487</td>
          <td>25.602152</td>
          <td>0.090442</td>
          <td>22.276963</td>
          <td>0.010117</td>
          <td>26.530292</td>
          <td>0.715131</td>
          <td>22.220833</td>
          <td>20.752538</td>
          <td>20.198838</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.825220</td>
          <td>0.008365</td>
          <td>23.466950</td>
          <td>0.010550</td>
          <td>23.109065</td>
          <td>0.007699</td>
          <td>20.728807</td>
          <td>0.005151</td>
          <td>16.786211</td>
          <td>0.005003</td>
          <td>23.767585</td>
          <td>0.077018</td>
          <td>24.946526</td>
          <td>19.704319</td>
          <td>22.932772</td>
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
          <td>20.650502</td>
          <td>0.005663</td>
          <td>23.461486</td>
          <td>0.010511</td>
          <td>21.665409</td>
          <td>0.005278</td>
          <td>25.880294</td>
          <td>0.115374</td>
          <td>19.884380</td>
          <td>0.005129</td>
          <td>21.529397</td>
          <td>0.011416</td>
          <td>21.864111</td>
          <td>21.473661</td>
          <td>24.881177</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.972714</td>
          <td>0.537974</td>
          <td>22.343225</td>
          <td>0.006107</td>
          <td>25.852833</td>
          <td>0.069227</td>
          <td>24.096754</td>
          <td>0.023948</td>
          <td>24.380429</td>
          <td>0.058714</td>
          <td>22.559194</td>
          <td>0.026466</td>
          <td>19.860603</td>
          <td>22.904191</td>
          <td>20.081222</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.306570</td>
          <td>0.058962</td>
          <td>21.382357</td>
          <td>0.005254</td>
          <td>25.480313</td>
          <td>0.049745</td>
          <td>22.350461</td>
          <td>0.006986</td>
          <td>16.312472</td>
          <td>0.005002</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.121403</td>
          <td>23.694598</td>
          <td>25.694318</td>
        </tr>
        <tr>
          <th>998</th>
          <td>15.996382</td>
          <td>0.005004</td>
          <td>22.996946</td>
          <td>0.007921</td>
          <td>25.266170</td>
          <td>0.041135</td>
          <td>23.799331</td>
          <td>0.018583</td>
          <td>20.295177</td>
          <td>0.005245</td>
          <td>19.339857</td>
          <td>0.005232</td>
          <td>23.651572</td>
          <td>23.799870</td>
          <td>25.331995</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.776987</td>
          <td>0.005022</td>
          <td>25.744882</td>
          <td>0.071566</td>
          <td>24.986385</td>
          <td>0.032122</td>
          <td>21.243869</td>
          <td>0.005342</td>
          <td>18.009124</td>
          <td>0.005010</td>
          <td>23.813790</td>
          <td>0.080224</td>
          <td>26.080264</td>
          <td>23.413978</td>
          <td>19.946877</td>
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
          <td>17.474708</td>
          <td>19.690731</td>
          <td>17.190230</td>
          <td>23.359190</td>
          <td>20.293470</td>
          <td>19.636710</td>
          <td>21.777304</td>
          <td>0.005302</td>
          <td>24.240823</td>
          <td>0.029731</td>
          <td>27.876645</td>
          <td>0.625613</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.253032</td>
          <td>20.089829</td>
          <td>19.794113</td>
          <td>26.445079</td>
          <td>20.874359</td>
          <td>21.613576</td>
          <td>21.703727</td>
          <td>0.005264</td>
          <td>19.788519</td>
          <td>0.005024</td>
          <td>23.093748</td>
          <td>0.011424</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.668988</td>
          <td>24.927721</td>
          <td>22.161515</td>
          <td>18.450531</td>
          <td>25.729953</td>
          <td>17.699815</td>
          <td>24.091927</td>
          <td>0.015618</td>
          <td>21.137908</td>
          <td>0.005281</td>
          <td>21.663980</td>
          <td>0.005711</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.223352</td>
          <td>21.311994</td>
          <td>23.287924</td>
          <td>25.501063</td>
          <td>22.271854</td>
          <td>28.967068</td>
          <td>22.220625</td>
          <td>0.005659</td>
          <td>20.747949</td>
          <td>0.005139</td>
          <td>20.198222</td>
          <td>0.005051</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.826989</td>
          <td>23.467035</td>
          <td>23.098899</td>
          <td>20.724904</td>
          <td>16.786124</td>
          <td>23.798659</td>
          <td>24.972851</td>
          <td>0.033418</td>
          <td>19.704793</td>
          <td>0.005021</td>
          <td>22.910902</td>
          <td>0.010022</td>
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
          <td>20.649553</td>
          <td>23.453922</td>
          <td>21.672076</td>
          <td>25.764339</td>
          <td>19.885743</td>
          <td>21.538517</td>
          <td>21.857978</td>
          <td>0.005348</td>
          <td>21.471205</td>
          <td>0.005508</td>
          <td>24.843473</td>
          <td>0.050812</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.447011</td>
          <td>22.339311</td>
          <td>25.888948</td>
          <td>24.078751</td>
          <td>24.345890</td>
          <td>22.581505</td>
          <td>19.864682</td>
          <td>0.005009</td>
          <td>22.896134</td>
          <td>0.009920</td>
          <td>20.080351</td>
          <td>0.005041</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.323807</td>
          <td>21.382274</td>
          <td>25.490166</td>
          <td>22.343304</td>
          <td>16.309252</td>
          <td>28.630340</td>
          <td>21.116898</td>
          <td>0.005091</td>
          <td>23.674224</td>
          <td>0.018183</td>
          <td>25.699666</td>
          <td>0.108436</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.005744</td>
          <td>23.008908</td>
          <td>25.249924</td>
          <td>23.800483</td>
          <td>20.296720</td>
          <td>19.343442</td>
          <td>23.654209</td>
          <td>0.011097</td>
          <td>23.819547</td>
          <td>0.020581</td>
          <td>25.334243</td>
          <td>0.078606</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.774198</td>
          <td>25.765226</td>
          <td>24.977679</td>
          <td>21.242733</td>
          <td>18.010141</td>
          <td>23.902406</td>
          <td>26.040633</td>
          <td>0.086357</td>
          <td>23.421839</td>
          <td>0.014750</td>
          <td>19.950465</td>
          <td>0.005032</td>
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
          <td>17.474708</td>
          <td>19.690731</td>
          <td>17.190230</td>
          <td>23.359190</td>
          <td>20.293470</td>
          <td>19.636710</td>
          <td>21.778936</td>
          <td>0.021662</td>
          <td>24.216379</td>
          <td>0.155600</td>
          <td>26.387252</td>
          <td>0.879101</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.253032</td>
          <td>20.089829</td>
          <td>19.794113</td>
          <td>26.445079</td>
          <td>20.874359</td>
          <td>21.613576</td>
          <td>21.724449</td>
          <td>0.020668</td>
          <td>19.783346</td>
          <td>0.005735</td>
          <td>22.984233</td>
          <td>0.057602</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.668988</td>
          <td>24.927721</td>
          <td>22.161515</td>
          <td>18.450531</td>
          <td>25.729953</td>
          <td>17.699815</td>
          <td>24.112697</td>
          <td>0.168954</td>
          <td>21.141211</td>
          <td>0.010992</td>
          <td>21.639943</td>
          <td>0.017665</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.223352</td>
          <td>21.311994</td>
          <td>23.287924</td>
          <td>25.501063</td>
          <td>22.271854</td>
          <td>28.967068</td>
          <td>22.214881</td>
          <td>0.031744</td>
          <td>20.775068</td>
          <td>0.008597</td>
          <td>20.193840</td>
          <td>0.006722</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.826989</td>
          <td>23.467035</td>
          <td>23.098899</td>
          <td>20.724904</td>
          <td>16.786124</td>
          <td>23.798659</td>
          <td>24.745292</td>
          <td>0.286059</td>
          <td>19.711640</td>
          <td>0.005649</td>
          <td>22.867520</td>
          <td>0.051913</td>
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
          <td>20.649553</td>
          <td>23.453922</td>
          <td>21.672076</td>
          <td>25.764339</td>
          <td>19.885743</td>
          <td>21.538517</td>
          <td>21.866774</td>
          <td>0.023376</td>
          <td>21.461136</td>
          <td>0.014046</td>
          <td>24.583624</td>
          <td>0.230835</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.447011</td>
          <td>22.339311</td>
          <td>25.888948</td>
          <td>24.078751</td>
          <td>24.345890</td>
          <td>22.581505</td>
          <td>19.859504</td>
          <td>0.006174</td>
          <td>22.957605</td>
          <td>0.051456</td>
          <td>20.088083</td>
          <td>0.006451</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.323807</td>
          <td>21.382274</td>
          <td>25.490166</td>
          <td>22.343304</td>
          <td>16.309252</td>
          <td>28.630340</td>
          <td>21.110153</td>
          <td>0.012475</td>
          <td>23.673791</td>
          <td>0.097103</td>
          <td>24.713718</td>
          <td>0.256979</td>
        </tr>
        <tr>
          <th>998</th>
          <td>16.005744</td>
          <td>23.008908</td>
          <td>25.249924</td>
          <td>23.800483</td>
          <td>20.296720</td>
          <td>19.343442</td>
          <td>23.768563</td>
          <td>0.125637</td>
          <td>23.840014</td>
          <td>0.112329</td>
          <td>26.092487</td>
          <td>0.725460</td>
        </tr>
        <tr>
          <th>999</th>
          <td>17.774198</td>
          <td>25.765226</td>
          <td>24.977679</td>
          <td>21.242733</td>
          <td>18.010141</td>
          <td>23.902406</td>
          <td>25.772422</td>
          <td>0.623766</td>
          <td>23.375871</td>
          <td>0.074645</td>
          <td>19.945953</td>
          <td>0.006147</td>
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


