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
          <td>25.047768</td>
          <td>21.998712</td>
          <td>19.554718</td>
          <td>22.073269</td>
          <td>22.143255</td>
          <td>24.286328</td>
          <td>22.708508</td>
          <td>19.592998</td>
          <td>23.769978</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.841666</td>
          <td>24.997580</td>
          <td>22.199866</td>
          <td>27.111513</td>
          <td>28.236061</td>
          <td>18.375507</td>
          <td>22.860904</td>
          <td>17.113473</td>
          <td>27.219030</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.307842</td>
          <td>24.751349</td>
          <td>25.602181</td>
          <td>20.630784</td>
          <td>23.168609</td>
          <td>23.544503</td>
          <td>18.901707</td>
          <td>28.271896</td>
          <td>24.755397</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.151452</td>
          <td>25.764884</td>
          <td>25.394715</td>
          <td>25.646835</td>
          <td>22.547081</td>
          <td>26.289553</td>
          <td>21.971469</td>
          <td>25.054221</td>
          <td>29.243855</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.570347</td>
          <td>19.719560</td>
          <td>24.903202</td>
          <td>17.951669</td>
          <td>22.323589</td>
          <td>24.364119</td>
          <td>19.770980</td>
          <td>21.160803</td>
          <td>20.470218</td>
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
          <td>30.728205</td>
          <td>26.934073</td>
          <td>21.638948</td>
          <td>25.430113</td>
          <td>23.694059</td>
          <td>23.837814</td>
          <td>22.176849</td>
          <td>28.649575</td>
          <td>24.143754</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.500131</td>
          <td>18.454987</td>
          <td>20.571832</td>
          <td>21.661687</td>
          <td>23.973252</td>
          <td>26.640472</td>
          <td>26.016277</td>
          <td>24.403037</td>
          <td>21.254212</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.249896</td>
          <td>27.778883</td>
          <td>19.389278</td>
          <td>27.656137</td>
          <td>23.910235</td>
          <td>24.119050</td>
          <td>25.733498</td>
          <td>23.483671</td>
          <td>22.187454</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.505640</td>
          <td>24.774725</td>
          <td>23.536702</td>
          <td>19.320363</td>
          <td>20.122488</td>
          <td>24.422072</td>
          <td>26.255174</td>
          <td>26.444794</td>
          <td>24.062768</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.408969</td>
          <td>17.478403</td>
          <td>21.932382</td>
          <td>25.280930</td>
          <td>22.503606</td>
          <td>23.547314</td>
          <td>21.933654</td>
          <td>21.596932</td>
          <td>27.476664</td>
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
          <td>25.090106</td>
          <td>0.117083</td>
          <td>21.990249</td>
          <td>0.005643</td>
          <td>19.550748</td>
          <td>0.005013</td>
          <td>22.079810</td>
          <td>0.006308</td>
          <td>22.140483</td>
          <td>0.009251</td>
          <td>24.163445</td>
          <td>0.109052</td>
          <td>22.708508</td>
          <td>19.592998</td>
          <td>23.769978</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.841502</td>
          <td>0.005226</td>
          <td>25.063681</td>
          <td>0.039167</td>
          <td>22.203733</td>
          <td>0.005661</td>
          <td>27.092283</td>
          <td>0.319141</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.376596</td>
          <td>0.005054</td>
          <td>22.860904</td>
          <td>17.113473</td>
          <td>27.219030</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.389217</td>
          <td>0.151487</td>
          <td>24.719395</td>
          <td>0.028946</td>
          <td>25.610966</td>
          <td>0.055864</td>
          <td>20.633980</td>
          <td>0.005130</td>
          <td>23.150823</td>
          <td>0.019965</td>
          <td>23.383197</td>
          <td>0.054790</td>
          <td>18.901707</td>
          <td>28.271896</td>
          <td>24.755397</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.078104</td>
          <td>0.048226</td>
          <td>25.771821</td>
          <td>0.073289</td>
          <td>25.396480</td>
          <td>0.046177</td>
          <td>25.716291</td>
          <td>0.099973</td>
          <td>22.557322</td>
          <td>0.012370</td>
          <td>26.075636</td>
          <td>0.519316</td>
          <td>21.971469</td>
          <td>25.054221</td>
          <td>29.243855</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.570223</td>
          <td>0.030944</td>
          <td>19.714671</td>
          <td>0.005027</td>
          <td>24.877110</td>
          <td>0.029182</td>
          <td>17.945230</td>
          <td>0.005004</td>
          <td>22.335509</td>
          <td>0.010532</td>
          <td>24.086103</td>
          <td>0.101923</td>
          <td>19.770980</td>
          <td>21.160803</td>
          <td>20.470218</td>
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
          <td>27.670270</td>
          <td>0.865213</td>
          <td>26.709231</td>
          <td>0.165690</td>
          <td>21.647963</td>
          <td>0.005270</td>
          <td>25.444053</td>
          <td>0.078680</td>
          <td>23.692257</td>
          <td>0.031914</td>
          <td>23.869203</td>
          <td>0.084241</td>
          <td>22.176849</td>
          <td>28.649575</td>
          <td>24.143754</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.513275</td>
          <td>0.007205</td>
          <td>18.451006</td>
          <td>0.005007</td>
          <td>20.579659</td>
          <td>0.005053</td>
          <td>21.654695</td>
          <td>0.005664</td>
          <td>24.035102</td>
          <td>0.043215</td>
          <td>26.314072</td>
          <td>0.616194</td>
          <td>26.016277</td>
          <td>24.403037</td>
          <td>21.254212</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.256605</td>
          <td>0.006545</td>
          <td>27.649945</td>
          <td>0.358763</td>
          <td>19.384924</td>
          <td>0.005011</td>
          <td>27.357825</td>
          <td>0.393123</td>
          <td>23.917926</td>
          <td>0.038952</td>
          <td>23.979873</td>
          <td>0.092856</td>
          <td>25.733498</td>
          <td>23.483671</td>
          <td>22.187454</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.441542</td>
          <td>0.066398</td>
          <td>24.771469</td>
          <td>0.030292</td>
          <td>23.530848</td>
          <td>0.009890</td>
          <td>19.317532</td>
          <td>0.005020</td>
          <td>20.127250</td>
          <td>0.005188</td>
          <td>24.402877</td>
          <td>0.134265</td>
          <td>26.255174</td>
          <td>26.444794</td>
          <td>24.062768</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.586979</td>
          <td>0.403232</td>
          <td>17.467836</td>
          <td>0.005003</td>
          <td>21.934237</td>
          <td>0.005428</td>
          <td>25.231542</td>
          <td>0.065194</td>
          <td>22.505646</td>
          <td>0.011901</td>
          <td>23.439948</td>
          <td>0.057620</td>
          <td>21.933654</td>
          <td>21.596932</td>
          <td>27.476664</td>
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
          <td>25.047768</td>
          <td>21.998712</td>
          <td>19.554718</td>
          <td>22.073269</td>
          <td>22.143255</td>
          <td>24.286328</td>
          <td>22.704146</td>
          <td>0.006490</td>
          <td>19.587586</td>
          <td>0.005017</td>
          <td>23.768120</td>
          <td>0.019694</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.841666</td>
          <td>24.997580</td>
          <td>22.199866</td>
          <td>27.111513</td>
          <td>28.236061</td>
          <td>18.375507</td>
          <td>22.854918</td>
          <td>0.006898</td>
          <td>17.106958</td>
          <td>0.005000</td>
          <td>27.396547</td>
          <td>0.440810</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.307842</td>
          <td>24.751349</td>
          <td>25.602181</td>
          <td>20.630784</td>
          <td>23.168609</td>
          <td>23.544503</td>
          <td>18.903274</td>
          <td>0.005002</td>
          <td>32.612516</td>
          <td>4.482717</td>
          <td>24.754127</td>
          <td>0.046922</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.151452</td>
          <td>25.764884</td>
          <td>25.394715</td>
          <td>25.646835</td>
          <td>22.547081</td>
          <td>26.289553</td>
          <td>21.970959</td>
          <td>0.005426</td>
          <td>25.019388</td>
          <td>0.059434</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.570347</td>
          <td>19.719560</td>
          <td>24.903202</td>
          <td>17.951669</td>
          <td>22.323589</td>
          <td>24.364119</td>
          <td>19.771277</td>
          <td>0.005008</td>
          <td>21.155042</td>
          <td>0.005290</td>
          <td>20.465812</td>
          <td>0.005083</td>
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
          <td>30.728205</td>
          <td>26.934073</td>
          <td>21.638948</td>
          <td>25.430113</td>
          <td>23.694059</td>
          <td>23.837814</td>
          <td>22.171165</td>
          <td>0.005605</td>
          <td>28.709294</td>
          <td>1.069459</td>
          <td>24.119438</td>
          <td>0.026715</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.500131</td>
          <td>18.454987</td>
          <td>20.571832</td>
          <td>21.661687</td>
          <td>23.973252</td>
          <td>26.640472</td>
          <td>25.869495</td>
          <td>0.074225</td>
          <td>24.457360</td>
          <td>0.036023</td>
          <td>21.264286</td>
          <td>0.005352</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.249896</td>
          <td>27.778883</td>
          <td>19.389278</td>
          <td>27.656137</td>
          <td>23.910235</td>
          <td>24.119050</td>
          <td>25.759493</td>
          <td>0.067321</td>
          <td>23.471096</td>
          <td>0.015354</td>
          <td>22.187650</td>
          <td>0.006705</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.505640</td>
          <td>24.774725</td>
          <td>23.536702</td>
          <td>19.320363</td>
          <td>20.122488</td>
          <td>24.422072</td>
          <td>26.476987</td>
          <td>0.126559</td>
          <td>26.527021</td>
          <td>0.220224</td>
          <td>24.089563</td>
          <td>0.026024</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.408969</td>
          <td>17.478403</td>
          <td>21.932382</td>
          <td>25.280930</td>
          <td>22.503606</td>
          <td>23.547314</td>
          <td>21.934818</td>
          <td>0.005399</td>
          <td>21.598063</td>
          <td>0.005634</td>
          <td>27.035043</td>
          <td>0.333041</td>
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
          <td>25.047768</td>
          <td>21.998712</td>
          <td>19.554718</td>
          <td>22.073269</td>
          <td>22.143255</td>
          <td>24.286328</td>
          <td>22.720250</td>
          <td>0.049771</td>
          <td>19.595134</td>
          <td>0.005530</td>
          <td>23.638354</td>
          <td>0.102766</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.841666</td>
          <td>24.997580</td>
          <td>22.199866</td>
          <td>27.111513</td>
          <td>28.236061</td>
          <td>18.375507</td>
          <td>22.951542</td>
          <td>0.061160</td>
          <td>17.108736</td>
          <td>0.005006</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.307842</td>
          <td>24.751349</td>
          <td>25.602181</td>
          <td>20.630784</td>
          <td>23.168609</td>
          <td>23.544503</td>
          <td>18.907002</td>
          <td>0.005222</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.657136</td>
          <td>0.245300</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.151452</td>
          <td>25.764884</td>
          <td>25.394715</td>
          <td>25.646835</td>
          <td>22.547081</td>
          <td>26.289553</td>
          <td>21.994447</td>
          <td>0.026136</td>
          <td>25.529911</td>
          <td>0.452056</td>
          <td>26.000489</td>
          <td>0.681598</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.570347</td>
          <td>19.719560</td>
          <td>24.903202</td>
          <td>17.951669</td>
          <td>22.323589</td>
          <td>24.364119</td>
          <td>19.766983</td>
          <td>0.006005</td>
          <td>21.174543</td>
          <td>0.011263</td>
          <td>20.460531</td>
          <td>0.007614</td>
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
          <td>30.728205</td>
          <td>26.934073</td>
          <td>21.638948</td>
          <td>25.430113</td>
          <td>23.694059</td>
          <td>23.837814</td>
          <td>22.182349</td>
          <td>0.030843</td>
          <td>26.743120</td>
          <td>1.028472</td>
          <td>24.145752</td>
          <td>0.159565</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.500131</td>
          <td>18.454987</td>
          <td>20.571832</td>
          <td>21.661687</td>
          <td>23.973252</td>
          <td>26.640472</td>
          <td>26.907151</td>
          <td>1.265401</td>
          <td>24.574813</td>
          <td>0.210830</td>
          <td>21.269802</td>
          <td>0.013067</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.249896</td>
          <td>27.778883</td>
          <td>19.389278</td>
          <td>27.656137</td>
          <td>23.910235</td>
          <td>24.119050</td>
          <td>24.921013</td>
          <td>0.329353</td>
          <td>23.404630</td>
          <td>0.076572</td>
          <td>22.180814</td>
          <td>0.028198</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.505640</td>
          <td>24.774725</td>
          <td>23.536702</td>
          <td>19.320363</td>
          <td>20.122488</td>
          <td>24.422072</td>
          <td>25.648824</td>
          <td>0.571474</td>
          <td>26.135788</td>
          <td>0.698198</td>
          <td>23.939151</td>
          <td>0.133567</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.408969</td>
          <td>17.478403</td>
          <td>21.932382</td>
          <td>25.280930</td>
          <td>22.503606</td>
          <td>23.547314</td>
          <td>21.944021</td>
          <td>0.025006</td>
          <td>21.571969</td>
          <td>0.015365</td>
          <td>26.538223</td>
          <td>0.965466</td>
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


