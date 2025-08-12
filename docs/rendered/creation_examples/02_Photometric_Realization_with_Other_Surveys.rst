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
          <td>21.903785</td>
          <td>21.760425</td>
          <td>24.309006</td>
          <td>26.795652</td>
          <td>24.122152</td>
          <td>19.726739</td>
          <td>20.570258</td>
          <td>25.274745</td>
          <td>22.893759</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.249192</td>
          <td>19.859864</td>
          <td>23.644446</td>
          <td>18.330824</td>
          <td>24.432829</td>
          <td>19.701743</td>
          <td>27.233267</td>
          <td>17.789849</td>
          <td>18.865972</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.050884</td>
          <td>28.741263</td>
          <td>23.709107</td>
          <td>17.494134</td>
          <td>26.484200</td>
          <td>23.424059</td>
          <td>27.722720</td>
          <td>25.348502</td>
          <td>16.486305</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.788579</td>
          <td>19.513627</td>
          <td>22.313876</td>
          <td>22.052501</td>
          <td>25.371641</td>
          <td>22.880575</td>
          <td>19.540759</td>
          <td>19.087545</td>
          <td>28.700685</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.337263</td>
          <td>17.751578</td>
          <td>21.596767</td>
          <td>23.100271</td>
          <td>25.356445</td>
          <td>24.416644</td>
          <td>24.709485</td>
          <td>24.792617</td>
          <td>19.992837</td>
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
          <td>24.507660</td>
          <td>23.806922</td>
          <td>21.980896</td>
          <td>23.640281</td>
          <td>17.713438</td>
          <td>21.866113</td>
          <td>22.307822</td>
          <td>19.851597</td>
          <td>18.358839</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.118436</td>
          <td>22.960039</td>
          <td>19.614389</td>
          <td>23.575054</td>
          <td>24.263408</td>
          <td>22.792409</td>
          <td>19.633041</td>
          <td>19.809357</td>
          <td>20.429043</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.545218</td>
          <td>20.885719</td>
          <td>18.423965</td>
          <td>21.932139</td>
          <td>20.825525</td>
          <td>19.887936</td>
          <td>24.681912</td>
          <td>19.873742</td>
          <td>24.100921</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.541432</td>
          <td>23.713737</td>
          <td>22.247661</td>
          <td>24.997789</td>
          <td>24.121224</td>
          <td>22.323170</td>
          <td>28.690592</td>
          <td>23.757155</td>
          <td>24.884154</td>
        </tr>
        <tr>
          <th>999</th>
          <td>14.900699</td>
          <td>23.306596</td>
          <td>26.684344</td>
          <td>22.048573</td>
          <td>26.911642</td>
          <td>22.922926</td>
          <td>28.193311</td>
          <td>21.254340</td>
          <td>19.903971</td>
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
          <td>21.905815</td>
          <td>0.008743</td>
          <td>21.757237</td>
          <td>0.005449</td>
          <td>24.296217</td>
          <td>0.017727</td>
          <td>26.889406</td>
          <td>0.270996</td>
          <td>24.080283</td>
          <td>0.044982</td>
          <td>19.733500</td>
          <td>0.005435</td>
          <td>20.570258</td>
          <td>25.274745</td>
          <td>22.893759</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.171711</td>
          <td>0.052364</td>
          <td>19.858549</td>
          <td>0.005032</td>
          <td>23.648029</td>
          <td>0.010718</td>
          <td>18.329359</td>
          <td>0.005006</td>
          <td>24.405891</td>
          <td>0.060056</td>
          <td>19.698197</td>
          <td>0.005411</td>
          <td>27.233267</td>
          <td>17.789849</td>
          <td>18.865972</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.057147</td>
          <td>0.005298</td>
          <td>28.510363</td>
          <td>0.676407</td>
          <td>23.692276</td>
          <td>0.011059</td>
          <td>17.488069</td>
          <td>0.005002</td>
          <td>26.599517</td>
          <td>0.385464</td>
          <td>23.463217</td>
          <td>0.058822</td>
          <td>27.722720</td>
          <td>25.348502</td>
          <td>16.486305</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.711362</td>
          <td>0.084152</td>
          <td>19.513706</td>
          <td>0.005021</td>
          <td>22.306285</td>
          <td>0.005779</td>
          <td>22.050560</td>
          <td>0.006249</td>
          <td>25.487740</td>
          <td>0.154825</td>
          <td>22.863164</td>
          <td>0.034555</td>
          <td>19.540759</td>
          <td>19.087545</td>
          <td>28.700685</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.323978</td>
          <td>0.011386</td>
          <td>17.744558</td>
          <td>0.005003</td>
          <td>21.594629</td>
          <td>0.005248</td>
          <td>23.080529</td>
          <td>0.010610</td>
          <td>25.340177</td>
          <td>0.136373</td>
          <td>24.556935</td>
          <td>0.153300</td>
          <td>24.709485</td>
          <td>24.792617</td>
          <td>19.992837</td>
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
          <td>24.500143</td>
          <td>0.069909</td>
          <td>23.820586</td>
          <td>0.013674</td>
          <td>21.982509</td>
          <td>0.005462</td>
          <td>23.655736</td>
          <td>0.016496</td>
          <td>17.710043</td>
          <td>0.005007</td>
          <td>21.848403</td>
          <td>0.014562</td>
          <td>22.307822</td>
          <td>19.851597</td>
          <td>18.358839</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.121465</td>
          <td>0.005012</td>
          <td>22.968056</td>
          <td>0.007803</td>
          <td>19.612742</td>
          <td>0.005014</td>
          <td>23.567883</td>
          <td>0.015357</td>
          <td>24.228979</td>
          <td>0.051329</td>
          <td>22.772768</td>
          <td>0.031909</td>
          <td>19.633041</td>
          <td>19.809357</td>
          <td>20.429043</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.532986</td>
          <td>0.007266</td>
          <td>20.885760</td>
          <td>0.005123</td>
          <td>18.412268</td>
          <td>0.005004</td>
          <td>21.934530</td>
          <td>0.006040</td>
          <td>20.818804</td>
          <td>0.005567</td>
          <td>19.902539</td>
          <td>0.005571</td>
          <td>24.681912</td>
          <td>19.873742</td>
          <td>24.100921</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.532780</td>
          <td>0.013231</td>
          <td>23.728654</td>
          <td>0.012746</td>
          <td>22.247800</td>
          <td>0.005709</td>
          <td>24.956588</td>
          <td>0.051079</td>
          <td>24.102266</td>
          <td>0.045868</td>
          <td>22.284925</td>
          <td>0.020892</td>
          <td>28.690592</td>
          <td>23.757155</td>
          <td>24.884154</td>
        </tr>
        <tr>
          <th>999</th>
          <td>14.900758</td>
          <td>0.005001</td>
          <td>23.300956</td>
          <td>0.009453</td>
          <td>26.465682</td>
          <td>0.118613</td>
          <td>22.049853</td>
          <td>0.006248</td>
          <td>26.778517</td>
          <td>0.442084</td>
          <td>22.939623</td>
          <td>0.036970</td>
          <td>28.193311</td>
          <td>21.254340</td>
          <td>19.903971</td>
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
          <td>21.903785</td>
          <td>21.760425</td>
          <td>24.309006</td>
          <td>26.795652</td>
          <td>24.122152</td>
          <td>19.726739</td>
          <td>20.568686</td>
          <td>0.005033</td>
          <td>25.269770</td>
          <td>0.074243</td>
          <td>22.908365</td>
          <td>0.010004</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.249192</td>
          <td>19.859864</td>
          <td>23.644446</td>
          <td>18.330824</td>
          <td>24.432829</td>
          <td>19.701743</td>
          <td>27.205072</td>
          <td>0.234974</td>
          <td>17.784748</td>
          <td>0.005001</td>
          <td>18.871760</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.050884</td>
          <td>28.741263</td>
          <td>23.709107</td>
          <td>17.494134</td>
          <td>26.484200</td>
          <td>23.424059</td>
          <td>28.523102</td>
          <td>0.646204</td>
          <td>25.424429</td>
          <td>0.085131</td>
          <td>16.484736</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.788579</td>
          <td>19.513627</td>
          <td>22.313876</td>
          <td>22.052501</td>
          <td>25.371641</td>
          <td>22.880575</td>
          <td>19.543552</td>
          <td>0.005005</td>
          <td>19.083729</td>
          <td>0.005007</td>
          <td>28.916125</td>
          <td>1.203582</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.337263</td>
          <td>17.751578</td>
          <td>21.596767</td>
          <td>23.100271</td>
          <td>25.356445</td>
          <td>24.416644</td>
          <td>24.681930</td>
          <td>0.025850</td>
          <td>24.719915</td>
          <td>0.045512</td>
          <td>19.988572</td>
          <td>0.005035</td>
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
          <td>24.507660</td>
          <td>23.806922</td>
          <td>21.980896</td>
          <td>23.640281</td>
          <td>17.713438</td>
          <td>21.866113</td>
          <td>22.306444</td>
          <td>0.005765</td>
          <td>19.854741</td>
          <td>0.005027</td>
          <td>18.356398</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.118436</td>
          <td>22.960039</td>
          <td>19.614389</td>
          <td>23.575054</td>
          <td>24.263408</td>
          <td>22.792409</td>
          <td>19.635323</td>
          <td>0.005006</td>
          <td>19.806486</td>
          <td>0.005025</td>
          <td>20.421609</td>
          <td>0.005077</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.545218</td>
          <td>20.885719</td>
          <td>18.423965</td>
          <td>21.932139</td>
          <td>20.825525</td>
          <td>19.887936</td>
          <td>24.696492</td>
          <td>0.026183</td>
          <td>19.872573</td>
          <td>0.005028</td>
          <td>24.067902</td>
          <td>0.025534</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.541432</td>
          <td>23.713737</td>
          <td>22.247661</td>
          <td>24.997789</td>
          <td>24.121224</td>
          <td>22.323170</td>
          <td>28.325769</td>
          <td>0.562099</td>
          <td>23.760602</td>
          <td>0.019568</td>
          <td>24.930526</td>
          <td>0.054911</td>
        </tr>
        <tr>
          <th>999</th>
          <td>14.900699</td>
          <td>23.306596</td>
          <td>26.684344</td>
          <td>22.048573</td>
          <td>26.911642</td>
          <td>22.922926</td>
          <td>27.857712</td>
          <td>0.396431</td>
          <td>21.252452</td>
          <td>0.005345</td>
          <td>19.911008</td>
          <td>0.005030</td>
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
          <td>21.903785</td>
          <td>21.760425</td>
          <td>24.309006</td>
          <td>26.795652</td>
          <td>24.122152</td>
          <td>19.726739</td>
          <td>20.574752</td>
          <td>0.008595</td>
          <td>25.610200</td>
          <td>0.480069</td>
          <td>22.904944</td>
          <td>0.053674</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.249192</td>
          <td>19.859864</td>
          <td>23.644446</td>
          <td>18.330824</td>
          <td>24.432829</td>
          <td>19.701743</td>
          <td>25.987473</td>
          <td>0.723019</td>
          <td>17.784034</td>
          <td>0.005020</td>
          <td>18.871854</td>
          <td>0.005174</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.050884</td>
          <td>28.741263</td>
          <td>23.709107</td>
          <td>17.494134</td>
          <td>26.484200</td>
          <td>23.424059</td>
          <td>27.137184</td>
          <td>1.428800</td>
          <td>24.871086</td>
          <td>0.269321</td>
          <td>16.482025</td>
          <td>0.005002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.788579</td>
          <td>19.513627</td>
          <td>22.313876</td>
          <td>22.052501</td>
          <td>25.371641</td>
          <td>22.880575</td>
          <td>19.532800</td>
          <td>0.005673</td>
          <td>19.093872</td>
          <td>0.005217</td>
          <td>25.832552</td>
          <td>0.606522</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.337263</td>
          <td>17.751578</td>
          <td>21.596767</td>
          <td>23.100271</td>
          <td>25.356445</td>
          <td>24.416644</td>
          <td>25.180292</td>
          <td>0.403387</td>
          <td>27.211480</td>
          <td>1.338265</td>
          <td>19.991175</td>
          <td>0.006237</td>
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
          <td>24.507660</td>
          <td>23.806922</td>
          <td>21.980896</td>
          <td>23.640281</td>
          <td>17.713438</td>
          <td>21.866113</td>
          <td>22.271452</td>
          <td>0.033377</td>
          <td>19.851826</td>
          <td>0.005826</td>
          <td>18.360498</td>
          <td>0.005069</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.118436</td>
          <td>22.960039</td>
          <td>19.614389</td>
          <td>23.575054</td>
          <td>24.263408</td>
          <td>22.792409</td>
          <td>19.628960</td>
          <td>0.005795</td>
          <td>19.815223</td>
          <td>0.005776</td>
          <td>20.426173</td>
          <td>0.007480</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.545218</td>
          <td>20.885719</td>
          <td>18.423965</td>
          <td>21.932139</td>
          <td>20.825525</td>
          <td>19.887936</td>
          <td>25.335018</td>
          <td>0.453798</td>
          <td>19.877479</td>
          <td>0.005863</td>
          <td>24.463732</td>
          <td>0.208883</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.541432</td>
          <td>23.713737</td>
          <td>22.247661</td>
          <td>24.997789</td>
          <td>24.121224</td>
          <td>22.323170</td>
          <td>28.751204</td>
          <td>2.790237</td>
          <td>23.750629</td>
          <td>0.103878</td>
          <td>24.827809</td>
          <td>0.282035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>14.900699</td>
          <td>23.306596</td>
          <td>26.684344</td>
          <td>22.048573</td>
          <td>26.911642</td>
          <td>22.922926</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.270764</td>
          <td>0.012105</td>
          <td>19.902149</td>
          <td>0.006066</td>
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


