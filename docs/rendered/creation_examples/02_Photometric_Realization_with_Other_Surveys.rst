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
          <td>20.598371</td>
          <td>19.297150</td>
          <td>23.957103</td>
          <td>23.583848</td>
          <td>20.229774</td>
          <td>24.128403</td>
          <td>22.902762</td>
          <td>21.202215</td>
          <td>18.969957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.306515</td>
          <td>20.845235</td>
          <td>21.924387</td>
          <td>22.451512</td>
          <td>20.869617</td>
          <td>15.631481</td>
          <td>25.169863</td>
          <td>19.496282</td>
          <td>21.770096</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.763673</td>
          <td>25.315762</td>
          <td>23.915984</td>
          <td>22.684868</td>
          <td>17.951825</td>
          <td>23.498322</td>
          <td>22.445617</td>
          <td>20.073656</td>
          <td>27.206595</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.307588</td>
          <td>16.456122</td>
          <td>22.139283</td>
          <td>26.712351</td>
          <td>23.806534</td>
          <td>24.462026</td>
          <td>23.534726</td>
          <td>18.490846</td>
          <td>20.792820</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.133303</td>
          <td>29.338692</td>
          <td>21.672762</td>
          <td>19.852074</td>
          <td>20.333490</td>
          <td>20.493468</td>
          <td>27.417395</td>
          <td>18.541400</td>
          <td>21.948432</td>
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
          <td>20.354596</td>
          <td>22.251479</td>
          <td>26.555908</td>
          <td>21.663828</td>
          <td>26.891673</td>
          <td>28.467886</td>
          <td>22.966656</td>
          <td>18.908961</td>
          <td>25.614211</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990976</td>
          <td>24.840660</td>
          <td>19.662452</td>
          <td>27.311242</td>
          <td>21.778766</td>
          <td>20.554785</td>
          <td>26.894775</td>
          <td>27.616774</td>
          <td>23.207046</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.005934</td>
          <td>24.216715</td>
          <td>23.185927</td>
          <td>22.431209</td>
          <td>25.378936</td>
          <td>20.154778</td>
          <td>21.233489</td>
          <td>22.829658</td>
          <td>19.365136</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.577676</td>
          <td>24.634198</td>
          <td>24.877191</td>
          <td>19.056823</td>
          <td>25.669024</td>
          <td>20.812546</td>
          <td>18.896630</td>
          <td>24.945341</td>
          <td>21.306631</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.498871</td>
          <td>21.049926</td>
          <td>25.098142</td>
          <td>22.215318</td>
          <td>23.474626</td>
          <td>26.978007</td>
          <td>26.257033</td>
          <td>23.208498</td>
          <td>21.329319</td>
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
          <td>20.598012</td>
          <td>0.005617</td>
          <td>19.293846</td>
          <td>0.005016</td>
          <td>23.963032</td>
          <td>0.013545</td>
          <td>23.613169</td>
          <td>0.015931</td>
          <td>20.231112</td>
          <td>0.005221</td>
          <td>24.223748</td>
          <td>0.114940</td>
          <td>22.902762</td>
          <td>21.202215</td>
          <td>18.969957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.307572</td>
          <td>0.011258</td>
          <td>20.840120</td>
          <td>0.005116</td>
          <td>21.925072</td>
          <td>0.005421</td>
          <td>22.445061</td>
          <td>0.007290</td>
          <td>20.864995</td>
          <td>0.005611</td>
          <td>15.637254</td>
          <td>0.005002</td>
          <td>25.169863</td>
          <td>19.496282</td>
          <td>21.770096</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.759813</td>
          <td>0.005063</td>
          <td>25.321066</td>
          <td>0.049181</td>
          <td>23.908451</td>
          <td>0.012985</td>
          <td>22.689843</td>
          <td>0.008276</td>
          <td>17.952057</td>
          <td>0.005009</td>
          <td>23.446907</td>
          <td>0.057977</td>
          <td>22.445617</td>
          <td>20.073656</td>
          <td>27.206595</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.306292</td>
          <td>0.006656</td>
          <td>16.462705</td>
          <td>0.005001</td>
          <td>22.148517</td>
          <td>0.005604</td>
          <td>26.348771</td>
          <td>0.172709</td>
          <td>23.786642</td>
          <td>0.034683</td>
          <td>24.516916</td>
          <td>0.148126</td>
          <td>23.534726</td>
          <td>18.490846</td>
          <td>20.792820</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.885812</td>
          <td>0.230149</td>
          <td>28.844809</td>
          <td>0.844274</td>
          <td>21.671340</td>
          <td>0.005280</td>
          <td>19.848830</td>
          <td>0.005040</td>
          <td>20.335433</td>
          <td>0.005261</td>
          <td>20.480472</td>
          <td>0.006429</td>
          <td>27.417395</td>
          <td>18.541400</td>
          <td>21.948432</td>
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
          <td>20.360735</td>
          <td>0.005446</td>
          <td>22.259858</td>
          <td>0.005974</td>
          <td>26.578257</td>
          <td>0.130782</td>
          <td>21.673037</td>
          <td>0.005684</td>
          <td>29.169304</td>
          <td>1.859354</td>
          <td>26.818550</td>
          <td>0.863726</td>
          <td>22.966656</td>
          <td>18.908961</td>
          <td>25.614211</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.978849</td>
          <td>0.018771</td>
          <td>24.839346</td>
          <td>0.032148</td>
          <td>19.661711</td>
          <td>0.005015</td>
          <td>27.196287</td>
          <td>0.346569</td>
          <td>21.771264</td>
          <td>0.007509</td>
          <td>20.567893</td>
          <td>0.006636</td>
          <td>26.894775</td>
          <td>27.616774</td>
          <td>23.207046</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.008652</td>
          <td>0.009282</td>
          <td>24.242591</td>
          <td>0.019234</td>
          <td>23.179277</td>
          <td>0.007990</td>
          <td>22.428367</td>
          <td>0.007233</td>
          <td>25.469405</td>
          <td>0.152411</td>
          <td>20.154054</td>
          <td>0.005854</td>
          <td>21.233489</td>
          <td>22.829658</td>
          <td>19.365136</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.772908</td>
          <td>1.599562</td>
          <td>24.606312</td>
          <td>0.026235</td>
          <td>24.839681</td>
          <td>0.028241</td>
          <td>19.063072</td>
          <td>0.005014</td>
          <td>25.931436</td>
          <td>0.225185</td>
          <td>20.810638</td>
          <td>0.007361</td>
          <td>18.896630</td>
          <td>24.945341</td>
          <td>21.306631</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.507673</td>
          <td>0.070374</td>
          <td>21.061053</td>
          <td>0.005159</td>
          <td>25.108351</td>
          <td>0.035771</td>
          <td>22.220738</td>
          <td>0.006628</td>
          <td>23.452496</td>
          <td>0.025873</td>
          <td>26.966690</td>
          <td>0.947479</td>
          <td>26.257033</td>
          <td>23.208498</td>
          <td>21.329319</td>
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
          <td>20.598371</td>
          <td>19.297150</td>
          <td>23.957103</td>
          <td>23.583848</td>
          <td>20.229774</td>
          <td>24.128403</td>
          <td>22.902140</td>
          <td>0.007045</td>
          <td>21.203333</td>
          <td>0.005316</td>
          <td>18.971667</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.306515</td>
          <td>20.845235</td>
          <td>21.924387</td>
          <td>22.451512</td>
          <td>20.869617</td>
          <td>15.631481</td>
          <td>25.171351</td>
          <td>0.039868</td>
          <td>19.492951</td>
          <td>0.005014</td>
          <td>21.786233</td>
          <td>0.005876</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.763673</td>
          <td>25.315762</td>
          <td>23.915984</td>
          <td>22.684868</td>
          <td>17.951825</td>
          <td>23.498322</td>
          <td>22.445659</td>
          <td>0.005970</td>
          <td>20.076328</td>
          <td>0.005041</td>
          <td>27.086999</td>
          <td>0.347007</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.307588</td>
          <td>16.456122</td>
          <td>22.139283</td>
          <td>26.712351</td>
          <td>23.806534</td>
          <td>24.462026</td>
          <td>23.533875</td>
          <td>0.010182</td>
          <td>18.484735</td>
          <td>0.005002</td>
          <td>20.797736</td>
          <td>0.005152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.133303</td>
          <td>29.338692</td>
          <td>21.672762</td>
          <td>19.852074</td>
          <td>20.333490</td>
          <td>20.493468</td>
          <td>27.832409</td>
          <td>0.388754</td>
          <td>18.541936</td>
          <td>0.005002</td>
          <td>21.944173</td>
          <td>0.006144</td>
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
          <td>20.354596</td>
          <td>22.251479</td>
          <td>26.555908</td>
          <td>21.663828</td>
          <td>26.891673</td>
          <td>28.467886</td>
          <td>22.961618</td>
          <td>0.007245</td>
          <td>18.912138</td>
          <td>0.005005</td>
          <td>25.630346</td>
          <td>0.102047</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990976</td>
          <td>24.840660</td>
          <td>19.662452</td>
          <td>27.311242</td>
          <td>21.778766</td>
          <td>20.554785</td>
          <td>26.665494</td>
          <td>0.148948</td>
          <td>27.658063</td>
          <td>0.535245</td>
          <td>23.195896</td>
          <td>0.012340</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.005934</td>
          <td>24.216715</td>
          <td>23.185927</td>
          <td>22.431209</td>
          <td>25.378936</td>
          <td>20.154778</td>
          <td>21.235186</td>
          <td>0.005113</td>
          <td>22.828175</td>
          <td>0.009476</td>
          <td>19.359365</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.577676</td>
          <td>24.634198</td>
          <td>24.877191</td>
          <td>19.056823</td>
          <td>25.669024</td>
          <td>20.812546</td>
          <td>18.898123</td>
          <td>0.005002</td>
          <td>25.061566</td>
          <td>0.061707</td>
          <td>21.309692</td>
          <td>0.005382</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.498871</td>
          <td>21.049926</td>
          <td>25.098142</td>
          <td>22.215318</td>
          <td>23.474626</td>
          <td>26.978007</td>
          <td>26.418426</td>
          <td>0.120278</td>
          <td>23.221834</td>
          <td>0.012588</td>
          <td>21.327686</td>
          <td>0.005394</td>
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
          <td>20.598371</td>
          <td>19.297150</td>
          <td>23.957103</td>
          <td>23.583848</td>
          <td>20.229774</td>
          <td>24.128403</td>
          <td>22.780746</td>
          <td>0.052529</td>
          <td>21.198505</td>
          <td>0.011465</td>
          <td>18.973505</td>
          <td>0.005209</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.306515</td>
          <td>20.845235</td>
          <td>21.924387</td>
          <td>22.451512</td>
          <td>20.869617</td>
          <td>15.631481</td>
          <td>25.266488</td>
          <td>0.430872</td>
          <td>19.497460</td>
          <td>0.005446</td>
          <td>21.775856</td>
          <td>0.019824</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.763673</td>
          <td>25.315762</td>
          <td>23.915984</td>
          <td>22.684868</td>
          <td>17.951825</td>
          <td>23.498322</td>
          <td>22.479464</td>
          <td>0.040158</td>
          <td>20.075550</td>
          <td>0.006205</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>3</th>
          <td>21.307588</td>
          <td>16.456122</td>
          <td>22.139283</td>
          <td>26.712351</td>
          <td>23.806534</td>
          <td>24.462026</td>
          <td>23.580958</td>
          <td>0.106675</td>
          <td>18.488228</td>
          <td>0.005072</td>
          <td>20.782614</td>
          <td>0.009198</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.133303</td>
          <td>29.338692</td>
          <td>21.672762</td>
          <td>19.852074</td>
          <td>20.333490</td>
          <td>20.493468</td>
          <td>inf</td>
          <td>inf</td>
          <td>18.543309</td>
          <td>0.005080</td>
          <td>21.942929</td>
          <td>0.022897</td>
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
          <td>20.354596</td>
          <td>22.251479</td>
          <td>26.555908</td>
          <td>21.663828</td>
          <td>26.891673</td>
          <td>28.467886</td>
          <td>22.865400</td>
          <td>0.056644</td>
          <td>18.915022</td>
          <td>0.005157</td>
          <td>25.473638</td>
          <td>0.467145</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.990976</td>
          <td>24.840660</td>
          <td>19.662452</td>
          <td>27.311242</td>
          <td>21.778766</td>
          <td>20.554785</td>
          <td>25.594602</td>
          <td>0.549612</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.143948</td>
          <td>0.066398</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.005934</td>
          <td>24.216715</td>
          <td>23.185927</td>
          <td>22.431209</td>
          <td>25.378936</td>
          <td>20.154778</td>
          <td>21.233402</td>
          <td>0.013739</td>
          <td>22.878597</td>
          <td>0.047956</td>
          <td>19.368619</td>
          <td>0.005424</td>
        </tr>
        <tr>
          <th>998</th>
          <td>27.577676</td>
          <td>24.634198</td>
          <td>24.877191</td>
          <td>19.056823</td>
          <td>25.669024</td>
          <td>20.812546</td>
          <td>18.903378</td>
          <td>0.005221</td>
          <td>24.953620</td>
          <td>0.287993</td>
          <td>21.295100</td>
          <td>0.013329</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.498871</td>
          <td>21.049926</td>
          <td>25.098142</td>
          <td>22.215318</td>
          <td>23.474626</td>
          <td>26.978007</td>
          <td>25.495493</td>
          <td>0.511321</td>
          <td>23.280154</td>
          <td>0.068568</td>
          <td>21.312961</td>
          <td>0.013518</td>
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


