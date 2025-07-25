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
          <td>25.900564</td>
          <td>26.176753</td>
          <td>17.228986</td>
          <td>25.627583</td>
          <td>18.549372</td>
          <td>18.921553</td>
          <td>21.692523</td>
          <td>21.921700</td>
          <td>27.772008</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.213338</td>
          <td>27.246628</td>
          <td>25.936229</td>
          <td>21.436581</td>
          <td>23.490792</td>
          <td>21.511540</td>
          <td>19.753496</td>
          <td>22.668805</td>
          <td>22.144078</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.468531</td>
          <td>26.769969</td>
          <td>22.675038</td>
          <td>23.020542</td>
          <td>17.770585</td>
          <td>21.442560</td>
          <td>23.235860</td>
          <td>25.057929</td>
          <td>23.666229</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.402804</td>
          <td>23.295785</td>
          <td>17.988373</td>
          <td>19.906252</td>
          <td>27.283789</td>
          <td>17.881651</td>
          <td>23.292569</td>
          <td>23.659804</td>
          <td>21.106019</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.815892</td>
          <td>20.338867</td>
          <td>26.078648</td>
          <td>23.276711</td>
          <td>19.819437</td>
          <td>23.960033</td>
          <td>19.747575</td>
          <td>21.684086</td>
          <td>21.693070</td>
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
          <td>25.814589</td>
          <td>25.350445</td>
          <td>20.930571</td>
          <td>20.039541</td>
          <td>19.867558</td>
          <td>25.014904</td>
          <td>26.705157</td>
          <td>20.819588</td>
          <td>19.830546</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.800108</td>
          <td>20.576552</td>
          <td>22.777080</td>
          <td>22.957920</td>
          <td>23.680476</td>
          <td>23.096273</td>
          <td>18.736467</td>
          <td>13.903684</td>
          <td>21.571271</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.287171</td>
          <td>23.723269</td>
          <td>26.279924</td>
          <td>23.196062</td>
          <td>22.556418</td>
          <td>25.311866</td>
          <td>23.263493</td>
          <td>24.713458</td>
          <td>24.093121</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.101579</td>
          <td>24.399786</td>
          <td>19.347136</td>
          <td>21.370818</td>
          <td>22.543288</td>
          <td>26.529790</td>
          <td>18.948202</td>
          <td>22.743041</td>
          <td>16.940296</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.339446</td>
          <td>21.498093</td>
          <td>24.876234</td>
          <td>31.771224</td>
          <td>19.858150</td>
          <td>27.946532</td>
          <td>22.833683</td>
          <td>23.626900</td>
          <td>25.852980</td>
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
          <td>25.834491</td>
          <td>0.220562</td>
          <td>26.366383</td>
          <td>0.123383</td>
          <td>17.233559</td>
          <td>0.005001</td>
          <td>25.593505</td>
          <td>0.089757</td>
          <td>18.551907</td>
          <td>0.005020</td>
          <td>18.919293</td>
          <td>0.005120</td>
          <td>21.692523</td>
          <td>21.921700</td>
          <td>27.772008</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.201809</td>
          <td>0.010483</td>
          <td>27.332683</td>
          <td>0.278530</td>
          <td>25.936932</td>
          <td>0.074574</td>
          <td>21.436965</td>
          <td>0.005467</td>
          <td>23.479119</td>
          <td>0.026479</td>
          <td>21.501674</td>
          <td>0.011189</td>
          <td>19.753496</td>
          <td>22.668805</td>
          <td>22.144078</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.547587</td>
          <td>0.072886</td>
          <td>26.505681</td>
          <td>0.139171</td>
          <td>22.669930</td>
          <td>0.006387</td>
          <td>23.006128</td>
          <td>0.010081</td>
          <td>17.770910</td>
          <td>0.005008</td>
          <td>21.444044</td>
          <td>0.010740</td>
          <td>23.235860</td>
          <td>25.057929</td>
          <td>23.666229</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.408173</td>
          <td>0.005132</td>
          <td>23.281714</td>
          <td>0.009338</td>
          <td>17.983206</td>
          <td>0.005002</td>
          <td>19.912868</td>
          <td>0.005044</td>
          <td>26.984316</td>
          <td>0.515312</td>
          <td>17.881554</td>
          <td>0.005027</td>
          <td>23.292569</td>
          <td>23.659804</td>
          <td>21.106019</td>
        </tr>
        <tr>
          <th>4</th>
          <td>26.364092</td>
          <td>0.338954</td>
          <td>20.337283</td>
          <td>0.005058</td>
          <td>26.171430</td>
          <td>0.091700</td>
          <td>23.278329</td>
          <td>0.012249</td>
          <td>19.815650</td>
          <td>0.005116</td>
          <td>23.950286</td>
          <td>0.090473</td>
          <td>19.747575</td>
          <td>21.684086</td>
          <td>21.693070</td>
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
          <td>25.700916</td>
          <td>0.197285</td>
          <td>25.432893</td>
          <td>0.054302</td>
          <td>20.922775</td>
          <td>0.005087</td>
          <td>20.033291</td>
          <td>0.005053</td>
          <td>19.857487</td>
          <td>0.005124</td>
          <td>25.137349</td>
          <td>0.249724</td>
          <td>26.705157</td>
          <td>20.819588</td>
          <td>19.830546</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.805213</td>
          <td>0.005216</td>
          <td>20.576544</td>
          <td>0.005080</td>
          <td>22.771585</td>
          <td>0.006624</td>
          <td>22.960243</td>
          <td>0.009776</td>
          <td>23.687534</td>
          <td>0.031782</td>
          <td>23.139191</td>
          <td>0.044120</td>
          <td>18.736467</td>
          <td>13.903684</td>
          <td>21.571271</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.362694</td>
          <td>0.338580</td>
          <td>23.700437</td>
          <td>0.012478</td>
          <td>26.269428</td>
          <td>0.099936</td>
          <td>23.182625</td>
          <td>0.011412</td>
          <td>22.544167</td>
          <td>0.012248</td>
          <td>24.960256</td>
          <td>0.215657</td>
          <td>23.263493</td>
          <td>24.713458</td>
          <td>24.093121</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.088579</td>
          <td>0.009747</td>
          <td>24.370448</td>
          <td>0.021425</td>
          <td>19.343923</td>
          <td>0.005010</td>
          <td>21.374750</td>
          <td>0.005422</td>
          <td>22.536988</td>
          <td>0.012182</td>
          <td>25.573814</td>
          <td>0.354747</td>
          <td>18.948202</td>
          <td>22.743041</td>
          <td>16.940296</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.075982</td>
          <td>0.115658</td>
          <td>21.497457</td>
          <td>0.005303</td>
          <td>24.839899</td>
          <td>0.028246</td>
          <td>29.209584</td>
          <td>1.328904</td>
          <td>19.855900</td>
          <td>0.005123</td>
          <td>25.899373</td>
          <td>0.455656</td>
          <td>22.833683</td>
          <td>23.626900</td>
          <td>25.852980</td>
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
          <td>25.900564</td>
          <td>26.176753</td>
          <td>17.228986</td>
          <td>25.627583</td>
          <td>18.549372</td>
          <td>18.921553</td>
          <td>21.695833</td>
          <td>0.005261</td>
          <td>21.924247</td>
          <td>0.006107</td>
          <td>27.485248</td>
          <td>0.471218</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.213338</td>
          <td>27.246628</td>
          <td>25.936229</td>
          <td>21.436581</td>
          <td>23.490792</td>
          <td>21.511540</td>
          <td>19.748831</td>
          <td>0.005007</td>
          <td>22.678875</td>
          <td>0.008617</td>
          <td>22.143051</td>
          <td>0.006587</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.468531</td>
          <td>26.769969</td>
          <td>22.675038</td>
          <td>23.020542</td>
          <td>17.770585</td>
          <td>21.442560</td>
          <td>23.225695</td>
          <td>0.008347</td>
          <td>25.041603</td>
          <td>0.060621</td>
          <td>23.666513</td>
          <td>0.018065</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.402804</td>
          <td>23.295785</td>
          <td>17.988373</td>
          <td>19.906252</td>
          <td>27.283789</td>
          <td>17.881651</td>
          <td>23.299970</td>
          <td>0.008729</td>
          <td>23.675942</td>
          <td>0.018209</td>
          <td>21.102106</td>
          <td>0.005264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.815892</td>
          <td>20.338867</td>
          <td>26.078648</td>
          <td>23.276711</td>
          <td>19.819437</td>
          <td>23.960033</td>
          <td>19.752246</td>
          <td>0.005007</td>
          <td>21.678037</td>
          <td>0.005728</td>
          <td>21.692725</td>
          <td>0.005747</td>
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
          <td>25.814589</td>
          <td>25.350445</td>
          <td>20.930571</td>
          <td>20.039541</td>
          <td>19.867558</td>
          <td>25.014904</td>
          <td>26.563869</td>
          <td>0.136453</td>
          <td>20.815184</td>
          <td>0.005157</td>
          <td>19.830146</td>
          <td>0.005026</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.800108</td>
          <td>20.576552</td>
          <td>22.777080</td>
          <td>22.957920</td>
          <td>23.680476</td>
          <td>23.096273</td>
          <td>18.725767</td>
          <td>0.005001</td>
          <td>13.900199</td>
          <td>0.005000</td>
          <td>21.573909</td>
          <td>0.005608</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.287171</td>
          <td>23.723269</td>
          <td>26.279924</td>
          <td>23.196062</td>
          <td>22.556418</td>
          <td>25.311866</td>
          <td>23.264772</td>
          <td>0.008544</td>
          <td>24.710902</td>
          <td>0.045148</td>
          <td>24.123229</td>
          <td>0.026805</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.101579</td>
          <td>24.399786</td>
          <td>19.347136</td>
          <td>21.370818</td>
          <td>22.543288</td>
          <td>26.529790</td>
          <td>18.950445</td>
          <td>0.005002</td>
          <td>22.728258</td>
          <td>0.008884</td>
          <td>16.935674</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.339446</td>
          <td>21.498093</td>
          <td>24.876234</td>
          <td>31.771224</td>
          <td>19.858150</td>
          <td>27.946532</td>
          <td>22.842552</td>
          <td>0.006861</td>
          <td>23.604073</td>
          <td>0.017141</td>
          <td>25.750809</td>
          <td>0.113392</td>
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
          <td>25.900564</td>
          <td>26.176753</td>
          <td>17.228986</td>
          <td>25.627583</td>
          <td>18.549372</td>
          <td>18.921553</td>
          <td>21.674629</td>
          <td>0.019803</td>
          <td>21.910772</td>
          <td>0.020426</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.213338</td>
          <td>27.246628</td>
          <td>25.936229</td>
          <td>21.436581</td>
          <td>23.490792</td>
          <td>21.511540</td>
          <td>19.749156</td>
          <td>0.005975</td>
          <td>22.640892</td>
          <td>0.038802</td>
          <td>22.113155</td>
          <td>0.026568</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.468531</td>
          <td>26.769969</td>
          <td>22.675038</td>
          <td>23.020542</td>
          <td>17.770585</td>
          <td>21.442560</td>
          <td>23.254515</td>
          <td>0.080029</td>
          <td>24.760997</td>
          <td>0.246082</td>
          <td>23.617288</td>
          <td>0.100885</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.402804</td>
          <td>23.295785</td>
          <td>17.988373</td>
          <td>19.906252</td>
          <td>27.283789</td>
          <td>17.881651</td>
          <td>23.294791</td>
          <td>0.082931</td>
          <td>23.772498</td>
          <td>0.105888</td>
          <td>21.107906</td>
          <td>0.011545</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.815892</td>
          <td>20.338867</td>
          <td>26.078648</td>
          <td>23.276711</td>
          <td>19.819437</td>
          <td>23.960033</td>
          <td>19.743739</td>
          <td>0.005966</td>
          <td>21.671517</td>
          <td>0.016681</td>
          <td>21.717928</td>
          <td>0.018869</td>
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
          <td>25.814589</td>
          <td>25.350445</td>
          <td>20.930571</td>
          <td>20.039541</td>
          <td>19.867558</td>
          <td>25.014904</td>
          <td>25.843670</td>
          <td>0.655478</td>
          <td>20.828579</td>
          <td>0.008886</td>
          <td>19.826620</td>
          <td>0.005939</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.800108</td>
          <td>20.576552</td>
          <td>22.777080</td>
          <td>22.957920</td>
          <td>23.680476</td>
          <td>23.096273</td>
          <td>18.739397</td>
          <td>0.005164</td>
          <td>13.899737</td>
          <td>0.005000</td>
          <td>21.558090</td>
          <td>0.016496</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.287171</td>
          <td>23.723269</td>
          <td>26.279924</td>
          <td>23.196062</td>
          <td>22.556418</td>
          <td>25.311866</td>
          <td>23.246826</td>
          <td>0.079487</td>
          <td>25.101271</td>
          <td>0.324223</td>
          <td>24.182096</td>
          <td>0.164600</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.101579</td>
          <td>24.399786</td>
          <td>19.347136</td>
          <td>21.370818</td>
          <td>22.543288</td>
          <td>26.529790</td>
          <td>18.940544</td>
          <td>0.005236</td>
          <td>22.686159</td>
          <td>0.040398</td>
          <td>16.938069</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.339446</td>
          <td>21.498093</td>
          <td>24.876234</td>
          <td>31.771224</td>
          <td>19.858150</td>
          <td>27.946532</td>
          <td>22.856537</td>
          <td>0.056199</td>
          <td>23.490463</td>
          <td>0.082614</td>
          <td>27.779008</td>
          <td>1.849723</td>
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


