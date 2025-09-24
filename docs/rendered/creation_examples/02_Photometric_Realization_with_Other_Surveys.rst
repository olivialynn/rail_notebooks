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
          <td>29.103583</td>
          <td>20.780625</td>
          <td>19.404047</td>
          <td>20.934711</td>
          <td>19.533993</td>
          <td>27.378112</td>
          <td>21.464443</td>
          <td>23.912009</td>
          <td>22.028079</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.088612</td>
          <td>25.767251</td>
          <td>21.512650</td>
          <td>24.606321</td>
          <td>22.841006</td>
          <td>24.204940</td>
          <td>27.403119</td>
          <td>19.158990</td>
          <td>24.476657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.667796</td>
          <td>25.394621</td>
          <td>26.058319</td>
          <td>22.759369</td>
          <td>20.228149</td>
          <td>21.960703</td>
          <td>24.648357</td>
          <td>27.738724</td>
          <td>27.444327</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.940446</td>
          <td>20.514286</td>
          <td>19.247138</td>
          <td>22.872866</td>
          <td>21.396971</td>
          <td>19.576637</td>
          <td>21.510778</td>
          <td>19.438124</td>
          <td>22.873539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.672985</td>
          <td>22.149696</td>
          <td>23.440419</td>
          <td>31.563232</td>
          <td>26.791552</td>
          <td>19.489477</td>
          <td>23.999282</td>
          <td>23.043983</td>
          <td>24.413866</td>
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
          <td>22.552898</td>
          <td>23.198446</td>
          <td>22.030030</td>
          <td>22.223995</td>
          <td>17.070472</td>
          <td>24.917858</td>
          <td>21.333058</td>
          <td>22.327568</td>
          <td>22.362646</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.561862</td>
          <td>26.133685</td>
          <td>25.424320</td>
          <td>19.560413</td>
          <td>27.139365</td>
          <td>25.115156</td>
          <td>21.406023</td>
          <td>26.382596</td>
          <td>27.339928</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.948011</td>
          <td>22.462495</td>
          <td>22.945581</td>
          <td>22.656842</td>
          <td>22.967568</td>
          <td>26.449581</td>
          <td>28.686805</td>
          <td>23.582110</td>
          <td>18.924880</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.432266</td>
          <td>28.505516</td>
          <td>19.327144</td>
          <td>23.449620</td>
          <td>21.176776</td>
          <td>18.958419</td>
          <td>23.617460</td>
          <td>24.002810</td>
          <td>24.958635</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.670533</td>
          <td>20.855644</td>
          <td>23.132699</td>
          <td>21.954332</td>
          <td>20.122994</td>
          <td>19.177121</td>
          <td>22.092246</td>
          <td>26.502718</td>
          <td>22.548117</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>20.783602</td>
          <td>0.005107</td>
          <td>19.411199</td>
          <td>0.005011</td>
          <td>20.927726</td>
          <td>0.005206</td>
          <td>19.531991</td>
          <td>0.005076</td>
          <td>25.554441</td>
          <td>0.349386</td>
          <td>21.464443</td>
          <td>23.912009</td>
          <td>22.028079</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.093452</td>
          <td>0.006230</td>
          <td>25.722697</td>
          <td>0.070177</td>
          <td>21.517195</td>
          <td>0.005219</td>
          <td>24.611625</td>
          <td>0.037613</td>
          <td>22.850203</td>
          <td>0.015558</td>
          <td>24.440545</td>
          <td>0.138702</td>
          <td>27.403119</td>
          <td>19.158990</td>
          <td>24.476657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.669200</td>
          <td>0.007729</td>
          <td>25.299056</td>
          <td>0.048231</td>
          <td>26.038080</td>
          <td>0.081541</td>
          <td>22.760947</td>
          <td>0.008625</td>
          <td>20.231051</td>
          <td>0.005221</td>
          <td>21.954070</td>
          <td>0.015854</td>
          <td>24.648357</td>
          <td>27.738724</td>
          <td>27.444327</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.939389</td>
          <td>0.005256</td>
          <td>20.507444</td>
          <td>0.005073</td>
          <td>19.242304</td>
          <td>0.005009</td>
          <td>22.878005</td>
          <td>0.009268</td>
          <td>21.398565</td>
          <td>0.006425</td>
          <td>19.571213</td>
          <td>0.005335</td>
          <td>21.510778</td>
          <td>19.438124</td>
          <td>22.873539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.675570</td>
          <td>0.014744</td>
          <td>22.154899</td>
          <td>0.005829</td>
          <td>23.455307</td>
          <td>0.009412</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.252877</td>
          <td>0.293007</td>
          <td>19.487852</td>
          <td>0.005294</td>
          <td>23.999282</td>
          <td>23.043983</td>
          <td>24.413866</td>
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
          <td>22.547517</td>
          <td>0.013377</td>
          <td>23.214686</td>
          <td>0.008960</td>
          <td>22.038740</td>
          <td>0.005506</td>
          <td>22.222220</td>
          <td>0.006632</td>
          <td>17.073624</td>
          <td>0.005004</td>
          <td>25.049093</td>
          <td>0.232186</td>
          <td>21.333058</td>
          <td>22.327568</td>
          <td>22.362646</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.597998</td>
          <td>0.031698</td>
          <td>26.062132</td>
          <td>0.094624</td>
          <td>25.494425</td>
          <td>0.050372</td>
          <td>19.559601</td>
          <td>0.005027</td>
          <td>27.310639</td>
          <td>0.650348</td>
          <td>25.854470</td>
          <td>0.440481</td>
          <td>21.406023</td>
          <td>26.382596</td>
          <td>27.339928</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.923939</td>
          <td>0.017950</td>
          <td>22.448200</td>
          <td>0.006299</td>
          <td>22.958898</td>
          <td>0.007160</td>
          <td>22.675136</td>
          <td>0.008208</td>
          <td>22.958820</td>
          <td>0.017003</td>
          <td>25.719888</td>
          <td>0.397449</td>
          <td>28.686805</td>
          <td>23.582110</td>
          <td>18.924880</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.421504</td>
          <td>0.006943</td>
          <td>28.961917</td>
          <td>0.909116</td>
          <td>19.325457</td>
          <td>0.005010</td>
          <td>23.446698</td>
          <td>0.013943</td>
          <td>21.169845</td>
          <td>0.005995</td>
          <td>18.962822</td>
          <td>0.005129</td>
          <td>23.617460</td>
          <td>24.002810</td>
          <td>24.958635</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.669932</td>
          <td>0.005057</td>
          <td>20.856555</td>
          <td>0.005118</td>
          <td>23.138081</td>
          <td>0.007816</td>
          <td>21.953117</td>
          <td>0.006071</td>
          <td>20.125882</td>
          <td>0.005188</td>
          <td>19.179265</td>
          <td>0.005180</td>
          <td>22.092246</td>
          <td>26.502718</td>
          <td>22.548117</td>
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
          <td>29.103583</td>
          <td>20.780625</td>
          <td>19.404047</td>
          <td>20.934711</td>
          <td>19.533993</td>
          <td>27.378112</td>
          <td>21.459501</td>
          <td>0.005170</td>
          <td>23.907626</td>
          <td>0.022206</td>
          <td>22.033926</td>
          <td>0.006328</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.088612</td>
          <td>25.767251</td>
          <td>21.512650</td>
          <td>24.606321</td>
          <td>22.841006</td>
          <td>24.204940</td>
          <td>27.079726</td>
          <td>0.211698</td>
          <td>19.155755</td>
          <td>0.005007</td>
          <td>24.484455</td>
          <td>0.036901</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.667796</td>
          <td>25.394621</td>
          <td>26.058319</td>
          <td>22.759369</td>
          <td>20.228149</td>
          <td>21.960703</td>
          <td>24.634212</td>
          <td>0.024793</td>
          <td>27.991307</td>
          <td>0.677327</td>
          <td>27.691568</td>
          <td>0.548408</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.940446</td>
          <td>20.514286</td>
          <td>19.247138</td>
          <td>22.872866</td>
          <td>21.396971</td>
          <td>19.576637</td>
          <td>21.513473</td>
          <td>0.005188</td>
          <td>19.441564</td>
          <td>0.005013</td>
          <td>22.864836</td>
          <td>0.009711</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.672985</td>
          <td>22.149696</td>
          <td>23.440419</td>
          <td>31.563232</td>
          <td>26.791552</td>
          <td>19.489477</td>
          <td>24.000439</td>
          <td>0.014497</td>
          <td>23.048689</td>
          <td>0.011052</td>
          <td>24.376806</td>
          <td>0.033536</td>
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
          <td>22.552898</td>
          <td>23.198446</td>
          <td>22.030030</td>
          <td>22.223995</td>
          <td>17.070472</td>
          <td>24.917858</td>
          <td>21.328637</td>
          <td>0.005134</td>
          <td>22.318932</td>
          <td>0.007100</td>
          <td>22.367270</td>
          <td>0.007264</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.561862</td>
          <td>26.133685</td>
          <td>25.424320</td>
          <td>19.560413</td>
          <td>27.139365</td>
          <td>25.115156</td>
          <td>21.406205</td>
          <td>0.005154</td>
          <td>26.510109</td>
          <td>0.217140</td>
          <td>26.736112</td>
          <td>0.261736</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.948011</td>
          <td>22.462495</td>
          <td>22.945581</td>
          <td>22.656842</td>
          <td>22.967568</td>
          <td>26.449581</td>
          <td>28.516544</td>
          <td>0.643268</td>
          <td>23.553129</td>
          <td>0.016428</td>
          <td>18.928459</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.432266</td>
          <td>28.505516</td>
          <td>19.327144</td>
          <td>23.449620</td>
          <td>21.176776</td>
          <td>18.958419</td>
          <td>23.617722</td>
          <td>0.010806</td>
          <td>23.999864</td>
          <td>0.024060</td>
          <td>24.924455</td>
          <td>0.054615</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.670533</td>
          <td>20.855644</td>
          <td>23.132699</td>
          <td>21.954332</td>
          <td>20.122994</td>
          <td>19.177121</td>
          <td>22.090087</td>
          <td>0.005525</td>
          <td>26.219020</td>
          <td>0.169866</td>
          <td>22.543524</td>
          <td>0.007963</td>
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
          <td>29.103583</td>
          <td>20.780625</td>
          <td>19.404047</td>
          <td>20.934711</td>
          <td>19.533993</td>
          <td>27.378112</td>
          <td>21.455975</td>
          <td>0.016467</td>
          <td>23.856080</td>
          <td>0.113915</td>
          <td>22.013182</td>
          <td>0.024341</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.088612</td>
          <td>25.767251</td>
          <td>21.512650</td>
          <td>24.606321</td>
          <td>22.841006</td>
          <td>24.204940</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.155239</td>
          <td>0.005242</td>
          <td>24.152158</td>
          <td>0.160441</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.667796</td>
          <td>25.394621</td>
          <td>26.058319</td>
          <td>22.759369</td>
          <td>20.228149</td>
          <td>21.960703</td>
          <td>24.778160</td>
          <td>0.293759</td>
          <td>31.321926</td>
          <td>5.084592</td>
          <td>26.754448</td>
          <td>1.097970</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.940446</td>
          <td>20.514286</td>
          <td>19.247138</td>
          <td>22.872866</td>
          <td>21.396971</td>
          <td>19.576637</td>
          <td>21.464297</td>
          <td>0.016581</td>
          <td>19.455889</td>
          <td>0.005414</td>
          <td>22.894137</td>
          <td>0.053159</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.672985</td>
          <td>22.149696</td>
          <td>23.440419</td>
          <td>31.563232</td>
          <td>26.791552</td>
          <td>19.489477</td>
          <td>23.708416</td>
          <td>0.119234</td>
          <td>23.093471</td>
          <td>0.058078</td>
          <td>24.323298</td>
          <td>0.185592</td>
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
          <td>22.552898</td>
          <td>23.198446</td>
          <td>22.030030</td>
          <td>22.223995</td>
          <td>17.070472</td>
          <td>24.917858</td>
          <td>21.353596</td>
          <td>0.015136</td>
          <td>22.310952</td>
          <td>0.028957</td>
          <td>22.396300</td>
          <td>0.034121</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.561862</td>
          <td>26.133685</td>
          <td>25.424320</td>
          <td>19.560413</td>
          <td>27.139365</td>
          <td>25.115156</td>
          <td>21.430966</td>
          <td>0.016129</td>
          <td>25.471470</td>
          <td>0.432507</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.948011</td>
          <td>22.462495</td>
          <td>22.945581</td>
          <td>22.656842</td>
          <td>22.967568</td>
          <td>26.449581</td>
          <td>29.821802</td>
          <td>3.807441</td>
          <td>23.510776</td>
          <td>0.084110</td>
          <td>18.917317</td>
          <td>0.005189</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.432266</td>
          <td>28.505516</td>
          <td>19.327144</td>
          <td>23.449620</td>
          <td>21.176776</td>
          <td>18.958419</td>
          <td>23.753819</td>
          <td>0.124038</td>
          <td>23.900456</td>
          <td>0.118410</td>
          <td>25.028467</td>
          <td>0.331308</td>
        </tr>
        <tr>
          <th>999</th>
          <td>18.670533</td>
          <td>20.855644</td>
          <td>23.132699</td>
          <td>21.954332</td>
          <td>20.122994</td>
          <td>19.177121</td>
          <td>22.101706</td>
          <td>0.028722</td>
          <td>25.557712</td>
          <td>0.461603</td>
          <td>22.592076</td>
          <td>0.040611</td>
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


