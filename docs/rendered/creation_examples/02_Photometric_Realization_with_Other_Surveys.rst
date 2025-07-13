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
          <td>24.186491</td>
          <td>19.913621</td>
          <td>18.692975</td>
          <td>20.278699</td>
          <td>25.033284</td>
          <td>22.546392</td>
          <td>28.605238</td>
          <td>24.762507</td>
          <td>21.733318</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.402012</td>
          <td>26.406537</td>
          <td>29.177406</td>
          <td>27.412522</td>
          <td>28.037283</td>
          <td>29.470755</td>
          <td>22.606549</td>
          <td>22.523679</td>
          <td>24.741682</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.296783</td>
          <td>23.430518</td>
          <td>21.442559</td>
          <td>22.826418</td>
          <td>21.306559</td>
          <td>21.802722</td>
          <td>21.900781</td>
          <td>21.465585</td>
          <td>20.671131</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.130794</td>
          <td>23.897564</td>
          <td>21.094488</td>
          <td>21.804194</td>
          <td>20.956197</td>
          <td>25.827833</td>
          <td>24.886686</td>
          <td>15.846461</td>
          <td>24.748101</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.058625</td>
          <td>21.976673</td>
          <td>25.371894</td>
          <td>20.515684</td>
          <td>26.334782</td>
          <td>21.265713</td>
          <td>19.806993</td>
          <td>27.059297</td>
          <td>22.902832</td>
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
          <td>23.637106</td>
          <td>25.841099</td>
          <td>21.242693</td>
          <td>26.713129</td>
          <td>25.015178</td>
          <td>26.048039</td>
          <td>20.378326</td>
          <td>21.171222</td>
          <td>22.855791</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.291902</td>
          <td>22.846186</td>
          <td>22.114126</td>
          <td>18.849951</td>
          <td>24.891017</td>
          <td>18.911009</td>
          <td>20.537299</td>
          <td>26.170240</td>
          <td>24.432317</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.357789</td>
          <td>26.673068</td>
          <td>24.757783</td>
          <td>23.920864</td>
          <td>28.567766</td>
          <td>22.678069</td>
          <td>21.882229</td>
          <td>23.199975</td>
          <td>17.060642</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.388636</td>
          <td>25.686051</td>
          <td>23.942786</td>
          <td>22.371048</td>
          <td>19.569141</td>
          <td>23.082648</td>
          <td>21.293030</td>
          <td>24.657753</td>
          <td>22.325594</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.522213</td>
          <td>23.339647</td>
          <td>14.147923</td>
          <td>25.393621</td>
          <td>23.376262</td>
          <td>26.810312</td>
          <td>20.038588</td>
          <td>22.791622</td>
          <td>24.663710</td>
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
          <td>24.266993</td>
          <td>0.056944</td>
          <td>19.922270</td>
          <td>0.005034</td>
          <td>18.682081</td>
          <td>0.005005</td>
          <td>20.275729</td>
          <td>0.005075</td>
          <td>25.064285</td>
          <td>0.107312</td>
          <td>22.580392</td>
          <td>0.026959</td>
          <td>28.605238</td>
          <td>24.762507</td>
          <td>21.733318</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.760981</td>
          <td>0.207460</td>
          <td>26.397017</td>
          <td>0.126702</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.427634</td>
          <td>0.414797</td>
          <td>27.484000</td>
          <td>0.731891</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.606549</td>
          <td>22.523679</td>
          <td>24.741682</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.349007</td>
          <td>0.061206</td>
          <td>23.435839</td>
          <td>0.010329</td>
          <td>21.445779</td>
          <td>0.005196</td>
          <td>22.813177</td>
          <td>0.008901</td>
          <td>21.313731</td>
          <td>0.006248</td>
          <td>21.802213</td>
          <td>0.014040</td>
          <td>21.900781</td>
          <td>21.465585</td>
          <td>20.671131</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.234325</td>
          <td>0.305714</td>
          <td>23.931378</td>
          <td>0.014915</td>
          <td>21.097013</td>
          <td>0.005114</td>
          <td>21.799915</td>
          <td>0.005839</td>
          <td>20.958276</td>
          <td>0.005710</td>
          <td>25.339197</td>
          <td>0.294326</td>
          <td>24.886686</td>
          <td>15.846461</td>
          <td>24.748101</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.069531</td>
          <td>0.009633</td>
          <td>21.977000</td>
          <td>0.005630</td>
          <td>25.310583</td>
          <td>0.042788</td>
          <td>20.513207</td>
          <td>0.005108</td>
          <td>26.234259</td>
          <td>0.288636</td>
          <td>21.271851</td>
          <td>0.009558</td>
          <td>19.806993</td>
          <td>27.059297</td>
          <td>22.902832</td>
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
          <td>23.626867</td>
          <td>0.032501</td>
          <td>25.824278</td>
          <td>0.076761</td>
          <td>21.234835</td>
          <td>0.005141</td>
          <td>27.007009</td>
          <td>0.298066</td>
          <td>25.047455</td>
          <td>0.105745</td>
          <td>25.677262</td>
          <td>0.384567</td>
          <td>20.378326</td>
          <td>21.171222</td>
          <td>22.855791</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.345078</td>
          <td>0.060995</td>
          <td>22.841964</td>
          <td>0.007337</td>
          <td>22.118737</td>
          <td>0.005576</td>
          <td>18.853904</td>
          <td>0.005011</td>
          <td>24.756422</td>
          <td>0.081895</td>
          <td>18.902135</td>
          <td>0.005117</td>
          <td>20.537299</td>
          <td>26.170240</td>
          <td>24.432317</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.351155</td>
          <td>0.011604</td>
          <td>26.565164</td>
          <td>0.146478</td>
          <td>24.774273</td>
          <td>0.026672</td>
          <td>23.907405</td>
          <td>0.020359</td>
          <td>26.566906</td>
          <td>0.375826</td>
          <td>22.696890</td>
          <td>0.029851</td>
          <td>21.882229</td>
          <td>23.199975</td>
          <td>17.060642</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.387027</td>
          <td>0.005462</td>
          <td>25.649034</td>
          <td>0.065753</td>
          <td>23.940637</td>
          <td>0.013312</td>
          <td>22.377818</td>
          <td>0.007070</td>
          <td>19.572638</td>
          <td>0.005080</td>
          <td>23.025423</td>
          <td>0.039887</td>
          <td>21.293030</td>
          <td>24.657753</td>
          <td>22.325594</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.519517</td>
          <td>0.029618</td>
          <td>23.354816</td>
          <td>0.009786</td>
          <td>14.143453</td>
          <td>0.005000</td>
          <td>25.245116</td>
          <td>0.065983</td>
          <td>23.384529</td>
          <td>0.024391</td>
          <td>26.222847</td>
          <td>0.577635</td>
          <td>20.038588</td>
          <td>22.791622</td>
          <td>24.663710</td>
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
          <td>24.186491</td>
          <td>19.913621</td>
          <td>18.692975</td>
          <td>20.278699</td>
          <td>25.033284</td>
          <td>22.546392</td>
          <td>28.343561</td>
          <td>0.569324</td>
          <td>24.764545</td>
          <td>0.047359</td>
          <td>21.727461</td>
          <td>0.005793</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.402012</td>
          <td>26.406537</td>
          <td>29.177406</td>
          <td>27.412522</td>
          <td>28.037283</td>
          <td>29.470755</td>
          <td>22.609965</td>
          <td>0.006276</td>
          <td>22.520027</td>
          <td>0.007860</td>
          <td>24.740150</td>
          <td>0.046340</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.296783</td>
          <td>23.430518</td>
          <td>21.442559</td>
          <td>22.826418</td>
          <td>21.306559</td>
          <td>21.802722</td>
          <td>21.906980</td>
          <td>0.005380</td>
          <td>21.468929</td>
          <td>0.005506</td>
          <td>20.674099</td>
          <td>0.005121</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.130794</td>
          <td>23.897564</td>
          <td>21.094488</td>
          <td>21.804194</td>
          <td>20.956197</td>
          <td>25.827833</td>
          <td>24.902597</td>
          <td>0.031401</td>
          <td>15.852552</td>
          <td>0.005000</td>
          <td>24.666243</td>
          <td>0.043386</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.058625</td>
          <td>21.976673</td>
          <td>25.371894</td>
          <td>20.515684</td>
          <td>26.334782</td>
          <td>21.265713</td>
          <td>19.802797</td>
          <td>0.005008</td>
          <td>27.129507</td>
          <td>0.358800</td>
          <td>22.887830</td>
          <td>0.009864</td>
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
          <td>23.637106</td>
          <td>25.841099</td>
          <td>21.242693</td>
          <td>26.713129</td>
          <td>25.015178</td>
          <td>26.048039</td>
          <td>20.375871</td>
          <td>0.005023</td>
          <td>21.171595</td>
          <td>0.005298</td>
          <td>22.860752</td>
          <td>0.009685</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.291902</td>
          <td>22.846186</td>
          <td>22.114126</td>
          <td>18.849951</td>
          <td>24.891017</td>
          <td>18.911009</td>
          <td>20.534963</td>
          <td>0.005031</td>
          <td>26.517905</td>
          <td>0.218557</td>
          <td>24.526642</td>
          <td>0.038313</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.357789</td>
          <td>26.673068</td>
          <td>24.757783</td>
          <td>23.920864</td>
          <td>28.567766</td>
          <td>22.678069</td>
          <td>21.888043</td>
          <td>0.005367</td>
          <td>23.193641</td>
          <td>0.012318</td>
          <td>17.061841</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.388636</td>
          <td>25.686051</td>
          <td>23.942786</td>
          <td>22.371048</td>
          <td>19.569141</td>
          <td>23.082648</td>
          <td>21.286594</td>
          <td>0.005124</td>
          <td>24.684313</td>
          <td>0.044090</td>
          <td>22.327848</td>
          <td>0.007129</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.522213</td>
          <td>23.339647</td>
          <td>14.147923</td>
          <td>25.393621</td>
          <td>23.376262</td>
          <td>26.810312</td>
          <td>20.039562</td>
          <td>0.005013</td>
          <td>22.794157</td>
          <td>0.009267</td>
          <td>24.661106</td>
          <td>0.043188</td>
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
          <td>24.186491</td>
          <td>19.913621</td>
          <td>18.692975</td>
          <td>20.278699</td>
          <td>25.033284</td>
          <td>22.546392</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.975081</td>
          <td>0.293030</td>
          <td>21.720907</td>
          <td>0.018917</td>
        </tr>
        <tr>
          <th>1</th>
          <td>25.402012</td>
          <td>26.406537</td>
          <td>29.177406</td>
          <td>27.412522</td>
          <td>28.037283</td>
          <td>29.470755</td>
          <td>22.607687</td>
          <td>0.045018</td>
          <td>22.500614</td>
          <td>0.034252</td>
          <td>25.135781</td>
          <td>0.360569</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.296783</td>
          <td>23.430518</td>
          <td>21.442559</td>
          <td>22.826418</td>
          <td>21.306559</td>
          <td>21.802722</td>
          <td>21.870060</td>
          <td>0.023443</td>
          <td>21.463811</td>
          <td>0.014076</td>
          <td>20.678282</td>
          <td>0.008614</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.130794</td>
          <td>23.897564</td>
          <td>21.094488</td>
          <td>21.804194</td>
          <td>20.956197</td>
          <td>25.827833</td>
          <td>25.168028</td>
          <td>0.399596</td>
          <td>15.843086</td>
          <td>0.005001</td>
          <td>24.920887</td>
          <td>0.304038</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.058625</td>
          <td>21.976673</td>
          <td>25.371894</td>
          <td>20.515684</td>
          <td>26.334782</td>
          <td>21.265713</td>
          <td>19.816810</td>
          <td>0.006093</td>
          <td>27.723861</td>
          <td>1.724516</td>
          <td>22.925981</td>
          <td>0.054689</td>
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
          <td>23.637106</td>
          <td>25.841099</td>
          <td>21.242693</td>
          <td>26.713129</td>
          <td>25.015178</td>
          <td>26.048039</td>
          <td>20.386855</td>
          <td>0.007720</td>
          <td>21.171447</td>
          <td>0.011238</td>
          <td>22.823301</td>
          <td>0.049906</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.291902</td>
          <td>22.846186</td>
          <td>22.114126</td>
          <td>18.849951</td>
          <td>24.891017</td>
          <td>18.911009</td>
          <td>20.531356</td>
          <td>0.008375</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.365163</td>
          <td>0.192274</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.357789</td>
          <td>26.673068</td>
          <td>24.757783</td>
          <td>23.920864</td>
          <td>28.567766</td>
          <td>22.678069</td>
          <td>21.887205</td>
          <td>0.023796</td>
          <td>23.190764</td>
          <td>0.063331</td>
          <td>17.065607</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>998</th>
          <td>20.388636</td>
          <td>25.686051</td>
          <td>23.942786</td>
          <td>22.371048</td>
          <td>19.569141</td>
          <td>23.082648</td>
          <td>21.300425</td>
          <td>0.014497</td>
          <td>24.414796</td>
          <td>0.184262</td>
          <td>22.331339</td>
          <td>0.032210</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.522213</td>
          <td>23.339647</td>
          <td>14.147923</td>
          <td>25.393621</td>
          <td>23.376262</td>
          <td>26.810312</td>
          <td>20.038113</td>
          <td>0.006574</td>
          <td>22.798766</td>
          <td>0.044662</td>
          <td>24.804694</td>
          <td>0.276792</td>
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


