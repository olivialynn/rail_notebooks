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
          <td>24.179739</td>
          <td>19.145503</td>
          <td>22.032718</td>
          <td>19.941632</td>
          <td>24.421745</td>
          <td>25.964591</td>
          <td>23.494075</td>
          <td>21.515306</td>
          <td>24.662662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.525591</td>
          <td>22.567050</td>
          <td>24.610950</td>
          <td>22.974018</td>
          <td>25.422553</td>
          <td>24.751561</td>
          <td>23.616951</td>
          <td>20.977913</td>
          <td>17.403907</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.178520</td>
          <td>28.162588</td>
          <td>23.407787</td>
          <td>22.810000</td>
          <td>22.356474</td>
          <td>24.962623</td>
          <td>28.736558</td>
          <td>25.462333</td>
          <td>24.507776</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.166400</td>
          <td>22.728455</td>
          <td>28.415328</td>
          <td>20.103498</td>
          <td>23.120516</td>
          <td>24.954302</td>
          <td>18.529963</td>
          <td>22.287135</td>
          <td>20.905399</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.800762</td>
          <td>22.128051</td>
          <td>27.012850</td>
          <td>19.331404</td>
          <td>25.303465</td>
          <td>20.349003</td>
          <td>30.131569</td>
          <td>32.692702</td>
          <td>22.334233</td>
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
          <td>18.243093</td>
          <td>24.794636</td>
          <td>24.029305</td>
          <td>22.734701</td>
          <td>26.307478</td>
          <td>25.042963</td>
          <td>22.356385</td>
          <td>27.477990</td>
          <td>19.622800</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.219318</td>
          <td>22.060144</td>
          <td>21.708501</td>
          <td>23.176541</td>
          <td>26.754285</td>
          <td>25.045414</td>
          <td>22.620936</td>
          <td>22.478305</td>
          <td>25.117861</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.305619</td>
          <td>23.298114</td>
          <td>19.057978</td>
          <td>25.223561</td>
          <td>27.487269</td>
          <td>21.341888</td>
          <td>21.970580</td>
          <td>26.037122</td>
          <td>21.089697</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.064416</td>
          <td>22.165079</td>
          <td>22.925272</td>
          <td>16.239666</td>
          <td>23.596575</td>
          <td>18.070553</td>
          <td>24.123332</td>
          <td>18.120121</td>
          <td>23.788983</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.635458</td>
          <td>23.153663</td>
          <td>23.422726</td>
          <td>22.065585</td>
          <td>22.438893</td>
          <td>23.679711</td>
          <td>23.237487</td>
          <td>21.236754</td>
          <td>32.162383</td>
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
          <td>24.138643</td>
          <td>0.050863</td>
          <td>19.142755</td>
          <td>0.005014</td>
          <td>22.034511</td>
          <td>0.005503</td>
          <td>19.941851</td>
          <td>0.005046</td>
          <td>24.512622</td>
          <td>0.066016</td>
          <td>26.074720</td>
          <td>0.518968</td>
          <td>23.494075</td>
          <td>21.515306</td>
          <td>24.662662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.929548</td>
          <td>0.521330</td>
          <td>22.560707</td>
          <td>0.006540</td>
          <td>24.612518</td>
          <td>0.023182</td>
          <td>22.974906</td>
          <td>0.009872</td>
          <td>25.481768</td>
          <td>0.154034</td>
          <td>24.748368</td>
          <td>0.180464</td>
          <td>23.616951</td>
          <td>20.977913</td>
          <td>17.403907</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.173558</td>
          <td>0.005100</td>
          <td>27.829895</td>
          <td>0.412437</td>
          <td>23.401782</td>
          <td>0.009098</td>
          <td>22.810312</td>
          <td>0.008885</td>
          <td>22.327259</td>
          <td>0.010472</td>
          <td>25.166950</td>
          <td>0.255865</td>
          <td>28.736558</td>
          <td>25.462333</td>
          <td>24.507776</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.172113</td>
          <td>0.022031</td>
          <td>22.720085</td>
          <td>0.006954</td>
          <td>27.875457</td>
          <td>0.382330</td>
          <td>20.097442</td>
          <td>0.005058</td>
          <td>23.092256</td>
          <td>0.019002</td>
          <td>25.339865</td>
          <td>0.294484</td>
          <td>18.529963</td>
          <td>22.287135</td>
          <td>20.905399</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.840596</td>
          <td>0.221683</td>
          <td>22.125698</td>
          <td>0.005792</td>
          <td>26.844946</td>
          <td>0.164470</td>
          <td>19.333720</td>
          <td>0.005020</td>
          <td>25.506671</td>
          <td>0.157354</td>
          <td>20.357357</td>
          <td>0.006179</td>
          <td>30.131569</td>
          <td>32.692702</td>
          <td>22.334233</td>
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
          <td>18.254725</td>
          <td>0.005037</td>
          <td>24.844895</td>
          <td>0.032305</td>
          <td>24.028873</td>
          <td>0.014264</td>
          <td>22.725799</td>
          <td>0.008449</td>
          <td>25.928166</td>
          <td>0.224574</td>
          <td>24.931372</td>
          <td>0.210518</td>
          <td>22.356385</td>
          <td>27.477990</td>
          <td>19.622800</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.226767</td>
          <td>0.005013</td>
          <td>22.058711</td>
          <td>0.005715</td>
          <td>21.702878</td>
          <td>0.005295</td>
          <td>23.169695</td>
          <td>0.011306</td>
          <td>26.833120</td>
          <td>0.460646</td>
          <td>25.148936</td>
          <td>0.252112</td>
          <td>22.620936</td>
          <td>22.478305</td>
          <td>25.117861</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.310232</td>
          <td>0.005417</td>
          <td>23.316966</td>
          <td>0.009550</td>
          <td>19.066870</td>
          <td>0.005007</td>
          <td>25.183537</td>
          <td>0.062478</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.350830</td>
          <td>0.010071</td>
          <td>21.970580</td>
          <td>26.037122</td>
          <td>21.089697</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.087496</td>
          <td>0.048626</td>
          <td>22.165837</td>
          <td>0.005843</td>
          <td>22.928204</td>
          <td>0.007062</td>
          <td>16.248125</td>
          <td>0.005001</td>
          <td>23.612039</td>
          <td>0.029742</td>
          <td>18.074162</td>
          <td>0.005035</td>
          <td>24.123332</td>
          <td>18.120121</td>
          <td>23.788983</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.329750</td>
          <td>0.143965</td>
          <td>23.140980</td>
          <td>0.008577</td>
          <td>23.419579</td>
          <td>0.009201</td>
          <td>22.073882</td>
          <td>0.006296</td>
          <td>22.440215</td>
          <td>0.011344</td>
          <td>23.619812</td>
          <td>0.067581</td>
          <td>23.237487</td>
          <td>21.236754</td>
          <td>32.162383</td>
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
          <td>24.179739</td>
          <td>19.145503</td>
          <td>22.032718</td>
          <td>19.941632</td>
          <td>24.421745</td>
          <td>25.964591</td>
          <td>23.500120</td>
          <td>0.009948</td>
          <td>21.517374</td>
          <td>0.005551</td>
          <td>24.651391</td>
          <td>0.042815</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.525591</td>
          <td>22.567050</td>
          <td>24.610950</td>
          <td>22.974018</td>
          <td>25.422553</td>
          <td>24.751561</td>
          <td>23.631390</td>
          <td>0.010914</td>
          <td>20.990215</td>
          <td>0.005215</td>
          <td>17.399253</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.178520</td>
          <td>28.162588</td>
          <td>23.407787</td>
          <td>22.810000</td>
          <td>22.356474</td>
          <td>24.962623</td>
          <td>29.673202</td>
          <td>1.311286</td>
          <td>25.396523</td>
          <td>0.083058</td>
          <td>24.552953</td>
          <td>0.039221</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.166400</td>
          <td>22.728455</td>
          <td>28.415328</td>
          <td>20.103498</td>
          <td>23.120516</td>
          <td>24.954302</td>
          <td>18.537035</td>
          <td>0.005001</td>
          <td>22.287798</td>
          <td>0.007000</td>
          <td>20.908610</td>
          <td>0.005186</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.800762</td>
          <td>22.128051</td>
          <td>27.012850</td>
          <td>19.331404</td>
          <td>25.303465</td>
          <td>20.349003</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.825362</td>
          <td>1.143621</td>
          <td>22.341397</td>
          <td>0.007175</td>
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
          <td>18.243093</td>
          <td>24.794636</td>
          <td>24.029305</td>
          <td>22.734701</td>
          <td>26.307478</td>
          <td>25.042963</td>
          <td>22.356457</td>
          <td>0.005833</td>
          <td>27.338846</td>
          <td>0.421896</td>
          <td>19.624990</td>
          <td>0.005018</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.219318</td>
          <td>22.060144</td>
          <td>21.708501</td>
          <td>23.176541</td>
          <td>26.754285</td>
          <td>25.045414</td>
          <td>22.626899</td>
          <td>0.006312</td>
          <td>22.487161</td>
          <td>0.007722</td>
          <td>25.152719</td>
          <td>0.066917</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.305619</td>
          <td>23.298114</td>
          <td>19.057978</td>
          <td>25.223561</td>
          <td>27.487269</td>
          <td>21.341888</td>
          <td>21.963399</td>
          <td>0.005420</td>
          <td>26.170816</td>
          <td>0.163021</td>
          <td>21.088182</td>
          <td>0.005257</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.064416</td>
          <td>22.165079</td>
          <td>22.925272</td>
          <td>16.239666</td>
          <td>23.596575</td>
          <td>18.070553</td>
          <td>24.125849</td>
          <td>0.016061</td>
          <td>18.121279</td>
          <td>0.005001</td>
          <td>23.771472</td>
          <td>0.019750</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.635458</td>
          <td>23.153663</td>
          <td>23.422726</td>
          <td>22.065585</td>
          <td>22.438893</td>
          <td>23.679711</td>
          <td>23.244533</td>
          <td>0.008440</td>
          <td>21.238117</td>
          <td>0.005336</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>24.179739</td>
          <td>19.145503</td>
          <td>22.032718</td>
          <td>19.941632</td>
          <td>24.421745</td>
          <td>25.964591</td>
          <td>23.577589</td>
          <td>0.106361</td>
          <td>21.523794</td>
          <td>0.014773</td>
          <td>24.550877</td>
          <td>0.224642</td>
        </tr>
        <tr>
          <th>1</th>
          <td>28.525591</td>
          <td>22.567050</td>
          <td>24.610950</td>
          <td>22.974018</td>
          <td>25.422553</td>
          <td>24.751561</td>
          <td>23.461304</td>
          <td>0.096043</td>
          <td>20.983903</td>
          <td>0.009838</td>
          <td>17.402227</td>
          <td>0.005012</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.178520</td>
          <td>28.162588</td>
          <td>23.407787</td>
          <td>22.810000</td>
          <td>22.356474</td>
          <td>24.962623</td>
          <td>27.993750</td>
          <td>2.113557</td>
          <td>27.096189</td>
          <td>1.257868</td>
          <td>24.511074</td>
          <td>0.217315</td>
        </tr>
        <tr>
          <th>3</th>
          <td>23.166400</td>
          <td>22.728455</td>
          <td>28.415328</td>
          <td>20.103498</td>
          <td>23.120516</td>
          <td>24.954302</td>
          <td>18.537559</td>
          <td>0.005114</td>
          <td>22.289060</td>
          <td>0.028404</td>
          <td>20.890542</td>
          <td>0.009882</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.800762</td>
          <td>22.128051</td>
          <td>27.012850</td>
          <td>19.331404</td>
          <td>25.303465</td>
          <td>20.349003</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.506174</td>
          <td>0.889643</td>
          <td>22.270702</td>
          <td>0.030527</td>
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
          <td>18.243093</td>
          <td>24.794636</td>
          <td>24.029305</td>
          <td>22.734701</td>
          <td>26.307478</td>
          <td>25.042963</td>
          <td>22.419640</td>
          <td>0.038075</td>
          <td>27.456963</td>
          <td>1.517726</td>
          <td>19.629646</td>
          <td>0.005670</td>
        </tr>
        <tr>
          <th>996</th>
          <td>17.219318</td>
          <td>22.060144</td>
          <td>21.708501</td>
          <td>23.176541</td>
          <td>26.754285</td>
          <td>25.045414</td>
          <td>22.668834</td>
          <td>0.047541</td>
          <td>22.466019</td>
          <td>0.033216</td>
          <td>25.083164</td>
          <td>0.345960</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.305619</td>
          <td>23.298114</td>
          <td>19.057978</td>
          <td>25.223561</td>
          <td>27.487269</td>
          <td>21.341888</td>
          <td>21.921976</td>
          <td>0.024529</td>
          <td>25.213532</td>
          <td>0.354329</td>
          <td>21.090995</td>
          <td>0.011401</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.064416</td>
          <td>22.165079</td>
          <td>22.925272</td>
          <td>16.239666</td>
          <td>23.596575</td>
          <td>18.070553</td>
          <td>23.998335</td>
          <td>0.153210</td>
          <td>18.127791</td>
          <td>0.005037</td>
          <td>24.185610</td>
          <td>0.165095</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.635458</td>
          <td>23.153663</td>
          <td>23.422726</td>
          <td>22.065585</td>
          <td>22.438893</td>
          <td>23.679711</td>
          <td>23.238970</td>
          <td>0.078936</td>
          <td>21.236487</td>
          <td>0.011795</td>
          <td>30.557746</td>
          <td>4.428851</td>
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


