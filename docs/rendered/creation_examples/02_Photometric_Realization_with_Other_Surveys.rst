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
          <td>24.100446</td>
          <td>19.905336</td>
          <td>23.594533</td>
          <td>25.034824</td>
          <td>20.785985</td>
          <td>19.288284</td>
          <td>22.372163</td>
          <td>20.942048</td>
          <td>21.746563</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.480967</td>
          <td>17.858589</td>
          <td>23.487026</td>
          <td>23.007471</td>
          <td>20.947907</td>
          <td>27.348770</td>
          <td>19.489932</td>
          <td>26.368648</td>
          <td>24.043305</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.088229</td>
          <td>21.919528</td>
          <td>24.182359</td>
          <td>24.933246</td>
          <td>16.812712</td>
          <td>23.117611</td>
          <td>21.305680</td>
          <td>20.735082</td>
          <td>25.207272</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.004085</td>
          <td>25.856245</td>
          <td>17.958055</td>
          <td>26.546051</td>
          <td>23.184812</td>
          <td>23.573557</td>
          <td>22.779656</td>
          <td>22.311070</td>
          <td>17.979740</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.655367</td>
          <td>24.720750</td>
          <td>22.415976</td>
          <td>22.484733</td>
          <td>26.141921</td>
          <td>24.054413</td>
          <td>27.985632</td>
          <td>19.965725</td>
          <td>17.202283</td>
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
          <td>21.491006</td>
          <td>25.791433</td>
          <td>20.360908</td>
          <td>26.775844</td>
          <td>29.173080</td>
          <td>20.008535</td>
          <td>23.959580</td>
          <td>23.817751</td>
          <td>25.513012</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.875834</td>
          <td>19.536389</td>
          <td>20.995500</td>
          <td>22.942333</td>
          <td>17.173846</td>
          <td>21.120722</td>
          <td>22.567585</td>
          <td>29.638452</td>
          <td>18.814553</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.414991</td>
          <td>27.761304</td>
          <td>23.489573</td>
          <td>26.221483</td>
          <td>22.934067</td>
          <td>22.867411</td>
          <td>21.499894</td>
          <td>30.895568</td>
          <td>23.276434</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.242919</td>
          <td>23.715567</td>
          <td>23.855010</td>
          <td>25.147450</td>
          <td>23.356042</td>
          <td>27.594353</td>
          <td>23.124822</td>
          <td>19.871579</td>
          <td>27.204339</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.429103</td>
          <td>27.489596</td>
          <td>23.292829</td>
          <td>19.489592</td>
          <td>19.378145</td>
          <td>26.381471</td>
          <td>23.930537</td>
          <td>25.067013</td>
          <td>23.965603</td>
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
          <td>24.212038</td>
          <td>0.054255</td>
          <td>19.901335</td>
          <td>0.005034</td>
          <td>23.604417</td>
          <td>0.010397</td>
          <td>25.170990</td>
          <td>0.061786</td>
          <td>20.786512</td>
          <td>0.005539</td>
          <td>19.281043</td>
          <td>0.005211</td>
          <td>22.372163</td>
          <td>20.942048</td>
          <td>21.746563</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.480187</td>
          <td>0.005144</td>
          <td>17.853860</td>
          <td>0.005004</td>
          <td>23.481323</td>
          <td>0.009572</td>
          <td>23.001421</td>
          <td>0.010049</td>
          <td>20.946901</td>
          <td>0.005697</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.489932</td>
          <td>26.368648</td>
          <td>24.043305</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.088298</td>
          <td>0.009746</td>
          <td>21.914477</td>
          <td>0.005572</td>
          <td>24.192289</td>
          <td>0.016270</td>
          <td>24.939623</td>
          <td>0.050315</td>
          <td>16.813490</td>
          <td>0.005003</td>
          <td>23.086077</td>
          <td>0.042090</td>
          <td>21.305680</td>
          <td>20.735082</td>
          <td>25.207272</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.527809</td>
          <td>0.385246</td>
          <td>25.781588</td>
          <td>0.073923</td>
          <td>17.949469</td>
          <td>0.005002</td>
          <td>26.683192</td>
          <td>0.228738</td>
          <td>23.178062</td>
          <td>0.020432</td>
          <td>23.611420</td>
          <td>0.067081</td>
          <td>22.779656</td>
          <td>22.311070</td>
          <td>17.979740</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.681674</td>
          <td>0.081990</td>
          <td>24.716336</td>
          <td>0.028868</td>
          <td>22.401539</td>
          <td>0.005908</td>
          <td>22.493689</td>
          <td>0.007462</td>
          <td>26.838342</td>
          <td>0.462453</td>
          <td>24.253736</td>
          <td>0.117980</td>
          <td>27.985632</td>
          <td>19.965725</td>
          <td>17.202283</td>
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
          <td>21.485058</td>
          <td>0.007121</td>
          <td>25.880326</td>
          <td>0.080649</td>
          <td>20.365743</td>
          <td>0.005039</td>
          <td>26.431481</td>
          <td>0.185253</td>
          <td>inf</td>
          <td>inf</td>
          <td>20.011064</td>
          <td>0.005680</td>
          <td>23.959580</td>
          <td>23.817751</td>
          <td>25.513012</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.760438</td>
          <td>0.087848</td>
          <td>19.525006</td>
          <td>0.005021</td>
          <td>20.996952</td>
          <td>0.005097</td>
          <td>22.958061</td>
          <td>0.009762</td>
          <td>17.175079</td>
          <td>0.005004</td>
          <td>21.122478</td>
          <td>0.008707</td>
          <td>22.567585</td>
          <td>29.638452</td>
          <td>18.814553</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.718775</td>
          <td>0.200261</td>
          <td>27.533746</td>
          <td>0.327333</td>
          <td>23.496925</td>
          <td>0.009670</td>
          <td>26.276051</td>
          <td>0.162334</td>
          <td>22.938192</td>
          <td>0.016716</td>
          <td>22.888867</td>
          <td>0.035349</td>
          <td>21.499894</td>
          <td>30.895568</td>
          <td>23.276434</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.247514</td>
          <td>0.005109</td>
          <td>23.715125</td>
          <td>0.012617</td>
          <td>23.842730</td>
          <td>0.012352</td>
          <td>25.077948</td>
          <td>0.056890</td>
          <td>23.344250</td>
          <td>0.023557</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.124822</td>
          <td>19.871579</td>
          <td>27.204339</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.432941</td>
          <td>0.005492</td>
          <td>27.126882</td>
          <td>0.235318</td>
          <td>23.309382</td>
          <td>0.008601</td>
          <td>19.485217</td>
          <td>0.005025</td>
          <td>19.379782</td>
          <td>0.005061</td>
          <td>26.641507</td>
          <td>0.770198</td>
          <td>23.930537</td>
          <td>25.067013</td>
          <td>23.965603</td>
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
          <td>24.100446</td>
          <td>19.905336</td>
          <td>23.594533</td>
          <td>25.034824</td>
          <td>20.785985</td>
          <td>19.288284</td>
          <td>22.372913</td>
          <td>0.005857</td>
          <td>20.941169</td>
          <td>0.005197</td>
          <td>21.739700</td>
          <td>0.005809</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.480967</td>
          <td>17.858589</td>
          <td>23.487026</td>
          <td>23.007471</td>
          <td>20.947907</td>
          <td>27.348770</td>
          <td>19.489807</td>
          <td>0.005005</td>
          <td>26.300488</td>
          <td>0.182042</td>
          <td>24.059030</td>
          <td>0.025337</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.088229</td>
          <td>21.919528</td>
          <td>24.182359</td>
          <td>24.933246</td>
          <td>16.812712</td>
          <td>23.117611</td>
          <td>21.305829</td>
          <td>0.005129</td>
          <td>20.738315</td>
          <td>0.005137</td>
          <td>25.323738</td>
          <td>0.077879</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.004085</td>
          <td>25.856245</td>
          <td>17.958055</td>
          <td>26.546051</td>
          <td>23.184812</td>
          <td>23.573557</td>
          <td>22.773416</td>
          <td>0.006666</td>
          <td>22.314334</td>
          <td>0.007085</td>
          <td>17.989514</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.655367</td>
          <td>24.720750</td>
          <td>22.415976</td>
          <td>22.484733</td>
          <td>26.141921</td>
          <td>24.054413</td>
          <td>28.891845</td>
          <td>0.827189</td>
          <td>19.966304</td>
          <td>0.005033</td>
          <td>17.201457</td>
          <td>0.005000</td>
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
          <td>21.491006</td>
          <td>25.791433</td>
          <td>20.360908</td>
          <td>26.775844</td>
          <td>29.173080</td>
          <td>20.008535</td>
          <td>23.976968</td>
          <td>0.014226</td>
          <td>23.803151</td>
          <td>0.020293</td>
          <td>25.606496</td>
          <td>0.099934</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.875834</td>
          <td>19.536389</td>
          <td>20.995500</td>
          <td>22.942333</td>
          <td>17.173846</td>
          <td>21.120722</td>
          <td>22.559627</td>
          <td>0.006174</td>
          <td>27.337819</td>
          <td>0.421565</td>
          <td>18.811003</td>
          <td>0.005004</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.414991</td>
          <td>27.761304</td>
          <td>23.489573</td>
          <td>26.221483</td>
          <td>22.934067</td>
          <td>22.867411</td>
          <td>21.498981</td>
          <td>0.005183</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.267786</td>
          <td>0.013046</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.242919</td>
          <td>23.715567</td>
          <td>23.855010</td>
          <td>25.147450</td>
          <td>23.356042</td>
          <td>27.594353</td>
          <td>23.120933</td>
          <td>0.007864</td>
          <td>19.871498</td>
          <td>0.005028</td>
          <td>27.850055</td>
          <td>0.614048</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.429103</td>
          <td>27.489596</td>
          <td>23.292829</td>
          <td>19.489592</td>
          <td>19.378145</td>
          <td>26.381471</td>
          <td>23.931073</td>
          <td>0.013713</td>
          <td>25.104715</td>
          <td>0.064122</td>
          <td>23.956185</td>
          <td>0.023162</td>
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
          <td>24.100446</td>
          <td>19.905336</td>
          <td>23.594533</td>
          <td>25.034824</td>
          <td>20.785985</td>
          <td>19.288284</td>
          <td>22.355695</td>
          <td>0.035970</td>
          <td>20.939324</td>
          <td>0.009547</td>
          <td>21.759448</td>
          <td>0.019548</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.480967</td>
          <td>17.858589</td>
          <td>23.487026</td>
          <td>23.007471</td>
          <td>20.947907</td>
          <td>27.348770</td>
          <td>19.483352</td>
          <td>0.005618</td>
          <td>26.362625</td>
          <td>0.811707</td>
          <td>23.901222</td>
          <td>0.129249</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.088229</td>
          <td>21.919528</td>
          <td>24.182359</td>
          <td>24.933246</td>
          <td>16.812712</td>
          <td>23.117611</td>
          <td>21.285763</td>
          <td>0.014326</td>
          <td>20.729031</td>
          <td>0.008363</td>
          <td>25.201567</td>
          <td>0.379564</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.004085</td>
          <td>25.856245</td>
          <td>17.958055</td>
          <td>26.546051</td>
          <td>23.184812</td>
          <td>23.573557</td>
          <td>22.684501</td>
          <td>0.048210</td>
          <td>22.356330</td>
          <td>0.030141</td>
          <td>17.973217</td>
          <td>0.005034</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.655367</td>
          <td>24.720750</td>
          <td>22.415976</td>
          <td>22.484733</td>
          <td>26.141921</td>
          <td>24.054413</td>
          <td>25.469187</td>
          <td>0.501516</td>
          <td>19.960791</td>
          <td>0.005995</td>
          <td>17.198360</td>
          <td>0.005008</td>
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
          <td>21.491006</td>
          <td>25.791433</td>
          <td>20.360908</td>
          <td>26.775844</td>
          <td>29.173080</td>
          <td>20.008535</td>
          <td>24.180434</td>
          <td>0.178972</td>
          <td>23.736640</td>
          <td>0.102612</td>
          <td>25.196979</td>
          <td>0.378213</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.875834</td>
          <td>19.536389</td>
          <td>20.995500</td>
          <td>22.942333</td>
          <td>17.173846</td>
          <td>21.120722</td>
          <td>22.587002</td>
          <td>0.044196</td>
          <td>26.279914</td>
          <td>0.768946</td>
          <td>18.822304</td>
          <td>0.005159</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.414991</td>
          <td>27.761304</td>
          <td>23.489573</td>
          <td>26.221483</td>
          <td>22.934067</td>
          <td>22.867411</td>
          <td>21.481541</td>
          <td>0.016821</td>
          <td>28.080345</td>
          <td>2.017084</td>
          <td>23.300798</td>
          <td>0.076313</td>
        </tr>
        <tr>
          <th>998</th>
          <td>19.242919</td>
          <td>23.715567</td>
          <td>23.855010</td>
          <td>25.147450</td>
          <td>23.356042</td>
          <td>27.594353</td>
          <td>23.061903</td>
          <td>0.067465</td>
          <td>19.868666</td>
          <td>0.005851</td>
          <td>24.992222</td>
          <td>0.321895</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.429103</td>
          <td>27.489596</td>
          <td>23.292829</td>
          <td>19.489592</td>
          <td>19.378145</td>
          <td>26.381471</td>
          <td>23.853354</td>
          <td>0.135218</td>
          <td>25.136328</td>
          <td>0.333381</td>
          <td>23.933144</td>
          <td>0.132874</td>
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


