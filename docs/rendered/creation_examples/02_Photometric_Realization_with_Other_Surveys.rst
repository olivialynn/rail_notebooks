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
          <td>20.873523</td>
          <td>25.460103</td>
          <td>20.089732</td>
          <td>19.332700</td>
          <td>26.468729</td>
          <td>28.648021</td>
          <td>21.840935</td>
          <td>29.222761</td>
          <td>25.700447</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.359098</td>
          <td>22.553829</td>
          <td>29.893376</td>
          <td>19.545191</td>
          <td>24.198076</td>
          <td>22.574881</td>
          <td>26.181609</td>
          <td>30.805225</td>
          <td>24.217369</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.633372</td>
          <td>23.330198</td>
          <td>25.505726</td>
          <td>26.440446</td>
          <td>28.811293</td>
          <td>22.253757</td>
          <td>27.280288</td>
          <td>22.084448</td>
          <td>22.299883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.310867</td>
          <td>22.805613</td>
          <td>24.645564</td>
          <td>27.205760</td>
          <td>21.625565</td>
          <td>25.516542</td>
          <td>22.613656</td>
          <td>28.437145</td>
          <td>21.819325</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.560443</td>
          <td>19.957627</td>
          <td>27.374256</td>
          <td>28.865016</td>
          <td>17.590282</td>
          <td>23.516059</td>
          <td>29.058795</td>
          <td>24.439937</td>
          <td>25.338432</td>
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
          <td>24.348570</td>
          <td>23.045860</td>
          <td>25.109447</td>
          <td>16.997793</td>
          <td>20.145465</td>
          <td>25.231901</td>
          <td>18.572010</td>
          <td>22.582896</td>
          <td>20.178628</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.507369</td>
          <td>25.438010</td>
          <td>24.246669</td>
          <td>21.793258</td>
          <td>29.956785</td>
          <td>25.405863</td>
          <td>21.268483</td>
          <td>25.993362</td>
          <td>21.731000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.521144</td>
          <td>22.425483</td>
          <td>26.085372</td>
          <td>27.946877</td>
          <td>24.123125</td>
          <td>23.218494</td>
          <td>24.856384</td>
          <td>22.515372</td>
          <td>19.352889</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.900486</td>
          <td>22.039579</td>
          <td>21.498867</td>
          <td>20.183446</td>
          <td>17.838134</td>
          <td>18.906066</td>
          <td>20.103339</td>
          <td>25.020461</td>
          <td>22.050518</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.006991</td>
          <td>24.672507</td>
          <td>19.904841</td>
          <td>27.098418</td>
          <td>26.450335</td>
          <td>22.552894</td>
          <td>29.521847</td>
          <td>21.668173</td>
          <td>21.848717</td>
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
          <td>20.881432</td>
          <td>0.005914</td>
          <td>25.436086</td>
          <td>0.054455</td>
          <td>20.083518</td>
          <td>0.005026</td>
          <td>19.339463</td>
          <td>0.005020</td>
          <td>26.471333</td>
          <td>0.348739</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.840935</td>
          <td>29.222761</td>
          <td>25.700447</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.357111</td>
          <td>0.005444</td>
          <td>22.551962</td>
          <td>0.006520</td>
          <td>30.103582</td>
          <td>1.579766</td>
          <td>19.546781</td>
          <td>0.005027</td>
          <td>24.183775</td>
          <td>0.049310</td>
          <td>22.563107</td>
          <td>0.026556</td>
          <td>26.181609</td>
          <td>30.805225</td>
          <td>24.217369</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.590079</td>
          <td>0.075658</td>
          <td>23.322500</td>
          <td>0.009584</td>
          <td>25.391360</td>
          <td>0.045967</td>
          <td>26.412316</td>
          <td>0.182274</td>
          <td>27.423351</td>
          <td>0.702579</td>
          <td>22.287450</td>
          <td>0.020937</td>
          <td>27.280288</td>
          <td>22.084448</td>
          <td>22.299883</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.306755</td>
          <td>0.005415</td>
          <td>22.803311</td>
          <td>0.007209</td>
          <td>24.602161</td>
          <td>0.022976</td>
          <td>27.383066</td>
          <td>0.400849</td>
          <td>21.619188</td>
          <td>0.006999</td>
          <td>25.015752</td>
          <td>0.225853</td>
          <td>22.613656</td>
          <td>28.437145</td>
          <td>21.819325</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.561005</td>
          <td>0.007355</td>
          <td>19.966957</td>
          <td>0.005036</td>
          <td>27.174200</td>
          <td>0.217138</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.590982</td>
          <td>0.005006</td>
          <td>23.517214</td>
          <td>0.061708</td>
          <td>29.058795</td>
          <td>24.439937</td>
          <td>25.338432</td>
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
          <td>24.375406</td>
          <td>0.062645</td>
          <td>23.043264</td>
          <td>0.008120</td>
          <td>25.104460</td>
          <td>0.035649</td>
          <td>16.997319</td>
          <td>0.005001</td>
          <td>20.141777</td>
          <td>0.005192</td>
          <td>25.909347</td>
          <td>0.459083</td>
          <td>18.572010</td>
          <td>22.582896</td>
          <td>20.178628</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.607225</td>
          <td>0.076806</td>
          <td>25.488013</td>
          <td>0.057018</td>
          <td>24.247691</td>
          <td>0.017028</td>
          <td>21.790268</td>
          <td>0.005826</td>
          <td>27.855280</td>
          <td>0.929742</td>
          <td>26.067281</td>
          <td>0.516148</td>
          <td>21.268483</td>
          <td>25.993362</td>
          <td>21.731000</td>
        </tr>
        <tr>
          <th>997</th>
          <td>26.776042</td>
          <td>0.465388</td>
          <td>22.428714</td>
          <td>0.006261</td>
          <td>26.213672</td>
          <td>0.095167</td>
          <td>29.918149</td>
          <td>1.872484</td>
          <td>24.098071</td>
          <td>0.045698</td>
          <td>23.151515</td>
          <td>0.044605</td>
          <td>24.856384</td>
          <td>22.515372</td>
          <td>19.352889</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.995196</td>
          <td>0.251846</td>
          <td>22.030133</td>
          <td>0.005684</td>
          <td>21.495374</td>
          <td>0.005212</td>
          <td>20.178289</td>
          <td>0.005065</td>
          <td>17.836941</td>
          <td>0.005008</td>
          <td>18.897627</td>
          <td>0.005117</td>
          <td>20.103339</td>
          <td>25.020461</td>
          <td>22.050518</td>
        </tr>
        <tr>
          <th>999</th>
          <td>23.953304</td>
          <td>0.043220</td>
          <td>24.635350</td>
          <td>0.026904</td>
          <td>19.900163</td>
          <td>0.005021</td>
          <td>28.397264</td>
          <td>0.824086</td>
          <td>26.361254</td>
          <td>0.319612</td>
          <td>22.538445</td>
          <td>0.025992</td>
          <td>29.521847</td>
          <td>21.668173</td>
          <td>21.848717</td>
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
          <td>20.873523</td>
          <td>25.460103</td>
          <td>20.089732</td>
          <td>19.332700</td>
          <td>26.468729</td>
          <td>28.648021</td>
          <td>21.840013</td>
          <td>0.005337</td>
          <td>28.992832</td>
          <td>1.255566</td>
          <td>25.738297</td>
          <td>0.112160</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.359098</td>
          <td>22.553829</td>
          <td>29.893376</td>
          <td>19.545191</td>
          <td>24.198076</td>
          <td>22.574881</td>
          <td>26.211475</td>
          <td>0.100371</td>
          <td>28.297700</td>
          <td>0.830315</td>
          <td>24.199636</td>
          <td>0.028670</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.633372</td>
          <td>23.330198</td>
          <td>25.505726</td>
          <td>26.440446</td>
          <td>28.811293</td>
          <td>22.253757</td>
          <td>27.386042</td>
          <td>0.272623</td>
          <td>22.086462</td>
          <td>0.006447</td>
          <td>22.303344</td>
          <td>0.007049</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.310867</td>
          <td>22.805613</td>
          <td>24.645564</td>
          <td>27.205760</td>
          <td>21.625565</td>
          <td>25.516542</td>
          <td>22.602550</td>
          <td>0.006261</td>
          <td>27.395868</td>
          <td>0.440583</td>
          <td>21.819994</td>
          <td>0.005928</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.560443</td>
          <td>19.957627</td>
          <td>27.374256</td>
          <td>28.865016</td>
          <td>17.590282</td>
          <td>23.516059</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.454258</td>
          <td>0.035924</td>
          <td>25.218422</td>
          <td>0.070937</td>
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
          <td>24.348570</td>
          <td>23.045860</td>
          <td>25.109447</td>
          <td>16.997793</td>
          <td>20.145465</td>
          <td>25.231901</td>
          <td>18.570368</td>
          <td>0.005001</td>
          <td>22.588919</td>
          <td>0.008170</td>
          <td>20.183993</td>
          <td>0.005050</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.507369</td>
          <td>25.438010</td>
          <td>24.246669</td>
          <td>21.793258</td>
          <td>29.956785</td>
          <td>25.405863</td>
          <td>21.268493</td>
          <td>0.005120</td>
          <td>26.004510</td>
          <td>0.141327</td>
          <td>21.722407</td>
          <td>0.005786</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.521144</td>
          <td>22.425483</td>
          <td>26.085372</td>
          <td>27.946877</td>
          <td>24.123125</td>
          <td>23.218494</td>
          <td>24.822759</td>
          <td>0.029261</td>
          <td>22.512892</td>
          <td>0.007829</td>
          <td>19.344267</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.900486</td>
          <td>22.039579</td>
          <td>21.498867</td>
          <td>20.183446</td>
          <td>17.838134</td>
          <td>18.906066</td>
          <td>20.099661</td>
          <td>0.005014</td>
          <td>25.041216</td>
          <td>0.060600</td>
          <td>22.055264</td>
          <td>0.006375</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.006991</td>
          <td>24.672507</td>
          <td>19.904841</td>
          <td>27.098418</td>
          <td>26.450335</td>
          <td>22.552894</td>
          <td>27.921041</td>
          <td>0.416195</td>
          <td>21.667242</td>
          <td>0.005715</td>
          <td>21.843994</td>
          <td>0.005967</td>
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
          <td>20.873523</td>
          <td>25.460103</td>
          <td>20.089732</td>
          <td>19.332700</td>
          <td>26.468729</td>
          <td>28.648021</td>
          <td>21.866172</td>
          <td>0.023364</td>
          <td>26.646811</td>
          <td>0.970533</td>
          <td>25.327087</td>
          <td>0.418124</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.359098</td>
          <td>22.553829</td>
          <td>29.893376</td>
          <td>19.545191</td>
          <td>24.198076</td>
          <td>22.574881</td>
          <td>25.777530</td>
          <td>0.626001</td>
          <td>25.781180</td>
          <td>0.544300</td>
          <td>24.324356</td>
          <td>0.185758</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.633372</td>
          <td>23.330198</td>
          <td>25.505726</td>
          <td>26.440446</td>
          <td>28.811293</td>
          <td>22.253757</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.098449</td>
          <td>0.024030</td>
          <td>22.290501</td>
          <td>0.031066</td>
        </tr>
        <tr>
          <th>3</th>
          <td>20.310867</td>
          <td>22.805613</td>
          <td>24.645564</td>
          <td>27.205760</td>
          <td>21.625565</td>
          <td>25.516542</td>
          <td>22.646893</td>
          <td>0.046620</td>
          <td>25.314414</td>
          <td>0.383370</td>
          <td>21.816833</td>
          <td>0.020533</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.560443</td>
          <td>19.957627</td>
          <td>27.374256</td>
          <td>28.865016</td>
          <td>17.590282</td>
          <td>23.516059</td>
          <td>26.129298</td>
          <td>0.794288</td>
          <td>24.185862</td>
          <td>0.151578</td>
          <td>24.758424</td>
          <td>0.266553</td>
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
          <td>24.348570</td>
          <td>23.045860</td>
          <td>25.109447</td>
          <td>16.997793</td>
          <td>20.145465</td>
          <td>25.231901</td>
          <td>18.571033</td>
          <td>0.005121</td>
          <td>22.537831</td>
          <td>0.035403</td>
          <td>20.193306</td>
          <td>0.006721</td>
        </tr>
        <tr>
          <th>996</th>
          <td>24.507369</td>
          <td>25.438010</td>
          <td>24.246669</td>
          <td>21.793258</td>
          <td>29.956785</td>
          <td>25.405863</td>
          <td>21.256979</td>
          <td>0.014000</td>
          <td>25.112682</td>
          <td>0.327180</td>
          <td>21.763800</td>
          <td>0.019621</td>
        </tr>
        <tr>
          <th>997</th>
          <td>27.521144</td>
          <td>22.425483</td>
          <td>26.085372</td>
          <td>27.946877</td>
          <td>24.123125</td>
          <td>23.218494</td>
          <td>24.797100</td>
          <td>0.298277</td>
          <td>22.536299</td>
          <td>0.035355</td>
          <td>19.353058</td>
          <td>0.005412</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.900486</td>
          <td>22.039579</td>
          <td>21.498867</td>
          <td>20.183446</td>
          <td>17.838134</td>
          <td>18.906066</td>
          <td>20.109372</td>
          <td>0.006765</td>
          <td>25.049249</td>
          <td>0.311032</td>
          <td>22.039405</td>
          <td>0.024905</td>
        </tr>
        <tr>
          <th>999</th>
          <td>24.006991</td>
          <td>24.672507</td>
          <td>19.904841</td>
          <td>27.098418</td>
          <td>26.450335</td>
          <td>22.552894</td>
          <td>28.717400</td>
          <td>2.759058</td>
          <td>21.666104</td>
          <td>0.016606</td>
          <td>21.833443</td>
          <td>0.020828</td>
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


