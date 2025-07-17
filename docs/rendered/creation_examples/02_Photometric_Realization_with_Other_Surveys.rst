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
          <td>22.924421</td>
          <td>26.623145</td>
          <td>27.394586</td>
          <td>25.305398</td>
          <td>24.927602</td>
          <td>22.374212</td>
          <td>22.157869</td>
          <td>23.830213</td>
          <td>32.020728</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.969920</td>
          <td>25.268529</td>
          <td>25.569740</td>
          <td>27.877423</td>
          <td>22.157410</td>
          <td>21.795464</td>
          <td>26.496968</td>
          <td>25.922360</td>
          <td>24.376712</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.263006</td>
          <td>21.980421</td>
          <td>23.784552</td>
          <td>23.168203</td>
          <td>21.190007</td>
          <td>28.171118</td>
          <td>21.306932</td>
          <td>17.829858</td>
          <td>23.622871</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.901991</td>
          <td>25.456661</td>
          <td>18.299273</td>
          <td>24.347278</td>
          <td>18.826936</td>
          <td>23.330937</td>
          <td>20.965424</td>
          <td>27.517862</td>
          <td>17.510298</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.598138</td>
          <td>26.626787</td>
          <td>18.591515</td>
          <td>21.119736</td>
          <td>21.997115</td>
          <td>21.655542</td>
          <td>19.797726</td>
          <td>27.272744</td>
          <td>27.031754</td>
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
          <td>22.926140</td>
          <td>23.611894</td>
          <td>19.308360</td>
          <td>21.517998</td>
          <td>27.563088</td>
          <td>24.956842</td>
          <td>17.342274</td>
          <td>22.261844</td>
          <td>22.257695</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.297483</td>
          <td>21.388896</td>
          <td>25.769455</td>
          <td>26.412470</td>
          <td>21.458691</td>
          <td>26.146942</td>
          <td>21.798906</td>
          <td>23.558383</td>
          <td>22.976960</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.462950</td>
          <td>28.581384</td>
          <td>23.206086</td>
          <td>20.439122</td>
          <td>18.935462</td>
          <td>24.604148</td>
          <td>23.079099</td>
          <td>20.876411</td>
          <td>26.884633</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.168948</td>
          <td>25.781877</td>
          <td>22.857215</td>
          <td>20.284897</td>
          <td>25.813058</td>
          <td>21.787951</td>
          <td>22.677858</td>
          <td>23.089003</td>
          <td>19.535574</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.270968</td>
          <td>20.582937</td>
          <td>23.187563</td>
          <td>24.288095</td>
          <td>24.132315</td>
          <td>29.104991</td>
          <td>23.603891</td>
          <td>23.226932</td>
          <td>20.226010</td>
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
          <td>22.927759</td>
          <td>0.018006</td>
          <td>26.743997</td>
          <td>0.170666</td>
          <td>27.633664</td>
          <td>0.316046</td>
          <td>25.316502</td>
          <td>0.070290</td>
          <td>24.905758</td>
          <td>0.093398</td>
          <td>22.365421</td>
          <td>0.022381</td>
          <td>22.157869</td>
          <td>23.830213</td>
          <td>32.020728</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.960758</td>
          <td>0.009023</td>
          <td>25.242715</td>
          <td>0.045884</td>
          <td>25.641826</td>
          <td>0.057415</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.140643</td>
          <td>0.009252</td>
          <td>21.801015</td>
          <td>0.014027</td>
          <td>26.496968</td>
          <td>25.922360</td>
          <td>24.376712</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.265249</td>
          <td>0.005112</td>
          <td>21.979218</td>
          <td>0.005632</td>
          <td>23.780071</td>
          <td>0.011788</td>
          <td>23.167470</td>
          <td>0.011287</td>
          <td>21.195053</td>
          <td>0.006036</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.306932</td>
          <td>17.829858</td>
          <td>23.622871</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.902116</td>
          <td>0.005074</td>
          <td>25.550188</td>
          <td>0.060246</td>
          <td>18.295287</td>
          <td>0.005003</td>
          <td>24.357024</td>
          <td>0.030049</td>
          <td>18.829090</td>
          <td>0.005028</td>
          <td>23.379424</td>
          <td>0.054606</td>
          <td>20.965424</td>
          <td>27.517862</td>
          <td>17.510298</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.605939</td>
          <td>0.076719</td>
          <td>26.861667</td>
          <td>0.188551</td>
          <td>18.594083</td>
          <td>0.005004</td>
          <td>21.117394</td>
          <td>0.005279</td>
          <td>22.002321</td>
          <td>0.008505</td>
          <td>21.672105</td>
          <td>0.012694</td>
          <td>19.797726</td>
          <td>27.272744</td>
          <td>27.031754</td>
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
          <td>22.921683</td>
          <td>0.017918</td>
          <td>23.615634</td>
          <td>0.011721</td>
          <td>19.316653</td>
          <td>0.005010</td>
          <td>21.516905</td>
          <td>0.005531</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.974303</td>
          <td>0.218198</td>
          <td>17.342274</td>
          <td>22.261844</td>
          <td>22.257695</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.031162</td>
          <td>0.111246</td>
          <td>21.395537</td>
          <td>0.005260</td>
          <td>25.744387</td>
          <td>0.062884</td>
          <td>26.394939</td>
          <td>0.179611</td>
          <td>21.455412</td>
          <td>0.006557</td>
          <td>26.088264</td>
          <td>0.524132</td>
          <td>21.798906</td>
          <td>23.558383</td>
          <td>22.976960</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.462957</td>
          <td>0.007058</td>
          <td>27.463258</td>
          <td>0.309441</td>
          <td>23.209654</td>
          <td>0.008124</td>
          <td>20.435750</td>
          <td>0.005096</td>
          <td>18.939434</td>
          <td>0.005033</td>
          <td>24.719670</td>
          <td>0.176126</td>
          <td>23.079099</td>
          <td>20.876411</td>
          <td>26.884633</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.177654</td>
          <td>0.022134</td>
          <td>25.695526</td>
          <td>0.068512</td>
          <td>22.852612</td>
          <td>0.006839</td>
          <td>20.293172</td>
          <td>0.005077</td>
          <td>25.950749</td>
          <td>0.228824</td>
          <td>21.782027</td>
          <td>0.013819</td>
          <td>22.677858</td>
          <td>23.089003</td>
          <td>19.535574</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.326498</td>
          <td>0.143563</td>
          <td>20.583006</td>
          <td>0.005081</td>
          <td>23.194141</td>
          <td>0.008055</td>
          <td>24.256108</td>
          <td>0.027507</td>
          <td>24.102983</td>
          <td>0.045897</td>
          <td>26.435966</td>
          <td>0.670655</td>
          <td>23.603891</td>
          <td>23.226932</td>
          <td>20.226010</td>
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
          <td>22.924421</td>
          <td>26.623145</td>
          <td>27.394586</td>
          <td>25.305398</td>
          <td>24.927602</td>
          <td>22.374212</td>
          <td>22.158937</td>
          <td>0.005592</td>
          <td>23.845310</td>
          <td>0.021042</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.969920</td>
          <td>25.268529</td>
          <td>25.569740</td>
          <td>27.877423</td>
          <td>22.157410</td>
          <td>21.795464</td>
          <td>26.670936</td>
          <td>0.149647</td>
          <td>25.967327</td>
          <td>0.136862</td>
          <td>24.295871</td>
          <td>0.031214</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.263006</td>
          <td>21.980421</td>
          <td>23.784552</td>
          <td>23.168203</td>
          <td>21.190007</td>
          <td>28.171118</td>
          <td>21.306654</td>
          <td>0.005129</td>
          <td>17.834826</td>
          <td>0.005001</td>
          <td>23.647012</td>
          <td>0.017770</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.901991</td>
          <td>25.456661</td>
          <td>18.299273</td>
          <td>24.347278</td>
          <td>18.826936</td>
          <td>23.330937</td>
          <td>20.956311</td>
          <td>0.005068</td>
          <td>27.453027</td>
          <td>0.459983</td>
          <td>17.524662</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.598138</td>
          <td>26.626787</td>
          <td>18.591515</td>
          <td>21.119736</td>
          <td>21.997115</td>
          <td>21.655542</td>
          <td>19.796693</td>
          <td>0.005008</td>
          <td>27.477328</td>
          <td>0.468437</td>
          <td>26.980017</td>
          <td>0.318777</td>
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
          <td>22.926140</td>
          <td>23.611894</td>
          <td>19.308360</td>
          <td>21.517998</td>
          <td>27.563088</td>
          <td>24.956842</td>
          <td>17.340272</td>
          <td>0.005000</td>
          <td>22.258099</td>
          <td>0.006908</td>
          <td>22.260349</td>
          <td>0.006915</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.297483</td>
          <td>21.388896</td>
          <td>25.769455</td>
          <td>26.412470</td>
          <td>21.458691</td>
          <td>26.146942</td>
          <td>21.798324</td>
          <td>0.005313</td>
          <td>23.539876</td>
          <td>0.016249</td>
          <td>22.996752</td>
          <td>0.010645</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.462950</td>
          <td>28.581384</td>
          <td>23.206086</td>
          <td>20.439122</td>
          <td>18.935462</td>
          <td>24.604148</td>
          <td>23.073134</td>
          <td>0.007664</td>
          <td>20.875555</td>
          <td>0.005175</td>
          <td>26.702854</td>
          <td>0.254699</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.168948</td>
          <td>25.781877</td>
          <td>22.857215</td>
          <td>20.284897</td>
          <td>25.813058</td>
          <td>21.787951</td>
          <td>22.681655</td>
          <td>0.006436</td>
          <td>23.074976</td>
          <td>0.011267</td>
          <td>19.543028</td>
          <td>0.005015</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.270968</td>
          <td>20.582937</td>
          <td>23.187563</td>
          <td>24.288095</td>
          <td>24.132315</td>
          <td>29.104991</td>
          <td>23.597194</td>
          <td>0.010648</td>
          <td>23.204988</td>
          <td>0.012426</td>
          <td>20.223213</td>
          <td>0.005053</td>
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
          <td>22.924421</td>
          <td>26.623145</td>
          <td>27.394586</td>
          <td>25.305398</td>
          <td>24.927602</td>
          <td>22.374212</td>
          <td>22.146573</td>
          <td>0.029883</td>
          <td>23.978347</td>
          <td>0.126709</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.969920</td>
          <td>25.268529</td>
          <td>25.569740</td>
          <td>27.877423</td>
          <td>22.157410</td>
          <td>21.795464</td>
          <td>26.399273</td>
          <td>0.942693</td>
          <td>26.158630</td>
          <td>0.709092</td>
          <td>24.605813</td>
          <td>0.235119</td>
        </tr>
        <tr>
          <th>2</th>
          <td>19.263006</td>
          <td>21.980421</td>
          <td>23.784552</td>
          <td>23.168203</td>
          <td>21.190007</td>
          <td>28.171118</td>
          <td>21.300542</td>
          <td>0.014498</td>
          <td>17.825822</td>
          <td>0.005021</td>
          <td>23.529863</td>
          <td>0.093423</td>
        </tr>
        <tr>
          <th>3</th>
          <td>18.901991</td>
          <td>25.456661</td>
          <td>18.299273</td>
          <td>24.347278</td>
          <td>18.826936</td>
          <td>23.330937</td>
          <td>20.956475</td>
          <td>0.011115</td>
          <td>inf</td>
          <td>inf</td>
          <td>17.512900</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>4</th>
          <td>24.598138</td>
          <td>26.626787</td>
          <td>18.591515</td>
          <td>21.119736</td>
          <td>21.997115</td>
          <td>21.655542</td>
          <td>19.794113</td>
          <td>0.006052</td>
          <td>29.051352</td>
          <td>2.883038</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>22.926140</td>
          <td>23.611894</td>
          <td>19.308360</td>
          <td>21.517998</td>
          <td>27.563088</td>
          <td>24.956842</td>
          <td>17.348172</td>
          <td>0.005013</td>
          <td>22.247303</td>
          <td>0.027378</td>
          <td>22.244758</td>
          <td>0.029835</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.297483</td>
          <td>21.388896</td>
          <td>25.769455</td>
          <td>26.412470</td>
          <td>21.458691</td>
          <td>26.146942</td>
          <td>21.811843</td>
          <td>0.022287</td>
          <td>23.546837</td>
          <td>0.086831</td>
          <td>23.037628</td>
          <td>0.060407</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.462950</td>
          <td>28.581384</td>
          <td>23.206086</td>
          <td>20.439122</td>
          <td>18.935462</td>
          <td>24.604148</td>
          <td>23.050297</td>
          <td>0.066773</td>
          <td>20.853744</td>
          <td>0.009029</td>
          <td>25.662761</td>
          <td>0.537076</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.168948</td>
          <td>25.781877</td>
          <td>22.857215</td>
          <td>20.284897</td>
          <td>25.813058</td>
          <td>21.787951</td>
          <td>22.654059</td>
          <td>0.046919</td>
          <td>23.127555</td>
          <td>0.059868</td>
          <td>19.540707</td>
          <td>0.005574</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.270968</td>
          <td>20.582937</td>
          <td>23.187563</td>
          <td>24.288095</td>
          <td>24.132315</td>
          <td>29.104991</td>
          <td>23.420999</td>
          <td>0.092697</td>
          <td>23.288099</td>
          <td>0.069053</td>
          <td>20.226581</td>
          <td>0.006815</td>
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


