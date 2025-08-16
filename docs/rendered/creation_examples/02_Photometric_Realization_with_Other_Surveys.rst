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
          <td>24.045319</td>
          <td>20.300750</td>
          <td>21.762237</td>
          <td>25.212422</td>
          <td>19.348047</td>
          <td>26.281083</td>
          <td>19.055271</td>
          <td>22.524786</td>
          <td>22.986802</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.986618</td>
          <td>24.707170</td>
          <td>23.341554</td>
          <td>24.597821</td>
          <td>23.274219</td>
          <td>23.990844</td>
          <td>25.328774</td>
          <td>23.617130</td>
          <td>17.944587</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.181495</td>
          <td>23.284047</td>
          <td>26.914646</td>
          <td>22.139564</td>
          <td>25.500890</td>
          <td>17.304624</td>
          <td>19.907707</td>
          <td>20.273633</td>
          <td>23.978421</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.549986</td>
          <td>20.306990</td>
          <td>24.530448</td>
          <td>23.618290</td>
          <td>20.754715</td>
          <td>16.757909</td>
          <td>25.464720</td>
          <td>23.911626</td>
          <td>23.288518</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.546018</td>
          <td>25.280600</td>
          <td>23.816520</td>
          <td>24.745796</td>
          <td>28.358460</td>
          <td>21.375890</td>
          <td>21.862554</td>
          <td>18.969783</td>
          <td>26.184382</td>
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
          <td>20.663956</td>
          <td>24.206213</td>
          <td>24.824602</td>
          <td>26.880820</td>
          <td>17.338150</td>
          <td>20.024561</td>
          <td>22.937471</td>
          <td>21.643098</td>
          <td>23.010180</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.672692</td>
          <td>20.508622</td>
          <td>20.839812</td>
          <td>25.525446</td>
          <td>22.271490</td>
          <td>25.170213</td>
          <td>21.519332</td>
          <td>21.862998</td>
          <td>25.702892</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.657261</td>
          <td>19.040155</td>
          <td>21.829368</td>
          <td>24.386958</td>
          <td>22.181441</td>
          <td>25.118612</td>
          <td>31.175502</td>
          <td>21.663407</td>
          <td>21.393114</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.971624</td>
          <td>21.816765</td>
          <td>17.674366</td>
          <td>18.942768</td>
          <td>22.151547</td>
          <td>20.293142</td>
          <td>20.948491</td>
          <td>25.275506</td>
          <td>19.510297</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.562614</td>
          <td>23.230641</td>
          <td>21.116145</td>
          <td>25.102357</td>
          <td>21.091997</td>
          <td>25.110828</td>
          <td>26.722364</td>
          <td>19.703532</td>
          <td>25.746557</td>
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
          <td>24.063283</td>
          <td>0.047602</td>
          <td>20.291848</td>
          <td>0.005055</td>
          <td>21.762934</td>
          <td>0.005325</td>
          <td>25.090092</td>
          <td>0.057507</td>
          <td>19.349814</td>
          <td>0.005058</td>
          <td>26.210266</td>
          <td>0.572463</td>
          <td>19.055271</td>
          <td>22.524786</td>
          <td>22.986802</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.982512</td>
          <td>0.009139</td>
          <td>24.680816</td>
          <td>0.027989</td>
          <td>23.341109</td>
          <td>0.008766</td>
          <td>24.536579</td>
          <td>0.035198</td>
          <td>23.303115</td>
          <td>0.022737</td>
          <td>24.001784</td>
          <td>0.094660</td>
          <td>25.328774</td>
          <td>23.617130</td>
          <td>17.944587</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.573391</td>
          <td>0.399041</td>
          <td>23.293010</td>
          <td>0.009405</td>
          <td>27.048804</td>
          <td>0.195486</td>
          <td>22.146157</td>
          <td>0.006451</td>
          <td>25.552653</td>
          <td>0.163659</td>
          <td>17.301432</td>
          <td>0.005013</td>
          <td>19.907707</td>
          <td>20.273633</td>
          <td>23.978421</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.615383</td>
          <td>0.412110</td>
          <td>20.313766</td>
          <td>0.005057</td>
          <td>24.554007</td>
          <td>0.022045</td>
          <td>23.632614</td>
          <td>0.016186</td>
          <td>20.756440</td>
          <td>0.005513</td>
          <td>16.758522</td>
          <td>0.005007</td>
          <td>25.464720</td>
          <td>23.911626</td>
          <td>23.288518</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.536013</td>
          <td>0.007275</td>
          <td>25.302760</td>
          <td>0.048390</td>
          <td>23.829073</td>
          <td>0.012226</td>
          <td>24.746112</td>
          <td>0.042374</td>
          <td>32.581280</td>
          <td>5.065546</td>
          <td>21.375530</td>
          <td>0.010242</td>
          <td>21.862554</td>
          <td>18.969783</td>
          <td>26.184382</td>
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
          <td>20.660279</td>
          <td>0.005672</td>
          <td>24.220915</td>
          <td>0.018889</td>
          <td>24.771538</td>
          <td>0.026609</td>
          <td>26.939356</td>
          <td>0.282218</td>
          <td>17.341450</td>
          <td>0.005005</td>
          <td>20.017375</td>
          <td>0.005686</td>
          <td>22.937471</td>
          <td>21.643098</td>
          <td>23.010180</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.614778</td>
          <td>0.411919</td>
          <td>20.506750</td>
          <td>0.005073</td>
          <td>20.837604</td>
          <td>0.005077</td>
          <td>25.626806</td>
          <td>0.092423</td>
          <td>22.274862</td>
          <td>0.010102</td>
          <td>24.841410</td>
          <td>0.195215</td>
          <td>21.519332</td>
          <td>21.862998</td>
          <td>25.702892</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.624991</td>
          <td>0.078013</td>
          <td>19.044753</td>
          <td>0.005012</td>
          <td>21.832631</td>
          <td>0.005363</td>
          <td>24.344338</td>
          <td>0.029716</td>
          <td>22.192326</td>
          <td>0.009564</td>
          <td>26.152149</td>
          <td>0.549031</td>
          <td>31.175502</td>
          <td>21.663407</td>
          <td>21.393114</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.973916</td>
          <td>0.005027</td>
          <td>21.810278</td>
          <td>0.005487</td>
          <td>17.678165</td>
          <td>0.005002</td>
          <td>18.943747</td>
          <td>0.005012</td>
          <td>22.152437</td>
          <td>0.009321</td>
          <td>20.280093</td>
          <td>0.006044</td>
          <td>20.948491</td>
          <td>25.275506</td>
          <td>19.510297</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.556416</td>
          <td>0.005158</td>
          <td>23.257235</td>
          <td>0.009197</td>
          <td>21.111180</td>
          <td>0.005116</td>
          <td>25.263024</td>
          <td>0.067038</td>
          <td>21.094463</td>
          <td>0.005883</td>
          <td>25.763534</td>
          <td>0.411008</td>
          <td>26.722364</td>
          <td>19.703532</td>
          <td>25.746557</td>
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
          <td>24.045319</td>
          <td>20.300750</td>
          <td>21.762237</td>
          <td>25.212422</td>
          <td>19.348047</td>
          <td>26.281083</td>
          <td>19.052466</td>
          <td>0.005002</td>
          <td>22.526479</td>
          <td>0.007888</td>
          <td>23.000725</td>
          <td>0.010675</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.986618</td>
          <td>24.707170</td>
          <td>23.341554</td>
          <td>24.597821</td>
          <td>23.274219</td>
          <td>23.990844</td>
          <td>25.357761</td>
          <td>0.047074</td>
          <td>23.599908</td>
          <td>0.017081</td>
          <td>17.944444</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.181495</td>
          <td>23.284047</td>
          <td>26.914646</td>
          <td>22.139564</td>
          <td>25.500890</td>
          <td>17.304624</td>
          <td>19.903630</td>
          <td>0.005010</td>
          <td>20.276667</td>
          <td>0.005059</td>
          <td>23.935121</td>
          <td>0.022742</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.549986</td>
          <td>20.306990</td>
          <td>24.530448</td>
          <td>23.618290</td>
          <td>20.754715</td>
          <td>16.757909</td>
          <td>25.481433</td>
          <td>0.052561</td>
          <td>23.928679</td>
          <td>0.022615</td>
          <td>23.296507</td>
          <td>0.013343</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.546018</td>
          <td>25.280600</td>
          <td>23.816520</td>
          <td>24.745796</td>
          <td>28.358460</td>
          <td>21.375890</td>
          <td>21.860641</td>
          <td>0.005350</td>
          <td>18.966590</td>
          <td>0.005005</td>
          <td>26.206700</td>
          <td>0.168092</td>
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
          <td>20.663956</td>
          <td>24.206213</td>
          <td>24.824602</td>
          <td>26.880820</td>
          <td>17.338150</td>
          <td>20.024561</td>
          <td>22.937880</td>
          <td>0.007163</td>
          <td>21.635641</td>
          <td>0.005677</td>
          <td>23.012914</td>
          <td>0.010769</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.672692</td>
          <td>20.508622</td>
          <td>20.839812</td>
          <td>25.525446</td>
          <td>22.271490</td>
          <td>25.170213</td>
          <td>21.517202</td>
          <td>0.005189</td>
          <td>21.871362</td>
          <td>0.006013</td>
          <td>25.673137</td>
          <td>0.105947</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.657261</td>
          <td>19.040155</td>
          <td>21.829368</td>
          <td>24.386958</td>
          <td>22.181441</td>
          <td>25.118612</td>
          <td>29.081448</td>
          <td>0.932385</td>
          <td>21.671071</td>
          <td>0.005719</td>
          <td>21.392647</td>
          <td>0.005442</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.971624</td>
          <td>21.816765</td>
          <td>17.674366</td>
          <td>18.942768</td>
          <td>22.151547</td>
          <td>20.293142</td>
          <td>20.945081</td>
          <td>0.005067</td>
          <td>25.430062</td>
          <td>0.085555</td>
          <td>19.508255</td>
          <td>0.005014</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.562614</td>
          <td>23.230641</td>
          <td>21.116145</td>
          <td>25.102357</td>
          <td>21.091997</td>
          <td>25.110828</td>
          <td>26.675117</td>
          <td>0.150186</td>
          <td>19.702586</td>
          <td>0.005020</td>
          <td>25.773544</td>
          <td>0.115664</td>
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
          <td>24.045319</td>
          <td>20.300750</td>
          <td>21.762237</td>
          <td>25.212422</td>
          <td>19.348047</td>
          <td>26.281083</td>
          <td>19.048863</td>
          <td>0.005287</td>
          <td>22.530403</td>
          <td>0.035170</td>
          <td>22.983011</td>
          <td>0.057540</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.986618</td>
          <td>24.707170</td>
          <td>23.341554</td>
          <td>24.597821</td>
          <td>23.274219</td>
          <td>23.990844</td>
          <td>26.075709</td>
          <td>0.766815</td>
          <td>23.479324</td>
          <td>0.081804</td>
          <td>17.952183</td>
          <td>0.005032</td>
        </tr>
        <tr>
          <th>2</th>
          <td>27.181495</td>
          <td>23.284047</td>
          <td>26.914646</td>
          <td>22.139564</td>
          <td>25.500890</td>
          <td>17.304624</td>
          <td>19.907744</td>
          <td>0.006272</td>
          <td>20.265913</td>
          <td>0.006646</td>
          <td>24.261082</td>
          <td>0.176055</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.549986</td>
          <td>20.306990</td>
          <td>24.530448</td>
          <td>23.618290</td>
          <td>20.754715</td>
          <td>16.757909</td>
          <td>25.071533</td>
          <td>0.370791</td>
          <td>23.714527</td>
          <td>0.100640</td>
          <td>23.330654</td>
          <td>0.078357</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.546018</td>
          <td>25.280600</td>
          <td>23.816520</td>
          <td>24.745796</td>
          <td>28.358460</td>
          <td>21.375890</td>
          <td>21.889686</td>
          <td>0.023847</td>
          <td>18.970533</td>
          <td>0.005174</td>
          <td>27.008754</td>
          <td>1.266504</td>
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
          <td>20.663956</td>
          <td>24.206213</td>
          <td>24.824602</td>
          <td>26.880820</td>
          <td>17.338150</td>
          <td>20.024561</td>
          <td>22.908622</td>
          <td>0.058867</td>
          <td>21.623850</td>
          <td>0.016035</td>
          <td>22.977456</td>
          <td>0.057256</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.672692</td>
          <td>20.508622</td>
          <td>20.839812</td>
          <td>25.525446</td>
          <td>22.271490</td>
          <td>25.170213</td>
          <td>21.532329</td>
          <td>0.017552</td>
          <td>21.851601</td>
          <td>0.019418</td>
          <td>25.602913</td>
          <td>0.514114</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.657261</td>
          <td>19.040155</td>
          <td>21.829368</td>
          <td>24.386958</td>
          <td>22.181441</td>
          <td>25.118612</td>
          <td>26.967166</td>
          <td>1.307058</td>
          <td>21.684681</td>
          <td>0.016865</td>
          <td>21.408164</td>
          <td>0.014588</td>
        </tr>
        <tr>
          <th>998</th>
          <td>17.971624</td>
          <td>21.816765</td>
          <td>17.674366</td>
          <td>18.942768</td>
          <td>22.151547</td>
          <td>20.293142</td>
          <td>20.948887</td>
          <td>0.011053</td>
          <td>25.776214</td>
          <td>0.542345</td>
          <td>19.498750</td>
          <td>0.005533</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.562614</td>
          <td>23.230641</td>
          <td>21.116145</td>
          <td>25.102357</td>
          <td>21.091997</td>
          <td>25.110828</td>
          <td>27.217480</td>
          <td>1.488135</td>
          <td>19.702094</td>
          <td>0.005638</td>
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


