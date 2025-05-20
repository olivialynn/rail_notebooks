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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.613276</td>
          <td>0.014057</td>
          <td>22.952921</td>
          <td>0.007743</td>
          <td>25.808037</td>
          <td>0.066534</td>
          <td>18.276940</td>
          <td>0.005006</td>
          <td>23.025837</td>
          <td>0.017974</td>
          <td>23.601125</td>
          <td>0.066472</td>
          <td>24.988174</td>
          <td>17.297281</td>
          <td>25.691319</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.679585</td>
          <td>0.014790</td>
          <td>24.403499</td>
          <td>0.022036</td>
          <td>20.928895</td>
          <td>0.005088</td>
          <td>24.340159</td>
          <td>0.029607</td>
          <td>22.950606</td>
          <td>0.016888</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.246999</td>
          <td>24.175778</td>
          <td>24.107270</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.122074</td>
          <td>0.009955</td>
          <td>23.423666</td>
          <td>0.010244</td>
          <td>21.993195</td>
          <td>0.005470</td>
          <td>25.353024</td>
          <td>0.072598</td>
          <td>22.946863</td>
          <td>0.016836</td>
          <td>23.433566</td>
          <td>0.057294</td>
          <td>24.789983</td>
          <td>25.735875</td>
          <td>24.365248</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.997138</td>
          <td>0.019054</td>
          <td>22.922741</td>
          <td>0.007627</td>
          <td>28.462356</td>
          <td>0.591705</td>
          <td>19.631121</td>
          <td>0.005030</td>
          <td>18.622517</td>
          <td>0.005021</td>
          <td>22.581075</td>
          <td>0.026975</td>
          <td>25.217366</td>
          <td>22.214593</td>
          <td>23.033418</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.548583</td>
          <td>0.005157</td>
          <td>22.307237</td>
          <td>0.006048</td>
          <td>24.283241</td>
          <td>0.017537</td>
          <td>12.208556</td>
          <td>0.005000</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.896318</td>
          <td>0.035582</td>
          <td>23.057799</td>
          <td>22.879641</td>
          <td>16.979636</td>
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
          <td>18.293438</td>
          <td>0.005038</td>
          <td>18.220895</td>
          <td>0.005005</td>
          <td>14.344954</td>
          <td>0.005000</td>
          <td>22.230137</td>
          <td>0.006652</td>
          <td>23.042314</td>
          <td>0.018223</td>
          <td>23.337886</td>
          <td>0.052630</td>
          <td>24.704066</td>
          <td>21.638246</td>
          <td>19.862898</td>
        </tr>
        <tr>
          <th>996</th>
          <td>inf</td>
          <td>inf</td>
          <td>19.192817</td>
          <td>0.005015</td>
          <td>23.760407</td>
          <td>0.011619</td>
          <td>23.085437</td>
          <td>0.010647</td>
          <td>24.809972</td>
          <td>0.085852</td>
          <td>21.779757</td>
          <td>0.013794</td>
          <td>21.322435</td>
          <td>22.318098</td>
          <td>22.769207</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.785343</td>
          <td>0.005800</td>
          <td>25.008394</td>
          <td>0.037302</td>
          <td>20.556220</td>
          <td>0.005051</td>
          <td>20.922988</td>
          <td>0.005204</td>
          <td>24.358256</td>
          <td>0.057570</td>
          <td>23.549797</td>
          <td>0.063516</td>
          <td>24.396760</td>
          <td>26.426820</td>
          <td>24.437075</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.939849</td>
          <td>0.102750</td>
          <td>24.884855</td>
          <td>0.033458</td>
          <td>26.357286</td>
          <td>0.107920</td>
          <td>26.058572</td>
          <td>0.134671</td>
          <td>17.081645</td>
          <td>0.005004</td>
          <td>26.558679</td>
          <td>0.728916</td>
          <td>24.020885</td>
          <td>26.171105</td>
          <td>19.988612</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.162891</td>
          <td>0.010219</td>
          <td>17.800657</td>
          <td>0.005003</td>
          <td>20.734605</td>
          <td>0.005066</td>
          <td>20.696784</td>
          <td>0.005143</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.844306</td>
          <td>0.005520</td>
          <td>26.681762</td>
          <td>22.623372</td>
          <td>24.677581</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>24.948129</td>
          <td>0.032693</td>
          <td>17.294033</td>
          <td>0.005000</td>
          <td>25.543613</td>
          <td>0.094560</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.254849</td>
          <td>0.008493</td>
          <td>24.169009</td>
          <td>0.027906</td>
          <td>24.094224</td>
          <td>0.026131</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>24.741038</td>
          <td>0.027228</td>
          <td>25.662677</td>
          <td>0.104981</td>
          <td>24.334964</td>
          <td>0.032314</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.227638</td>
          <td>0.041918</td>
          <td>22.219383</td>
          <td>0.006794</td>
          <td>23.032008</td>
          <td>0.010919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.044568</td>
          <td>0.007551</td>
          <td>22.868517</td>
          <td>0.009736</td>
          <td>16.975224</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.704770</td>
          <td>0.026373</td>
          <td>21.629201</td>
          <td>0.005669</td>
          <td>19.864487</td>
          <td>0.005028</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.328651</td>
          <td>0.005134</td>
          <td>22.324293</td>
          <td>0.007118</td>
          <td>22.762803</td>
          <td>0.009081</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.421371</td>
          <td>0.020613</td>
          <td>26.342828</td>
          <td>0.188682</td>
          <td>24.457044</td>
          <td>0.036013</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.027619</td>
          <td>0.014819</td>
          <td>26.347273</td>
          <td>0.189392</td>
          <td>19.992392</td>
          <td>0.005035</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>26.546138</td>
          <td>0.134377</td>
          <td>22.634641</td>
          <td>0.008391</td>
          <td>24.680251</td>
          <td>0.043931</td>
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
          <td>22.600424</td>
          <td>22.961977</td>
          <td>25.818134</td>
          <td>18.274340</td>
          <td>23.027885</td>
          <td>23.638181</td>
          <td>25.379345</td>
          <td>0.469144</td>
          <td>17.297955</td>
          <td>0.005008</td>
          <td>25.221117</td>
          <td>0.385368</td>
        </tr>
        <tr>
          <th>1</th>
          <td>22.682690</td>
          <td>24.413747</td>
          <td>20.926504</td>
          <td>24.363033</td>
          <td>22.927275</td>
          <td>26.444150</td>
          <td>23.327018</td>
          <td>0.085325</td>
          <td>23.985282</td>
          <td>0.127474</td>
          <td>23.896702</td>
          <td>0.128743</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.120187</td>
          <td>23.411418</td>
          <td>22.001105</td>
          <td>25.361529</td>
          <td>22.966012</td>
          <td>23.444323</td>
          <td>25.146031</td>
          <td>0.392871</td>
          <td>25.043075</td>
          <td>0.309498</td>
          <td>24.457996</td>
          <td>0.207882</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.986236</td>
          <td>22.927064</td>
          <td>28.331736</td>
          <td>19.622063</td>
          <td>18.624798</td>
          <td>22.598671</td>
          <td>25.010130</td>
          <td>0.353383</td>
          <td>22.242839</td>
          <td>0.027271</td>
          <td>22.959223</td>
          <td>0.056333</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.541318</td>
          <td>22.305423</td>
          <td>24.290190</td>
          <td>12.204110</td>
          <td>27.382203</td>
          <td>22.867527</td>
          <td>23.194857</td>
          <td>0.075912</td>
          <td>22.948775</td>
          <td>0.051053</td>
          <td>16.972382</td>
          <td>0.005005</td>
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
          <td>18.286397</td>
          <td>18.219970</td>
          <td>14.348508</td>
          <td>22.215665</td>
          <td>23.057058</td>
          <td>23.365618</td>
          <td>24.924718</td>
          <td>0.330324</td>
          <td>21.632710</td>
          <td>0.016153</td>
          <td>19.862461</td>
          <td>0.005997</td>
        </tr>
        <tr>
          <th>996</th>
          <td>27.785847</td>
          <td>19.202302</td>
          <td>23.726031</td>
          <td>23.101989</td>
          <td>24.929284</td>
          <td>21.791072</td>
          <td>21.298684</td>
          <td>0.014476</td>
          <td>22.304623</td>
          <td>0.028796</td>
          <td>22.716232</td>
          <td>0.045363</td>
        </tr>
        <tr>
          <th>997</th>
          <td>20.784442</td>
          <td>24.969209</td>
          <td>20.555581</td>
          <td>20.927701</td>
          <td>24.437681</td>
          <td>23.567993</td>
          <td>24.446435</td>
          <td>0.223813</td>
          <td>25.825355</td>
          <td>0.561932</td>
          <td>24.298130</td>
          <td>0.181679</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.817794</td>
          <td>24.877437</td>
          <td>26.242721</td>
          <td>26.028165</td>
          <td>17.081520</td>
          <td>28.302186</td>
          <td>24.278331</td>
          <td>0.194421</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.996828</td>
          <td>0.006249</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.159231</td>
          <td>17.803560</td>
          <td>20.729490</td>
          <td>20.693127</td>
          <td>30.756824</td>
          <td>19.849953</td>
          <td>25.465993</td>
          <td>0.500336</td>
          <td>22.634213</td>
          <td>0.038572</td>
          <td>25.115459</td>
          <td>0.354866</td>
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


