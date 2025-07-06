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
          <td>19.209968</td>
          <td>19.677317</td>
          <td>17.358430</td>
          <td>25.163205</td>
          <td>18.925505</td>
          <td>26.811069</td>
          <td>17.708265</td>
          <td>21.459790</td>
          <td>21.226525</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.710375</td>
          <td>19.313440</td>
          <td>27.959407</td>
          <td>21.263009</td>
          <td>24.054788</td>
          <td>24.910869</td>
          <td>26.857051</td>
          <td>21.361612</td>
          <td>24.552861</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.869575</td>
          <td>21.602492</td>
          <td>25.529264</td>
          <td>23.068049</td>
          <td>17.447842</td>
          <td>22.680342</td>
          <td>26.683137</td>
          <td>21.915803</td>
          <td>22.972333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.589882</td>
          <td>27.576717</td>
          <td>21.552096</td>
          <td>20.573637</td>
          <td>16.624106</td>
          <td>22.520046</td>
          <td>27.661765</td>
          <td>22.595669</td>
          <td>19.085920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.972696</td>
          <td>20.223403</td>
          <td>17.737244</td>
          <td>22.827789</td>
          <td>26.691005</td>
          <td>20.327054</td>
          <td>22.709751</td>
          <td>20.132075</td>
          <td>19.906697</td>
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
          <td>21.400958</td>
          <td>19.909868</td>
          <td>26.160232</td>
          <td>20.127856</td>
          <td>22.536843</td>
          <td>24.035025</td>
          <td>19.374816</td>
          <td>23.780542</td>
          <td>20.314771</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.485905</td>
          <td>21.068944</td>
          <td>27.823563</td>
          <td>20.002640</td>
          <td>23.165620</td>
          <td>19.545039</td>
          <td>20.041089</td>
          <td>27.277971</td>
          <td>22.245472</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.021091</td>
          <td>25.239927</td>
          <td>21.660126</td>
          <td>25.513290</td>
          <td>23.697363</td>
          <td>23.979353</td>
          <td>28.062145</td>
          <td>23.986698</td>
          <td>19.546256</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.004101</td>
          <td>23.597287</td>
          <td>19.289497</td>
          <td>22.935598</td>
          <td>24.964263</td>
          <td>21.316885</td>
          <td>25.536937</td>
          <td>20.179491</td>
          <td>23.299173</td>
        </tr>
        <tr>
          <th>999</th>
          <td>12.880496</td>
          <td>21.223576</td>
          <td>23.169376</td>
          <td>24.949464</td>
          <td>21.582061</td>
          <td>24.592514</td>
          <td>24.970312</td>
          <td>23.636063</td>
          <td>25.334016</td>
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
          <td>19.217593</td>
          <td>0.005106</td>
          <td>19.671491</td>
          <td>0.005025</td>
          <td>17.363913</td>
          <td>0.005001</td>
          <td>25.249451</td>
          <td>0.066237</td>
          <td>18.928997</td>
          <td>0.005032</td>
          <td>27.462862</td>
          <td>1.262973</td>
          <td>17.708265</td>
          <td>21.459790</td>
          <td>21.226525</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.712989</td>
          <td>0.005021</td>
          <td>19.318196</td>
          <td>0.005017</td>
          <td>27.505758</td>
          <td>0.285161</td>
          <td>21.259427</td>
          <td>0.005351</td>
          <td>24.053268</td>
          <td>0.043917</td>
          <td>24.921971</td>
          <td>0.208869</td>
          <td>26.857051</td>
          <td>21.361612</td>
          <td>24.552861</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.882975</td>
          <td>0.008632</td>
          <td>21.608406</td>
          <td>0.005358</td>
          <td>25.422214</td>
          <td>0.047244</td>
          <td>23.077271</td>
          <td>0.010586</td>
          <td>17.455064</td>
          <td>0.005005</td>
          <td>22.689546</td>
          <td>0.029660</td>
          <td>26.683137</td>
          <td>21.915803</td>
          <td>22.972333</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.668373</td>
          <td>0.081039</td>
          <td>27.699240</td>
          <td>0.372849</td>
          <td>21.552763</td>
          <td>0.005232</td>
          <td>20.574132</td>
          <td>0.005118</td>
          <td>16.623576</td>
          <td>0.005002</td>
          <td>22.498478</td>
          <td>0.025105</td>
          <td>27.661765</td>
          <td>22.595669</td>
          <td>19.085920</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.974488</td>
          <td>0.005268</td>
          <td>20.229307</td>
          <td>0.005051</td>
          <td>17.738112</td>
          <td>0.005002</td>
          <td>22.827401</td>
          <td>0.008979</td>
          <td>26.735085</td>
          <td>0.427756</td>
          <td>20.322155</td>
          <td>0.006115</td>
          <td>22.709751</td>
          <td>20.132075</td>
          <td>19.906697</td>
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
          <td>21.400683</td>
          <td>0.006888</td>
          <td>19.899759</td>
          <td>0.005034</td>
          <td>25.964620</td>
          <td>0.076421</td>
          <td>20.129648</td>
          <td>0.005061</td>
          <td>22.529957</td>
          <td>0.012119</td>
          <td>23.957002</td>
          <td>0.091009</td>
          <td>19.374816</td>
          <td>23.780542</td>
          <td>20.314771</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.485801</td>
          <td>0.005145</td>
          <td>21.061078</td>
          <td>0.005159</td>
          <td>27.672539</td>
          <td>0.325989</td>
          <td>20.003256</td>
          <td>0.005050</td>
          <td>23.229240</td>
          <td>0.021342</td>
          <td>19.544169</td>
          <td>0.005321</td>
          <td>20.041089</td>
          <td>27.277971</td>
          <td>22.245472</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.857271</td>
          <td>0.095610</td>
          <td>25.215829</td>
          <td>0.044805</td>
          <td>21.663099</td>
          <td>0.005277</td>
          <td>25.530402</td>
          <td>0.084907</td>
          <td>23.746783</td>
          <td>0.033485</td>
          <td>23.938535</td>
          <td>0.089543</td>
          <td>28.062145</td>
          <td>23.986698</td>
          <td>19.546256</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.498684</td>
          <td>0.376642</td>
          <td>23.589275</td>
          <td>0.011500</td>
          <td>19.295435</td>
          <td>0.005010</td>
          <td>22.925980</td>
          <td>0.009558</td>
          <td>24.967836</td>
          <td>0.098626</td>
          <td>21.307908</td>
          <td>0.009786</td>
          <td>25.536937</td>
          <td>20.179491</td>
          <td>23.299173</td>
        </tr>
        <tr>
          <th>999</th>
          <td>12.876631</td>
          <td>0.005000</td>
          <td>21.214189</td>
          <td>0.005198</td>
          <td>23.167755</td>
          <td>0.007941</td>
          <td>24.994812</td>
          <td>0.052842</td>
          <td>21.574689</td>
          <td>0.006869</td>
          <td>24.708923</td>
          <td>0.174527</td>
          <td>24.970312</td>
          <td>23.636063</td>
          <td>25.334016</td>
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
          <td>19.209968</td>
          <td>19.677317</td>
          <td>17.358430</td>
          <td>25.163205</td>
          <td>18.925505</td>
          <td>26.811069</td>
          <td>17.703304</td>
          <td>0.005000</td>
          <td>21.465141</td>
          <td>0.005503</td>
          <td>21.224259</td>
          <td>0.005328</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.710375</td>
          <td>19.313440</td>
          <td>27.959407</td>
          <td>21.263009</td>
          <td>24.054788</td>
          <td>24.910869</td>
          <td>26.638898</td>
          <td>0.145578</td>
          <td>21.366619</td>
          <td>0.005422</td>
          <td>24.578020</td>
          <td>0.040106</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.869575</td>
          <td>21.602492</td>
          <td>25.529264</td>
          <td>23.068049</td>
          <td>17.447842</td>
          <td>22.680342</td>
          <td>26.562594</td>
          <td>0.136303</td>
          <td>21.910869</td>
          <td>0.006082</td>
          <td>22.982930</td>
          <td>0.010540</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.589882</td>
          <td>27.576717</td>
          <td>21.552096</td>
          <td>20.573637</td>
          <td>16.624106</td>
          <td>22.520046</td>
          <td>27.440650</td>
          <td>0.284985</td>
          <td>22.600349</td>
          <td>0.008224</td>
          <td>19.086786</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.972696</td>
          <td>20.223403</td>
          <td>17.737244</td>
          <td>22.827789</td>
          <td>26.691005</td>
          <td>20.327054</td>
          <td>22.708159</td>
          <td>0.006499</td>
          <td>20.134227</td>
          <td>0.005045</td>
          <td>19.911063</td>
          <td>0.005030</td>
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
          <td>21.400958</td>
          <td>19.909868</td>
          <td>26.160232</td>
          <td>20.127856</td>
          <td>22.536843</td>
          <td>24.035025</td>
          <td>19.382071</td>
          <td>0.005004</td>
          <td>23.760065</td>
          <td>0.019559</td>
          <td>20.309232</td>
          <td>0.005062</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.485905</td>
          <td>21.068944</td>
          <td>27.823563</td>
          <td>20.002640</td>
          <td>23.165620</td>
          <td>19.545039</td>
          <td>20.042528</td>
          <td>0.005013</td>
          <td>27.537012</td>
          <td>0.489726</td>
          <td>22.256857</td>
          <td>0.006904</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.021091</td>
          <td>25.239927</td>
          <td>21.660126</td>
          <td>25.513290</td>
          <td>23.697363</td>
          <td>23.979353</td>
          <td>28.044403</td>
          <td>0.457013</td>
          <td>23.949985</td>
          <td>0.023037</td>
          <td>19.549356</td>
          <td>0.005015</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.004101</td>
          <td>23.597287</td>
          <td>19.289497</td>
          <td>22.935598</td>
          <td>24.964263</td>
          <td>21.316885</td>
          <td>25.576198</td>
          <td>0.057192</td>
          <td>20.187024</td>
          <td>0.005050</td>
          <td>23.305835</td>
          <td>0.013442</td>
        </tr>
        <tr>
          <th>999</th>
          <td>12.880496</td>
          <td>21.223576</td>
          <td>23.169376</td>
          <td>24.949464</td>
          <td>21.582061</td>
          <td>24.592514</td>
          <td>24.973332</td>
          <td>0.033432</td>
          <td>23.624475</td>
          <td>0.017437</td>
          <td>25.287732</td>
          <td>0.075434</td>
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
          <td>19.209968</td>
          <td>19.677317</td>
          <td>17.358430</td>
          <td>25.163205</td>
          <td>18.925505</td>
          <td>26.811069</td>
          <td>17.705683</td>
          <td>0.005025</td>
          <td>21.464588</td>
          <td>0.014085</td>
          <td>21.233308</td>
          <td>0.012701</td>
        </tr>
        <tr>
          <th>1</th>
          <td>17.710375</td>
          <td>19.313440</td>
          <td>27.959407</td>
          <td>21.263009</td>
          <td>24.054788</td>
          <td>24.910869</td>
          <td>26.339044</td>
          <td>0.908150</td>
          <td>21.351568</td>
          <td>0.012882</td>
          <td>24.255242</td>
          <td>0.175184</td>
        </tr>
        <tr>
          <th>2</th>
          <td>21.869575</td>
          <td>21.602492</td>
          <td>25.529264</td>
          <td>23.068049</td>
          <td>17.447842</td>
          <td>22.680342</td>
          <td>28.544087</td>
          <td>2.600448</td>
          <td>21.927401</td>
          <td>0.020720</td>
          <td>23.071330</td>
          <td>0.062246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.589882</td>
          <td>27.576717</td>
          <td>21.552096</td>
          <td>20.573637</td>
          <td>16.624106</td>
          <td>22.520046</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.576334</td>
          <td>0.036636</td>
          <td>19.086367</td>
          <td>0.005256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.972696</td>
          <td>20.223403</td>
          <td>17.737244</td>
          <td>22.827789</td>
          <td>26.691005</td>
          <td>20.327054</td>
          <td>22.715727</td>
          <td>0.049571</td>
          <td>20.131581</td>
          <td>0.006323</td>
          <td>19.912594</td>
          <td>0.006085</td>
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
          <td>21.400958</td>
          <td>19.909868</td>
          <td>26.160232</td>
          <td>20.127856</td>
          <td>22.536843</td>
          <td>24.035025</td>
          <td>19.377452</td>
          <td>0.005514</td>
          <td>23.846111</td>
          <td>0.112928</td>
          <td>20.318335</td>
          <td>0.007098</td>
        </tr>
        <tr>
          <th>996</th>
          <td>19.485905</td>
          <td>21.068944</td>
          <td>27.823563</td>
          <td>20.002640</td>
          <td>23.165620</td>
          <td>19.545039</td>
          <td>20.029617</td>
          <td>0.006553</td>
          <td>28.465572</td>
          <td>2.350498</td>
          <td>22.249122</td>
          <td>0.029950</td>
        </tr>
        <tr>
          <th>997</th>
          <td>25.021091</td>
          <td>25.239927</td>
          <td>21.660126</td>
          <td>25.513290</td>
          <td>23.697363</td>
          <td>23.979353</td>
          <td>25.377196</td>
          <td>0.468390</td>
          <td>23.943453</td>
          <td>0.122925</td>
          <td>19.545584</td>
          <td>0.005579</td>
        </tr>
        <tr>
          <th>998</th>
          <td>26.004101</td>
          <td>23.597287</td>
          <td>19.289497</td>
          <td>22.935598</td>
          <td>24.964263</td>
          <td>21.316885</td>
          <td>26.484681</td>
          <td>0.993071</td>
          <td>20.177384</td>
          <td>0.006426</td>
          <td>23.461759</td>
          <td>0.087982</td>
        </tr>
        <tr>
          <th>999</th>
          <td>12.880496</td>
          <td>21.223576</td>
          <td>23.169376</td>
          <td>24.949464</td>
          <td>21.582061</td>
          <td>24.592514</td>
          <td>24.759384</td>
          <td>0.289338</td>
          <td>23.573343</td>
          <td>0.088885</td>
          <td>24.822767</td>
          <td>0.280884</td>
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


