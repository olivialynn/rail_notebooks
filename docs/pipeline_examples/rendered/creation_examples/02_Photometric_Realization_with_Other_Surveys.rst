Photometric error stage demo
----------------------------

author: Tianqing Zhang, John-Franklin Crenshaw

This notebook demonstrate the use of
``rail.creation.degraders.photometric_errors``, which adds column for
the photometric noise to the catalog based on the package PhotErr
developed by John-Franklin Crenshaw. The RAIL stage PhotoErrorModel
inherit from the Noisifier base classes, and the LSST, Roman, Euclid
child classes inherit from the PhotoErrorModel

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization_with_Other_Surveys.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.creation.degraders.photometric_errors import RomanErrorModel
    from rail.creation.degraders.photometric_errors import EuclidErrorModel
    
    from rail.core.data import PqHandle
    from rail.core.stage import RailStage
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    


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
          <td>20.936982</td>
          <td>20.736446</td>
          <td>26.616009</td>
          <td>29.956404</td>
          <td>21.661493</td>
          <td>22.237001</td>
          <td>25.269038</td>
          <td>23.795367</td>
          <td>22.027562</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.098666</td>
          <td>19.682055</td>
          <td>20.016410</td>
          <td>21.446775</td>
          <td>20.803682</td>
          <td>23.941803</td>
          <td>25.731604</td>
          <td>22.975758</td>
          <td>19.372556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.872042</td>
          <td>20.432118</td>
          <td>25.907850</td>
          <td>18.545273</td>
          <td>23.991779</td>
          <td>21.848953</td>
          <td>27.267761</td>
          <td>27.419774</td>
          <td>23.728812</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.502026</td>
          <td>25.194770</td>
          <td>24.631939</td>
          <td>22.949805</td>
          <td>23.591656</td>
          <td>27.400602</td>
          <td>25.889720</td>
          <td>23.973273</td>
          <td>24.312091</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.835505</td>
          <td>25.775658</td>
          <td>20.469206</td>
          <td>24.252495</td>
          <td>23.439923</td>
          <td>21.786737</td>
          <td>20.983546</td>
          <td>24.945897</td>
          <td>23.632077</td>
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
          <td>17.046286</td>
          <td>22.994568</td>
          <td>24.858258</td>
          <td>23.644375</td>
          <td>17.875426</td>
          <td>27.813937</td>
          <td>20.340302</td>
          <td>21.688638</td>
          <td>29.161574</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.360019</td>
          <td>22.693263</td>
          <td>25.508198</td>
          <td>21.208447</td>
          <td>23.247426</td>
          <td>23.639363</td>
          <td>22.536559</td>
          <td>25.302439</td>
          <td>26.447996</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.354647</td>
          <td>25.430417</td>
          <td>23.373977</td>
          <td>24.852231</td>
          <td>24.214054</td>
          <td>24.819854</td>
          <td>23.124508</td>
          <td>23.719451</td>
          <td>24.371762</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.160358</td>
          <td>21.115443</td>
          <td>17.181666</td>
          <td>23.991828</td>
          <td>20.555711</td>
          <td>22.442137</td>
          <td>18.039826</td>
          <td>17.891280</td>
          <td>24.137021</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882166</td>
          <td>23.641862</td>
          <td>26.522531</td>
          <td>24.901472</td>
          <td>26.490984</td>
          <td>18.124788</td>
          <td>28.611585</td>
          <td>23.898042</td>
          <td>17.689596</td>
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
          <td>20.926752</td>
          <td>0.005974</td>
          <td>20.734163</td>
          <td>0.005100</td>
          <td>26.609191</td>
          <td>0.134328</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.667739</td>
          <td>0.007151</td>
          <td>22.231314</td>
          <td>0.019962</td>
          <td>25.269038</td>
          <td>23.795367</td>
          <td>22.027562</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.098033</td>
          <td>0.273889</td>
          <td>19.683547</td>
          <td>0.005026</td>
          <td>20.023765</td>
          <td>0.005024</td>
          <td>21.443468</td>
          <td>0.005472</td>
          <td>20.801365</td>
          <td>0.005552</td>
          <td>23.859068</td>
          <td>0.083492</td>
          <td>25.731604</td>
          <td>22.975758</td>
          <td>19.372556</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.930684</td>
          <td>0.101933</td>
          <td>20.437230</td>
          <td>0.005067</td>
          <td>25.881520</td>
          <td>0.071007</td>
          <td>18.538094</td>
          <td>0.005008</td>
          <td>24.031218</td>
          <td>0.043066</td>
          <td>21.864809</td>
          <td>0.014754</td>
          <td>27.267761</td>
          <td>27.419774</td>
          <td>23.728812</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.524660</td>
          <td>0.013152</td>
          <td>25.290627</td>
          <td>0.047873</td>
          <td>24.627531</td>
          <td>0.023484</td>
          <td>22.961157</td>
          <td>0.009782</td>
          <td>23.641605</td>
          <td>0.030524</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.889720</td>
          <td>23.973273</td>
          <td>24.312091</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.834251</td>
          <td>0.016701</td>
          <td>25.749742</td>
          <td>0.071874</td>
          <td>20.465320</td>
          <td>0.005045</td>
          <td>24.240876</td>
          <td>0.027143</td>
          <td>23.410582</td>
          <td>0.024948</td>
          <td>21.787239</td>
          <td>0.013876</td>
          <td>20.983546</td>
          <td>24.945897</td>
          <td>23.632077</td>
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
          <td>17.058188</td>
          <td>0.005011</td>
          <td>23.018488</td>
          <td>0.008012</td>
          <td>24.906717</td>
          <td>0.029949</td>
          <td>23.609428</td>
          <td>0.015883</td>
          <td>17.865423</td>
          <td>0.005008</td>
          <td>26.118840</td>
          <td>0.535938</td>
          <td>20.340302</td>
          <td>21.688638</td>
          <td>29.161574</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.266489</td>
          <td>0.136350</td>
          <td>22.681358</td>
          <td>0.006845</td>
          <td>25.552553</td>
          <td>0.053040</td>
          <td>21.215253</td>
          <td>0.005326</td>
          <td>23.229863</td>
          <td>0.021353</td>
          <td>23.681125</td>
          <td>0.071351</td>
          <td>22.536559</td>
          <td>25.302439</td>
          <td>26.447996</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.306551</td>
          <td>0.058961</td>
          <td>25.441455</td>
          <td>0.054715</td>
          <td>23.378262</td>
          <td>0.008967</td>
          <td>24.924734</td>
          <td>0.049654</td>
          <td>24.195332</td>
          <td>0.049818</td>
          <td>24.968404</td>
          <td>0.217128</td>
          <td>23.124508</td>
          <td>23.719451</td>
          <td>24.371762</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.164894</td>
          <td>0.010233</td>
          <td>21.119030</td>
          <td>0.005172</td>
          <td>17.180436</td>
          <td>0.005001</td>
          <td>23.971507</td>
          <td>0.021502</td>
          <td>20.567903</td>
          <td>0.005379</td>
          <td>22.413905</td>
          <td>0.023334</td>
          <td>18.039826</td>
          <td>17.891280</td>
          <td>24.137021</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.888019</td>
          <td>0.008657</td>
          <td>23.651274</td>
          <td>0.012031</td>
          <td>26.609783</td>
          <td>0.134396</td>
          <td>24.958350</td>
          <td>0.051159</td>
          <td>26.320116</td>
          <td>0.309277</td>
          <td>18.127229</td>
          <td>0.005038</td>
          <td>28.611585</td>
          <td>23.898042</td>
          <td>17.689596</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_7_0.png


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
          <td>20.936982</td>
          <td>20.736446</td>
          <td>26.616009</td>
          <td>29.956404</td>
          <td>21.661493</td>
          <td>22.237001</td>
          <td>25.345613</td>
          <td>0.046567</td>
          <td>23.789682</td>
          <td>0.020060</td>
          <td>22.028382</td>
          <td>0.006316</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.098666</td>
          <td>19.682055</td>
          <td>20.016410</td>
          <td>21.446775</td>
          <td>20.803682</td>
          <td>23.941803</td>
          <td>25.739564</td>
          <td>0.066139</td>
          <td>22.971047</td>
          <td>0.010452</td>
          <td>19.372723</td>
          <td>0.005011</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.872042</td>
          <td>20.432118</td>
          <td>25.907850</td>
          <td>18.545273</td>
          <td>23.991779</td>
          <td>21.848953</td>
          <td>27.327535</td>
          <td>0.259905</td>
          <td>28.383176</td>
          <td>0.876841</td>
          <td>23.718637</td>
          <td>0.018880</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.502026</td>
          <td>25.194770</td>
          <td>24.631939</td>
          <td>22.949805</td>
          <td>23.591656</td>
          <td>27.400602</td>
          <td>25.848202</td>
          <td>0.072837</td>
          <td>23.986017</td>
          <td>0.023771</td>
          <td>24.266148</td>
          <td>0.030404</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.835505</td>
          <td>25.775658</td>
          <td>20.469206</td>
          <td>24.252495</td>
          <td>23.439923</td>
          <td>21.786737</td>
          <td>20.986772</td>
          <td>0.005072</td>
          <td>24.984856</td>
          <td>0.057634</td>
          <td>23.639085</td>
          <td>0.017652</td>
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
          <td>17.046286</td>
          <td>22.994568</td>
          <td>24.858258</td>
          <td>23.644375</td>
          <td>17.875426</td>
          <td>27.813937</td>
          <td>20.350109</td>
          <td>0.005022</td>
          <td>21.688857</td>
          <td>0.005742</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.360019</td>
          <td>22.693263</td>
          <td>25.508198</td>
          <td>21.208447</td>
          <td>23.247426</td>
          <td>23.639363</td>
          <td>22.537002</td>
          <td>0.006130</td>
          <td>25.337417</td>
          <td>0.078827</td>
          <td>26.091184</td>
          <td>0.152273</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.354647</td>
          <td>25.430417</td>
          <td>23.373977</td>
          <td>24.852231</td>
          <td>24.214054</td>
          <td>24.819854</td>
          <td>23.128171</td>
          <td>0.007895</td>
          <td>23.712598</td>
          <td>0.018784</td>
          <td>24.336811</td>
          <td>0.032367</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.160358</td>
          <td>21.115443</td>
          <td>17.181666</td>
          <td>23.991828</td>
          <td>20.555711</td>
          <td>22.442137</td>
          <td>18.039348</td>
          <td>0.005000</td>
          <td>17.890437</td>
          <td>0.005001</td>
          <td>24.137525</td>
          <td>0.027144</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882166</td>
          <td>23.641862</td>
          <td>26.522531</td>
          <td>24.901472</td>
          <td>26.490984</td>
          <td>18.124788</td>
          <td>29.379064</td>
          <td>1.113696</td>
          <td>23.887634</td>
          <td>0.021825</td>
          <td>17.683589</td>
          <td>0.005000</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_13_0.png


The Euclid error model adds noise to YJH bands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    errorModel_Euclid = EuclidErrorModel.make_stage(name="error_model")
    
    samples_w_errs_Euclid = errorModel_Euclid(data_truth)
    samples_w_errs_Euclid()


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
          <td>20.936982</td>
          <td>20.736446</td>
          <td>26.616009</td>
          <td>29.956404</td>
          <td>21.661493</td>
          <td>22.237001</td>
          <td>25.259060</td>
          <td>0.428445</td>
          <td>23.879324</td>
          <td>0.116249</td>
          <td>22.069706</td>
          <td>0.025575</td>
        </tr>
        <tr>
          <th>1</th>
          <td>26.098666</td>
          <td>19.682055</td>
          <td>20.016410</td>
          <td>21.446775</td>
          <td>20.803682</td>
          <td>23.941803</td>
          <td>25.312775</td>
          <td>0.446251</td>
          <td>22.985199</td>
          <td>0.052737</td>
          <td>19.366002</td>
          <td>0.005422</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.872042</td>
          <td>20.432118</td>
          <td>25.907850</td>
          <td>18.545273</td>
          <td>23.991779</td>
          <td>21.848953</td>
          <td>26.400939</td>
          <td>0.943660</td>
          <td>25.762223</td>
          <td>0.536866</td>
          <td>23.661974</td>
          <td>0.104916</td>
        </tr>
        <tr>
          <th>3</th>
          <td>22.502026</td>
          <td>25.194770</td>
          <td>24.631939</td>
          <td>22.949805</td>
          <td>23.591656</td>
          <td>27.400602</td>
          <td>26.072965</td>
          <td>0.765425</td>
          <td>23.926829</td>
          <td>0.121160</td>
          <td>24.487554</td>
          <td>0.213088</td>
        </tr>
        <tr>
          <th>4</th>
          <td>22.835505</td>
          <td>25.775658</td>
          <td>20.469206</td>
          <td>24.252495</td>
          <td>23.439923</td>
          <td>21.786737</td>
          <td>20.992475</td>
          <td>0.011414</td>
          <td>24.858193</td>
          <td>0.266502</td>
          <td>23.744865</td>
          <td>0.112805</td>
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
          <td>17.046286</td>
          <td>22.994568</td>
          <td>24.858258</td>
          <td>23.644375</td>
          <td>17.875426</td>
          <td>27.813937</td>
          <td>20.339386</td>
          <td>0.007531</td>
          <td>21.680684</td>
          <td>0.016809</td>
          <td>25.721085</td>
          <td>0.560209</td>
        </tr>
        <tr>
          <th>996</th>
          <td>25.360019</td>
          <td>22.693263</td>
          <td>25.508198</td>
          <td>21.208447</td>
          <td>23.247426</td>
          <td>23.639363</td>
          <td>22.546445</td>
          <td>0.042627</td>
          <td>25.612992</td>
          <td>0.481067</td>
          <td>26.019007</td>
          <td>0.690270</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.354647</td>
          <td>25.430417</td>
          <td>23.373977</td>
          <td>24.852231</td>
          <td>24.214054</td>
          <td>24.819854</td>
          <td>23.045997</td>
          <td>0.066519</td>
          <td>23.659782</td>
          <td>0.095915</td>
          <td>24.561232</td>
          <td>0.226584</td>
        </tr>
        <tr>
          <th>998</th>
          <td>22.160358</td>
          <td>21.115443</td>
          <td>17.181666</td>
          <td>23.991828</td>
          <td>20.555711</td>
          <td>22.442137</td>
          <td>18.047129</td>
          <td>0.005046</td>
          <td>17.894755</td>
          <td>0.005024</td>
          <td>24.406603</td>
          <td>0.199104</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.882166</td>
          <td>23.641862</td>
          <td>26.522531</td>
          <td>24.901472</td>
          <td>26.490984</td>
          <td>18.124788</td>
          <td>25.339798</td>
          <td>0.455433</td>
          <td>24.064709</td>
          <td>0.136552</td>
          <td>17.680898</td>
          <td>0.005020</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


