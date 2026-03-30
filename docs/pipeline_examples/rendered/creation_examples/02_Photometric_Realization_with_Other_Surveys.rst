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
          <td>19.798390</td>
          <td>14.422915</td>
          <td>20.211869</td>
          <td>19.105667</td>
          <td>21.069618</td>
          <td>24.059826</td>
          <td>19.377839</td>
          <td>25.310530</td>
          <td>21.753734</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.795655</td>
          <td>18.998452</td>
          <td>26.302669</td>
          <td>22.817169</td>
          <td>20.163512</td>
          <td>20.504588</td>
          <td>23.965031</td>
          <td>17.883764</td>
          <td>19.011135</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.625052</td>
          <td>30.445840</td>
          <td>19.090848</td>
          <td>21.922084</td>
          <td>20.266353</td>
          <td>24.846552</td>
          <td>21.280508</td>
          <td>23.715372</td>
          <td>20.388556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.474125</td>
          <td>22.786980</td>
          <td>21.579549</td>
          <td>26.759474</td>
          <td>21.552674</td>
          <td>22.085975</td>
          <td>29.134318</td>
          <td>23.800487</td>
          <td>22.489767</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.120431</td>
          <td>29.867282</td>
          <td>23.765124</td>
          <td>22.833673</td>
          <td>23.500949</td>
          <td>23.635034</td>
          <td>19.314703</td>
          <td>19.324829</td>
          <td>24.120844</td>
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
          <td>23.596092</td>
          <td>27.577209</td>
          <td>19.628775</td>
          <td>27.318086</td>
          <td>22.072626</td>
          <td>25.086841</td>
          <td>23.960719</td>
          <td>26.122226</td>
          <td>25.432280</td>
        </tr>
        <tr>
          <th>996</th>
          <td>15.725450</td>
          <td>29.954225</td>
          <td>22.544118</td>
          <td>26.069699</td>
          <td>23.089159</td>
          <td>25.962455</td>
          <td>25.074829</td>
          <td>25.914264</td>
          <td>21.481986</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.053269</td>
          <td>27.488969</td>
          <td>19.280517</td>
          <td>27.156197</td>
          <td>21.735883</td>
          <td>24.731386</td>
          <td>18.422529</td>
          <td>26.823820</td>
          <td>21.199971</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.173858</td>
          <td>31.942683</td>
          <td>19.434568</td>
          <td>21.420930</td>
          <td>20.302463</td>
          <td>23.193555</td>
          <td>25.904415</td>
          <td>22.436282</td>
          <td>26.952297</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.121978</td>
          <td>24.337055</td>
          <td>25.128113</td>
          <td>23.834809</td>
          <td>24.669095</td>
          <td>20.445682</td>
          <td>20.975932</td>
          <td>23.087508</td>
          <td>25.398213</td>
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
          <td>19.792264</td>
          <td>0.005212</td>
          <td>14.425088</td>
          <td>0.005000</td>
          <td>20.216928</td>
          <td>0.005032</td>
          <td>19.098704</td>
          <td>0.005015</td>
          <td>21.076640</td>
          <td>0.005858</td>
          <td>24.054366</td>
          <td>0.099128</td>
          <td>19.377839</td>
          <td>25.310530</td>
          <td>21.753734</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.794550</td>
          <td>0.008230</td>
          <td>18.993878</td>
          <td>0.005012</td>
          <td>26.281722</td>
          <td>0.101018</td>
          <td>22.805644</td>
          <td>0.008860</td>
          <td>20.162009</td>
          <td>0.005199</td>
          <td>20.494593</td>
          <td>0.006461</td>
          <td>23.965031</td>
          <td>17.883764</td>
          <td>19.011135</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.704764</td>
          <td>0.083667</td>
          <td>27.720392</td>
          <td>0.379034</td>
          <td>19.084210</td>
          <td>0.005008</td>
          <td>21.921803</td>
          <td>0.006019</td>
          <td>20.267137</td>
          <td>0.005234</td>
          <td>25.120804</td>
          <td>0.246348</td>
          <td>21.280508</td>
          <td>23.715372</td>
          <td>20.388556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.438820</td>
          <td>0.744293</td>
          <td>22.780687</td>
          <td>0.007137</td>
          <td>21.581076</td>
          <td>0.005243</td>
          <td>26.630808</td>
          <td>0.218991</td>
          <td>21.557458</td>
          <td>0.006820</td>
          <td>22.095049</td>
          <td>0.017804</td>
          <td>29.134318</td>
          <td>23.800487</td>
          <td>22.489767</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.089317</td>
          <td>0.117003</td>
          <td>28.702931</td>
          <td>0.769920</td>
          <td>23.765232</td>
          <td>0.011660</td>
          <td>22.829469</td>
          <td>0.008990</td>
          <td>23.527618</td>
          <td>0.027624</td>
          <td>23.551099</td>
          <td>0.063590</td>
          <td>19.314703</td>
          <td>19.324829</td>
          <td>24.120844</td>
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
          <td>23.585353</td>
          <td>0.031352</td>
          <td>28.353280</td>
          <td>0.606415</td>
          <td>19.632379</td>
          <td>0.005015</td>
          <td>27.188447</td>
          <td>0.344434</td>
          <td>22.074143</td>
          <td>0.008877</td>
          <td>24.892294</td>
          <td>0.203741</td>
          <td>23.960719</td>
          <td>26.122226</td>
          <td>25.432280</td>
        </tr>
        <tr>
          <th>996</th>
          <td>15.724797</td>
          <td>0.005003</td>
          <td>29.431090</td>
          <td>1.199370</td>
          <td>22.543371</td>
          <td>0.006137</td>
          <td>26.173367</td>
          <td>0.148669</td>
          <td>23.066954</td>
          <td>0.018603</td>
          <td>27.686856</td>
          <td>1.421793</td>
          <td>25.074829</td>
          <td>25.914264</td>
          <td>21.481986</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.062199</td>
          <td>0.020104</td>
          <td>27.787985</td>
          <td>0.399375</td>
          <td>19.279800</td>
          <td>0.005010</td>
          <td>27.061383</td>
          <td>0.311360</td>
          <td>21.752307</td>
          <td>0.007440</td>
          <td>24.658608</td>
          <td>0.167215</td>
          <td>18.422529</td>
          <td>26.823820</td>
          <td>21.199971</td>
        </tr>
        <tr>
          <th>998</th>
          <td>34.642802</td>
          <td>7.188080</td>
          <td>inf</td>
          <td>inf</td>
          <td>19.435680</td>
          <td>0.005011</td>
          <td>21.420740</td>
          <td>0.005455</td>
          <td>20.306334</td>
          <td>0.005250</td>
          <td>23.134334</td>
          <td>0.043931</td>
          <td>25.904415</td>
          <td>22.436282</td>
          <td>26.952297</td>
        </tr>
        <tr>
          <th>999</th>
          <td>25.667242</td>
          <td>0.191782</td>
          <td>24.336318</td>
          <td>0.020814</td>
          <td>25.136927</td>
          <td>0.036687</td>
          <td>23.860990</td>
          <td>0.019574</td>
          <td>24.568837</td>
          <td>0.069385</td>
          <td>20.440008</td>
          <td>0.006342</td>
          <td>20.975932</td>
          <td>23.087508</td>
          <td>25.398213</td>
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
          <td>19.798390</td>
          <td>14.422915</td>
          <td>20.211869</td>
          <td>19.105667</td>
          <td>21.069618</td>
          <td>24.059826</td>
          <td>19.377676</td>
          <td>0.005004</td>
          <td>25.301911</td>
          <td>0.076388</td>
          <td>21.757780</td>
          <td>0.005835</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.795655</td>
          <td>18.998452</td>
          <td>26.302669</td>
          <td>22.817169</td>
          <td>20.163512</td>
          <td>20.504588</td>
          <td>23.956124</td>
          <td>0.013990</td>
          <td>17.890863</td>
          <td>0.005001</td>
          <td>19.011605</td>
          <td>0.005006</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.625052</td>
          <td>30.445840</td>
          <td>19.090848</td>
          <td>21.922084</td>
          <td>20.266353</td>
          <td>24.846552</td>
          <td>21.283466</td>
          <td>0.005124</td>
          <td>23.686212</td>
          <td>0.018368</td>
          <td>20.395428</td>
          <td>0.005073</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.474125</td>
          <td>22.786980</td>
          <td>21.579549</td>
          <td>26.759474</td>
          <td>21.552674</td>
          <td>22.085975</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.797233</td>
          <td>0.020191</td>
          <td>22.492024</td>
          <td>0.007742</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.120431</td>
          <td>29.867282</td>
          <td>23.765124</td>
          <td>22.833673</td>
          <td>23.500949</td>
          <td>23.635034</td>
          <td>19.317291</td>
          <td>0.005003</td>
          <td>19.330777</td>
          <td>0.005010</td>
          <td>24.119725</td>
          <td>0.026722</td>
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
          <td>23.596092</td>
          <td>27.577209</td>
          <td>19.628775</td>
          <td>27.318086</td>
          <td>22.072626</td>
          <td>25.086841</td>
          <td>23.946326</td>
          <td>0.013881</td>
          <td>26.018282</td>
          <td>0.143016</td>
          <td>25.407761</td>
          <td>0.083886</td>
        </tr>
        <tr>
          <th>996</th>
          <td>15.725450</td>
          <td>29.954225</td>
          <td>22.544118</td>
          <td>26.069699</td>
          <td>23.089159</td>
          <td>25.962455</td>
          <td>25.026896</td>
          <td>0.035061</td>
          <td>26.045683</td>
          <td>0.146431</td>
          <td>21.481970</td>
          <td>0.005518</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.053269</td>
          <td>27.488969</td>
          <td>19.280517</td>
          <td>27.156197</td>
          <td>21.735883</td>
          <td>24.731386</td>
          <td>18.418214</td>
          <td>0.005001</td>
          <td>27.448234</td>
          <td>0.458330</td>
          <td>21.205209</td>
          <td>0.005317</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.173858</td>
          <td>31.942683</td>
          <td>19.434568</td>
          <td>21.420930</td>
          <td>20.302463</td>
          <td>23.193555</td>
          <td>25.976163</td>
          <td>0.081576</td>
          <td>22.435426</td>
          <td>0.007515</td>
          <td>27.357623</td>
          <td>0.427977</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.121978</td>
          <td>24.337055</td>
          <td>25.128113</td>
          <td>23.834809</td>
          <td>24.669095</td>
          <td>20.445682</td>
          <td>20.984091</td>
          <td>0.005072</td>
          <td>23.084810</td>
          <td>0.011349</td>
          <td>25.450820</td>
          <td>0.087137</td>
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
          <td>19.798390</td>
          <td>14.422915</td>
          <td>20.211869</td>
          <td>19.105667</td>
          <td>21.069618</td>
          <td>24.059826</td>
          <td>19.367090</td>
          <td>0.005504</td>
          <td>24.882143</td>
          <td>0.271759</td>
          <td>21.730176</td>
          <td>0.019066</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.795655</td>
          <td>18.998452</td>
          <td>26.302669</td>
          <td>22.817169</td>
          <td>20.163512</td>
          <td>20.504588</td>
          <td>24.313966</td>
          <td>0.200341</td>
          <td>17.889705</td>
          <td>0.005024</td>
          <td>19.002650</td>
          <td>0.005220</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.625052</td>
          <td>30.445840</td>
          <td>19.090848</td>
          <td>21.922084</td>
          <td>20.266353</td>
          <td>24.846552</td>
          <td>21.274005</td>
          <td>0.014192</td>
          <td>23.633794</td>
          <td>0.093747</td>
          <td>20.377512</td>
          <td>0.007301</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30.474125</td>
          <td>22.786980</td>
          <td>21.579549</td>
          <td>26.759474</td>
          <td>21.552674</td>
          <td>22.085975</td>
          <td>inf</td>
          <td>inf</td>
          <td>23.773424</td>
          <td>0.105973</td>
          <td>22.508592</td>
          <td>0.037702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.120431</td>
          <td>29.867282</td>
          <td>23.765124</td>
          <td>22.833673</td>
          <td>23.500949</td>
          <td>23.635034</td>
          <td>19.311719</td>
          <td>0.005457</td>
          <td>19.331900</td>
          <td>0.005332</td>
          <td>23.967239</td>
          <td>0.136851</td>
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
          <td>23.596092</td>
          <td>27.577209</td>
          <td>19.628775</td>
          <td>27.318086</td>
          <td>22.072626</td>
          <td>25.086841</td>
          <td>24.119239</td>
          <td>0.169898</td>
          <td>28.884996</td>
          <td>2.729242</td>
          <td>25.324187</td>
          <td>0.417198</td>
        </tr>
        <tr>
          <th>996</th>
          <td>15.725450</td>
          <td>29.954225</td>
          <td>22.544118</td>
          <td>26.069699</td>
          <td>23.089159</td>
          <td>25.962455</td>
          <td>25.203046</td>
          <td>0.410498</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.480253</td>
          <td>0.015469</td>
        </tr>
        <tr>
          <th>997</th>
          <td>23.053269</td>
          <td>27.488969</td>
          <td>19.280517</td>
          <td>27.156197</td>
          <td>21.735883</td>
          <td>24.731386</td>
          <td>18.423579</td>
          <td>0.005092</td>
          <td>26.708217</td>
          <td>1.007238</td>
          <td>21.208611</td>
          <td>0.012461</td>
        </tr>
        <tr>
          <th>998</th>
          <td>28.173858</td>
          <td>31.942683</td>
          <td>19.434568</td>
          <td>21.420930</td>
          <td>20.302463</td>
          <td>23.193555</td>
          <td>inf</td>
          <td>inf</td>
          <td>22.422205</td>
          <td>0.031951</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>26.121978</td>
          <td>24.337055</td>
          <td>25.128113</td>
          <td>23.834809</td>
          <td>24.669095</td>
          <td>20.445682</td>
          <td>20.981938</td>
          <td>0.011325</td>
          <td>23.060343</td>
          <td>0.056390</td>
          <td>24.864381</td>
          <td>0.290509</td>
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


