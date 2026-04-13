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
          <td>26.684448</td>
          <td>24.401359</td>
          <td>20.420772</td>
          <td>25.089791</td>
          <td>20.710853</td>
          <td>20.968550</td>
          <td>23.092407</td>
          <td>27.248114</td>
          <td>29.934624</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.100038</td>
          <td>25.270937</td>
          <td>25.721887</td>
          <td>23.240645</td>
          <td>22.198033</td>
          <td>24.458891</td>
          <td>22.625371</td>
          <td>25.582538</td>
          <td>27.928136</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.076120</td>
          <td>26.353996</td>
          <td>25.149638</td>
          <td>21.282377</td>
          <td>23.989843</td>
          <td>19.014465</td>
          <td>25.183478</td>
          <td>24.149752</td>
          <td>22.783929</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.977985</td>
          <td>22.317156</td>
          <td>20.762362</td>
          <td>21.380671</td>
          <td>19.394189</td>
          <td>22.464806</td>
          <td>24.773643</td>
          <td>23.124688</td>
          <td>23.086461</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.883862</td>
          <td>26.435798</td>
          <td>22.988800</td>
          <td>23.188953</td>
          <td>23.980104</td>
          <td>18.081590</td>
          <td>20.387366</td>
          <td>24.999970</td>
          <td>22.432658</td>
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
          <td>23.026208</td>
          <td>24.494611</td>
          <td>22.153018</td>
          <td>25.991057</td>
          <td>24.085272</td>
          <td>23.239008</td>
          <td>24.821031</td>
          <td>19.157318</td>
          <td>13.640717</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.126011</td>
          <td>26.538878</td>
          <td>24.576373</td>
          <td>29.084697</td>
          <td>23.569826</td>
          <td>21.159550</td>
          <td>22.866887</td>
          <td>23.001004</td>
          <td>22.864816</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.361038</td>
          <td>22.098051</td>
          <td>20.950619</td>
          <td>26.458797</td>
          <td>25.021527</td>
          <td>20.755995</td>
          <td>20.419510</td>
          <td>18.083690</td>
          <td>24.423932</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.045403</td>
          <td>18.855375</td>
          <td>28.427942</td>
          <td>24.013015</td>
          <td>23.892424</td>
          <td>18.258662</td>
          <td>27.230938</td>
          <td>22.597613</td>
          <td>22.869280</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.357597</td>
          <td>21.770590</td>
          <td>15.411428</td>
          <td>23.404007</td>
          <td>25.957345</td>
          <td>19.060094</td>
          <td>20.275973</td>
          <td>26.126130</td>
          <td>30.363959</td>
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
          <td>26.216026</td>
          <td>0.301262</td>
          <td>24.408111</td>
          <td>0.022123</td>
          <td>20.419686</td>
          <td>0.005042</td>
          <td>25.076552</td>
          <td>0.056820</td>
          <td>20.704911</td>
          <td>0.005472</td>
          <td>20.976417</td>
          <td>0.008011</td>
          <td>23.092407</td>
          <td>27.248114</td>
          <td>29.934624</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.110559</td>
          <td>0.006260</td>
          <td>25.346092</td>
          <td>0.050283</td>
          <td>25.695534</td>
          <td>0.060218</td>
          <td>23.236220</td>
          <td>0.011870</td>
          <td>22.215429</td>
          <td>0.009709</td>
          <td>24.428646</td>
          <td>0.137286</td>
          <td>22.625371</td>
          <td>25.582538</td>
          <td>27.928136</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.073119</td>
          <td>0.005304</td>
          <td>26.424804</td>
          <td>0.129786</td>
          <td>25.103963</td>
          <td>0.035633</td>
          <td>21.279784</td>
          <td>0.005362</td>
          <td>23.981039</td>
          <td>0.041192</td>
          <td>19.021231</td>
          <td>0.005141</td>
          <td>25.183478</td>
          <td>24.149752</td>
          <td>22.783929</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.973071</td>
          <td>0.005267</td>
          <td>22.327045</td>
          <td>0.006080</td>
          <td>20.766546</td>
          <td>0.005069</td>
          <td>21.379888</td>
          <td>0.005426</td>
          <td>19.399654</td>
          <td>0.005062</td>
          <td>22.486542</td>
          <td>0.024846</td>
          <td>24.773643</td>
          <td>23.124688</td>
          <td>23.086461</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.894852</td>
          <td>0.005242</td>
          <td>26.284412</td>
          <td>0.114904</td>
          <td>22.988275</td>
          <td>0.007257</td>
          <td>23.185207</td>
          <td>0.011434</td>
          <td>23.962799</td>
          <td>0.040531</td>
          <td>18.082771</td>
          <td>0.005036</td>
          <td>20.387366</td>
          <td>24.999970</td>
          <td>22.432658</td>
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
          <td>23.049177</td>
          <td>0.019888</td>
          <td>24.529217</td>
          <td>0.024544</td>
          <td>22.160971</td>
          <td>0.005617</td>
          <td>26.238517</td>
          <td>0.157209</td>
          <td>24.016755</td>
          <td>0.042517</td>
          <td>23.201608</td>
          <td>0.046633</td>
          <td>24.821031</td>
          <td>19.157318</td>
          <td>13.640717</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.119925</td>
          <td>0.006276</td>
          <td>26.515341</td>
          <td>0.140334</td>
          <td>24.576557</td>
          <td>0.022476</td>
          <td>28.274721</td>
          <td>0.760646</td>
          <td>23.570242</td>
          <td>0.028673</td>
          <td>21.164267</td>
          <td>0.008930</td>
          <td>22.866887</td>
          <td>23.001004</td>
          <td>22.864816</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.357234</td>
          <td>0.011653</td>
          <td>22.104285</td>
          <td>0.005767</td>
          <td>20.953042</td>
          <td>0.005091</td>
          <td>27.006549</td>
          <td>0.297956</td>
          <td>24.972603</td>
          <td>0.099039</td>
          <td>20.768145</td>
          <td>0.007217</td>
          <td>20.419510</td>
          <td>18.083690</td>
          <td>24.423932</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.047446</td>
          <td>0.005029</td>
          <td>18.849675</td>
          <td>0.005010</td>
          <td>29.439220</td>
          <td>1.110648</td>
          <td>23.995478</td>
          <td>0.021948</td>
          <td>23.904455</td>
          <td>0.038490</td>
          <td>18.262741</td>
          <td>0.005046</td>
          <td>27.230938</td>
          <td>22.597613</td>
          <td>22.869280</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.356232</td>
          <td>0.005124</td>
          <td>21.771510</td>
          <td>0.005459</td>
          <td>15.420060</td>
          <td>0.005000</td>
          <td>23.418978</td>
          <td>0.013643</td>
          <td>26.402293</td>
          <td>0.330218</td>
          <td>19.050206</td>
          <td>0.005147</td>
          <td>20.275973</td>
          <td>26.126130</td>
          <td>30.363959</td>
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
          <td>26.684448</td>
          <td>24.401359</td>
          <td>20.420772</td>
          <td>25.089791</td>
          <td>20.710853</td>
          <td>20.968550</td>
          <td>23.089362</td>
          <td>0.007731</td>
          <td>30.112923</td>
          <td>2.130013</td>
          <td>27.910862</td>
          <td>0.640731</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.100038</td>
          <td>25.270937</td>
          <td>25.721887</td>
          <td>23.240645</td>
          <td>22.198033</td>
          <td>24.458891</td>
          <td>22.629045</td>
          <td>0.006317</td>
          <td>25.593714</td>
          <td>0.098818</td>
          <td>28.534877</td>
          <td>0.963497</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.076120</td>
          <td>26.353996</td>
          <td>25.149638</td>
          <td>21.282377</td>
          <td>23.989843</td>
          <td>19.014465</td>
          <td>25.181404</td>
          <td>0.040227</td>
          <td>24.134017</td>
          <td>0.027060</td>
          <td>22.780329</td>
          <td>0.009184</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.977985</td>
          <td>22.317156</td>
          <td>20.762362</td>
          <td>21.380671</td>
          <td>19.394189</td>
          <td>22.464806</td>
          <td>24.775468</td>
          <td>0.028065</td>
          <td>23.107896</td>
          <td>0.011545</td>
          <td>23.077796</td>
          <td>0.011290</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.883862</td>
          <td>26.435798</td>
          <td>22.988800</td>
          <td>23.188953</td>
          <td>23.980104</td>
          <td>18.081590</td>
          <td>20.393029</td>
          <td>0.005024</td>
          <td>25.073791</td>
          <td>0.062382</td>
          <td>22.436481</td>
          <td>0.007519</td>
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
          <td>23.026208</td>
          <td>24.494611</td>
          <td>22.153018</td>
          <td>25.991057</td>
          <td>24.085272</td>
          <td>23.239008</td>
          <td>24.843856</td>
          <td>0.029811</td>
          <td>19.159794</td>
          <td>0.005008</td>
          <td>13.632143</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.126011</td>
          <td>26.538878</td>
          <td>24.576373</td>
          <td>29.084697</td>
          <td>23.569826</td>
          <td>21.159550</td>
          <td>22.869919</td>
          <td>0.006944</td>
          <td>23.001694</td>
          <td>0.010682</td>
          <td>22.862663</td>
          <td>0.009697</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.361038</td>
          <td>22.098051</td>
          <td>20.950619</td>
          <td>26.458797</td>
          <td>25.021527</td>
          <td>20.755995</td>
          <td>20.425099</td>
          <td>0.005026</td>
          <td>18.091306</td>
          <td>0.005001</td>
          <td>24.500169</td>
          <td>0.037421</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.045403</td>
          <td>18.855375</td>
          <td>28.427942</td>
          <td>24.013015</td>
          <td>23.892424</td>
          <td>18.258662</td>
          <td>27.117633</td>
          <td>0.218508</td>
          <td>22.600003</td>
          <td>0.008222</td>
          <td>22.875247</td>
          <td>0.009780</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.357597</td>
          <td>21.770590</td>
          <td>15.411428</td>
          <td>23.404007</td>
          <td>25.957345</td>
          <td>19.060094</td>
          <td>20.277695</td>
          <td>0.005020</td>
          <td>26.024957</td>
          <td>0.143841</td>
          <td>27.492127</td>
          <td>0.473645</td>
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
          <td>26.684448</td>
          <td>24.401359</td>
          <td>20.420772</td>
          <td>25.089791</td>
          <td>20.710853</td>
          <td>20.968550</td>
          <td>23.246153</td>
          <td>0.079439</td>
          <td>26.857222</td>
          <td>1.099736</td>
          <td>26.901536</td>
          <td>1.193829</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.100038</td>
          <td>25.270937</td>
          <td>25.721887</td>
          <td>23.240645</td>
          <td>22.198033</td>
          <td>24.458891</td>
          <td>22.683791</td>
          <td>0.048179</td>
          <td>24.950055</td>
          <td>0.287163</td>
          <td>25.399846</td>
          <td>0.441911</td>
        </tr>
        <tr>
          <th>2</th>
          <td>20.076120</td>
          <td>26.353996</td>
          <td>25.149638</td>
          <td>21.282377</td>
          <td>23.989843</td>
          <td>19.014465</td>
          <td>29.036573</td>
          <td>3.056226</td>
          <td>24.211863</td>
          <td>0.154999</td>
          <td>22.760342</td>
          <td>0.047182</td>
        </tr>
        <tr>
          <th>3</th>
          <td>19.977985</td>
          <td>22.317156</td>
          <td>20.762362</td>
          <td>21.380671</td>
          <td>19.394189</td>
          <td>22.464806</td>
          <td>24.996919</td>
          <td>0.349729</td>
          <td>23.054785</td>
          <td>0.056111</td>
          <td>23.019794</td>
          <td>0.059455</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.883862</td>
          <td>26.435798</td>
          <td>22.988800</td>
          <td>23.188953</td>
          <td>23.980104</td>
          <td>18.081590</td>
          <td>20.371558</td>
          <td>0.007658</td>
          <td>25.297256</td>
          <td>0.378295</td>
          <td>22.423472</td>
          <td>0.034954</td>
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
          <td>23.026208</td>
          <td>24.494611</td>
          <td>22.153018</td>
          <td>25.991057</td>
          <td>24.085272</td>
          <td>23.239008</td>
          <td>24.719933</td>
          <td>0.280239</td>
          <td>19.160202</td>
          <td>0.005244</td>
          <td>13.639581</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>996</th>
          <td>21.126011</td>
          <td>26.538878</td>
          <td>24.576373</td>
          <td>29.084697</td>
          <td>23.569826</td>
          <td>21.159550</td>
          <td>22.851888</td>
          <td>0.055966</td>
          <td>23.043437</td>
          <td>0.055547</td>
          <td>22.899130</td>
          <td>0.053396</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.361038</td>
          <td>22.098051</td>
          <td>20.950619</td>
          <td>26.458797</td>
          <td>25.021527</td>
          <td>20.755995</td>
          <td>20.402571</td>
          <td>0.007786</td>
          <td>18.079199</td>
          <td>0.005034</td>
          <td>24.471453</td>
          <td>0.210238</td>
        </tr>
        <tr>
          <th>998</th>
          <td>18.045403</td>
          <td>18.855375</td>
          <td>28.427942</td>
          <td>24.013015</td>
          <td>23.892424</td>
          <td>18.258662</td>
          <td>24.559146</td>
          <td>0.245707</td>
          <td>22.566226</td>
          <td>0.036308</td>
          <td>22.774600</td>
          <td>0.047786</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.357597</td>
          <td>21.770590</td>
          <td>15.411428</td>
          <td>23.404007</td>
          <td>25.957345</td>
          <td>19.060094</td>
          <td>20.275587</td>
          <td>0.007294</td>
          <td>25.387723</td>
          <td>0.405698</td>
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




.. image:: 02_Photometric_Realization_with_Other_Surveys_files/02_Photometric_Realization_with_Other_Surveys_16_0.png


