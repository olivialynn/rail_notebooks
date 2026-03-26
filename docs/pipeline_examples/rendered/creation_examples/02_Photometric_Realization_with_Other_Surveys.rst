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
```Photometric_Realization_with_Other_Surveys.ipynb`` <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization_with_Other_Surveys.ipynb>`__
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
          <td>23.141914</td>
          <td>24.173927</td>
          <td>24.982166</td>
          <td>18.541625</td>
          <td>20.429770</td>
          <td>21.829888</td>
          <td>27.095922</td>
          <td>19.944948</td>
          <td>23.278464</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.323063</td>
          <td>22.839295</td>
          <td>22.122903</td>
          <td>20.701176</td>
          <td>24.732650</td>
          <td>26.991680</td>
          <td>25.325013</td>
          <td>23.786281</td>
          <td>18.891986</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.213103</td>
          <td>23.414368</td>
          <td>17.421274</td>
          <td>23.321106</td>
          <td>19.095892</td>
          <td>15.631140</td>
          <td>24.351522</td>
          <td>30.085376</td>
          <td>24.786310</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.506766</td>
          <td>18.090103</td>
          <td>24.462432</td>
          <td>29.918263</td>
          <td>25.173536</td>
          <td>20.249903</td>
          <td>25.443808</td>
          <td>18.973447</td>
          <td>20.344264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.558310</td>
          <td>24.678915</td>
          <td>22.539343</td>
          <td>23.320639</td>
          <td>20.968950</td>
          <td>20.665749</td>
          <td>20.063002</td>
          <td>26.789910</td>
          <td>25.349250</td>
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
          <td>22.383009</td>
          <td>18.790097</td>
          <td>22.254451</td>
          <td>30.217751</td>
          <td>20.049243</td>
          <td>23.272401</td>
          <td>16.864328</td>
          <td>23.886127</td>
          <td>21.862413</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.596739</td>
          <td>20.802226</td>
          <td>24.977085</td>
          <td>25.902007</td>
          <td>27.514435</td>
          <td>29.619925</td>
          <td>17.565344</td>
          <td>22.433643</td>
          <td>26.225177</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.381045</td>
          <td>22.490098</td>
          <td>25.528559</td>
          <td>23.232676</td>
          <td>24.838891</td>
          <td>26.135397</td>
          <td>20.934115</td>
          <td>22.758842</td>
          <td>16.617892</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.660534</td>
          <td>18.890324</td>
          <td>22.913271</td>
          <td>26.103125</td>
          <td>18.273929</td>
          <td>23.631992</td>
          <td>18.232623</td>
          <td>19.259348</td>
          <td>27.368631</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.735018</td>
          <td>21.472022</td>
          <td>23.203327</td>
          <td>21.082705</td>
          <td>24.205770</td>
          <td>22.460892</td>
          <td>22.954405</td>
          <td>24.444754</td>
          <td>21.384473</td>
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
          <td>23.178310</td>
          <td>0.022146</td>
          <td>24.158157</td>
          <td>0.017929</td>
          <td>25.003742</td>
          <td>0.032617</td>
          <td>18.539860</td>
          <td>0.005008</td>
          <td>20.433340</td>
          <td>0.005305</td>
          <td>21.830059</td>
          <td>0.014352</td>
          <td>27.095922</td>
          <td>19.944948</td>
          <td>23.278464</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.317848</td>
          <td>0.005421</td>
          <td>22.836291</td>
          <td>0.007318</td>
          <td>22.129562</td>
          <td>0.005586</td>
          <td>20.698562</td>
          <td>0.005144</td>
          <td>24.788652</td>
          <td>0.084255</td>
          <td>26.913743</td>
          <td>0.916975</td>
          <td>25.325013</td>
          <td>23.786281</td>
          <td>18.891986</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.594918</td>
          <td>0.180439</td>
          <td>23.415049</td>
          <td>0.010185</td>
          <td>17.432407</td>
          <td>0.005001</td>
          <td>23.307409</td>
          <td>0.012520</td>
          <td>19.089214</td>
          <td>0.005040</td>
          <td>15.633747</td>
          <td>0.005002</td>
          <td>24.351522</td>
          <td>30.085376</td>
          <td>24.786310</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.037349</td>
          <td>0.260683</td>
          <td>18.087345</td>
          <td>0.005005</td>
          <td>24.469564</td>
          <td>0.020511</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.162240</td>
          <td>0.116881</td>
          <td>20.254163</td>
          <td>0.006002</td>
          <td>25.443808</td>
          <td>18.973447</td>
          <td>20.344264</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.569456</td>
          <td>0.007382</td>
          <td>24.685174</td>
          <td>0.028095</td>
          <td>22.539763</td>
          <td>0.006130</td>
          <td>23.288716</td>
          <td>0.012345</td>
          <td>20.968202</td>
          <td>0.005721</td>
          <td>20.661361</td>
          <td>0.006887</td>
          <td>20.063002</td>
          <td>26.789910</td>
          <td>25.349250</td>
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
          <td>22.384157</td>
          <td>0.011877</td>
          <td>18.789642</td>
          <td>0.005009</td>
          <td>22.253415</td>
          <td>0.005716</td>
          <td>28.227425</td>
          <td>0.737085</td>
          <td>20.048871</td>
          <td>0.005166</td>
          <td>23.197685</td>
          <td>0.046471</td>
          <td>16.864328</td>
          <td>23.886127</td>
          <td>21.862413</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.614171</td>
          <td>0.014066</td>
          <td>20.800113</td>
          <td>0.005109</td>
          <td>24.946531</td>
          <td>0.031015</td>
          <td>25.783009</td>
          <td>0.105985</td>
          <td>29.200906</td>
          <td>1.885321</td>
          <td>27.829372</td>
          <td>1.527635</td>
          <td>17.565344</td>
          <td>22.433643</td>
          <td>26.225177</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.385466</td>
          <td>0.011888</td>
          <td>22.498089</td>
          <td>0.006401</td>
          <td>25.526071</td>
          <td>0.051808</td>
          <td>23.255862</td>
          <td>0.012045</td>
          <td>24.829169</td>
          <td>0.087315</td>
          <td>25.875660</td>
          <td>0.447591</td>
          <td>20.934115</td>
          <td>22.758842</td>
          <td>16.617892</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.748322</td>
          <td>0.205276</td>
          <td>18.883762</td>
          <td>0.005010</td>
          <td>22.921808</td>
          <td>0.007043</td>
          <td>26.235948</td>
          <td>0.156863</td>
          <td>18.267284</td>
          <td>0.005014</td>
          <td>23.560389</td>
          <td>0.064115</td>
          <td>18.232623</td>
          <td>19.259348</td>
          <td>27.368631</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.724127</td>
          <td>0.015310</td>
          <td>21.464017</td>
          <td>0.005288</td>
          <td>23.206431</td>
          <td>0.008110</td>
          <td>21.071585</td>
          <td>0.005259</td>
          <td>24.241675</td>
          <td>0.051911</td>
          <td>22.454348</td>
          <td>0.024163</td>
          <td>22.954405</td>
          <td>24.444754</td>
          <td>21.384473</td>
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
          <td>23.141914</td>
          <td>24.173927</td>
          <td>24.982166</td>
          <td>18.541625</td>
          <td>20.429770</td>
          <td>21.829888</td>
          <td>27.176573</td>
          <td>0.229489</td>
          <td>19.934188</td>
          <td>0.005031</td>
          <td>23.273961</td>
          <td>0.013109</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.323063</td>
          <td>22.839295</td>
          <td>22.122903</td>
          <td>20.701176</td>
          <td>24.732650</td>
          <td>26.991680</td>
          <td>25.291155</td>
          <td>0.044360</td>
          <td>23.792167</td>
          <td>0.020103</td>
          <td>18.889005</td>
          <td>0.005005</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.213103</td>
          <td>23.414368</td>
          <td>17.421274</td>
          <td>23.321106</td>
          <td>19.095892</td>
          <td>15.631140</td>
          <td>24.320382</td>
          <td>0.018908</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.773162</td>
          <td>0.047725</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.506766</td>
          <td>18.090103</td>
          <td>24.462432</td>
          <td>29.918263</td>
          <td>25.173536</td>
          <td>20.249903</td>
          <td>25.443255</td>
          <td>0.050802</td>
          <td>18.975787</td>
          <td>0.005005</td>
          <td>20.338579</td>
          <td>0.005066</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.558310</td>
          <td>24.678915</td>
          <td>22.539343</td>
          <td>23.320639</td>
          <td>20.968950</td>
          <td>20.665749</td>
          <td>20.062470</td>
          <td>0.005013</td>
          <td>26.723984</td>
          <td>0.259150</td>
          <td>25.478166</td>
          <td>0.089264</td>
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
          <td>22.383009</td>
          <td>18.790097</td>
          <td>22.254451</td>
          <td>30.217751</td>
          <td>20.049243</td>
          <td>23.272401</td>
          <td>16.859968</td>
          <td>0.005000</td>
          <td>23.863857</td>
          <td>0.021381</td>
          <td>21.855506</td>
          <td>0.005986</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.596739</td>
          <td>20.802226</td>
          <td>24.977085</td>
          <td>25.902007</td>
          <td>27.514435</td>
          <td>29.619925</td>
          <td>17.572720</td>
          <td>0.005000</td>
          <td>22.449093</td>
          <td>0.007568</td>
          <td>26.303471</td>
          <td>0.182503</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.381045</td>
          <td>22.490098</td>
          <td>25.528559</td>
          <td>23.232676</td>
          <td>24.838891</td>
          <td>26.135397</td>
          <td>20.933938</td>
          <td>0.005065</td>
          <td>22.761680</td>
          <td>0.009075</td>
          <td>16.621664</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.660534</td>
          <td>18.890324</td>
          <td>22.913271</td>
          <td>26.103125</td>
          <td>18.273929</td>
          <td>23.631992</td>
          <td>18.230837</td>
          <td>0.005000</td>
          <td>19.266807</td>
          <td>0.005009</td>
          <td>28.285641</td>
          <td>0.823886</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.735018</td>
          <td>21.472022</td>
          <td>23.203327</td>
          <td>21.082705</td>
          <td>24.205770</td>
          <td>22.460892</td>
          <td>22.951624</td>
          <td>0.007210</td>
          <td>24.452697</td>
          <td>0.035874</td>
          <td>21.379345</td>
          <td>0.005432</td>
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
          <td>23.141914</td>
          <td>24.173927</td>
          <td>24.982166</td>
          <td>18.541625</td>
          <td>20.429770</td>
          <td>21.829888</td>
          <td>25.841067</td>
          <td>0.654299</td>
          <td>19.943670</td>
          <td>0.005966</td>
          <td>23.326549</td>
          <td>0.078073</td>
        </tr>
        <tr>
          <th>1</th>
          <td>20.323063</td>
          <td>22.839295</td>
          <td>22.122903</td>
          <td>20.701176</td>
          <td>24.732650</td>
          <td>26.991680</td>
          <td>26.136057</td>
          <td>0.797800</td>
          <td>23.969100</td>
          <td>0.125696</td>
          <td>18.882953</td>
          <td>0.005177</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.213103</td>
          <td>23.414368</td>
          <td>17.421274</td>
          <td>23.321106</td>
          <td>19.095892</td>
          <td>15.631140</td>
          <td>24.605153</td>
          <td>0.255180</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.076907</td>
          <td>0.344256</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.506766</td>
          <td>18.090103</td>
          <td>24.462432</td>
          <td>29.918263</td>
          <td>25.173536</td>
          <td>20.249903</td>
          <td>25.225700</td>
          <td>0.417681</td>
          <td>18.979254</td>
          <td>0.005176</td>
          <td>20.348312</td>
          <td>0.007199</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.558310</td>
          <td>24.678915</td>
          <td>22.539343</td>
          <td>23.320639</td>
          <td>20.968950</td>
          <td>20.665749</td>
          <td>20.055551</td>
          <td>0.006619</td>
          <td>25.921152</td>
          <td>0.601658</td>
          <td>30.556078</td>
          <td>4.427212</td>
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
          <td>22.383009</td>
          <td>18.790097</td>
          <td>22.254451</td>
          <td>30.217751</td>
          <td>20.049243</td>
          <td>23.272401</td>
          <td>16.867132</td>
          <td>0.005005</td>
          <td>23.841227</td>
          <td>0.112448</td>
          <td>21.862788</td>
          <td>0.021362</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.596739</td>
          <td>20.802226</td>
          <td>24.977085</td>
          <td>25.902007</td>
          <td>27.514435</td>
          <td>29.619925</td>
          <td>17.568277</td>
          <td>0.005019</td>
          <td>22.480132</td>
          <td>0.033635</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>997</th>
          <td>22.381045</td>
          <td>22.490098</td>
          <td>25.528559</td>
          <td>23.232676</td>
          <td>24.838891</td>
          <td>26.135397</td>
          <td>20.925546</td>
          <td>0.010868</td>
          <td>22.723365</td>
          <td>0.041759</td>
          <td>16.616807</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>25.660534</td>
          <td>18.890324</td>
          <td>22.913271</td>
          <td>26.103125</td>
          <td>18.273929</td>
          <td>23.631992</td>
          <td>18.232321</td>
          <td>0.005065</td>
          <td>19.261260</td>
          <td>0.005293</td>
          <td>26.318149</td>
          <td>0.841294</td>
        </tr>
        <tr>
          <th>999</th>
          <td>22.735018</td>
          <td>21.472022</td>
          <td>23.203327</td>
          <td>21.082705</td>
          <td>24.205770</td>
          <td>22.460892</td>
          <td>22.966182</td>
          <td>0.061961</td>
          <td>24.385775</td>
          <td>0.179785</td>
          <td>21.408865</td>
          <td>0.014596</td>
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


