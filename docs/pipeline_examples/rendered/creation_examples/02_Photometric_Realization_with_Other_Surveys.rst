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
          <td>25.468770</td>
          <td>24.874110</td>
          <td>23.871990</td>
          <td>19.780658</td>
          <td>25.996625</td>
          <td>21.494870</td>
          <td>19.775766</td>
          <td>18.759278</td>
          <td>24.019840</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.836721</td>
          <td>19.004286</td>
          <td>20.729692</td>
          <td>29.241382</td>
          <td>21.220022</td>
          <td>25.855877</td>
          <td>22.527862</td>
          <td>18.706927</td>
          <td>25.378633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.357447</td>
          <td>25.194249</td>
          <td>23.461680</td>
          <td>19.296505</td>
          <td>20.937006</td>
          <td>20.716325</td>
          <td>25.475760</td>
          <td>16.625803</td>
          <td>24.377165</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.674184</td>
          <td>25.407137</td>
          <td>24.619111</td>
          <td>16.122080</td>
          <td>20.477143</td>
          <td>24.371543</td>
          <td>24.801303</td>
          <td>19.641893</td>
          <td>23.337518</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.129946</td>
          <td>24.924979</td>
          <td>22.186146</td>
          <td>21.418786</td>
          <td>21.740473</td>
          <td>25.585063</td>
          <td>25.079158</td>
          <td>24.130943</td>
          <td>23.731829</td>
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
          <td>27.236975</td>
          <td>26.239315</td>
          <td>24.613416</td>
          <td>25.674417</td>
          <td>23.746404</td>
          <td>24.539178</td>
          <td>15.601424</td>
          <td>21.227136</td>
          <td>19.098766</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.728821</td>
          <td>22.449322</td>
          <td>25.029206</td>
          <td>19.832398</td>
          <td>23.263659</td>
          <td>20.332547</td>
          <td>20.117226</td>
          <td>21.984942</td>
          <td>23.513858</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.236817</td>
          <td>21.124839</td>
          <td>23.926885</td>
          <td>19.272726</td>
          <td>21.671680</td>
          <td>25.338771</td>
          <td>23.421108</td>
          <td>19.835997</td>
          <td>23.250070</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.668642</td>
          <td>23.051926</td>
          <td>21.827861</td>
          <td>19.261813</td>
          <td>26.699373</td>
          <td>20.993713</td>
          <td>19.052311</td>
          <td>24.524175</td>
          <td>18.048799</td>
        </tr>
        <tr>
          <th>999</th>
          <td>31.302837</td>
          <td>24.725669</td>
          <td>22.701613</td>
          <td>22.751241</td>
          <td>25.006437</td>
          <td>24.942641</td>
          <td>24.770256</td>
          <td>25.306286</td>
          <td>21.638280</td>
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
          <td>25.274378</td>
          <td>0.137278</td>
          <td>24.861222</td>
          <td>0.032771</td>
          <td>23.883504</td>
          <td>0.012740</td>
          <td>19.787902</td>
          <td>0.005037</td>
          <td>25.786372</td>
          <td>0.199481</td>
          <td>21.480398</td>
          <td>0.011020</td>
          <td>19.775766</td>
          <td>18.759278</td>
          <td>24.019840</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.830972</td>
          <td>0.005223</td>
          <td>19.005278</td>
          <td>0.005012</td>
          <td>20.729665</td>
          <td>0.005065</td>
          <td>29.323553</td>
          <td>1.410577</td>
          <td>21.219536</td>
          <td>0.006076</td>
          <td>25.356621</td>
          <td>0.298485</td>
          <td>22.527862</td>
          <td>18.706927</td>
          <td>25.378633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.371159</td>
          <td>0.011768</td>
          <td>25.240560</td>
          <td>0.045797</td>
          <td>23.454254</td>
          <td>0.009406</td>
          <td>19.295422</td>
          <td>0.005019</td>
          <td>20.933512</td>
          <td>0.005682</td>
          <td>20.705512</td>
          <td>0.007017</td>
          <td>25.475760</td>
          <td>16.625803</td>
          <td>24.377165</td>
        </tr>
        <tr>
          <th>3</th>
          <td>26.740037</td>
          <td>0.452988</td>
          <td>25.456884</td>
          <td>0.055468</td>
          <td>24.604394</td>
          <td>0.023020</td>
          <td>16.125551</td>
          <td>0.005001</td>
          <td>20.470919</td>
          <td>0.005324</td>
          <td>24.766368</td>
          <td>0.183236</td>
          <td>24.801303</td>
          <td>19.641893</td>
          <td>23.337518</td>
        </tr>
        <tr>
          <th>4</th>
          <td>28.316932</td>
          <td>1.266331</td>
          <td>24.918677</td>
          <td>0.034468</td>
          <td>22.182564</td>
          <td>0.005639</td>
          <td>21.429027</td>
          <td>0.005461</td>
          <td>21.745045</td>
          <td>0.007414</td>
          <td>25.883877</td>
          <td>0.450372</td>
          <td>25.079158</td>
          <td>24.130943</td>
          <td>23.731829</td>
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
          <td>27.181464</td>
          <td>0.624289</td>
          <td>26.139074</td>
          <td>0.101220</td>
          <td>24.602435</td>
          <td>0.022981</td>
          <td>25.516813</td>
          <td>0.083896</td>
          <td>23.718194</td>
          <td>0.032651</td>
          <td>24.430244</td>
          <td>0.137475</td>
          <td>15.601424</td>
          <td>21.227136</td>
          <td>19.098766</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.737803</td>
          <td>0.035794</td>
          <td>22.448925</td>
          <td>0.006301</td>
          <td>25.057522</td>
          <td>0.034201</td>
          <td>19.825442</td>
          <td>0.005039</td>
          <td>23.287739</td>
          <td>0.022438</td>
          <td>20.327196</td>
          <td>0.006124</td>
          <td>20.117226</td>
          <td>21.984942</td>
          <td>23.513858</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.214798</td>
          <td>0.054387</td>
          <td>21.122989</td>
          <td>0.005173</td>
          <td>23.939667</td>
          <td>0.013302</td>
          <td>19.267430</td>
          <td>0.005018</td>
          <td>21.670445</td>
          <td>0.007160</td>
          <td>26.490765</td>
          <td>0.696245</td>
          <td>23.421108</td>
          <td>19.835997</td>
          <td>23.250070</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.729162</td>
          <td>0.085474</td>
          <td>23.054202</td>
          <td>0.008168</td>
          <td>21.830831</td>
          <td>0.005362</td>
          <td>19.262681</td>
          <td>0.005018</td>
          <td>27.062812</td>
          <td>0.545634</td>
          <td>21.001721</td>
          <td>0.008122</td>
          <td>19.052311</td>
          <td>24.524175</td>
          <td>18.048799</td>
        </tr>
        <tr>
          <th>999</th>
          <td>inf</td>
          <td>inf</td>
          <td>24.694617</td>
          <td>0.028327</td>
          <td>22.694971</td>
          <td>0.006442</td>
          <td>22.752462</td>
          <td>0.008582</td>
          <td>24.932840</td>
          <td>0.095645</td>
          <td>24.641451</td>
          <td>0.164787</td>
          <td>24.770256</td>
          <td>25.306286</td>
          <td>21.638280</td>
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
          <td>25.468770</td>
          <td>24.874110</td>
          <td>23.871990</td>
          <td>19.780658</td>
          <td>25.996625</td>
          <td>21.494870</td>
          <td>19.769871</td>
          <td>0.005008</td>
          <td>18.757872</td>
          <td>0.005004</td>
          <td>24.051229</td>
          <td>0.025164</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.836721</td>
          <td>19.004286</td>
          <td>20.729692</td>
          <td>29.241382</td>
          <td>21.220022</td>
          <td>25.855877</td>
          <td>22.532472</td>
          <td>0.006122</td>
          <td>18.706999</td>
          <td>0.005003</td>
          <td>25.400327</td>
          <td>0.083337</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.357447</td>
          <td>25.194249</td>
          <td>23.461680</td>
          <td>19.296505</td>
          <td>20.937006</td>
          <td>20.716325</td>
          <td>25.409264</td>
          <td>0.049286</td>
          <td>16.628351</td>
          <td>0.005000</td>
          <td>24.327195</td>
          <td>0.032092</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.674184</td>
          <td>25.407137</td>
          <td>24.619111</td>
          <td>16.122080</td>
          <td>20.477143</td>
          <td>24.371543</td>
          <td>24.827733</td>
          <td>0.029389</td>
          <td>19.640380</td>
          <td>0.005018</td>
          <td>23.318710</td>
          <td>0.013580</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.129946</td>
          <td>24.924979</td>
          <td>22.186146</td>
          <td>21.418786</td>
          <td>21.740473</td>
          <td>25.585063</td>
          <td>25.077874</td>
          <td>0.036686</td>
          <td>24.127823</td>
          <td>0.026913</td>
          <td>23.749329</td>
          <td>0.019380</td>
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
          <td>27.236975</td>
          <td>26.239315</td>
          <td>24.613416</td>
          <td>25.674417</td>
          <td>23.746404</td>
          <td>24.539178</td>
          <td>15.599203</td>
          <td>0.005000</td>
          <td>21.226648</td>
          <td>0.005329</td>
          <td>19.106584</td>
          <td>0.005007</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.728821</td>
          <td>22.449322</td>
          <td>25.029206</td>
          <td>19.832398</td>
          <td>23.263659</td>
          <td>20.332547</td>
          <td>20.117321</td>
          <td>0.005015</td>
          <td>21.987561</td>
          <td>0.006230</td>
          <td>23.506140</td>
          <td>0.015802</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.236817</td>
          <td>21.124839</td>
          <td>23.926885</td>
          <td>19.272726</td>
          <td>21.671680</td>
          <td>25.338771</td>
          <td>23.420359</td>
          <td>0.009427</td>
          <td>19.834294</td>
          <td>0.005026</td>
          <td>23.245529</td>
          <td>0.012822</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.668642</td>
          <td>23.051926</td>
          <td>21.827861</td>
          <td>19.261813</td>
          <td>26.699373</td>
          <td>20.993713</td>
          <td>19.054502</td>
          <td>0.005002</td>
          <td>24.441628</td>
          <td>0.035523</td>
          <td>18.045052</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>999</th>
          <td>31.302837</td>
          <td>24.725669</td>
          <td>22.701613</td>
          <td>22.751241</td>
          <td>25.006437</td>
          <td>24.942641</td>
          <td>24.799500</td>
          <td>0.028666</td>
          <td>25.322951</td>
          <td>0.077824</td>
          <td>21.631112</td>
          <td>0.005671</td>
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
          <td>25.468770</td>
          <td>24.874110</td>
          <td>23.871990</td>
          <td>19.780658</td>
          <td>25.996625</td>
          <td>21.494870</td>
          <td>19.786415</td>
          <td>0.006039</td>
          <td>18.753739</td>
          <td>0.005117</td>
          <td>24.069029</td>
          <td>0.149402</td>
        </tr>
        <tr>
          <th>1</th>
          <td>19.836721</td>
          <td>19.004286</td>
          <td>20.729692</td>
          <td>29.241382</td>
          <td>21.220022</td>
          <td>25.855877</td>
          <td>22.551044</td>
          <td>0.042802</td>
          <td>18.707154</td>
          <td>0.005108</td>
          <td>26.028310</td>
          <td>0.694657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>22.357447</td>
          <td>25.194249</td>
          <td>23.461680</td>
          <td>19.296505</td>
          <td>20.937006</td>
          <td>20.716325</td>
          <td>24.927283</td>
          <td>0.330997</td>
          <td>16.631320</td>
          <td>0.005002</td>
          <td>24.310335</td>
          <td>0.183567</td>
        </tr>
        <tr>
          <th>3</th>
          <td>27.674184</td>
          <td>25.407137</td>
          <td>24.619111</td>
          <td>16.122080</td>
          <td>20.477143</td>
          <td>24.371543</td>
          <td>24.661069</td>
          <td>0.267129</td>
          <td>19.643241</td>
          <td>0.005576</td>
          <td>23.323776</td>
          <td>0.077881</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.129946</td>
          <td>24.924979</td>
          <td>22.186146</td>
          <td>21.418786</td>
          <td>21.740473</td>
          <td>25.585063</td>
          <td>24.790809</td>
          <td>0.296770</td>
          <td>24.261686</td>
          <td>0.161754</td>
          <td>23.643545</td>
          <td>0.103235</td>
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
          <td>27.236975</td>
          <td>26.239315</td>
          <td>24.613416</td>
          <td>25.674417</td>
          <td>23.746404</td>
          <td>24.539178</td>
          <td>15.602676</td>
          <td>0.005001</td>
          <td>21.229899</td>
          <td>0.011737</td>
          <td>19.090495</td>
          <td>0.005258</td>
        </tr>
        <tr>
          <th>996</th>
          <td>23.728821</td>
          <td>22.449322</td>
          <td>25.029206</td>
          <td>19.832398</td>
          <td>23.263659</td>
          <td>20.332547</td>
          <td>20.113547</td>
          <td>0.006777</td>
          <td>21.984096</td>
          <td>0.021758</td>
          <td>23.601258</td>
          <td>0.099475</td>
        </tr>
        <tr>
          <th>997</th>
          <td>24.236817</td>
          <td>21.124839</td>
          <td>23.926885</td>
          <td>19.272726</td>
          <td>21.671680</td>
          <td>25.338771</td>
          <td>23.739041</td>
          <td>0.122454</td>
          <td>19.835926</td>
          <td>0.005804</td>
          <td>23.121655</td>
          <td>0.065095</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.668642</td>
          <td>23.051926</td>
          <td>21.827861</td>
          <td>19.261813</td>
          <td>26.699373</td>
          <td>20.993713</td>
          <td>19.062730</td>
          <td>0.005294</td>
          <td>24.771275</td>
          <td>0.248173</td>
          <td>18.051280</td>
          <td>0.005039</td>
        </tr>
        <tr>
          <th>999</th>
          <td>31.302837</td>
          <td>24.725669</td>
          <td>22.701613</td>
          <td>22.751241</td>
          <td>25.006437</td>
          <td>24.942641</td>
          <td>24.882755</td>
          <td>0.319474</td>
          <td>24.847204</td>
          <td>0.264121</td>
          <td>21.657931</td>
          <td>0.017934</td>
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


