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
          <td>24.651168</td>
          <td>22.479405</td>
          <td>26.720420</td>
          <td>29.476924</td>
          <td>21.902155</td>
          <td>22.199306</td>
          <td>23.610365</td>
          <td>22.065903</td>
          <td>21.161574</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.514438</td>
          <td>21.234691</td>
          <td>20.894524</td>
          <td>22.102717</td>
          <td>20.653214</td>
          <td>24.482550</td>
          <td>19.448352</td>
          <td>24.940032</td>
          <td>26.490537</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.188305</td>
          <td>26.860545</td>
          <td>27.366471</td>
          <td>21.102870</td>
          <td>23.045840</td>
          <td>25.503393</td>
          <td>20.031719</td>
          <td>19.239058</td>
          <td>25.256879</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.521089</td>
          <td>22.240759</td>
          <td>25.212710</td>
          <td>17.964801</td>
          <td>22.744474</td>
          <td>19.100505</td>
          <td>21.242425</td>
          <td>18.833706</td>
          <td>20.836772</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.296711</td>
          <td>19.788159</td>
          <td>24.050087</td>
          <td>20.610071</td>
          <td>22.876115</td>
          <td>22.282437</td>
          <td>25.595458</td>
          <td>26.086874</td>
          <td>18.412504</td>
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
          <td>22.733414</td>
          <td>16.861620</td>
          <td>22.521772</td>
          <td>27.201808</td>
          <td>23.185033</td>
          <td>22.784255</td>
          <td>22.647197</td>
          <td>27.861195</td>
          <td>26.216899</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.356930</td>
          <td>24.602171</td>
          <td>24.001158</td>
          <td>24.645902</td>
          <td>20.587300</td>
          <td>20.706643</td>
          <td>23.774636</td>
          <td>19.968097</td>
          <td>23.205284</td>
        </tr>
        <tr>
          <th>997</th>
          <td>16.632954</td>
          <td>14.834630</td>
          <td>25.229883</td>
          <td>28.162282</td>
          <td>25.633721</td>
          <td>21.831107</td>
          <td>23.289367</td>
          <td>24.245431</td>
          <td>16.691857</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.030299</td>
          <td>24.066104</td>
          <td>20.252478</td>
          <td>20.402251</td>
          <td>29.342032</td>
          <td>21.348544</td>
          <td>22.519794</td>
          <td>22.538796</td>
          <td>28.231409</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.013076</td>
          <td>23.684646</td>
          <td>24.838941</td>
          <td>24.266819</td>
          <td>22.061643</td>
          <td>18.002945</td>
          <td>27.420458</td>
          <td>28.223690</td>
          <td>26.921644</td>
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
          <td>24.708563</td>
          <td>0.083946</td>
          <td>22.490124</td>
          <td>0.006384</td>
          <td>26.779328</td>
          <td>0.155500</td>
          <td>inf</td>
          <td>inf</td>
          <td>21.902762</td>
          <td>0.008040</td>
          <td>22.187188</td>
          <td>0.019232</td>
          <td>23.610365</td>
          <td>22.065903</td>
          <td>21.161574</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.509730</td>
          <td>0.007195</td>
          <td>21.232204</td>
          <td>0.005204</td>
          <td>20.901552</td>
          <td>0.005084</td>
          <td>22.103187</td>
          <td>0.006357</td>
          <td>20.664495</td>
          <td>0.005443</td>
          <td>24.424383</td>
          <td>0.136782</td>
          <td>19.448352</td>
          <td>24.940032</td>
          <td>26.490537</td>
        </tr>
        <tr>
          <th>2</th>
          <td>25.987636</td>
          <td>0.250289</td>
          <td>26.889951</td>
          <td>0.193099</td>
          <td>28.032735</td>
          <td>0.431409</td>
          <td>21.096545</td>
          <td>0.005270</td>
          <td>23.034300</td>
          <td>0.018101</td>
          <td>25.864315</td>
          <td>0.443773</td>
          <td>20.031719</td>
          <td>19.239058</td>
          <td>25.256879</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.353109</td>
          <td>0.146876</td>
          <td>22.231204</td>
          <td>0.005932</td>
          <td>25.199576</td>
          <td>0.038778</td>
          <td>17.965424</td>
          <td>0.005004</td>
          <td>22.738706</td>
          <td>0.014230</td>
          <td>19.097575</td>
          <td>0.005159</td>
          <td>21.242425</td>
          <td>18.833706</td>
          <td>20.836772</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.288969</td>
          <td>0.005115</td>
          <td>19.794717</td>
          <td>0.005030</td>
          <td>24.061898</td>
          <td>0.014643</td>
          <td>20.616086</td>
          <td>0.005126</td>
          <td>22.862125</td>
          <td>0.015709</td>
          <td>22.253815</td>
          <td>0.020347</td>
          <td>25.595458</td>
          <td>26.086874</td>
          <td>18.412504</td>
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
          <td>22.718881</td>
          <td>0.015248</td>
          <td>16.863638</td>
          <td>0.005001</td>
          <td>22.520555</td>
          <td>0.006097</td>
          <td>27.053852</td>
          <td>0.309489</td>
          <td>23.154554</td>
          <td>0.020028</td>
          <td>22.859642</td>
          <td>0.034448</td>
          <td>22.647197</td>
          <td>27.861195</td>
          <td>26.216899</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.300302</td>
          <td>0.322245</td>
          <td>24.590401</td>
          <td>0.025876</td>
          <td>24.001653</td>
          <td>0.013961</td>
          <td>24.638102</td>
          <td>0.038505</td>
          <td>20.591072</td>
          <td>0.005393</td>
          <td>20.706192</td>
          <td>0.007020</td>
          <td>23.774636</td>
          <td>19.968097</td>
          <td>23.205284</td>
        </tr>
        <tr>
          <th>997</th>
          <td>16.627992</td>
          <td>0.005007</td>
          <td>14.824013</td>
          <td>0.005000</td>
          <td>25.215068</td>
          <td>0.039314</td>
          <td>28.330014</td>
          <td>0.788843</td>
          <td>25.261249</td>
          <td>0.127375</td>
          <td>21.833304</td>
          <td>0.014389</td>
          <td>23.289367</td>
          <td>24.245431</td>
          <td>16.691857</td>
        </tr>
        <tr>
          <th>998</th>
          <td>23.978810</td>
          <td>0.044198</td>
          <td>24.061610</td>
          <td>0.016562</td>
          <td>20.251054</td>
          <td>0.005033</td>
          <td>20.406854</td>
          <td>0.005092</td>
          <td>28.643353</td>
          <td>1.449113</td>
          <td>21.361472</td>
          <td>0.010144</td>
          <td>22.519794</td>
          <td>22.538796</td>
          <td>28.231409</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.005730</td>
          <td>0.005083</td>
          <td>23.687949</td>
          <td>0.012363</td>
          <td>24.860845</td>
          <td>0.028769</td>
          <td>24.293668</td>
          <td>0.028425</td>
          <td>22.066512</td>
          <td>0.008836</td>
          <td>17.989156</td>
          <td>0.005031</td>
          <td>27.420458</td>
          <td>28.223690</td>
          <td>26.921644</td>
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
          <td>24.651168</td>
          <td>22.479405</td>
          <td>26.720420</td>
          <td>29.476924</td>
          <td>21.902155</td>
          <td>22.199306</td>
          <td>23.608741</td>
          <td>0.010737</td>
          <td>22.068784</td>
          <td>0.006406</td>
          <td>21.158799</td>
          <td>0.005292</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.514438</td>
          <td>21.234691</td>
          <td>20.894524</td>
          <td>22.102717</td>
          <td>20.653214</td>
          <td>24.482550</td>
          <td>19.447169</td>
          <td>0.005004</td>
          <td>24.909287</td>
          <td>0.053882</td>
          <td>26.331180</td>
          <td>0.186834</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.188305</td>
          <td>26.860545</td>
          <td>27.366471</td>
          <td>21.102870</td>
          <td>23.045840</td>
          <td>25.503393</td>
          <td>20.034465</td>
          <td>0.005013</td>
          <td>19.246520</td>
          <td>0.005009</td>
          <td>25.237433</td>
          <td>0.072144</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.521089</td>
          <td>22.240759</td>
          <td>25.212710</td>
          <td>17.964801</td>
          <td>22.744474</td>
          <td>19.100505</td>
          <td>21.238038</td>
          <td>0.005114</td>
          <td>18.843581</td>
          <td>0.005004</td>
          <td>20.842835</td>
          <td>0.005165</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.296711</td>
          <td>19.788159</td>
          <td>24.050087</td>
          <td>20.610071</td>
          <td>22.876115</td>
          <td>22.282437</td>
          <td>25.578816</td>
          <td>0.057325</td>
          <td>26.183736</td>
          <td>0.164831</td>
          <td>18.415424</td>
          <td>0.005002</td>
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
          <td>22.733414</td>
          <td>16.861620</td>
          <td>22.521772</td>
          <td>27.201808</td>
          <td>23.185033</td>
          <td>22.784255</td>
          <td>22.640519</td>
          <td>0.006342</td>
          <td>27.167640</td>
          <td>0.369666</td>
          <td>25.889140</td>
          <td>0.127901</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.356930</td>
          <td>24.602171</td>
          <td>24.001158</td>
          <td>24.645902</td>
          <td>20.587300</td>
          <td>20.706643</td>
          <td>23.738480</td>
          <td>0.011813</td>
          <td>19.968137</td>
          <td>0.005033</td>
          <td>23.209822</td>
          <td>0.012472</td>
        </tr>
        <tr>
          <th>997</th>
          <td>16.632954</td>
          <td>14.834630</td>
          <td>25.229883</td>
          <td>28.162282</td>
          <td>25.633721</td>
          <td>21.831107</td>
          <td>23.305087</td>
          <td>0.008757</td>
          <td>24.238076</td>
          <td>0.029659</td>
          <td>16.683051</td>
          <td>0.005000</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.030299</td>
          <td>24.066104</td>
          <td>20.252478</td>
          <td>20.402251</td>
          <td>29.342032</td>
          <td>21.348544</td>
          <td>22.518098</td>
          <td>0.006095</td>
          <td>22.537648</td>
          <td>0.007937</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.013076</td>
          <td>23.684646</td>
          <td>24.838941</td>
          <td>24.266819</td>
          <td>22.061643</td>
          <td>18.002945</td>
          <td>27.392792</td>
          <td>0.274125</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.451587</td>
          <td>0.459486</td>
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
          <td>24.651168</td>
          <td>22.479405</td>
          <td>26.720420</td>
          <td>29.476924</td>
          <td>21.902155</td>
          <td>22.199306</td>
          <td>23.467555</td>
          <td>0.096573</td>
          <td>22.071812</td>
          <td>0.023479</td>
          <td>21.188906</td>
          <td>0.012274</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.514438</td>
          <td>21.234691</td>
          <td>20.894524</td>
          <td>22.102717</td>
          <td>20.653214</td>
          <td>24.482550</td>
          <td>19.453570</td>
          <td>0.005587</td>
          <td>24.766306</td>
          <td>0.247160</td>
          <td>28.068691</td>
          <td>2.092110</td>
        </tr>
        <tr>
          <th>2</th>
          <td>26.188305</td>
          <td>26.860545</td>
          <td>27.366471</td>
          <td>21.102870</td>
          <td>23.045840</td>
          <td>25.503393</td>
          <td>20.029640</td>
          <td>0.006553</td>
          <td>19.241946</td>
          <td>0.005283</td>
          <td>24.717585</td>
          <td>0.257795</td>
        </tr>
        <tr>
          <th>3</th>
          <td>25.521089</td>
          <td>22.240759</td>
          <td>25.212710</td>
          <td>17.964801</td>
          <td>22.744474</td>
          <td>19.100505</td>
          <td>21.236398</td>
          <td>0.013772</td>
          <td>18.831140</td>
          <td>0.005135</td>
          <td>20.848575</td>
          <td>0.009606</td>
        </tr>
        <tr>
          <th>4</th>
          <td>19.296711</td>
          <td>19.788159</td>
          <td>24.050087</td>
          <td>20.610071</td>
          <td>22.876115</td>
          <td>22.282437</td>
          <td>25.192992</td>
          <td>0.407343</td>
          <td>25.555160</td>
          <td>0.460720</td>
          <td>18.402761</td>
          <td>0.005074</td>
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
          <td>22.733414</td>
          <td>16.861620</td>
          <td>22.521772</td>
          <td>27.201808</td>
          <td>23.185033</td>
          <td>22.784255</td>
          <td>22.673895</td>
          <td>0.047756</td>
          <td>28.260829</td>
          <td>2.171311</td>
          <td>25.973066</td>
          <td>0.668899</td>
        </tr>
        <tr>
          <th>996</th>
          <td>26.356930</td>
          <td>24.602171</td>
          <td>24.001158</td>
          <td>24.645902</td>
          <td>20.587300</td>
          <td>20.706643</td>
          <td>23.691186</td>
          <td>0.117457</td>
          <td>19.957831</td>
          <td>0.005990</td>
          <td>23.114969</td>
          <td>0.064709</td>
        </tr>
        <tr>
          <th>997</th>
          <td>16.632954</td>
          <td>14.834630</td>
          <td>25.229883</td>
          <td>28.162282</td>
          <td>25.633721</td>
          <td>21.831107</td>
          <td>23.335035</td>
          <td>0.085931</td>
          <td>24.338388</td>
          <td>0.172691</td>
          <td>16.691266</td>
          <td>0.005003</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.030299</td>
          <td>24.066104</td>
          <td>20.252478</td>
          <td>20.402251</td>
          <td>29.342032</td>
          <td>21.348544</td>
          <td>22.502227</td>
          <td>0.040980</td>
          <td>22.545200</td>
          <td>0.035636</td>
          <td>26.832625</td>
          <td>1.148356</td>
        </tr>
        <tr>
          <th>999</th>
          <td>19.013076</td>
          <td>23.684646</td>
          <td>24.838941</td>
          <td>24.266819</td>
          <td>22.061643</td>
          <td>18.002945</td>
          <td>24.883819</td>
          <td>0.319745</td>
          <td>28.388416</td>
          <td>2.282481</td>
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


