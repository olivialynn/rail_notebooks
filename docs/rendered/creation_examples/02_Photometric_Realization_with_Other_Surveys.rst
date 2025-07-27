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
          <td>25.115821</td>
          <td>23.635735</td>
          <td>17.804598</td>
          <td>23.215472</td>
          <td>22.770758</td>
          <td>21.427868</td>
          <td>23.482313</td>
          <td>27.015061</td>
          <td>21.413897</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.102749</td>
          <td>21.478506</td>
          <td>23.584137</td>
          <td>24.350449</td>
          <td>21.686557</td>
          <td>18.422865</td>
          <td>25.868742</td>
          <td>21.211569</td>
          <td>24.333139</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.735998</td>
          <td>18.793369</td>
          <td>19.063401</td>
          <td>28.816046</td>
          <td>27.043125</td>
          <td>22.218105</td>
          <td>25.135996</td>
          <td>23.193197</td>
          <td>22.037994</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.182941</td>
          <td>15.951134</td>
          <td>22.273621</td>
          <td>19.504070</td>
          <td>27.771019</td>
          <td>23.698619</td>
          <td>23.094215</td>
          <td>19.384096</td>
          <td>27.037578</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.047467</td>
          <td>18.877125</td>
          <td>21.308809</td>
          <td>22.352723</td>
          <td>24.619465</td>
          <td>20.526776</td>
          <td>25.250517</td>
          <td>20.678023</td>
          <td>20.051526</td>
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
          <td>24.124411</td>
          <td>27.850721</td>
          <td>22.889233</td>
          <td>25.143905</td>
          <td>21.208115</td>
          <td>25.922464</td>
          <td>22.405467</td>
          <td>26.825087</td>
          <td>23.772143</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.439643</td>
          <td>22.544920</td>
          <td>21.207710</td>
          <td>19.785157</td>
          <td>26.539721</td>
          <td>23.431689</td>
          <td>18.439370</td>
          <td>20.760538</td>
          <td>19.456022</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.411468</td>
          <td>22.750965</td>
          <td>26.172398</td>
          <td>25.795807</td>
          <td>22.394124</td>
          <td>25.751225</td>
          <td>20.630273</td>
          <td>26.714885</td>
          <td>21.325026</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.528914</td>
          <td>18.047077</td>
          <td>21.907181</td>
          <td>18.759660</td>
          <td>24.684787</td>
          <td>25.233606</td>
          <td>21.820861</td>
          <td>21.994287</td>
          <td>22.769991</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.689360</td>
          <td>23.226261</td>
          <td>24.832386</td>
          <td>23.661843</td>
          <td>21.571665</td>
          <td>25.178646</td>
          <td>26.636297</td>
          <td>25.423507</td>
          <td>21.930060</td>
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
          <td>25.288560</td>
          <td>0.138962</td>
          <td>23.632933</td>
          <td>0.011870</td>
          <td>17.800389</td>
          <td>0.005002</td>
          <td>23.225227</td>
          <td>0.011774</td>
          <td>22.763586</td>
          <td>0.014514</td>
          <td>21.431691</td>
          <td>0.010647</td>
          <td>23.482313</td>
          <td>27.015061</td>
          <td>21.413897</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.035488</td>
          <td>0.046454</td>
          <td>21.471583</td>
          <td>0.005291</td>
          <td>23.578552</td>
          <td>0.010214</td>
          <td>24.314832</td>
          <td>0.028957</td>
          <td>21.685559</td>
          <td>0.007209</td>
          <td>18.425526</td>
          <td>0.005058</td>
          <td>25.868742</td>
          <td>21.211569</td>
          <td>24.333139</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.821778</td>
          <td>0.092690</td>
          <td>18.788890</td>
          <td>0.005009</td>
          <td>19.060622</td>
          <td>0.005007</td>
          <td>27.736614</td>
          <td>0.522695</td>
          <td>26.808527</td>
          <td>0.452209</td>
          <td>22.221201</td>
          <td>0.019792</td>
          <td>25.135996</td>
          <td>23.193197</td>
          <td>22.037994</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.239986</td>
          <td>0.055606</td>
          <td>15.944456</td>
          <td>0.005001</td>
          <td>22.276039</td>
          <td>0.005742</td>
          <td>19.504285</td>
          <td>0.005025</td>
          <td>27.570428</td>
          <td>0.775124</td>
          <td>23.780168</td>
          <td>0.077878</td>
          <td>23.094215</td>
          <td>19.384096</td>
          <td>27.037578</td>
        </tr>
        <tr>
          <th>4</th>
          <td>25.887375</td>
          <td>0.230447</td>
          <td>18.879393</td>
          <td>0.005010</td>
          <td>21.304079</td>
          <td>0.005157</td>
          <td>22.350945</td>
          <td>0.006987</td>
          <td>24.598196</td>
          <td>0.071212</td>
          <td>20.529193</td>
          <td>0.006541</td>
          <td>25.250517</td>
          <td>20.678023</td>
          <td>20.051526</td>
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
          <td>24.131671</td>
          <td>0.050552</td>
          <td>28.106538</td>
          <td>0.507644</td>
          <td>22.883604</td>
          <td>0.006928</td>
          <td>25.138450</td>
          <td>0.060028</td>
          <td>21.204901</td>
          <td>0.006052</td>
          <td>25.412510</td>
          <td>0.312172</td>
          <td>22.405467</td>
          <td>26.825087</td>
          <td>23.772143</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.426679</td>
          <td>0.012243</td>
          <td>22.548500</td>
          <td>0.006512</td>
          <td>21.209239</td>
          <td>0.005135</td>
          <td>19.781532</td>
          <td>0.005037</td>
          <td>26.681113</td>
          <td>0.410483</td>
          <td>23.409586</td>
          <td>0.056088</td>
          <td>18.439370</td>
          <td>20.760538</td>
          <td>19.456022</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.407912</td>
          <td>0.006907</td>
          <td>22.747877</td>
          <td>0.007036</td>
          <td>26.259428</td>
          <td>0.099064</td>
          <td>25.753070</td>
          <td>0.103245</td>
          <td>22.390196</td>
          <td>0.010944</td>
          <td>25.716502</td>
          <td>0.396413</td>
          <td>20.630273</td>
          <td>26.714885</td>
          <td>21.325026</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.603443</td>
          <td>0.076551</td>
          <td>18.051364</td>
          <td>0.005004</td>
          <td>21.916779</td>
          <td>0.005416</td>
          <td>18.764803</td>
          <td>0.005010</td>
          <td>24.600973</td>
          <td>0.071387</td>
          <td>25.143154</td>
          <td>0.250918</td>
          <td>21.820861</td>
          <td>21.994287</td>
          <td>22.769991</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.692313</td>
          <td>0.007815</td>
          <td>23.221949</td>
          <td>0.008999</td>
          <td>24.837833</td>
          <td>0.028195</td>
          <td>23.685789</td>
          <td>0.016908</td>
          <td>21.567214</td>
          <td>0.006848</td>
          <td>25.352039</td>
          <td>0.297386</td>
          <td>26.636297</td>
          <td>25.423507</td>
          <td>21.930060</td>
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
          <td>25.115821</td>
          <td>23.635735</td>
          <td>17.804598</td>
          <td>23.215472</td>
          <td>22.770758</td>
          <td>21.427868</td>
          <td>23.486941</td>
          <td>0.009858</td>
          <td>27.002717</td>
          <td>0.324597</td>
          <td>21.410684</td>
          <td>0.005457</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.102749</td>
          <td>21.478506</td>
          <td>23.584137</td>
          <td>24.350449</td>
          <td>21.686557</td>
          <td>18.422865</td>
          <td>25.791118</td>
          <td>0.069239</td>
          <td>21.207696</td>
          <td>0.005318</td>
          <td>24.326579</td>
          <td>0.032075</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.735998</td>
          <td>18.793369</td>
          <td>19.063401</td>
          <td>28.816046</td>
          <td>27.043125</td>
          <td>22.218105</td>
          <td>25.144254</td>
          <td>0.038918</td>
          <td>23.199145</td>
          <td>0.012370</td>
          <td>22.037550</td>
          <td>0.006336</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.182941</td>
          <td>15.951134</td>
          <td>22.273621</td>
          <td>19.504070</td>
          <td>27.771019</td>
          <td>23.698619</td>
          <td>23.094543</td>
          <td>0.007752</td>
          <td>19.376738</td>
          <td>0.005011</td>
          <td>27.082332</td>
          <td>0.345733</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.047467</td>
          <td>18.877125</td>
          <td>21.308809</td>
          <td>22.352723</td>
          <td>24.619465</td>
          <td>20.526776</td>
          <td>25.175042</td>
          <td>0.040000</td>
          <td>20.682039</td>
          <td>0.005123</td>
          <td>20.053143</td>
          <td>0.005039</td>
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
          <td>24.124411</td>
          <td>27.850721</td>
          <td>22.889233</td>
          <td>25.143905</td>
          <td>21.208115</td>
          <td>25.922464</td>
          <td>22.409271</td>
          <td>0.005912</td>
          <td>26.725145</td>
          <td>0.259396</td>
          <td>23.790457</td>
          <td>0.020074</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.439643</td>
          <td>22.544920</td>
          <td>21.207710</td>
          <td>19.785157</td>
          <td>26.539721</td>
          <td>23.431689</td>
          <td>18.449243</td>
          <td>0.005001</td>
          <td>20.773791</td>
          <td>0.005146</td>
          <td>19.448229</td>
          <td>0.005013</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.411468</td>
          <td>22.750965</td>
          <td>26.172398</td>
          <td>25.795807</td>
          <td>22.394124</td>
          <td>25.751225</td>
          <td>20.630805</td>
          <td>0.005037</td>
          <td>27.153623</td>
          <td>0.365640</td>
          <td>21.324840</td>
          <td>0.005392</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.528914</td>
          <td>18.047077</td>
          <td>21.907181</td>
          <td>18.759660</td>
          <td>24.684787</td>
          <td>25.233606</td>
          <td>21.828885</td>
          <td>0.005331</td>
          <td>21.993975</td>
          <td>0.006243</td>
          <td>22.778232</td>
          <td>0.009172</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.689360</td>
          <td>23.226261</td>
          <td>24.832386</td>
          <td>23.661843</td>
          <td>21.571665</td>
          <td>25.178646</td>
          <td>26.645548</td>
          <td>0.146414</td>
          <td>25.561939</td>
          <td>0.096097</td>
          <td>21.941233</td>
          <td>0.006138</td>
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
          <td>25.115821</td>
          <td>23.635735</td>
          <td>17.804598</td>
          <td>23.215472</td>
          <td>22.770758</td>
          <td>21.427868</td>
          <td>23.475578</td>
          <td>0.097256</td>
          <td>25.331722</td>
          <td>0.388547</td>
          <td>21.401286</td>
          <td>0.014507</td>
        </tr>
        <tr>
          <th>1</th>
          <td>24.102749</td>
          <td>21.478506</td>
          <td>23.584137</td>
          <td>24.350449</td>
          <td>21.686557</td>
          <td>18.422865</td>
          <td>26.182850</td>
          <td>0.822403</td>
          <td>21.208080</td>
          <td>0.011547</td>
          <td>24.540943</td>
          <td>0.222793</td>
        </tr>
        <tr>
          <th>2</th>
          <td>24.735998</td>
          <td>18.793369</td>
          <td>19.063401</td>
          <td>28.816046</td>
          <td>27.043125</td>
          <td>22.218105</td>
          <td>25.778959</td>
          <td>0.626627</td>
          <td>23.154114</td>
          <td>0.061300</td>
          <td>22.013778</td>
          <td>0.024354</td>
        </tr>
        <tr>
          <th>3</th>
          <td>24.182941</td>
          <td>15.951134</td>
          <td>22.273621</td>
          <td>19.504070</td>
          <td>27.771019</td>
          <td>23.698619</td>
          <td>23.064966</td>
          <td>0.067649</td>
          <td>19.377318</td>
          <td>0.005360</td>
          <td>inf</td>
          <td>inf</td>
        </tr>
        <tr>
          <th>4</th>
          <td>27.047467</td>
          <td>18.877125</td>
          <td>21.308809</td>
          <td>22.352723</td>
          <td>24.619465</td>
          <td>20.526776</td>
          <td>25.358166</td>
          <td>0.461761</td>
          <td>20.677067</td>
          <td>0.008115</td>
          <td>20.051516</td>
          <td>0.006367</td>
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
          <td>24.124411</td>
          <td>27.850721</td>
          <td>22.889233</td>
          <td>25.143905</td>
          <td>21.208115</td>
          <td>25.922464</td>
          <td>22.380305</td>
          <td>0.036765</td>
          <td>26.792922</td>
          <td>1.059231</td>
          <td>23.555229</td>
          <td>0.095532</td>
        </tr>
        <tr>
          <th>996</th>
          <td>22.439643</td>
          <td>22.544920</td>
          <td>21.207710</td>
          <td>19.785157</td>
          <td>26.539721</td>
          <td>23.431689</td>
          <td>18.447614</td>
          <td>0.005096</td>
          <td>20.752284</td>
          <td>0.008480</td>
          <td>19.458953</td>
          <td>0.005497</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.411468</td>
          <td>22.750965</td>
          <td>26.172398</td>
          <td>25.795807</td>
          <td>22.394124</td>
          <td>25.751225</td>
          <td>20.626463</td>
          <td>0.008874</td>
          <td>26.470717</td>
          <td>0.869955</td>
          <td>21.318706</td>
          <td>0.013580</td>
        </tr>
        <tr>
          <th>998</th>
          <td>24.528914</td>
          <td>18.047077</td>
          <td>21.907181</td>
          <td>18.759660</td>
          <td>24.684787</td>
          <td>25.233606</td>
          <td>21.810455</td>
          <td>0.022261</td>
          <td>21.990250</td>
          <td>0.021875</td>
          <td>22.762794</td>
          <td>0.047285</td>
        </tr>
        <tr>
          <th>999</th>
          <td>21.689360</td>
          <td>23.226261</td>
          <td>24.832386</td>
          <td>23.661843</td>
          <td>21.571665</td>
          <td>25.178646</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.301715</td>
          <td>0.780065</td>
          <td>21.935885</td>
          <td>0.022757</td>
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


