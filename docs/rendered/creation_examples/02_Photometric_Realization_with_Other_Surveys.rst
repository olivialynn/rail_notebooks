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
          <td>25.304883</td>
          <td>22.058332</td>
          <td>24.254380</td>
          <td>27.413820</td>
          <td>22.597435</td>
          <td>21.037787</td>
          <td>26.034366</td>
          <td>29.719932</td>
          <td>24.444568</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.279411</td>
          <td>25.738319</td>
          <td>21.211093</td>
          <td>18.492386</td>
          <td>26.267760</td>
          <td>21.213590</td>
          <td>29.671552</td>
          <td>24.026941</td>
          <td>26.167710</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.846128</td>
          <td>18.026834</td>
          <td>23.738532</td>
          <td>18.924827</td>
          <td>27.368588</td>
          <td>24.068488</td>
          <td>13.846258</td>
          <td>20.362969</td>
          <td>24.827871</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.051107</td>
          <td>26.724214</td>
          <td>24.054839</td>
          <td>25.227698</td>
          <td>22.011011</td>
          <td>24.319362</td>
          <td>26.229927</td>
          <td>25.443678</td>
          <td>21.069779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.837077</td>
          <td>27.424184</td>
          <td>21.609521</td>
          <td>23.149877</td>
          <td>25.502406</td>
          <td>24.163304</td>
          <td>19.789257</td>
          <td>16.568189</td>
          <td>25.776315</td>
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
          <td>26.331742</td>
          <td>24.529277</td>
          <td>24.063293</td>
          <td>24.668682</td>
          <td>24.642778</td>
          <td>25.398770</td>
          <td>21.368837</td>
          <td>20.631450</td>
          <td>23.615064</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.577742</td>
          <td>23.738217</td>
          <td>23.381833</td>
          <td>20.847969</td>
          <td>22.920890</td>
          <td>19.686091</td>
          <td>25.943399</td>
          <td>19.402930</td>
          <td>17.773181</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.174209</td>
          <td>23.017301</td>
          <td>23.931259</td>
          <td>21.989212</td>
          <td>24.256786</td>
          <td>23.330359</td>
          <td>25.636995</td>
          <td>24.855099</td>
          <td>20.099619</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.233359</td>
          <td>23.093428</td>
          <td>24.218735</td>
          <td>17.636143</td>
          <td>23.591282</td>
          <td>18.566829</td>
          <td>22.571282</td>
          <td>22.434721</td>
          <td>27.664484</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.943097</td>
          <td>23.694977</td>
          <td>19.595860</td>
          <td>20.529159</td>
          <td>20.815311</td>
          <td>24.528727</td>
          <td>21.863855</td>
          <td>21.977573</td>
          <td>28.057831</td>
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
          <td>25.089477</td>
          <td>0.117020</td>
          <td>22.068279</td>
          <td>0.005725</td>
          <td>24.243321</td>
          <td>0.016967</td>
          <td>27.166929</td>
          <td>0.338630</td>
          <td>22.585190</td>
          <td>0.012633</td>
          <td>21.041347</td>
          <td>0.008305</td>
          <td>26.034366</td>
          <td>29.719932</td>
          <td>24.444568</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.282082</td>
          <td>0.006601</td>
          <td>25.822510</td>
          <td>0.076642</td>
          <td>21.213626</td>
          <td>0.005136</td>
          <td>18.489537</td>
          <td>0.005007</td>
          <td>26.300760</td>
          <td>0.304515</td>
          <td>21.217374</td>
          <td>0.009230</td>
          <td>29.671552</td>
          <td>24.026941</td>
          <td>26.167710</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.848016</td>
          <td>0.005069</td>
          <td>18.034029</td>
          <td>0.005004</td>
          <td>23.725131</td>
          <td>0.011324</td>
          <td>18.928490</td>
          <td>0.005012</td>
          <td>28.724087</td>
          <td>1.509166</td>
          <td>23.935586</td>
          <td>0.089311</td>
          <td>13.846258</td>
          <td>20.362969</td>
          <td>24.827871</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.056417</td>
          <td>0.005011</td>
          <td>26.481587</td>
          <td>0.136310</td>
          <td>24.054428</td>
          <td>0.014556</td>
          <td>25.240674</td>
          <td>0.065724</td>
          <td>21.994989</td>
          <td>0.008469</td>
          <td>24.432682</td>
          <td>0.137765</td>
          <td>26.229927</td>
          <td>25.443678</td>
          <td>21.069779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.883423</td>
          <td>0.040652</td>
          <td>28.017333</td>
          <td>0.475199</td>
          <td>21.620388</td>
          <td>0.005258</td>
          <td>23.151661</td>
          <td>0.011159</td>
          <td>25.656683</td>
          <td>0.178800</td>
          <td>24.219729</td>
          <td>0.114538</td>
          <td>19.789257</td>
          <td>16.568189</td>
          <td>25.776315</td>
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
          <td>26.696586</td>
          <td>0.438380</td>
          <td>24.536528</td>
          <td>0.024699</td>
          <td>24.072754</td>
          <td>0.014771</td>
          <td>24.658458</td>
          <td>0.039206</td>
          <td>24.663993</td>
          <td>0.075478</td>
          <td>25.469326</td>
          <td>0.326639</td>
          <td>21.368837</td>
          <td>20.631450</td>
          <td>23.615064</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.582564</td>
          <td>0.005604</td>
          <td>23.733138</td>
          <td>0.012789</td>
          <td>23.375060</td>
          <td>0.008949</td>
          <td>20.846374</td>
          <td>0.005181</td>
          <td>22.929048</td>
          <td>0.016591</td>
          <td>19.685679</td>
          <td>0.005403</td>
          <td>25.943399</td>
          <td>19.402930</td>
          <td>17.773181</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.168998</td>
          <td>0.006367</td>
          <td>23.014523</td>
          <td>0.007995</td>
          <td>23.928520</td>
          <td>0.013187</td>
          <td>21.979980</td>
          <td>0.006117</td>
          <td>24.179490</td>
          <td>0.049123</td>
          <td>23.286116</td>
          <td>0.050266</td>
          <td>25.636995</td>
          <td>24.855099</td>
          <td>20.099619</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.244952</td>
          <td>0.006520</td>
          <td>23.098841</td>
          <td>0.008373</td>
          <td>24.241008</td>
          <td>0.016935</td>
          <td>17.633819</td>
          <td>0.005003</td>
          <td>23.499938</td>
          <td>0.026964</td>
          <td>18.567987</td>
          <td>0.005071</td>
          <td>22.571282</td>
          <td>22.434721</td>
          <td>27.664484</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.946105</td>
          <td>0.006001</td>
          <td>23.681286</td>
          <td>0.012301</td>
          <td>19.590027</td>
          <td>0.005014</td>
          <td>20.523880</td>
          <td>0.005110</td>
          <td>20.817402</td>
          <td>0.005566</td>
          <td>24.957622</td>
          <td>0.215184</td>
          <td>21.863855</td>
          <td>21.977573</td>
          <td>28.057831</td>
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
          <td>25.304883</td>
          <td>22.058332</td>
          <td>24.254380</td>
          <td>27.413820</td>
          <td>22.597435</td>
          <td>21.037787</td>
          <td>25.754989</td>
          <td>0.067052</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.425097</td>
          <td>0.035005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.279411</td>
          <td>25.738319</td>
          <td>21.211093</td>
          <td>18.492386</td>
          <td>26.267760</td>
          <td>21.213590</td>
          <td>30.466586</td>
          <td>1.921876</td>
          <td>24.030238</td>
          <td>0.024707</td>
          <td>26.152836</td>
          <td>0.160535</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.846128</td>
          <td>18.026834</td>
          <td>23.738532</td>
          <td>18.924827</td>
          <td>27.368588</td>
          <td>24.068488</td>
          <td>13.843934</td>
          <td>0.005000</td>
          <td>20.369730</td>
          <td>0.005070</td>
          <td>24.889083</td>
          <td>0.052920</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.051107</td>
          <td>26.724214</td>
          <td>24.054839</td>
          <td>25.227698</td>
          <td>22.011011</td>
          <td>24.319362</td>
          <td>26.247962</td>
          <td>0.103636</td>
          <td>25.438208</td>
          <td>0.086172</td>
          <td>21.068794</td>
          <td>0.005248</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.837077</td>
          <td>27.424184</td>
          <td>21.609521</td>
          <td>23.149877</td>
          <td>25.502406</td>
          <td>24.163304</td>
          <td>19.789244</td>
          <td>0.005008</td>
          <td>16.574851</td>
          <td>0.005000</td>
          <td>25.781095</td>
          <td>0.116429</td>
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
          <td>26.331742</td>
          <td>24.529277</td>
          <td>24.063293</td>
          <td>24.668682</td>
          <td>24.642778</td>
          <td>25.398770</td>
          <td>21.362723</td>
          <td>0.005143</td>
          <td>20.626469</td>
          <td>0.005111</td>
          <td>23.608959</td>
          <td>0.017211</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.577742</td>
          <td>23.738217</td>
          <td>23.381833</td>
          <td>20.847969</td>
          <td>22.920890</td>
          <td>19.686091</td>
          <td>25.906901</td>
          <td>0.076726</td>
          <td>19.402774</td>
          <td>0.005012</td>
          <td>17.774902</td>
          <td>0.005001</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.174209</td>
          <td>23.017301</td>
          <td>23.931259</td>
          <td>21.989212</td>
          <td>24.256786</td>
          <td>23.330359</td>
          <td>25.669086</td>
          <td>0.062122</td>
          <td>24.889645</td>
          <td>0.052947</td>
          <td>20.106690</td>
          <td>0.005043</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.233359</td>
          <td>23.093428</td>
          <td>24.218735</td>
          <td>17.636143</td>
          <td>23.591282</td>
          <td>18.566829</td>
          <td>22.564809</td>
          <td>0.006184</td>
          <td>22.448171</td>
          <td>0.007565</td>
          <td>26.861648</td>
          <td>0.289868</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.943097</td>
          <td>23.694977</td>
          <td>19.595860</td>
          <td>20.529159</td>
          <td>20.815311</td>
          <td>24.528727</td>
          <td>21.861813</td>
          <td>0.005351</td>
          <td>21.977181</td>
          <td>0.006209</td>
          <td>28.396165</td>
          <td>0.884057</td>
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
          <td>25.304883</td>
          <td>22.058332</td>
          <td>24.254380</td>
          <td>27.413820</td>
          <td>22.597435</td>
          <td>21.037787</td>
          <td>26.363884</td>
          <td>0.922297</td>
          <td>inf</td>
          <td>inf</td>
          <td>24.578777</td>
          <td>0.229909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>21.279411</td>
          <td>25.738319</td>
          <td>21.211093</td>
          <td>18.492386</td>
          <td>26.267760</td>
          <td>21.213590</td>
          <td>26.404104</td>
          <td>0.945499</td>
          <td>24.003747</td>
          <td>0.129532</td>
          <td>26.419359</td>
          <td>0.897037</td>
        </tr>
        <tr>
          <th>2</th>
          <td>18.846128</td>
          <td>18.026834</td>
          <td>23.738532</td>
          <td>18.924827</td>
          <td>27.368588</td>
          <td>24.068488</td>
          <td>13.849801</td>
          <td>0.005000</td>
          <td>20.359246</td>
          <td>0.006911</td>
          <td>24.837347</td>
          <td>0.284224</td>
        </tr>
        <tr>
          <th>3</th>
          <td>17.051107</td>
          <td>26.724214</td>
          <td>24.054839</td>
          <td>25.227698</td>
          <td>22.011011</td>
          <td>24.319362</td>
          <td>26.252061</td>
          <td>0.859711</td>
          <td>25.563241</td>
          <td>0.463521</td>
          <td>21.065324</td>
          <td>0.011187</td>
        </tr>
        <tr>
          <th>4</th>
          <td>23.837077</td>
          <td>27.424184</td>
          <td>21.609521</td>
          <td>23.149877</td>
          <td>25.502406</td>
          <td>24.163304</td>
          <td>19.798464</td>
          <td>0.006060</td>
          <td>16.565520</td>
          <td>0.005002</td>
          <td>25.272935</td>
          <td>0.401109</td>
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
          <td>26.331742</td>
          <td>24.529277</td>
          <td>24.063293</td>
          <td>24.668682</td>
          <td>24.642778</td>
          <td>25.398770</td>
          <td>21.383468</td>
          <td>0.015510</td>
          <td>20.623592</td>
          <td>0.007875</td>
          <td>23.501603</td>
          <td>0.091127</td>
        </tr>
        <tr>
          <th>996</th>
          <td>20.577742</td>
          <td>23.738217</td>
          <td>23.381833</td>
          <td>20.847969</td>
          <td>22.920890</td>
          <td>19.686091</td>
          <td>25.053512</td>
          <td>0.365608</td>
          <td>19.406848</td>
          <td>0.005380</td>
          <td>17.768155</td>
          <td>0.005023</td>
        </tr>
        <tr>
          <th>997</th>
          <td>21.174209</td>
          <td>23.017301</td>
          <td>23.931259</td>
          <td>21.989212</td>
          <td>24.256786</td>
          <td>23.330359</td>
          <td>25.350025</td>
          <td>0.458948</td>
          <td>24.732080</td>
          <td>0.240281</td>
          <td>20.101298</td>
          <td>0.006483</td>
        </tr>
        <tr>
          <th>998</th>
          <td>21.233359</td>
          <td>23.093428</td>
          <td>24.218735</td>
          <td>17.636143</td>
          <td>23.591282</td>
          <td>18.566829</td>
          <td>22.660318</td>
          <td>0.047181</td>
          <td>22.405489</td>
          <td>0.031481</td>
          <td>25.104803</td>
          <td>0.351906</td>
        </tr>
        <tr>
          <th>999</th>
          <td>20.943097</td>
          <td>23.694977</td>
          <td>19.595860</td>
          <td>20.529159</td>
          <td>20.815311</td>
          <td>24.528727</td>
          <td>21.859572</td>
          <td>0.023230</td>
          <td>21.991981</td>
          <td>0.021907</td>
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


