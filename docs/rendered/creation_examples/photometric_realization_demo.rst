Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f09dbc17fd0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



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
          <th>redshift</th>
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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>27.985053</td>
          <td>1.049175</td>
          <td>26.539081</td>
          <td>0.143230</td>
          <td>26.162273</td>
          <td>0.090965</td>
          <td>25.308713</td>
          <td>0.069807</td>
          <td>24.926984</td>
          <td>0.095155</td>
          <td>24.692576</td>
          <td>0.172119</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.990456</td>
          <td>1.052524</td>
          <td>27.805370</td>
          <td>0.404752</td>
          <td>27.517513</td>
          <td>0.287885</td>
          <td>26.806748</td>
          <td>0.253288</td>
          <td>27.678071</td>
          <td>0.831372</td>
          <td>27.158822</td>
          <td>1.063405</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.722049</td>
          <td>0.446894</td>
          <td>25.946499</td>
          <td>0.085486</td>
          <td>24.784501</td>
          <td>0.026911</td>
          <td>23.904153</td>
          <td>0.020303</td>
          <td>23.169849</td>
          <td>0.020290</td>
          <td>22.843631</td>
          <td>0.033965</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.946516</td>
          <td>0.527824</td>
          <td>29.226144</td>
          <td>1.066721</td>
          <td>26.932773</td>
          <td>0.177228</td>
          <td>26.593277</td>
          <td>0.212240</td>
          <td>26.099870</td>
          <td>0.258747</td>
          <td>25.175401</td>
          <td>0.257643</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.115893</td>
          <td>0.277887</td>
          <td>25.822575</td>
          <td>0.076646</td>
          <td>25.489781</td>
          <td>0.050165</td>
          <td>24.822823</td>
          <td>0.045359</td>
          <td>24.344122</td>
          <td>0.056853</td>
          <td>23.590738</td>
          <td>0.065863</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <th>99995</th>
          <td>2.147172</td>
          <td>26.327538</td>
          <td>0.329290</td>
          <td>26.306316</td>
          <td>0.117113</td>
          <td>26.439962</td>
          <td>0.115988</td>
          <td>25.861637</td>
          <td>0.113514</td>
          <td>26.030912</td>
          <td>0.244501</td>
          <td>27.301891</td>
          <td>1.154922</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.034266</td>
          <td>0.562416</td>
          <td>26.921185</td>
          <td>0.198239</td>
          <td>26.890651</td>
          <td>0.170999</td>
          <td>26.362694</td>
          <td>0.174764</td>
          <td>26.130824</td>
          <td>0.265378</td>
          <td>24.940654</td>
          <td>0.212157</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.845577</td>
          <td>0.490106</td>
          <td>27.191475</td>
          <td>0.248189</td>
          <td>27.040280</td>
          <td>0.194088</td>
          <td>26.702882</td>
          <td>0.232501</td>
          <td>25.603633</td>
          <td>0.170924</td>
          <td>25.596825</td>
          <td>0.361206</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.013879</td>
          <td>0.554228</td>
          <td>26.759878</td>
          <td>0.172985</td>
          <td>26.695328</td>
          <td>0.144683</td>
          <td>25.795029</td>
          <td>0.107104</td>
          <td>25.678302</td>
          <td>0.182104</td>
          <td>25.590370</td>
          <td>0.359384</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.903722</td>
          <td>0.195350</td>
          <td>26.083046</td>
          <td>0.084838</td>
          <td>25.494787</td>
          <td>0.082282</td>
          <td>24.955896</td>
          <td>0.097599</td>
          <td>25.274887</td>
          <td>0.279410</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




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
          <th>redshift</th>
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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.540314</td>
          <td>0.164825</td>
          <td>25.965356</td>
          <td>0.089944</td>
          <td>25.367193</td>
          <td>0.087081</td>
          <td>25.063574</td>
          <td>0.125850</td>
          <td>24.737139</td>
          <td>0.209769</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.344800</td>
          <td>2.021945</td>
          <td>31.425783</td>
          <td>2.876811</td>
          <td>27.218921</td>
          <td>0.409450</td>
          <td>26.448350</td>
          <td>0.395309</td>
          <td>25.655184</td>
          <td>0.437429</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.839048</td>
          <td>0.545522</td>
          <td>25.927876</td>
          <td>0.099013</td>
          <td>24.755577</td>
          <td>0.031553</td>
          <td>23.863132</td>
          <td>0.023662</td>
          <td>23.154201</td>
          <td>0.023984</td>
          <td>22.856319</td>
          <td>0.041618</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.948240</td>
          <td>1.148060</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.595268</td>
          <td>0.376410</td>
          <td>26.382257</td>
          <td>0.223036</td>
          <td>25.940253</td>
          <td>0.280782</td>
          <td>25.661355</td>
          <td>0.466091</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.885692</td>
          <td>0.256944</td>
          <td>25.797348</td>
          <td>0.086566</td>
          <td>25.488743</td>
          <td>0.059048</td>
          <td>24.842784</td>
          <td>0.054780</td>
          <td>24.461092</td>
          <td>0.074253</td>
          <td>23.648093</td>
          <td>0.082012</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <th>99995</th>
          <td>2.147172</td>
          <td>27.570089</td>
          <td>0.893769</td>
          <td>26.325499</td>
          <td>0.139684</td>
          <td>26.103705</td>
          <td>0.103699</td>
          <td>25.951015</td>
          <td>0.147917</td>
          <td>25.517167</td>
          <td>0.189390</td>
          <td>25.091968</td>
          <td>0.286640</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.814626</td>
          <td>0.208538</td>
          <td>26.731101</td>
          <td>0.175274</td>
          <td>26.445381</td>
          <td>0.221026</td>
          <td>25.602848</td>
          <td>0.200300</td>
          <td>25.234923</td>
          <td>0.316491</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.363612</td>
          <td>0.779213</td>
          <td>27.135526</td>
          <td>0.273733</td>
          <td>26.947315</td>
          <td>0.212031</td>
          <td>26.522495</td>
          <td>0.237610</td>
          <td>26.118360</td>
          <td>0.308377</td>
          <td>25.371876</td>
          <td>0.355554</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.942449</td>
          <td>0.590816</td>
          <td>27.591559</td>
          <td>0.398899</td>
          <td>26.656929</td>
          <td>0.168961</td>
          <td>25.922655</td>
          <td>0.145885</td>
          <td>25.437715</td>
          <td>0.178867</td>
          <td>26.210673</td>
          <td>0.671218</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.930158</td>
          <td>1.837157</td>
          <td>26.598344</td>
          <td>0.174687</td>
          <td>26.040197</td>
          <td>0.097023</td>
          <td>25.815287</td>
          <td>0.130126</td>
          <td>25.410751</td>
          <td>0.171248</td>
          <td>25.209382</td>
          <td>0.311800</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




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
          <th>redshift</th>
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
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.890625</td>
          <td>30.312859</td>
          <td>2.932316</td>
          <td>26.643322</td>
          <td>0.156643</td>
          <td>26.118917</td>
          <td>0.087573</td>
          <td>25.278074</td>
          <td>0.067948</td>
          <td>25.136812</td>
          <td>0.114336</td>
          <td>24.930644</td>
          <td>0.210417</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.929742</td>
          <td>1.722971</td>
          <td>27.862584</td>
          <td>0.423168</td>
          <td>27.826884</td>
          <td>0.368456</td>
          <td>27.009381</td>
          <td>0.298908</td>
          <td>27.482508</td>
          <td>0.731686</td>
          <td>27.128731</td>
          <td>1.045406</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.143703</td>
          <td>0.636974</td>
          <td>25.979626</td>
          <td>0.094586</td>
          <td>24.770126</td>
          <td>0.028845</td>
          <td>23.868607</td>
          <td>0.021416</td>
          <td>23.197311</td>
          <td>0.022505</td>
          <td>22.847289</td>
          <td>0.037120</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.393196</td>
          <td>3.185469</td>
          <td>28.009048</td>
          <td>0.558247</td>
          <td>26.993171</td>
          <td>0.231126</td>
          <td>26.594823</td>
          <td>0.264819</td>
          <td>25.893972</td>
          <td>0.269527</td>
          <td>25.277290</td>
          <td>0.345810</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.258491</td>
          <td>0.311961</td>
          <td>25.869918</td>
          <td>0.080012</td>
          <td>25.489237</td>
          <td>0.050213</td>
          <td>24.797573</td>
          <td>0.044420</td>
          <td>24.247411</td>
          <td>0.052251</td>
          <td>23.623135</td>
          <td>0.067882</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <th>99995</th>
          <td>2.147172</td>
          <td>26.255813</td>
          <td>0.327174</td>
          <td>26.409602</td>
          <td>0.137086</td>
          <td>26.140039</td>
          <td>0.096516</td>
          <td>26.177807</td>
          <td>0.161710</td>
          <td>25.817014</td>
          <td>0.220560</td>
          <td>26.014601</td>
          <td>0.531482</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.976971</td>
          <td>1.052255</td>
          <td>27.271029</td>
          <td>0.268445</td>
          <td>26.916399</td>
          <td>0.177586</td>
          <td>26.473327</td>
          <td>0.195120</td>
          <td>25.764352</td>
          <td>0.198929</td>
          <td>25.155029</td>
          <td>0.257452</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.729068</td>
          <td>0.462687</td>
          <td>27.340973</td>
          <td>0.291649</td>
          <td>26.507947</td>
          <td>0.129138</td>
          <td>26.217379</td>
          <td>0.162285</td>
          <td>25.875555</td>
          <td>0.225131</td>
          <td>25.332814</td>
          <td>0.306726</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.494707</td>
          <td>0.403624</td>
          <td>27.317749</td>
          <td>0.301908</td>
          <td>26.529286</td>
          <td>0.140326</td>
          <td>25.743385</td>
          <td>0.115322</td>
          <td>25.625689</td>
          <td>0.194455</td>
          <td>25.151150</td>
          <td>0.281974</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.896215</td>
          <td>0.520450</td>
          <td>26.713066</td>
          <td>0.171784</td>
          <td>26.161182</td>
          <td>0.094487</td>
          <td>25.642118</td>
          <td>0.097574</td>
          <td>25.313066</td>
          <td>0.138405</td>
          <td>24.709469</td>
          <td>0.181547</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
