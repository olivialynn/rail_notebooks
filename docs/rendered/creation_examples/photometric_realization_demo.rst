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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fa759d7afb0>



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
          <td>27.283447</td>
          <td>0.670030</td>
          <td>26.425218</td>
          <td>0.129833</td>
          <td>25.945191</td>
          <td>0.075120</td>
          <td>25.320743</td>
          <td>0.070554</td>
          <td>25.229330</td>
          <td>0.123898</td>
          <td>24.782879</td>
          <td>0.185812</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.670887</td>
          <td>0.753767</td>
          <td>28.537026</td>
          <td>0.623696</td>
          <td>27.636384</td>
          <td>0.485495</td>
          <td>26.639008</td>
          <td>0.397410</td>
          <td>25.608709</td>
          <td>0.364580</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.474988</td>
          <td>0.369762</td>
          <td>25.935041</td>
          <td>0.084629</td>
          <td>24.788700</td>
          <td>0.027010</td>
          <td>23.872183</td>
          <td>0.019760</td>
          <td>23.178740</td>
          <td>0.020443</td>
          <td>22.826635</td>
          <td>0.033460</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.081266</td>
          <td>0.226593</td>
          <td>27.680523</td>
          <td>0.328064</td>
          <td>26.935056</td>
          <td>0.281236</td>
          <td>25.678303</td>
          <td>0.182104</td>
          <td>25.859460</td>
          <td>0.442147</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.234176</td>
          <td>0.647635</td>
          <td>25.658454</td>
          <td>0.066303</td>
          <td>25.408084</td>
          <td>0.046655</td>
          <td>24.804770</td>
          <td>0.044638</td>
          <td>24.426670</td>
          <td>0.061173</td>
          <td>23.614289</td>
          <td>0.067252</td>
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
          <td>25.746083</td>
          <td>0.204892</td>
          <td>26.548931</td>
          <td>0.144449</td>
          <td>26.187783</td>
          <td>0.093028</td>
          <td>26.180619</td>
          <td>0.149598</td>
          <td>26.260333</td>
          <td>0.294774</td>
          <td>25.688266</td>
          <td>0.387858</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.708707</td>
          <td>0.442416</td>
          <td>26.855908</td>
          <td>0.187637</td>
          <td>26.434553</td>
          <td>0.115443</td>
          <td>26.392957</td>
          <td>0.179309</td>
          <td>25.821843</td>
          <td>0.205507</td>
          <td>25.653200</td>
          <td>0.377450</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.940583</td>
          <td>0.525546</td>
          <td>27.121237</td>
          <td>0.234222</td>
          <td>27.084440</td>
          <td>0.201430</td>
          <td>26.762051</td>
          <td>0.244147</td>
          <td>25.915259</td>
          <td>0.222177</td>
          <td>25.342503</td>
          <td>0.295111</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.261823</td>
          <td>0.262917</td>
          <td>26.751610</td>
          <td>0.151850</td>
          <td>25.816886</td>
          <td>0.109168</td>
          <td>25.615452</td>
          <td>0.172650</td>
          <td>25.408687</td>
          <td>0.311219</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.732447</td>
          <td>0.450409</td>
          <td>26.452646</td>
          <td>0.132947</td>
          <td>25.993838</td>
          <td>0.078419</td>
          <td>25.578194</td>
          <td>0.088556</td>
          <td>25.096670</td>
          <td>0.110390</td>
          <td>24.897918</td>
          <td>0.204704</td>
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
          <td>27.208001</td>
          <td>0.697108</td>
          <td>26.478029</td>
          <td>0.156291</td>
          <td>26.139893</td>
          <td>0.104818</td>
          <td>25.349154</td>
          <td>0.085709</td>
          <td>24.924092</td>
          <td>0.111478</td>
          <td>25.194816</td>
          <td>0.305301</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.100978</td>
          <td>0.569002</td>
          <td>28.196236</td>
          <td>0.557940</td>
          <td>28.372942</td>
          <td>0.916305</td>
          <td>26.484695</td>
          <td>0.406523</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.015826</td>
          <td>0.618731</td>
          <td>25.980754</td>
          <td>0.103697</td>
          <td>24.761413</td>
          <td>0.031715</td>
          <td>23.888003</td>
          <td>0.024176</td>
          <td>23.112856</td>
          <td>0.023145</td>
          <td>22.772368</td>
          <td>0.038639</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.831496</td>
          <td>1.073338</td>
          <td>30.674347</td>
          <td>2.367409</td>
          <td>26.850070</td>
          <td>0.205847</td>
          <td>27.065085</td>
          <td>0.386397</td>
          <td>26.095158</td>
          <td>0.318037</td>
          <td>25.044339</td>
          <td>0.288081</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.097372</td>
          <td>0.304972</td>
          <td>25.904229</td>
          <td>0.095078</td>
          <td>25.464448</td>
          <td>0.057790</td>
          <td>24.766250</td>
          <td>0.051183</td>
          <td>24.254563</td>
          <td>0.061849</td>
          <td>23.588698</td>
          <td>0.077826</td>
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
          <td>27.934394</td>
          <td>1.112798</td>
          <td>26.566994</td>
          <td>0.171736</td>
          <td>26.105122</td>
          <td>0.103828</td>
          <td>26.175340</td>
          <td>0.179123</td>
          <td>25.515894</td>
          <td>0.189186</td>
          <td>28.688663</td>
          <td>2.419903</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.045593</td>
          <td>0.548374</td>
          <td>26.752805</td>
          <td>0.178531</td>
          <td>26.272203</td>
          <td>0.191178</td>
          <td>26.060005</td>
          <td>0.291936</td>
          <td>25.258975</td>
          <td>0.322617</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.345493</td>
          <td>0.374155</td>
          <td>27.340833</td>
          <td>0.322891</td>
          <td>26.853753</td>
          <td>0.196035</td>
          <td>26.446466</td>
          <td>0.223100</td>
          <td>26.168290</td>
          <td>0.320926</td>
          <td>26.082552</td>
          <td>0.604695</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.246604</td>
          <td>0.304100</td>
          <td>26.363496</td>
          <td>0.131344</td>
          <td>25.852138</td>
          <td>0.137290</td>
          <td>25.152662</td>
          <td>0.140187</td>
          <td>25.940107</td>
          <td>0.554803</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.147853</td>
          <td>0.673051</td>
          <td>26.462604</td>
          <td>0.155610</td>
          <td>26.004206</td>
          <td>0.094007</td>
          <td>25.653976</td>
          <td>0.113117</td>
          <td>25.038329</td>
          <td>0.124351</td>
          <td>25.272689</td>
          <td>0.327938</td>
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
          <td>27.798550</td>
          <td>0.937563</td>
          <td>26.665855</td>
          <td>0.159689</td>
          <td>26.114302</td>
          <td>0.087218</td>
          <td>25.472087</td>
          <td>0.080662</td>
          <td>24.824074</td>
          <td>0.086936</td>
          <td>24.709807</td>
          <td>0.174681</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.167210</td>
          <td>0.618408</td>
          <td>27.806789</td>
          <td>0.405484</td>
          <td>26.898074</td>
          <td>0.172240</td>
          <td>27.088119</td>
          <td>0.318370</td>
          <td>27.017338</td>
          <td>0.528318</td>
          <td>26.625539</td>
          <td>0.762681</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.888188</td>
          <td>0.531010</td>
          <td>26.030634</td>
          <td>0.098908</td>
          <td>24.800138</td>
          <td>0.029613</td>
          <td>23.859899</td>
          <td>0.021257</td>
          <td>23.156663</td>
          <td>0.021735</td>
          <td>22.795923</td>
          <td>0.035473</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.081772</td>
          <td>0.248661</td>
          <td>26.927176</td>
          <td>0.345790</td>
          <td>25.832196</td>
          <td>0.256260</td>
          <td>24.919066</td>
          <td>0.259297</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.062651</td>
          <td>0.266362</td>
          <td>25.800409</td>
          <td>0.075255</td>
          <td>25.414148</td>
          <td>0.046974</td>
          <td>24.831591</td>
          <td>0.045782</td>
          <td>24.351253</td>
          <td>0.057296</td>
          <td>23.878388</td>
          <td>0.085052</td>
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
          <td>26.289425</td>
          <td>0.336001</td>
          <td>26.528733</td>
          <td>0.151867</td>
          <td>26.053182</td>
          <td>0.089426</td>
          <td>25.812502</td>
          <td>0.118013</td>
          <td>25.983111</td>
          <td>0.253020</td>
          <td>25.359368</td>
          <td>0.322207</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.472798</td>
          <td>1.385357</td>
          <td>26.924825</td>
          <td>0.201586</td>
          <td>26.863614</td>
          <td>0.169797</td>
          <td>26.618519</td>
          <td>0.220340</td>
          <td>26.210270</td>
          <td>0.287409</td>
          <td>25.251826</td>
          <td>0.278596</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.623385</td>
          <td>1.515516</td>
          <td>27.274798</td>
          <td>0.276440</td>
          <td>27.279958</td>
          <td>0.248193</td>
          <td>26.731113</td>
          <td>0.249705</td>
          <td>26.146221</td>
          <td>0.281153</td>
          <td>25.199319</td>
          <td>0.275390</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.585612</td>
          <td>0.432623</td>
          <td>27.227991</td>
          <td>0.280819</td>
          <td>26.408469</td>
          <td>0.126412</td>
          <td>25.952416</td>
          <td>0.138228</td>
          <td>25.638618</td>
          <td>0.196582</td>
          <td>25.817920</td>
          <td>0.474285</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.621960</td>
          <td>0.855749</td>
          <td>26.803337</td>
          <td>0.185437</td>
          <td>26.087740</td>
          <td>0.088581</td>
          <td>25.597977</td>
          <td>0.093867</td>
          <td>25.286520</td>
          <td>0.135270</td>
          <td>24.740715</td>
          <td>0.186409</td>
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
