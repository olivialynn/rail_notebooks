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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1d7cf59ea0>



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
          <td>27.754101</td>
          <td>0.912033</td>
          <td>26.999503</td>
          <td>0.211681</td>
          <td>26.120865</td>
          <td>0.087711</td>
          <td>25.383917</td>
          <td>0.074609</td>
          <td>24.961928</td>
          <td>0.098117</td>
          <td>24.770901</td>
          <td>0.183940</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.336913</td>
          <td>2.057686</td>
          <td>27.960291</td>
          <td>0.455334</td>
          <td>28.243919</td>
          <td>0.505245</td>
          <td>26.891803</td>
          <td>0.271525</td>
          <td>27.792866</td>
          <td>0.894282</td>
          <td>26.839344</td>
          <td>0.875182</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.585340</td>
          <td>0.402724</td>
          <td>25.757389</td>
          <td>0.072361</td>
          <td>24.761141</td>
          <td>0.026369</td>
          <td>23.855541</td>
          <td>0.019484</td>
          <td>23.132631</td>
          <td>0.019660</td>
          <td>22.865018</td>
          <td>0.034612</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.747252</td>
          <td>0.792650</td>
          <td>27.672008</td>
          <td>0.325851</td>
          <td>26.770946</td>
          <td>0.245943</td>
          <td>26.127972</td>
          <td>0.264761</td>
          <td>25.153580</td>
          <td>0.253075</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.065719</td>
          <td>0.266785</td>
          <td>25.691672</td>
          <td>0.068279</td>
          <td>25.436446</td>
          <td>0.047845</td>
          <td>24.899727</td>
          <td>0.048564</td>
          <td>24.469430</td>
          <td>0.063536</td>
          <td>23.679001</td>
          <td>0.071217</td>
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
          <td>27.136826</td>
          <td>0.605012</td>
          <td>26.351694</td>
          <td>0.121821</td>
          <td>26.114113</td>
          <td>0.087192</td>
          <td>25.976334</td>
          <td>0.125417</td>
          <td>25.519239</td>
          <td>0.159054</td>
          <td>25.936564</td>
          <td>0.468541</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.803501</td>
          <td>0.940368</td>
          <td>26.884296</td>
          <td>0.192182</td>
          <td>26.872211</td>
          <td>0.168336</td>
          <td>26.479282</td>
          <td>0.192880</td>
          <td>26.156895</td>
          <td>0.271079</td>
          <td>25.882803</td>
          <td>0.450008</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.102335</td>
          <td>1.123285</td>
          <td>27.206396</td>
          <td>0.251250</td>
          <td>27.232686</td>
          <td>0.227962</td>
          <td>26.500910</td>
          <td>0.196424</td>
          <td>26.362836</td>
          <td>0.320016</td>
          <td>25.355704</td>
          <td>0.298265</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.878501</td>
          <td>0.428024</td>
          <td>26.588832</td>
          <td>0.131984</td>
          <td>25.887426</td>
          <td>0.116093</td>
          <td>25.461075</td>
          <td>0.151326</td>
          <td>26.576611</td>
          <td>0.737719</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.349954</td>
          <td>0.121637</td>
          <td>26.092091</td>
          <td>0.085517</td>
          <td>25.508036</td>
          <td>0.083249</td>
          <td>25.378028</td>
          <td>0.140899</td>
          <td>24.574440</td>
          <td>0.155616</td>
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
          <td>26.529625</td>
          <td>0.163331</td>
          <td>26.157403</td>
          <td>0.106434</td>
          <td>25.329091</td>
          <td>0.084208</td>
          <td>25.017905</td>
          <td>0.120961</td>
          <td>25.160560</td>
          <td>0.297011</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.907176</td>
          <td>0.565076</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.959539</td>
          <td>0.468966</td>
          <td>28.208226</td>
          <td>0.825516</td>
          <td>28.401657</td>
          <td>1.406699</td>
          <td>25.609031</td>
          <td>0.422349</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.787354</td>
          <td>1.021593</td>
          <td>25.853637</td>
          <td>0.092780</td>
          <td>24.791999</td>
          <td>0.032579</td>
          <td>23.903168</td>
          <td>0.024495</td>
          <td>23.145304</td>
          <td>0.023801</td>
          <td>22.825908</td>
          <td>0.040513</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.738048</td>
          <td>0.458625</td>
          <td>27.614470</td>
          <td>0.382067</td>
          <td>26.810428</td>
          <td>0.316252</td>
          <td>25.864971</td>
          <td>0.264099</td>
          <td>25.569786</td>
          <td>0.435018</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.589978</td>
          <td>0.894860</td>
          <td>25.812445</td>
          <td>0.087722</td>
          <td>25.429026</td>
          <td>0.056002</td>
          <td>24.849465</td>
          <td>0.055106</td>
          <td>24.393250</td>
          <td>0.069930</td>
          <td>23.596663</td>
          <td>0.078375</td>
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
          <td>26.102585</td>
          <td>0.310681</td>
          <td>26.198538</td>
          <td>0.125178</td>
          <td>26.119657</td>
          <td>0.105156</td>
          <td>26.028869</td>
          <td>0.158123</td>
          <td>25.860531</td>
          <td>0.252077</td>
          <td>25.160698</td>
          <td>0.302963</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.105254</td>
          <td>1.215489</td>
          <td>26.749535</td>
          <td>0.197466</td>
          <td>26.718652</td>
          <td>0.173431</td>
          <td>26.780533</td>
          <td>0.290948</td>
          <td>26.056211</td>
          <td>0.291044</td>
          <td>25.640707</td>
          <td>0.434177</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.946152</td>
          <td>1.852176</td>
          <td>27.078990</td>
          <td>0.261405</td>
          <td>26.914666</td>
          <td>0.206319</td>
          <td>26.601355</td>
          <td>0.253550</td>
          <td>25.874304</td>
          <td>0.252996</td>
          <td>26.364040</td>
          <td>0.733875</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.419724</td>
          <td>0.348948</td>
          <td>26.543386</td>
          <td>0.153346</td>
          <td>25.956094</td>
          <td>0.150135</td>
          <td>26.426868</td>
          <td>0.399594</td>
          <td>25.114298</td>
          <td>0.294734</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.314925</td>
          <td>0.364668</td>
          <td>26.407775</td>
          <td>0.148471</td>
          <td>26.003073</td>
          <td>0.093914</td>
          <td>25.704777</td>
          <td>0.118231</td>
          <td>25.285183</td>
          <td>0.153844</td>
          <td>24.632879</td>
          <td>0.194100</td>
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
          <td>28.471295</td>
          <td>1.374963</td>
          <td>26.687040</td>
          <td>0.162603</td>
          <td>26.002110</td>
          <td>0.079004</td>
          <td>25.393755</td>
          <td>0.075271</td>
          <td>24.963248</td>
          <td>0.098243</td>
          <td>24.819860</td>
          <td>0.191728</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.950828</td>
          <td>0.452421</td>
          <td>27.772510</td>
          <td>0.353095</td>
          <td>26.859574</td>
          <td>0.264727</td>
          <td>26.386339</td>
          <td>0.326338</td>
          <td>25.857185</td>
          <td>0.441761</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.809950</td>
          <td>0.501446</td>
          <td>25.915538</td>
          <td>0.089414</td>
          <td>24.780181</td>
          <td>0.029100</td>
          <td>23.856928</td>
          <td>0.021203</td>
          <td>23.162730</td>
          <td>0.021848</td>
          <td>22.789777</td>
          <td>0.035281</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.760425</td>
          <td>0.465089</td>
          <td>27.323817</td>
          <td>0.302726</td>
          <td>27.509776</td>
          <td>0.537896</td>
          <td>26.079256</td>
          <td>0.313007</td>
          <td>26.576719</td>
          <td>0.875930</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.609946</td>
          <td>0.833111</td>
          <td>25.826916</td>
          <td>0.077035</td>
          <td>25.433335</td>
          <td>0.047781</td>
          <td>24.788326</td>
          <td>0.044057</td>
          <td>24.375684</td>
          <td>0.058552</td>
          <td>23.643576</td>
          <td>0.069121</td>
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
          <td>26.104048</td>
          <td>0.289766</td>
          <td>26.408661</td>
          <td>0.136975</td>
          <td>26.147970</td>
          <td>0.097189</td>
          <td>25.834815</td>
          <td>0.120325</td>
          <td>25.500990</td>
          <td>0.169022</td>
          <td>25.255049</td>
          <td>0.296383</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.208436</td>
          <td>0.302590</td>
          <td>26.681478</td>
          <td>0.164082</td>
          <td>26.917957</td>
          <td>0.177820</td>
          <td>26.470304</td>
          <td>0.194624</td>
          <td>26.545248</td>
          <td>0.374966</td>
          <td>25.878089</td>
          <td>0.455031</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>30.466836</td>
          <td>3.113392</td>
          <td>27.371981</td>
          <td>0.299025</td>
          <td>27.128569</td>
          <td>0.218959</td>
          <td>26.109750</td>
          <td>0.147994</td>
          <td>26.197184</td>
          <td>0.292981</td>
          <td>25.089687</td>
          <td>0.251797</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.647083</td>
          <td>0.453181</td>
          <td>28.025746</td>
          <td>0.520431</td>
          <td>26.761504</td>
          <td>0.171195</td>
          <td>25.885248</td>
          <td>0.130435</td>
          <td>25.502973</td>
          <td>0.175293</td>
          <td>24.950069</td>
          <td>0.239201</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.984361</td>
          <td>0.554828</td>
          <td>26.837520</td>
          <td>0.190863</td>
          <td>25.990443</td>
          <td>0.081305</td>
          <td>25.618179</td>
          <td>0.095546</td>
          <td>25.081674</td>
          <td>0.113243</td>
          <td>25.249009</td>
          <td>0.283991</td>
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
