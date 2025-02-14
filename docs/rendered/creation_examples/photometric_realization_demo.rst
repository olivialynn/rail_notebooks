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

    <pzflow.flow.Flow at 0x7f37c0aeacb0>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>28.632982</td>
          <td>1.493356</td>
          <td>26.721637</td>
          <td>0.167450</td>
          <td>26.040923</td>
          <td>0.081746</td>
          <td>25.103502</td>
          <td>0.058195</td>
          <td>24.700776</td>
          <td>0.077970</td>
          <td>23.953016</td>
          <td>0.090691</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.775899</td>
          <td>0.924469</td>
          <td>27.055844</td>
          <td>0.221857</td>
          <td>26.672787</td>
          <td>0.141903</td>
          <td>26.439209</td>
          <td>0.186467</td>
          <td>26.823685</td>
          <td>0.457394</td>
          <td>25.323645</td>
          <td>0.290656</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.684271</td>
          <td>0.760485</td>
          <td>27.457475</td>
          <td>0.274207</td>
          <td>26.015395</td>
          <td>0.129735</td>
          <td>24.973044</td>
          <td>0.099078</td>
          <td>24.431008</td>
          <td>0.137566</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.119630</td>
          <td>0.512549</td>
          <td>27.285556</td>
          <td>0.238162</td>
          <td>26.292721</td>
          <td>0.164660</td>
          <td>25.580222</td>
          <td>0.167551</td>
          <td>25.250126</td>
          <td>0.273846</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.961319</td>
          <td>0.244938</td>
          <td>26.353716</td>
          <td>0.122035</td>
          <td>25.875795</td>
          <td>0.070648</td>
          <td>25.554083</td>
          <td>0.086696</td>
          <td>25.496804</td>
          <td>0.156031</td>
          <td>25.228122</td>
          <td>0.268984</td>
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
          <td>0.389450</td>
          <td>27.768803</td>
          <td>0.920409</td>
          <td>26.346913</td>
          <td>0.121317</td>
          <td>25.422916</td>
          <td>0.047273</td>
          <td>25.131378</td>
          <td>0.059653</td>
          <td>24.797920</td>
          <td>0.084946</td>
          <td>25.121112</td>
          <td>0.246411</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.477922</td>
          <td>0.763857</td>
          <td>26.562694</td>
          <td>0.146167</td>
          <td>26.082281</td>
          <td>0.084781</td>
          <td>25.192549</td>
          <td>0.062979</td>
          <td>24.899029</td>
          <td>0.092848</td>
          <td>24.304921</td>
          <td>0.123345</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.765585</td>
          <td>0.173825</td>
          <td>26.282290</td>
          <td>0.101068</td>
          <td>25.962922</td>
          <td>0.123967</td>
          <td>25.788164</td>
          <td>0.199781</td>
          <td>25.535519</td>
          <td>0.344216</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.705563</td>
          <td>0.441366</td>
          <td>26.216488</td>
          <td>0.108301</td>
          <td>25.898855</td>
          <td>0.072104</td>
          <td>25.770108</td>
          <td>0.104796</td>
          <td>25.901882</td>
          <td>0.219717</td>
          <td>25.375220</td>
          <td>0.302981</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.925896</td>
          <td>1.012922</td>
          <td>26.538635</td>
          <td>0.143175</td>
          <td>26.491636</td>
          <td>0.121320</td>
          <td>26.244106</td>
          <td>0.157962</td>
          <td>26.237014</td>
          <td>0.289279</td>
          <td>25.251118</td>
          <td>0.274067</td>
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
          <td>1.398945</td>
          <td>30.278712</td>
          <td>3.025042</td>
          <td>26.949649</td>
          <td>0.232500</td>
          <td>25.940306</td>
          <td>0.087985</td>
          <td>25.247233</td>
          <td>0.078344</td>
          <td>24.665249</td>
          <td>0.088866</td>
          <td>24.043200</td>
          <td>0.115905</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.831383</td>
          <td>0.534992</td>
          <td>28.151060</td>
          <td>0.589698</td>
          <td>26.491903</td>
          <td>0.142303</td>
          <td>26.710249</td>
          <td>0.273804</td>
          <td>25.625353</td>
          <td>0.203347</td>
          <td>24.984470</td>
          <td>0.257484</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.487934</td>
          <td>1.499131</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.175763</td>
          <td>0.258444</td>
          <td>26.424285</td>
          <td>0.221070</td>
          <td>25.052157</td>
          <td>0.127382</td>
          <td>24.456853</td>
          <td>0.169270</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.547794</td>
          <td>0.396820</td>
          <td>26.925479</td>
          <td>0.219228</td>
          <td>26.768338</td>
          <td>0.305779</td>
          <td>25.554985</td>
          <td>0.204330</td>
          <td>25.060899</td>
          <td>0.291959</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.998906</td>
          <td>0.281729</td>
          <td>26.460799</td>
          <td>0.154048</td>
          <td>25.890957</td>
          <td>0.084272</td>
          <td>25.764076</td>
          <td>0.123249</td>
          <td>25.421456</td>
          <td>0.171198</td>
          <td>24.884633</td>
          <td>0.237208</td>
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
          <td>0.389450</td>
          <td>26.216531</td>
          <td>0.340103</td>
          <td>26.430160</td>
          <td>0.152818</td>
          <td>25.452136</td>
          <td>0.058371</td>
          <td>25.050867</td>
          <td>0.067317</td>
          <td>24.653978</td>
          <td>0.089860</td>
          <td>24.517094</td>
          <td>0.177927</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.848770</td>
          <td>0.543129</td>
          <td>26.598154</td>
          <td>0.173767</td>
          <td>26.080676</td>
          <td>0.099939</td>
          <td>25.093373</td>
          <td>0.068680</td>
          <td>24.916836</td>
          <td>0.111235</td>
          <td>24.432036</td>
          <td>0.162761</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.508877</td>
          <td>0.162275</td>
          <td>26.211180</td>
          <td>0.112974</td>
          <td>26.377197</td>
          <td>0.210581</td>
          <td>26.105677</td>
          <td>0.305257</td>
          <td>25.307755</td>
          <td>0.338039</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.375427</td>
          <td>0.387802</td>
          <td>26.327654</td>
          <td>0.141244</td>
          <td>26.309370</td>
          <td>0.125330</td>
          <td>26.115710</td>
          <td>0.172064</td>
          <td>26.100734</td>
          <td>0.309255</td>
          <td>25.280963</td>
          <td>0.336695</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.990251</td>
          <td>0.603133</td>
          <td>26.915867</td>
          <td>0.228023</td>
          <td>26.467080</td>
          <td>0.140642</td>
          <td>26.813571</td>
          <td>0.300462</td>
          <td>25.645490</td>
          <td>0.208752</td>
          <td>25.521996</td>
          <td>0.398588</td>
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
          <td>1.398945</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.983728</td>
          <td>0.208931</td>
          <td>25.957462</td>
          <td>0.075949</td>
          <td>25.143070</td>
          <td>0.060283</td>
          <td>24.758156</td>
          <td>0.082031</td>
          <td>23.830032</td>
          <td>0.081393</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.238814</td>
          <td>0.258214</td>
          <td>26.632294</td>
          <td>0.137162</td>
          <td>26.392348</td>
          <td>0.179389</td>
          <td>25.679920</td>
          <td>0.182520</td>
          <td>25.615154</td>
          <td>0.366742</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.508802</td>
          <td>0.814143</td>
          <td>30.013030</td>
          <td>1.684609</td>
          <td>27.863542</td>
          <td>0.407047</td>
          <td>26.310703</td>
          <td>0.181614</td>
          <td>24.943655</td>
          <td>0.104763</td>
          <td>24.268109</td>
          <td>0.129892</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>31.277460</td>
          <td>4.037049</td>
          <td>28.399262</td>
          <td>0.732184</td>
          <td>27.013300</td>
          <td>0.235009</td>
          <td>25.985111</td>
          <td>0.158974</td>
          <td>26.207181</td>
          <td>0.346458</td>
          <td>24.557042</td>
          <td>0.191932</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.244532</td>
          <td>0.652801</td>
          <td>26.123053</td>
          <td>0.099933</td>
          <td>25.962765</td>
          <td>0.076405</td>
          <td>25.581069</td>
          <td>0.088913</td>
          <td>25.402273</td>
          <td>0.144071</td>
          <td>24.865067</td>
          <td>0.199418</td>
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
          <td>0.389450</td>
          <td>26.828573</td>
          <td>0.507226</td>
          <td>26.309532</td>
          <td>0.125730</td>
          <td>25.335811</td>
          <td>0.047390</td>
          <td>25.165233</td>
          <td>0.066808</td>
          <td>25.155040</td>
          <td>0.125547</td>
          <td>24.845867</td>
          <td>0.211809</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.007331</td>
          <td>0.556833</td>
          <td>26.879579</td>
          <td>0.194066</td>
          <td>26.129210</td>
          <td>0.089829</td>
          <td>25.137033</td>
          <td>0.061011</td>
          <td>24.822280</td>
          <td>0.088232</td>
          <td>24.173873</td>
          <td>0.111936</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.411380</td>
          <td>0.362789</td>
          <td>26.893121</td>
          <td>0.201691</td>
          <td>26.444457</td>
          <td>0.122220</td>
          <td>26.344578</td>
          <td>0.180824</td>
          <td>25.710305</td>
          <td>0.196078</td>
          <td>25.520133</td>
          <td>0.355871</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.009275</td>
          <td>0.274971</td>
          <td>26.153946</td>
          <td>0.113367</td>
          <td>26.197367</td>
          <td>0.105190</td>
          <td>25.728117</td>
          <td>0.113799</td>
          <td>25.547185</td>
          <td>0.181986</td>
          <td>24.761894</td>
          <td>0.204536</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.438139</td>
          <td>0.759437</td>
          <td>26.762798</td>
          <td>0.179186</td>
          <td>26.359430</td>
          <td>0.112385</td>
          <td>26.459697</td>
          <td>0.197285</td>
          <td>25.911438</td>
          <td>0.229758</td>
          <td>25.399566</td>
          <td>0.320505</td>
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
