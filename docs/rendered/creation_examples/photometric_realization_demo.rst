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

    <pzflow.flow.Flow at 0x7f8bacbe39a0>



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
          <td>29.683122</td>
          <td>2.358374</td>
          <td>26.847622</td>
          <td>0.186329</td>
          <td>26.200433</td>
          <td>0.094067</td>
          <td>25.112573</td>
          <td>0.058666</td>
          <td>24.711747</td>
          <td>0.078729</td>
          <td>23.900492</td>
          <td>0.086595</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.088760</td>
          <td>0.228006</td>
          <td>26.528144</td>
          <td>0.125226</td>
          <td>26.149586</td>
          <td>0.145662</td>
          <td>26.100809</td>
          <td>0.258946</td>
          <td>25.330882</td>
          <td>0.292359</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.838254</td>
          <td>2.496900</td>
          <td>27.659489</td>
          <td>0.361454</td>
          <td>27.998570</td>
          <td>0.420330</td>
          <td>26.049815</td>
          <td>0.133656</td>
          <td>25.300549</td>
          <td>0.131783</td>
          <td>24.260757</td>
          <td>0.118702</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.114853</td>
          <td>0.510755</td>
          <td>27.373365</td>
          <td>0.256009</td>
          <td>26.083164</td>
          <td>0.137561</td>
          <td>25.801353</td>
          <td>0.202006</td>
          <td>25.308927</td>
          <td>0.287220</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.651002</td>
          <td>0.423469</td>
          <td>26.321826</td>
          <td>0.118703</td>
          <td>25.965118</td>
          <td>0.076455</td>
          <td>25.534687</td>
          <td>0.085228</td>
          <td>25.459900</td>
          <td>0.151173</td>
          <td>25.255112</td>
          <td>0.274959</td>
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
          <td>27.552081</td>
          <td>0.801928</td>
          <td>26.748769</td>
          <td>0.171360</td>
          <td>25.438460</td>
          <td>0.047930</td>
          <td>25.071812</td>
          <td>0.056581</td>
          <td>24.723654</td>
          <td>0.079561</td>
          <td>24.805452</td>
          <td>0.189388</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.418929</td>
          <td>0.734476</td>
          <td>26.935608</td>
          <td>0.200654</td>
          <td>26.070589</td>
          <td>0.083912</td>
          <td>25.244330</td>
          <td>0.065937</td>
          <td>24.752866</td>
          <td>0.081638</td>
          <td>24.448559</td>
          <td>0.139664</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.378200</td>
          <td>0.342747</td>
          <td>26.599798</td>
          <td>0.150896</td>
          <td>26.561061</td>
          <td>0.128850</td>
          <td>25.985772</td>
          <td>0.126448</td>
          <td>26.113269</td>
          <td>0.261599</td>
          <td>25.822909</td>
          <td>0.430063</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.228088</td>
          <td>0.304191</td>
          <td>26.284701</td>
          <td>0.114932</td>
          <td>26.162801</td>
          <td>0.091008</td>
          <td>25.979957</td>
          <td>0.125812</td>
          <td>25.696551</td>
          <td>0.184937</td>
          <td>25.418188</td>
          <td>0.313593</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.018376</td>
          <td>0.556026</td>
          <td>26.616875</td>
          <td>0.153121</td>
          <td>26.356739</td>
          <td>0.107868</td>
          <td>26.161740</td>
          <td>0.147191</td>
          <td>26.323869</td>
          <td>0.310208</td>
          <td>25.403705</td>
          <td>0.309980</td>
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
          <td>26.779242</td>
          <td>0.514961</td>
          <td>26.897902</td>
          <td>0.222733</td>
          <td>25.921775</td>
          <td>0.086562</td>
          <td>25.105277</td>
          <td>0.069106</td>
          <td>24.560962</td>
          <td>0.081069</td>
          <td>24.022501</td>
          <td>0.113835</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.380240</td>
          <td>0.329759</td>
          <td>26.716027</td>
          <td>0.172382</td>
          <td>25.935359</td>
          <td>0.142898</td>
          <td>25.757602</td>
          <td>0.227069</td>
          <td>26.037423</td>
          <td>0.579557</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.650253</td>
          <td>0.776352</td>
          <td>25.985949</td>
          <td>0.152618</td>
          <td>24.937414</td>
          <td>0.115301</td>
          <td>24.517787</td>
          <td>0.178261</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.171044</td>
          <td>1.298481</td>
          <td>27.343502</td>
          <td>0.338317</td>
          <td>27.483193</td>
          <td>0.344779</td>
          <td>26.333416</td>
          <td>0.214144</td>
          <td>25.464243</td>
          <td>0.189319</td>
          <td>25.522405</td>
          <td>0.419613</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.939421</td>
          <td>0.268455</td>
          <td>25.978567</td>
          <td>0.101470</td>
          <td>25.989697</td>
          <td>0.091919</td>
          <td>25.880101</td>
          <td>0.136268</td>
          <td>25.256031</td>
          <td>0.148630</td>
          <td>25.177003</td>
          <td>0.301057</td>
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
          <td>27.066731</td>
          <td>0.640594</td>
          <td>26.366447</td>
          <td>0.144691</td>
          <td>25.340384</td>
          <td>0.052862</td>
          <td>25.131429</td>
          <td>0.072290</td>
          <td>24.709630</td>
          <td>0.094361</td>
          <td>24.799105</td>
          <td>0.225451</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.726636</td>
          <td>0.193700</td>
          <td>25.943077</td>
          <td>0.088569</td>
          <td>25.376957</td>
          <td>0.088213</td>
          <td>24.779538</td>
          <td>0.098655</td>
          <td>24.187186</td>
          <td>0.131883</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.380977</td>
          <td>0.788134</td>
          <td>26.922800</td>
          <td>0.229875</td>
          <td>26.681350</td>
          <td>0.169423</td>
          <td>26.869941</td>
          <td>0.315160</td>
          <td>25.852307</td>
          <td>0.248466</td>
          <td>25.333666</td>
          <td>0.345026</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.329751</td>
          <td>0.374309</td>
          <td>26.185044</td>
          <td>0.124882</td>
          <td>26.078870</td>
          <td>0.102530</td>
          <td>25.869603</td>
          <td>0.139373</td>
          <td>25.552454</td>
          <td>0.197059</td>
          <td>25.202553</td>
          <td>0.316353</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.666154</td>
          <td>0.476752</td>
          <td>26.679266</td>
          <td>0.187068</td>
          <td>26.571438</td>
          <td>0.153834</td>
          <td>26.619982</td>
          <td>0.256770</td>
          <td>25.915337</td>
          <td>0.260977</td>
          <td>25.083637</td>
          <td>0.281777</td>
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
          <td>27.719894</td>
          <td>0.892794</td>
          <td>26.558766</td>
          <td>0.145691</td>
          <td>26.080838</td>
          <td>0.084685</td>
          <td>25.300989</td>
          <td>0.069340</td>
          <td>24.608942</td>
          <td>0.071901</td>
          <td>23.873281</td>
          <td>0.084556</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.059099</td>
          <td>0.572820</td>
          <td>27.648633</td>
          <td>0.358656</td>
          <td>26.704360</td>
          <td>0.145947</td>
          <td>26.222955</td>
          <td>0.155279</td>
          <td>25.743093</td>
          <td>0.192522</td>
          <td>25.389036</td>
          <td>0.306634</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.213888</td>
          <td>1.110727</td>
          <td>28.013483</td>
          <td>0.456161</td>
          <td>25.745596</td>
          <td>0.111684</td>
          <td>25.104783</td>
          <td>0.120560</td>
          <td>24.562240</td>
          <td>0.167226</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.434050</td>
          <td>0.414642</td>
          <td>27.890235</td>
          <td>0.512065</td>
          <td>27.264449</td>
          <td>0.288590</td>
          <td>26.366485</td>
          <td>0.219362</td>
          <td>25.711704</td>
          <td>0.232045</td>
          <td>25.607343</td>
          <td>0.446150</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.263553</td>
          <td>0.661430</td>
          <td>26.101504</td>
          <td>0.098066</td>
          <td>25.932948</td>
          <td>0.074418</td>
          <td>25.683858</td>
          <td>0.097316</td>
          <td>25.158282</td>
          <td>0.116643</td>
          <td>25.237845</td>
          <td>0.271494</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.180360</td>
          <td>0.112387</td>
          <td>25.442568</td>
          <td>0.052100</td>
          <td>25.070922</td>
          <td>0.061451</td>
          <td>24.820185</td>
          <td>0.093728</td>
          <td>24.876635</td>
          <td>0.217318</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.929463</td>
          <td>0.526290</td>
          <td>27.018088</td>
          <td>0.217931</td>
          <td>26.138781</td>
          <td>0.090588</td>
          <td>25.251797</td>
          <td>0.067544</td>
          <td>24.786122</td>
          <td>0.085467</td>
          <td>24.332858</td>
          <td>0.128522</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.353688</td>
          <td>0.346740</td>
          <td>26.737588</td>
          <td>0.176896</td>
          <td>26.358974</td>
          <td>0.113462</td>
          <td>26.269116</td>
          <td>0.169602</td>
          <td>25.873899</td>
          <td>0.224821</td>
          <td>26.246200</td>
          <td>0.611799</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.945222</td>
          <td>0.261003</td>
          <td>26.544777</td>
          <td>0.158830</td>
          <td>26.298544</td>
          <td>0.114898</td>
          <td>25.851130</td>
          <td>0.126638</td>
          <td>25.872393</td>
          <td>0.238879</td>
          <td>25.852375</td>
          <td>0.486599</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.781438</td>
          <td>0.478209</td>
          <td>26.981192</td>
          <td>0.215295</td>
          <td>26.635416</td>
          <td>0.142752</td>
          <td>26.154017</td>
          <td>0.152164</td>
          <td>26.070857</td>
          <td>0.261985</td>
          <td>26.726752</td>
          <td>0.838800</td>
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
