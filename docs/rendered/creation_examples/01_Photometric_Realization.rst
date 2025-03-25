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

    <pzflow.flow.Flow at 0x7fde3a513310>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>26.498204</td>
          <td>0.376502</td>
          <td>26.912039</td>
          <td>0.196721</td>
          <td>26.013737</td>
          <td>0.079808</td>
          <td>25.231049</td>
          <td>0.065166</td>
          <td>24.655026</td>
          <td>0.074882</td>
          <td>23.978586</td>
          <td>0.092752</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.701419</td>
          <td>0.373482</td>
          <td>26.771142</td>
          <td>0.154413</td>
          <td>26.382154</td>
          <td>0.177674</td>
          <td>25.662937</td>
          <td>0.179750</td>
          <td>25.970970</td>
          <td>0.480719</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.354473</td>
          <td>1.292309</td>
          <td>27.815131</td>
          <td>0.407796</td>
          <td>28.456403</td>
          <td>0.589209</td>
          <td>26.182248</td>
          <td>0.149807</td>
          <td>24.883919</td>
          <td>0.091623</td>
          <td>24.288685</td>
          <td>0.121618</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.647370</td>
          <td>1.504123</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.349230</td>
          <td>0.250989</td>
          <td>26.243648</td>
          <td>0.157900</td>
          <td>25.467802</td>
          <td>0.152202</td>
          <td>25.278266</td>
          <td>0.280177</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.160924</td>
          <td>0.288194</td>
          <td>26.124075</td>
          <td>0.099900</td>
          <td>25.866314</td>
          <td>0.070058</td>
          <td>25.728767</td>
          <td>0.101072</td>
          <td>25.644009</td>
          <td>0.176888</td>
          <td>24.809818</td>
          <td>0.190087</td>
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
          <td>27.126318</td>
          <td>0.600539</td>
          <td>26.452481</td>
          <td>0.132928</td>
          <td>25.459024</td>
          <td>0.048813</td>
          <td>25.060896</td>
          <td>0.056036</td>
          <td>24.768915</td>
          <td>0.082802</td>
          <td>24.527895</td>
          <td>0.149529</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.676947</td>
          <td>0.868884</td>
          <td>26.517420</td>
          <td>0.140585</td>
          <td>26.105039</td>
          <td>0.086498</td>
          <td>25.177656</td>
          <td>0.062153</td>
          <td>24.782249</td>
          <td>0.083781</td>
          <td>24.186185</td>
          <td>0.111238</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.545463</td>
          <td>0.798480</td>
          <td>27.211640</td>
          <td>0.252333</td>
          <td>26.322788</td>
          <td>0.104715</td>
          <td>26.246101</td>
          <td>0.158232</td>
          <td>25.669321</td>
          <td>0.180725</td>
          <td>25.781160</td>
          <td>0.416591</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.443641</td>
          <td>0.360822</td>
          <td>26.159619</td>
          <td>0.103054</td>
          <td>26.058635</td>
          <td>0.083033</td>
          <td>25.889896</td>
          <td>0.116343</td>
          <td>25.702607</td>
          <td>0.185886</td>
          <td>24.989323</td>
          <td>0.220945</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.657171</td>
          <td>0.858042</td>
          <td>26.766792</td>
          <td>0.174003</td>
          <td>26.878163</td>
          <td>0.169192</td>
          <td>26.221571</td>
          <td>0.154944</td>
          <td>25.751786</td>
          <td>0.193761</td>
          <td>25.992554</td>
          <td>0.488487</td>
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.946031</td>
          <td>0.231805</td>
          <td>26.015498</td>
          <td>0.093995</td>
          <td>25.107352</td>
          <td>0.069233</td>
          <td>24.572574</td>
          <td>0.081903</td>
          <td>24.118074</td>
          <td>0.123696</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.865292</td>
          <td>0.216809</td>
          <td>26.718481</td>
          <td>0.172742</td>
          <td>26.236352</td>
          <td>0.184748</td>
          <td>26.418190</td>
          <td>0.386200</td>
          <td>25.437787</td>
          <td>0.370088</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>33.805376</td>
          <td>5.329930</td>
          <td>29.626158</td>
          <td>1.380208</td>
          <td>26.276825</td>
          <td>0.195404</td>
          <td>24.925148</td>
          <td>0.114076</td>
          <td>24.401533</td>
          <td>0.161473</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.007064</td>
          <td>1.941936</td>
          <td>28.326351</td>
          <td>0.698803</td>
          <td>27.311517</td>
          <td>0.300729</td>
          <td>26.458613</td>
          <td>0.237607</td>
          <td>25.271935</td>
          <td>0.160802</td>
          <td>25.778760</td>
          <td>0.508503</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.158761</td>
          <td>0.320292</td>
          <td>25.934195</td>
          <td>0.097606</td>
          <td>26.035507</td>
          <td>0.095691</td>
          <td>25.797939</td>
          <td>0.126921</td>
          <td>25.286587</td>
          <td>0.152578</td>
          <td>24.945399</td>
          <td>0.249389</td>
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
          <td>26.926667</td>
          <td>0.580450</td>
          <td>26.549744</td>
          <td>0.169237</td>
          <td>25.482683</td>
          <td>0.059974</td>
          <td>25.022789</td>
          <td>0.065664</td>
          <td>24.755318</td>
          <td>0.098218</td>
          <td>24.923556</td>
          <td>0.249865</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>32.083968</td>
          <td>4.778164</td>
          <td>26.688969</td>
          <td>0.187648</td>
          <td>25.905129</td>
          <td>0.085660</td>
          <td>25.193992</td>
          <td>0.075071</td>
          <td>25.023756</td>
          <td>0.122079</td>
          <td>24.429596</td>
          <td>0.162423</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.903369</td>
          <td>0.567959</td>
          <td>26.584997</td>
          <td>0.173134</td>
          <td>26.590911</td>
          <td>0.156840</td>
          <td>26.084076</td>
          <td>0.164394</td>
          <td>25.755512</td>
          <td>0.229381</td>
          <td>25.412158</td>
          <td>0.366946</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.684328</td>
          <td>0.489966</td>
          <td>26.099825</td>
          <td>0.115980</td>
          <td>25.940479</td>
          <td>0.090813</td>
          <td>25.907307</td>
          <td>0.143972</td>
          <td>25.644127</td>
          <td>0.212794</td>
          <td>25.241290</td>
          <td>0.326266</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.807096</td>
          <td>0.528890</td>
          <td>26.679711</td>
          <td>0.187138</td>
          <td>26.383224</td>
          <td>0.130820</td>
          <td>26.279047</td>
          <td>0.193406</td>
          <td>25.702524</td>
          <td>0.218930</td>
          <td>25.711593</td>
          <td>0.460406</td>
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.924897</td>
          <td>0.198880</td>
          <td>25.881067</td>
          <td>0.070988</td>
          <td>25.142433</td>
          <td>0.060249</td>
          <td>24.653898</td>
          <td>0.074817</td>
          <td>23.994538</td>
          <td>0.094073</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.428427</td>
          <td>0.739520</td>
          <td>27.098846</td>
          <td>0.230098</td>
          <td>26.566617</td>
          <td>0.129592</td>
          <td>25.993607</td>
          <td>0.127435</td>
          <td>25.883997</td>
          <td>0.216661</td>
          <td>25.388198</td>
          <td>0.306428</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.375300</td>
          <td>1.216813</td>
          <td>27.897764</td>
          <td>0.417856</td>
          <td>26.076654</td>
          <td>0.148751</td>
          <td>25.347500</td>
          <td>0.148686</td>
          <td>24.626313</td>
          <td>0.176588</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.603962</td>
          <td>0.934366</td>
          <td>28.893028</td>
          <td>1.002189</td>
          <td>27.121094</td>
          <td>0.256816</td>
          <td>26.113174</td>
          <td>0.177289</td>
          <td>25.490041</td>
          <td>0.192820</td>
          <td>25.878492</td>
          <td>0.545218</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.486851</td>
          <td>0.373518</td>
          <td>26.097391</td>
          <td>0.097713</td>
          <td>25.901862</td>
          <td>0.072400</td>
          <td>25.481699</td>
          <td>0.081460</td>
          <td>25.436658</td>
          <td>0.148393</td>
          <td>24.959414</td>
          <td>0.215808</td>
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
          <td>26.790387</td>
          <td>0.493148</td>
          <td>26.070580</td>
          <td>0.102121</td>
          <td>25.458252</td>
          <td>0.052831</td>
          <td>25.112781</td>
          <td>0.063774</td>
          <td>24.775949</td>
          <td>0.090155</td>
          <td>24.566835</td>
          <td>0.167368</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.024559</td>
          <td>1.082021</td>
          <td>26.826264</td>
          <td>0.185535</td>
          <td>25.972979</td>
          <td>0.078275</td>
          <td>25.141662</td>
          <td>0.061262</td>
          <td>24.883251</td>
          <td>0.093090</td>
          <td>24.125227</td>
          <td>0.107283</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.256320</td>
          <td>0.321027</td>
          <td>26.548337</td>
          <td>0.150534</td>
          <td>26.542622</td>
          <td>0.133070</td>
          <td>26.258455</td>
          <td>0.168070</td>
          <td>25.601320</td>
          <td>0.178837</td>
          <td>25.604477</td>
          <td>0.380094</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.469952</td>
          <td>0.396011</td>
          <td>26.027042</td>
          <td>0.101486</td>
          <td>25.962795</td>
          <td>0.085620</td>
          <td>25.844038</td>
          <td>0.125862</td>
          <td>25.556478</td>
          <td>0.183423</td>
          <td>24.907197</td>
          <td>0.230868</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.416804</td>
          <td>1.358073</td>
          <td>26.712098</td>
          <td>0.171643</td>
          <td>26.504718</td>
          <td>0.127513</td>
          <td>26.155679</td>
          <td>0.152381</td>
          <td>26.157424</td>
          <td>0.281115</td>
          <td>25.185220</td>
          <td>0.269652</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
