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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f0794fe78b0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.793640</td>
          <td>0.178011</td>
          <td>25.910182</td>
          <td>0.072830</td>
          <td>25.275159</td>
          <td>0.067763</td>
          <td>24.720535</td>
          <td>0.079342</td>
          <td>24.053292</td>
          <td>0.099035</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.937019</td>
          <td>0.524181</td>
          <td>28.690189</td>
          <td>0.763468</td>
          <td>26.693122</td>
          <td>0.144409</td>
          <td>26.515847</td>
          <td>0.198906</td>
          <td>26.046936</td>
          <td>0.247747</td>
          <td>25.330467</td>
          <td>0.292261</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.803181</td>
          <td>0.361387</td>
          <td>25.973342</td>
          <td>0.125092</td>
          <td>25.015241</td>
          <td>0.102808</td>
          <td>24.295419</td>
          <td>0.122332</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.557511</td>
          <td>0.804766</td>
          <td>27.256596</td>
          <td>0.261796</td>
          <td>27.757466</td>
          <td>0.348646</td>
          <td>26.110091</td>
          <td>0.140793</td>
          <td>25.501161</td>
          <td>0.156614</td>
          <td>25.442719</td>
          <td>0.319794</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.703237</td>
          <td>0.197669</td>
          <td>26.129586</td>
          <td>0.100383</td>
          <td>25.878741</td>
          <td>0.070832</td>
          <td>25.561626</td>
          <td>0.087274</td>
          <td>25.423818</td>
          <td>0.146562</td>
          <td>25.145644</td>
          <td>0.251431</td>
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
          <td>26.747279</td>
          <td>0.455461</td>
          <td>26.226737</td>
          <td>0.109274</td>
          <td>25.401546</td>
          <td>0.046385</td>
          <td>25.021984</td>
          <td>0.054133</td>
          <td>24.832046</td>
          <td>0.087537</td>
          <td>24.652141</td>
          <td>0.166296</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.561466</td>
          <td>1.440376</td>
          <td>27.096886</td>
          <td>0.229547</td>
          <td>26.092945</td>
          <td>0.085581</td>
          <td>25.214753</td>
          <td>0.064231</td>
          <td>24.705679</td>
          <td>0.078309</td>
          <td>24.253464</td>
          <td>0.117952</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.672810</td>
          <td>0.866609</td>
          <td>26.598737</td>
          <td>0.150759</td>
          <td>26.431787</td>
          <td>0.115165</td>
          <td>26.057015</td>
          <td>0.134490</td>
          <td>26.093077</td>
          <td>0.257311</td>
          <td>25.913739</td>
          <td>0.460599</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.009234</td>
          <td>1.064213</td>
          <td>25.961850</td>
          <td>0.086648</td>
          <td>26.186706</td>
          <td>0.092940</td>
          <td>25.782984</td>
          <td>0.105982</td>
          <td>25.471396</td>
          <td>0.152671</td>
          <td>25.190303</td>
          <td>0.260805</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.163191</td>
          <td>0.616343</td>
          <td>26.745679</td>
          <td>0.170910</td>
          <td>26.793394</td>
          <td>0.157383</td>
          <td>26.527738</td>
          <td>0.200903</td>
          <td>25.949832</td>
          <td>0.228650</td>
          <td>25.509517</td>
          <td>0.337217</td>
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
          <td>27.975901</td>
          <td>1.127781</td>
          <td>26.523821</td>
          <td>0.162524</td>
          <td>25.995269</td>
          <td>0.092340</td>
          <td>25.177803</td>
          <td>0.073684</td>
          <td>24.687261</td>
          <td>0.090603</td>
          <td>24.238524</td>
          <td>0.137282</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.813646</td>
          <td>0.460876</td>
          <td>26.621981</td>
          <td>0.159103</td>
          <td>26.262747</td>
          <td>0.188913</td>
          <td>25.549644</td>
          <td>0.190805</td>
          <td>25.194837</td>
          <td>0.305365</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.499942</td>
          <td>0.761034</td>
          <td>29.685912</td>
          <td>1.423527</td>
          <td>25.760329</td>
          <td>0.125638</td>
          <td>24.840096</td>
          <td>0.105919</td>
          <td>24.104743</td>
          <td>0.125072</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.641774</td>
          <td>0.859881</td>
          <td>27.136474</td>
          <td>0.260936</td>
          <td>26.514068</td>
          <td>0.248718</td>
          <td>25.511140</td>
          <td>0.196946</td>
          <td>25.641797</td>
          <td>0.459309</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.747734</td>
          <td>0.503286</td>
          <td>25.985533</td>
          <td>0.102089</td>
          <td>25.867067</td>
          <td>0.082516</td>
          <td>25.947165</td>
          <td>0.144375</td>
          <td>25.328754</td>
          <td>0.158186</td>
          <td>25.121689</td>
          <td>0.287931</td>
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
          <td>27.151425</td>
          <td>0.679122</td>
          <td>26.253495</td>
          <td>0.131273</td>
          <td>25.458087</td>
          <td>0.058680</td>
          <td>25.044843</td>
          <td>0.066959</td>
          <td>24.841116</td>
          <td>0.105876</td>
          <td>24.485461</td>
          <td>0.173214</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.015153</td>
          <td>1.155652</td>
          <td>26.410114</td>
          <td>0.148001</td>
          <td>25.947662</td>
          <td>0.088927</td>
          <td>25.290741</td>
          <td>0.081763</td>
          <td>24.875029</td>
          <td>0.107250</td>
          <td>24.417787</td>
          <td>0.160793</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.920052</td>
          <td>1.099077</td>
          <td>26.860886</td>
          <td>0.218353</td>
          <td>26.378788</td>
          <td>0.130673</td>
          <td>26.217081</td>
          <td>0.184054</td>
          <td>25.792371</td>
          <td>0.236488</td>
          <td>25.071105</td>
          <td>0.279656</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.203182</td>
          <td>0.707825</td>
          <td>26.107034</td>
          <td>0.116709</td>
          <td>26.054337</td>
          <td>0.100352</td>
          <td>25.818609</td>
          <td>0.133372</td>
          <td>25.404590</td>
          <td>0.173910</td>
          <td>25.753986</td>
          <td>0.484148</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.885688</td>
          <td>1.075833</td>
          <td>27.117246</td>
          <td>0.269072</td>
          <td>26.503786</td>
          <td>0.145156</td>
          <td>26.172868</td>
          <td>0.176804</td>
          <td>26.193578</td>
          <td>0.326636</td>
          <td>26.462185</td>
          <td>0.781544</td>
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
          <td>26.798896</td>
          <td>0.178825</td>
          <td>25.921430</td>
          <td>0.073568</td>
          <td>25.119718</td>
          <td>0.059047</td>
          <td>24.724202</td>
          <td>0.079610</td>
          <td>24.036396</td>
          <td>0.097592</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.253330</td>
          <td>0.261298</td>
          <td>26.689580</td>
          <td>0.144103</td>
          <td>26.352280</td>
          <td>0.173391</td>
          <td>25.731494</td>
          <td>0.190648</td>
          <td>25.120450</td>
          <td>0.246504</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.262690</td>
          <td>1.142230</td>
          <td>28.894499</td>
          <td>0.843750</td>
          <td>25.742526</td>
          <td>0.111385</td>
          <td>25.124744</td>
          <td>0.122669</td>
          <td>24.257593</td>
          <td>0.128715</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.452413</td>
          <td>1.499737</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.125492</td>
          <td>0.257743</td>
          <td>26.479975</td>
          <td>0.240999</td>
          <td>25.284269</td>
          <td>0.161942</td>
          <td>25.663498</td>
          <td>0.465388</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.080306</td>
          <td>0.270218</td>
          <td>26.028288</td>
          <td>0.091969</td>
          <td>26.092714</td>
          <td>0.085686</td>
          <td>25.576741</td>
          <td>0.088575</td>
          <td>25.229145</td>
          <td>0.124051</td>
          <td>25.437830</td>
          <td>0.318977</td>
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
          <td>27.107053</td>
          <td>0.619550</td>
          <td>26.570399</td>
          <td>0.157380</td>
          <td>25.531983</td>
          <td>0.056403</td>
          <td>25.031272</td>
          <td>0.059327</td>
          <td>24.768289</td>
          <td>0.089550</td>
          <td>24.719991</td>
          <td>0.190569</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.276750</td>
          <td>1.247729</td>
          <td>26.635999</td>
          <td>0.157834</td>
          <td>26.068688</td>
          <td>0.085169</td>
          <td>25.237387</td>
          <td>0.066687</td>
          <td>24.818002</td>
          <td>0.087900</td>
          <td>24.262102</td>
          <td>0.120871</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.293836</td>
          <td>0.693119</td>
          <td>26.783359</td>
          <td>0.183884</td>
          <td>26.454101</td>
          <td>0.123248</td>
          <td>26.182997</td>
          <td>0.157585</td>
          <td>25.812598</td>
          <td>0.213631</td>
          <td>25.555322</td>
          <td>0.365814</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.091844</td>
          <td>0.293947</td>
          <td>26.295963</td>
          <td>0.128236</td>
          <td>26.086618</td>
          <td>0.095465</td>
          <td>26.027138</td>
          <td>0.147411</td>
          <td>25.594027</td>
          <td>0.189335</td>
          <td>25.808882</td>
          <td>0.471096</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.072120</td>
          <td>0.590761</td>
          <td>26.529858</td>
          <td>0.146896</td>
          <td>26.585561</td>
          <td>0.136747</td>
          <td>26.152333</td>
          <td>0.151945</td>
          <td>25.843799</td>
          <td>0.217194</td>
          <td>26.368361</td>
          <td>0.660720</td>
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
