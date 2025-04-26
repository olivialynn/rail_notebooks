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

    <pzflow.flow.Flow at 0x7f26d6f7ff40>



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
          <td>27.003332</td>
          <td>0.550028</td>
          <td>27.094965</td>
          <td>0.229182</td>
          <td>25.834738</td>
          <td>0.068126</td>
          <td>25.143562</td>
          <td>0.060301</td>
          <td>24.706704</td>
          <td>0.078380</td>
          <td>24.107751</td>
          <td>0.103872</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.876340</td>
          <td>0.501369</td>
          <td>26.808974</td>
          <td>0.180338</td>
          <td>26.425625</td>
          <td>0.114549</td>
          <td>26.299307</td>
          <td>0.165587</td>
          <td>26.126258</td>
          <td>0.264390</td>
          <td>24.994356</td>
          <td>0.221872</td>
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
          <td>28.739931</td>
          <td>0.717032</td>
          <td>26.002760</td>
          <td>0.128323</td>
          <td>25.001905</td>
          <td>0.101614</td>
          <td>24.225724</td>
          <td>0.115138</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.004888</td>
          <td>1.782554</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.132279</td>
          <td>0.209668</td>
          <td>26.470034</td>
          <td>0.191382</td>
          <td>25.560857</td>
          <td>0.164809</td>
          <td>25.811647</td>
          <td>0.426394</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.958314</td>
          <td>0.244334</td>
          <td>26.066919</td>
          <td>0.095022</td>
          <td>25.986269</td>
          <td>0.077896</td>
          <td>25.604203</td>
          <td>0.090605</td>
          <td>25.518006</td>
          <td>0.158887</td>
          <td>25.505871</td>
          <td>0.336246</td>
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
          <td>26.173658</td>
          <td>0.291170</td>
          <td>26.313864</td>
          <td>0.117884</td>
          <td>25.519130</td>
          <td>0.051489</td>
          <td>25.169116</td>
          <td>0.061684</td>
          <td>24.803357</td>
          <td>0.085353</td>
          <td>24.679578</td>
          <td>0.170227</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.489530</td>
          <td>0.373972</td>
          <td>26.337681</td>
          <td>0.120348</td>
          <td>25.996318</td>
          <td>0.078591</td>
          <td>25.205913</td>
          <td>0.063730</td>
          <td>24.787338</td>
          <td>0.084157</td>
          <td>24.279172</td>
          <td>0.120617</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.736195</td>
          <td>0.451681</td>
          <td>26.627951</td>
          <td>0.154580</td>
          <td>26.520687</td>
          <td>0.124418</td>
          <td>26.312689</td>
          <td>0.167487</td>
          <td>25.811294</td>
          <td>0.203698</td>
          <td>26.519188</td>
          <td>0.709789</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.211125</td>
          <td>0.300079</td>
          <td>26.548344</td>
          <td>0.144376</td>
          <td>26.031783</td>
          <td>0.081089</td>
          <td>25.818482</td>
          <td>0.109320</td>
          <td>25.645348</td>
          <td>0.177089</td>
          <td>25.516800</td>
          <td>0.339165</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.600817</td>
          <td>0.827637</td>
          <td>27.188585</td>
          <td>0.247599</td>
          <td>26.674497</td>
          <td>0.142112</td>
          <td>26.158836</td>
          <td>0.146824</td>
          <td>25.769300</td>
          <td>0.196639</td>
          <td>25.530971</td>
          <td>0.342983</td>
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
          <td>29.274512</td>
          <td>2.118274</td>
          <td>26.626131</td>
          <td>0.177294</td>
          <td>26.198279</td>
          <td>0.110301</td>
          <td>25.272600</td>
          <td>0.080117</td>
          <td>24.850904</td>
          <td>0.104578</td>
          <td>23.848044</td>
          <td>0.097739</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.255665</td>
          <td>0.298522</td>
          <td>26.632024</td>
          <td>0.160474</td>
          <td>26.026757</td>
          <td>0.154564</td>
          <td>25.518136</td>
          <td>0.185798</td>
          <td>25.660107</td>
          <td>0.439063</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.251906</td>
          <td>2.115511</td>
          <td>28.418283</td>
          <td>0.720671</td>
          <td>30.080475</td>
          <td>1.725031</td>
          <td>26.035223</td>
          <td>0.159192</td>
          <td>25.185701</td>
          <td>0.142951</td>
          <td>24.380882</td>
          <td>0.158649</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.986616</td>
          <td>3.589996</td>
          <td>27.095358</td>
          <td>0.252293</td>
          <td>26.052742</td>
          <td>0.169017</td>
          <td>25.488209</td>
          <td>0.193182</td>
          <td>25.613291</td>
          <td>0.449565</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.150015</td>
          <td>0.318069</td>
          <td>26.154740</td>
          <td>0.118307</td>
          <td>25.957424</td>
          <td>0.089348</td>
          <td>25.693519</td>
          <td>0.115918</td>
          <td>25.803667</td>
          <td>0.235927</td>
          <td>24.991921</td>
          <td>0.259089</td>
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
          <td>26.526594</td>
          <td>0.165936</td>
          <td>25.620580</td>
          <td>0.067767</td>
          <td>25.026643</td>
          <td>0.065889</td>
          <td>24.833345</td>
          <td>0.105160</td>
          <td>25.017135</td>
          <td>0.269748</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.401442</td>
          <td>0.794855</td>
          <td>26.962906</td>
          <td>0.235896</td>
          <td>26.086699</td>
          <td>0.100468</td>
          <td>25.156348</td>
          <td>0.072615</td>
          <td>24.717270</td>
          <td>0.093411</td>
          <td>24.142543</td>
          <td>0.126884</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.769746</td>
          <td>1.710035</td>
          <td>27.204968</td>
          <td>0.289573</td>
          <td>26.370632</td>
          <td>0.129754</td>
          <td>26.611353</td>
          <td>0.255637</td>
          <td>26.358582</td>
          <td>0.372843</td>
          <td>26.100371</td>
          <td>0.612337</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.680981</td>
          <td>0.488754</td>
          <td>26.033135</td>
          <td>0.109440</td>
          <td>26.010428</td>
          <td>0.096564</td>
          <td>25.689750</td>
          <td>0.119279</td>
          <td>25.863492</td>
          <td>0.255157</td>
          <td>25.469692</td>
          <td>0.390266</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.874374</td>
          <td>0.256337</td>
          <td>26.455912</td>
          <td>0.154722</td>
          <td>26.593964</td>
          <td>0.156830</td>
          <td>26.486079</td>
          <td>0.229940</td>
          <td>25.722449</td>
          <td>0.222591</td>
          <td>25.227840</td>
          <td>0.316433</td>
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
          <td>27.316045</td>
          <td>0.685199</td>
          <td>26.620129</td>
          <td>0.153565</td>
          <td>26.047663</td>
          <td>0.082244</td>
          <td>25.120933</td>
          <td>0.059111</td>
          <td>24.719632</td>
          <td>0.079290</td>
          <td>24.071167</td>
          <td>0.100612</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.711457</td>
          <td>0.376684</td>
          <td>27.023086</td>
          <td>0.191470</td>
          <td>26.135619</td>
          <td>0.144062</td>
          <td>26.147329</td>
          <td>0.269211</td>
          <td>25.332837</td>
          <td>0.293085</td>
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
          <td>28.708298</td>
          <td>0.747149</td>
          <td>26.116708</td>
          <td>0.153950</td>
          <td>24.974592</td>
          <td>0.107634</td>
          <td>24.864562</td>
          <td>0.215788</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.879983</td>
          <td>0.210347</td>
          <td>26.497775</td>
          <td>0.244561</td>
          <td>25.769141</td>
          <td>0.243320</td>
          <td>25.168043</td>
          <td>0.317109</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.221887</td>
          <td>0.302954</td>
          <td>25.963271</td>
          <td>0.086864</td>
          <td>25.860564</td>
          <td>0.069802</td>
          <td>25.512105</td>
          <td>0.083674</td>
          <td>25.381949</td>
          <td>0.141572</td>
          <td>24.993385</td>
          <td>0.222003</td>
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
          <td>26.500592</td>
          <td>0.148247</td>
          <td>25.409962</td>
          <td>0.050614</td>
          <td>25.107389</td>
          <td>0.063470</td>
          <td>25.199705</td>
          <td>0.130499</td>
          <td>24.585357</td>
          <td>0.170028</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.485150</td>
          <td>0.376506</td>
          <td>26.895369</td>
          <td>0.196661</td>
          <td>26.004590</td>
          <td>0.080489</td>
          <td>25.169250</td>
          <td>0.062779</td>
          <td>24.873722</td>
          <td>0.092314</td>
          <td>24.391054</td>
          <td>0.135156</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.380125</td>
          <td>0.354018</td>
          <td>26.638513</td>
          <td>0.162602</td>
          <td>26.500924</td>
          <td>0.128355</td>
          <td>26.571968</td>
          <td>0.218892</td>
          <td>26.393215</td>
          <td>0.342596</td>
          <td>25.438003</td>
          <td>0.333552</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.307031</td>
          <td>0.725459</td>
          <td>26.239422</td>
          <td>0.122107</td>
          <td>26.078103</td>
          <td>0.094755</td>
          <td>25.886398</td>
          <td>0.130565</td>
          <td>25.935944</td>
          <td>0.251712</td>
          <td>25.374827</td>
          <td>0.337296</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.751206</td>
          <td>0.467548</td>
          <td>26.799503</td>
          <td>0.184838</td>
          <td>26.277627</td>
          <td>0.104637</td>
          <td>26.218541</td>
          <td>0.160804</td>
          <td>25.913415</td>
          <td>0.230135</td>
          <td>25.453084</td>
          <td>0.334427</td>
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
