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

    <pzflow.flow.Flow at 0x7ff120078cd0>



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
    0      23.994413  0.048763  0.038909  
    1      25.391064  0.162548  0.109186  
    2      24.304707  0.026914  0.014150  
    3      25.291103  0.074303  0.057752  
    4      25.096743  0.039283  0.020326  
    ...          ...       ...       ...  
    99995  24.737946  0.005739  0.004572  
    99996  24.224169  0.103244  0.090551  
    99997  25.613836  0.126814  0.117970  
    99998  25.274899  0.001017  0.000585  
    99999  25.699642  0.059156  0.039530  
    
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
          <td>27.252742</td>
          <td>0.656009</td>
          <td>26.740880</td>
          <td>0.170214</td>
          <td>26.089876</td>
          <td>0.085350</td>
          <td>25.217735</td>
          <td>0.064401</td>
          <td>24.601955</td>
          <td>0.071449</td>
          <td>24.068035</td>
          <td>0.100323</td>
          <td>0.048763</td>
          <td>0.038909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.806310</td>
          <td>0.405044</td>
          <td>26.418580</td>
          <td>0.113848</td>
          <td>26.367319</td>
          <td>0.175452</td>
          <td>25.751192</td>
          <td>0.193664</td>
          <td>26.102848</td>
          <td>0.529737</td>
          <td>0.162548</td>
          <td>0.109186</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.792261</td>
          <td>0.933873</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.636429</td>
          <td>1.240990</td>
          <td>25.882209</td>
          <td>0.115567</td>
          <td>24.895645</td>
          <td>0.092572</td>
          <td>24.261833</td>
          <td>0.118813</td>
          <td>0.026914</td>
          <td>0.014150</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.221868</td>
          <td>1.064049</td>
          <td>27.592641</td>
          <td>0.305839</td>
          <td>26.095315</td>
          <td>0.139011</td>
          <td>25.519521</td>
          <td>0.159093</td>
          <td>25.065670</td>
          <td>0.235393</td>
          <td>0.074303</td>
          <td>0.057752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.766878</td>
          <td>0.208484</td>
          <td>26.041783</td>
          <td>0.092950</td>
          <td>26.043847</td>
          <td>0.081957</td>
          <td>25.683226</td>
          <td>0.097117</td>
          <td>25.646191</td>
          <td>0.177216</td>
          <td>24.831148</td>
          <td>0.193535</td>
          <td>0.039283</td>
          <td>0.020326</td>
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
          <td>26.362303</td>
          <td>0.122948</td>
          <td>25.553755</td>
          <td>0.053097</td>
          <td>25.102320</td>
          <td>0.058134</td>
          <td>24.913791</td>
          <td>0.094059</td>
          <td>24.509879</td>
          <td>0.147233</td>
          <td>0.005739</td>
          <td>0.004572</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.353668</td>
          <td>0.122030</td>
          <td>25.960292</td>
          <td>0.076129</td>
          <td>25.278544</td>
          <td>0.067966</td>
          <td>24.853901</td>
          <td>0.089236</td>
          <td>24.053069</td>
          <td>0.099016</td>
          <td>0.103244</td>
          <td>0.090551</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>32.373880</td>
          <td>4.929365</td>
          <td>26.807465</td>
          <td>0.180108</td>
          <td>26.440611</td>
          <td>0.116054</td>
          <td>26.380995</td>
          <td>0.177500</td>
          <td>25.792602</td>
          <td>0.200527</td>
          <td>25.808908</td>
          <td>0.425506</td>
          <td>0.126814</td>
          <td>0.117970</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.973229</td>
          <td>0.538175</td>
          <td>26.042979</td>
          <td>0.093048</td>
          <td>26.118229</td>
          <td>0.087508</td>
          <td>26.012316</td>
          <td>0.129390</td>
          <td>25.627854</td>
          <td>0.174479</td>
          <td>26.440398</td>
          <td>0.672699</td>
          <td>0.001017</td>
          <td>0.000585</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.860733</td>
          <td>0.495630</td>
          <td>26.507486</td>
          <td>0.139388</td>
          <td>26.354652</td>
          <td>0.107672</td>
          <td>26.534616</td>
          <td>0.202067</td>
          <td>25.656813</td>
          <td>0.178819</td>
          <td>25.436149</td>
          <td>0.318123</td>
          <td>0.059156</td>
          <td>0.039530</td>
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
          <td>27.641962</td>
          <td>0.927423</td>
          <td>26.772644</td>
          <td>0.201747</td>
          <td>25.956147</td>
          <td>0.089807</td>
          <td>25.132669</td>
          <td>0.071287</td>
          <td>24.674675</td>
          <td>0.090196</td>
          <td>23.930384</td>
          <td>0.105750</td>
          <td>0.048763</td>
          <td>0.038909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.545474</td>
          <td>0.899649</td>
          <td>27.025580</td>
          <td>0.260589</td>
          <td>26.599422</td>
          <td>0.165523</td>
          <td>26.667383</td>
          <td>0.280161</td>
          <td>26.060433</td>
          <td>0.307532</td>
          <td>25.346584</td>
          <td>0.364191</td>
          <td>0.162548</td>
          <td>0.109186</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.143308</td>
          <td>1.239708</td>
          <td>28.620501</td>
          <td>0.812141</td>
          <td>27.319492</td>
          <td>0.284990</td>
          <td>26.110111</td>
          <td>0.166196</td>
          <td>24.788841</td>
          <td>0.099201</td>
          <td>24.259374</td>
          <td>0.139989</td>
          <td>0.026914</td>
          <td>0.014150</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.798555</td>
          <td>1.024427</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.210569</td>
          <td>0.264060</td>
          <td>25.976576</td>
          <td>0.150257</td>
          <td>25.518618</td>
          <td>0.188500</td>
          <td>25.528173</td>
          <td>0.402212</td>
          <td>0.074303</td>
          <td>0.057752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.931370</td>
          <td>0.267259</td>
          <td>26.155058</td>
          <td>0.118653</td>
          <td>25.954329</td>
          <td>0.089371</td>
          <td>25.622855</td>
          <td>0.109327</td>
          <td>25.493198</td>
          <td>0.182461</td>
          <td>24.975984</td>
          <td>0.256456</td>
          <td>0.039283</td>
          <td>0.020326</td>
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
          <td>30.810463</td>
          <td>3.530586</td>
          <td>26.291468</td>
          <td>0.133147</td>
          <td>25.343551</td>
          <td>0.051898</td>
          <td>25.087999</td>
          <td>0.068061</td>
          <td>24.858445</td>
          <td>0.105276</td>
          <td>25.244602</td>
          <td>0.317723</td>
          <td>0.005739</td>
          <td>0.004572</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.911561</td>
          <td>0.577848</td>
          <td>26.503215</td>
          <td>0.164109</td>
          <td>26.222788</td>
          <td>0.116209</td>
          <td>25.222692</td>
          <td>0.079184</td>
          <td>24.873310</td>
          <td>0.109989</td>
          <td>24.200763</td>
          <td>0.137112</td>
          <td>0.103244</td>
          <td>0.090551</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.823694</td>
          <td>0.548552</td>
          <td>26.658472</td>
          <td>0.190081</td>
          <td>26.274909</td>
          <td>0.123714</td>
          <td>26.279226</td>
          <td>0.200941</td>
          <td>25.870133</td>
          <td>0.260670</td>
          <td>26.468552</td>
          <td>0.808159</td>
          <td>0.126814</td>
          <td>0.117970</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.581734</td>
          <td>0.444605</td>
          <td>26.215555</td>
          <td>0.124677</td>
          <td>26.048893</td>
          <td>0.096787</td>
          <td>25.868693</td>
          <td>0.134884</td>
          <td>25.581774</td>
          <td>0.195995</td>
          <td>24.746787</td>
          <td>0.211461</td>
          <td>0.001017</td>
          <td>0.000585</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.705093</td>
          <td>0.490250</td>
          <td>26.646615</td>
          <td>0.181736</td>
          <td>26.966789</td>
          <td>0.214633</td>
          <td>26.166580</td>
          <td>0.175594</td>
          <td>25.759853</td>
          <td>0.229282</td>
          <td>25.774509</td>
          <td>0.481914</td>
          <td>0.059156</td>
          <td>0.039530</td>
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
          <td>28.139341</td>
          <td>1.160624</td>
          <td>26.504339</td>
          <td>0.142094</td>
          <td>26.275278</td>
          <td>0.103059</td>
          <td>25.038664</td>
          <td>0.056462</td>
          <td>24.797321</td>
          <td>0.087118</td>
          <td>24.085969</td>
          <td>0.104659</td>
          <td>0.048763</td>
          <td>0.038909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.392982</td>
          <td>2.253467</td>
          <td>28.320997</td>
          <td>0.684284</td>
          <td>26.452849</td>
          <td>0.143373</td>
          <td>26.390908</td>
          <td>0.219200</td>
          <td>25.668692</td>
          <td>0.219421</td>
          <td>25.310943</td>
          <td>0.348147</td>
          <td>0.162548</td>
          <td>0.109186</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.730532</td>
          <td>1.570806</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.116283</td>
          <td>0.461955</td>
          <td>26.249684</td>
          <td>0.159731</td>
          <td>25.065768</td>
          <td>0.108119</td>
          <td>24.197525</td>
          <td>0.113068</td>
          <td>0.026914</td>
          <td>0.014150</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.215779</td>
          <td>1.095574</td>
          <td>28.067596</td>
          <td>0.464906</td>
          <td>25.987572</td>
          <td>0.134251</td>
          <td>25.627833</td>
          <td>0.184239</td>
          <td>25.104562</td>
          <td>0.256768</td>
          <td>0.074303</td>
          <td>0.057752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.391325</td>
          <td>0.349138</td>
          <td>26.031891</td>
          <td>0.093199</td>
          <td>25.761958</td>
          <td>0.064720</td>
          <td>25.726006</td>
          <td>0.102215</td>
          <td>25.773327</td>
          <td>0.199782</td>
          <td>25.059649</td>
          <td>0.237232</td>
          <td>0.039283</td>
          <td>0.020326</td>
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
          <td>26.422643</td>
          <td>0.129584</td>
          <td>25.393755</td>
          <td>0.046082</td>
          <td>25.173228</td>
          <td>0.061933</td>
          <td>25.050157</td>
          <td>0.106034</td>
          <td>24.516836</td>
          <td>0.148171</td>
          <td>0.005739</td>
          <td>0.004572</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.591070</td>
          <td>0.434072</td>
          <td>26.742331</td>
          <td>0.187635</td>
          <td>26.069162</td>
          <td>0.093894</td>
          <td>25.178339</td>
          <td>0.070095</td>
          <td>24.947152</td>
          <td>0.108431</td>
          <td>24.116672</td>
          <td>0.117629</td>
          <td>0.103244</td>
          <td>0.090551</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.539655</td>
          <td>0.432996</td>
          <td>26.581375</td>
          <td>0.171906</td>
          <td>26.200548</td>
          <td>0.111417</td>
          <td>26.448479</td>
          <td>0.222461</td>
          <td>26.039030</td>
          <td>0.288083</td>
          <td>25.513003</td>
          <td>0.395160</td>
          <td>0.126814</td>
          <td>0.117970</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.719574</td>
          <td>0.446063</td>
          <td>26.159812</td>
          <td>0.103073</td>
          <td>26.110017</td>
          <td>0.086879</td>
          <td>25.947749</td>
          <td>0.122346</td>
          <td>25.804936</td>
          <td>0.202616</td>
          <td>25.248099</td>
          <td>0.273397</td>
          <td>0.001017</td>
          <td>0.000585</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.843383</td>
          <td>0.979203</td>
          <td>26.648265</td>
          <td>0.161760</td>
          <td>26.465458</td>
          <td>0.122535</td>
          <td>26.687136</td>
          <td>0.237105</td>
          <td>26.257130</td>
          <td>0.303038</td>
          <td>25.518906</td>
          <td>0.350352</td>
          <td>0.059156</td>
          <td>0.039530</td>
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
