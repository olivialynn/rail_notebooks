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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7faad107c9d0>



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
    0      23.994413  0.051668  0.038376  
    1      25.391064  0.149728  0.087284  
    2      24.304707  0.046032  0.034066  
    3      25.291103  0.068580  0.041343  
    4      25.096743  0.159879  0.125351  
    ...          ...       ...       ...  
    99995  24.737946  0.036379  0.022280  
    99996  24.224169  0.124545  0.123918  
    99997  25.613836  0.008036  0.005385  
    99998  25.274899  0.045751  0.027897  
    99999  25.699642  0.017952  0.017897  
    
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
          <td>28.892287</td>
          <td>1.692701</td>
          <td>26.542022</td>
          <td>0.143593</td>
          <td>25.993733</td>
          <td>0.078412</td>
          <td>25.209500</td>
          <td>0.063933</td>
          <td>24.738519</td>
          <td>0.080612</td>
          <td>23.965367</td>
          <td>0.091680</td>
          <td>0.051668</td>
          <td>0.038376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.222972</td>
          <td>0.642620</td>
          <td>27.113312</td>
          <td>0.232691</td>
          <td>26.848639</td>
          <td>0.164989</td>
          <td>26.154748</td>
          <td>0.146310</td>
          <td>25.854440</td>
          <td>0.211190</td>
          <td>25.385927</td>
          <td>0.305596</td>
          <td>0.149728</td>
          <td>0.087284</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.909870</td>
          <td>0.392653</td>
          <td>25.992945</td>
          <td>0.127237</td>
          <td>25.065276</td>
          <td>0.107405</td>
          <td>24.309316</td>
          <td>0.123816</td>
          <td>0.046032</td>
          <td>0.034066</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.883027</td>
          <td>1.521304</td>
          <td>27.386174</td>
          <td>0.258709</td>
          <td>26.527466</td>
          <td>0.200857</td>
          <td>25.795515</td>
          <td>0.201018</td>
          <td>25.702186</td>
          <td>0.392056</td>
          <td>0.068580</td>
          <td>0.041343</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.305400</td>
          <td>0.323554</td>
          <td>26.144344</td>
          <td>0.101687</td>
          <td>26.087564</td>
          <td>0.085177</td>
          <td>25.638748</td>
          <td>0.093398</td>
          <td>25.557265</td>
          <td>0.164304</td>
          <td>25.440534</td>
          <td>0.319237</td>
          <td>0.159879</td>
          <td>0.125351</td>
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
          <td>28.713032</td>
          <td>1.553713</td>
          <td>26.562349</td>
          <td>0.146124</td>
          <td>25.450400</td>
          <td>0.048441</td>
          <td>25.107846</td>
          <td>0.058420</td>
          <td>24.890570</td>
          <td>0.092160</td>
          <td>24.656232</td>
          <td>0.166877</td>
          <td>0.036379</td>
          <td>0.022280</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.833465</td>
          <td>0.184113</td>
          <td>26.133893</td>
          <td>0.088723</td>
          <td>25.129311</td>
          <td>0.059543</td>
          <td>24.775431</td>
          <td>0.083279</td>
          <td>24.183960</td>
          <td>0.111022</td>
          <td>0.124545</td>
          <td>0.123918</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.651806</td>
          <td>0.423729</td>
          <td>27.342766</td>
          <td>0.280817</td>
          <td>26.514298</td>
          <td>0.123731</td>
          <td>26.389325</td>
          <td>0.178758</td>
          <td>25.738575</td>
          <td>0.191616</td>
          <td>25.964221</td>
          <td>0.478311</td>
          <td>0.008036</td>
          <td>0.005385</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.332220</td>
          <td>0.330514</td>
          <td>26.262376</td>
          <td>0.112721</td>
          <td>25.950475</td>
          <td>0.075472</td>
          <td>26.015816</td>
          <td>0.129782</td>
          <td>25.682512</td>
          <td>0.182754</td>
          <td>25.167126</td>
          <td>0.255902</td>
          <td>0.045751</td>
          <td>0.027897</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.719448</td>
          <td>0.446018</td>
          <td>26.716703</td>
          <td>0.166748</td>
          <td>26.579903</td>
          <td>0.130969</td>
          <td>26.154618</td>
          <td>0.146293</td>
          <td>25.688555</td>
          <td>0.183691</td>
          <td>25.884027</td>
          <td>0.450423</td>
          <td>0.017952</td>
          <td>0.017897</td>
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
          <td>27.133806</td>
          <td>0.272025</td>
          <td>26.148250</td>
          <td>0.106321</td>
          <td>25.182332</td>
          <td>0.074519</td>
          <td>24.627067</td>
          <td>0.086533</td>
          <td>24.138802</td>
          <td>0.126833</td>
          <td>0.051668</td>
          <td>0.038376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.867249</td>
          <td>1.085257</td>
          <td>27.622696</td>
          <td>0.413994</td>
          <td>26.557223</td>
          <td>0.157711</td>
          <td>26.087430</td>
          <td>0.170791</td>
          <td>25.610564</td>
          <td>0.210208</td>
          <td>25.706303</td>
          <td>0.474063</td>
          <td>0.149728</td>
          <td>0.087284</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.857652</td>
          <td>0.436322</td>
          <td>25.958520</td>
          <td>0.146561</td>
          <td>24.962876</td>
          <td>0.115942</td>
          <td>24.142968</td>
          <td>0.127105</td>
          <td>0.046032</td>
          <td>0.034066</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.572814</td>
          <td>1.419901</td>
          <td>27.862220</td>
          <td>0.439810</td>
          <td>26.201364</td>
          <td>0.181254</td>
          <td>25.453004</td>
          <td>0.177626</td>
          <td>24.637413</td>
          <td>0.194979</td>
          <td>0.068580</td>
          <td>0.041343</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.173876</td>
          <td>0.339182</td>
          <td>25.961420</td>
          <td>0.106016</td>
          <td>25.959338</td>
          <td>0.095520</td>
          <td>25.792216</td>
          <td>0.134937</td>
          <td>25.447865</td>
          <td>0.186404</td>
          <td>24.742871</td>
          <td>0.224620</td>
          <td>0.159879</td>
          <td>0.125351</td>
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
          <td>26.415436</td>
          <td>0.392409</td>
          <td>26.591440</td>
          <td>0.172612</td>
          <td>25.491194</td>
          <td>0.059341</td>
          <td>25.060544</td>
          <td>0.066635</td>
          <td>24.906916</td>
          <td>0.110155</td>
          <td>24.651760</td>
          <td>0.195859</td>
          <td>0.036379</td>
          <td>0.022280</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.092308</td>
          <td>2.000472</td>
          <td>26.670682</td>
          <td>0.192272</td>
          <td>25.854641</td>
          <td>0.085783</td>
          <td>25.147566</td>
          <td>0.075567</td>
          <td>24.940106</td>
          <td>0.118771</td>
          <td>24.064590</td>
          <td>0.124216</td>
          <td>0.124545</td>
          <td>0.123918</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.415485</td>
          <td>0.391657</td>
          <td>26.607630</td>
          <td>0.174554</td>
          <td>26.301805</td>
          <td>0.120718</td>
          <td>26.410006</td>
          <td>0.213754</td>
          <td>25.871108</td>
          <td>0.249369</td>
          <td>26.690373</td>
          <td>0.898053</td>
          <td>0.008036</td>
          <td>0.005385</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.858007</td>
          <td>0.251991</td>
          <td>25.991108</td>
          <td>0.103004</td>
          <td>25.951718</td>
          <td>0.089304</td>
          <td>25.737784</td>
          <td>0.121024</td>
          <td>25.937945</td>
          <td>0.264560</td>
          <td>26.404468</td>
          <td>0.749281</td>
          <td>0.045751</td>
          <td>0.027897</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.663303</td>
          <td>0.472993</td>
          <td>26.871112</td>
          <td>0.218022</td>
          <td>26.582218</td>
          <td>0.153909</td>
          <td>26.735094</td>
          <td>0.279615</td>
          <td>25.937966</td>
          <td>0.263640</td>
          <td>26.240134</td>
          <td>0.668464</td>
          <td>0.017952</td>
          <td>0.017897</td>
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
          <td>27.000838</td>
          <td>0.557690</td>
          <td>26.450372</td>
          <td>0.135823</td>
          <td>26.010383</td>
          <td>0.081787</td>
          <td>25.218967</td>
          <td>0.066364</td>
          <td>24.735843</td>
          <td>0.082656</td>
          <td>24.060608</td>
          <td>0.102530</td>
          <td>0.051668</td>
          <td>0.038376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.449205</td>
          <td>0.349214</td>
          <td>26.816077</td>
          <td>0.188168</td>
          <td>26.555099</td>
          <td>0.241782</td>
          <td>25.882667</td>
          <td>0.252469</td>
          <td>24.832537</td>
          <td>0.227792</td>
          <td>0.149728</td>
          <td>0.087284</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.245181</td>
          <td>1.229104</td>
          <td>29.048910</td>
          <td>0.971954</td>
          <td>27.600374</td>
          <td>0.313894</td>
          <td>26.046317</td>
          <td>0.136276</td>
          <td>25.214800</td>
          <td>0.124999</td>
          <td>24.427528</td>
          <td>0.140229</td>
          <td>0.046032</td>
          <td>0.034066</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.000583</td>
          <td>1.079510</td>
          <td>28.796745</td>
          <td>0.840473</td>
          <td>27.699293</td>
          <td>0.345599</td>
          <td>26.219380</td>
          <td>0.161317</td>
          <td>25.434419</td>
          <td>0.153990</td>
          <td>25.149728</td>
          <td>0.262566</td>
          <td>0.068580</td>
          <td>0.041343</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.828359</td>
          <td>0.255480</td>
          <td>26.155600</td>
          <td>0.124617</td>
          <td>26.018519</td>
          <td>0.099797</td>
          <td>25.656197</td>
          <td>0.118948</td>
          <td>25.231945</td>
          <td>0.153913</td>
          <td>25.351696</td>
          <td>0.364494</td>
          <td>0.159879</td>
          <td>0.125351</td>
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
          <td>26.650925</td>
          <td>0.159279</td>
          <td>25.586431</td>
          <td>0.055339</td>
          <td>25.098062</td>
          <td>0.058670</td>
          <td>24.900782</td>
          <td>0.094130</td>
          <td>24.786150</td>
          <td>0.188606</td>
          <td>0.036379</td>
          <td>0.022280</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>31.212459</td>
          <td>3.929865</td>
          <td>26.703688</td>
          <td>0.191348</td>
          <td>26.117331</td>
          <td>0.104045</td>
          <td>25.222142</td>
          <td>0.077622</td>
          <td>24.827233</td>
          <td>0.103706</td>
          <td>24.246387</td>
          <td>0.139952</td>
          <td>0.124545</td>
          <td>0.123918</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.627847</td>
          <td>0.842400</td>
          <td>26.766443</td>
          <td>0.174044</td>
          <td>26.529923</td>
          <td>0.125498</td>
          <td>26.226703</td>
          <td>0.155728</td>
          <td>25.619077</td>
          <td>0.173289</td>
          <td>25.522060</td>
          <td>0.340780</td>
          <td>0.008036</td>
          <td>0.005385</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.445318</td>
          <td>0.365588</td>
          <td>26.130447</td>
          <td>0.102132</td>
          <td>26.122947</td>
          <td>0.089567</td>
          <td>25.999278</td>
          <td>0.130483</td>
          <td>25.545303</td>
          <td>0.165669</td>
          <td>25.194189</td>
          <td>0.266490</td>
          <td>0.045751</td>
          <td>0.027897</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.098696</td>
          <td>0.590335</td>
          <td>26.748935</td>
          <td>0.172010</td>
          <td>26.620819</td>
          <td>0.136265</td>
          <td>26.219919</td>
          <td>0.155416</td>
          <td>26.886442</td>
          <td>0.481136</td>
          <td>26.140463</td>
          <td>0.546438</td>
          <td>0.017952</td>
          <td>0.017897</td>
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
