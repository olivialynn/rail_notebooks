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

    <pzflow.flow.Flow at 0x7f7cee3f8700>



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
          <td>27.222008</td>
          <td>0.642190</td>
          <td>26.786900</td>
          <td>0.176997</td>
          <td>25.990150</td>
          <td>0.078164</td>
          <td>25.245688</td>
          <td>0.066017</td>
          <td>24.814834</td>
          <td>0.086220</td>
          <td>24.056941</td>
          <td>0.099352</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.128680</td>
          <td>2.761448</td>
          <td>27.419215</td>
          <td>0.298697</td>
          <td>26.731173</td>
          <td>0.149210</td>
          <td>26.153771</td>
          <td>0.146187</td>
          <td>25.475829</td>
          <td>0.153253</td>
          <td>25.285904</td>
          <td>0.281917</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.408018</td>
          <td>2.118391</td>
          <td>28.187718</td>
          <td>0.538659</td>
          <td>29.217217</td>
          <td>0.973798</td>
          <td>25.997022</td>
          <td>0.127687</td>
          <td>25.100102</td>
          <td>0.110721</td>
          <td>24.285477</td>
          <td>0.121280</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.713468</td>
          <td>0.889141</td>
          <td>29.554726</td>
          <td>1.283568</td>
          <td>28.524094</td>
          <td>0.618065</td>
          <td>26.545906</td>
          <td>0.203989</td>
          <td>25.501374</td>
          <td>0.156642</td>
          <td>27.130569</td>
          <td>1.045850</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.703608</td>
          <td>0.197731</td>
          <td>26.144266</td>
          <td>0.101680</td>
          <td>25.887514</td>
          <td>0.071384</td>
          <td>25.608502</td>
          <td>0.090948</td>
          <td>25.634272</td>
          <td>0.175432</td>
          <td>24.492406</td>
          <td>0.145038</td>
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
          <td>26.814381</td>
          <td>0.478891</td>
          <td>26.255056</td>
          <td>0.112004</td>
          <td>25.499965</td>
          <td>0.050621</td>
          <td>25.100652</td>
          <td>0.058048</td>
          <td>25.077666</td>
          <td>0.108574</td>
          <td>25.065187</td>
          <td>0.235299</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.639382</td>
          <td>1.498141</td>
          <td>26.633448</td>
          <td>0.155309</td>
          <td>26.077341</td>
          <td>0.084413</td>
          <td>25.192622</td>
          <td>0.062983</td>
          <td>24.875210</td>
          <td>0.090924</td>
          <td>24.123845</td>
          <td>0.105344</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.650933</td>
          <td>0.423447</td>
          <td>26.695007</td>
          <td>0.163694</td>
          <td>26.331297</td>
          <td>0.105497</td>
          <td>26.209522</td>
          <td>0.153353</td>
          <td>26.404486</td>
          <td>0.330793</td>
          <td>25.683644</td>
          <td>0.386473</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.530277</td>
          <td>0.385983</td>
          <td>26.170100</td>
          <td>0.104003</td>
          <td>26.007105</td>
          <td>0.079343</td>
          <td>26.084303</td>
          <td>0.137696</td>
          <td>25.821892</td>
          <td>0.205515</td>
          <td>25.698928</td>
          <td>0.391070</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.464148</td>
          <td>0.756926</td>
          <td>27.157225</td>
          <td>0.241288</td>
          <td>26.308791</td>
          <td>0.103440</td>
          <td>26.207369</td>
          <td>0.153070</td>
          <td>26.027276</td>
          <td>0.243769</td>
          <td>25.305963</td>
          <td>0.286533</td>
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
          <td>26.706811</td>
          <td>0.189807</td>
          <td>25.980051</td>
          <td>0.091114</td>
          <td>25.129648</td>
          <td>0.070612</td>
          <td>24.673205</td>
          <td>0.089490</td>
          <td>23.973564</td>
          <td>0.109080</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.698869</td>
          <td>0.485402</td>
          <td>28.312354</td>
          <td>0.660194</td>
          <td>26.511723</td>
          <td>0.144750</td>
          <td>26.088885</td>
          <td>0.162996</td>
          <td>26.184076</td>
          <td>0.321306</td>
          <td>24.976105</td>
          <td>0.255725</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.876655</td>
          <td>1.658866</td>
          <td>27.594015</td>
          <td>0.361370</td>
          <td>26.453831</td>
          <td>0.226566</td>
          <td>25.138164</td>
          <td>0.137214</td>
          <td>24.325470</td>
          <td>0.151298</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.252780</td>
          <td>2.002559</td>
          <td>27.388427</td>
          <td>0.319824</td>
          <td>26.646139</td>
          <td>0.277062</td>
          <td>25.391679</td>
          <td>0.178051</td>
          <td>25.074526</td>
          <td>0.295184</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.033318</td>
          <td>0.289669</td>
          <td>26.476499</td>
          <td>0.156131</td>
          <td>25.838263</td>
          <td>0.080447</td>
          <td>25.672090</td>
          <td>0.113775</td>
          <td>25.738869</td>
          <td>0.223588</td>
          <td>25.782323</td>
          <td>0.481287</td>
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
          <td>27.057771</td>
          <td>0.636613</td>
          <td>26.398281</td>
          <td>0.148700</td>
          <td>25.315846</td>
          <td>0.051724</td>
          <td>25.070482</td>
          <td>0.068496</td>
          <td>24.826422</td>
          <td>0.104525</td>
          <td>24.794574</td>
          <td>0.224605</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.535950</td>
          <td>1.523345</td>
          <td>26.731701</td>
          <td>0.194527</td>
          <td>26.031555</td>
          <td>0.095727</td>
          <td>25.253697</td>
          <td>0.079135</td>
          <td>24.740459</td>
          <td>0.095332</td>
          <td>24.245707</td>
          <td>0.138717</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.104541</td>
          <td>0.654353</td>
          <td>26.515505</td>
          <td>0.163195</td>
          <td>26.319164</td>
          <td>0.124094</td>
          <td>26.219253</td>
          <td>0.184392</td>
          <td>26.098960</td>
          <td>0.303616</td>
          <td>25.223597</td>
          <td>0.316172</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.960908</td>
          <td>0.598593</td>
          <td>26.235099</td>
          <td>0.130409</td>
          <td>26.302575</td>
          <td>0.124594</td>
          <td>25.853606</td>
          <td>0.137464</td>
          <td>25.830457</td>
          <td>0.248330</td>
          <td>25.358161</td>
          <td>0.357804</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.177543</td>
          <td>0.327290</td>
          <td>26.648911</td>
          <td>0.182331</td>
          <td>26.784129</td>
          <td>0.184368</td>
          <td>26.134886</td>
          <td>0.171191</td>
          <td>25.844822</td>
          <td>0.246307</td>
          <td>24.943531</td>
          <td>0.251341</td>
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
          <td>26.765171</td>
          <td>0.173783</td>
          <td>25.997409</td>
          <td>0.078677</td>
          <td>25.202518</td>
          <td>0.063547</td>
          <td>24.548428</td>
          <td>0.068152</td>
          <td>24.206867</td>
          <td>0.113277</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.601423</td>
          <td>0.828357</td>
          <td>28.213374</td>
          <td>0.549131</td>
          <td>26.698145</td>
          <td>0.145169</td>
          <td>26.012019</td>
          <td>0.129483</td>
          <td>25.755067</td>
          <td>0.194473</td>
          <td>25.415863</td>
          <td>0.313291</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.530746</td>
          <td>2.282665</td>
          <td>28.758578</td>
          <td>0.841875</td>
          <td>27.693279</td>
          <td>0.356655</td>
          <td>25.908964</td>
          <td>0.128721</td>
          <td>25.176008</td>
          <td>0.128244</td>
          <td>24.506657</td>
          <td>0.159480</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.890262</td>
          <td>0.580916</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.922938</td>
          <td>0.481530</td>
          <td>26.502338</td>
          <td>0.245481</td>
          <td>25.618956</td>
          <td>0.214828</td>
          <td>25.606579</td>
          <td>0.445893</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.445502</td>
          <td>0.361664</td>
          <td>26.050098</td>
          <td>0.093746</td>
          <td>25.882903</td>
          <td>0.071196</td>
          <td>25.536629</td>
          <td>0.085502</td>
          <td>25.781084</td>
          <td>0.198866</td>
          <td>25.323855</td>
          <td>0.291100</td>
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
          <td>26.455612</td>
          <td>0.382683</td>
          <td>26.394959</td>
          <td>0.135366</td>
          <td>25.453760</td>
          <td>0.052620</td>
          <td>25.134417</td>
          <td>0.065009</td>
          <td>24.950110</td>
          <td>0.105029</td>
          <td>24.618687</td>
          <td>0.174913</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.749390</td>
          <td>0.916779</td>
          <td>26.774800</td>
          <td>0.177629</td>
          <td>26.111902</td>
          <td>0.088472</td>
          <td>25.310719</td>
          <td>0.071161</td>
          <td>25.134434</td>
          <td>0.115962</td>
          <td>24.373352</td>
          <td>0.133104</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.638148</td>
          <td>0.432037</td>
          <td>26.339792</td>
          <td>0.125768</td>
          <td>26.616775</td>
          <td>0.141863</td>
          <td>26.284790</td>
          <td>0.171878</td>
          <td>26.349299</td>
          <td>0.330895</td>
          <td>27.465684</td>
          <td>1.304929</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.157993</td>
          <td>0.309965</td>
          <td>26.081423</td>
          <td>0.106423</td>
          <td>26.184949</td>
          <td>0.104054</td>
          <td>26.061858</td>
          <td>0.151870</td>
          <td>26.167533</td>
          <td>0.303798</td>
          <td>25.522842</td>
          <td>0.378807</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.967416</td>
          <td>0.252525</td>
          <td>26.856084</td>
          <td>0.193871</td>
          <td>26.579237</td>
          <td>0.136002</td>
          <td>26.160240</td>
          <td>0.152978</td>
          <td>25.955900</td>
          <td>0.238369</td>
          <td>25.548436</td>
          <td>0.360513</td>
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
