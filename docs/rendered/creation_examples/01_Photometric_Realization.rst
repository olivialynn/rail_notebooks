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

    <pzflow.flow.Flow at 0x7fe02851f790>



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
    0      23.994413  0.094601  0.051828  
    1      25.391064  0.031011  0.018386  
    2      24.304707  0.198149  0.149471  
    3      25.291103  0.039206  0.036093  
    4      25.096743  0.142242  0.128260  
    ...          ...       ...       ...  
    99995  24.737946  0.174584  0.157995  
    99996  24.224169  0.143712  0.102268  
    99997  25.613836  0.006775  0.004116  
    99998  25.274899  0.094873  0.090266  
    99999  25.699642  0.022969  0.019192  
    
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
          <td>27.062589</td>
          <td>0.573945</td>
          <td>26.831592</td>
          <td>0.183822</td>
          <td>26.004964</td>
          <td>0.079193</td>
          <td>25.140137</td>
          <td>0.060118</td>
          <td>24.737282</td>
          <td>0.080524</td>
          <td>24.037988</td>
          <td>0.097715</td>
          <td>0.094601</td>
          <td>0.051828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.195384</td>
          <td>0.630393</td>
          <td>27.807606</td>
          <td>0.405447</td>
          <td>26.500512</td>
          <td>0.122259</td>
          <td>26.306566</td>
          <td>0.166615</td>
          <td>26.188453</td>
          <td>0.278124</td>
          <td>25.464478</td>
          <td>0.325383</td>
          <td>0.031011</td>
          <td>0.018386</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.326478</td>
          <td>0.595031</td>
          <td>28.981006</td>
          <td>0.840218</td>
          <td>26.083255</td>
          <td>0.137572</td>
          <td>25.144469</td>
          <td>0.115086</td>
          <td>24.412507</td>
          <td>0.135386</td>
          <td>0.198149</td>
          <td>0.149471</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.251273</td>
          <td>0.563948</td>
          <td>27.280915</td>
          <td>0.237251</td>
          <td>26.217292</td>
          <td>0.154378</td>
          <td>25.353270</td>
          <td>0.137923</td>
          <td>25.284718</td>
          <td>0.281646</td>
          <td>0.039206</td>
          <td>0.036093</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.173473</td>
          <td>0.291126</td>
          <td>26.121830</td>
          <td>0.099704</td>
          <td>25.981167</td>
          <td>0.077546</td>
          <td>25.874814</td>
          <td>0.114825</td>
          <td>25.592680</td>
          <td>0.169338</td>
          <td>25.182326</td>
          <td>0.259108</td>
          <td>0.142242</td>
          <td>0.128260</td>
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
          <td>27.425583</td>
          <td>0.737750</td>
          <td>26.641773</td>
          <td>0.156419</td>
          <td>25.473648</td>
          <td>0.049451</td>
          <td>25.099915</td>
          <td>0.058010</td>
          <td>24.849928</td>
          <td>0.088925</td>
          <td>24.754444</td>
          <td>0.181396</td>
          <td>0.174584</td>
          <td>0.157995</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.985234</td>
          <td>1.049287</td>
          <td>26.607978</td>
          <td>0.151958</td>
          <td>26.098287</td>
          <td>0.085985</td>
          <td>25.176473</td>
          <td>0.062088</td>
          <td>24.808707</td>
          <td>0.085756</td>
          <td>24.148991</td>
          <td>0.107685</td>
          <td>0.143712</td>
          <td>0.102268</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.724360</td>
          <td>0.447673</td>
          <td>26.682833</td>
          <td>0.162003</td>
          <td>26.245518</td>
          <td>0.097863</td>
          <td>26.362170</td>
          <td>0.174686</td>
          <td>25.896195</td>
          <td>0.218679</td>
          <td>25.827001</td>
          <td>0.431402</td>
          <td>0.006775</td>
          <td>0.004116</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.571178</td>
          <td>0.176853</td>
          <td>26.127196</td>
          <td>0.100173</td>
          <td>26.082938</td>
          <td>0.084830</td>
          <td>25.789184</td>
          <td>0.106558</td>
          <td>25.808384</td>
          <td>0.203201</td>
          <td>25.384580</td>
          <td>0.305266</td>
          <td>0.094873</td>
          <td>0.090266</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.488219</td>
          <td>0.373590</td>
          <td>27.248944</td>
          <td>0.260164</td>
          <td>26.636448</td>
          <td>0.137527</td>
          <td>26.482966</td>
          <td>0.193479</td>
          <td>25.996727</td>
          <td>0.237701</td>
          <td>25.582591</td>
          <td>0.357199</td>
          <td>0.022969</td>
          <td>0.019192</td>
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
          <td>27.197186</td>
          <td>0.699925</td>
          <td>26.622582</td>
          <td>0.179759</td>
          <td>26.005411</td>
          <td>0.094982</td>
          <td>25.317567</td>
          <td>0.085044</td>
          <td>24.664350</td>
          <td>0.090528</td>
          <td>24.064898</td>
          <td>0.120453</td>
          <td>0.094601</td>
          <td>0.051828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.596401</td>
          <td>1.567915</td>
          <td>26.699567</td>
          <td>0.189006</td>
          <td>26.374161</td>
          <td>0.128797</td>
          <td>26.129915</td>
          <td>0.169131</td>
          <td>25.849689</td>
          <td>0.245483</td>
          <td>25.540401</td>
          <td>0.401427</td>
          <td>0.031011</td>
          <td>0.018386</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.570649</td>
          <td>0.836680</td>
          <td>28.643440</td>
          <td>0.815332</td>
          <td>26.082332</td>
          <td>0.178068</td>
          <td>24.964789</td>
          <td>0.126786</td>
          <td>23.939551</td>
          <td>0.116561</td>
          <td>0.198149</td>
          <td>0.149471</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.959742</td>
          <td>0.588341</td>
          <td>27.473419</td>
          <td>0.356230</td>
          <td>26.941193</td>
          <td>0.209349</td>
          <td>26.358575</td>
          <td>0.205696</td>
          <td>25.616618</td>
          <td>0.202746</td>
          <td>24.808677</td>
          <td>0.223701</td>
          <td>0.039206</td>
          <td>0.036093</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.130181</td>
          <td>0.326103</td>
          <td>26.151240</td>
          <td>0.124286</td>
          <td>25.992303</td>
          <td>0.097668</td>
          <td>25.532356</td>
          <td>0.106940</td>
          <td>25.554920</td>
          <td>0.202681</td>
          <td>24.812429</td>
          <td>0.236417</td>
          <td>0.142242</td>
          <td>0.128260</td>
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
          <td>26.680393</td>
          <td>0.506391</td>
          <td>26.287593</td>
          <td>0.143299</td>
          <td>25.408809</td>
          <td>0.059999</td>
          <td>25.100160</td>
          <td>0.075248</td>
          <td>24.772985</td>
          <td>0.106436</td>
          <td>24.680013</td>
          <td>0.217508</td>
          <td>0.174584</td>
          <td>0.157995</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.426938</td>
          <td>0.408621</td>
          <td>26.666347</td>
          <td>0.191595</td>
          <td>26.032671</td>
          <td>0.100309</td>
          <td>25.210340</td>
          <td>0.079881</td>
          <td>24.713991</td>
          <td>0.097507</td>
          <td>24.228244</td>
          <td>0.143098</td>
          <td>0.143712</td>
          <td>0.102268</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.761192</td>
          <td>0.994097</td>
          <td>26.523756</td>
          <td>0.162526</td>
          <td>26.458729</td>
          <td>0.138275</td>
          <td>26.340652</td>
          <td>0.201688</td>
          <td>26.028488</td>
          <td>0.283524</td>
          <td>26.148270</td>
          <td>0.626702</td>
          <td>0.006775</td>
          <td>0.004116</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.517279</td>
          <td>0.431423</td>
          <td>26.233448</td>
          <td>0.129866</td>
          <td>26.094592</td>
          <td>0.103631</td>
          <td>25.726215</td>
          <td>0.122730</td>
          <td>25.589255</td>
          <td>0.202648</td>
          <td>26.032443</td>
          <td>0.591171</td>
          <td>0.094873</td>
          <td>0.090266</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.582613</td>
          <td>0.445342</td>
          <td>26.874876</td>
          <td>0.218787</td>
          <td>26.666616</td>
          <td>0.165489</td>
          <td>26.752069</td>
          <td>0.283605</td>
          <td>26.122056</td>
          <td>0.306126</td>
          <td>25.211802</td>
          <td>0.309918</td>
          <td>0.022969</td>
          <td>0.019192</td>
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
          <td>28.626814</td>
          <td>1.532359</td>
          <td>27.062622</td>
          <td>0.236761</td>
          <td>26.101572</td>
          <td>0.092667</td>
          <td>25.247311</td>
          <td>0.071317</td>
          <td>24.782965</td>
          <td>0.090086</td>
          <td>23.961658</td>
          <td>0.098437</td>
          <td>0.094601</td>
          <td>0.051828</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.383635</td>
          <td>0.292283</td>
          <td>26.953671</td>
          <td>0.181921</td>
          <td>26.177370</td>
          <td>0.150524</td>
          <td>25.872566</td>
          <td>0.216195</td>
          <td>25.867784</td>
          <td>0.448418</td>
          <td>0.031011</td>
          <td>0.018386</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.287380</td>
          <td>2.234168</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.003744</td>
          <td>0.542437</td>
          <td>26.183804</td>
          <td>0.201942</td>
          <td>25.025112</td>
          <td>0.139048</td>
          <td>24.246979</td>
          <td>0.158318</td>
          <td>0.198149</td>
          <td>0.149471</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.107465</td>
          <td>0.515011</td>
          <td>27.235335</td>
          <td>0.232577</td>
          <td>26.291975</td>
          <td>0.167756</td>
          <td>25.520607</td>
          <td>0.162194</td>
          <td>25.624773</td>
          <td>0.375671</td>
          <td>0.039206</td>
          <td>0.036093</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.788159</td>
          <td>0.531304</td>
          <td>26.178211</td>
          <td>0.124971</td>
          <td>26.044328</td>
          <td>0.100211</td>
          <td>25.586507</td>
          <td>0.109848</td>
          <td>25.441304</td>
          <td>0.180691</td>
          <td>25.411002</td>
          <td>0.375392</td>
          <td>0.142242</td>
          <td>0.128260</td>
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
          <td>26.398727</td>
          <td>0.418408</td>
          <td>26.312508</td>
          <td>0.150357</td>
          <td>25.503398</td>
          <td>0.067229</td>
          <td>25.105186</td>
          <td>0.077940</td>
          <td>24.791649</td>
          <td>0.111421</td>
          <td>24.717177</td>
          <td>0.230857</td>
          <td>0.174584</td>
          <td>0.157995</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.958810</td>
          <td>0.236535</td>
          <td>25.855634</td>
          <td>0.082579</td>
          <td>25.237741</td>
          <td>0.078587</td>
          <td>24.900560</td>
          <td>0.110425</td>
          <td>24.219755</td>
          <td>0.136591</td>
          <td>0.143712</td>
          <td>0.102268</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.862796</td>
          <td>0.975301</td>
          <td>26.433429</td>
          <td>0.130805</td>
          <td>26.683740</td>
          <td>0.143308</td>
          <td>26.136726</td>
          <td>0.144122</td>
          <td>25.661398</td>
          <td>0.179589</td>
          <td>27.667093</td>
          <td>1.407768</td>
          <td>0.006775</td>
          <td>0.004116</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.425914</td>
          <td>0.380183</td>
          <td>26.336304</td>
          <td>0.131584</td>
          <td>26.332915</td>
          <td>0.117176</td>
          <td>25.942767</td>
          <td>0.135636</td>
          <td>26.011192</td>
          <td>0.265135</td>
          <td>25.364872</td>
          <td>0.331437</td>
          <td>0.094873</td>
          <td>0.090266</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.931317</td>
          <td>0.523826</td>
          <td>26.424558</td>
          <td>0.130431</td>
          <td>26.628046</td>
          <td>0.137347</td>
          <td>26.257019</td>
          <td>0.160706</td>
          <td>25.732013</td>
          <td>0.191667</td>
          <td>26.449253</td>
          <td>0.680118</td>
          <td>0.022969</td>
          <td>0.019192</td>
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
