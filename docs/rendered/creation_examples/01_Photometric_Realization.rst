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

    <pzflow.flow.Flow at 0x7fa1fec94130>



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
          <td>27.935131</td>
          <td>1.018532</td>
          <td>26.888467</td>
          <td>0.192858</td>
          <td>26.105915</td>
          <td>0.086565</td>
          <td>25.252843</td>
          <td>0.066436</td>
          <td>24.753238</td>
          <td>0.081665</td>
          <td>24.026965</td>
          <td>0.096775</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.447371</td>
          <td>0.305527</td>
          <td>26.630398</td>
          <td>0.136810</td>
          <td>26.486574</td>
          <td>0.194068</td>
          <td>25.792514</td>
          <td>0.200513</td>
          <td>25.329413</td>
          <td>0.292013</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.524509</td>
          <td>0.682987</td>
          <td>29.003486</td>
          <td>0.852386</td>
          <td>25.971764</td>
          <td>0.124921</td>
          <td>24.939405</td>
          <td>0.096198</td>
          <td>24.187476</td>
          <td>0.111363</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.146064</td>
          <td>0.522566</td>
          <td>27.037630</td>
          <td>0.193655</td>
          <td>25.953162</td>
          <td>0.122921</td>
          <td>25.467434</td>
          <td>0.152154</td>
          <td>25.350776</td>
          <td>0.297084</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.224948</td>
          <td>0.303426</td>
          <td>26.108765</td>
          <td>0.098570</td>
          <td>25.958384</td>
          <td>0.076001</td>
          <td>25.731894</td>
          <td>0.101349</td>
          <td>25.455180</td>
          <td>0.150563</td>
          <td>24.968323</td>
          <td>0.217113</td>
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
          <td>29.419104</td>
          <td>2.127908</td>
          <td>26.553756</td>
          <td>0.145049</td>
          <td>25.511381</td>
          <td>0.051136</td>
          <td>25.071208</td>
          <td>0.056551</td>
          <td>24.794655</td>
          <td>0.084702</td>
          <td>24.717363</td>
          <td>0.175782</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.049948</td>
          <td>0.220772</td>
          <td>26.103625</td>
          <td>0.086390</td>
          <td>25.323896</td>
          <td>0.070751</td>
          <td>24.911539</td>
          <td>0.093873</td>
          <td>24.416702</td>
          <td>0.135878</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.525347</td>
          <td>0.384513</td>
          <td>26.623563</td>
          <td>0.154000</td>
          <td>26.496413</td>
          <td>0.121824</td>
          <td>26.240816</td>
          <td>0.157518</td>
          <td>25.914366</td>
          <td>0.222012</td>
          <td>25.514618</td>
          <td>0.338581</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.884273</td>
          <td>0.229856</td>
          <td>26.180394</td>
          <td>0.104942</td>
          <td>26.081842</td>
          <td>0.084748</td>
          <td>26.054153</td>
          <td>0.134158</td>
          <td>25.553822</td>
          <td>0.163822</td>
          <td>25.246210</td>
          <td>0.272975</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.237858</td>
          <td>0.649290</td>
          <td>26.714738</td>
          <td>0.166469</td>
          <td>26.491075</td>
          <td>0.121261</td>
          <td>26.163206</td>
          <td>0.147377</td>
          <td>25.434412</td>
          <td>0.147902</td>
          <td>25.755566</td>
          <td>0.408505</td>
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
          <td>28.504136</td>
          <td>1.496748</td>
          <td>26.469633</td>
          <td>0.155173</td>
          <td>26.068825</td>
          <td>0.098496</td>
          <td>25.192439</td>
          <td>0.074643</td>
          <td>24.626913</td>
          <td>0.085919</td>
          <td>23.917649</td>
          <td>0.103880</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.782150</td>
          <td>1.711231</td>
          <td>27.369812</td>
          <td>0.327040</td>
          <td>26.545211</td>
          <td>0.148976</td>
          <td>26.962649</td>
          <td>0.335302</td>
          <td>25.575208</td>
          <td>0.194959</td>
          <td>24.813636</td>
          <td>0.223628</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.653460</td>
          <td>0.415305</td>
          <td>28.110588</td>
          <td>0.534034</td>
          <td>26.093471</td>
          <td>0.167304</td>
          <td>24.980831</td>
          <td>0.119738</td>
          <td>24.341475</td>
          <td>0.153388</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.282859</td>
          <td>3.084285</td>
          <td>27.529614</td>
          <td>0.391292</td>
          <td>27.689205</td>
          <td>0.404763</td>
          <td>26.188884</td>
          <td>0.189685</td>
          <td>26.055933</td>
          <td>0.308219</td>
          <td>25.152245</td>
          <td>0.314176</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.035788</td>
          <td>0.290247</td>
          <td>26.043735</td>
          <td>0.107411</td>
          <td>25.823511</td>
          <td>0.079407</td>
          <td>25.651644</td>
          <td>0.111766</td>
          <td>25.184875</td>
          <td>0.139806</td>
          <td>25.026365</td>
          <td>0.266484</td>
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
          <td>27.400752</td>
          <td>0.802102</td>
          <td>26.548043</td>
          <td>0.168992</td>
          <td>25.495945</td>
          <td>0.060683</td>
          <td>24.934442</td>
          <td>0.060720</td>
          <td>24.868691</td>
          <td>0.108457</td>
          <td>24.899455</td>
          <td>0.244960</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.344266</td>
          <td>0.765572</td>
          <td>26.531160</td>
          <td>0.164143</td>
          <td>25.961400</td>
          <td>0.090008</td>
          <td>25.383252</td>
          <td>0.088703</td>
          <td>24.902353</td>
          <td>0.109838</td>
          <td>24.144882</td>
          <td>0.127141</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.823502</td>
          <td>0.211651</td>
          <td>26.420677</td>
          <td>0.135490</td>
          <td>26.162442</td>
          <td>0.175728</td>
          <td>25.728123</td>
          <td>0.224225</td>
          <td>26.295027</td>
          <td>0.700525</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.361270</td>
          <td>0.383577</td>
          <td>26.112209</td>
          <td>0.117235</td>
          <td>26.081731</td>
          <td>0.102787</td>
          <td>25.852292</td>
          <td>0.137308</td>
          <td>25.847366</td>
          <td>0.251804</td>
          <td>25.150566</td>
          <td>0.303458</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.705544</td>
          <td>0.191259</td>
          <td>26.211716</td>
          <td>0.112719</td>
          <td>26.215132</td>
          <td>0.183249</td>
          <td>26.035812</td>
          <td>0.287834</td>
          <td>25.092351</td>
          <td>0.283773</td>
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
          <td>26.615718</td>
          <td>0.152986</td>
          <td>25.944501</td>
          <td>0.075084</td>
          <td>25.213598</td>
          <td>0.064174</td>
          <td>24.659282</td>
          <td>0.075174</td>
          <td>23.993339</td>
          <td>0.093974</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.052645</td>
          <td>0.221440</td>
          <td>26.547459</td>
          <td>0.127459</td>
          <td>26.050752</td>
          <td>0.133895</td>
          <td>25.619903</td>
          <td>0.173463</td>
          <td>25.721788</td>
          <td>0.398375</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.229110</td>
          <td>0.675677</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.969124</td>
          <td>0.441154</td>
          <td>25.939236</td>
          <td>0.132138</td>
          <td>24.987497</td>
          <td>0.108854</td>
          <td>24.547923</td>
          <td>0.165198</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.316923</td>
          <td>0.378877</td>
          <td>27.921937</td>
          <td>0.524085</td>
          <td>27.185244</td>
          <td>0.270632</td>
          <td>26.375154</td>
          <td>0.220951</td>
          <td>25.405986</td>
          <td>0.179603</td>
          <td>24.998461</td>
          <td>0.276636</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.661627</td>
          <td>0.191059</td>
          <td>26.132435</td>
          <td>0.100757</td>
          <td>25.863684</td>
          <td>0.069995</td>
          <td>25.715342</td>
          <td>0.100039</td>
          <td>25.555357</td>
          <td>0.164263</td>
          <td>24.630321</td>
          <td>0.163464</td>
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
          <td>27.720135</td>
          <td>0.928913</td>
          <td>26.402782</td>
          <td>0.136283</td>
          <td>25.372336</td>
          <td>0.048951</td>
          <td>25.074316</td>
          <td>0.061636</td>
          <td>24.987815</td>
          <td>0.108546</td>
          <td>24.772218</td>
          <td>0.199134</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.009372</td>
          <td>2.663931</td>
          <td>26.860762</td>
          <td>0.191015</td>
          <td>26.031071</td>
          <td>0.082392</td>
          <td>25.094874</td>
          <td>0.058771</td>
          <td>24.831666</td>
          <td>0.088963</td>
          <td>24.187549</td>
          <td>0.113279</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.643253</td>
          <td>1.530495</td>
          <td>26.444438</td>
          <td>0.137671</td>
          <td>26.508827</td>
          <td>0.129236</td>
          <td>26.338713</td>
          <td>0.179928</td>
          <td>26.469911</td>
          <td>0.363876</td>
          <td>25.479274</td>
          <td>0.344614</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.032722</td>
          <td>0.280247</td>
          <td>26.308184</td>
          <td>0.129599</td>
          <td>26.175874</td>
          <td>0.103231</td>
          <td>25.832022</td>
          <td>0.124557</td>
          <td>25.715635</td>
          <td>0.209699</td>
          <td>25.278660</td>
          <td>0.312458</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.386216</td>
          <td>0.733640</td>
          <td>26.545170</td>
          <td>0.148839</td>
          <td>26.384120</td>
          <td>0.114829</td>
          <td>26.534753</td>
          <td>0.210102</td>
          <td>25.720260</td>
          <td>0.195844</td>
          <td>25.243769</td>
          <td>0.282788</td>
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
