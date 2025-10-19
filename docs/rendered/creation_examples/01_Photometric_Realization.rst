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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f41e23f4c40>



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
    0      23.994413  0.143131  0.106349  
    1      25.391064  0.088305  0.083048  
    2      24.304707  0.136844  0.126873  
    3      25.291103  0.203302  0.143440  
    4      25.096743  0.121614  0.078995  
    ...          ...       ...       ...  
    99995  24.737946  0.085461  0.058599  
    99996  24.224169  0.195009  0.113706  
    99997  25.613836  0.103863  0.099705  
    99998  25.274899  0.027406  0.023874  
    99999  25.699642  0.077917  0.066743  
    
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
          <td>26.884488</td>
          <td>0.504386</td>
          <td>26.692474</td>
          <td>0.163341</td>
          <td>26.104789</td>
          <td>0.086479</td>
          <td>25.242511</td>
          <td>0.065831</td>
          <td>24.708163</td>
          <td>0.078481</td>
          <td>23.905261</td>
          <td>0.086959</td>
          <td>0.143131</td>
          <td>0.106349</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.692286</td>
          <td>0.436955</td>
          <td>27.800973</td>
          <td>0.403386</td>
          <td>26.539587</td>
          <td>0.126475</td>
          <td>26.452358</td>
          <td>0.188549</td>
          <td>25.750087</td>
          <td>0.193484</td>
          <td>25.236399</td>
          <td>0.270804</td>
          <td>0.088305</td>
          <td>0.083048</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.424871</td>
          <td>1.952603</td>
          <td>27.790104</td>
          <td>0.357703</td>
          <td>25.660366</td>
          <td>0.095188</td>
          <td>24.996055</td>
          <td>0.101095</td>
          <td>24.484270</td>
          <td>0.144026</td>
          <td>0.136844</td>
          <td>0.126873</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.520025</td>
          <td>0.323782</td>
          <td>27.082497</td>
          <td>0.201101</td>
          <td>26.190732</td>
          <td>0.150902</td>
          <td>25.414632</td>
          <td>0.145409</td>
          <td>25.884980</td>
          <td>0.450747</td>
          <td>0.203302</td>
          <td>0.143440</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.172938</td>
          <td>0.291001</td>
          <td>26.094378</td>
          <td>0.097336</td>
          <td>25.967151</td>
          <td>0.076592</td>
          <td>25.758358</td>
          <td>0.103724</td>
          <td>25.704952</td>
          <td>0.186255</td>
          <td>25.281091</td>
          <td>0.280819</td>
          <td>0.121614</td>
          <td>0.078995</td>
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
          <td>26.725360</td>
          <td>0.448010</td>
          <td>26.240217</td>
          <td>0.110565</td>
          <td>25.459264</td>
          <td>0.048824</td>
          <td>25.019217</td>
          <td>0.054000</td>
          <td>24.776045</td>
          <td>0.083324</td>
          <td>24.589860</td>
          <td>0.157683</td>
          <td>0.085461</td>
          <td>0.058599</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.393331</td>
          <td>0.721976</td>
          <td>26.688318</td>
          <td>0.162763</td>
          <td>25.954173</td>
          <td>0.075719</td>
          <td>25.126982</td>
          <td>0.059421</td>
          <td>24.748224</td>
          <td>0.081305</td>
          <td>24.238384</td>
          <td>0.116414</td>
          <td>0.195009</td>
          <td>0.113706</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.033232</td>
          <td>0.561999</td>
          <td>26.478281</td>
          <td>0.135922</td>
          <td>26.299763</td>
          <td>0.102626</td>
          <td>26.184547</td>
          <td>0.150103</td>
          <td>25.470331</td>
          <td>0.152532</td>
          <td>25.271391</td>
          <td>0.278618</td>
          <td>0.103863</td>
          <td>0.099705</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.373007</td>
          <td>0.341347</td>
          <td>26.112667</td>
          <td>0.098907</td>
          <td>26.080761</td>
          <td>0.084668</td>
          <td>25.637850</td>
          <td>0.093325</td>
          <td>25.646986</td>
          <td>0.177335</td>
          <td>25.557090</td>
          <td>0.350115</td>
          <td>0.027406</td>
          <td>0.023874</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.265302</td>
          <td>1.231059</td>
          <td>26.824132</td>
          <td>0.182666</td>
          <td>26.415712</td>
          <td>0.113564</td>
          <td>26.384076</td>
          <td>0.177964</td>
          <td>26.210011</td>
          <td>0.283029</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.077917</td>
          <td>0.066743</td>
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
          <td>26.360021</td>
          <td>0.388400</td>
          <td>26.476868</td>
          <td>0.163330</td>
          <td>26.051206</td>
          <td>0.102062</td>
          <td>25.224727</td>
          <td>0.080994</td>
          <td>24.749241</td>
          <td>0.100675</td>
          <td>23.841338</td>
          <td>0.102388</td>
          <td>0.143131</td>
          <td>0.106349</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.193238</td>
          <td>0.289620</td>
          <td>26.542452</td>
          <td>0.152175</td>
          <td>26.305474</td>
          <td>0.200580</td>
          <td>25.870499</td>
          <td>0.254950</td>
          <td>25.447966</td>
          <td>0.381319</td>
          <td>0.088305</td>
          <td>0.083048</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.691935</td>
          <td>0.981913</td>
          <td>27.505024</td>
          <td>0.380461</td>
          <td>30.334603</td>
          <td>1.963718</td>
          <td>25.965540</td>
          <td>0.155135</td>
          <td>25.138026</td>
          <td>0.141794</td>
          <td>24.207363</td>
          <td>0.141385</td>
          <td>0.136844</td>
          <td>0.126873</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.441008</td>
          <td>0.341761</td>
          <td>26.359298</td>
          <td>0.224709</td>
          <td>25.303854</td>
          <td>0.169669</td>
          <td>26.315791</td>
          <td>0.757154</td>
          <td>0.203302</td>
          <td>0.143440</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.189811</td>
          <td>0.336101</td>
          <td>26.179510</td>
          <td>0.124613</td>
          <td>25.982670</td>
          <td>0.094517</td>
          <td>25.924904</td>
          <td>0.146599</td>
          <td>25.328826</td>
          <td>0.163512</td>
          <td>25.934721</td>
          <td>0.554013</td>
          <td>0.121614</td>
          <td>0.078995</td>
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
          <td>26.960529</td>
          <td>0.593446</td>
          <td>26.321963</td>
          <td>0.138875</td>
          <td>25.338871</td>
          <td>0.052625</td>
          <td>25.050970</td>
          <td>0.067105</td>
          <td>24.917035</td>
          <td>0.112778</td>
          <td>24.518533</td>
          <td>0.177596</td>
          <td>0.085461</td>
          <td>0.058599</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.255277</td>
          <td>0.364947</td>
          <td>26.379737</td>
          <td>0.154029</td>
          <td>25.798144</td>
          <td>0.083990</td>
          <td>25.239290</td>
          <td>0.084389</td>
          <td>25.015044</td>
          <td>0.130364</td>
          <td>24.262601</td>
          <td>0.151633</td>
          <td>0.195009</td>
          <td>0.113706</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.006379</td>
          <td>0.619081</td>
          <td>26.545902</td>
          <td>0.170629</td>
          <td>26.533024</td>
          <td>0.152382</td>
          <td>26.066276</td>
          <td>0.165410</td>
          <td>25.837114</td>
          <td>0.250309</td>
          <td>25.122875</td>
          <td>0.297514</td>
          <td>0.103863</td>
          <td>0.099705</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.983187</td>
          <td>0.278535</td>
          <td>26.109354</td>
          <td>0.113923</td>
          <td>26.168645</td>
          <td>0.107721</td>
          <td>25.927621</td>
          <td>0.142239</td>
          <td>26.070677</td>
          <td>0.293932</td>
          <td>25.309838</td>
          <td>0.335302</td>
          <td>0.027406</td>
          <td>0.023874</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.244718</td>
          <td>0.722014</td>
          <td>26.727787</td>
          <td>0.196148</td>
          <td>26.859192</td>
          <td>0.197838</td>
          <td>26.227984</td>
          <td>0.186645</td>
          <td>25.728436</td>
          <td>0.225300</td>
          <td>26.821394</td>
          <td>0.985810</td>
          <td>0.077917</td>
          <td>0.066743</td>
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
          <td>28.262862</td>
          <td>1.330868</td>
          <td>26.669791</td>
          <td>0.186379</td>
          <td>26.014463</td>
          <td>0.095311</td>
          <td>25.181415</td>
          <td>0.075065</td>
          <td>24.693769</td>
          <td>0.092483</td>
          <td>23.929420</td>
          <td>0.106551</td>
          <td>0.143131</td>
          <td>0.106349</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.836121</td>
          <td>0.513754</td>
          <td>27.687749</td>
          <td>0.396383</td>
          <td>26.601522</td>
          <td>0.145860</td>
          <td>26.504717</td>
          <td>0.215693</td>
          <td>26.029037</td>
          <td>0.265651</td>
          <td>25.644056</td>
          <td>0.407041</td>
          <td>0.088305</td>
          <td>0.083048</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.870401</td>
          <td>1.803146</td>
          <td>29.758440</td>
          <td>1.573928</td>
          <td>28.332997</td>
          <td>0.630015</td>
          <td>25.710354</td>
          <td>0.121326</td>
          <td>24.959090</td>
          <td>0.118461</td>
          <td>24.249757</td>
          <td>0.142951</td>
          <td>0.136844</td>
          <td>0.126873</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.365777</td>
          <td>1.363373</td>
          <td>27.221400</td>
          <td>0.297420</td>
          <td>26.491797</td>
          <td>0.260425</td>
          <td>25.546704</td>
          <td>0.216293</td>
          <td>26.313755</td>
          <td>0.779524</td>
          <td>0.203302</td>
          <td>0.143440</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.198208</td>
          <td>0.321975</td>
          <td>25.993161</td>
          <td>0.099273</td>
          <td>26.022578</td>
          <td>0.090997</td>
          <td>25.679150</td>
          <td>0.109971</td>
          <td>25.560185</td>
          <td>0.185472</td>
          <td>25.126172</td>
          <td>0.278491</td>
          <td>0.121614</td>
          <td>0.078995</td>
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
          <td>26.443328</td>
          <td>0.139753</td>
          <td>25.488551</td>
          <td>0.053651</td>
          <td>25.033883</td>
          <td>0.058756</td>
          <td>24.806625</td>
          <td>0.091574</td>
          <td>24.575964</td>
          <td>0.166766</td>
          <td>0.085461</td>
          <td>0.058599</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.375686</td>
          <td>0.821413</td>
          <td>27.008029</td>
          <td>0.263195</td>
          <td>25.921253</td>
          <td>0.094539</td>
          <td>25.179706</td>
          <td>0.080888</td>
          <td>24.782371</td>
          <td>0.107561</td>
          <td>24.152368</td>
          <td>0.139292</td>
          <td>0.195009</td>
          <td>0.113706</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.836053</td>
          <td>0.524166</td>
          <td>26.650655</td>
          <td>0.175140</td>
          <td>26.323248</td>
          <td>0.118423</td>
          <td>26.460742</td>
          <td>0.214761</td>
          <td>26.034908</td>
          <td>0.275148</td>
          <td>25.726142</td>
          <td>0.446160</td>
          <td>0.103863</td>
          <td>0.099705</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.202374</td>
          <td>0.299665</td>
          <td>26.104241</td>
          <td>0.098940</td>
          <td>25.973435</td>
          <td>0.077711</td>
          <td>25.731809</td>
          <td>0.102290</td>
          <td>25.663943</td>
          <td>0.181450</td>
          <td>24.955560</td>
          <td>0.216707</td>
          <td>0.027406</td>
          <td>0.023874</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.861585</td>
          <td>1.006381</td>
          <td>26.635907</td>
          <td>0.164703</td>
          <td>26.356689</td>
          <td>0.115237</td>
          <td>26.337709</td>
          <td>0.183001</td>
          <td>25.836625</td>
          <td>0.221629</td>
          <td>26.003925</td>
          <td>0.521899</td>
          <td>0.077917</td>
          <td>0.066743</td>
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
