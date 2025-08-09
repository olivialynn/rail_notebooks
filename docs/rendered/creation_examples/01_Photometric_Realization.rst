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

    <pzflow.flow.Flow at 0x7fc3fe056b30>



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
    0      23.994413  0.090240  0.054313  
    1      25.391064  0.150932  0.112526  
    2      24.304707  0.001330  0.000852  
    3      25.291103  0.002432  0.001411  
    4      25.096743  0.001566  0.000812  
    ...          ...       ...       ...  
    99995  24.737946  0.021929  0.018797  
    99996  24.224169  0.029033  0.028874  
    99997  25.613836  0.021656  0.020931  
    99998  25.274899  0.043100  0.032587  
    99999  25.699642  0.014722  0.010327  
    
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
          <td>27.871578</td>
          <td>0.980313</td>
          <td>26.774079</td>
          <td>0.175083</td>
          <td>25.984104</td>
          <td>0.077748</td>
          <td>25.230909</td>
          <td>0.065158</td>
          <td>24.698080</td>
          <td>0.077785</td>
          <td>24.202896</td>
          <td>0.112871</td>
          <td>0.090240</td>
          <td>0.054313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.898500</td>
          <td>0.996394</td>
          <td>27.304176</td>
          <td>0.272153</td>
          <td>26.716264</td>
          <td>0.147311</td>
          <td>26.309061</td>
          <td>0.166970</td>
          <td>26.315845</td>
          <td>0.308221</td>
          <td>25.308017</td>
          <td>0.287009</td>
          <td>0.150932</td>
          <td>0.112526</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.738478</td>
          <td>0.343467</td>
          <td>26.121862</td>
          <td>0.142228</td>
          <td>25.043840</td>
          <td>0.105412</td>
          <td>24.273063</td>
          <td>0.119979</td>
          <td>0.001330</td>
          <td>0.000852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.048052</td>
          <td>2.687382</td>
          <td>29.499272</td>
          <td>1.245425</td>
          <td>27.347938</td>
          <td>0.250722</td>
          <td>26.566702</td>
          <td>0.207575</td>
          <td>25.619111</td>
          <td>0.173188</td>
          <td>25.547344</td>
          <td>0.347439</td>
          <td>0.002432</td>
          <td>0.001411</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.300490</td>
          <td>0.322293</td>
          <td>26.100547</td>
          <td>0.097863</td>
          <td>25.907285</td>
          <td>0.072644</td>
          <td>25.638538</td>
          <td>0.093381</td>
          <td>25.439160</td>
          <td>0.148506</td>
          <td>25.391652</td>
          <td>0.307002</td>
          <td>0.001566</td>
          <td>0.000812</td>
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
          <td>27.058042</td>
          <td>0.572082</td>
          <td>26.430363</td>
          <td>0.130412</td>
          <td>25.449282</td>
          <td>0.048393</td>
          <td>25.146481</td>
          <td>0.060458</td>
          <td>24.720862</td>
          <td>0.079365</td>
          <td>24.955757</td>
          <td>0.214849</td>
          <td>0.021929</td>
          <td>0.018797</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.347607</td>
          <td>0.700024</td>
          <td>26.698119</td>
          <td>0.164129</td>
          <td>26.130524</td>
          <td>0.088460</td>
          <td>25.383278</td>
          <td>0.074567</td>
          <td>24.782560</td>
          <td>0.083804</td>
          <td>24.238050</td>
          <td>0.116380</td>
          <td>0.029033</td>
          <td>0.028874</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.732724</td>
          <td>0.450502</td>
          <td>26.272170</td>
          <td>0.113686</td>
          <td>26.280990</td>
          <td>0.100953</td>
          <td>26.172677</td>
          <td>0.148581</td>
          <td>25.719303</td>
          <td>0.188526</td>
          <td>25.404290</td>
          <td>0.310126</td>
          <td>0.021656</td>
          <td>0.020931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.137478</td>
          <td>0.282786</td>
          <td>26.242577</td>
          <td>0.110793</td>
          <td>26.109699</td>
          <td>0.086854</td>
          <td>25.701342</td>
          <td>0.098672</td>
          <td>25.377463</td>
          <td>0.140830</td>
          <td>24.936658</td>
          <td>0.211450</td>
          <td>0.043100</td>
          <td>0.032587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.485377</td>
          <td>0.372765</td>
          <td>26.624665</td>
          <td>0.154145</td>
          <td>26.640098</td>
          <td>0.137960</td>
          <td>26.108484</td>
          <td>0.140598</td>
          <td>25.824966</td>
          <td>0.206045</td>
          <td>25.640328</td>
          <td>0.373688</td>
          <td>0.014722</td>
          <td>0.010327</td>
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
          <td>26.803110</td>
          <td>0.209090</td>
          <td>25.856928</td>
          <td>0.083284</td>
          <td>25.078972</td>
          <td>0.068828</td>
          <td>24.898810</td>
          <td>0.111062</td>
          <td>24.084939</td>
          <td>0.122461</td>
          <td>0.090240</td>
          <td>0.054313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.001753</td>
          <td>0.625766</td>
          <td>27.335502</td>
          <td>0.333294</td>
          <td>26.620150</td>
          <td>0.167800</td>
          <td>26.557364</td>
          <td>0.255115</td>
          <td>26.050338</td>
          <td>0.303908</td>
          <td>25.601668</td>
          <td>0.441526</td>
          <td>0.150932</td>
          <td>0.112526</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.525755</td>
          <td>0.335682</td>
          <td>26.057671</td>
          <td>0.158670</td>
          <td>25.130192</td>
          <td>0.133316</td>
          <td>24.221916</td>
          <td>0.135324</td>
          <td>0.001330</td>
          <td>0.000852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.333905</td>
          <td>1.243436</td>
          <td>27.236553</td>
          <td>0.266030</td>
          <td>26.146494</td>
          <td>0.171156</td>
          <td>25.647410</td>
          <td>0.207096</td>
          <td>25.567696</td>
          <td>0.409125</td>
          <td>0.002432</td>
          <td>0.001411</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.967783</td>
          <td>0.274646</td>
          <td>26.159540</td>
          <td>0.118764</td>
          <td>25.917474</td>
          <td>0.086232</td>
          <td>25.717741</td>
          <td>0.118344</td>
          <td>25.468020</td>
          <td>0.178042</td>
          <td>25.088612</td>
          <td>0.280230</td>
          <td>0.001566</td>
          <td>0.000812</td>
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
          <td>26.641706</td>
          <td>0.465523</td>
          <td>26.717343</td>
          <td>0.191728</td>
          <td>25.335365</td>
          <td>0.051592</td>
          <td>25.032931</td>
          <td>0.064912</td>
          <td>24.955702</td>
          <td>0.114749</td>
          <td>24.603436</td>
          <td>0.187733</td>
          <td>0.021929</td>
          <td>0.018797</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.196045</td>
          <td>0.330481</td>
          <td>27.071375</td>
          <td>0.257615</td>
          <td>26.097140</td>
          <td>0.101253</td>
          <td>25.126250</td>
          <td>0.070608</td>
          <td>24.713188</td>
          <td>0.092951</td>
          <td>24.469041</td>
          <td>0.167751</td>
          <td>0.029033</td>
          <td>0.028874</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.046206</td>
          <td>0.624079</td>
          <td>26.584543</td>
          <td>0.171372</td>
          <td>26.558028</td>
          <td>0.150816</td>
          <td>26.988892</td>
          <td>0.342747</td>
          <td>26.016407</td>
          <td>0.281134</td>
          <td>25.303043</td>
          <td>0.333285</td>
          <td>0.021656</td>
          <td>0.020931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.170032</td>
          <td>0.324198</td>
          <td>26.525964</td>
          <td>0.163524</td>
          <td>26.007218</td>
          <td>0.093774</td>
          <td>26.137379</td>
          <td>0.170677</td>
          <td>25.921701</td>
          <td>0.261092</td>
          <td>26.029072</td>
          <td>0.578382</td>
          <td>0.043100</td>
          <td>0.032587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.461174</td>
          <td>0.824569</td>
          <td>26.859883</td>
          <td>0.215891</td>
          <td>26.461456</td>
          <td>0.138661</td>
          <td>26.332572</td>
          <td>0.200412</td>
          <td>27.181857</td>
          <td>0.675619</td>
          <td>25.495578</td>
          <td>0.387197</td>
          <td>0.014722</td>
          <td>0.010327</td>
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
          <td>26.748992</td>
          <td>0.475472</td>
          <td>26.744323</td>
          <td>0.180985</td>
          <td>26.030612</td>
          <td>0.086822</td>
          <td>25.137637</td>
          <td>0.064531</td>
          <td>24.927633</td>
          <td>0.101997</td>
          <td>24.090528</td>
          <td>0.109871</td>
          <td>0.090240</td>
          <td>0.054313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.571824</td>
          <td>3.328550</td>
          <td>27.459807</td>
          <td>0.360132</td>
          <td>26.656792</td>
          <td>0.168925</td>
          <td>26.377921</td>
          <td>0.214565</td>
          <td>26.046640</td>
          <td>0.296082</td>
          <td>25.853076</td>
          <td>0.520760</td>
          <td>0.150932</td>
          <td>0.112526</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.085136</td>
          <td>0.980698</td>
          <td>27.728803</td>
          <td>0.340859</td>
          <td>26.142025</td>
          <td>0.144720</td>
          <td>25.120690</td>
          <td>0.112728</td>
          <td>24.367801</td>
          <td>0.130256</td>
          <td>0.001330</td>
          <td>0.000852</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.281803</td>
          <td>0.669293</td>
          <td>29.328615</td>
          <td>1.131974</td>
          <td>26.967893</td>
          <td>0.182591</td>
          <td>26.445963</td>
          <td>0.187544</td>
          <td>25.357950</td>
          <td>0.138488</td>
          <td>25.306904</td>
          <td>0.286766</td>
          <td>0.002432</td>
          <td>0.001411</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.868950</td>
          <td>0.226962</td>
          <td>26.146155</td>
          <td>0.101850</td>
          <td>25.964793</td>
          <td>0.076434</td>
          <td>25.793467</td>
          <td>0.106960</td>
          <td>25.337971</td>
          <td>0.136117</td>
          <td>25.152855</td>
          <td>0.252929</td>
          <td>0.001566</td>
          <td>0.000812</td>
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
          <td>30.617472</td>
          <td>3.223046</td>
          <td>26.132116</td>
          <td>0.101097</td>
          <td>25.433792</td>
          <td>0.048005</td>
          <td>25.004963</td>
          <td>0.053641</td>
          <td>24.877436</td>
          <td>0.091617</td>
          <td>24.851856</td>
          <td>0.198045</td>
          <td>0.021929</td>
          <td>0.018797</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.811365</td>
          <td>0.182414</td>
          <td>26.158706</td>
          <td>0.091709</td>
          <td>25.209221</td>
          <td>0.064685</td>
          <td>24.833901</td>
          <td>0.088675</td>
          <td>24.344048</td>
          <td>0.129083</td>
          <td>0.029033</td>
          <td>0.028874</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.187743</td>
          <td>1.182284</td>
          <td>26.743454</td>
          <td>0.171467</td>
          <td>26.294401</td>
          <td>0.102771</td>
          <td>26.005904</td>
          <td>0.129492</td>
          <td>26.439300</td>
          <td>0.341913</td>
          <td>25.994685</td>
          <td>0.491890</td>
          <td>0.021656</td>
          <td>0.020931</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.352325</td>
          <td>0.339926</td>
          <td>25.935493</td>
          <td>0.086109</td>
          <td>26.148611</td>
          <td>0.091644</td>
          <td>25.915808</td>
          <td>0.121419</td>
          <td>25.485093</td>
          <td>0.157421</td>
          <td>24.980172</td>
          <td>0.223495</td>
          <td>0.043100</td>
          <td>0.032587</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.502427</td>
          <td>0.378246</td>
          <td>26.876734</td>
          <td>0.191310</td>
          <td>26.656905</td>
          <td>0.140276</td>
          <td>26.118893</td>
          <td>0.142185</td>
          <td>27.094597</td>
          <td>0.559298</td>
          <td>26.150598</td>
          <td>0.549444</td>
          <td>0.014722</td>
          <td>0.010327</td>
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
