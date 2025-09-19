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

    <pzflow.flow.Flow at 0x7f4500513340>



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
    0      23.994413  0.067302  0.050641  
    1      25.391064  0.089146  0.075692  
    2      24.304707  0.043325  0.038315  
    3      25.291103  0.031393  0.019881  
    4      25.096743  0.009267  0.007033  
    ...          ...       ...       ...  
    99995  24.737946  0.036432  0.018745  
    99996  24.224169  0.017446  0.012787  
    99997  25.613836  0.158459  0.115133  
    99998  25.274899  0.100523  0.078402  
    99999  25.699642  0.077340  0.055310  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.559688</td>
          <td>0.145790</td>
          <td>26.000592</td>
          <td>0.078888</td>
          <td>25.191245</td>
          <td>0.062906</td>
          <td>24.677222</td>
          <td>0.076365</td>
          <td>23.931775</td>
          <td>0.089012</td>
          <td>0.067302</td>
          <td>0.050641</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.559829</td>
          <td>0.805979</td>
          <td>28.042046</td>
          <td>0.484018</td>
          <td>26.533991</td>
          <td>0.125862</td>
          <td>26.108405</td>
          <td>0.140588</td>
          <td>26.133490</td>
          <td>0.265956</td>
          <td>25.832618</td>
          <td>0.433246</td>
          <td>0.089146</td>
          <td>0.075692</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.544730</td>
          <td>0.798099</td>
          <td>28.577207</td>
          <td>0.707901</td>
          <td>27.877594</td>
          <td>0.382964</td>
          <td>26.295521</td>
          <td>0.165054</td>
          <td>25.000741</td>
          <td>0.101511</td>
          <td>24.318122</td>
          <td>0.124766</td>
          <td>0.043325</td>
          <td>0.038315</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.628830</td>
          <td>0.842660</td>
          <td>29.230178</td>
          <td>1.069246</td>
          <td>27.187062</td>
          <td>0.219478</td>
          <td>26.359179</td>
          <td>0.174243</td>
          <td>25.807953</td>
          <td>0.203128</td>
          <td>25.184873</td>
          <td>0.259649</td>
          <td>0.031393</td>
          <td>0.019881</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.120732</td>
          <td>0.278979</td>
          <td>25.974868</td>
          <td>0.087645</td>
          <td>25.893333</td>
          <td>0.071753</td>
          <td>25.571181</td>
          <td>0.088011</td>
          <td>25.628023</td>
          <td>0.174504</td>
          <td>24.871918</td>
          <td>0.200287</td>
          <td>0.009267</td>
          <td>0.007033</td>
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
          <td>26.380139</td>
          <td>0.343272</td>
          <td>26.580073</td>
          <td>0.148365</td>
          <td>25.487458</td>
          <td>0.050062</td>
          <td>25.139655</td>
          <td>0.060093</td>
          <td>24.985559</td>
          <td>0.100170</td>
          <td>24.504653</td>
          <td>0.146573</td>
          <td>0.036432</td>
          <td>0.018745</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.707929</td>
          <td>0.442156</td>
          <td>26.479469</td>
          <td>0.136061</td>
          <td>25.830514</td>
          <td>0.067872</td>
          <td>25.312170</td>
          <td>0.070021</td>
          <td>24.827432</td>
          <td>0.087182</td>
          <td>24.507953</td>
          <td>0.146989</td>
          <td>0.017446</td>
          <td>0.012787</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.893456</td>
          <td>0.507723</td>
          <td>26.591530</td>
          <td>0.149830</td>
          <td>26.524632</td>
          <td>0.124845</td>
          <td>26.314025</td>
          <td>0.167677</td>
          <td>26.277197</td>
          <td>0.298804</td>
          <td>25.215912</td>
          <td>0.266319</td>
          <td>0.158459</td>
          <td>0.115133</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.544735</td>
          <td>0.390322</td>
          <td>26.252610</td>
          <td>0.111766</td>
          <td>25.905189</td>
          <td>0.072509</td>
          <td>26.167127</td>
          <td>0.147874</td>
          <td>26.473189</td>
          <td>0.349249</td>
          <td>25.789247</td>
          <td>0.419173</td>
          <td>0.100523</td>
          <td>0.078402</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.248107</td>
          <td>0.309105</td>
          <td>26.641787</td>
          <td>0.156421</td>
          <td>26.784400</td>
          <td>0.156177</td>
          <td>25.994714</td>
          <td>0.127432</td>
          <td>25.925543</td>
          <td>0.224085</td>
          <td>26.767283</td>
          <td>0.835904</td>
          <td>0.077340</td>
          <td>0.055310</td>
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
          <td>27.502534</td>
          <td>0.852103</td>
          <td>26.553131</td>
          <td>0.168378</td>
          <td>26.232880</td>
          <td>0.115028</td>
          <td>25.165319</td>
          <td>0.073785</td>
          <td>24.723812</td>
          <td>0.094678</td>
          <td>24.103078</td>
          <td>0.123581</td>
          <td>0.067302</td>
          <td>0.050641</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.265233</td>
          <td>0.353805</td>
          <td>27.402172</td>
          <td>0.341732</td>
          <td>26.681843</td>
          <td>0.171132</td>
          <td>26.227478</td>
          <td>0.187521</td>
          <td>26.073356</td>
          <td>0.300145</td>
          <td>25.310098</td>
          <td>0.341783</td>
          <td>0.089146</td>
          <td>0.075692</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.431124</td>
          <td>1.446194</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.090833</td>
          <td>1.011959</td>
          <td>26.009137</td>
          <td>0.153085</td>
          <td>25.279746</td>
          <td>0.152473</td>
          <td>24.362630</td>
          <td>0.153608</td>
          <td>0.043325</td>
          <td>0.038315</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.124704</td>
          <td>0.579630</td>
          <td>27.638147</td>
          <td>0.367467</td>
          <td>26.233672</td>
          <td>0.184718</td>
          <td>26.049340</td>
          <td>0.288941</td>
          <td>25.002096</td>
          <td>0.261752</td>
          <td>0.031393</td>
          <td>0.019881</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.925365</td>
          <td>0.265376</td>
          <td>26.134822</td>
          <td>0.116264</td>
          <td>25.853018</td>
          <td>0.081490</td>
          <td>25.532034</td>
          <td>0.100661</td>
          <td>25.106318</td>
          <td>0.130621</td>
          <td>25.151644</td>
          <td>0.294941</td>
          <td>0.009267</td>
          <td>0.007033</td>
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
          <td>27.207786</td>
          <td>0.698159</td>
          <td>26.124909</td>
          <td>0.115537</td>
          <td>25.387089</td>
          <td>0.054093</td>
          <td>25.026524</td>
          <td>0.064641</td>
          <td>24.937832</td>
          <td>0.113137</td>
          <td>24.572393</td>
          <td>0.183131</td>
          <td>0.036432</td>
          <td>0.018745</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.057621</td>
          <td>0.628796</td>
          <td>26.572369</td>
          <td>0.169497</td>
          <td>25.818441</td>
          <td>0.079088</td>
          <td>25.138045</td>
          <td>0.071195</td>
          <td>25.010410</td>
          <td>0.120267</td>
          <td>23.937724</td>
          <td>0.105802</td>
          <td>0.017446</td>
          <td>0.012787</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.099312</td>
          <td>1.245232</td>
          <td>27.112258</td>
          <td>0.279685</td>
          <td>26.283176</td>
          <td>0.126139</td>
          <td>26.437217</td>
          <td>0.232033</td>
          <td>26.451164</td>
          <td>0.417722</td>
          <td>25.430378</td>
          <td>0.388788</td>
          <td>0.158459</td>
          <td>0.115133</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.248375</td>
          <td>1.327779</td>
          <td>26.736792</td>
          <td>0.199233</td>
          <td>26.103671</td>
          <td>0.104303</td>
          <td>26.119104</td>
          <td>0.171776</td>
          <td>25.681790</td>
          <td>0.218637</td>
          <td>26.099151</td>
          <td>0.618916</td>
          <td>0.100523</td>
          <td>0.078402</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.193155</td>
          <td>0.332526</td>
          <td>27.382362</td>
          <td>0.334346</td>
          <td>26.556666</td>
          <td>0.152653</td>
          <td>26.314872</td>
          <td>0.200326</td>
          <td>26.325855</td>
          <td>0.364191</td>
          <td>25.978078</td>
          <td>0.562416</td>
          <td>0.077340</td>
          <td>0.055310</td>
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
          <td>26.648109</td>
          <td>0.163498</td>
          <td>26.083003</td>
          <td>0.088823</td>
          <td>25.138302</td>
          <td>0.063005</td>
          <td>24.644604</td>
          <td>0.077693</td>
          <td>24.067826</td>
          <td>0.105169</td>
          <td>0.067302</td>
          <td>0.050641</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.134323</td>
          <td>0.633690</td>
          <td>27.299319</td>
          <td>0.290314</td>
          <td>26.463103</td>
          <td>0.128690</td>
          <td>26.107978</td>
          <td>0.153272</td>
          <td>25.949222</td>
          <td>0.247469</td>
          <td>25.370957</td>
          <td>0.327040</td>
          <td>0.089146</td>
          <td>0.075692</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.780630</td>
          <td>0.473168</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.295199</td>
          <td>1.036097</td>
          <td>26.174537</td>
          <td>0.152244</td>
          <td>25.042938</td>
          <td>0.107680</td>
          <td>24.276087</td>
          <td>0.123070</td>
          <td>0.043325</td>
          <td>0.038315</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.531876</td>
          <td>0.690730</td>
          <td>27.106187</td>
          <td>0.206960</td>
          <td>26.399659</td>
          <td>0.182031</td>
          <td>25.456437</td>
          <td>0.152095</td>
          <td>25.939689</td>
          <td>0.473498</td>
          <td>0.031393</td>
          <td>0.019881</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.892610</td>
          <td>0.231586</td>
          <td>26.098906</td>
          <td>0.097801</td>
          <td>25.938087</td>
          <td>0.074719</td>
          <td>25.644037</td>
          <td>0.093924</td>
          <td>25.069308</td>
          <td>0.107882</td>
          <td>24.896682</td>
          <td>0.204678</td>
          <td>0.009267</td>
          <td>0.007033</td>
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
          <td>26.732445</td>
          <td>0.453429</td>
          <td>26.202839</td>
          <td>0.108063</td>
          <td>25.465144</td>
          <td>0.049641</td>
          <td>24.938424</td>
          <td>0.050866</td>
          <td>25.085568</td>
          <td>0.110551</td>
          <td>24.636108</td>
          <td>0.165907</td>
          <td>0.036432</td>
          <td>0.018745</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.383721</td>
          <td>2.099717</td>
          <td>26.836334</td>
          <td>0.185049</td>
          <td>26.025325</td>
          <td>0.080886</td>
          <td>25.205018</td>
          <td>0.063894</td>
          <td>24.881795</td>
          <td>0.091742</td>
          <td>24.360028</td>
          <td>0.129801</td>
          <td>0.017446</td>
          <td>0.012787</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.591751</td>
          <td>0.461207</td>
          <td>27.119866</td>
          <td>0.277419</td>
          <td>26.652722</td>
          <td>0.170369</td>
          <td>26.421368</td>
          <td>0.225171</td>
          <td>25.794683</td>
          <td>0.243908</td>
          <td>26.302317</td>
          <td>0.721301</td>
          <td>0.158459</td>
          <td>0.115133</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.892998</td>
          <td>0.247399</td>
          <td>26.121681</td>
          <td>0.108702</td>
          <td>26.105318</td>
          <td>0.095526</td>
          <td>25.766586</td>
          <td>0.115774</td>
          <td>25.783594</td>
          <td>0.218640</td>
          <td>25.547411</td>
          <td>0.380588</td>
          <td>0.100523</td>
          <td>0.078402</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.516656</td>
          <td>0.395809</td>
          <td>26.543359</td>
          <td>0.150999</td>
          <td>26.552959</td>
          <td>0.135401</td>
          <td>26.742728</td>
          <td>0.254216</td>
          <td>25.745228</td>
          <td>0.203574</td>
          <td>25.313185</td>
          <td>0.304389</td>
          <td>0.077340</td>
          <td>0.055310</td>
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
