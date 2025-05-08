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

    <pzflow.flow.Flow at 0x7fde69ea5630>



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
          <td>27.720537</td>
          <td>0.893096</td>
          <td>26.750316</td>
          <td>0.171585</td>
          <td>26.186303</td>
          <td>0.092907</td>
          <td>25.155169</td>
          <td>0.060925</td>
          <td>24.691971</td>
          <td>0.077367</td>
          <td>24.060450</td>
          <td>0.099658</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.011754</td>
          <td>0.473226</td>
          <td>26.539432</td>
          <td>0.126458</td>
          <td>26.298996</td>
          <td>0.165543</td>
          <td>25.583540</td>
          <td>0.168025</td>
          <td>25.238697</td>
          <td>0.271311</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.489878</td>
          <td>2.007109</td>
          <td>27.870248</td>
          <td>0.380787</td>
          <td>25.838629</td>
          <td>0.111259</td>
          <td>25.275484</td>
          <td>0.128955</td>
          <td>24.216811</td>
          <td>0.114248</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.559320</td>
          <td>1.438800</td>
          <td>27.841102</td>
          <td>0.415989</td>
          <td>27.413010</td>
          <td>0.264449</td>
          <td>26.689017</td>
          <td>0.229845</td>
          <td>25.784337</td>
          <td>0.199140</td>
          <td>25.357133</td>
          <td>0.298608</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.904944</td>
          <td>0.233819</td>
          <td>26.192805</td>
          <td>0.106086</td>
          <td>26.011455</td>
          <td>0.079648</td>
          <td>25.584163</td>
          <td>0.089022</td>
          <td>25.314788</td>
          <td>0.133415</td>
          <td>24.972496</td>
          <td>0.217870</td>
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
          <td>28.101926</td>
          <td>1.123021</td>
          <td>26.270984</td>
          <td>0.113569</td>
          <td>25.430323</td>
          <td>0.047585</td>
          <td>25.030399</td>
          <td>0.054539</td>
          <td>24.839639</td>
          <td>0.088124</td>
          <td>24.789635</td>
          <td>0.186876</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.857923</td>
          <td>2.514610</td>
          <td>27.078477</td>
          <td>0.226069</td>
          <td>26.054406</td>
          <td>0.082724</td>
          <td>25.179565</td>
          <td>0.062258</td>
          <td>24.848767</td>
          <td>0.088834</td>
          <td>24.193038</td>
          <td>0.111905</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.918531</td>
          <td>0.517147</td>
          <td>26.860625</td>
          <td>0.188385</td>
          <td>26.643810</td>
          <td>0.138403</td>
          <td>26.371750</td>
          <td>0.176113</td>
          <td>27.068326</td>
          <td>0.547815</td>
          <td>25.926378</td>
          <td>0.464983</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.995342</td>
          <td>0.546863</td>
          <td>26.008582</td>
          <td>0.090280</td>
          <td>26.045231</td>
          <td>0.082057</td>
          <td>25.927559</td>
          <td>0.120217</td>
          <td>25.417169</td>
          <td>0.145726</td>
          <td>24.921482</td>
          <td>0.208783</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.260262</td>
          <td>0.312123</td>
          <td>26.912130</td>
          <td>0.196736</td>
          <td>26.380171</td>
          <td>0.110098</td>
          <td>26.232423</td>
          <td>0.156391</td>
          <td>25.579243</td>
          <td>0.167412</td>
          <td>26.219581</td>
          <td>0.576289</td>
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
          <td>27.688724</td>
          <td>0.951204</td>
          <td>27.509671</td>
          <td>0.365080</td>
          <td>25.906896</td>
          <td>0.085435</td>
          <td>25.143837</td>
          <td>0.071504</td>
          <td>24.693061</td>
          <td>0.091065</td>
          <td>23.814855</td>
          <td>0.094935</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.172396</td>
          <td>0.279107</td>
          <td>27.088630</td>
          <td>0.235638</td>
          <td>26.245636</td>
          <td>0.186203</td>
          <td>26.013155</td>
          <td>0.280056</td>
          <td>25.018914</td>
          <td>0.264839</td>
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
          <td>28.703297</td>
          <td>0.803767</td>
          <td>26.182646</td>
          <td>0.180469</td>
          <td>24.905173</td>
          <td>0.112108</td>
          <td>24.229255</td>
          <td>0.139286</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.844594</td>
          <td>1.081577</td>
          <td>28.977160</td>
          <td>1.055959</td>
          <td>27.096854</td>
          <td>0.252602</td>
          <td>26.254099</td>
          <td>0.200387</td>
          <td>25.922476</td>
          <td>0.276761</td>
          <td>25.313059</td>
          <td>0.356839</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.846860</td>
          <td>0.248900</td>
          <td>26.000454</td>
          <td>0.103429</td>
          <td>25.848825</td>
          <td>0.081200</td>
          <td>25.872070</td>
          <td>0.135327</td>
          <td>25.728600</td>
          <td>0.221687</td>
          <td>24.866789</td>
          <td>0.233734</td>
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
          <td>26.408323</td>
          <td>0.149986</td>
          <td>25.405630</td>
          <td>0.056012</td>
          <td>25.084214</td>
          <td>0.069334</td>
          <td>25.052210</td>
          <td>0.127224</td>
          <td>24.767049</td>
          <td>0.219521</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.996045</td>
          <td>1.143178</td>
          <td>26.884384</td>
          <td>0.221030</td>
          <td>26.235590</td>
          <td>0.114420</td>
          <td>25.209302</td>
          <td>0.076094</td>
          <td>24.913323</td>
          <td>0.110894</td>
          <td>24.325635</td>
          <td>0.148592</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.829784</td>
          <td>0.212764</td>
          <td>26.359743</td>
          <td>0.128536</td>
          <td>26.219445</td>
          <td>0.184422</td>
          <td>26.627164</td>
          <td>0.457926</td>
          <td>25.463626</td>
          <td>0.381946</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.908911</td>
          <td>0.267712</td>
          <td>26.118251</td>
          <td>0.117852</td>
          <td>25.909307</td>
          <td>0.088358</td>
          <td>25.649066</td>
          <td>0.115131</td>
          <td>25.425671</td>
          <td>0.177050</td>
          <td>24.654094</td>
          <td>0.201785</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.829404</td>
          <td>0.537537</td>
          <td>26.556152</td>
          <td>0.168536</td>
          <td>26.814191</td>
          <td>0.189110</td>
          <td>26.340824</td>
          <td>0.203712</td>
          <td>25.548372</td>
          <td>0.192406</td>
          <td>25.612305</td>
          <td>0.427127</td>
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
          <td>26.794474</td>
          <td>0.178156</td>
          <td>26.000235</td>
          <td>0.078873</td>
          <td>25.185380</td>
          <td>0.062589</td>
          <td>24.680354</td>
          <td>0.076587</td>
          <td>24.014208</td>
          <td>0.095711</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.071790</td>
          <td>0.224991</td>
          <td>26.538666</td>
          <td>0.126492</td>
          <td>26.044407</td>
          <td>0.133163</td>
          <td>25.892533</td>
          <td>0.218208</td>
          <td>25.457832</td>
          <td>0.323955</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.667790</td>
          <td>0.451071</td>
          <td>30.626353</td>
          <td>2.193759</td>
          <td>28.001274</td>
          <td>0.451990</td>
          <td>25.949154</td>
          <td>0.133276</td>
          <td>24.779001</td>
          <td>0.090681</td>
          <td>24.461602</td>
          <td>0.153448</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.781398</td>
          <td>0.537174</td>
          <td>27.726004</td>
          <td>0.453228</td>
          <td>27.636016</td>
          <td>0.387280</td>
          <td>26.284715</td>
          <td>0.204878</td>
          <td>25.550230</td>
          <td>0.202826</td>
          <td>24.988681</td>
          <td>0.274446</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.519254</td>
          <td>0.383034</td>
          <td>26.311402</td>
          <td>0.117776</td>
          <td>26.180343</td>
          <td>0.092553</td>
          <td>25.797235</td>
          <td>0.107470</td>
          <td>25.485536</td>
          <td>0.154746</td>
          <td>25.015222</td>
          <td>0.226069</td>
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
          <td>26.767152</td>
          <td>0.484734</td>
          <td>26.233110</td>
          <td>0.117664</td>
          <td>25.371569</td>
          <td>0.048918</td>
          <td>25.078976</td>
          <td>0.061891</td>
          <td>24.824513</td>
          <td>0.094085</td>
          <td>25.116923</td>
          <td>0.264977</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.339307</td>
          <td>1.290837</td>
          <td>26.591391</td>
          <td>0.151922</td>
          <td>25.951095</td>
          <td>0.076776</td>
          <td>25.167344</td>
          <td>0.062673</td>
          <td>24.691021</td>
          <td>0.078593</td>
          <td>24.217000</td>
          <td>0.116222</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.920959</td>
          <td>0.206451</td>
          <td>26.372752</td>
          <td>0.114832</td>
          <td>26.323380</td>
          <td>0.177604</td>
          <td>25.945858</td>
          <td>0.238630</td>
          <td>25.994901</td>
          <td>0.510619</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.395591</td>
          <td>0.373859</td>
          <td>26.087896</td>
          <td>0.107026</td>
          <td>26.418684</td>
          <td>0.127536</td>
          <td>26.085566</td>
          <td>0.154987</td>
          <td>25.701127</td>
          <td>0.207169</td>
          <td>24.788908</td>
          <td>0.209215</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.716030</td>
          <td>0.172217</td>
          <td>26.481238</td>
          <td>0.124944</td>
          <td>26.236874</td>
          <td>0.163341</td>
          <td>25.818578</td>
          <td>0.212671</td>
          <td>25.457842</td>
          <td>0.335689</td>
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
