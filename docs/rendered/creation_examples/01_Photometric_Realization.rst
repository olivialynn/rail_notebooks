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

    <pzflow.flow.Flow at 0x7f6dd00a0fd0>



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
    0      23.994413  0.133408  0.095796  
    1      25.391064  0.048508  0.028169  
    2      24.304707  0.040548  0.021524  
    3      25.291103  0.056526  0.050287  
    4      25.096743  0.056268  0.041998  
    ...          ...       ...       ...  
    99995  24.737946  0.030357  0.030144  
    99996  24.224169  0.026796  0.013805  
    99997  25.613836  0.210639  0.188076  
    99998  25.274899  0.094940  0.080726  
    99999  25.699642  0.043909  0.042424  
    
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
          <td>27.417460</td>
          <td>0.733755</td>
          <td>26.654292</td>
          <td>0.158102</td>
          <td>26.003306</td>
          <td>0.079077</td>
          <td>25.242028</td>
          <td>0.065803</td>
          <td>24.645925</td>
          <td>0.074282</td>
          <td>23.854567</td>
          <td>0.083162</td>
          <td>0.133408</td>
          <td>0.095796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.134120</td>
          <td>1.143873</td>
          <td>27.660325</td>
          <td>0.361691</td>
          <td>26.765588</td>
          <td>0.153680</td>
          <td>26.245501</td>
          <td>0.158151</td>
          <td>25.782933</td>
          <td>0.198905</td>
          <td>25.076999</td>
          <td>0.237608</td>
          <td>0.048508</td>
          <td>0.028169</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.563609</td>
          <td>2.069555</td>
          <td>27.861882</td>
          <td>0.378321</td>
          <td>25.882965</td>
          <td>0.115643</td>
          <td>25.121296</td>
          <td>0.112786</td>
          <td>24.137290</td>
          <td>0.106589</td>
          <td>0.040548</td>
          <td>0.021524</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.123933</td>
          <td>1.137252</td>
          <td>29.683544</td>
          <td>1.374478</td>
          <td>27.302756</td>
          <td>0.241568</td>
          <td>26.165866</td>
          <td>0.147714</td>
          <td>25.460546</td>
          <td>0.151257</td>
          <td>25.538164</td>
          <td>0.344935</td>
          <td>0.056526</td>
          <td>0.050287</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.257822</td>
          <td>0.311515</td>
          <td>26.106989</td>
          <td>0.098417</td>
          <td>25.811421</td>
          <td>0.066733</td>
          <td>25.729179</td>
          <td>0.101108</td>
          <td>25.406085</td>
          <td>0.144344</td>
          <td>24.980146</td>
          <td>0.219263</td>
          <td>0.056268</td>
          <td>0.041998</td>
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
          <td>26.654896</td>
          <td>0.424727</td>
          <td>26.480220</td>
          <td>0.136150</td>
          <td>25.451615</td>
          <td>0.048493</td>
          <td>25.041657</td>
          <td>0.055087</td>
          <td>25.023488</td>
          <td>0.103552</td>
          <td>24.569032</td>
          <td>0.154897</td>
          <td>0.030357</td>
          <td>0.030144</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.591553</td>
          <td>0.822708</td>
          <td>26.599730</td>
          <td>0.150887</td>
          <td>25.928101</td>
          <td>0.073994</td>
          <td>25.245793</td>
          <td>0.066023</td>
          <td>24.654452</td>
          <td>0.074844</td>
          <td>24.247704</td>
          <td>0.117362</td>
          <td>0.026796</td>
          <td>0.013805</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.640552</td>
          <td>0.420110</td>
          <td>27.057178</td>
          <td>0.222104</td>
          <td>26.378426</td>
          <td>0.109930</td>
          <td>26.454039</td>
          <td>0.188817</td>
          <td>25.936772</td>
          <td>0.226185</td>
          <td>25.358125</td>
          <td>0.298846</td>
          <td>0.210639</td>
          <td>0.188076</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.427262</td>
          <td>0.356224</td>
          <td>26.143994</td>
          <td>0.101656</td>
          <td>26.100896</td>
          <td>0.086183</td>
          <td>25.796572</td>
          <td>0.107248</td>
          <td>25.796309</td>
          <td>0.201153</td>
          <td>24.879566</td>
          <td>0.201577</td>
          <td>0.094940</td>
          <td>0.080726</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.436496</td>
          <td>2.142865</td>
          <td>26.800484</td>
          <td>0.179046</td>
          <td>26.704815</td>
          <td>0.145869</td>
          <td>26.478865</td>
          <td>0.192812</td>
          <td>25.742099</td>
          <td>0.192186</td>
          <td>25.254567</td>
          <td>0.274837</td>
          <td>0.043909</td>
          <td>0.042424</td>
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
          <td>26.591851</td>
          <td>0.178919</td>
          <td>26.103933</td>
          <td>0.106090</td>
          <td>25.070314</td>
          <td>0.070125</td>
          <td>24.546623</td>
          <td>0.083640</td>
          <td>24.003304</td>
          <td>0.117038</td>
          <td>0.133408</td>
          <td>0.095796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.127088</td>
          <td>0.270163</td>
          <td>26.848489</td>
          <td>0.193773</td>
          <td>26.178657</td>
          <td>0.176834</td>
          <td>26.347216</td>
          <td>0.367107</td>
          <td>25.540947</td>
          <td>0.402722</td>
          <td>0.048508</td>
          <td>0.028169</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.105452</td>
          <td>1.095350</td>
          <td>28.797447</td>
          <td>0.842454</td>
          <td>26.007185</td>
          <td>0.152507</td>
          <td>25.072358</td>
          <td>0.127255</td>
          <td>24.273153</td>
          <td>0.141947</td>
          <td>0.040548</td>
          <td>0.021524</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.508420</td>
          <td>0.758734</td>
          <td>27.603128</td>
          <td>0.359872</td>
          <td>26.218240</td>
          <td>0.183655</td>
          <td>25.456912</td>
          <td>0.178022</td>
          <td>24.870288</td>
          <td>0.236529</td>
          <td>0.056526</td>
          <td>0.050287</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.345995</td>
          <td>0.373165</td>
          <td>26.347397</td>
          <td>0.140747</td>
          <td>26.006420</td>
          <td>0.094026</td>
          <td>25.788522</td>
          <td>0.126919</td>
          <td>25.648974</td>
          <td>0.209014</td>
          <td>24.749088</td>
          <td>0.213598</td>
          <td>0.056268</td>
          <td>0.041998</td>
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
          <td>26.847226</td>
          <td>0.542150</td>
          <td>26.556066</td>
          <td>0.167501</td>
          <td>25.498039</td>
          <td>0.059703</td>
          <td>25.119068</td>
          <td>0.070180</td>
          <td>24.830848</td>
          <td>0.103075</td>
          <td>24.503800</td>
          <td>0.172829</td>
          <td>0.030357</td>
          <td>0.030144</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.519169</td>
          <td>0.856260</td>
          <td>26.806884</td>
          <td>0.206720</td>
          <td>25.995846</td>
          <td>0.092527</td>
          <td>25.201183</td>
          <td>0.075341</td>
          <td>24.752880</td>
          <td>0.096121</td>
          <td>24.278393</td>
          <td>0.142297</td>
          <td>0.026796</td>
          <td>0.013805</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.270604</td>
          <td>0.380084</td>
          <td>26.773440</td>
          <td>0.222643</td>
          <td>26.245375</td>
          <td>0.129329</td>
          <td>26.300957</td>
          <td>0.219343</td>
          <td>25.616242</td>
          <td>0.225982</td>
          <td>26.465292</td>
          <td>0.850462</td>
          <td>0.210639</td>
          <td>0.188076</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.007749</td>
          <td>0.288861</td>
          <td>26.442900</td>
          <td>0.155139</td>
          <td>26.108218</td>
          <td>0.104600</td>
          <td>25.849071</td>
          <td>0.136137</td>
          <td>25.975109</td>
          <td>0.278009</td>
          <td>25.248977</td>
          <td>0.326534</td>
          <td>0.094940</td>
          <td>0.080726</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.004411</td>
          <td>0.607788</td>
          <td>27.323901</td>
          <td>0.316872</td>
          <td>26.381443</td>
          <td>0.130136</td>
          <td>26.530172</td>
          <td>0.237614</td>
          <td>25.592865</td>
          <td>0.199028</td>
          <td>26.068696</td>
          <td>0.595588</td>
          <td>0.043909</td>
          <td>0.042424</td>
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
          <td>26.747268</td>
          <td>0.195022</td>
          <td>26.115276</td>
          <td>0.101744</td>
          <td>25.199400</td>
          <td>0.074454</td>
          <td>24.669007</td>
          <td>0.088427</td>
          <td>23.839202</td>
          <td>0.096160</td>
          <td>0.133408</td>
          <td>0.095796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.772299</td>
          <td>0.400887</td>
          <td>26.520774</td>
          <td>0.127011</td>
          <td>26.195353</td>
          <td>0.154764</td>
          <td>25.791485</td>
          <td>0.204356</td>
          <td>26.531878</td>
          <td>0.727830</td>
          <td>0.048508</td>
          <td>0.028169</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.889119</td>
          <td>2.364713</td>
          <td>28.313359</td>
          <td>0.537806</td>
          <td>26.086167</td>
          <td>0.139929</td>
          <td>25.004047</td>
          <td>0.103240</td>
          <td>24.326287</td>
          <td>0.127476</td>
          <td>0.040548</td>
          <td>0.021524</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.276427</td>
          <td>0.589562</td>
          <td>27.334276</td>
          <td>0.256732</td>
          <td>26.166279</td>
          <td>0.153533</td>
          <td>25.555695</td>
          <td>0.170135</td>
          <td>25.291986</td>
          <td>0.293600</td>
          <td>0.056526</td>
          <td>0.050287</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.022383</td>
          <td>0.262987</td>
          <td>26.207424</td>
          <td>0.110496</td>
          <td>26.108526</td>
          <td>0.089620</td>
          <td>25.588388</td>
          <td>0.092437</td>
          <td>25.312671</td>
          <td>0.137464</td>
          <td>24.816397</td>
          <td>0.197386</td>
          <td>0.056268</td>
          <td>0.041998</td>
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
          <td>26.896381</td>
          <td>0.512453</td>
          <td>26.370592</td>
          <td>0.125145</td>
          <td>25.403720</td>
          <td>0.047057</td>
          <td>25.120764</td>
          <td>0.059870</td>
          <td>24.957935</td>
          <td>0.098979</td>
          <td>24.895758</td>
          <td>0.206835</td>
          <td>0.030357</td>
          <td>0.030144</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.470946</td>
          <td>0.135772</td>
          <td>26.059517</td>
          <td>0.083610</td>
          <td>25.217853</td>
          <td>0.064828</td>
          <td>24.808522</td>
          <td>0.086270</td>
          <td>24.084523</td>
          <td>0.102430</td>
          <td>0.026796</td>
          <td>0.013805</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.574086</td>
          <td>0.505844</td>
          <td>26.591432</td>
          <td>0.205325</td>
          <td>26.276537</td>
          <td>0.143773</td>
          <td>26.013813</td>
          <td>0.186504</td>
          <td>26.277416</td>
          <td>0.412682</td>
          <td>25.070604</td>
          <td>0.332824</td>
          <td>0.210639</td>
          <td>0.188076</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.887918</td>
          <td>0.245730</td>
          <td>26.163194</td>
          <td>0.112335</td>
          <td>26.282202</td>
          <td>0.111104</td>
          <td>25.915255</td>
          <td>0.131219</td>
          <td>26.011782</td>
          <td>0.263030</td>
          <td>25.524783</td>
          <td>0.372690</td>
          <td>0.094940</td>
          <td>0.080726</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.289744</td>
          <td>0.324571</td>
          <td>26.700331</td>
          <td>0.167899</td>
          <td>26.593621</td>
          <td>0.135790</td>
          <td>26.354795</td>
          <td>0.177994</td>
          <td>25.798496</td>
          <td>0.206320</td>
          <td>27.462079</td>
          <td>1.282556</td>
          <td>0.043909</td>
          <td>0.042424</td>
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
