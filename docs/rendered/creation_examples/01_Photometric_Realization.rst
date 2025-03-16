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

    <pzflow.flow.Flow at 0x7f34194bc340>



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
          <td>27.255947</td>
          <td>0.657462</td>
          <td>26.758504</td>
          <td>0.172783</td>
          <td>26.014163</td>
          <td>0.079838</td>
          <td>25.120749</td>
          <td>0.059093</td>
          <td>24.569677</td>
          <td>0.069437</td>
          <td>23.881270</td>
          <td>0.085142</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.460776</td>
          <td>0.755235</td>
          <td>26.970448</td>
          <td>0.206600</td>
          <td>26.755916</td>
          <td>0.152411</td>
          <td>26.185628</td>
          <td>0.150242</td>
          <td>25.344675</td>
          <td>0.136904</td>
          <td>25.159969</td>
          <td>0.254405</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.837753</td>
          <td>1.487361</td>
          <td>27.594972</td>
          <td>0.306411</td>
          <td>25.757525</td>
          <td>0.103649</td>
          <td>25.137827</td>
          <td>0.114422</td>
          <td>24.267041</td>
          <td>0.119352</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.647903</td>
          <td>0.852992</td>
          <td>28.897963</td>
          <td>0.873318</td>
          <td>27.414245</td>
          <td>0.264716</td>
          <td>26.245215</td>
          <td>0.158112</td>
          <td>25.819766</td>
          <td>0.205149</td>
          <td>25.474584</td>
          <td>0.328007</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.601679</td>
          <td>0.407806</td>
          <td>25.892390</td>
          <td>0.081510</td>
          <td>26.000084</td>
          <td>0.078852</td>
          <td>25.784467</td>
          <td>0.106120</td>
          <td>25.356940</td>
          <td>0.138360</td>
          <td>25.036427</td>
          <td>0.229761</td>
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
          <td>26.358274</td>
          <td>0.337399</td>
          <td>26.440089</td>
          <td>0.131513</td>
          <td>25.447532</td>
          <td>0.048318</td>
          <td>25.218185</td>
          <td>0.064427</td>
          <td>24.738422</td>
          <td>0.080605</td>
          <td>24.687716</td>
          <td>0.171410</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.615496</td>
          <td>0.835487</td>
          <td>26.740300</td>
          <td>0.170130</td>
          <td>26.164222</td>
          <td>0.091121</td>
          <td>25.184777</td>
          <td>0.062546</td>
          <td>24.864869</td>
          <td>0.090101</td>
          <td>24.151777</td>
          <td>0.107947</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.310282</td>
          <td>0.324811</td>
          <td>26.693336</td>
          <td>0.163461</td>
          <td>26.377183</td>
          <td>0.109811</td>
          <td>26.225951</td>
          <td>0.155527</td>
          <td>25.984060</td>
          <td>0.235225</td>
          <td>24.939536</td>
          <td>0.211959</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.333522</td>
          <td>0.330856</td>
          <td>26.236863</td>
          <td>0.110243</td>
          <td>26.164032</td>
          <td>0.091106</td>
          <td>25.947325</td>
          <td>0.122299</td>
          <td>25.852099</td>
          <td>0.210777</td>
          <td>25.634081</td>
          <td>0.371873</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.699990</td>
          <td>0.439511</td>
          <td>26.686979</td>
          <td>0.162577</td>
          <td>26.628539</td>
          <td>0.136591</td>
          <td>26.316335</td>
          <td>0.168008</td>
          <td>26.068658</td>
          <td>0.252210</td>
          <td>25.825071</td>
          <td>0.430770</td>
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
          <td>27.925720</td>
          <td>1.095639</td>
          <td>26.620006</td>
          <td>0.176376</td>
          <td>25.948617</td>
          <td>0.088630</td>
          <td>25.192858</td>
          <td>0.074671</td>
          <td>24.913747</td>
          <td>0.110477</td>
          <td>23.812424</td>
          <td>0.094733</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.357569</td>
          <td>0.770575</td>
          <td>28.429324</td>
          <td>0.715027</td>
          <td>26.772110</td>
          <td>0.180783</td>
          <td>26.206631</td>
          <td>0.180159</td>
          <td>26.424202</td>
          <td>0.388002</td>
          <td>25.157031</td>
          <td>0.296226</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.821292</td>
          <td>0.934731</td>
          <td>28.159118</td>
          <td>0.553139</td>
          <td>25.900491</td>
          <td>0.141814</td>
          <td>25.096711</td>
          <td>0.132389</td>
          <td>24.316824</td>
          <td>0.150180</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.981508</td>
          <td>1.920698</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.995016</td>
          <td>0.232260</td>
          <td>26.208745</td>
          <td>0.192887</td>
          <td>25.650318</td>
          <td>0.221263</td>
          <td>25.246430</td>
          <td>0.338598</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.750985</td>
          <td>1.686670</td>
          <td>26.355818</td>
          <td>0.140775</td>
          <td>26.053205</td>
          <td>0.097188</td>
          <td>25.808941</td>
          <td>0.128137</td>
          <td>25.805987</td>
          <td>0.236380</td>
          <td>24.890748</td>
          <td>0.238409</td>
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
          <td>26.033518</td>
          <td>0.293941</td>
          <td>26.437262</td>
          <td>0.153750</td>
          <td>25.389989</td>
          <td>0.055241</td>
          <td>25.060793</td>
          <td>0.067911</td>
          <td>24.657949</td>
          <td>0.090174</td>
          <td>24.868994</td>
          <td>0.238884</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.243703</td>
          <td>1.310620</td>
          <td>26.649556</td>
          <td>0.181503</td>
          <td>26.131887</td>
          <td>0.104520</td>
          <td>25.253483</td>
          <td>0.079120</td>
          <td>24.887591</td>
          <td>0.108433</td>
          <td>24.237106</td>
          <td>0.137692</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.350593</td>
          <td>1.392000</td>
          <td>26.686042</td>
          <td>0.188590</td>
          <td>26.707762</td>
          <td>0.173271</td>
          <td>26.357895</td>
          <td>0.207208</td>
          <td>25.997881</td>
          <td>0.279835</td>
          <td>26.475286</td>
          <td>0.789942</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.148826</td>
          <td>0.324690</td>
          <td>26.100290</td>
          <td>0.116027</td>
          <td>26.226280</td>
          <td>0.116604</td>
          <td>25.874303</td>
          <td>0.139939</td>
          <td>25.878048</td>
          <td>0.258218</td>
          <td>25.384151</td>
          <td>0.365158</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.902997</td>
          <td>0.262400</td>
          <td>27.293977</td>
          <td>0.310341</td>
          <td>26.772080</td>
          <td>0.182499</td>
          <td>26.037110</td>
          <td>0.157494</td>
          <td>25.693961</td>
          <td>0.217374</td>
          <td>24.826125</td>
          <td>0.228127</td>
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
          <td>27.810350</td>
          <td>0.944400</td>
          <td>26.666388</td>
          <td>0.159762</td>
          <td>26.034849</td>
          <td>0.081320</td>
          <td>25.185554</td>
          <td>0.062598</td>
          <td>24.607855</td>
          <td>0.071832</td>
          <td>24.072731</td>
          <td>0.100750</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.814832</td>
          <td>0.407995</td>
          <td>26.790820</td>
          <td>0.157182</td>
          <td>26.206418</td>
          <td>0.153094</td>
          <td>25.424346</td>
          <td>0.146764</td>
          <td>25.353493</td>
          <td>0.298003</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.786697</td>
          <td>0.492914</td>
          <td>30.858136</td>
          <td>2.397502</td>
          <td>27.653442</td>
          <td>0.345655</td>
          <td>25.974124</td>
          <td>0.136182</td>
          <td>25.092876</td>
          <td>0.119319</td>
          <td>24.552479</td>
          <td>0.165841</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.685813</td>
          <td>2.529909</td>
          <td>28.321196</td>
          <td>0.694608</td>
          <td>27.384942</td>
          <td>0.317905</td>
          <td>26.333671</td>
          <td>0.213442</td>
          <td>26.509607</td>
          <td>0.437714</td>
          <td>25.064772</td>
          <td>0.291890</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.439351</td>
          <td>0.359928</td>
          <td>26.085368</td>
          <td>0.096690</td>
          <td>25.901154</td>
          <td>0.072355</td>
          <td>25.626305</td>
          <td>0.092521</td>
          <td>25.332888</td>
          <td>0.135707</td>
          <td>25.469916</td>
          <td>0.327229</td>
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
          <td>27.098331</td>
          <td>0.615770</td>
          <td>26.547430</td>
          <td>0.154318</td>
          <td>25.470763</td>
          <td>0.053420</td>
          <td>25.100869</td>
          <td>0.063104</td>
          <td>24.972456</td>
          <td>0.107100</td>
          <td>24.559177</td>
          <td>0.166279</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.502027</td>
          <td>0.782776</td>
          <td>26.623379</td>
          <td>0.156140</td>
          <td>26.086189</td>
          <td>0.086492</td>
          <td>25.273064</td>
          <td>0.068828</td>
          <td>24.852774</td>
          <td>0.090630</td>
          <td>24.197238</td>
          <td>0.114239</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.340745</td>
          <td>0.715489</td>
          <td>26.674499</td>
          <td>0.167665</td>
          <td>26.347290</td>
          <td>0.112312</td>
          <td>26.033117</td>
          <td>0.138547</td>
          <td>25.955566</td>
          <td>0.240551</td>
          <td>26.587959</td>
          <td>0.772304</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.378569</td>
          <td>0.368936</td>
          <td>26.302805</td>
          <td>0.128997</td>
          <td>26.053820</td>
          <td>0.092756</td>
          <td>25.747429</td>
          <td>0.115729</td>
          <td>25.604531</td>
          <td>0.191020</td>
          <td>25.322397</td>
          <td>0.323552</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.305872</td>
          <td>0.331747</td>
          <td>26.664964</td>
          <td>0.164894</td>
          <td>26.351500</td>
          <td>0.111611</td>
          <td>26.241249</td>
          <td>0.163952</td>
          <td>25.592029</td>
          <td>0.175729</td>
          <td>25.428928</td>
          <td>0.328080</td>
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
