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

    <pzflow.flow.Flow at 0x7f6b2dad4b20>



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
    0      23.994413  0.058003  0.048201  
    1      25.391064  0.058097  0.056221  
    2      24.304707  0.126514  0.067453  
    3      25.291103  0.139830  0.076355  
    4      25.096743  0.014762  0.008881  
    ...          ...       ...       ...  
    99995  24.737946  0.048318  0.042478  
    99996  24.224169  0.028689  0.027657  
    99997  25.613836  0.041644  0.031350  
    99998  25.274899  0.034230  0.021133  
    99999  25.699642  0.084951  0.051996  
    
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
          <td>27.423980</td>
          <td>0.736960</td>
          <td>26.761797</td>
          <td>0.173267</td>
          <td>25.960236</td>
          <td>0.076126</td>
          <td>25.123171</td>
          <td>0.059220</td>
          <td>24.674774</td>
          <td>0.076200</td>
          <td>24.126287</td>
          <td>0.105569</td>
          <td>0.058003</td>
          <td>0.048201</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.249768</td>
          <td>0.654662</td>
          <td>27.036992</td>
          <td>0.218404</td>
          <td>26.464475</td>
          <td>0.118489</td>
          <td>26.105168</td>
          <td>0.140197</td>
          <td>26.046645</td>
          <td>0.247688</td>
          <td>25.065407</td>
          <td>0.235342</td>
          <td>0.058097</td>
          <td>0.056221</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.167260</td>
          <td>1.165564</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.659128</td>
          <td>0.322529</td>
          <td>25.950367</td>
          <td>0.122623</td>
          <td>25.138994</td>
          <td>0.114539</td>
          <td>24.466689</td>
          <td>0.141863</td>
          <td>0.126514</td>
          <td>0.067453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.859783</td>
          <td>0.421965</td>
          <td>27.539486</td>
          <td>0.293039</td>
          <td>26.332348</td>
          <td>0.170314</td>
          <td>25.462173</td>
          <td>0.151469</td>
          <td>25.269220</td>
          <td>0.278128</td>
          <td>0.139830</td>
          <td>0.076355</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.972695</td>
          <td>0.537967</td>
          <td>26.034955</td>
          <td>0.092395</td>
          <td>25.924191</td>
          <td>0.073738</td>
          <td>25.682149</td>
          <td>0.097025</td>
          <td>25.631322</td>
          <td>0.174994</td>
          <td>25.191650</td>
          <td>0.261092</td>
          <td>0.014762</td>
          <td>0.008881</td>
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
          <td>26.917474</td>
          <td>0.516747</td>
          <td>26.232692</td>
          <td>0.109843</td>
          <td>25.392560</td>
          <td>0.046016</td>
          <td>25.136607</td>
          <td>0.059930</td>
          <td>24.900182</td>
          <td>0.092942</td>
          <td>24.719358</td>
          <td>0.176080</td>
          <td>0.048318</td>
          <td>0.042478</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.653270</td>
          <td>0.424201</td>
          <td>26.740918</td>
          <td>0.170220</td>
          <td>25.915581</td>
          <td>0.073179</td>
          <td>25.256020</td>
          <td>0.066624</td>
          <td>24.748541</td>
          <td>0.081327</td>
          <td>24.276316</td>
          <td>0.120319</td>
          <td>0.028689</td>
          <td>0.027657</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.000984</td>
          <td>0.549097</td>
          <td>26.865875</td>
          <td>0.189221</td>
          <td>26.329027</td>
          <td>0.105288</td>
          <td>26.035940</td>
          <td>0.132062</td>
          <td>25.918362</td>
          <td>0.222751</td>
          <td>25.771154</td>
          <td>0.413414</td>
          <td>0.041644</td>
          <td>0.031350</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.667095</td>
          <td>0.428685</td>
          <td>26.238430</td>
          <td>0.110393</td>
          <td>26.060507</td>
          <td>0.083170</td>
          <td>25.875875</td>
          <td>0.114931</td>
          <td>25.641097</td>
          <td>0.176451</td>
          <td>24.898463</td>
          <td>0.204797</td>
          <td>0.034230</td>
          <td>0.021133</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.313237</td>
          <td>0.325574</td>
          <td>26.742303</td>
          <td>0.170420</td>
          <td>26.801891</td>
          <td>0.158531</td>
          <td>26.246200</td>
          <td>0.158245</td>
          <td>25.830797</td>
          <td>0.207054</td>
          <td>25.899086</td>
          <td>0.455558</td>
          <td>0.084951</td>
          <td>0.051996</td>
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
          <td>28.719661</td>
          <td>1.668318</td>
          <td>26.842056</td>
          <td>0.214360</td>
          <td>26.124440</td>
          <td>0.104401</td>
          <td>25.206562</td>
          <td>0.076337</td>
          <td>24.805938</td>
          <td>0.101506</td>
          <td>24.169423</td>
          <td>0.130583</td>
          <td>0.058003</td>
          <td>0.048201</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.338619</td>
          <td>0.765700</td>
          <td>27.441115</td>
          <td>0.349045</td>
          <td>26.651768</td>
          <td>0.164919</td>
          <td>26.164737</td>
          <td>0.175754</td>
          <td>26.170711</td>
          <td>0.321020</td>
          <td>25.738114</td>
          <td>0.470028</td>
          <td>0.058097</td>
          <td>0.056221</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.528115</td>
          <td>2.365034</td>
          <td>29.715983</td>
          <td>1.543852</td>
          <td>27.916978</td>
          <td>0.467296</td>
          <td>25.941879</td>
          <td>0.148616</td>
          <td>24.991485</td>
          <td>0.122189</td>
          <td>24.521357</td>
          <td>0.180781</td>
          <td>0.126514</td>
          <td>0.067453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.107821</td>
          <td>0.589150</td>
          <td>27.504710</td>
          <td>0.342521</td>
          <td>26.239031</td>
          <td>0.192800</td>
          <td>25.869261</td>
          <td>0.258642</td>
          <td>25.135922</td>
          <td>0.302585</td>
          <td>0.139830</td>
          <td>0.076355</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.190463</td>
          <td>0.328494</td>
          <td>26.089087</td>
          <td>0.111757</td>
          <td>25.961514</td>
          <td>0.089684</td>
          <td>25.528831</td>
          <td>0.100408</td>
          <td>25.302523</td>
          <td>0.154698</td>
          <td>25.148986</td>
          <td>0.294386</td>
          <td>0.014762</td>
          <td>0.008881</td>
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
          <td>27.502640</td>
          <td>0.849839</td>
          <td>26.113464</td>
          <td>0.114818</td>
          <td>25.511056</td>
          <td>0.060635</td>
          <td>25.089630</td>
          <td>0.068654</td>
          <td>24.942728</td>
          <td>0.114089</td>
          <td>25.004073</td>
          <td>0.263344</td>
          <td>0.048318</td>
          <td>0.042478</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.445943</td>
          <td>0.401632</td>
          <td>27.093038</td>
          <td>0.262186</td>
          <td>25.938781</td>
          <td>0.088102</td>
          <td>25.201689</td>
          <td>0.075465</td>
          <td>24.851628</td>
          <td>0.104922</td>
          <td>24.206424</td>
          <td>0.133890</td>
          <td>0.028689</td>
          <td>0.027657</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.590737</td>
          <td>0.172737</td>
          <td>26.183190</td>
          <td>0.109354</td>
          <td>26.221562</td>
          <td>0.183248</td>
          <td>25.882052</td>
          <td>0.252667</td>
          <td>25.567411</td>
          <td>0.410723</td>
          <td>0.041644</td>
          <td>0.031350</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.451989</td>
          <td>0.403516</td>
          <td>26.209659</td>
          <td>0.124345</td>
          <td>26.088246</td>
          <td>0.100458</td>
          <td>26.126955</td>
          <td>0.168797</td>
          <td>25.569725</td>
          <td>0.194528</td>
          <td>25.212990</td>
          <td>0.310568</td>
          <td>0.034230</td>
          <td>0.021133</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.919983</td>
          <td>1.101108</td>
          <td>26.996552</td>
          <td>0.245089</td>
          <td>26.590091</td>
          <td>0.157302</td>
          <td>26.504140</td>
          <td>0.234886</td>
          <td>26.108625</td>
          <td>0.307025</td>
          <td>25.998436</td>
          <td>0.571366</td>
          <td>0.084951</td>
          <td>0.051996</td>
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
          <td>27.103541</td>
          <td>0.237952</td>
          <td>26.042207</td>
          <td>0.084949</td>
          <td>25.088760</td>
          <td>0.059747</td>
          <td>24.553942</td>
          <td>0.071088</td>
          <td>23.866389</td>
          <td>0.087347</td>
          <td>0.058003</td>
          <td>0.048201</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.973684</td>
          <td>0.474656</td>
          <td>26.459017</td>
          <td>0.122991</td>
          <td>26.426217</td>
          <td>0.192490</td>
          <td>25.812206</td>
          <td>0.212261</td>
          <td>25.434183</td>
          <td>0.330555</td>
          <td>0.058097</td>
          <td>0.056221</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.346126</td>
          <td>0.746792</td>
          <td>27.824282</td>
          <td>0.449637</td>
          <td>28.836731</td>
          <td>0.835232</td>
          <td>25.797254</td>
          <td>0.121343</td>
          <td>24.845829</td>
          <td>0.099768</td>
          <td>24.370413</td>
          <td>0.147269</td>
          <td>0.126514</td>
          <td>0.067453</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.066342</td>
          <td>0.547320</td>
          <td>27.279304</td>
          <td>0.270604</td>
          <td>26.223488</td>
          <td>0.179205</td>
          <td>25.932224</td>
          <td>0.257504</td>
          <td>25.793140</td>
          <td>0.476933</td>
          <td>0.139830</td>
          <td>0.076355</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.357584</td>
          <td>0.337638</td>
          <td>26.036987</td>
          <td>0.092721</td>
          <td>26.119874</td>
          <td>0.087812</td>
          <td>26.051341</td>
          <td>0.134111</td>
          <td>25.558994</td>
          <td>0.164868</td>
          <td>25.566299</td>
          <td>0.353319</td>
          <td>0.014762</td>
          <td>0.008881</td>
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
          <td>26.579534</td>
          <td>0.407670</td>
          <td>26.564469</td>
          <td>0.149832</td>
          <td>25.519752</td>
          <td>0.052964</td>
          <td>25.149303</td>
          <td>0.062392</td>
          <td>24.948250</td>
          <td>0.099626</td>
          <td>24.866979</td>
          <td>0.204935</td>
          <td>0.048318</td>
          <td>0.042478</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.533942</td>
          <td>0.143899</td>
          <td>26.114892</td>
          <td>0.088190</td>
          <td>25.203429</td>
          <td>0.064314</td>
          <td>24.689206</td>
          <td>0.078011</td>
          <td>24.140704</td>
          <td>0.108094</td>
          <td>0.028689</td>
          <td>0.027657</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.176221</td>
          <td>0.628269</td>
          <td>27.020134</td>
          <td>0.218594</td>
          <td>26.484651</td>
          <td>0.122763</td>
          <td>26.301641</td>
          <td>0.169000</td>
          <td>25.570902</td>
          <td>0.169165</td>
          <td>25.442255</td>
          <td>0.325165</td>
          <td>0.041644</td>
          <td>0.031350</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.180497</td>
          <td>0.294809</td>
          <td>26.502727</td>
          <td>0.140107</td>
          <td>26.099669</td>
          <td>0.087032</td>
          <td>25.906616</td>
          <td>0.119387</td>
          <td>25.893695</td>
          <td>0.220477</td>
          <td>25.315409</td>
          <td>0.291735</td>
          <td>0.034230</td>
          <td>0.021133</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.176033</td>
          <td>0.644073</td>
          <td>26.715226</td>
          <td>0.175567</td>
          <td>26.710682</td>
          <td>0.155805</td>
          <td>26.396924</td>
          <td>0.191561</td>
          <td>26.220643</td>
          <td>0.302249</td>
          <td>25.373958</td>
          <td>0.321000</td>
          <td>0.084951</td>
          <td>0.051996</td>
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
