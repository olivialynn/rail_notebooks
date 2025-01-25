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

    <pzflow.flow.Flow at 0x7f9ff35a6b30>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>26.778310</td>
          <td>0.466179</td>
          <td>26.860648</td>
          <td>0.188389</td>
          <td>25.957550</td>
          <td>0.075945</td>
          <td>25.145656</td>
          <td>0.060413</td>
          <td>24.699179</td>
          <td>0.077861</td>
          <td>23.939832</td>
          <td>0.089645</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.065085</td>
          <td>0.223568</td>
          <td>26.587952</td>
          <td>0.131884</td>
          <td>26.215494</td>
          <td>0.154140</td>
          <td>25.842785</td>
          <td>0.209142</td>
          <td>26.094219</td>
          <td>0.526415</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.920641</td>
          <td>0.441923</td>
          <td>28.624549</td>
          <td>0.662802</td>
          <td>25.919343</td>
          <td>0.119362</td>
          <td>24.958045</td>
          <td>0.097783</td>
          <td>24.351809</td>
          <td>0.128463</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.227759</td>
          <td>1.790709</td>
          <td>27.430782</td>
          <td>0.268312</td>
          <td>26.158125</td>
          <td>0.146735</td>
          <td>25.748382</td>
          <td>0.193206</td>
          <td>24.677894</td>
          <td>0.169983</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.037187</td>
          <td>0.260649</td>
          <td>26.100788</td>
          <td>0.097884</td>
          <td>25.845034</td>
          <td>0.068750</td>
          <td>25.504547</td>
          <td>0.082994</td>
          <td>25.859444</td>
          <td>0.212075</td>
          <td>25.240540</td>
          <td>0.271718</td>
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
          <td>27.239510</td>
          <td>0.650033</td>
          <td>26.474328</td>
          <td>0.135459</td>
          <td>25.478679</td>
          <td>0.049673</td>
          <td>25.154694</td>
          <td>0.060900</td>
          <td>24.898931</td>
          <td>0.092840</td>
          <td>24.670158</td>
          <td>0.168868</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.819485</td>
          <td>0.480712</td>
          <td>26.866857</td>
          <td>0.189378</td>
          <td>26.088280</td>
          <td>0.085231</td>
          <td>25.093554</td>
          <td>0.057684</td>
          <td>24.733453</td>
          <td>0.080252</td>
          <td>24.157461</td>
          <td>0.108484</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>29.772036</td>
          <td>2.437511</td>
          <td>26.996694</td>
          <td>0.211185</td>
          <td>26.523141</td>
          <td>0.124684</td>
          <td>26.104983</td>
          <td>0.140174</td>
          <td>25.814556</td>
          <td>0.204255</td>
          <td>25.902906</td>
          <td>0.456868</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.057023</td>
          <td>0.264901</td>
          <td>26.164248</td>
          <td>0.103472</td>
          <td>25.951946</td>
          <td>0.075570</td>
          <td>25.800772</td>
          <td>0.107643</td>
          <td>25.879645</td>
          <td>0.215682</td>
          <td>24.872153</td>
          <td>0.200326</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.511409</td>
          <td>0.380382</td>
          <td>26.702302</td>
          <td>0.164715</td>
          <td>26.671514</td>
          <td>0.141748</td>
          <td>26.258528</td>
          <td>0.159922</td>
          <td>25.918232</td>
          <td>0.222727</td>
          <td>26.620967</td>
          <td>0.759813</td>
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
          <td>1.398945</td>
          <td>27.265825</td>
          <td>0.724865</td>
          <td>26.885250</td>
          <td>0.220403</td>
          <td>26.222782</td>
          <td>0.112682</td>
          <td>25.138247</td>
          <td>0.071151</td>
          <td>24.592575</td>
          <td>0.083360</td>
          <td>24.033588</td>
          <td>0.114939</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.212267</td>
          <td>0.699213</td>
          <td>29.148662</td>
          <td>1.120793</td>
          <td>26.521904</td>
          <td>0.146023</td>
          <td>26.235154</td>
          <td>0.184561</td>
          <td>25.475800</td>
          <td>0.179260</td>
          <td>25.085801</td>
          <td>0.279654</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.885345</td>
          <td>1.082357</td>
          <td>28.106792</td>
          <td>0.580722</td>
          <td>29.447211</td>
          <td>1.254513</td>
          <td>25.848054</td>
          <td>0.135545</td>
          <td>24.911867</td>
          <td>0.112764</td>
          <td>24.130934</td>
          <td>0.127943</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.334112</td>
          <td>2.071425</td>
          <td>27.159569</td>
          <td>0.265906</td>
          <td>26.062402</td>
          <td>0.170412</td>
          <td>25.970872</td>
          <td>0.287828</td>
          <td>25.428420</td>
          <td>0.390382</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.039454</td>
          <td>0.291106</td>
          <td>26.123191</td>
          <td>0.115108</td>
          <td>25.955665</td>
          <td>0.089210</td>
          <td>25.589295</td>
          <td>0.105845</td>
          <td>25.640428</td>
          <td>0.205956</td>
          <td>24.894016</td>
          <td>0.239053</td>
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
          <td>28.180585</td>
          <td>1.276860</td>
          <td>26.892641</td>
          <td>0.225774</td>
          <td>25.417528</td>
          <td>0.056607</td>
          <td>25.090339</td>
          <td>0.069710</td>
          <td>24.871658</td>
          <td>0.108738</td>
          <td>24.625218</td>
          <td>0.194943</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.912283</td>
          <td>0.568545</td>
          <td>27.103866</td>
          <td>0.264848</td>
          <td>25.983221</td>
          <td>0.091750</td>
          <td>25.214507</td>
          <td>0.076444</td>
          <td>24.844249</td>
          <td>0.104404</td>
          <td>24.194657</td>
          <td>0.132737</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.934462</td>
          <td>1.108268</td>
          <td>26.729851</td>
          <td>0.195677</td>
          <td>26.183787</td>
          <td>0.110307</td>
          <td>26.048898</td>
          <td>0.159530</td>
          <td>25.550855</td>
          <td>0.193317</td>
          <td>26.627382</td>
          <td>0.871198</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.927261</td>
          <td>0.271737</td>
          <td>26.106997</td>
          <td>0.116705</td>
          <td>26.293788</td>
          <td>0.123648</td>
          <td>25.610824</td>
          <td>0.111358</td>
          <td>25.601275</td>
          <td>0.205302</td>
          <td>25.173488</td>
          <td>0.309086</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.983830</td>
          <td>0.280217</td>
          <td>26.796944</td>
          <td>0.206515</td>
          <td>26.456335</td>
          <td>0.139346</td>
          <td>26.105177</td>
          <td>0.166916</td>
          <td>27.303896</td>
          <td>0.739174</td>
          <td>25.594451</td>
          <td>0.421355</td>
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
          <td>1.398945</td>
          <td>28.911170</td>
          <td>1.707718</td>
          <td>26.542390</td>
          <td>0.143655</td>
          <td>26.003423</td>
          <td>0.079096</td>
          <td>25.204572</td>
          <td>0.063663</td>
          <td>24.715296</td>
          <td>0.078987</td>
          <td>24.036637</td>
          <td>0.097613</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.912054</td>
          <td>0.514981</td>
          <td>26.998581</td>
          <td>0.211683</td>
          <td>26.814812</td>
          <td>0.160440</td>
          <td>26.289967</td>
          <td>0.164432</td>
          <td>25.975785</td>
          <td>0.233828</td>
          <td>26.201953</td>
          <td>0.569522</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.720934</td>
          <td>0.931124</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.027124</td>
          <td>0.460857</td>
          <td>25.914877</td>
          <td>0.129382</td>
          <td>24.898925</td>
          <td>0.100742</td>
          <td>24.381338</td>
          <td>0.143227</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.027464</td>
          <td>0.639820</td>
          <td>28.320775</td>
          <td>0.694410</td>
          <td>27.049121</td>
          <td>0.242065</td>
          <td>26.454631</td>
          <td>0.236007</td>
          <td>25.435082</td>
          <td>0.184081</td>
          <td>24.966479</td>
          <td>0.269532</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.769404</td>
          <td>0.209120</td>
          <td>26.119408</td>
          <td>0.099615</td>
          <td>25.902247</td>
          <td>0.072425</td>
          <td>25.545619</td>
          <td>0.086181</td>
          <td>25.327610</td>
          <td>0.135090</td>
          <td>25.646144</td>
          <td>0.375874</td>
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
          <td>29.089626</td>
          <td>1.903595</td>
          <td>26.421285</td>
          <td>0.138474</td>
          <td>25.423511</td>
          <td>0.051226</td>
          <td>25.116661</td>
          <td>0.063994</td>
          <td>24.764269</td>
          <td>0.089234</td>
          <td>25.025838</td>
          <td>0.245911</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.059807</td>
          <td>0.268346</td>
          <td>26.621674</td>
          <td>0.155913</td>
          <td>26.165464</td>
          <td>0.092738</td>
          <td>25.272611</td>
          <td>0.068801</td>
          <td>24.726353</td>
          <td>0.081081</td>
          <td>24.536129</td>
          <td>0.153125</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.793543</td>
          <td>0.485465</td>
          <td>26.559777</td>
          <td>0.152017</td>
          <td>26.133784</td>
          <td>0.093169</td>
          <td>26.408160</td>
          <td>0.190807</td>
          <td>25.883475</td>
          <td>0.226617</td>
          <td>26.074699</td>
          <td>0.541237</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.712220</td>
          <td>0.475818</td>
          <td>26.180436</td>
          <td>0.116009</td>
          <td>26.063169</td>
          <td>0.093521</td>
          <td>26.054263</td>
          <td>0.150884</td>
          <td>25.820197</td>
          <td>0.228780</td>
          <td>24.930704</td>
          <td>0.235404</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.579691</td>
          <td>0.410643</td>
          <td>26.426866</td>
          <td>0.134431</td>
          <td>26.561692</td>
          <td>0.133957</td>
          <td>26.452000</td>
          <td>0.196012</td>
          <td>25.929259</td>
          <td>0.233175</td>
          <td>25.205644</td>
          <td>0.274172</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
