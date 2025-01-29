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

    <pzflow.flow.Flow at 0x7f798c5eb1c0>



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
          <td>26.808037</td>
          <td>0.476635</td>
          <td>26.854593</td>
          <td>0.187429</td>
          <td>25.858733</td>
          <td>0.069589</td>
          <td>25.108226</td>
          <td>0.058440</td>
          <td>24.637912</td>
          <td>0.073758</td>
          <td>24.156768</td>
          <td>0.108418</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.600121</td>
          <td>0.718933</td>
          <td>26.393108</td>
          <td>0.111348</td>
          <td>26.424103</td>
          <td>0.184101</td>
          <td>25.970003</td>
          <td>0.232505</td>
          <td>25.399804</td>
          <td>0.309014</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.839332</td>
          <td>0.841318</td>
          <td>27.686133</td>
          <td>0.329528</td>
          <td>26.096836</td>
          <td>0.139193</td>
          <td>25.193769</td>
          <td>0.120130</td>
          <td>24.253487</td>
          <td>0.117954</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.004816</td>
          <td>1.061456</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.500928</td>
          <td>0.284048</td>
          <td>26.365136</td>
          <td>0.175127</td>
          <td>25.500823</td>
          <td>0.156568</td>
          <td>25.055468</td>
          <td>0.233414</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.399008</td>
          <td>0.348409</td>
          <td>26.047030</td>
          <td>0.093379</td>
          <td>25.899416</td>
          <td>0.072140</td>
          <td>25.615087</td>
          <td>0.091476</td>
          <td>25.392292</td>
          <td>0.142640</td>
          <td>25.738750</td>
          <td>0.403263</td>
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
          <td>26.202857</td>
          <td>0.107021</td>
          <td>25.503545</td>
          <td>0.050782</td>
          <td>25.168376</td>
          <td>0.061643</td>
          <td>24.688513</td>
          <td>0.077131</td>
          <td>25.261720</td>
          <td>0.276439</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.724196</td>
          <td>0.895148</td>
          <td>26.785009</td>
          <td>0.176713</td>
          <td>25.910107</td>
          <td>0.072826</td>
          <td>25.180236</td>
          <td>0.062295</td>
          <td>24.969895</td>
          <td>0.098805</td>
          <td>24.185327</td>
          <td>0.111155</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.311266</td>
          <td>0.682919</td>
          <td>26.836908</td>
          <td>0.184650</td>
          <td>26.325269</td>
          <td>0.104942</td>
          <td>26.288098</td>
          <td>0.164012</td>
          <td>26.071671</td>
          <td>0.252834</td>
          <td>25.875663</td>
          <td>0.447591</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.219510</td>
          <td>0.302105</td>
          <td>26.161493</td>
          <td>0.103223</td>
          <td>26.260673</td>
          <td>0.099172</td>
          <td>25.832396</td>
          <td>0.110656</td>
          <td>26.097606</td>
          <td>0.258268</td>
          <td>25.363784</td>
          <td>0.300210</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.101512</td>
          <td>0.590079</td>
          <td>27.011901</td>
          <td>0.213883</td>
          <td>26.719275</td>
          <td>0.147693</td>
          <td>26.221282</td>
          <td>0.154906</td>
          <td>26.177629</td>
          <td>0.275690</td>
          <td>25.519865</td>
          <td>0.339988</td>
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
          <td>27.756632</td>
          <td>0.991328</td>
          <td>26.662713</td>
          <td>0.182870</td>
          <td>26.065390</td>
          <td>0.098200</td>
          <td>25.339588</td>
          <td>0.084990</td>
          <td>24.747044</td>
          <td>0.095485</td>
          <td>23.925147</td>
          <td>0.104563</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.719870</td>
          <td>0.969583</td>
          <td>27.797411</td>
          <td>0.455292</td>
          <td>26.628045</td>
          <td>0.159929</td>
          <td>26.459947</td>
          <td>0.222853</td>
          <td>26.045469</td>
          <td>0.287481</td>
          <td>25.379896</td>
          <td>0.353693</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.675518</td>
          <td>0.954816</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.382245</td>
          <td>0.647738</td>
          <td>26.017012</td>
          <td>0.156732</td>
          <td>25.022999</td>
          <td>0.124203</td>
          <td>24.503050</td>
          <td>0.176047</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.210913</td>
          <td>0.726820</td>
          <td>28.042553</td>
          <td>0.573351</td>
          <td>26.825377</td>
          <td>0.201629</td>
          <td>26.476235</td>
          <td>0.241088</td>
          <td>25.369132</td>
          <td>0.174677</td>
          <td>24.909033</td>
          <td>0.258054</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.039953</td>
          <td>0.291223</td>
          <td>26.023347</td>
          <td>0.105517</td>
          <td>25.972919</td>
          <td>0.090574</td>
          <td>25.688129</td>
          <td>0.115375</td>
          <td>25.995533</td>
          <td>0.276109</td>
          <td>24.723527</td>
          <td>0.207459</td>
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
          <td>26.226504</td>
          <td>0.342788</td>
          <td>26.409431</td>
          <td>0.150128</td>
          <td>25.407023</td>
          <td>0.056082</td>
          <td>25.079234</td>
          <td>0.069029</td>
          <td>25.008038</td>
          <td>0.122443</td>
          <td>24.758405</td>
          <td>0.217946</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.263701</td>
          <td>0.348930</td>
          <td>26.958468</td>
          <td>0.235033</td>
          <td>26.222740</td>
          <td>0.113146</td>
          <td>25.272362</td>
          <td>0.080449</td>
          <td>24.951393</td>
          <td>0.114635</td>
          <td>24.230200</td>
          <td>0.136874</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.435248</td>
          <td>0.401043</td>
          <td>26.760192</td>
          <td>0.200727</td>
          <td>26.494008</td>
          <td>0.144329</td>
          <td>26.668900</td>
          <td>0.267951</td>
          <td>25.687127</td>
          <td>0.216703</td>
          <td>25.053793</td>
          <td>0.275753</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.899940</td>
          <td>0.265764</td>
          <td>26.237317</td>
          <td>0.130659</td>
          <td>26.073313</td>
          <td>0.102033</td>
          <td>25.722148</td>
          <td>0.122683</td>
          <td>26.026460</td>
          <td>0.291330</td>
          <td>25.713914</td>
          <td>0.469908</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.908560</td>
          <td>0.569098</td>
          <td>27.014423</td>
          <td>0.247355</td>
          <td>26.607992</td>
          <td>0.158723</td>
          <td>26.660090</td>
          <td>0.265331</td>
          <td>26.139290</td>
          <td>0.312802</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>26.714464</td>
          <td>0.444378</td>
          <td>26.805844</td>
          <td>0.179881</td>
          <td>26.002111</td>
          <td>0.079004</td>
          <td>25.072752</td>
          <td>0.056636</td>
          <td>24.629869</td>
          <td>0.073245</td>
          <td>23.810916</td>
          <td>0.080032</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.234123</td>
          <td>2.859650</td>
          <td>27.160290</td>
          <td>0.242084</td>
          <td>26.685808</td>
          <td>0.143636</td>
          <td>26.648098</td>
          <td>0.222375</td>
          <td>26.019230</td>
          <td>0.242373</td>
          <td>25.183863</td>
          <td>0.259672</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.707267</td>
          <td>1.600252</td>
          <td>28.157614</td>
          <td>0.559164</td>
          <td>28.083147</td>
          <td>0.480554</td>
          <td>26.077775</td>
          <td>0.148894</td>
          <td>25.113171</td>
          <td>0.121442</td>
          <td>24.134213</td>
          <td>0.115640</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.878276</td>
          <td>0.507586</td>
          <td>27.959929</td>
          <td>0.494919</td>
          <td>26.206715</td>
          <td>0.191879</td>
          <td>25.581964</td>
          <td>0.208290</td>
          <td>25.556765</td>
          <td>0.429379</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.891324</td>
          <td>0.231415</td>
          <td>26.251643</td>
          <td>0.111808</td>
          <td>25.988813</td>
          <td>0.078183</td>
          <td>25.792036</td>
          <td>0.106983</td>
          <td>25.448975</td>
          <td>0.149971</td>
          <td>24.926452</td>
          <td>0.209948</td>
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
          <td>27.526401</td>
          <td>0.821840</td>
          <td>26.424672</td>
          <td>0.138878</td>
          <td>25.516942</td>
          <td>0.055655</td>
          <td>25.066969</td>
          <td>0.061236</td>
          <td>24.748063</td>
          <td>0.087971</td>
          <td>24.901440</td>
          <td>0.221853</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.934148</td>
          <td>0.528090</td>
          <td>26.768292</td>
          <td>0.176652</td>
          <td>25.964384</td>
          <td>0.077683</td>
          <td>25.206684</td>
          <td>0.064897</td>
          <td>24.795370</td>
          <td>0.086166</td>
          <td>24.171612</td>
          <td>0.111716</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>28.991263</td>
          <td>1.803183</td>
          <td>27.528757</td>
          <td>0.338840</td>
          <td>26.356519</td>
          <td>0.113219</td>
          <td>26.037139</td>
          <td>0.139029</td>
          <td>26.193932</td>
          <td>0.292213</td>
          <td>25.193169</td>
          <td>0.274016</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.979464</td>
          <td>0.268390</td>
          <td>26.282583</td>
          <td>0.126760</td>
          <td>26.121064</td>
          <td>0.098393</td>
          <td>25.850803</td>
          <td>0.126602</td>
          <td>25.663421</td>
          <td>0.200723</td>
          <td>25.337586</td>
          <td>0.327483</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.761815</td>
          <td>0.179037</td>
          <td>26.710291</td>
          <td>0.152237</td>
          <td>26.412988</td>
          <td>0.189674</td>
          <td>25.693287</td>
          <td>0.191445</td>
          <td>25.503468</td>
          <td>0.348003</td>
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
