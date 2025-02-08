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

    <pzflow.flow.Flow at 0x7fba1f327d00>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.715199</td>
          <td>0.166535</td>
          <td>25.953530</td>
          <td>0.075676</td>
          <td>25.125786</td>
          <td>0.059358</td>
          <td>24.759060</td>
          <td>0.082085</td>
          <td>24.000808</td>
          <td>0.094579</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.417118</td>
          <td>3.029778</td>
          <td>28.236732</td>
          <td>0.558084</td>
          <td>26.739690</td>
          <td>0.150305</td>
          <td>26.253135</td>
          <td>0.159187</td>
          <td>25.624826</td>
          <td>0.174031</td>
          <td>24.966820</td>
          <td>0.216841</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.344672</td>
          <td>0.543819</td>
          <td>25.983473</td>
          <td>0.126196</td>
          <td>24.936913</td>
          <td>0.095988</td>
          <td>24.376743</td>
          <td>0.131265</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.433646</td>
          <td>0.741731</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.089251</td>
          <td>0.450258</td>
          <td>26.153401</td>
          <td>0.146140</td>
          <td>25.670304</td>
          <td>0.180875</td>
          <td>26.104929</td>
          <td>0.530541</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.326871</td>
          <td>0.329116</td>
          <td>26.077620</td>
          <td>0.095917</td>
          <td>25.931592</td>
          <td>0.074222</td>
          <td>25.625170</td>
          <td>0.092291</td>
          <td>25.817526</td>
          <td>0.204765</td>
          <td>25.041601</td>
          <td>0.230749</td>
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
          <td>26.497437</td>
          <td>0.138186</td>
          <td>25.463299</td>
          <td>0.048999</td>
          <td>25.058103</td>
          <td>0.055897</td>
          <td>24.927204</td>
          <td>0.095173</td>
          <td>24.515070</td>
          <td>0.147891</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.739948</td>
          <td>0.452958</td>
          <td>26.305128</td>
          <td>0.116993</td>
          <td>26.067971</td>
          <td>0.083719</td>
          <td>25.140063</td>
          <td>0.060114</td>
          <td>24.800530</td>
          <td>0.085141</td>
          <td>24.229548</td>
          <td>0.115522</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>28.192863</td>
          <td>1.182478</td>
          <td>26.575585</td>
          <td>0.147794</td>
          <td>26.325217</td>
          <td>0.104937</td>
          <td>26.329629</td>
          <td>0.169920</td>
          <td>25.748854</td>
          <td>0.193283</td>
          <td>25.700279</td>
          <td>0.391479</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.727883</td>
          <td>0.201795</td>
          <td>26.134991</td>
          <td>0.100859</td>
          <td>26.029943</td>
          <td>0.080958</td>
          <td>25.991472</td>
          <td>0.127074</td>
          <td>25.747874</td>
          <td>0.193124</td>
          <td>24.947022</td>
          <td>0.213288</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.021389</td>
          <td>0.257305</td>
          <td>26.671482</td>
          <td>0.160441</td>
          <td>26.421096</td>
          <td>0.114098</td>
          <td>26.172433</td>
          <td>0.148550</td>
          <td>25.729031</td>
          <td>0.190080</td>
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
          <td>27.949917</td>
          <td>1.111071</td>
          <td>26.798980</td>
          <td>0.205090</td>
          <td>26.088902</td>
          <td>0.100244</td>
          <td>25.277980</td>
          <td>0.080498</td>
          <td>24.631911</td>
          <td>0.086298</td>
          <td>24.187775</td>
          <td>0.131394</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.054407</td>
          <td>0.550257</td>
          <td>26.649134</td>
          <td>0.162836</td>
          <td>26.370057</td>
          <td>0.206748</td>
          <td>25.481818</td>
          <td>0.180176</td>
          <td>25.085264</td>
          <td>0.279532</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.745383</td>
          <td>0.509532</td>
          <td>29.158392</td>
          <td>1.141783</td>
          <td>28.727676</td>
          <td>0.816583</td>
          <td>25.818671</td>
          <td>0.132147</td>
          <td>25.083026</td>
          <td>0.130831</td>
          <td>24.264109</td>
          <td>0.143530</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.694354</td>
          <td>1.553890</td>
          <td>27.102140</td>
          <td>0.253700</td>
          <td>26.006627</td>
          <td>0.162503</td>
          <td>25.411741</td>
          <td>0.181103</td>
          <td>25.854518</td>
          <td>0.537443</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.729560</td>
          <td>0.225953</td>
          <td>26.337917</td>
          <td>0.138622</td>
          <td>25.919193</td>
          <td>0.086393</td>
          <td>25.672801</td>
          <td>0.113845</td>
          <td>25.465167</td>
          <td>0.177672</td>
          <td>24.900597</td>
          <td>0.240355</td>
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
          <td>27.133372</td>
          <td>0.670772</td>
          <td>26.277465</td>
          <td>0.134018</td>
          <td>25.375928</td>
          <td>0.054556</td>
          <td>25.093813</td>
          <td>0.069925</td>
          <td>24.724647</td>
          <td>0.095612</td>
          <td>24.939171</td>
          <td>0.253090</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.918849</td>
          <td>0.571223</td>
          <td>26.851194</td>
          <td>0.215004</td>
          <td>26.177367</td>
          <td>0.108756</td>
          <td>25.305243</td>
          <td>0.082815</td>
          <td>24.853193</td>
          <td>0.105223</td>
          <td>23.894825</td>
          <td>0.102263</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>29.596683</td>
          <td>2.409914</td>
          <td>27.068942</td>
          <td>0.259266</td>
          <td>26.245355</td>
          <td>0.116386</td>
          <td>26.554618</td>
          <td>0.243992</td>
          <td>25.760885</td>
          <td>0.230405</td>
          <td>25.359108</td>
          <td>0.352006</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.744908</td>
          <td>0.512331</td>
          <td>26.340160</td>
          <td>0.142772</td>
          <td>26.082237</td>
          <td>0.102832</td>
          <td>25.690827</td>
          <td>0.119390</td>
          <td>25.879818</td>
          <td>0.258593</td>
          <td>25.171266</td>
          <td>0.308537</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.254372</td>
          <td>0.347768</td>
          <td>26.698808</td>
          <td>0.190176</td>
          <td>26.628324</td>
          <td>0.161504</td>
          <td>26.050022</td>
          <td>0.159242</td>
          <td>25.567597</td>
          <td>0.195545</td>
          <td>25.312983</td>
          <td>0.338578</td>
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
          <td>26.338252</td>
          <td>0.332125</td>
          <td>27.179424</td>
          <td>0.245767</td>
          <td>25.976511</td>
          <td>0.077238</td>
          <td>25.193670</td>
          <td>0.063050</td>
          <td>24.719460</td>
          <td>0.079278</td>
          <td>23.987590</td>
          <td>0.093501</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.273060</td>
          <td>0.665604</td>
          <td>27.082625</td>
          <td>0.227024</td>
          <td>26.532444</td>
          <td>0.125811</td>
          <td>26.159059</td>
          <td>0.146996</td>
          <td>25.962948</td>
          <td>0.231356</td>
          <td>25.592557</td>
          <td>0.360317</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.636495</td>
          <td>2.088569</td>
          <td>25.978225</td>
          <td>0.136665</td>
          <td>25.211679</td>
          <td>0.132264</td>
          <td>24.566300</td>
          <td>0.167805</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.012355</td>
          <td>1.188200</td>
          <td>30.011647</td>
          <td>1.800317</td>
          <td>27.158187</td>
          <td>0.264726</td>
          <td>25.977474</td>
          <td>0.157939</td>
          <td>25.406008</td>
          <td>0.179606</td>
          <td>25.776765</td>
          <td>0.506206</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.715096</td>
          <td>0.890676</td>
          <td>26.128573</td>
          <td>0.100417</td>
          <td>25.909426</td>
          <td>0.072886</td>
          <td>25.717294</td>
          <td>0.100210</td>
          <td>25.228967</td>
          <td>0.124032</td>
          <td>25.860918</td>
          <td>0.443197</td>
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
          <td>27.498281</td>
          <td>0.807010</td>
          <td>26.510648</td>
          <td>0.149531</td>
          <td>25.469082</td>
          <td>0.053341</td>
          <td>25.190975</td>
          <td>0.068348</td>
          <td>24.789664</td>
          <td>0.091249</td>
          <td>24.673778</td>
          <td>0.183274</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.721808</td>
          <td>1.570356</td>
          <td>26.842231</td>
          <td>0.188053</td>
          <td>26.033806</td>
          <td>0.082590</td>
          <td>25.294448</td>
          <td>0.070144</td>
          <td>24.846354</td>
          <td>0.090120</td>
          <td>24.105657</td>
          <td>0.105464</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.846544</td>
          <td>0.504850</td>
          <td>26.788293</td>
          <td>0.184653</td>
          <td>26.310986</td>
          <td>0.108810</td>
          <td>26.109436</td>
          <td>0.147955</td>
          <td>25.814113</td>
          <td>0.213901</td>
          <td>26.082554</td>
          <td>0.544326</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.068223</td>
          <td>0.615700</td>
          <td>26.000219</td>
          <td>0.099133</td>
          <td>26.198406</td>
          <td>0.105285</td>
          <td>25.783746</td>
          <td>0.119443</td>
          <td>25.533708</td>
          <td>0.179922</td>
          <td>25.934443</td>
          <td>0.516949</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.555316</td>
          <td>0.403037</td>
          <td>26.945162</td>
          <td>0.208914</td>
          <td>26.602649</td>
          <td>0.138778</td>
          <td>26.190751</td>
          <td>0.157029</td>
          <td>25.705907</td>
          <td>0.193492</td>
          <td>25.417590</td>
          <td>0.325137</td>
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
