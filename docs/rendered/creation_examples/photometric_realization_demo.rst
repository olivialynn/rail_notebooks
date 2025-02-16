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

    <pzflow.flow.Flow at 0x7ff7e0c2ed70>



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
          <td>26.166592</td>
          <td>0.289515</td>
          <td>26.697285</td>
          <td>0.164012</td>
          <td>26.100693</td>
          <td>0.086167</td>
          <td>25.233209</td>
          <td>0.065291</td>
          <td>24.630055</td>
          <td>0.073247</td>
          <td>23.866369</td>
          <td>0.084031</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.248819</td>
          <td>0.260137</td>
          <td>26.447096</td>
          <td>0.116711</td>
          <td>26.486739</td>
          <td>0.194095</td>
          <td>26.153060</td>
          <td>0.270233</td>
          <td>25.581792</td>
          <td>0.356975</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.766262</td>
          <td>1.434495</td>
          <td>28.620274</td>
          <td>0.660851</td>
          <td>25.858447</td>
          <td>0.113199</td>
          <td>24.992321</td>
          <td>0.100765</td>
          <td>24.234696</td>
          <td>0.116041</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.525415</td>
          <td>0.683410</td>
          <td>27.632702</td>
          <td>0.315803</td>
          <td>26.142757</td>
          <td>0.144809</td>
          <td>25.286121</td>
          <td>0.130148</td>
          <td>25.689509</td>
          <td>0.388232</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.090239</td>
          <td>0.272161</td>
          <td>26.121640</td>
          <td>0.099687</td>
          <td>25.918289</td>
          <td>0.073354</td>
          <td>25.720737</td>
          <td>0.100363</td>
          <td>25.529705</td>
          <td>0.160484</td>
          <td>24.835603</td>
          <td>0.194263</td>
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
          <td>27.047591</td>
          <td>0.567818</td>
          <td>26.298966</td>
          <td>0.116367</td>
          <td>25.504003</td>
          <td>0.050802</td>
          <td>24.990093</td>
          <td>0.052621</td>
          <td>24.891645</td>
          <td>0.092247</td>
          <td>24.801159</td>
          <td>0.188703</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.796156</td>
          <td>0.178391</td>
          <td>26.094169</td>
          <td>0.085674</td>
          <td>25.094692</td>
          <td>0.057742</td>
          <td>24.842680</td>
          <td>0.088360</td>
          <td>24.212505</td>
          <td>0.113820</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.817801</td>
          <td>0.480110</td>
          <td>26.954602</td>
          <td>0.203876</td>
          <td>26.420468</td>
          <td>0.114036</td>
          <td>26.527950</td>
          <td>0.200939</td>
          <td>26.188714</td>
          <td>0.278183</td>
          <td>26.801735</td>
          <td>0.854535</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.500887</td>
          <td>0.377287</td>
          <td>26.259102</td>
          <td>0.112400</td>
          <td>26.121619</td>
          <td>0.087770</td>
          <td>25.879914</td>
          <td>0.115336</td>
          <td>25.507347</td>
          <td>0.157445</td>
          <td>25.541490</td>
          <td>0.345840</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.762035</td>
          <td>0.173302</td>
          <td>26.475954</td>
          <td>0.119678</td>
          <td>26.259602</td>
          <td>0.160069</td>
          <td>26.049308</td>
          <td>0.248231</td>
          <td>25.960605</td>
          <td>0.477024</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.785633</td>
          <td>0.202809</td>
          <td>25.941507</td>
          <td>0.088078</td>
          <td>25.190716</td>
          <td>0.074529</td>
          <td>24.770870</td>
          <td>0.097502</td>
          <td>24.138722</td>
          <td>0.125931</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.112410</td>
          <td>0.265817</td>
          <td>26.809475</td>
          <td>0.186587</td>
          <td>26.140445</td>
          <td>0.170316</td>
          <td>25.674636</td>
          <td>0.211911</td>
          <td>25.383046</td>
          <td>0.354569</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.920837</td>
          <td>1.104876</td>
          <td>27.842740</td>
          <td>0.479065</td>
          <td>27.883734</td>
          <td>0.451468</td>
          <td>26.028177</td>
          <td>0.158236</td>
          <td>25.143931</td>
          <td>0.137898</td>
          <td>24.387407</td>
          <td>0.159536</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.792999</td>
          <td>0.542960</td>
          <td>28.607571</td>
          <td>0.841310</td>
          <td>27.673063</td>
          <td>0.399769</td>
          <td>26.568731</td>
          <td>0.260121</td>
          <td>25.666250</td>
          <td>0.224213</td>
          <td>25.538236</td>
          <td>0.424710</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.813719</td>
          <td>0.528202</td>
          <td>26.037501</td>
          <td>0.106829</td>
          <td>25.965050</td>
          <td>0.089949</td>
          <td>25.860579</td>
          <td>0.133990</td>
          <td>25.637523</td>
          <td>0.205455</td>
          <td>24.677221</td>
          <td>0.199559</td>
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
          <td>26.284617</td>
          <td>0.134848</td>
          <td>25.576730</td>
          <td>0.065186</td>
          <td>24.988465</td>
          <td>0.063698</td>
          <td>24.980284</td>
          <td>0.119527</td>
          <td>24.987369</td>
          <td>0.263277</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.027799</td>
          <td>0.617074</td>
          <td>26.322528</td>
          <td>0.137262</td>
          <td>26.068719</td>
          <td>0.098898</td>
          <td>25.171294</td>
          <td>0.073581</td>
          <td>24.933734</td>
          <td>0.112885</td>
          <td>24.102636</td>
          <td>0.122568</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.633748</td>
          <td>0.466195</td>
          <td>26.527974</td>
          <td>0.164938</td>
          <td>26.484514</td>
          <td>0.143155</td>
          <td>25.923917</td>
          <td>0.143315</td>
          <td>26.076207</td>
          <td>0.298115</td>
          <td>25.537730</td>
          <td>0.404440</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>30.252182</td>
          <td>3.025778</td>
          <td>26.360075</td>
          <td>0.145237</td>
          <td>26.120863</td>
          <td>0.106364</td>
          <td>25.665662</td>
          <td>0.116806</td>
          <td>25.756684</td>
          <td>0.233668</td>
          <td>25.036586</td>
          <td>0.276775</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.157455</td>
          <td>0.677494</td>
          <td>26.608944</td>
          <td>0.176264</td>
          <td>26.766813</td>
          <td>0.181687</td>
          <td>26.419182</td>
          <td>0.217503</td>
          <td>25.885144</td>
          <td>0.254603</td>
          <td>25.200747</td>
          <td>0.309653</td>
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
          <td>28.903387</td>
          <td>1.701556</td>
          <td>26.721524</td>
          <td>0.167453</td>
          <td>25.988302</td>
          <td>0.078047</td>
          <td>25.176617</td>
          <td>0.062104</td>
          <td>24.820645</td>
          <td>0.086674</td>
          <td>23.937456</td>
          <td>0.089470</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.804843</td>
          <td>0.404878</td>
          <td>26.624650</td>
          <td>0.136260</td>
          <td>26.534255</td>
          <td>0.202197</td>
          <td>25.973079</td>
          <td>0.233305</td>
          <td>25.968909</td>
          <td>0.480382</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.128853</td>
          <td>0.630414</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.254587</td>
          <td>0.544995</td>
          <td>25.980928</td>
          <td>0.136984</td>
          <td>24.931300</td>
          <td>0.103638</td>
          <td>24.301350</td>
          <td>0.133680</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.799450</td>
          <td>1.769392</td>
          <td>28.296333</td>
          <td>0.682935</td>
          <td>27.155650</td>
          <td>0.264179</td>
          <td>26.515207</td>
          <td>0.248095</td>
          <td>25.277235</td>
          <td>0.160972</td>
          <td>24.863521</td>
          <td>0.247747</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.538337</td>
          <td>0.388733</td>
          <td>26.031360</td>
          <td>0.092217</td>
          <td>26.026078</td>
          <td>0.080798</td>
          <td>25.735729</td>
          <td>0.101841</td>
          <td>25.378450</td>
          <td>0.141146</td>
          <td>25.174242</td>
          <td>0.257753</td>
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
          <td>28.135660</td>
          <td>1.186658</td>
          <td>26.328374</td>
          <td>0.127797</td>
          <td>25.474194</td>
          <td>0.053583</td>
          <td>25.012505</td>
          <td>0.058348</td>
          <td>25.064041</td>
          <td>0.116004</td>
          <td>24.806586</td>
          <td>0.204960</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.817894</td>
          <td>0.184228</td>
          <td>25.999140</td>
          <td>0.080103</td>
          <td>25.266220</td>
          <td>0.068412</td>
          <td>24.789083</td>
          <td>0.085691</td>
          <td>24.130932</td>
          <td>0.107819</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>29.691394</td>
          <td>2.400496</td>
          <td>27.035714</td>
          <td>0.227170</td>
          <td>26.253716</td>
          <td>0.103498</td>
          <td>26.444472</td>
          <td>0.196732</td>
          <td>25.598185</td>
          <td>0.178362</td>
          <td>25.031348</td>
          <td>0.239989</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.515392</td>
          <td>0.410078</td>
          <td>26.118745</td>
          <td>0.109944</td>
          <td>26.128747</td>
          <td>0.099058</td>
          <td>25.821994</td>
          <td>0.123478</td>
          <td>26.200133</td>
          <td>0.311838</td>
          <td>24.988086</td>
          <td>0.246815</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.689953</td>
          <td>0.446535</td>
          <td>26.974897</td>
          <td>0.214168</td>
          <td>26.583449</td>
          <td>0.136498</td>
          <td>26.296429</td>
          <td>0.171842</td>
          <td>25.755200</td>
          <td>0.201680</td>
          <td>25.584128</td>
          <td>0.370710</td>
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
