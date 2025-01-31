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

    <pzflow.flow.Flow at 0x7f6332a309a0>



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
          <td>27.174875</td>
          <td>0.244822</td>
          <td>26.049058</td>
          <td>0.082334</td>
          <td>25.233079</td>
          <td>0.065283</td>
          <td>24.658530</td>
          <td>0.075114</td>
          <td>24.110010</td>
          <td>0.104077</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>31.040317</td>
          <td>3.623452</td>
          <td>27.562502</td>
          <td>0.334883</td>
          <td>26.691772</td>
          <td>0.144241</td>
          <td>26.331402</td>
          <td>0.170177</td>
          <td>25.718068</td>
          <td>0.188330</td>
          <td>25.602063</td>
          <td>0.362690</td>
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
          <td>30.404476</td>
          <td>1.817510</td>
          <td>25.893908</td>
          <td>0.116750</td>
          <td>25.137538</td>
          <td>0.114394</td>
          <td>24.205294</td>
          <td>0.113107</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.120021</td>
          <td>1.134715</td>
          <td>28.024411</td>
          <td>0.477712</td>
          <td>27.414123</td>
          <td>0.264690</td>
          <td>26.600090</td>
          <td>0.213451</td>
          <td>25.527212</td>
          <td>0.160142</td>
          <td>26.584827</td>
          <td>0.741777</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.147863</td>
          <td>0.285171</td>
          <td>26.256240</td>
          <td>0.112120</td>
          <td>26.087469</td>
          <td>0.085170</td>
          <td>25.704692</td>
          <td>0.098962</td>
          <td>25.478423</td>
          <td>0.153594</td>
          <td>25.198991</td>
          <td>0.262664</td>
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
          <td>26.258881</td>
          <td>0.112378</td>
          <td>25.445459</td>
          <td>0.048229</td>
          <td>25.032450</td>
          <td>0.054638</td>
          <td>24.776342</td>
          <td>0.083346</td>
          <td>25.177491</td>
          <td>0.258085</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.456231</td>
          <td>1.364095</td>
          <td>26.891110</td>
          <td>0.193288</td>
          <td>25.940846</td>
          <td>0.074832</td>
          <td>25.228113</td>
          <td>0.064996</td>
          <td>25.008145</td>
          <td>0.102171</td>
          <td>24.196518</td>
          <td>0.112245</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.547676</td>
          <td>0.799632</td>
          <td>26.807035</td>
          <td>0.180042</td>
          <td>26.473691</td>
          <td>0.119442</td>
          <td>26.129939</td>
          <td>0.143220</td>
          <td>26.123445</td>
          <td>0.263783</td>
          <td>25.565634</td>
          <td>0.352475</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.140984</td>
          <td>0.283589</td>
          <td>26.169713</td>
          <td>0.103968</td>
          <td>26.012056</td>
          <td>0.079690</td>
          <td>25.793666</td>
          <td>0.106976</td>
          <td>25.829887</td>
          <td>0.206896</td>
          <td>25.247173</td>
          <td>0.273189</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.040165</td>
          <td>1.083633</td>
          <td>26.660582</td>
          <td>0.158954</td>
          <td>26.683341</td>
          <td>0.143199</td>
          <td>26.325765</td>
          <td>0.169362</td>
          <td>25.967969</td>
          <td>0.232113</td>
          <td>25.090185</td>
          <td>0.240209</td>
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
          <td>27.215067</td>
          <td>0.700458</td>
          <td>26.825401</td>
          <td>0.209673</td>
          <td>25.986880</td>
          <td>0.091662</td>
          <td>25.211067</td>
          <td>0.075882</td>
          <td>24.753023</td>
          <td>0.095988</td>
          <td>24.066989</td>
          <td>0.118328</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.613020</td>
          <td>0.455272</td>
          <td>27.149485</td>
          <td>0.273963</td>
          <td>26.469567</td>
          <td>0.139591</td>
          <td>26.868838</td>
          <td>0.311177</td>
          <td>26.282813</td>
          <td>0.347445</td>
          <td>24.916790</td>
          <td>0.243558</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.578555</td>
          <td>1.567656</td>
          <td>32.211799</td>
          <td>3.762794</td>
          <td>28.015807</td>
          <td>0.498202</td>
          <td>25.860895</td>
          <td>0.137056</td>
          <td>24.919485</td>
          <td>0.113515</td>
          <td>24.161902</td>
          <td>0.131418</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.654640</td>
          <td>0.866936</td>
          <td>27.617206</td>
          <td>0.382879</td>
          <td>26.286066</td>
          <td>0.205832</td>
          <td>25.331463</td>
          <td>0.169173</td>
          <td>26.308236</td>
          <td>0.737609</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.249107</td>
          <td>0.716896</td>
          <td>26.113286</td>
          <td>0.114120</td>
          <td>25.955076</td>
          <td>0.089164</td>
          <td>25.894158</td>
          <td>0.137931</td>
          <td>25.359280</td>
          <td>0.162366</td>
          <td>24.857993</td>
          <td>0.232038</td>
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
          <td>26.208315</td>
          <td>0.337905</td>
          <td>26.641745</td>
          <td>0.182967</td>
          <td>25.451554</td>
          <td>0.058341</td>
          <td>25.068405</td>
          <td>0.068370</td>
          <td>24.769908</td>
          <td>0.099482</td>
          <td>24.957435</td>
          <td>0.256908</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.938962</td>
          <td>0.231271</td>
          <td>26.119172</td>
          <td>0.103364</td>
          <td>25.278511</td>
          <td>0.080886</td>
          <td>24.786763</td>
          <td>0.099281</td>
          <td>24.116608</td>
          <td>0.124063</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.462479</td>
          <td>0.409511</td>
          <td>26.537306</td>
          <td>0.166254</td>
          <td>26.077988</td>
          <td>0.100565</td>
          <td>26.195228</td>
          <td>0.180681</td>
          <td>25.605866</td>
          <td>0.202466</td>
          <td>25.213038</td>
          <td>0.313517</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.414096</td>
          <td>0.399540</td>
          <td>26.240310</td>
          <td>0.130998</td>
          <td>26.106394</td>
          <td>0.105028</td>
          <td>25.981254</td>
          <td>0.153409</td>
          <td>25.572568</td>
          <td>0.200418</td>
          <td>25.778831</td>
          <td>0.493148</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.831376</td>
          <td>0.538306</td>
          <td>27.053200</td>
          <td>0.255355</td>
          <td>26.737565</td>
          <td>0.177240</td>
          <td>26.438405</td>
          <td>0.221013</td>
          <td>26.185996</td>
          <td>0.324673</td>
          <td>25.769798</td>
          <td>0.480867</td>
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
          <td>28.016071</td>
          <td>1.068552</td>
          <td>26.637215</td>
          <td>0.155827</td>
          <td>26.008914</td>
          <td>0.079480</td>
          <td>25.230741</td>
          <td>0.065157</td>
          <td>24.822784</td>
          <td>0.086837</td>
          <td>23.878748</td>
          <td>0.084964</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.593484</td>
          <td>0.343437</td>
          <td>26.754868</td>
          <td>0.152415</td>
          <td>26.482826</td>
          <td>0.193641</td>
          <td>25.821319</td>
          <td>0.205602</td>
          <td>25.180881</td>
          <td>0.259039</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>25.869122</td>
          <td>0.124352</td>
          <td>24.952950</td>
          <td>0.105618</td>
          <td>24.199944</td>
          <td>0.122440</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>29.855282</td>
          <td>2.683994</td>
          <td>29.909447</td>
          <td>1.718349</td>
          <td>27.556582</td>
          <td>0.364068</td>
          <td>26.359999</td>
          <td>0.218180</td>
          <td>25.461455</td>
          <td>0.188228</td>
          <td>25.169702</td>
          <td>0.317529</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.193623</td>
          <td>0.630110</td>
          <td>26.174033</td>
          <td>0.104489</td>
          <td>26.093003</td>
          <td>0.085708</td>
          <td>25.740402</td>
          <td>0.102259</td>
          <td>25.465228</td>
          <td>0.152076</td>
          <td>25.133318</td>
          <td>0.249241</td>
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
          <td>27.305109</td>
          <td>0.710059</td>
          <td>26.417415</td>
          <td>0.138013</td>
          <td>25.456335</td>
          <td>0.052741</td>
          <td>25.042139</td>
          <td>0.059902</td>
          <td>25.081264</td>
          <td>0.117755</td>
          <td>25.037933</td>
          <td>0.248370</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.965658</td>
          <td>0.540322</td>
          <td>26.792884</td>
          <td>0.180371</td>
          <td>25.999652</td>
          <td>0.080139</td>
          <td>25.254730</td>
          <td>0.067720</td>
          <td>24.981200</td>
          <td>0.101440</td>
          <td>24.127832</td>
          <td>0.107528</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.311830</td>
          <td>0.701640</td>
          <td>26.587696</td>
          <td>0.155694</td>
          <td>26.272575</td>
          <td>0.105219</td>
          <td>26.494442</td>
          <td>0.205162</td>
          <td>25.932876</td>
          <td>0.236084</td>
          <td>25.126768</td>
          <td>0.259568</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.598346</td>
          <td>0.195824</td>
          <td>25.978294</td>
          <td>0.097248</td>
          <td>26.088888</td>
          <td>0.095656</td>
          <td>25.984360</td>
          <td>0.142086</td>
          <td>25.945486</td>
          <td>0.253691</td>
          <td>25.413987</td>
          <td>0.347884</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.517436</td>
          <td>1.430819</td>
          <td>26.522891</td>
          <td>0.146019</td>
          <td>26.703626</td>
          <td>0.151369</td>
          <td>26.365319</td>
          <td>0.182184</td>
          <td>25.900742</td>
          <td>0.227729</td>
          <td>26.195957</td>
          <td>0.585526</td>
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
