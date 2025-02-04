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

    <pzflow.flow.Flow at 0x7f016f1f3310>



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
          <td>26.970358</td>
          <td>0.206585</td>
          <td>25.922004</td>
          <td>0.073596</td>
          <td>25.263233</td>
          <td>0.067051</td>
          <td>24.869649</td>
          <td>0.090481</td>
          <td>24.118439</td>
          <td>0.104847</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>33.308395</td>
          <td>5.857168</td>
          <td>27.112145</td>
          <td>0.232466</td>
          <td>26.643620</td>
          <td>0.138380</td>
          <td>26.403003</td>
          <td>0.180842</td>
          <td>26.090850</td>
          <td>0.256842</td>
          <td>25.423918</td>
          <td>0.315032</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.310353</td>
          <td>1.261807</td>
          <td>29.531814</td>
          <td>1.267735</td>
          <td>27.966527</td>
          <td>0.410152</td>
          <td>25.981405</td>
          <td>0.125970</td>
          <td>25.065513</td>
          <td>0.107427</td>
          <td>24.407688</td>
          <td>0.134824</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.897661</td>
          <td>0.509294</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.128486</td>
          <td>0.209004</td>
          <td>26.192087</td>
          <td>0.151077</td>
          <td>26.021814</td>
          <td>0.242674</td>
          <td>25.284787</td>
          <td>0.281662</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.929033</td>
          <td>0.238513</td>
          <td>26.008781</td>
          <td>0.090296</td>
          <td>25.925215</td>
          <td>0.073805</td>
          <td>25.719633</td>
          <td>0.100266</td>
          <td>25.511470</td>
          <td>0.158001</td>
          <td>25.534613</td>
          <td>0.343970</td>
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
          <td>30.316879</td>
          <td>2.935970</td>
          <td>26.289502</td>
          <td>0.115413</td>
          <td>25.442302</td>
          <td>0.048094</td>
          <td>25.097802</td>
          <td>0.057902</td>
          <td>24.753600</td>
          <td>0.081691</td>
          <td>24.561250</td>
          <td>0.153868</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.693217</td>
          <td>0.163444</td>
          <td>26.074336</td>
          <td>0.084190</td>
          <td>25.170174</td>
          <td>0.061742</td>
          <td>24.832480</td>
          <td>0.087570</td>
          <td>24.425192</td>
          <td>0.136877</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.868438</td>
          <td>0.189631</td>
          <td>26.367629</td>
          <td>0.108899</td>
          <td>26.390023</td>
          <td>0.178864</td>
          <td>25.818678</td>
          <td>0.204963</td>
          <td>25.153969</td>
          <td>0.253156</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.857978</td>
          <td>0.224904</td>
          <td>26.158557</td>
          <td>0.102959</td>
          <td>26.093796</td>
          <td>0.085646</td>
          <td>26.009358</td>
          <td>0.129059</td>
          <td>25.598811</td>
          <td>0.170224</td>
          <td>25.219726</td>
          <td>0.267149</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.425167</td>
          <td>0.355639</td>
          <td>27.052325</td>
          <td>0.221209</td>
          <td>26.581737</td>
          <td>0.131177</td>
          <td>26.269376</td>
          <td>0.161411</td>
          <td>25.774669</td>
          <td>0.197528</td>
          <td>25.807239</td>
          <td>0.424965</td>
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
          <td>28.298159</td>
          <td>1.346502</td>
          <td>26.963441</td>
          <td>0.235167</td>
          <td>25.982304</td>
          <td>0.091294</td>
          <td>25.114085</td>
          <td>0.069647</td>
          <td>24.717674</td>
          <td>0.093056</td>
          <td>24.039497</td>
          <td>0.115532</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.039194</td>
          <td>0.620531</td>
          <td>27.067491</td>
          <td>0.256234</td>
          <td>26.531091</td>
          <td>0.147180</td>
          <td>26.145028</td>
          <td>0.170981</td>
          <td>26.115042</td>
          <td>0.304051</td>
          <td>25.521158</td>
          <td>0.394817</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.925655</td>
          <td>1.107954</td>
          <td>28.021634</td>
          <td>0.546272</td>
          <td>28.167418</td>
          <td>0.556459</td>
          <td>26.270131</td>
          <td>0.194306</td>
          <td>25.137236</td>
          <td>0.137104</td>
          <td>24.290751</td>
          <td>0.146856</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.920116</td>
          <td>1.020846</td>
          <td>27.349114</td>
          <td>0.309937</td>
          <td>26.300217</td>
          <td>0.208285</td>
          <td>25.450661</td>
          <td>0.187161</td>
          <td>25.337654</td>
          <td>0.363779</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.063264</td>
          <td>0.296738</td>
          <td>26.130852</td>
          <td>0.115877</td>
          <td>26.041544</td>
          <td>0.096199</td>
          <td>25.652310</td>
          <td>0.111830</td>
          <td>25.566329</td>
          <td>0.193529</td>
          <td>25.382741</td>
          <td>0.354523</td>
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
          <td>27.776447</td>
          <td>1.014247</td>
          <td>26.456597</td>
          <td>0.156315</td>
          <td>25.435055</td>
          <td>0.057493</td>
          <td>25.115874</td>
          <td>0.071303</td>
          <td>24.875919</td>
          <td>0.109143</td>
          <td>24.410118</td>
          <td>0.162450</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.549470</td>
          <td>0.874155</td>
          <td>27.164460</td>
          <td>0.278233</td>
          <td>26.110646</td>
          <td>0.102596</td>
          <td>25.384302</td>
          <td>0.088785</td>
          <td>24.853501</td>
          <td>0.105252</td>
          <td>24.246453</td>
          <td>0.138807</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.899388</td>
          <td>1.085973</td>
          <td>26.654093</td>
          <td>0.183570</td>
          <td>26.186836</td>
          <td>0.110601</td>
          <td>26.402404</td>
          <td>0.215061</td>
          <td>25.558106</td>
          <td>0.194501</td>
          <td>27.075164</td>
          <td>1.140574</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.925601</td>
          <td>0.271370</td>
          <td>26.194817</td>
          <td>0.125943</td>
          <td>26.070036</td>
          <td>0.101740</td>
          <td>26.033640</td>
          <td>0.160442</td>
          <td>26.239328</td>
          <td>0.345258</td>
          <td>25.388927</td>
          <td>0.366523</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.376615</td>
          <td>0.382592</td>
          <td>26.944194</td>
          <td>0.233435</td>
          <td>26.649522</td>
          <td>0.164452</td>
          <td>26.575178</td>
          <td>0.247497</td>
          <td>26.053764</td>
          <td>0.292037</td>
          <td>25.439637</td>
          <td>0.373956</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.915720</td>
          <td>0.197352</td>
          <td>25.945432</td>
          <td>0.075146</td>
          <td>25.253666</td>
          <td>0.066494</td>
          <td>24.707566</td>
          <td>0.078450</td>
          <td>24.017930</td>
          <td>0.096024</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.899936</td>
          <td>0.997703</td>
          <td>28.135162</td>
          <td>0.518771</td>
          <td>26.684129</td>
          <td>0.143429</td>
          <td>26.278499</td>
          <td>0.162831</td>
          <td>25.839282</td>
          <td>0.208718</td>
          <td>25.163944</td>
          <td>0.255470</td>
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
          <td>28.684838</td>
          <td>0.735544</td>
          <td>26.244579</td>
          <td>0.171705</td>
          <td>25.047146</td>
          <td>0.114665</td>
          <td>24.237282</td>
          <td>0.126470</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.440870</td>
          <td>0.752785</td>
          <td>27.397859</td>
          <td>0.321195</td>
          <td>26.297646</td>
          <td>0.207109</td>
          <td>25.742154</td>
          <td>0.237963</td>
          <td>24.797673</td>
          <td>0.234650</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.445074</td>
          <td>0.361543</td>
          <td>26.137494</td>
          <td>0.101204</td>
          <td>25.918402</td>
          <td>0.073467</td>
          <td>25.822500</td>
          <td>0.109867</td>
          <td>25.570312</td>
          <td>0.166371</td>
          <td>25.258618</td>
          <td>0.276120</td>
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
          <td>27.103477</td>
          <td>0.617998</td>
          <td>26.388612</td>
          <td>0.134627</td>
          <td>25.312056</td>
          <td>0.046401</td>
          <td>25.049381</td>
          <td>0.060288</td>
          <td>25.119365</td>
          <td>0.121720</td>
          <td>24.705377</td>
          <td>0.188234</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.672809</td>
          <td>0.434838</td>
          <td>27.025557</td>
          <td>0.219291</td>
          <td>26.114035</td>
          <td>0.088638</td>
          <td>25.293121</td>
          <td>0.070061</td>
          <td>24.999355</td>
          <td>0.103065</td>
          <td>24.368498</td>
          <td>0.132547</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.954147</td>
          <td>1.054297</td>
          <td>26.524397</td>
          <td>0.147473</td>
          <td>26.444565</td>
          <td>0.122232</td>
          <td>26.219442</td>
          <td>0.162571</td>
          <td>25.422817</td>
          <td>0.153595</td>
          <td>25.837284</td>
          <td>0.454155</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.399085</td>
          <td>0.374876</td>
          <td>26.167828</td>
          <td>0.114744</td>
          <td>26.082088</td>
          <td>0.095087</td>
          <td>26.028676</td>
          <td>0.147606</td>
          <td>26.151226</td>
          <td>0.299844</td>
          <td>25.386263</td>
          <td>0.340359</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.548414</td>
          <td>0.816284</td>
          <td>26.582271</td>
          <td>0.153648</td>
          <td>26.490369</td>
          <td>0.125937</td>
          <td>26.407685</td>
          <td>0.188827</td>
          <td>26.023353</td>
          <td>0.251985</td>
          <td>25.672350</td>
          <td>0.396957</td>
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
