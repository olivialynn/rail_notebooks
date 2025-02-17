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

    <pzflow.flow.Flow at 0x7f7c88a50610>



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
          <td>27.358604</td>
          <td>0.705260</td>
          <td>26.523887</td>
          <td>0.141370</td>
          <td>26.024324</td>
          <td>0.080558</td>
          <td>25.172526</td>
          <td>0.061871</td>
          <td>24.650331</td>
          <td>0.074572</td>
          <td>24.138964</td>
          <td>0.106745</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.644461</td>
          <td>0.851121</td>
          <td>27.142943</td>
          <td>0.238461</td>
          <td>26.526729</td>
          <td>0.125072</td>
          <td>26.645305</td>
          <td>0.221650</td>
          <td>25.457373</td>
          <td>0.150846</td>
          <td>25.712939</td>
          <td>0.395325</td>
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
          <td>28.223238</td>
          <td>0.497599</td>
          <td>26.172806</td>
          <td>0.148597</td>
          <td>25.077221</td>
          <td>0.108531</td>
          <td>24.155007</td>
          <td>0.108252</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.458785</td>
          <td>1.217964</td>
          <td>26.977049</td>
          <td>0.184002</td>
          <td>26.134994</td>
          <td>0.143845</td>
          <td>25.438348</td>
          <td>0.148403</td>
          <td>25.292453</td>
          <td>0.283417</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.899095</td>
          <td>0.232691</td>
          <td>26.194262</td>
          <td>0.106221</td>
          <td>25.974430</td>
          <td>0.077086</td>
          <td>25.772732</td>
          <td>0.105037</td>
          <td>25.489665</td>
          <td>0.155080</td>
          <td>24.753569</td>
          <td>0.181261</td>
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
          <td>27.085145</td>
          <td>0.583253</td>
          <td>26.325971</td>
          <td>0.119131</td>
          <td>25.415425</td>
          <td>0.046960</td>
          <td>25.026639</td>
          <td>0.054357</td>
          <td>24.864957</td>
          <td>0.090108</td>
          <td>24.569765</td>
          <td>0.154994</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>32.628329</td>
          <td>5.181390</td>
          <td>26.661528</td>
          <td>0.159083</td>
          <td>25.953550</td>
          <td>0.075677</td>
          <td>25.230549</td>
          <td>0.065137</td>
          <td>24.764423</td>
          <td>0.082474</td>
          <td>24.012907</td>
          <td>0.095589</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>28.704420</td>
          <td>1.547167</td>
          <td>26.625157</td>
          <td>0.154210</td>
          <td>26.434534</td>
          <td>0.115441</td>
          <td>26.404756</td>
          <td>0.181111</td>
          <td>25.663402</td>
          <td>0.179821</td>
          <td>24.998556</td>
          <td>0.222648</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.540985</td>
          <td>0.389193</td>
          <td>26.167608</td>
          <td>0.103776</td>
          <td>26.132256</td>
          <td>0.088595</td>
          <td>25.840174</td>
          <td>0.111409</td>
          <td>25.653865</td>
          <td>0.178373</td>
          <td>25.509403</td>
          <td>0.337187</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.125386</td>
          <td>0.600144</td>
          <td>27.259802</td>
          <td>0.262483</td>
          <td>26.737258</td>
          <td>0.149991</td>
          <td>26.268241</td>
          <td>0.161255</td>
          <td>25.974802</td>
          <td>0.233430</td>
          <td>25.581735</td>
          <td>0.356960</td>
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
          <td>27.063758</td>
          <td>0.631205</td>
          <td>27.347225</td>
          <td>0.321164</td>
          <td>26.117204</td>
          <td>0.102758</td>
          <td>25.139687</td>
          <td>0.071242</td>
          <td>24.733564</td>
          <td>0.094363</td>
          <td>24.338491</td>
          <td>0.149613</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.059117</td>
          <td>1.182366</td>
          <td>27.887145</td>
          <td>0.486846</td>
          <td>26.457111</td>
          <td>0.138100</td>
          <td>26.329499</td>
          <td>0.199834</td>
          <td>26.268588</td>
          <td>0.343571</td>
          <td>25.546815</td>
          <td>0.402698</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.977367</td>
          <td>2.597953</td>
          <td>27.850593</td>
          <td>0.440313</td>
          <td>26.059078</td>
          <td>0.162469</td>
          <td>24.957459</td>
          <td>0.117330</td>
          <td>24.008651</td>
          <td>0.115055</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.198596</td>
          <td>0.720832</td>
          <td>28.567528</td>
          <td>0.819909</td>
          <td>27.472991</td>
          <td>0.342015</td>
          <td>26.032201</td>
          <td>0.166087</td>
          <td>25.575436</td>
          <td>0.207860</td>
          <td>24.851132</td>
          <td>0.246076</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.070573</td>
          <td>0.298486</td>
          <td>26.107880</td>
          <td>0.113585</td>
          <td>26.070465</td>
          <td>0.098669</td>
          <td>25.698374</td>
          <td>0.116409</td>
          <td>25.658005</td>
          <td>0.209009</td>
          <td>24.669075</td>
          <td>0.198198</td>
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
          <td>27.542822</td>
          <td>0.878566</td>
          <td>26.407568</td>
          <td>0.149889</td>
          <td>25.429554</td>
          <td>0.057214</td>
          <td>25.113893</td>
          <td>0.071178</td>
          <td>24.686991</td>
          <td>0.092504</td>
          <td>24.193981</td>
          <td>0.134935</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.160340</td>
          <td>0.676475</td>
          <td>26.972673</td>
          <td>0.237807</td>
          <td>26.221474</td>
          <td>0.113021</td>
          <td>25.164271</td>
          <td>0.073125</td>
          <td>24.706517</td>
          <td>0.092534</td>
          <td>24.407599</td>
          <td>0.159400</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.336712</td>
          <td>0.371607</td>
          <td>26.496269</td>
          <td>0.160539</td>
          <td>26.232160</td>
          <td>0.115057</td>
          <td>25.931298</td>
          <td>0.144228</td>
          <td>26.043892</td>
          <td>0.290451</td>
          <td>25.882862</td>
          <td>0.523896</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.354664</td>
          <td>0.381619</td>
          <td>26.305645</td>
          <td>0.138593</td>
          <td>26.119461</td>
          <td>0.106234</td>
          <td>25.934245</td>
          <td>0.147345</td>
          <td>25.890169</td>
          <td>0.260792</td>
          <td>25.347239</td>
          <td>0.354751</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.537413</td>
          <td>0.432791</td>
          <td>26.643200</td>
          <td>0.181453</td>
          <td>26.522485</td>
          <td>0.147508</td>
          <td>26.460143</td>
          <td>0.225044</td>
          <td>26.416123</td>
          <td>0.388932</td>
          <td>25.949919</td>
          <td>0.548771</td>
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
          <td>27.294321</td>
          <td>0.270007</td>
          <td>26.094472</td>
          <td>0.085708</td>
          <td>25.184401</td>
          <td>0.062534</td>
          <td>24.691891</td>
          <td>0.077371</td>
          <td>24.058832</td>
          <td>0.099531</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.484578</td>
          <td>0.314996</td>
          <td>26.782956</td>
          <td>0.156127</td>
          <td>26.443032</td>
          <td>0.187249</td>
          <td>25.864757</td>
          <td>0.213210</td>
          <td>25.497259</td>
          <td>0.334257</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.262817</td>
          <td>1.142313</td>
          <td>28.322401</td>
          <td>0.572256</td>
          <td>25.960271</td>
          <td>0.134562</td>
          <td>25.109506</td>
          <td>0.121056</td>
          <td>24.199191</td>
          <td>0.122360</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.986473</td>
          <td>0.504708</td>
          <td>26.347649</td>
          <td>0.215946</td>
          <td>25.467696</td>
          <td>0.189222</td>
          <td>25.764684</td>
          <td>0.501722</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.681402</td>
          <td>0.194261</td>
          <td>26.244514</td>
          <td>0.111116</td>
          <td>25.967337</td>
          <td>0.076714</td>
          <td>25.713822</td>
          <td>0.099906</td>
          <td>25.660789</td>
          <td>0.179668</td>
          <td>24.887518</td>
          <td>0.203212</td>
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
          <td>27.223798</td>
          <td>0.671816</td>
          <td>26.228077</td>
          <td>0.117150</td>
          <td>25.490892</td>
          <td>0.054383</td>
          <td>25.106371</td>
          <td>0.063413</td>
          <td>24.810353</td>
          <td>0.092922</td>
          <td>24.915504</td>
          <td>0.224463</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.848717</td>
          <td>0.974494</td>
          <td>26.822845</td>
          <td>0.185000</td>
          <td>25.880748</td>
          <td>0.072147</td>
          <td>25.167064</td>
          <td>0.062657</td>
          <td>24.737424</td>
          <td>0.081877</td>
          <td>24.200864</td>
          <td>0.114600</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.489123</td>
          <td>0.143071</td>
          <td>26.402160</td>
          <td>0.117809</td>
          <td>26.844043</td>
          <td>0.273861</td>
          <td>26.232815</td>
          <td>0.301506</td>
          <td>26.547537</td>
          <td>0.751922</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.527658</td>
          <td>0.413945</td>
          <td>26.511886</td>
          <td>0.154426</td>
          <td>26.159583</td>
          <td>0.101770</td>
          <td>25.831762</td>
          <td>0.124529</td>
          <td>25.597249</td>
          <td>0.189850</td>
          <td>25.087857</td>
          <td>0.267834</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.328812</td>
          <td>0.705843</td>
          <td>26.720532</td>
          <td>0.172877</td>
          <td>26.610629</td>
          <td>0.139736</td>
          <td>26.265107</td>
          <td>0.167321</td>
          <td>25.854530</td>
          <td>0.219145</td>
          <td>25.256852</td>
          <td>0.285799</td>
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
