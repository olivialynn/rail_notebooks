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

    <pzflow.flow.Flow at 0x7fb31bb498a0>



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
          <td>26.568614</td>
          <td>0.146912</td>
          <td>26.207630</td>
          <td>0.094663</td>
          <td>25.200645</td>
          <td>0.063433</td>
          <td>24.754884</td>
          <td>0.081784</td>
          <td>24.008491</td>
          <td>0.095219</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.801408</td>
          <td>1.621592</td>
          <td>27.685685</td>
          <td>0.368930</td>
          <td>26.704347</td>
          <td>0.145810</td>
          <td>26.234630</td>
          <td>0.156687</td>
          <td>25.676664</td>
          <td>0.181852</td>
          <td>25.293937</td>
          <td>0.283758</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.548136</td>
          <td>2.056397</td>
          <td>28.168290</td>
          <td>0.477727</td>
          <td>25.991288</td>
          <td>0.127054</td>
          <td>24.848150</td>
          <td>0.088786</td>
          <td>24.196124</td>
          <td>0.112206</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.986327</td>
          <td>0.923023</td>
          <td>27.319332</td>
          <td>0.244891</td>
          <td>26.354368</td>
          <td>0.173532</td>
          <td>25.629236</td>
          <td>0.174684</td>
          <td>25.308077</td>
          <td>0.287023</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.172673</td>
          <td>0.290938</td>
          <td>26.128524</td>
          <td>0.100290</td>
          <td>25.847101</td>
          <td>0.068876</td>
          <td>25.732185</td>
          <td>0.101375</td>
          <td>25.494588</td>
          <td>0.155735</td>
          <td>24.637296</td>
          <td>0.164204</td>
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
          <td>26.495860</td>
          <td>0.375817</td>
          <td>26.430201</td>
          <td>0.130393</td>
          <td>25.334528</td>
          <td>0.043706</td>
          <td>25.074326</td>
          <td>0.056708</td>
          <td>24.889491</td>
          <td>0.092073</td>
          <td>24.987486</td>
          <td>0.220607</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.964127</td>
          <td>1.036264</td>
          <td>26.857864</td>
          <td>0.187947</td>
          <td>26.251150</td>
          <td>0.098348</td>
          <td>25.171722</td>
          <td>0.061826</td>
          <td>24.775197</td>
          <td>0.083262</td>
          <td>24.271907</td>
          <td>0.119858</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.155748</td>
          <td>0.286993</td>
          <td>26.878848</td>
          <td>0.191302</td>
          <td>26.444824</td>
          <td>0.116480</td>
          <td>26.173978</td>
          <td>0.148747</td>
          <td>25.967022</td>
          <td>0.231931</td>
          <td>25.302283</td>
          <td>0.285681</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.241030</td>
          <td>0.307360</td>
          <td>26.293630</td>
          <td>0.115829</td>
          <td>26.081661</td>
          <td>0.084735</td>
          <td>25.781185</td>
          <td>0.105816</td>
          <td>25.534578</td>
          <td>0.161153</td>
          <td>25.250809</td>
          <td>0.273998</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.619304</td>
          <td>0.837531</td>
          <td>26.920966</td>
          <td>0.198203</td>
          <td>26.579125</td>
          <td>0.130881</td>
          <td>26.151692</td>
          <td>0.145926</td>
          <td>25.745760</td>
          <td>0.192780</td>
          <td>25.356647</td>
          <td>0.298491</td>
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
          <td>26.518843</td>
          <td>0.423917</td>
          <td>27.012751</td>
          <td>0.244930</td>
          <td>26.030826</td>
          <td>0.095268</td>
          <td>25.144515</td>
          <td>0.071547</td>
          <td>24.724005</td>
          <td>0.093574</td>
          <td>23.928478</td>
          <td>0.104868</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.462446</td>
          <td>0.825102</td>
          <td>26.913138</td>
          <td>0.225609</td>
          <td>26.584940</td>
          <td>0.154139</td>
          <td>25.968513</td>
          <td>0.147031</td>
          <td>26.066199</td>
          <td>0.292333</td>
          <td>25.024267</td>
          <td>0.265998</td>
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
          <td>28.181705</td>
          <td>0.562208</td>
          <td>25.750471</td>
          <td>0.124569</td>
          <td>24.918329</td>
          <td>0.113401</td>
          <td>24.408343</td>
          <td>0.162415</td>
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
          <td>27.280000</td>
          <td>0.293195</td>
          <td>26.092201</td>
          <td>0.174783</td>
          <td>25.668091</td>
          <td>0.224556</td>
          <td>25.390931</td>
          <td>0.379203</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.221367</td>
          <td>0.336595</td>
          <td>26.084219</td>
          <td>0.111269</td>
          <td>25.927369</td>
          <td>0.087017</td>
          <td>25.665767</td>
          <td>0.113150</td>
          <td>25.459520</td>
          <td>0.176823</td>
          <td>25.162852</td>
          <td>0.297651</td>
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
          <td>30.460756</td>
          <td>3.213956</td>
          <td>26.195332</td>
          <td>0.124831</td>
          <td>25.396003</td>
          <td>0.055536</td>
          <td>25.402737</td>
          <td>0.091823</td>
          <td>24.887135</td>
          <td>0.110216</td>
          <td>24.621626</td>
          <td>0.194355</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.162208</td>
          <td>2.919459</td>
          <td>27.061425</td>
          <td>0.255815</td>
          <td>26.242684</td>
          <td>0.115129</td>
          <td>25.351199</td>
          <td>0.086236</td>
          <td>24.767148</td>
          <td>0.097589</td>
          <td>24.184631</td>
          <td>0.131592</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>25.995335</td>
          <td>0.283385</td>
          <td>26.367348</td>
          <td>0.143754</td>
          <td>26.377025</td>
          <td>0.130473</td>
          <td>26.155154</td>
          <td>0.174644</td>
          <td>25.879981</td>
          <td>0.254178</td>
          <td>25.120801</td>
          <td>0.291129</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.876650</td>
          <td>0.563716</td>
          <td>26.271485</td>
          <td>0.134571</td>
          <td>26.125130</td>
          <td>0.106761</td>
          <td>25.931284</td>
          <td>0.146971</td>
          <td>25.778185</td>
          <td>0.237859</td>
          <td>25.075037</td>
          <td>0.285537</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.833330</td>
          <td>1.758833</td>
          <td>26.927303</td>
          <td>0.230194</td>
          <td>26.619624</td>
          <td>0.160309</td>
          <td>26.270924</td>
          <td>0.192087</td>
          <td>25.668179</td>
          <td>0.212748</td>
          <td>26.258837</td>
          <td>0.681977</td>
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
          <td>28.176072</td>
          <td>1.171438</td>
          <td>26.573175</td>
          <td>0.147505</td>
          <td>25.983914</td>
          <td>0.077745</td>
          <td>25.149649</td>
          <td>0.060636</td>
          <td>24.669856</td>
          <td>0.075880</td>
          <td>24.040212</td>
          <td>0.097919</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.095488</td>
          <td>0.229459</td>
          <td>26.696511</td>
          <td>0.144965</td>
          <td>25.932541</td>
          <td>0.120858</td>
          <td>26.141059</td>
          <td>0.267838</td>
          <td>26.703954</td>
          <td>0.802950</td>
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
          <td>29.472163</td>
          <td>1.192996</td>
          <td>25.690956</td>
          <td>0.106482</td>
          <td>25.052530</td>
          <td>0.115204</td>
          <td>24.478362</td>
          <td>0.155667</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.485257</td>
          <td>0.775201</td>
          <td>27.680527</td>
          <td>0.400817</td>
          <td>26.108429</td>
          <td>0.176577</td>
          <td>25.602065</td>
          <td>0.211819</td>
          <td>25.168643</td>
          <td>0.317261</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.479994</td>
          <td>0.371530</td>
          <td>26.028247</td>
          <td>0.091966</td>
          <td>25.969911</td>
          <td>0.076889</td>
          <td>25.716043</td>
          <td>0.100100</td>
          <td>25.428903</td>
          <td>0.147408</td>
          <td>24.693257</td>
          <td>0.172465</td>
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
          <td>26.412949</td>
          <td>0.370206</td>
          <td>26.450225</td>
          <td>0.141967</td>
          <td>25.374738</td>
          <td>0.049056</td>
          <td>25.152876</td>
          <td>0.066081</td>
          <td>24.976225</td>
          <td>0.107453</td>
          <td>24.635997</td>
          <td>0.177501</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.823954</td>
          <td>0.959898</td>
          <td>26.621205</td>
          <td>0.155850</td>
          <td>25.965719</td>
          <td>0.077774</td>
          <td>25.295250</td>
          <td>0.070193</td>
          <td>24.808006</td>
          <td>0.087130</td>
          <td>24.124420</td>
          <td>0.107208</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.291341</td>
          <td>0.691943</td>
          <td>26.524709</td>
          <td>0.147513</td>
          <td>26.485887</td>
          <td>0.126693</td>
          <td>26.079539</td>
          <td>0.144200</td>
          <td>26.125485</td>
          <td>0.276461</td>
          <td>26.154285</td>
          <td>0.573166</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.191267</td>
          <td>0.318305</td>
          <td>26.118716</td>
          <td>0.109941</td>
          <td>26.107063</td>
          <td>0.097193</td>
          <td>26.052478</td>
          <td>0.150653</td>
          <td>25.569596</td>
          <td>0.185469</td>
          <td>25.722055</td>
          <td>0.441332</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.840238</td>
          <td>0.499497</td>
          <td>26.861987</td>
          <td>0.194836</td>
          <td>26.374722</td>
          <td>0.113893</td>
          <td>26.223925</td>
          <td>0.161546</td>
          <td>26.087441</td>
          <td>0.265558</td>
          <td>26.371466</td>
          <td>0.662136</td>
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
