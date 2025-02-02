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

    <pzflow.flow.Flow at 0x7f5b24da4a00>



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
          <td>27.089622</td>
          <td>0.585114</td>
          <td>26.360166</td>
          <td>0.122720</td>
          <td>26.037085</td>
          <td>0.081470</td>
          <td>25.073997</td>
          <td>0.056691</td>
          <td>24.782242</td>
          <td>0.083780</td>
          <td>23.996472</td>
          <td>0.094220</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.787374</td>
          <td>1.610728</td>
          <td>27.511733</td>
          <td>0.321652</td>
          <td>26.746767</td>
          <td>0.151220</td>
          <td>26.043056</td>
          <td>0.132877</td>
          <td>25.786951</td>
          <td>0.199578</td>
          <td>25.235104</td>
          <td>0.270518</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.578087</td>
          <td>0.708323</td>
          <td>27.983947</td>
          <td>0.415660</td>
          <td>25.828828</td>
          <td>0.110312</td>
          <td>24.998560</td>
          <td>0.101317</td>
          <td>24.378633</td>
          <td>0.131480</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.622423</td>
          <td>1.330948</td>
          <td>27.281158</td>
          <td>0.237299</td>
          <td>26.172838</td>
          <td>0.148602</td>
          <td>25.635437</td>
          <td>0.175606</td>
          <td>25.230215</td>
          <td>0.269443</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.280240</td>
          <td>0.317138</td>
          <td>26.003365</td>
          <td>0.089868</td>
          <td>26.022326</td>
          <td>0.080416</td>
          <td>25.752391</td>
          <td>0.103184</td>
          <td>25.673070</td>
          <td>0.181299</td>
          <td>25.000663</td>
          <td>0.223039</td>
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
          <td>26.560438</td>
          <td>0.395080</td>
          <td>26.099631</td>
          <td>0.097785</td>
          <td>25.470582</td>
          <td>0.049317</td>
          <td>25.106608</td>
          <td>0.058356</td>
          <td>25.056472</td>
          <td>0.106582</td>
          <td>24.561597</td>
          <td>0.153914</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.086513</td>
          <td>0.583821</td>
          <td>26.946179</td>
          <td>0.202442</td>
          <td>26.063256</td>
          <td>0.083372</td>
          <td>25.082841</td>
          <td>0.057138</td>
          <td>24.728907</td>
          <td>0.079931</td>
          <td>24.182297</td>
          <td>0.110861</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.042677</td>
          <td>0.565821</td>
          <td>26.785947</td>
          <td>0.176854</td>
          <td>26.271652</td>
          <td>0.100131</td>
          <td>26.287032</td>
          <td>0.163863</td>
          <td>26.217076</td>
          <td>0.284653</td>
          <td>26.834116</td>
          <td>0.872292</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.544781</td>
          <td>0.390336</td>
          <td>26.226099</td>
          <td>0.109213</td>
          <td>26.034140</td>
          <td>0.081258</td>
          <td>26.054467</td>
          <td>0.134194</td>
          <td>25.905832</td>
          <td>0.220441</td>
          <td>26.001514</td>
          <td>0.491740</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.170078</td>
          <td>1.167419</td>
          <td>26.900396</td>
          <td>0.194804</td>
          <td>26.406551</td>
          <td>0.112661</td>
          <td>26.478139</td>
          <td>0.192694</td>
          <td>26.705963</td>
          <td>0.418364</td>
          <td>26.281867</td>
          <td>0.602368</td>
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
          <td>31.782246</td>
          <td>4.477158</td>
          <td>26.489492</td>
          <td>0.157830</td>
          <td>26.060145</td>
          <td>0.097749</td>
          <td>25.212322</td>
          <td>0.075966</td>
          <td>24.843435</td>
          <td>0.103897</td>
          <td>23.978095</td>
          <td>0.109512</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.249187</td>
          <td>0.296971</td>
          <td>26.687253</td>
          <td>0.168214</td>
          <td>26.466772</td>
          <td>0.224120</td>
          <td>26.176992</td>
          <td>0.319498</td>
          <td>25.843404</td>
          <td>0.503493</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.754206</td>
          <td>0.896526</td>
          <td>28.809594</td>
          <td>0.860646</td>
          <td>25.862180</td>
          <td>0.137208</td>
          <td>25.083165</td>
          <td>0.130847</td>
          <td>24.390922</td>
          <td>0.160016</td>
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
          <td>27.344849</td>
          <td>0.308880</td>
          <td>26.083267</td>
          <td>0.173462</td>
          <td>25.865815</td>
          <td>0.264281</td>
          <td>25.284104</td>
          <td>0.348812</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.664037</td>
          <td>0.473027</td>
          <td>26.272010</td>
          <td>0.130959</td>
          <td>25.875119</td>
          <td>0.083104</td>
          <td>25.785820</td>
          <td>0.125595</td>
          <td>25.592140</td>
          <td>0.197778</td>
          <td>26.235387</td>
          <td>0.665889</td>
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
          <td>26.619088</td>
          <td>0.463584</td>
          <td>26.296250</td>
          <td>0.136207</td>
          <td>25.402510</td>
          <td>0.055857</td>
          <td>25.033860</td>
          <td>0.066311</td>
          <td>24.848000</td>
          <td>0.106515</td>
          <td>24.558344</td>
          <td>0.184251</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.882615</td>
          <td>1.070716</td>
          <td>26.586563</td>
          <td>0.172065</td>
          <td>26.080977</td>
          <td>0.099965</td>
          <td>25.147598</td>
          <td>0.072055</td>
          <td>24.846834</td>
          <td>0.104640</td>
          <td>24.092552</td>
          <td>0.121500</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.916209</td>
          <td>1.096633</td>
          <td>26.642279</td>
          <td>0.181746</td>
          <td>26.460575</td>
          <td>0.140234</td>
          <td>26.619066</td>
          <td>0.257258</td>
          <td>26.107552</td>
          <td>0.305717</td>
          <td>25.578990</td>
          <td>0.417432</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.911594</td>
          <td>0.268297</td>
          <td>26.318506</td>
          <td>0.140137</td>
          <td>26.353265</td>
          <td>0.130187</td>
          <td>25.721540</td>
          <td>0.122618</td>
          <td>25.657814</td>
          <td>0.215239</td>
          <td>25.668960</td>
          <td>0.454335</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.271116</td>
          <td>0.731691</td>
          <td>26.976104</td>
          <td>0.239671</td>
          <td>26.780245</td>
          <td>0.183764</td>
          <td>26.433190</td>
          <td>0.220056</td>
          <td>25.648124</td>
          <td>0.209212</td>
          <td>25.634517</td>
          <td>0.434396</td>
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
          <td>29.436381</td>
          <td>2.142854</td>
          <td>26.692831</td>
          <td>0.163408</td>
          <td>26.051058</td>
          <td>0.082491</td>
          <td>25.215307</td>
          <td>0.064272</td>
          <td>24.682224</td>
          <td>0.076713</td>
          <td>24.044062</td>
          <td>0.098251</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.102103</td>
          <td>0.590639</td>
          <td>27.413438</td>
          <td>0.297535</td>
          <td>26.764548</td>
          <td>0.153685</td>
          <td>26.379126</td>
          <td>0.177389</td>
          <td>25.805833</td>
          <td>0.202950</td>
          <td>25.242638</td>
          <td>0.272431</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.046916</td>
          <td>0.595117</td>
          <td>28.800037</td>
          <td>0.864431</td>
          <td>28.329579</td>
          <td>0.575202</td>
          <td>26.074268</td>
          <td>0.148446</td>
          <td>25.027425</td>
          <td>0.112712</td>
          <td>24.357157</td>
          <td>0.140276</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.309543</td>
          <td>1.271692</td>
          <td>27.583983</td>
          <td>0.371939</td>
          <td>26.173194</td>
          <td>0.186528</td>
          <td>25.713736</td>
          <td>0.232436</td>
          <td>24.936668</td>
          <td>0.263056</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.630785</td>
          <td>0.417347</td>
          <td>26.075427</td>
          <td>0.095851</td>
          <td>26.020704</td>
          <td>0.080415</td>
          <td>25.745236</td>
          <td>0.102692</td>
          <td>25.318287</td>
          <td>0.134006</td>
          <td>25.363726</td>
          <td>0.300601</td>
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
          <td>26.815503</td>
          <td>0.502372</td>
          <td>26.293619</td>
          <td>0.124008</td>
          <td>25.410858</td>
          <td>0.050654</td>
          <td>24.989511</td>
          <td>0.057169</td>
          <td>24.933044</td>
          <td>0.103473</td>
          <td>25.048100</td>
          <td>0.250455</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.036724</td>
          <td>1.089709</td>
          <td>26.838134</td>
          <td>0.187404</td>
          <td>25.966585</td>
          <td>0.077834</td>
          <td>25.196772</td>
          <td>0.064330</td>
          <td>25.006459</td>
          <td>0.103707</td>
          <td>24.296903</td>
          <td>0.124578</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.429589</td>
          <td>0.367983</td>
          <td>26.551843</td>
          <td>0.150987</td>
          <td>26.360719</td>
          <td>0.113634</td>
          <td>26.340539</td>
          <td>0.180207</td>
          <td>25.816634</td>
          <td>0.214352</td>
          <td>25.934089</td>
          <td>0.488210</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.121057</td>
          <td>0.300930</td>
          <td>26.464685</td>
          <td>0.148304</td>
          <td>26.069075</td>
          <td>0.094007</td>
          <td>25.953522</td>
          <td>0.138360</td>
          <td>25.653900</td>
          <td>0.199124</td>
          <td>25.413948</td>
          <td>0.347874</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.244445</td>
          <td>0.315941</td>
          <td>27.104725</td>
          <td>0.238534</td>
          <td>26.633725</td>
          <td>0.142544</td>
          <td>26.358465</td>
          <td>0.181130</td>
          <td>25.796442</td>
          <td>0.208772</td>
          <td>25.529430</td>
          <td>0.355180</td>
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
