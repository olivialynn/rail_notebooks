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

    <pzflow.flow.Flow at 0x7ff1f0626860>



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
          <td>27.125623</td>
          <td>0.235073</td>
          <td>26.039095</td>
          <td>0.081614</td>
          <td>25.093222</td>
          <td>0.057667</td>
          <td>24.671183</td>
          <td>0.075959</td>
          <td>23.998786</td>
          <td>0.094412</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.571677</td>
          <td>0.398515</td>
          <td>27.417115</td>
          <td>0.298193</td>
          <td>26.485975</td>
          <td>0.120724</td>
          <td>26.575092</td>
          <td>0.209038</td>
          <td>25.996768</td>
          <td>0.237710</td>
          <td>25.691194</td>
          <td>0.388738</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.426480</td>
          <td>1.342902</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.427521</td>
          <td>0.577210</td>
          <td>26.178362</td>
          <td>0.149308</td>
          <td>24.972583</td>
          <td>0.099038</td>
          <td>24.344281</td>
          <td>0.127628</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.792929</td>
          <td>0.816549</td>
          <td>27.417077</td>
          <td>0.265329</td>
          <td>26.137881</td>
          <td>0.144202</td>
          <td>25.797988</td>
          <td>0.201436</td>
          <td>25.134840</td>
          <td>0.249209</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.157303</td>
          <td>0.287353</td>
          <td>25.997080</td>
          <td>0.089373</td>
          <td>25.894359</td>
          <td>0.071818</td>
          <td>25.653940</td>
          <td>0.094653</td>
          <td>25.237954</td>
          <td>0.124828</td>
          <td>25.653135</td>
          <td>0.377430</td>
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
          <td>27.345455</td>
          <td>0.699003</td>
          <td>26.446820</td>
          <td>0.132280</td>
          <td>25.383729</td>
          <td>0.045657</td>
          <td>25.076481</td>
          <td>0.056816</td>
          <td>24.919449</td>
          <td>0.094528</td>
          <td>24.577472</td>
          <td>0.156021</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.613219</td>
          <td>2.296672</td>
          <td>26.620865</td>
          <td>0.153645</td>
          <td>26.038826</td>
          <td>0.081595</td>
          <td>25.214954</td>
          <td>0.064243</td>
          <td>24.699984</td>
          <td>0.077916</td>
          <td>24.176138</td>
          <td>0.110267</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.341876</td>
          <td>0.333052</td>
          <td>26.639320</td>
          <td>0.156091</td>
          <td>26.374620</td>
          <td>0.109566</td>
          <td>26.307162</td>
          <td>0.166700</td>
          <td>26.137637</td>
          <td>0.266857</td>
          <td>26.202122</td>
          <td>0.569135</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.383979</td>
          <td>0.344312</td>
          <td>26.393992</td>
          <td>0.126371</td>
          <td>26.035622</td>
          <td>0.081365</td>
          <td>25.828016</td>
          <td>0.110234</td>
          <td>25.630504</td>
          <td>0.174872</td>
          <td>25.269308</td>
          <td>0.278148</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.677923</td>
          <td>0.432225</td>
          <td>26.866870</td>
          <td>0.189380</td>
          <td>26.408546</td>
          <td>0.112857</td>
          <td>26.221223</td>
          <td>0.154898</td>
          <td>26.109935</td>
          <td>0.260887</td>
          <td>26.979194</td>
          <td>0.954775</td>
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
          <td>27.603845</td>
          <td>0.902505</td>
          <td>26.405499</td>
          <td>0.146875</td>
          <td>26.167023</td>
          <td>0.107332</td>
          <td>25.192875</td>
          <td>0.074672</td>
          <td>24.711177</td>
          <td>0.092526</td>
          <td>24.033461</td>
          <td>0.114926</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.984381</td>
          <td>0.522971</td>
          <td>26.643957</td>
          <td>0.162118</td>
          <td>26.599018</td>
          <td>0.250005</td>
          <td>25.998235</td>
          <td>0.276686</td>
          <td>25.403381</td>
          <td>0.360269</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.818506</td>
          <td>2.617263</td>
          <td>30.095141</td>
          <td>1.833519</td>
          <td>28.087152</td>
          <td>0.524993</td>
          <td>26.048129</td>
          <td>0.160958</td>
          <td>25.193568</td>
          <td>0.143922</td>
          <td>24.396958</td>
          <td>0.160844</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.298238</td>
          <td>0.685557</td>
          <td>27.195963</td>
          <td>0.273906</td>
          <td>26.251038</td>
          <td>0.199872</td>
          <td>25.180717</td>
          <td>0.148720</td>
          <td>25.752607</td>
          <td>0.498801</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.175491</td>
          <td>0.324580</td>
          <td>26.229383</td>
          <td>0.126218</td>
          <td>25.988239</td>
          <td>0.091801</td>
          <td>25.566418</td>
          <td>0.103750</td>
          <td>25.732231</td>
          <td>0.222357</td>
          <td>25.242882</td>
          <td>0.317366</td>
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
          <td>27.062413</td>
          <td>0.638673</td>
          <td>26.863621</td>
          <td>0.220395</td>
          <td>25.349870</td>
          <td>0.053309</td>
          <td>25.222924</td>
          <td>0.078376</td>
          <td>24.746516</td>
          <td>0.097463</td>
          <td>24.550824</td>
          <td>0.183083</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.679470</td>
          <td>0.479681</td>
          <td>27.482471</td>
          <td>0.358591</td>
          <td>26.042956</td>
          <td>0.096689</td>
          <td>25.229490</td>
          <td>0.077462</td>
          <td>25.139922</td>
          <td>0.134997</td>
          <td>23.957439</td>
          <td>0.108015</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.865516</td>
          <td>0.552716</td>
          <td>26.863007</td>
          <td>0.218739</td>
          <td>26.285089</td>
          <td>0.120477</td>
          <td>26.325246</td>
          <td>0.201614</td>
          <td>26.263724</td>
          <td>0.346136</td>
          <td>25.049800</td>
          <td>0.274860</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.488696</td>
          <td>0.853837</td>
          <td>26.297574</td>
          <td>0.137633</td>
          <td>26.110758</td>
          <td>0.105429</td>
          <td>26.063617</td>
          <td>0.164600</td>
          <td>26.137773</td>
          <td>0.318545</td>
          <td>25.216416</td>
          <td>0.319870</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.682394</td>
          <td>2.484479</td>
          <td>26.777429</td>
          <td>0.203167</td>
          <td>26.406881</td>
          <td>0.133524</td>
          <td>26.496723</td>
          <td>0.231977</td>
          <td>26.223642</td>
          <td>0.334521</td>
          <td>25.522671</td>
          <td>0.398795</td>
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
          <td>26.598918</td>
          <td>0.150799</td>
          <td>26.076655</td>
          <td>0.084373</td>
          <td>25.260596</td>
          <td>0.066904</td>
          <td>24.687351</td>
          <td>0.077062</td>
          <td>24.131038</td>
          <td>0.106023</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.607375</td>
          <td>1.474835</td>
          <td>27.415874</td>
          <td>0.298119</td>
          <td>26.572497</td>
          <td>0.130253</td>
          <td>26.440166</td>
          <td>0.186796</td>
          <td>25.470230</td>
          <td>0.152660</td>
          <td>26.318935</td>
          <td>0.618787</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.260761</td>
          <td>1.273480</td>
          <td>28.364548</td>
          <td>0.647205</td>
          <td>29.248577</td>
          <td>1.049168</td>
          <td>25.878746</td>
          <td>0.125395</td>
          <td>24.966223</td>
          <td>0.106850</td>
          <td>24.338262</td>
          <td>0.138009</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.151175</td>
          <td>1.282462</td>
          <td>29.116711</td>
          <td>1.142399</td>
          <td>27.113958</td>
          <td>0.255319</td>
          <td>26.286271</td>
          <td>0.205145</td>
          <td>25.528818</td>
          <td>0.199213</td>
          <td>25.475997</td>
          <td>0.403668</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.413928</td>
          <td>0.352827</td>
          <td>26.176114</td>
          <td>0.104679</td>
          <td>25.972652</td>
          <td>0.077075</td>
          <td>25.788853</td>
          <td>0.106686</td>
          <td>25.557201</td>
          <td>0.164522</td>
          <td>25.117795</td>
          <td>0.246079</td>
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
          <td>27.677309</td>
          <td>0.904509</td>
          <td>26.396177</td>
          <td>0.135508</td>
          <td>25.440271</td>
          <td>0.051994</td>
          <td>25.057113</td>
          <td>0.060703</td>
          <td>24.874650</td>
          <td>0.098315</td>
          <td>25.211683</td>
          <td>0.286189</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.947074</td>
          <td>1.033806</td>
          <td>26.713823</td>
          <td>0.168665</td>
          <td>26.202837</td>
          <td>0.095831</td>
          <td>25.280156</td>
          <td>0.069262</td>
          <td>24.755579</td>
          <td>0.083198</td>
          <td>24.080918</td>
          <td>0.103207</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.188767</td>
          <td>0.644844</td>
          <td>26.720549</td>
          <td>0.174357</td>
          <td>26.373956</td>
          <td>0.114952</td>
          <td>26.338982</td>
          <td>0.179969</td>
          <td>26.400679</td>
          <td>0.344620</td>
          <td>26.291302</td>
          <td>0.631452</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.808808</td>
          <td>0.233341</td>
          <td>26.269425</td>
          <td>0.125324</td>
          <td>26.188075</td>
          <td>0.104338</td>
          <td>25.843907</td>
          <td>0.125847</td>
          <td>26.568278</td>
          <td>0.415997</td>
          <td>24.872702</td>
          <td>0.224354</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.467668</td>
          <td>0.774385</td>
          <td>26.737799</td>
          <td>0.175429</td>
          <td>26.743273</td>
          <td>0.156600</td>
          <td>26.203293</td>
          <td>0.158722</td>
          <td>25.640597</td>
          <td>0.183113</td>
          <td>25.769705</td>
          <td>0.427690</td>
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
