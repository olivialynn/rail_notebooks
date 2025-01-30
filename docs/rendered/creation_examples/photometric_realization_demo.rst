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

    <pzflow.flow.Flow at 0x7fe819ec8250>



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
          <td>28.015436</td>
          <td>1.068090</td>
          <td>26.846527</td>
          <td>0.186156</td>
          <td>25.942110</td>
          <td>0.074916</td>
          <td>25.312520</td>
          <td>0.070042</td>
          <td>24.838804</td>
          <td>0.088059</td>
          <td>24.112572</td>
          <td>0.104311</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.947939</td>
          <td>0.451121</td>
          <td>26.881370</td>
          <td>0.169654</td>
          <td>26.342393</td>
          <td>0.171775</td>
          <td>25.741041</td>
          <td>0.192015</td>
          <td>25.011816</td>
          <td>0.225116</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.848301</td>
          <td>2.505942</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.332220</td>
          <td>0.247503</td>
          <td>25.858723</td>
          <td>0.113226</td>
          <td>24.971085</td>
          <td>0.098908</td>
          <td>24.312397</td>
          <td>0.124148</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.653517</td>
          <td>0.856048</td>
          <td>27.852973</td>
          <td>0.419778</td>
          <td>27.518551</td>
          <td>0.288127</td>
          <td>26.004431</td>
          <td>0.128509</td>
          <td>25.547636</td>
          <td>0.162960</td>
          <td>25.897298</td>
          <td>0.454946</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.488138</td>
          <td>0.164822</td>
          <td>26.018176</td>
          <td>0.091044</td>
          <td>25.855572</td>
          <td>0.069395</td>
          <td>25.759962</td>
          <td>0.103870</td>
          <td>25.687114</td>
          <td>0.183467</td>
          <td>24.924296</td>
          <td>0.209275</td>
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
          <td>28.023964</td>
          <td>1.073436</td>
          <td>26.487179</td>
          <td>0.136969</td>
          <td>25.442895</td>
          <td>0.048119</td>
          <td>25.135412</td>
          <td>0.059867</td>
          <td>24.845734</td>
          <td>0.088598</td>
          <td>25.070078</td>
          <td>0.236252</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.049675</td>
          <td>1.089645</td>
          <td>26.567987</td>
          <td>0.146833</td>
          <td>26.073227</td>
          <td>0.084108</td>
          <td>25.239225</td>
          <td>0.065640</td>
          <td>24.790773</td>
          <td>0.084412</td>
          <td>24.279666</td>
          <td>0.120669</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.882565</td>
          <td>0.503673</td>
          <td>26.615162</td>
          <td>0.152896</td>
          <td>26.333961</td>
          <td>0.105743</td>
          <td>26.246235</td>
          <td>0.158250</td>
          <td>25.815224</td>
          <td>0.204370</td>
          <td>25.361430</td>
          <td>0.299642</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.893950</td>
          <td>0.231704</td>
          <td>26.243279</td>
          <td>0.110861</td>
          <td>26.033191</td>
          <td>0.081190</td>
          <td>25.884956</td>
          <td>0.115843</td>
          <td>26.058430</td>
          <td>0.250100</td>
          <td>25.537653</td>
          <td>0.344796</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.465068</td>
          <td>0.366912</td>
          <td>27.001813</td>
          <td>0.212090</td>
          <td>26.479742</td>
          <td>0.120072</td>
          <td>26.325543</td>
          <td>0.169330</td>
          <td>26.046785</td>
          <td>0.247717</td>
          <td>25.499319</td>
          <td>0.334506</td>
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
          <td>26.412846</td>
          <td>0.147804</td>
          <td>25.908115</td>
          <td>0.085527</td>
          <td>25.273502</td>
          <td>0.080181</td>
          <td>24.666592</td>
          <td>0.088971</td>
          <td>24.001070</td>
          <td>0.111729</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.452727</td>
          <td>0.819943</td>
          <td>27.418695</td>
          <td>0.339952</td>
          <td>26.496486</td>
          <td>0.142865</td>
          <td>26.007144</td>
          <td>0.151988</td>
          <td>25.555660</td>
          <td>0.191775</td>
          <td>25.121376</td>
          <td>0.287826</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.670384</td>
          <td>1.500821</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.905542</td>
          <td>0.142432</td>
          <td>24.785698</td>
          <td>0.100998</td>
          <td>24.535939</td>
          <td>0.181023</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.732675</td>
          <td>0.910533</td>
          <td>27.221801</td>
          <td>0.279715</td>
          <td>26.267493</td>
          <td>0.202652</td>
          <td>25.337144</td>
          <td>0.169993</td>
          <td>25.032635</td>
          <td>0.285367</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.938268</td>
          <td>0.577827</td>
          <td>26.070297</td>
          <td>0.109928</td>
          <td>26.159426</td>
          <td>0.106656</td>
          <td>25.833774</td>
          <td>0.130921</td>
          <td>25.997377</td>
          <td>0.276523</td>
          <td>25.227702</td>
          <td>0.313542</td>
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
          <td>28.886646</td>
          <td>1.809519</td>
          <td>26.542801</td>
          <td>0.168241</td>
          <td>25.391427</td>
          <td>0.055311</td>
          <td>25.077591</td>
          <td>0.068928</td>
          <td>24.920712</td>
          <td>0.113490</td>
          <td>24.766050</td>
          <td>0.219338</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.547783</td>
          <td>0.434528</td>
          <td>26.690185</td>
          <td>0.187841</td>
          <td>26.229728</td>
          <td>0.113837</td>
          <td>25.206923</td>
          <td>0.075934</td>
          <td>24.815130</td>
          <td>0.101778</td>
          <td>24.385755</td>
          <td>0.156450</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.194477</td>
          <td>0.695951</td>
          <td>26.778038</td>
          <td>0.203753</td>
          <td>26.339081</td>
          <td>0.126256</td>
          <td>26.176237</td>
          <td>0.177797</td>
          <td>26.028324</td>
          <td>0.286820</td>
          <td>25.039231</td>
          <td>0.272507</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.969203</td>
          <td>0.281137</td>
          <td>26.077554</td>
          <td>0.113756</td>
          <td>26.077299</td>
          <td>0.102389</td>
          <td>26.034887</td>
          <td>0.160613</td>
          <td>25.703535</td>
          <td>0.223591</td>
          <td>25.622102</td>
          <td>0.438548</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.455001</td>
          <td>0.406424</td>
          <td>26.811191</td>
          <td>0.208991</td>
          <td>26.420584</td>
          <td>0.135113</td>
          <td>26.330280</td>
          <td>0.201919</td>
          <td>25.982034</td>
          <td>0.275559</td>
          <td>25.106070</td>
          <td>0.286940</td>
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
          <td>28.252579</td>
          <td>1.222520</td>
          <td>27.062363</td>
          <td>0.223087</td>
          <td>26.004330</td>
          <td>0.079159</td>
          <td>25.164437</td>
          <td>0.061437</td>
          <td>24.672294</td>
          <td>0.076043</td>
          <td>24.078287</td>
          <td>0.101241</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.548276</td>
          <td>0.331375</td>
          <td>26.532419</td>
          <td>0.125809</td>
          <td>26.096874</td>
          <td>0.139334</td>
          <td>26.394228</td>
          <td>0.328390</td>
          <td>25.581995</td>
          <td>0.357346</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.977242</td>
          <td>0.566289</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.019032</td>
          <td>0.912787</td>
          <td>25.992062</td>
          <td>0.138306</td>
          <td>24.931331</td>
          <td>0.103640</td>
          <td>24.497002</td>
          <td>0.158169</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.829056</td>
          <td>1.069817</td>
          <td>28.802132</td>
          <td>0.948337</td>
          <td>27.657469</td>
          <td>0.393756</td>
          <td>26.099337</td>
          <td>0.175220</td>
          <td>25.843315</td>
          <td>0.258604</td>
          <td>25.430809</td>
          <td>0.389847</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.965167</td>
          <td>0.535470</td>
          <td>26.362166</td>
          <td>0.123083</td>
          <td>26.026655</td>
          <td>0.080839</td>
          <td>25.792094</td>
          <td>0.106988</td>
          <td>25.586517</td>
          <td>0.168684</td>
          <td>25.598026</td>
          <td>0.362021</td>
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
          <td>27.828428</td>
          <td>0.992457</td>
          <td>26.469757</td>
          <td>0.144372</td>
          <td>25.417905</td>
          <td>0.050972</td>
          <td>25.103671</td>
          <td>0.063261</td>
          <td>24.893858</td>
          <td>0.099984</td>
          <td>25.298907</td>
          <td>0.307013</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.599285</td>
          <td>0.152953</td>
          <td>25.986143</td>
          <td>0.079190</td>
          <td>25.100483</td>
          <td>0.059064</td>
          <td>24.819439</td>
          <td>0.088011</td>
          <td>24.247904</td>
          <td>0.119389</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.016109</td>
          <td>0.570958</td>
          <td>27.188049</td>
          <td>0.257561</td>
          <td>26.453495</td>
          <td>0.123183</td>
          <td>26.566803</td>
          <td>0.217952</td>
          <td>26.232235</td>
          <td>0.301366</td>
          <td>25.266393</td>
          <td>0.290766</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.001569</td>
          <td>0.273257</td>
          <td>26.355746</td>
          <td>0.135034</td>
          <td>26.001776</td>
          <td>0.088608</td>
          <td>26.051358</td>
          <td>0.150509</td>
          <td>25.520898</td>
          <td>0.177979</td>
          <td>25.144294</td>
          <td>0.280411</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.734487</td>
          <td>0.174937</td>
          <td>26.552720</td>
          <td>0.132922</td>
          <td>26.670935</td>
          <td>0.235300</td>
          <td>26.208612</td>
          <td>0.292995</td>
          <td>26.738180</td>
          <td>0.844963</td>
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
