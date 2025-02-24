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

    <pzflow.flow.Flow at 0x7f99cde2eb90>



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
          <td>26.924396</td>
          <td>0.519370</td>
          <td>26.640744</td>
          <td>0.156281</td>
          <td>25.993145</td>
          <td>0.078371</td>
          <td>25.240751</td>
          <td>0.065728</td>
          <td>24.780467</td>
          <td>0.083649</td>
          <td>24.033190</td>
          <td>0.097305</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.410947</td>
          <td>0.730562</td>
          <td>27.160919</td>
          <td>0.242023</td>
          <td>26.554961</td>
          <td>0.128171</td>
          <td>26.271032</td>
          <td>0.161640</td>
          <td>25.867374</td>
          <td>0.213485</td>
          <td>27.170259</td>
          <td>1.070560</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.277655</td>
          <td>0.667368</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.288354</td>
          <td>0.521985</td>
          <td>25.919290</td>
          <td>0.119356</td>
          <td>24.988334</td>
          <td>0.100414</td>
          <td>24.199837</td>
          <td>0.112570</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.531110</td>
          <td>0.791034</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.269436</td>
          <td>0.235010</td>
          <td>26.383187</td>
          <td>0.177830</td>
          <td>25.390045</td>
          <td>0.142365</td>
          <td>25.077159</td>
          <td>0.237639</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.894860</td>
          <td>0.508248</td>
          <td>26.020896</td>
          <td>0.091262</td>
          <td>25.943112</td>
          <td>0.074982</td>
          <td>25.839155</td>
          <td>0.111310</td>
          <td>25.526248</td>
          <td>0.160010</td>
          <td>24.927798</td>
          <td>0.209889</td>
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
          <td>27.109178</td>
          <td>0.593297</td>
          <td>26.244998</td>
          <td>0.111027</td>
          <td>25.386473</td>
          <td>0.045768</td>
          <td>25.139722</td>
          <td>0.060096</td>
          <td>24.841421</td>
          <td>0.088262</td>
          <td>24.821127</td>
          <td>0.191908</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.446060</td>
          <td>0.747889</td>
          <td>26.621039</td>
          <td>0.153668</td>
          <td>26.052928</td>
          <td>0.082616</td>
          <td>25.190925</td>
          <td>0.062888</td>
          <td>24.841811</td>
          <td>0.088292</td>
          <td>24.401403</td>
          <td>0.134094</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.691560</td>
          <td>0.876953</td>
          <td>26.791769</td>
          <td>0.177729</td>
          <td>26.375927</td>
          <td>0.109691</td>
          <td>26.001347</td>
          <td>0.128166</td>
          <td>25.957286</td>
          <td>0.230068</td>
          <td>24.951040</td>
          <td>0.214005</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.284219</td>
          <td>0.318146</td>
          <td>26.254895</td>
          <td>0.111988</td>
          <td>25.841085</td>
          <td>0.068510</td>
          <td>25.977040</td>
          <td>0.125494</td>
          <td>26.033806</td>
          <td>0.245084</td>
          <td>25.434046</td>
          <td>0.317590</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.091898</td>
          <td>1.116571</td>
          <td>26.738171</td>
          <td>0.169823</td>
          <td>26.352346</td>
          <td>0.107455</td>
          <td>26.247717</td>
          <td>0.158451</td>
          <td>25.865453</td>
          <td>0.213142</td>
          <td>26.595051</td>
          <td>0.746849</td>
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

.. parsed-literal::

    




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
          <td>27.242579</td>
          <td>0.713614</td>
          <td>26.507746</td>
          <td>0.160310</td>
          <td>26.013343</td>
          <td>0.093817</td>
          <td>25.294964</td>
          <td>0.081713</td>
          <td>24.610744</td>
          <td>0.084704</td>
          <td>23.918266</td>
          <td>0.103936</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.866095</td>
          <td>0.479292</td>
          <td>26.850088</td>
          <td>0.193090</td>
          <td>26.531096</td>
          <td>0.236394</td>
          <td>26.116792</td>
          <td>0.304478</td>
          <td>25.002226</td>
          <td>0.261252</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.618646</td>
          <td>0.822445</td>
          <td>28.533108</td>
          <td>0.718105</td>
          <td>26.078635</td>
          <td>0.165203</td>
          <td>24.961385</td>
          <td>0.117731</td>
          <td>24.218538</td>
          <td>0.138005</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.283059</td>
          <td>0.678480</td>
          <td>26.759390</td>
          <td>0.190742</td>
          <td>25.992700</td>
          <td>0.160583</td>
          <td>25.466952</td>
          <td>0.189752</td>
          <td>24.754492</td>
          <td>0.227186</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.107558</td>
          <td>0.307469</td>
          <td>26.098621</td>
          <td>0.112673</td>
          <td>26.120622</td>
          <td>0.103099</td>
          <td>25.558892</td>
          <td>0.103069</td>
          <td>25.705111</td>
          <td>0.217393</td>
          <td>28.509070</td>
          <td>2.240035</td>
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
          <td>27.157388</td>
          <td>0.681896</td>
          <td>26.266038</td>
          <td>0.132702</td>
          <td>25.415636</td>
          <td>0.056512</td>
          <td>25.196565</td>
          <td>0.076573</td>
          <td>24.858998</td>
          <td>0.107543</td>
          <td>25.071810</td>
          <td>0.282001</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.107331</td>
          <td>1.216888</td>
          <td>26.678185</td>
          <td>0.185948</td>
          <td>26.123843</td>
          <td>0.103787</td>
          <td>25.053418</td>
          <td>0.066293</td>
          <td>24.969232</td>
          <td>0.116429</td>
          <td>24.358315</td>
          <td>0.152816</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.882621</td>
          <td>1.075408</td>
          <td>26.433390</td>
          <td>0.152135</td>
          <td>26.249076</td>
          <td>0.116764</td>
          <td>26.321860</td>
          <td>0.201042</td>
          <td>25.868857</td>
          <td>0.251868</td>
          <td>25.723860</td>
          <td>0.465777</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.037184</td>
          <td>0.296981</td>
          <td>26.026349</td>
          <td>0.108795</td>
          <td>26.192370</td>
          <td>0.113211</td>
          <td>25.935528</td>
          <td>0.147508</td>
          <td>25.954091</td>
          <td>0.274745</td>
          <td>24.964235</td>
          <td>0.260927</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.241219</td>
          <td>0.344187</td>
          <td>26.804530</td>
          <td>0.207830</td>
          <td>26.727850</td>
          <td>0.175785</td>
          <td>26.285529</td>
          <td>0.194465</td>
          <td>26.446886</td>
          <td>0.398280</td>
          <td>25.595543</td>
          <td>0.421707</td>
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

.. parsed-literal::

    




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
          <td>26.636325</td>
          <td>0.155709</td>
          <td>26.034798</td>
          <td>0.081316</td>
          <td>25.240246</td>
          <td>0.065708</td>
          <td>24.609932</td>
          <td>0.071964</td>
          <td>23.994198</td>
          <td>0.094045</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.098101</td>
          <td>1.858930</td>
          <td>27.384257</td>
          <td>0.290620</td>
          <td>26.849053</td>
          <td>0.165199</td>
          <td>26.316840</td>
          <td>0.168242</td>
          <td>26.208428</td>
          <td>0.282913</td>
          <td>25.334737</td>
          <td>0.293534</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.747991</td>
          <td>0.946778</td>
          <td>29.141102</td>
          <td>1.064683</td>
          <td>29.124425</td>
          <td>0.973951</td>
          <td>26.135387</td>
          <td>0.156432</td>
          <td>25.032120</td>
          <td>0.113174</td>
          <td>24.421726</td>
          <td>0.148288</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.263815</td>
          <td>1.361732</td>
          <td>29.139599</td>
          <td>1.157346</td>
          <td>27.625767</td>
          <td>0.384217</td>
          <td>26.307024</td>
          <td>0.208741</td>
          <td>25.847831</td>
          <td>0.259562</td>
          <td>25.147770</td>
          <td>0.312014</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.998730</td>
          <td>0.252808</td>
          <td>26.006982</td>
          <td>0.090265</td>
          <td>25.847922</td>
          <td>0.069025</td>
          <td>25.674356</td>
          <td>0.096508</td>
          <td>25.410143</td>
          <td>0.145050</td>
          <td>24.904298</td>
          <td>0.206091</td>
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
          <td>27.633232</td>
          <td>0.879827</td>
          <td>26.229491</td>
          <td>0.117294</td>
          <td>25.422299</td>
          <td>0.051171</td>
          <td>25.115309</td>
          <td>0.063917</td>
          <td>24.717566</td>
          <td>0.085641</td>
          <td>24.643193</td>
          <td>0.178588</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.838015</td>
          <td>0.492105</td>
          <td>26.857064</td>
          <td>0.190420</td>
          <td>25.947893</td>
          <td>0.076559</td>
          <td>25.200726</td>
          <td>0.064556</td>
          <td>24.790897</td>
          <td>0.085828</td>
          <td>24.127050</td>
          <td>0.107454</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.257938</td>
          <td>0.676341</td>
          <td>26.782965</td>
          <td>0.183823</td>
          <td>26.330246</td>
          <td>0.110655</td>
          <td>26.224205</td>
          <td>0.163233</td>
          <td>26.028794</td>
          <td>0.255484</td>
          <td>25.634347</td>
          <td>0.388997</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.519371</td>
          <td>0.411329</td>
          <td>26.088859</td>
          <td>0.107116</td>
          <td>26.157386</td>
          <td>0.101574</td>
          <td>25.814122</td>
          <td>0.122637</td>
          <td>25.759452</td>
          <td>0.217513</td>
          <td>25.253189</td>
          <td>0.306150</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.213512</td>
          <td>0.652294</td>
          <td>26.376487</td>
          <td>0.128704</td>
          <td>26.539298</td>
          <td>0.131388</td>
          <td>26.280628</td>
          <td>0.169547</td>
          <td>25.872483</td>
          <td>0.222445</td>
          <td>25.563535</td>
          <td>0.364797</td>
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
