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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f7e98ebb700>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.023191</td>
          <td>0.557956</td>
          <td>26.604655</td>
          <td>0.151526</td>
          <td>26.078443</td>
          <td>0.084495</td>
          <td>25.364003</td>
          <td>0.073307</td>
          <td>24.892524</td>
          <td>0.092318</td>
          <td>24.963686</td>
          <td>0.216275</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.538521</td>
          <td>0.328577</td>
          <td>28.424388</td>
          <td>0.575919</td>
          <td>26.737688</td>
          <td>0.239290</td>
          <td>26.388120</td>
          <td>0.326521</td>
          <td>26.725922</td>
          <td>0.813895</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.845788</td>
          <td>0.490183</td>
          <td>26.034908</td>
          <td>0.092391</td>
          <td>24.799393</td>
          <td>0.027263</td>
          <td>23.878593</td>
          <td>0.019867</td>
          <td>23.135090</td>
          <td>0.019701</td>
          <td>22.813610</td>
          <td>0.033078</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.344796</td>
          <td>0.333823</td>
          <td>28.317584</td>
          <td>0.591290</td>
          <td>27.642854</td>
          <td>0.318373</td>
          <td>26.548810</td>
          <td>0.204487</td>
          <td>26.043449</td>
          <td>0.247038</td>
          <td>25.704765</td>
          <td>0.392838</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.701399</td>
          <td>0.197365</td>
          <td>25.693742</td>
          <td>0.068404</td>
          <td>25.420458</td>
          <td>0.047170</td>
          <td>24.816522</td>
          <td>0.045106</td>
          <td>24.502122</td>
          <td>0.065404</td>
          <td>23.716376</td>
          <td>0.073610</td>
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
          <td>2.147172</td>
          <td>26.845124</td>
          <td>0.489942</td>
          <td>26.319375</td>
          <td>0.118450</td>
          <td>26.016867</td>
          <td>0.080029</td>
          <td>26.002974</td>
          <td>0.128347</td>
          <td>25.843269</td>
          <td>0.209227</td>
          <td>26.148105</td>
          <td>0.547428</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.918813</td>
          <td>0.517253</td>
          <td>27.325444</td>
          <td>0.276898</td>
          <td>26.601314</td>
          <td>0.133416</td>
          <td>26.195755</td>
          <td>0.151553</td>
          <td>26.160902</td>
          <td>0.271965</td>
          <td>25.349437</td>
          <td>0.296764</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.462948</td>
          <td>0.756324</td>
          <td>27.964149</td>
          <td>0.456656</td>
          <td>26.853670</td>
          <td>0.165698</td>
          <td>26.805165</td>
          <td>0.252959</td>
          <td>25.929897</td>
          <td>0.224897</td>
          <td>25.468368</td>
          <td>0.326391</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.338293</td>
          <td>0.279800</td>
          <td>26.685658</td>
          <td>0.143484</td>
          <td>25.872422</td>
          <td>0.114586</td>
          <td>25.481059</td>
          <td>0.153941</td>
          <td>25.573381</td>
          <td>0.354627</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.585646</td>
          <td>1.458187</td>
          <td>26.485398</td>
          <td>0.136759</td>
          <td>26.234083</td>
          <td>0.096887</td>
          <td>25.713553</td>
          <td>0.099734</td>
          <td>25.285957</td>
          <td>0.130129</td>
          <td>25.108170</td>
          <td>0.243798</td>
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
          <td>0.890625</td>
          <td>26.267590</td>
          <td>0.349006</td>
          <td>27.032170</td>
          <td>0.248873</td>
          <td>26.172149</td>
          <td>0.107813</td>
          <td>25.237736</td>
          <td>0.077690</td>
          <td>25.024527</td>
          <td>0.121658</td>
          <td>24.732222</td>
          <td>0.208908</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.893768</td>
          <td>0.489242</td>
          <td>27.462612</td>
          <td>0.319325</td>
          <td>27.142131</td>
          <td>0.385919</td>
          <td>25.994369</td>
          <td>0.275818</td>
          <td>26.086373</td>
          <td>0.600065</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.998376</td>
          <td>1.155003</td>
          <td>25.854992</td>
          <td>0.092890</td>
          <td>24.748266</td>
          <td>0.031351</td>
          <td>23.827039</td>
          <td>0.022939</td>
          <td>23.190510</td>
          <td>0.024748</td>
          <td>22.823563</td>
          <td>0.040429</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.264731</td>
          <td>2.160527</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.563156</td>
          <td>0.367107</td>
          <td>26.136551</td>
          <td>0.181479</td>
          <td>26.021194</td>
          <td>0.299744</td>
          <td>25.872878</td>
          <td>0.544645</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.822501</td>
          <td>0.531589</td>
          <td>25.990207</td>
          <td>0.102507</td>
          <td>25.464214</td>
          <td>0.057778</td>
          <td>24.840032</td>
          <td>0.054646</td>
          <td>24.324068</td>
          <td>0.065777</td>
          <td>23.880047</td>
          <td>0.100551</td>
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
          <td>2.147172</td>
          <td>26.985300</td>
          <td>0.605089</td>
          <td>26.141932</td>
          <td>0.119182</td>
          <td>26.122071</td>
          <td>0.105378</td>
          <td>26.432781</td>
          <td>0.222355</td>
          <td>26.243725</td>
          <td>0.343237</td>
          <td>25.938615</td>
          <td>0.549364</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.702482</td>
          <td>0.425043</td>
          <td>26.774577</td>
          <td>0.181854</td>
          <td>27.010352</td>
          <td>0.349455</td>
          <td>25.751628</td>
          <td>0.226793</td>
          <td>25.873615</td>
          <td>0.516548</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.846085</td>
          <td>0.476767</td>
          <td>26.843307</td>
          <td>0.194319</td>
          <td>26.318631</td>
          <td>0.200497</td>
          <td>26.472392</td>
          <td>0.407146</td>
          <td>26.732506</td>
          <td>0.930442</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.078350</td>
          <td>1.213345</td>
          <td>28.242056</td>
          <td>0.642959</td>
          <td>26.636326</td>
          <td>0.166022</td>
          <td>25.868123</td>
          <td>0.139195</td>
          <td>25.705115</td>
          <td>0.223885</td>
          <td>25.992934</td>
          <td>0.576240</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.188625</td>
          <td>0.330178</td>
          <td>26.454991</td>
          <td>0.154600</td>
          <td>26.149557</td>
          <td>0.106768</td>
          <td>25.421284</td>
          <td>0.092278</td>
          <td>25.117712</td>
          <td>0.133198</td>
          <td>25.604980</td>
          <td>0.424751</td>
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
          <td>0.890625</td>
          <td>31.736214</td>
          <td>4.301027</td>
          <td>26.970604</td>
          <td>0.206650</td>
          <td>25.922572</td>
          <td>0.073642</td>
          <td>25.270653</td>
          <td>0.067502</td>
          <td>25.005608</td>
          <td>0.101958</td>
          <td>24.749381</td>
          <td>0.180643</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.970797</td>
          <td>0.537517</td>
          <td>27.871087</td>
          <td>0.425918</td>
          <td>27.418221</td>
          <td>0.265810</td>
          <td>27.259184</td>
          <td>0.364433</td>
          <td>27.328730</td>
          <td>0.659025</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.250897</td>
          <td>0.685817</td>
          <td>25.894923</td>
          <td>0.087810</td>
          <td>24.766307</td>
          <td>0.028748</td>
          <td>23.894820</td>
          <td>0.021901</td>
          <td>23.144358</td>
          <td>0.021507</td>
          <td>22.909317</td>
          <td>0.039214</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.780265</td>
          <td>0.472038</td>
          <td>29.417365</td>
          <td>1.266236</td>
          <td>26.607919</td>
          <td>0.267664</td>
          <td>27.666714</td>
          <td>0.968869</td>
          <td>26.232917</td>
          <td>0.699159</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.940802</td>
          <td>0.241060</td>
          <td>25.829437</td>
          <td>0.077207</td>
          <td>25.381495</td>
          <td>0.045632</td>
          <td>24.876199</td>
          <td>0.047632</td>
          <td>24.341582</td>
          <td>0.056806</td>
          <td>23.682258</td>
          <td>0.071528</td>
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
          <td>2.147172</td>
          <td>26.421083</td>
          <td>0.372558</td>
          <td>26.237272</td>
          <td>0.118090</td>
          <td>26.118823</td>
          <td>0.094735</td>
          <td>26.335772</td>
          <td>0.184941</td>
          <td>26.153568</td>
          <td>0.290684</td>
          <td>25.424942</td>
          <td>0.339413</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.537533</td>
          <td>0.801149</td>
          <td>27.563602</td>
          <td>0.339522</td>
          <td>26.852165</td>
          <td>0.168150</td>
          <td>26.351486</td>
          <td>0.176024</td>
          <td>26.616278</td>
          <td>0.396181</td>
          <td>25.471706</td>
          <td>0.332352</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.635287</td>
          <td>1.524481</td>
          <td>27.167944</td>
          <td>0.253353</td>
          <td>26.825882</td>
          <td>0.169680</td>
          <td>26.746286</td>
          <td>0.252837</td>
          <td>25.967836</td>
          <td>0.242997</td>
          <td>25.470091</td>
          <td>0.342126</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.837907</td>
          <td>1.723635</td>
          <td>27.725371</td>
          <td>0.415673</td>
          <td>26.589105</td>
          <td>0.147737</td>
          <td>26.023383</td>
          <td>0.146936</td>
          <td>25.486036</td>
          <td>0.172789</td>
          <td>24.978562</td>
          <td>0.244888</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.092644</td>
          <td>0.279651</td>
          <td>26.492341</td>
          <td>0.142234</td>
          <td>26.079918</td>
          <td>0.087973</td>
          <td>25.618088</td>
          <td>0.095539</td>
          <td>25.228781</td>
          <td>0.128682</td>
          <td>24.595413</td>
          <td>0.164777</td>
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
