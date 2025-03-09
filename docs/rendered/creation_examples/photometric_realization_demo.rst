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

    <pzflow.flow.Flow at 0x7efc67cb3c70>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>30.076801</td>
          <td>2.713740</td>
          <td>26.938977</td>
          <td>0.201223</td>
          <td>26.030847</td>
          <td>0.081023</td>
          <td>25.193508</td>
          <td>0.063033</td>
          <td>24.897883</td>
          <td>0.092754</td>
          <td>23.984693</td>
          <td>0.093250</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.198672</td>
          <td>0.631841</td>
          <td>27.678973</td>
          <td>0.367002</td>
          <td>26.473822</td>
          <td>0.119456</td>
          <td>26.504925</td>
          <td>0.197088</td>
          <td>26.108313</td>
          <td>0.260541</td>
          <td>25.144725</td>
          <td>0.251242</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.605022</td>
          <td>0.829880</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.125856</td>
          <td>0.920637</td>
          <td>25.862038</td>
          <td>0.113554</td>
          <td>25.002706</td>
          <td>0.101686</td>
          <td>24.410703</td>
          <td>0.135176</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.470006</td>
          <td>0.759868</td>
          <td>27.943003</td>
          <td>0.449447</td>
          <td>27.807299</td>
          <td>0.362554</td>
          <td>26.427458</td>
          <td>0.184624</td>
          <td>25.772781</td>
          <td>0.197215</td>
          <td>25.670278</td>
          <td>0.382489</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.218232</td>
          <td>0.301796</td>
          <td>26.005434</td>
          <td>0.090031</td>
          <td>25.933547</td>
          <td>0.074351</td>
          <td>25.661978</td>
          <td>0.095323</td>
          <td>25.467483</td>
          <td>0.152160</td>
          <td>25.511500</td>
          <td>0.337747</td>
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
          <td>27.338792</td>
          <td>0.695847</td>
          <td>26.230718</td>
          <td>0.109654</td>
          <td>25.461348</td>
          <td>0.048914</td>
          <td>25.107338</td>
          <td>0.058394</td>
          <td>24.805551</td>
          <td>0.085518</td>
          <td>24.709506</td>
          <td>0.174613</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.754041</td>
          <td>0.172129</td>
          <td>26.225539</td>
          <td>0.096163</td>
          <td>25.281070</td>
          <td>0.068119</td>
          <td>24.753229</td>
          <td>0.081664</td>
          <td>24.164699</td>
          <td>0.109172</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.190203</td>
          <td>0.628116</td>
          <td>26.605831</td>
          <td>0.151679</td>
          <td>26.321623</td>
          <td>0.104608</td>
          <td>26.300439</td>
          <td>0.165747</td>
          <td>26.008716</td>
          <td>0.240067</td>
          <td>25.587432</td>
          <td>0.358558</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.322374</td>
          <td>0.327944</td>
          <td>26.253874</td>
          <td>0.111889</td>
          <td>26.021919</td>
          <td>0.080387</td>
          <td>25.861864</td>
          <td>0.113536</td>
          <td>25.955367</td>
          <td>0.229702</td>
          <td>25.602661</td>
          <td>0.362860</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.917934</td>
          <td>0.516921</td>
          <td>27.010770</td>
          <td>0.213681</td>
          <td>26.765796</td>
          <td>0.153708</td>
          <td>26.318978</td>
          <td>0.168386</td>
          <td>25.729883</td>
          <td>0.190217</td>
          <td>26.478432</td>
          <td>0.690426</td>
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
          <td>1.398944</td>
          <td>27.458771</td>
          <td>0.823055</td>
          <td>26.820595</td>
          <td>0.208833</td>
          <td>26.004533</td>
          <td>0.093094</td>
          <td>25.175294</td>
          <td>0.073521</td>
          <td>24.882282</td>
          <td>0.107485</td>
          <td>23.762235</td>
          <td>0.090649</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.733925</td>
          <td>0.977901</td>
          <td>27.487685</td>
          <td>0.358912</td>
          <td>26.614854</td>
          <td>0.158136</td>
          <td>26.569404</td>
          <td>0.243986</td>
          <td>25.836472</td>
          <td>0.242377</td>
          <td>25.405360</td>
          <td>0.360827</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.418760</td>
          <td>0.720902</td>
          <td>28.431658</td>
          <td>0.670211</td>
          <td>26.405368</td>
          <td>0.217614</td>
          <td>25.183946</td>
          <td>0.142735</td>
          <td>24.534389</td>
          <td>0.180786</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.405997</td>
          <td>0.355385</td>
          <td>28.275258</td>
          <td>0.622942</td>
          <td>26.294785</td>
          <td>0.207340</td>
          <td>25.340340</td>
          <td>0.170456</td>
          <td>24.804996</td>
          <td>0.236889</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.064942</td>
          <td>0.297138</td>
          <td>26.105508</td>
          <td>0.113350</td>
          <td>25.938326</td>
          <td>0.087860</td>
          <td>25.768326</td>
          <td>0.123704</td>
          <td>25.360348</td>
          <td>0.162514</td>
          <td>25.244250</td>
          <td>0.317712</td>
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
          <td>27.821039</td>
          <td>1.041523</td>
          <td>26.201999</td>
          <td>0.125554</td>
          <td>25.381490</td>
          <td>0.054826</td>
          <td>25.081480</td>
          <td>0.069166</td>
          <td>24.861238</td>
          <td>0.107753</td>
          <td>24.603451</td>
          <td>0.191402</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.582257</td>
          <td>0.892397</td>
          <td>26.557639</td>
          <td>0.167886</td>
          <td>26.090892</td>
          <td>0.100837</td>
          <td>25.208492</td>
          <td>0.076039</td>
          <td>24.940060</td>
          <td>0.113509</td>
          <td>24.158037</td>
          <td>0.128598</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.086088</td>
          <td>0.646047</td>
          <td>26.635151</td>
          <td>0.180653</td>
          <td>26.272038</td>
          <td>0.119119</td>
          <td>26.189822</td>
          <td>0.179856</td>
          <td>26.373877</td>
          <td>0.377308</td>
          <td>26.851882</td>
          <td>1.000730</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.385278</td>
          <td>0.390765</td>
          <td>26.205046</td>
          <td>0.127063</td>
          <td>26.202901</td>
          <td>0.114255</td>
          <td>26.213509</td>
          <td>0.186930</td>
          <td>25.682560</td>
          <td>0.219724</td>
          <td>25.151301</td>
          <td>0.303637</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.224085</td>
          <td>0.708905</td>
          <td>27.037187</td>
          <td>0.252024</td>
          <td>26.447860</td>
          <td>0.138331</td>
          <td>26.022621</td>
          <td>0.155553</td>
          <td>26.278377</td>
          <td>0.349294</td>
          <td>24.930909</td>
          <td>0.248748</td>
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
          <td>1.398944</td>
          <td>26.903729</td>
          <td>0.511606</td>
          <td>26.802956</td>
          <td>0.179441</td>
          <td>26.043201</td>
          <td>0.081921</td>
          <td>25.109949</td>
          <td>0.058537</td>
          <td>24.669945</td>
          <td>0.075886</td>
          <td>24.003428</td>
          <td>0.094810</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.784910</td>
          <td>0.398717</td>
          <td>26.452836</td>
          <td>0.117405</td>
          <td>26.251185</td>
          <td>0.159075</td>
          <td>25.861413</td>
          <td>0.212615</td>
          <td>25.435526</td>
          <td>0.318249</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.623511</td>
          <td>0.876127</td>
          <td>27.702235</td>
          <td>0.398170</td>
          <td>28.016439</td>
          <td>0.457175</td>
          <td>26.141879</td>
          <td>0.157304</td>
          <td>24.858974</td>
          <td>0.097277</td>
          <td>24.492624</td>
          <td>0.157578</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.425562</td>
          <td>0.835078</td>
          <td>28.450217</td>
          <td>0.757468</td>
          <td>27.384621</td>
          <td>0.317824</td>
          <td>26.095696</td>
          <td>0.174679</td>
          <td>25.677373</td>
          <td>0.225532</td>
          <td>25.187875</td>
          <td>0.322161</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.110232</td>
          <td>0.276865</td>
          <td>26.137187</td>
          <td>0.101177</td>
          <td>25.906955</td>
          <td>0.072727</td>
          <td>25.636089</td>
          <td>0.093320</td>
          <td>25.384420</td>
          <td>0.141874</td>
          <td>25.393859</td>
          <td>0.307959</td>
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
          <td>27.054433</td>
          <td>0.597003</td>
          <td>26.488120</td>
          <td>0.146668</td>
          <td>25.387626</td>
          <td>0.049620</td>
          <td>25.221689</td>
          <td>0.070232</td>
          <td>24.826048</td>
          <td>0.094212</td>
          <td>24.890297</td>
          <td>0.219805</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.123956</td>
          <td>0.605092</td>
          <td>26.647261</td>
          <td>0.159360</td>
          <td>26.111710</td>
          <td>0.088457</td>
          <td>25.145152</td>
          <td>0.061452</td>
          <td>24.742792</td>
          <td>0.082265</td>
          <td>24.263643</td>
          <td>0.121033</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.713191</td>
          <td>0.457212</td>
          <td>26.952898</td>
          <td>0.212038</td>
          <td>26.347180</td>
          <td>0.112301</td>
          <td>26.738698</td>
          <td>0.251266</td>
          <td>26.045181</td>
          <td>0.258937</td>
          <td>25.462105</td>
          <td>0.339975</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.234558</td>
          <td>0.329446</td>
          <td>26.134004</td>
          <td>0.111415</td>
          <td>26.007646</td>
          <td>0.089067</td>
          <td>25.834739</td>
          <td>0.124851</td>
          <td>25.307745</td>
          <td>0.148376</td>
          <td>25.379547</td>
          <td>0.338558</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.243155</td>
          <td>0.315616</td>
          <td>26.949154</td>
          <td>0.209613</td>
          <td>26.654990</td>
          <td>0.145176</td>
          <td>26.320893</td>
          <td>0.175451</td>
          <td>25.769970</td>
          <td>0.204194</td>
          <td>25.815817</td>
          <td>0.442912</td>
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
