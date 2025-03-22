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

    <pzflow.flow.Flow at 0x7f9785043df0>



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
          <td>28.037517</td>
          <td>1.081963</td>
          <td>26.484747</td>
          <td>0.136682</td>
          <td>25.842084</td>
          <td>0.068571</td>
          <td>25.113923</td>
          <td>0.058736</td>
          <td>24.681537</td>
          <td>0.076657</td>
          <td>24.058310</td>
          <td>0.099472</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.189971</td>
          <td>0.247882</td>
          <td>26.650748</td>
          <td>0.139233</td>
          <td>26.045572</td>
          <td>0.133167</td>
          <td>25.947611</td>
          <td>0.228229</td>
          <td>25.458871</td>
          <td>0.323934</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.349696</td>
          <td>1.288988</td>
          <td>29.555494</td>
          <td>1.284101</td>
          <td>28.952542</td>
          <td>0.824978</td>
          <td>26.407656</td>
          <td>0.181556</td>
          <td>25.072573</td>
          <td>0.108092</td>
          <td>24.199139</td>
          <td>0.112502</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.177166</td>
          <td>0.622414</td>
          <td>27.848204</td>
          <td>0.418253</td>
          <td>27.053123</td>
          <td>0.196197</td>
          <td>26.201630</td>
          <td>0.152319</td>
          <td>25.741395</td>
          <td>0.192072</td>
          <td>25.655589</td>
          <td>0.378151</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.237794</td>
          <td>0.306565</td>
          <td>26.063352</td>
          <td>0.094725</td>
          <td>26.053178</td>
          <td>0.082634</td>
          <td>25.688857</td>
          <td>0.097598</td>
          <td>25.361583</td>
          <td>0.138915</td>
          <td>24.919779</td>
          <td>0.208486</td>
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
          <td>28.986891</td>
          <td>1.768066</td>
          <td>26.336302</td>
          <td>0.120205</td>
          <td>25.396562</td>
          <td>0.046180</td>
          <td>25.046405</td>
          <td>0.055319</td>
          <td>24.898362</td>
          <td>0.092793</td>
          <td>24.729665</td>
          <td>0.177626</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.336623</td>
          <td>0.331670</td>
          <td>26.731760</td>
          <td>0.168899</td>
          <td>26.121434</td>
          <td>0.087755</td>
          <td>25.230451</td>
          <td>0.065131</td>
          <td>24.789207</td>
          <td>0.084296</td>
          <td>24.476265</td>
          <td>0.143037</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.134369</td>
          <td>0.282076</td>
          <td>26.672608</td>
          <td>0.160595</td>
          <td>26.496013</td>
          <td>0.121782</td>
          <td>26.558408</td>
          <td>0.206138</td>
          <td>26.325251</td>
          <td>0.310551</td>
          <td>25.898871</td>
          <td>0.455484</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.272581</td>
          <td>0.315208</td>
          <td>26.158636</td>
          <td>0.102966</td>
          <td>26.019918</td>
          <td>0.080245</td>
          <td>26.010728</td>
          <td>0.129212</td>
          <td>26.045854</td>
          <td>0.247527</td>
          <td>25.086735</td>
          <td>0.239526</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.727316</td>
          <td>0.168262</td>
          <td>26.750203</td>
          <td>0.151667</td>
          <td>26.346625</td>
          <td>0.172394</td>
          <td>26.049553</td>
          <td>0.248281</td>
          <td>25.500842</td>
          <td>0.334910</td>
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
          <td>27.480051</td>
          <td>0.834407</td>
          <td>26.784718</td>
          <td>0.202654</td>
          <td>26.146063</td>
          <td>0.105384</td>
          <td>25.240822</td>
          <td>0.077902</td>
          <td>24.745806</td>
          <td>0.095382</td>
          <td>23.927292</td>
          <td>0.104760</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.895123</td>
          <td>1.076415</td>
          <td>27.304878</td>
          <td>0.310542</td>
          <td>26.608603</td>
          <td>0.157293</td>
          <td>26.438623</td>
          <td>0.218932</td>
          <td>26.071415</td>
          <td>0.293566</td>
          <td>25.128134</td>
          <td>0.289402</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.703521</td>
          <td>1.664377</td>
          <td>29.369676</td>
          <td>1.283801</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.945976</td>
          <td>0.147472</td>
          <td>24.957444</td>
          <td>0.117328</td>
          <td>24.413006</td>
          <td>0.163062</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.914469</td>
          <td>0.522647</td>
          <td>27.567060</td>
          <td>0.368228</td>
          <td>26.518357</td>
          <td>0.249596</td>
          <td>25.508477</td>
          <td>0.196506</td>
          <td>25.940168</td>
          <td>0.571677</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.159733</td>
          <td>0.320540</td>
          <td>26.156250</td>
          <td>0.118462</td>
          <td>25.971993</td>
          <td>0.090500</td>
          <td>25.660938</td>
          <td>0.112675</td>
          <td>25.180637</td>
          <td>0.139296</td>
          <td>25.243725</td>
          <td>0.317579</td>
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
          <td>26.475373</td>
          <td>0.158843</td>
          <td>25.317904</td>
          <td>0.051818</td>
          <td>25.180380</td>
          <td>0.075486</td>
          <td>24.834014</td>
          <td>0.105221</td>
          <td>25.014466</td>
          <td>0.269162</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.636659</td>
          <td>0.179533</td>
          <td>26.036399</td>
          <td>0.096135</td>
          <td>25.130917</td>
          <td>0.071000</td>
          <td>24.662633</td>
          <td>0.089033</td>
          <td>24.242845</td>
          <td>0.138375</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.268350</td>
          <td>0.352270</td>
          <td>26.539536</td>
          <td>0.166570</td>
          <td>26.329466</td>
          <td>0.125208</td>
          <td>26.302282</td>
          <td>0.197762</td>
          <td>26.145087</td>
          <td>0.315040</td>
          <td>26.580495</td>
          <td>0.845584</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.967388</td>
          <td>0.280724</td>
          <td>26.293942</td>
          <td>0.137203</td>
          <td>26.076134</td>
          <td>0.102285</td>
          <td>25.813135</td>
          <td>0.132743</td>
          <td>25.588882</td>
          <td>0.203180</td>
          <td>25.552896</td>
          <td>0.416050</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.130258</td>
          <td>0.664964</td>
          <td>26.964656</td>
          <td>0.237417</td>
          <td>26.806532</td>
          <td>0.187891</td>
          <td>26.421462</td>
          <td>0.217917</td>
          <td>26.087846</td>
          <td>0.300164</td>
          <td>26.254713</td>
          <td>0.680056</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.537652</td>
          <td>0.143070</td>
          <td>26.049433</td>
          <td>0.082373</td>
          <td>25.128155</td>
          <td>0.059491</td>
          <td>24.578561</td>
          <td>0.069994</td>
          <td>24.045610</td>
          <td>0.098384</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.062369</td>
          <td>0.574160</td>
          <td>27.835840</td>
          <td>0.414614</td>
          <td>27.073196</td>
          <td>0.199717</td>
          <td>26.587496</td>
          <td>0.211417</td>
          <td>25.910281</td>
          <td>0.221457</td>
          <td>25.295190</td>
          <td>0.284303</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.585062</td>
          <td>0.751751</td>
          <td>28.152971</td>
          <td>0.506034</td>
          <td>26.087884</td>
          <td>0.150192</td>
          <td>25.215670</td>
          <td>0.132722</td>
          <td>24.173994</td>
          <td>0.119711</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.016973</td>
          <td>0.635165</td>
          <td>28.031531</td>
          <td>0.567336</td>
          <td>27.434934</td>
          <td>0.330803</td>
          <td>25.977862</td>
          <td>0.157992</td>
          <td>25.720555</td>
          <td>0.233751</td>
          <td>26.109472</td>
          <td>0.642312</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.479566</td>
          <td>0.371406</td>
          <td>26.107858</td>
          <td>0.098613</td>
          <td>26.038855</td>
          <td>0.081713</td>
          <td>25.547141</td>
          <td>0.086297</td>
          <td>25.553782</td>
          <td>0.164043</td>
          <td>24.927159</td>
          <td>0.210072</td>
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
          <td>27.013732</td>
          <td>0.579990</td>
          <td>26.296758</td>
          <td>0.124346</td>
          <td>25.447720</td>
          <td>0.052339</td>
          <td>25.027756</td>
          <td>0.059143</td>
          <td>24.797365</td>
          <td>0.091868</td>
          <td>24.769717</td>
          <td>0.198715</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.528394</td>
          <td>0.796392</td>
          <td>26.562545</td>
          <td>0.148210</td>
          <td>25.928496</td>
          <td>0.075258</td>
          <td>25.240013</td>
          <td>0.066843</td>
          <td>24.800103</td>
          <td>0.086526</td>
          <td>24.158009</td>
          <td>0.110398</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.659613</td>
          <td>0.439120</td>
          <td>26.849239</td>
          <td>0.194391</td>
          <td>26.362555</td>
          <td>0.113816</td>
          <td>26.398233</td>
          <td>0.189216</td>
          <td>26.169887</td>
          <td>0.286592</td>
          <td>25.556047</td>
          <td>0.366021</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.916683</td>
          <td>0.254985</td>
          <td>26.206056</td>
          <td>0.118621</td>
          <td>26.093758</td>
          <td>0.096065</td>
          <td>25.940926</td>
          <td>0.136864</td>
          <td>25.530436</td>
          <td>0.179423</td>
          <td>24.786659</td>
          <td>0.208822</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.705164</td>
          <td>0.451680</td>
          <td>26.688918</td>
          <td>0.168292</td>
          <td>26.586173</td>
          <td>0.136819</td>
          <td>25.920238</td>
          <td>0.124372</td>
          <td>25.973251</td>
          <td>0.241807</td>
          <td>25.490898</td>
          <td>0.344573</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
