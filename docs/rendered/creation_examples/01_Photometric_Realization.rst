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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f98404d6b30>



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
    0      23.994413  0.179687  0.120440  
    1      25.391064  0.236200  0.172064  
    2      24.304707  0.028239  0.026784  
    3      25.291103  0.162433  0.133156  
    4      25.096743  0.118931  0.103257  
    ...          ...       ...       ...  
    99995  24.737946  0.027429  0.022535  
    99996  24.224169  0.086398  0.070105  
    99997  25.613836  0.115062  0.074434  
    99998  25.274899  0.042559  0.023908  
    99999  25.699642  0.144102  0.134445  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>27.785384</td>
          <td>0.929913</td>
          <td>26.551097</td>
          <td>0.144718</td>
          <td>26.016650</td>
          <td>0.080014</td>
          <td>25.264412</td>
          <td>0.067121</td>
          <td>24.496150</td>
          <td>0.065059</td>
          <td>23.953394</td>
          <td>0.090721</td>
          <td>0.179687</td>
          <td>0.120440</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.074785</td>
          <td>1.105612</td>
          <td>27.268170</td>
          <td>0.264283</td>
          <td>26.560140</td>
          <td>0.128747</td>
          <td>26.288594</td>
          <td>0.164081</td>
          <td>25.661961</td>
          <td>0.179601</td>
          <td>25.003200</td>
          <td>0.223510</td>
          <td>0.236200</td>
          <td>0.172064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.913281</td>
          <td>1.709310</td>
          <td>29.870802</td>
          <td>1.512104</td>
          <td>28.670111</td>
          <td>0.683851</td>
          <td>25.838763</td>
          <td>0.111272</td>
          <td>25.053191</td>
          <td>0.106277</td>
          <td>24.269593</td>
          <td>0.119617</td>
          <td>0.028239</td>
          <td>0.026784</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.155042</td>
          <td>1.157540</td>
          <td>28.300969</td>
          <td>0.584346</td>
          <td>27.413721</td>
          <td>0.264603</td>
          <td>26.288428</td>
          <td>0.164058</td>
          <td>25.355899</td>
          <td>0.138236</td>
          <td>25.393922</td>
          <td>0.307561</td>
          <td>0.162433</td>
          <td>0.133156</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.486296</td>
          <td>0.373032</td>
          <td>26.175343</td>
          <td>0.104480</td>
          <td>26.020338</td>
          <td>0.080275</td>
          <td>25.781778</td>
          <td>0.105871</td>
          <td>25.312090</td>
          <td>0.133104</td>
          <td>24.519275</td>
          <td>0.148426</td>
          <td>0.118931</td>
          <td>0.103257</td>
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
          <td>27.200027</td>
          <td>0.632439</td>
          <td>26.388778</td>
          <td>0.125801</td>
          <td>25.373758</td>
          <td>0.045255</td>
          <td>25.075237</td>
          <td>0.056753</td>
          <td>24.968850</td>
          <td>0.098714</td>
          <td>24.948000</td>
          <td>0.213463</td>
          <td>0.027429</td>
          <td>0.022535</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.646039</td>
          <td>0.421871</td>
          <td>26.756461</td>
          <td>0.172483</td>
          <td>26.028423</td>
          <td>0.080849</td>
          <td>25.253535</td>
          <td>0.066477</td>
          <td>24.704894</td>
          <td>0.078254</td>
          <td>24.272192</td>
          <td>0.119888</td>
          <td>0.086398</td>
          <td>0.070105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.594965</td>
          <td>0.405712</td>
          <td>27.029324</td>
          <td>0.217013</td>
          <td>26.346784</td>
          <td>0.106934</td>
          <td>26.173557</td>
          <td>0.148693</td>
          <td>25.819115</td>
          <td>0.205038</td>
          <td>26.337941</td>
          <td>0.626592</td>
          <td>0.115062</td>
          <td>0.074434</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.105389</td>
          <td>0.275529</td>
          <td>26.257724</td>
          <td>0.112265</td>
          <td>26.194786</td>
          <td>0.093602</td>
          <td>25.816702</td>
          <td>0.109151</td>
          <td>25.883900</td>
          <td>0.216449</td>
          <td>26.734163</td>
          <td>0.818249</td>
          <td>0.042559</td>
          <td>0.023908</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.077048</td>
          <td>0.579899</td>
          <td>26.763499</td>
          <td>0.173517</td>
          <td>26.435372</td>
          <td>0.115526</td>
          <td>26.044478</td>
          <td>0.133041</td>
          <td>25.965590</td>
          <td>0.231656</td>
          <td>25.360714</td>
          <td>0.299469</td>
          <td>0.144102</td>
          <td>0.134445</td>
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
          <td>26.229322</td>
          <td>0.356234</td>
          <td>26.732777</td>
          <td>0.206699</td>
          <td>25.929791</td>
          <td>0.093790</td>
          <td>25.318518</td>
          <td>0.089982</td>
          <td>24.556677</td>
          <td>0.086904</td>
          <td>23.959389</td>
          <td>0.116057</td>
          <td>0.179687</td>
          <td>0.120440</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.130771</td>
          <td>0.342325</td>
          <td>27.130536</td>
          <td>0.299807</td>
          <td>26.812274</td>
          <td>0.210896</td>
          <td>27.112011</td>
          <td>0.422275</td>
          <td>25.960709</td>
          <td>0.301244</td>
          <td>25.410640</td>
          <td>0.405849</td>
          <td>0.236200</td>
          <td>0.172064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.897516</td>
          <td>0.967346</td>
          <td>27.718814</td>
          <td>0.391311</td>
          <td>25.931944</td>
          <td>0.142817</td>
          <td>24.978816</td>
          <td>0.117215</td>
          <td>24.246559</td>
          <td>0.138591</td>
          <td>0.028239</td>
          <td>0.026784</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.652246</td>
          <td>0.965670</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.909547</td>
          <td>0.216849</td>
          <td>26.325741</td>
          <td>0.213295</td>
          <td>25.607073</td>
          <td>0.213936</td>
          <td>25.992622</td>
          <td>0.594668</td>
          <td>0.162433</td>
          <td>0.133156</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.321969</td>
          <td>0.374360</td>
          <td>26.107655</td>
          <td>0.117726</td>
          <td>25.895447</td>
          <td>0.088082</td>
          <td>25.761158</td>
          <td>0.128078</td>
          <td>25.434170</td>
          <td>0.179891</td>
          <td>25.051162</td>
          <td>0.282473</td>
          <td>0.118931</td>
          <td>0.103257</td>
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
          <td>26.428235</td>
          <td>0.150046</td>
          <td>25.411071</td>
          <td>0.055217</td>
          <td>24.992939</td>
          <td>0.062700</td>
          <td>25.074580</td>
          <td>0.127321</td>
          <td>25.030822</td>
          <td>0.267911</td>
          <td>0.027429</td>
          <td>0.022535</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.555081</td>
          <td>0.885187</td>
          <td>26.813256</td>
          <td>0.211262</td>
          <td>26.064575</td>
          <td>0.100168</td>
          <td>25.264100</td>
          <td>0.081241</td>
          <td>24.766304</td>
          <td>0.099129</td>
          <td>24.138530</td>
          <td>0.128566</td>
          <td>0.086398</td>
          <td>0.070105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.655098</td>
          <td>0.479273</td>
          <td>26.798919</td>
          <td>0.210591</td>
          <td>26.350975</td>
          <td>0.129849</td>
          <td>26.214660</td>
          <td>0.186996</td>
          <td>26.185927</td>
          <td>0.330797</td>
          <td>25.641289</td>
          <td>0.444713</td>
          <td>0.115062</td>
          <td>0.074434</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.419378</td>
          <td>0.393854</td>
          <td>26.234823</td>
          <td>0.127230</td>
          <td>26.065304</td>
          <td>0.098586</td>
          <td>25.878083</td>
          <td>0.136542</td>
          <td>25.605622</td>
          <td>0.200735</td>
          <td>25.767853</td>
          <td>0.477660</td>
          <td>0.042559</td>
          <td>0.023908</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.627177</td>
          <td>0.479124</td>
          <td>26.782781</td>
          <td>0.213444</td>
          <td>26.613187</td>
          <td>0.167674</td>
          <td>26.513603</td>
          <td>0.247382</td>
          <td>26.314755</td>
          <td>0.376360</td>
          <td>25.423980</td>
          <td>0.387229</td>
          <td>0.144102</td>
          <td>0.134445</td>
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
          <td>28.113748</td>
          <td>1.265415</td>
          <td>26.833603</td>
          <td>0.225608</td>
          <td>25.989498</td>
          <td>0.099191</td>
          <td>25.141677</td>
          <td>0.077280</td>
          <td>24.945275</td>
          <td>0.122506</td>
          <td>24.032023</td>
          <td>0.124057</td>
          <td>0.179687</td>
          <td>0.120440</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.119193</td>
          <td>2.150300</td>
          <td>26.895680</td>
          <td>0.266840</td>
          <td>26.544935</td>
          <td>0.182836</td>
          <td>25.988579</td>
          <td>0.184655</td>
          <td>25.944633</td>
          <td>0.321504</td>
          <td>24.828153</td>
          <td>0.276946</td>
          <td>0.236200</td>
          <td>0.172064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.141416</td>
          <td>1.153882</td>
          <td>27.497742</td>
          <td>0.320647</td>
          <td>28.182588</td>
          <td>0.487007</td>
          <td>25.916062</td>
          <td>0.120288</td>
          <td>25.190580</td>
          <td>0.121010</td>
          <td>24.377744</td>
          <td>0.132756</td>
          <td>0.028239</td>
          <td>0.026784</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.014710</td>
          <td>1.081278</td>
          <td>27.369286</td>
          <td>0.315739</td>
          <td>26.056748</td>
          <td>0.170039</td>
          <td>25.322591</td>
          <td>0.168334</td>
          <td>26.042653</td>
          <td>0.616112</td>
          <td>0.162433</td>
          <td>0.133156</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.959435</td>
          <td>0.269499</td>
          <td>26.192011</td>
          <td>0.120282</td>
          <td>26.054029</td>
          <td>0.095545</td>
          <td>25.685625</td>
          <td>0.113038</td>
          <td>25.433081</td>
          <td>0.169951</td>
          <td>25.470659</td>
          <td>0.373624</td>
          <td>0.118931</td>
          <td>0.103257</td>
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
          <td>27.530939</td>
          <td>0.794414</td>
          <td>26.427672</td>
          <td>0.131056</td>
          <td>25.453527</td>
          <td>0.048995</td>
          <td>24.995776</td>
          <td>0.053367</td>
          <td>24.859840</td>
          <td>0.090469</td>
          <td>24.756584</td>
          <td>0.183275</td>
          <td>0.027429</td>
          <td>0.022535</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.432011</td>
          <td>1.391475</td>
          <td>26.192525</td>
          <td>0.113441</td>
          <td>26.101768</td>
          <td>0.093193</td>
          <td>25.137995</td>
          <td>0.065117</td>
          <td>24.776728</td>
          <td>0.090087</td>
          <td>24.502447</td>
          <td>0.158190</td>
          <td>0.086398</td>
          <td>0.070105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.576195</td>
          <td>2.344277</td>
          <td>26.966386</td>
          <td>0.225985</td>
          <td>26.254891</td>
          <td>0.110245</td>
          <td>26.232291</td>
          <td>0.175046</td>
          <td>25.873005</td>
          <td>0.238257</td>
          <td>25.003747</td>
          <td>0.249204</td>
          <td>0.115062</td>
          <td>0.074434</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.906642</td>
          <td>0.236592</td>
          <td>26.161301</td>
          <td>0.104633</td>
          <td>26.204706</td>
          <td>0.095930</td>
          <td>25.656094</td>
          <td>0.096423</td>
          <td>25.699490</td>
          <td>0.188242</td>
          <td>25.591812</td>
          <td>0.365105</td>
          <td>0.042559</td>
          <td>0.023908</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.054741</td>
          <td>0.646032</td>
          <td>27.145501</td>
          <td>0.284327</td>
          <td>26.667185</td>
          <td>0.173280</td>
          <td>26.350979</td>
          <td>0.213370</td>
          <td>26.031638</td>
          <td>0.297159</td>
          <td>26.383168</td>
          <td>0.764152</td>
          <td>0.144102</td>
          <td>0.134445</td>
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
