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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fc0e6ab6800>



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
    0      23.994413  0.123073  0.122805  
    1      25.391064  0.158462  0.124667  
    2      24.304707  0.021533  0.014599  
    3      25.291103  0.140619  0.127660  
    4      25.096743  0.063266  0.052875  
    ...          ...       ...       ...  
    99995  24.737946  0.145647  0.119625  
    99996  24.224169  0.110208  0.077847  
    99997  25.613836  0.009304  0.007202  
    99998  25.274899  0.007058  0.004670  
    99999  25.699642  0.063948  0.044139  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.757355</td>
          <td>0.172614</td>
          <td>25.955351</td>
          <td>0.075798</td>
          <td>25.235840</td>
          <td>0.065443</td>
          <td>24.731838</td>
          <td>0.080138</td>
          <td>24.103164</td>
          <td>0.103456</td>
          <td>0.123073</td>
          <td>0.122805</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.837839</td>
          <td>0.487305</td>
          <td>27.978531</td>
          <td>0.461612</td>
          <td>26.555426</td>
          <td>0.128222</td>
          <td>26.176040</td>
          <td>0.149011</td>
          <td>25.702709</td>
          <td>0.185902</td>
          <td>25.640719</td>
          <td>0.373801</td>
          <td>0.158462</td>
          <td>0.124667</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.707497</td>
          <td>0.772241</td>
          <td>28.251605</td>
          <td>0.508110</td>
          <td>26.026578</td>
          <td>0.130997</td>
          <td>24.838355</td>
          <td>0.088024</td>
          <td>24.204864</td>
          <td>0.113064</td>
          <td>0.021533</td>
          <td>0.014599</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.575463</td>
          <td>0.399678</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.234800</td>
          <td>0.228362</td>
          <td>26.438597</td>
          <td>0.186370</td>
          <td>25.640258</td>
          <td>0.176326</td>
          <td>25.283370</td>
          <td>0.281339</td>
          <td>0.140619</td>
          <td>0.127660</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.838022</td>
          <td>0.487371</td>
          <td>26.066546</td>
          <td>0.094991</td>
          <td>25.966436</td>
          <td>0.076544</td>
          <td>25.779010</td>
          <td>0.105615</td>
          <td>25.332798</td>
          <td>0.135507</td>
          <td>24.826910</td>
          <td>0.192845</td>
          <td>0.063266</td>
          <td>0.052875</td>
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
          <td>27.141373</td>
          <td>0.606955</td>
          <td>26.407409</td>
          <td>0.127848</td>
          <td>25.505265</td>
          <td>0.050859</td>
          <td>25.082515</td>
          <td>0.057121</td>
          <td>24.868667</td>
          <td>0.090403</td>
          <td>24.426073</td>
          <td>0.136981</td>
          <td>0.145647</td>
          <td>0.119625</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.129770</td>
          <td>1.141043</td>
          <td>26.551594</td>
          <td>0.144780</td>
          <td>26.225495</td>
          <td>0.096159</td>
          <td>25.172452</td>
          <td>0.061867</td>
          <td>24.776744</td>
          <td>0.083375</td>
          <td>24.382828</td>
          <td>0.131958</td>
          <td>0.110208</td>
          <td>0.077847</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.122044</td>
          <td>0.598727</td>
          <td>26.724322</td>
          <td>0.167833</td>
          <td>26.382942</td>
          <td>0.110365</td>
          <td>26.234383</td>
          <td>0.156653</td>
          <td>26.263346</td>
          <td>0.295490</td>
          <td>26.396554</td>
          <td>0.652676</td>
          <td>0.009304</td>
          <td>0.007202</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.024657</td>
          <td>0.257994</td>
          <td>26.111700</td>
          <td>0.098824</td>
          <td>26.076185</td>
          <td>0.084327</td>
          <td>26.069158</td>
          <td>0.135908</td>
          <td>25.863486</td>
          <td>0.212792</td>
          <td>24.819251</td>
          <td>0.191605</td>
          <td>0.007058</td>
          <td>0.004670</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.679787</td>
          <td>0.432836</td>
          <td>26.521513</td>
          <td>0.141082</td>
          <td>26.704903</td>
          <td>0.145880</td>
          <td>26.381897</td>
          <td>0.177636</td>
          <td>25.971804</td>
          <td>0.232852</td>
          <td>25.583396</td>
          <td>0.357425</td>
          <td>0.063948</td>
          <td>0.044139</td>
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
          <td>26.491232</td>
          <td>0.428829</td>
          <td>26.919060</td>
          <td>0.236342</td>
          <td>26.155055</td>
          <td>0.111508</td>
          <td>25.105226</td>
          <td>0.072718</td>
          <td>24.564047</td>
          <td>0.085379</td>
          <td>24.297393</td>
          <td>0.151688</td>
          <td>0.123073</td>
          <td>0.122805</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.912629</td>
          <td>0.238249</td>
          <td>26.649630</td>
          <td>0.173373</td>
          <td>26.318970</td>
          <td>0.211020</td>
          <td>25.942876</td>
          <td>0.280686</td>
          <td>25.938533</td>
          <td>0.569732</td>
          <td>0.158462</td>
          <td>0.124667</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.258008</td>
          <td>0.721537</td>
          <td>29.014609</td>
          <td>1.037002</td>
          <td>29.319638</td>
          <td>1.152952</td>
          <td>26.082483</td>
          <td>0.162257</td>
          <td>24.957714</td>
          <td>0.114920</td>
          <td>24.198495</td>
          <td>0.132768</td>
          <td>0.021533</td>
          <td>0.014599</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.484860</td>
          <td>0.864450</td>
          <td>29.930450</td>
          <td>1.730283</td>
          <td>27.300042</td>
          <td>0.295423</td>
          <td>26.005799</td>
          <td>0.160872</td>
          <td>25.378181</td>
          <td>0.174440</td>
          <td>25.511253</td>
          <td>0.412594</td>
          <td>0.140619</td>
          <td>0.127660</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.354715</td>
          <td>0.376495</td>
          <td>26.052401</td>
          <td>0.109308</td>
          <td>25.826230</td>
          <td>0.080490</td>
          <td>25.625871</td>
          <td>0.110535</td>
          <td>25.336366</td>
          <td>0.160944</td>
          <td>25.417195</td>
          <td>0.367931</td>
          <td>0.063266</td>
          <td>0.052875</td>
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
          <td>31.237346</td>
          <td>3.992139</td>
          <td>26.459680</td>
          <td>0.161775</td>
          <td>25.327910</td>
          <td>0.054215</td>
          <td>25.077306</td>
          <td>0.071535</td>
          <td>24.739794</td>
          <td>0.100421</td>
          <td>24.912373</td>
          <td>0.256234</td>
          <td>0.145647</td>
          <td>0.119625</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.230405</td>
          <td>0.720319</td>
          <td>26.655681</td>
          <td>0.186583</td>
          <td>26.111350</td>
          <td>0.105328</td>
          <td>25.199492</td>
          <td>0.077488</td>
          <td>24.811069</td>
          <td>0.104053</td>
          <td>24.212231</td>
          <td>0.138317</td>
          <td>0.110208</td>
          <td>0.077847</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.524820</td>
          <td>0.425906</td>
          <td>26.752895</td>
          <td>0.197347</td>
          <td>26.350224</td>
          <td>0.125908</td>
          <td>26.038893</td>
          <td>0.156178</td>
          <td>26.164630</td>
          <td>0.316362</td>
          <td>26.095288</td>
          <td>0.603857</td>
          <td>0.009304</td>
          <td>0.007202</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.425002</td>
          <td>0.394533</td>
          <td>26.027416</td>
          <td>0.105871</td>
          <td>25.960229</td>
          <td>0.089548</td>
          <td>25.809383</td>
          <td>0.128155</td>
          <td>26.223719</td>
          <td>0.331557</td>
          <td>26.206619</td>
          <td>0.652676</td>
          <td>0.007058</td>
          <td>0.004670</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.049092</td>
          <td>0.628615</td>
          <td>26.792769</td>
          <td>0.205812</td>
          <td>26.353740</td>
          <td>0.127536</td>
          <td>26.731931</td>
          <td>0.281325</td>
          <td>25.895926</td>
          <td>0.256887</td>
          <td>25.301583</td>
          <td>0.335569</td>
          <td>0.063948</td>
          <td>0.044139</td>
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
          <td>30.025112</td>
          <td>2.796397</td>
          <td>26.918704</td>
          <td>0.228426</td>
          <td>25.867972</td>
          <td>0.083335</td>
          <td>25.152933</td>
          <td>0.072789</td>
          <td>24.662990</td>
          <td>0.089524</td>
          <td>24.270424</td>
          <td>0.142445</td>
          <td>0.123073</td>
          <td>0.122805</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.974418</td>
          <td>0.613289</td>
          <td>27.201208</td>
          <td>0.298981</td>
          <td>26.489398</td>
          <td>0.149776</td>
          <td>26.459641</td>
          <td>0.234945</td>
          <td>25.670247</td>
          <td>0.222331</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.158462</td>
          <td>0.124667</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.151407</td>
          <td>0.526340</td>
          <td>27.432729</td>
          <td>0.269880</td>
          <td>25.932902</td>
          <td>0.121352</td>
          <td>25.060198</td>
          <td>0.107417</td>
          <td>24.343048</td>
          <td>0.128091</td>
          <td>0.021533</td>
          <td>0.014599</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.974590</td>
          <td>1.147358</td>
          <td>28.952583</td>
          <td>1.021060</td>
          <td>27.067792</td>
          <td>0.239562</td>
          <td>25.971624</td>
          <td>0.152866</td>
          <td>25.356714</td>
          <td>0.167729</td>
          <td>25.604815</td>
          <td>0.434629</td>
          <td>0.140619</td>
          <td>0.127660</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.863853</td>
          <td>0.232690</td>
          <td>26.212151</td>
          <td>0.112091</td>
          <td>26.074590</td>
          <td>0.088015</td>
          <td>25.597288</td>
          <td>0.094316</td>
          <td>25.396084</td>
          <td>0.149404</td>
          <td>25.324529</td>
          <td>0.303299</td>
          <td>0.063266</td>
          <td>0.052875</td>
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
          <td>26.382161</td>
          <td>0.390268</td>
          <td>26.416284</td>
          <td>0.152618</td>
          <td>25.477215</td>
          <td>0.060404</td>
          <td>25.017777</td>
          <td>0.066186</td>
          <td>25.169651</td>
          <td>0.142455</td>
          <td>24.452702</td>
          <td>0.170449</td>
          <td>0.145647</td>
          <td>0.119625</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.800276</td>
          <td>0.506133</td>
          <td>26.512780</td>
          <td>0.153790</td>
          <td>25.974233</td>
          <td>0.085989</td>
          <td>25.172149</td>
          <td>0.069384</td>
          <td>24.837799</td>
          <td>0.098102</td>
          <td>24.276266</td>
          <td>0.134466</td>
          <td>0.110208</td>
          <td>0.077847</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.599112</td>
          <td>1.468700</td>
          <td>26.669179</td>
          <td>0.160252</td>
          <td>26.365542</td>
          <td>0.108803</td>
          <td>26.168232</td>
          <td>0.148158</td>
          <td>25.959468</td>
          <td>0.230688</td>
          <td>25.604595</td>
          <td>0.363725</td>
          <td>0.009304</td>
          <td>0.007202</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.725982</td>
          <td>0.448350</td>
          <td>25.965836</td>
          <td>0.086989</td>
          <td>26.196417</td>
          <td>0.093781</td>
          <td>25.880292</td>
          <td>0.115432</td>
          <td>26.058451</td>
          <td>0.250218</td>
          <td>25.230013</td>
          <td>0.269525</td>
          <td>0.007058</td>
          <td>0.004670</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.755265</td>
          <td>0.469057</td>
          <td>26.753424</td>
          <td>0.177813</td>
          <td>26.744982</td>
          <td>0.156871</td>
          <td>26.187027</td>
          <td>0.156572</td>
          <td>25.949099</td>
          <td>0.237094</td>
          <td>26.278046</td>
          <td>0.620628</td>
          <td>0.063948</td>
          <td>0.044139</td>
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
