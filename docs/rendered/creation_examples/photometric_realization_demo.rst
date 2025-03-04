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

    <pzflow.flow.Flow at 0x7f3415527fa0>



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
          <td>26.345249</td>
          <td>0.333942</td>
          <td>27.276791</td>
          <td>0.266149</td>
          <td>26.117168</td>
          <td>0.087426</td>
          <td>25.289794</td>
          <td>0.068647</td>
          <td>24.738991</td>
          <td>0.080645</td>
          <td>23.955397</td>
          <td>0.090881</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.227870</td>
          <td>0.554532</td>
          <td>26.831119</td>
          <td>0.162540</td>
          <td>26.263320</td>
          <td>0.160578</td>
          <td>26.373538</td>
          <td>0.322756</td>
          <td>25.335668</td>
          <td>0.293490</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.487305</td>
          <td>0.768604</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.932754</td>
          <td>0.399645</td>
          <td>25.967348</td>
          <td>0.124444</td>
          <td>25.048076</td>
          <td>0.105803</td>
          <td>24.104772</td>
          <td>0.103602</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.264095</td>
          <td>0.661168</td>
          <td>28.656052</td>
          <td>0.746369</td>
          <td>27.306132</td>
          <td>0.242242</td>
          <td>26.218261</td>
          <td>0.154506</td>
          <td>25.608976</td>
          <td>0.171702</td>
          <td>25.248116</td>
          <td>0.273399</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.486989</td>
          <td>0.373233</td>
          <td>26.308228</td>
          <td>0.117308</td>
          <td>25.974892</td>
          <td>0.077118</td>
          <td>25.715586</td>
          <td>0.099912</td>
          <td>25.191067</td>
          <td>0.119848</td>
          <td>25.125842</td>
          <td>0.247372</td>
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
          <td>26.270104</td>
          <td>0.314585</td>
          <td>26.450409</td>
          <td>0.132691</td>
          <td>25.442004</td>
          <td>0.048081</td>
          <td>25.059892</td>
          <td>0.055986</td>
          <td>24.933500</td>
          <td>0.095701</td>
          <td>24.416469</td>
          <td>0.135850</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.116877</td>
          <td>1.132678</td>
          <td>26.566856</td>
          <td>0.146691</td>
          <td>26.023103</td>
          <td>0.080471</td>
          <td>25.177439</td>
          <td>0.062141</td>
          <td>24.951043</td>
          <td>0.097185</td>
          <td>24.242810</td>
          <td>0.116863</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.621614</td>
          <td>0.414079</td>
          <td>26.569247</td>
          <td>0.146992</td>
          <td>26.347868</td>
          <td>0.107036</td>
          <td>26.234114</td>
          <td>0.156617</td>
          <td>25.831441</td>
          <td>0.207166</td>
          <td>25.702432</td>
          <td>0.392131</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.350351</td>
          <td>0.335293</td>
          <td>26.100439</td>
          <td>0.097854</td>
          <td>26.365621</td>
          <td>0.108708</td>
          <td>26.078317</td>
          <td>0.136987</td>
          <td>25.762362</td>
          <td>0.195494</td>
          <td>25.831374</td>
          <td>0.432837</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.878877</td>
          <td>0.191307</td>
          <td>26.326145</td>
          <td>0.105022</td>
          <td>26.418362</td>
          <td>0.183209</td>
          <td>25.946626</td>
          <td>0.228043</td>
          <td>26.562501</td>
          <td>0.730786</td>
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
          <td>27.385915</td>
          <td>0.784972</td>
          <td>26.893047</td>
          <td>0.221836</td>
          <td>26.084909</td>
          <td>0.099894</td>
          <td>25.197980</td>
          <td>0.075009</td>
          <td>24.597174</td>
          <td>0.083698</td>
          <td>23.899308</td>
          <td>0.102227</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.242994</td>
          <td>0.295494</td>
          <td>26.532672</td>
          <td>0.147380</td>
          <td>26.388266</td>
          <td>0.209922</td>
          <td>25.961935</td>
          <td>0.268633</td>
          <td>25.501138</td>
          <td>0.388756</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.040973</td>
          <td>0.629715</td>
          <td>29.505330</td>
          <td>1.379629</td>
          <td>28.238492</td>
          <td>0.585511</td>
          <td>25.999705</td>
          <td>0.154427</td>
          <td>25.002294</td>
          <td>0.121991</td>
          <td>24.189630</td>
          <td>0.134605</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.840496</td>
          <td>0.494995</td>
          <td>27.351567</td>
          <td>0.310546</td>
          <td>26.052085</td>
          <td>0.168923</td>
          <td>25.615800</td>
          <td>0.214990</td>
          <td>25.247908</td>
          <td>0.338994</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.366782</td>
          <td>0.775316</td>
          <td>25.936121</td>
          <td>0.097771</td>
          <td>25.948964</td>
          <td>0.088686</td>
          <td>25.554232</td>
          <td>0.102649</td>
          <td>25.521746</td>
          <td>0.186387</td>
          <td>24.791321</td>
          <td>0.219540</td>
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
          <td>27.324311</td>
          <td>0.762877</td>
          <td>26.258176</td>
          <td>0.131805</td>
          <td>25.452314</td>
          <td>0.058380</td>
          <td>25.153504</td>
          <td>0.073715</td>
          <td>24.769720</td>
          <td>0.099465</td>
          <td>24.858646</td>
          <td>0.236851</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.908286</td>
          <td>0.225463</td>
          <td>26.368459</td>
          <td>0.128415</td>
          <td>25.285562</td>
          <td>0.081391</td>
          <td>24.824661</td>
          <td>0.102630</td>
          <td>24.338352</td>
          <td>0.150223</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.068175</td>
          <td>0.638058</td>
          <td>27.222260</td>
          <td>0.293640</td>
          <td>26.272861</td>
          <td>0.119204</td>
          <td>26.494370</td>
          <td>0.232146</td>
          <td>25.604741</td>
          <td>0.202274</td>
          <td>27.487363</td>
          <td>1.425619</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.588353</td>
          <td>0.205488</td>
          <td>26.328994</td>
          <td>0.141407</td>
          <td>26.200261</td>
          <td>0.113992</td>
          <td>26.009009</td>
          <td>0.157098</td>
          <td>25.986659</td>
          <td>0.282103</td>
          <td>24.720982</td>
          <td>0.213404</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.572767</td>
          <td>0.444522</td>
          <td>26.561431</td>
          <td>0.169294</td>
          <td>26.266900</td>
          <td>0.118266</td>
          <td>26.518094</td>
          <td>0.236116</td>
          <td>26.171790</td>
          <td>0.321022</td>
          <td>25.763738</td>
          <td>0.478704</td>
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
          <td>26.533846</td>
          <td>0.387081</td>
          <td>26.871688</td>
          <td>0.190172</td>
          <td>26.042573</td>
          <td>0.081876</td>
          <td>25.137903</td>
          <td>0.060008</td>
          <td>24.743348</td>
          <td>0.080966</td>
          <td>24.023517</td>
          <td>0.096496</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.346812</td>
          <td>0.700000</td>
          <td>28.194687</td>
          <td>0.541755</td>
          <td>26.460686</td>
          <td>0.118210</td>
          <td>26.598168</td>
          <td>0.213311</td>
          <td>25.993582</td>
          <td>0.237295</td>
          <td>26.095723</td>
          <td>0.527423</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.017600</td>
          <td>0.989181</td>
          <td>31.220925</td>
          <td>2.604876</td>
          <td>26.071572</td>
          <td>0.148103</td>
          <td>24.970516</td>
          <td>0.107252</td>
          <td>24.485927</td>
          <td>0.156678</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.173568</td>
          <td>0.268069</td>
          <td>26.136818</td>
          <td>0.180877</td>
          <td>25.122508</td>
          <td>0.140967</td>
          <td>25.053168</td>
          <td>0.289169</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.186324</td>
          <td>0.294421</td>
          <td>26.136093</td>
          <td>0.101080</td>
          <td>25.957819</td>
          <td>0.076072</td>
          <td>25.672395</td>
          <td>0.096342</td>
          <td>25.429377</td>
          <td>0.147468</td>
          <td>24.994314</td>
          <td>0.222174</td>
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
          <td>27.066350</td>
          <td>0.602055</td>
          <td>26.317139</td>
          <td>0.126561</td>
          <td>25.464203</td>
          <td>0.053110</td>
          <td>25.111723</td>
          <td>0.063714</td>
          <td>24.979454</td>
          <td>0.107756</td>
          <td>24.386790</td>
          <td>0.143456</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.745343</td>
          <td>0.173245</td>
          <td>26.131798</td>
          <td>0.090034</td>
          <td>25.213263</td>
          <td>0.065277</td>
          <td>24.964353</td>
          <td>0.099954</td>
          <td>24.125801</td>
          <td>0.107337</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.019815</td>
          <td>0.572474</td>
          <td>26.606384</td>
          <td>0.158202</td>
          <td>26.374564</td>
          <td>0.115013</td>
          <td>26.211175</td>
          <td>0.161427</td>
          <td>25.983873</td>
          <td>0.246228</td>
          <td>25.636508</td>
          <td>0.389648</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.919057</td>
          <td>0.553697</td>
          <td>26.220897</td>
          <td>0.120160</td>
          <td>26.258895</td>
          <td>0.110996</td>
          <td>25.734849</td>
          <td>0.114468</td>
          <td>25.541789</td>
          <td>0.181157</td>
          <td>25.456761</td>
          <td>0.359771</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.087838</td>
          <td>0.278564</td>
          <td>26.958848</td>
          <td>0.211317</td>
          <td>26.473235</td>
          <td>0.124079</td>
          <td>26.083743</td>
          <td>0.143249</td>
          <td>25.974267</td>
          <td>0.242010</td>
          <td>25.364217</td>
          <td>0.311585</td>
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
