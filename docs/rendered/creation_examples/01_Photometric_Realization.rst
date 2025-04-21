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

    <pzflow.flow.Flow at 0x7f15ff5ddf30>



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
          <td>28.172099</td>
          <td>1.168750</td>
          <td>26.623925</td>
          <td>0.154048</td>
          <td>26.039153</td>
          <td>0.081618</td>
          <td>25.263939</td>
          <td>0.067093</td>
          <td>24.716840</td>
          <td>0.079084</td>
          <td>24.019733</td>
          <td>0.096163</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.487042</td>
          <td>0.315381</td>
          <td>26.801099</td>
          <td>0.158424</td>
          <td>26.073505</td>
          <td>0.136419</td>
          <td>25.539099</td>
          <td>0.161777</td>
          <td>25.200190</td>
          <td>0.262922</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.090216</td>
          <td>2.532270</td>
          <td>27.837079</td>
          <td>0.371086</td>
          <td>25.776350</td>
          <td>0.105369</td>
          <td>25.017175</td>
          <td>0.102982</td>
          <td>24.287684</td>
          <td>0.121513</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.717003</td>
          <td>0.378037</td>
          <td>27.422611</td>
          <td>0.266530</td>
          <td>26.283424</td>
          <td>0.163359</td>
          <td>25.470250</td>
          <td>0.152521</td>
          <td>24.991197</td>
          <td>0.221289</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.854501</td>
          <td>0.224256</td>
          <td>26.062952</td>
          <td>0.094692</td>
          <td>25.882125</td>
          <td>0.071045</td>
          <td>25.567235</td>
          <td>0.087706</td>
          <td>25.608634</td>
          <td>0.171652</td>
          <td>25.051994</td>
          <td>0.232744</td>
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
          <td>26.862498</td>
          <td>0.496276</td>
          <td>26.432134</td>
          <td>0.130612</td>
          <td>25.435126</td>
          <td>0.047789</td>
          <td>25.104135</td>
          <td>0.058228</td>
          <td>24.922954</td>
          <td>0.094819</td>
          <td>24.774426</td>
          <td>0.184489</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.375256</td>
          <td>1.306809</td>
          <td>26.669791</td>
          <td>0.160209</td>
          <td>26.098610</td>
          <td>0.086010</td>
          <td>25.394895</td>
          <td>0.075336</td>
          <td>24.892091</td>
          <td>0.092283</td>
          <td>24.256379</td>
          <td>0.118251</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.962261</td>
          <td>0.245128</td>
          <td>27.088734</td>
          <td>0.228001</td>
          <td>26.247954</td>
          <td>0.098072</td>
          <td>26.469862</td>
          <td>0.191354</td>
          <td>25.549654</td>
          <td>0.163241</td>
          <td>25.015687</td>
          <td>0.225841</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.620364</td>
          <td>0.413683</td>
          <td>26.113562</td>
          <td>0.098985</td>
          <td>26.052991</td>
          <td>0.082620</td>
          <td>25.921817</td>
          <td>0.119619</td>
          <td>25.691210</td>
          <td>0.184104</td>
          <td>24.934751</td>
          <td>0.211113</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.708892</td>
          <td>0.886586</td>
          <td>26.893295</td>
          <td>0.193644</td>
          <td>26.345802</td>
          <td>0.106843</td>
          <td>26.677859</td>
          <td>0.227728</td>
          <td>26.024247</td>
          <td>0.243162</td>
          <td>25.264311</td>
          <td>0.277021</td>
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
          <td>32.167223</td>
          <td>4.856872</td>
          <td>26.643643</td>
          <td>0.179944</td>
          <td>25.946551</td>
          <td>0.088469</td>
          <td>25.132126</td>
          <td>0.070767</td>
          <td>24.528890</td>
          <td>0.078808</td>
          <td>24.128334</td>
          <td>0.124802</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.044671</td>
          <td>0.546399</td>
          <td>26.646219</td>
          <td>0.162431</td>
          <td>25.861175</td>
          <td>0.134043</td>
          <td>25.767484</td>
          <td>0.228938</td>
          <td>25.382731</td>
          <td>0.354481</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>31.032934</td>
          <td>3.763959</td>
          <td>28.447465</td>
          <td>0.734918</td>
          <td>28.211810</td>
          <td>0.574472</td>
          <td>25.797563</td>
          <td>0.129757</td>
          <td>25.175410</td>
          <td>0.141690</td>
          <td>24.722076</td>
          <td>0.211707</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.382946</td>
          <td>0.349006</td>
          <td>27.369651</td>
          <td>0.315068</td>
          <td>25.850232</td>
          <td>0.142114</td>
          <td>25.507596</td>
          <td>0.196360</td>
          <td>25.227886</td>
          <td>0.333665</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.425178</td>
          <td>0.394649</td>
          <td>26.054878</td>
          <td>0.108460</td>
          <td>25.855102</td>
          <td>0.081651</td>
          <td>25.693599</td>
          <td>0.115926</td>
          <td>25.748805</td>
          <td>0.225442</td>
          <td>25.312238</td>
          <td>0.335353</td>
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
          <td>26.933399</td>
          <td>0.583240</td>
          <td>26.224802</td>
          <td>0.128056</td>
          <td>25.380194</td>
          <td>0.054763</td>
          <td>25.052597</td>
          <td>0.067420</td>
          <td>24.879335</td>
          <td>0.109469</td>
          <td>25.463648</td>
          <td>0.384804</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.574970</td>
          <td>0.170379</td>
          <td>25.966315</td>
          <td>0.090397</td>
          <td>25.120772</td>
          <td>0.070366</td>
          <td>24.942050</td>
          <td>0.113706</td>
          <td>24.254959</td>
          <td>0.139828</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.718244</td>
          <td>0.975067</td>
          <td>26.983053</td>
          <td>0.241611</td>
          <td>26.304394</td>
          <td>0.122514</td>
          <td>26.444018</td>
          <td>0.222646</td>
          <td>26.387272</td>
          <td>0.381255</td>
          <td>25.217194</td>
          <td>0.314560</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.259905</td>
          <td>0.354441</td>
          <td>26.261912</td>
          <td>0.133464</td>
          <td>25.976188</td>
          <td>0.093706</td>
          <td>25.642995</td>
          <td>0.114524</td>
          <td>25.660515</td>
          <td>0.215724</td>
          <td>25.679027</td>
          <td>0.457786</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.478508</td>
          <td>0.838242</td>
          <td>26.931860</td>
          <td>0.231065</td>
          <td>26.448708</td>
          <td>0.138433</td>
          <td>26.265332</td>
          <td>0.191184</td>
          <td>26.007096</td>
          <td>0.281221</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>27.223432</td>
          <td>0.642871</td>
          <td>26.844356</td>
          <td>0.185836</td>
          <td>25.904995</td>
          <td>0.072507</td>
          <td>25.134032</td>
          <td>0.059802</td>
          <td>24.725076</td>
          <td>0.079672</td>
          <td>23.964526</td>
          <td>0.091625</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>31.645411</td>
          <td>4.212654</td>
          <td>27.269577</td>
          <td>0.264789</td>
          <td>26.629353</td>
          <td>0.136814</td>
          <td>26.424615</td>
          <td>0.184357</td>
          <td>26.230675</td>
          <td>0.288052</td>
          <td>25.171675</td>
          <td>0.257094</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.626698</td>
          <td>0.772745</td>
          <td>27.775094</td>
          <td>0.380175</td>
          <td>25.755312</td>
          <td>0.112634</td>
          <td>24.927035</td>
          <td>0.103252</td>
          <td>24.442329</td>
          <td>0.150933</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.853734</td>
          <td>0.565946</td>
          <td>28.079859</td>
          <td>0.587253</td>
          <td>27.514928</td>
          <td>0.352372</td>
          <td>26.405235</td>
          <td>0.226545</td>
          <td>25.642315</td>
          <td>0.219052</td>
          <td>25.034736</td>
          <td>0.284892</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.243303</td>
          <td>0.308196</td>
          <td>26.124754</td>
          <td>0.100082</td>
          <td>26.046384</td>
          <td>0.082258</td>
          <td>25.518607</td>
          <td>0.084155</td>
          <td>25.783536</td>
          <td>0.199276</td>
          <td>24.949886</td>
          <td>0.214099</td>
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
          <td>26.750194</td>
          <td>0.478665</td>
          <td>26.577284</td>
          <td>0.158309</td>
          <td>25.427476</td>
          <td>0.051407</td>
          <td>25.114931</td>
          <td>0.063896</td>
          <td>25.089632</td>
          <td>0.118615</td>
          <td>24.824322</td>
          <td>0.208027</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.644609</td>
          <td>0.858310</td>
          <td>26.766735</td>
          <td>0.176419</td>
          <td>26.095041</td>
          <td>0.087169</td>
          <td>25.049031</td>
          <td>0.056428</td>
          <td>24.733856</td>
          <td>0.081620</td>
          <td>24.359898</td>
          <td>0.131565</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.246629</td>
          <td>0.671117</td>
          <td>26.775154</td>
          <td>0.182613</td>
          <td>26.285812</td>
          <td>0.106444</td>
          <td>26.399156</td>
          <td>0.189363</td>
          <td>25.914898</td>
          <td>0.232598</td>
          <td>25.412150</td>
          <td>0.326778</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.426083</td>
          <td>0.382813</td>
          <td>26.178810</td>
          <td>0.115846</td>
          <td>26.019948</td>
          <td>0.090036</td>
          <td>25.822168</td>
          <td>0.123497</td>
          <td>25.641366</td>
          <td>0.197037</td>
          <td>25.116190</td>
          <td>0.274085</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.353515</td>
          <td>0.344469</td>
          <td>26.873878</td>
          <td>0.196794</td>
          <td>26.761342</td>
          <td>0.159039</td>
          <td>25.956220</td>
          <td>0.128312</td>
          <td>26.065289</td>
          <td>0.260795</td>
          <td>25.907912</td>
          <td>0.474625</td>
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
