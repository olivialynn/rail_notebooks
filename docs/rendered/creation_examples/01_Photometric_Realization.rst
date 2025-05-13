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

    <pzflow.flow.Flow at 0x7f228ae01330>



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
          <td>28.121578</td>
          <td>1.135724</td>
          <td>26.874329</td>
          <td>0.190575</td>
          <td>26.142333</td>
          <td>0.089384</td>
          <td>25.197980</td>
          <td>0.063283</td>
          <td>24.492904</td>
          <td>0.064872</td>
          <td>24.142562</td>
          <td>0.107082</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.687880</td>
          <td>0.369562</td>
          <td>26.820115</td>
          <td>0.161020</td>
          <td>26.574013</td>
          <td>0.208849</td>
          <td>25.673968</td>
          <td>0.181437</td>
          <td>26.074881</td>
          <td>0.519029</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.535309</td>
          <td>0.387488</td>
          <td>28.341081</td>
          <td>0.601214</td>
          <td>28.020890</td>
          <td>0.427541</td>
          <td>25.998719</td>
          <td>0.127875</td>
          <td>25.032814</td>
          <td>0.104400</td>
          <td>24.328869</td>
          <td>0.125934</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.371073</td>
          <td>0.614062</td>
          <td>27.179000</td>
          <td>0.218009</td>
          <td>25.998340</td>
          <td>0.127833</td>
          <td>25.649470</td>
          <td>0.177709</td>
          <td>25.335469</td>
          <td>0.293443</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.601100</td>
          <td>0.407625</td>
          <td>26.019154</td>
          <td>0.091122</td>
          <td>25.902073</td>
          <td>0.072310</td>
          <td>25.679687</td>
          <td>0.096816</td>
          <td>25.554569</td>
          <td>0.163927</td>
          <td>25.272581</td>
          <td>0.278888</td>
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
          <td>27.822529</td>
          <td>0.951429</td>
          <td>26.241710</td>
          <td>0.110709</td>
          <td>25.477131</td>
          <td>0.049605</td>
          <td>25.028169</td>
          <td>0.054431</td>
          <td>24.858205</td>
          <td>0.089575</td>
          <td>24.520009</td>
          <td>0.148520</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.747956</td>
          <td>0.455692</td>
          <td>27.051041</td>
          <td>0.220973</td>
          <td>26.048722</td>
          <td>0.082310</td>
          <td>25.040449</td>
          <td>0.055028</td>
          <td>24.889649</td>
          <td>0.092086</td>
          <td>24.318269</td>
          <td>0.124782</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.349183</td>
          <td>1.288632</td>
          <td>26.604338</td>
          <td>0.151485</td>
          <td>26.405931</td>
          <td>0.112600</td>
          <td>26.192902</td>
          <td>0.151183</td>
          <td>25.581711</td>
          <td>0.167764</td>
          <td>26.768328</td>
          <td>0.836464</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.290514</td>
          <td>0.673287</td>
          <td>26.073858</td>
          <td>0.095601</td>
          <td>26.083771</td>
          <td>0.084893</td>
          <td>25.971057</td>
          <td>0.124845</td>
          <td>25.847363</td>
          <td>0.209944</td>
          <td>25.285321</td>
          <td>0.281784</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.508974</td>
          <td>0.779644</td>
          <td>26.973625</td>
          <td>0.207150</td>
          <td>26.602275</td>
          <td>0.133527</td>
          <td>26.233935</td>
          <td>0.156593</td>
          <td>25.838092</td>
          <td>0.208322</td>
          <td>25.264230</td>
          <td>0.277003</td>
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
          <td>26.241163</td>
          <td>0.341820</td>
          <td>27.045954</td>
          <td>0.251706</td>
          <td>26.026230</td>
          <td>0.094885</td>
          <td>25.071314</td>
          <td>0.067059</td>
          <td>24.721161</td>
          <td>0.093341</td>
          <td>23.965408</td>
          <td>0.108306</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.336262</td>
          <td>2.171604</td>
          <td>27.892422</td>
          <td>0.488754</td>
          <td>26.258890</td>
          <td>0.116306</td>
          <td>26.003658</td>
          <td>0.151534</td>
          <td>25.438663</td>
          <td>0.173700</td>
          <td>25.084456</td>
          <td>0.279348</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.152807</td>
          <td>1.258538</td>
          <td>27.869896</td>
          <td>0.488824</td>
          <td>27.989260</td>
          <td>0.488512</td>
          <td>25.883503</td>
          <td>0.139754</td>
          <td>25.297605</td>
          <td>0.157357</td>
          <td>24.291074</td>
          <td>0.146896</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.126410</td>
          <td>1.151138</td>
          <td>26.964757</td>
          <td>0.226506</td>
          <td>26.416417</td>
          <td>0.229452</td>
          <td>25.414936</td>
          <td>0.181593</td>
          <td>26.413046</td>
          <td>0.790536</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.263152</td>
          <td>0.347867</td>
          <td>26.268640</td>
          <td>0.130578</td>
          <td>26.071583</td>
          <td>0.098766</td>
          <td>25.845002</td>
          <td>0.132199</td>
          <td>25.554187</td>
          <td>0.191559</td>
          <td>24.770404</td>
          <td>0.215746</td>
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
          <td>27.168680</td>
          <td>0.687173</td>
          <td>26.304573</td>
          <td>0.137188</td>
          <td>25.362120</td>
          <td>0.053892</td>
          <td>25.067999</td>
          <td>0.068346</td>
          <td>24.792888</td>
          <td>0.101504</td>
          <td>24.652418</td>
          <td>0.199453</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.841677</td>
          <td>1.045241</td>
          <td>26.774314</td>
          <td>0.201616</td>
          <td>26.106880</td>
          <td>0.102258</td>
          <td>25.145071</td>
          <td>0.071894</td>
          <td>24.772685</td>
          <td>0.098064</td>
          <td>24.244042</td>
          <td>0.138518</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.841704</td>
          <td>0.543290</td>
          <td>27.001826</td>
          <td>0.245375</td>
          <td>26.238401</td>
          <td>0.115684</td>
          <td>26.342775</td>
          <td>0.204600</td>
          <td>25.799029</td>
          <td>0.237792</td>
          <td>25.002039</td>
          <td>0.264370</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.491038</td>
          <td>0.423772</td>
          <td>26.148363</td>
          <td>0.120973</td>
          <td>26.397939</td>
          <td>0.135312</td>
          <td>25.723852</td>
          <td>0.122864</td>
          <td>25.795924</td>
          <td>0.241367</td>
          <td>24.710062</td>
          <td>0.211466</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.282442</td>
          <td>0.355518</td>
          <td>26.726417</td>
          <td>0.194649</td>
          <td>26.697972</td>
          <td>0.171379</td>
          <td>26.369085</td>
          <td>0.208592</td>
          <td>25.844693</td>
          <td>0.246281</td>
          <td>26.760674</td>
          <td>0.944875</td>
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
          <td>27.614880</td>
          <td>0.835212</td>
          <td>26.674507</td>
          <td>0.160873</td>
          <td>26.104832</td>
          <td>0.086494</td>
          <td>25.212201</td>
          <td>0.064095</td>
          <td>24.491345</td>
          <td>0.064791</td>
          <td>23.976407</td>
          <td>0.092587</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.052327</td>
          <td>1.652006</td>
          <td>26.717914</td>
          <td>0.147657</td>
          <td>26.122020</td>
          <td>0.142386</td>
          <td>25.913503</td>
          <td>0.222051</td>
          <td>25.811428</td>
          <td>0.426686</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.028124</td>
          <td>0.461203</td>
          <td>26.068216</td>
          <td>0.147677</td>
          <td>25.109963</td>
          <td>0.121104</td>
          <td>24.202352</td>
          <td>0.122696</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.975107</td>
          <td>0.544739</td>
          <td>28.319079</td>
          <td>0.640474</td>
          <td>26.278700</td>
          <td>0.203848</td>
          <td>25.289277</td>
          <td>0.162635</td>
          <td>25.981898</td>
          <td>0.587207</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.361202</td>
          <td>0.338480</td>
          <td>25.974900</td>
          <td>0.087756</td>
          <td>25.875994</td>
          <td>0.070762</td>
          <td>25.747098</td>
          <td>0.102860</td>
          <td>25.652237</td>
          <td>0.178371</td>
          <td>25.150634</td>
          <td>0.252812</td>
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
          <td>26.716224</td>
          <td>0.466692</td>
          <td>26.208279</td>
          <td>0.115151</td>
          <td>25.364029</td>
          <td>0.048592</td>
          <td>25.071398</td>
          <td>0.061477</td>
          <td>24.865733</td>
          <td>0.097549</td>
          <td>24.782936</td>
          <td>0.200934</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.872487</td>
          <td>0.504780</td>
          <td>26.872508</td>
          <td>0.192914</td>
          <td>26.111491</td>
          <td>0.088440</td>
          <td>25.215961</td>
          <td>0.065433</td>
          <td>24.743779</td>
          <td>0.082337</td>
          <td>24.338929</td>
          <td>0.129199</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.676518</td>
          <td>0.444764</td>
          <td>26.440308</td>
          <td>0.137181</td>
          <td>26.453956</td>
          <td>0.123233</td>
          <td>26.183767</td>
          <td>0.157689</td>
          <td>26.038022</td>
          <td>0.257423</td>
          <td>25.882277</td>
          <td>0.469738</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.393575</td>
          <td>0.373273</td>
          <td>26.247227</td>
          <td>0.122936</td>
          <td>26.060256</td>
          <td>0.093282</td>
          <td>25.769493</td>
          <td>0.117972</td>
          <td>26.152897</td>
          <td>0.300247</td>
          <td>25.205906</td>
          <td>0.294731</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.510835</td>
          <td>0.389459</td>
          <td>27.027549</td>
          <td>0.223765</td>
          <td>26.478456</td>
          <td>0.124642</td>
          <td>26.537366</td>
          <td>0.210562</td>
          <td>25.928598</td>
          <td>0.233048</td>
          <td>25.871016</td>
          <td>0.461708</td>
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
