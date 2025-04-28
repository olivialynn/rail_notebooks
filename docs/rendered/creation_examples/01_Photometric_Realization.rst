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

    <pzflow.flow.Flow at 0x7fdc947c1240>



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
          <td>26.989294</td>
          <td>0.544476</td>
          <td>26.860082</td>
          <td>0.188299</td>
          <td>25.993112</td>
          <td>0.078369</td>
          <td>25.239510</td>
          <td>0.065656</td>
          <td>24.842791</td>
          <td>0.088368</td>
          <td>24.122666</td>
          <td>0.105236</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.650031</td>
          <td>0.358787</td>
          <td>26.510823</td>
          <td>0.123358</td>
          <td>26.071977</td>
          <td>0.136239</td>
          <td>25.804761</td>
          <td>0.202585</td>
          <td>25.434412</td>
          <td>0.317682</td>
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
          <td>28.041903</td>
          <td>0.434422</td>
          <td>25.711032</td>
          <td>0.099514</td>
          <td>24.967927</td>
          <td>0.098634</td>
          <td>24.390614</td>
          <td>0.132849</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.446684</td>
          <td>1.357276</td>
          <td>28.365937</td>
          <td>0.611847</td>
          <td>27.099500</td>
          <td>0.203991</td>
          <td>26.315091</td>
          <td>0.167830</td>
          <td>25.357614</td>
          <td>0.138441</td>
          <td>25.859076</td>
          <td>0.442019</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.579711</td>
          <td>0.400986</td>
          <td>26.039516</td>
          <td>0.092765</td>
          <td>25.996081</td>
          <td>0.078574</td>
          <td>25.679023</td>
          <td>0.096759</td>
          <td>25.477003</td>
          <td>0.153407</td>
          <td>24.890439</td>
          <td>0.203424</td>
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
          <td>28.191489</td>
          <td>1.181567</td>
          <td>26.265258</td>
          <td>0.113004</td>
          <td>25.406390</td>
          <td>0.046585</td>
          <td>25.119708</td>
          <td>0.059038</td>
          <td>24.733890</td>
          <td>0.080283</td>
          <td>24.560234</td>
          <td>0.153734</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.289122</td>
          <td>0.319391</td>
          <td>26.464565</td>
          <td>0.134323</td>
          <td>26.091372</td>
          <td>0.085463</td>
          <td>25.316023</td>
          <td>0.070260</td>
          <td>24.806205</td>
          <td>0.085568</td>
          <td>24.252070</td>
          <td>0.117809</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.126779</td>
          <td>0.600735</td>
          <td>26.839014</td>
          <td>0.184979</td>
          <td>26.229841</td>
          <td>0.096527</td>
          <td>26.375437</td>
          <td>0.176665</td>
          <td>25.814059</td>
          <td>0.204170</td>
          <td>26.659867</td>
          <td>0.779562</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.158486</td>
          <td>0.287628</td>
          <td>26.179854</td>
          <td>0.104893</td>
          <td>26.092066</td>
          <td>0.085515</td>
          <td>25.787748</td>
          <td>0.106425</td>
          <td>25.324785</td>
          <td>0.134573</td>
          <td>25.900198</td>
          <td>0.455939</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.257916</td>
          <td>1.226058</td>
          <td>26.846507</td>
          <td>0.186153</td>
          <td>26.625191</td>
          <td>0.136197</td>
          <td>26.522105</td>
          <td>0.199955</td>
          <td>25.818105</td>
          <td>0.204864</td>
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
          <td>30.362657</td>
          <td>3.103991</td>
          <td>26.722951</td>
          <td>0.192405</td>
          <td>25.895820</td>
          <td>0.084606</td>
          <td>25.366400</td>
          <td>0.087020</td>
          <td>24.752295</td>
          <td>0.095926</td>
          <td>24.048162</td>
          <td>0.116406</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.018708</td>
          <td>1.902888</td>
          <td>27.210669</td>
          <td>0.287889</td>
          <td>27.005710</td>
          <td>0.219971</td>
          <td>26.340087</td>
          <td>0.201618</td>
          <td>26.120289</td>
          <td>0.305334</td>
          <td>25.219980</td>
          <td>0.311577</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.611597</td>
          <td>0.258043</td>
          <td>24.830314</td>
          <td>0.105017</td>
          <td>24.244202</td>
          <td>0.141092</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.978430</td>
          <td>1.167850</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.752203</td>
          <td>0.189590</td>
          <td>26.343372</td>
          <td>0.215930</td>
          <td>25.876633</td>
          <td>0.266625</td>
          <td>25.185400</td>
          <td>0.322595</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.290902</td>
          <td>0.737280</td>
          <td>26.278125</td>
          <td>0.131653</td>
          <td>26.091729</td>
          <td>0.100525</td>
          <td>25.732930</td>
          <td>0.119960</td>
          <td>25.437940</td>
          <td>0.173613</td>
          <td>24.908810</td>
          <td>0.241989</td>
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
          <td>27.009293</td>
          <td>0.615394</td>
          <td>26.399058</td>
          <td>0.148799</td>
          <td>25.473286</td>
          <td>0.059476</td>
          <td>25.067648</td>
          <td>0.068324</td>
          <td>24.896497</td>
          <td>0.111120</td>
          <td>24.753200</td>
          <td>0.217003</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.816279</td>
          <td>0.530471</td>
          <td>26.455848</td>
          <td>0.153917</td>
          <td>26.004107</td>
          <td>0.093449</td>
          <td>25.077850</td>
          <td>0.067743</td>
          <td>24.877778</td>
          <td>0.107508</td>
          <td>24.321744</td>
          <td>0.148097</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.250428</td>
          <td>1.320586</td>
          <td>27.219873</td>
          <td>0.293076</td>
          <td>26.298986</td>
          <td>0.121940</td>
          <td>25.995078</td>
          <td>0.152348</td>
          <td>25.615647</td>
          <td>0.204133</td>
          <td>26.250029</td>
          <td>0.679370</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.264563</td>
          <td>0.355737</td>
          <td>26.247909</td>
          <td>0.131860</td>
          <td>26.103697</td>
          <td>0.104780</td>
          <td>25.709710</td>
          <td>0.121365</td>
          <td>25.787568</td>
          <td>0.239709</td>
          <td>25.592152</td>
          <td>0.428692</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.800975</td>
          <td>1.732963</td>
          <td>27.382263</td>
          <td>0.332942</td>
          <td>26.502752</td>
          <td>0.145027</td>
          <td>26.456622</td>
          <td>0.224387</td>
          <td>25.973360</td>
          <td>0.273622</td>
          <td>25.900534</td>
          <td>0.529452</td>
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
          <td>26.440954</td>
          <td>0.360093</td>
          <td>26.827295</td>
          <td>0.183176</td>
          <td>26.022149</td>
          <td>0.080414</td>
          <td>25.183046</td>
          <td>0.062459</td>
          <td>24.789145</td>
          <td>0.084302</td>
          <td>24.040273</td>
          <td>0.097925</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.005871</td>
          <td>1.783946</td>
          <td>28.020141</td>
          <td>0.476526</td>
          <td>26.712740</td>
          <td>0.147002</td>
          <td>26.237677</td>
          <td>0.157248</td>
          <td>26.066793</td>
          <td>0.252046</td>
          <td>25.014634</td>
          <td>0.225853</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.350697</td>
          <td>1.958946</td>
          <td>28.605441</td>
          <td>0.697207</td>
          <td>25.727951</td>
          <td>0.109978</td>
          <td>24.988122</td>
          <td>0.108913</td>
          <td>24.361265</td>
          <td>0.140773</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.516371</td>
          <td>1.418493</td>
          <td>27.162493</td>
          <td>0.265658</td>
          <td>26.027540</td>
          <td>0.164838</td>
          <td>25.422858</td>
          <td>0.182187</td>
          <td>24.975578</td>
          <td>0.271536</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.340580</td>
          <td>0.333005</td>
          <td>26.059830</td>
          <td>0.094549</td>
          <td>25.805284</td>
          <td>0.066467</td>
          <td>25.783614</td>
          <td>0.106198</td>
          <td>25.612162</td>
          <td>0.172404</td>
          <td>25.577916</td>
          <td>0.356361</td>
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
          <td>27.085247</td>
          <td>0.610131</td>
          <td>26.287192</td>
          <td>0.123319</td>
          <td>25.428209</td>
          <td>0.051440</td>
          <td>25.189138</td>
          <td>0.068237</td>
          <td>24.821167</td>
          <td>0.093809</td>
          <td>24.714514</td>
          <td>0.189691</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.679088</td>
          <td>0.163748</td>
          <td>26.013219</td>
          <td>0.081104</td>
          <td>25.144765</td>
          <td>0.061430</td>
          <td>24.741887</td>
          <td>0.082200</td>
          <td>24.074048</td>
          <td>0.102588</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.462515</td>
          <td>0.775921</td>
          <td>26.374052</td>
          <td>0.129553</td>
          <td>26.339661</td>
          <td>0.111567</td>
          <td>26.214025</td>
          <td>0.161820</td>
          <td>25.711978</td>
          <td>0.196354</td>
          <td>25.081965</td>
          <td>0.250204</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.903267</td>
          <td>0.252199</td>
          <td>26.306211</td>
          <td>0.129378</td>
          <td>26.233535</td>
          <td>0.108566</td>
          <td>25.892492</td>
          <td>0.131255</td>
          <td>25.489372</td>
          <td>0.173280</td>
          <td>25.693722</td>
          <td>0.431955</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.595424</td>
          <td>0.415614</td>
          <td>26.623704</td>
          <td>0.159191</td>
          <td>26.454289</td>
          <td>0.122055</td>
          <td>26.359480</td>
          <td>0.181286</td>
          <td>26.258304</td>
          <td>0.304944</td>
          <td>25.259021</td>
          <td>0.286301</td>
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
