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

    <pzflow.flow.Flow at 0x7f84c7d7a980>



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
    0      23.994413  0.108925  0.065360  
    1      25.391064  0.193298  0.131914  
    2      24.304707  0.103521  0.084225  
    3      25.291103  0.012434  0.006387  
    4      25.096743  0.013698  0.010571  
    ...          ...       ...       ...  
    99995  24.737946  0.048813  0.037818  
    99996  24.224169  0.146624  0.080407  
    99997  25.613836  0.064109  0.061100  
    99998  25.274899  0.032825  0.024223  
    99999  25.699642  0.160826  0.154993  
    
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
          <td>27.598832</td>
          <td>0.826579</td>
          <td>26.690286</td>
          <td>0.163036</td>
          <td>26.198748</td>
          <td>0.093928</td>
          <td>25.195334</td>
          <td>0.063135</td>
          <td>24.731864</td>
          <td>0.080140</td>
          <td>23.991419</td>
          <td>0.093803</td>
          <td>0.108925</td>
          <td>0.065360</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.787358</td>
          <td>1.610716</td>
          <td>27.029077</td>
          <td>0.216969</td>
          <td>26.693682</td>
          <td>0.144479</td>
          <td>26.487777</td>
          <td>0.194265</td>
          <td>25.977632</td>
          <td>0.233978</td>
          <td>25.208513</td>
          <td>0.264715</td>
          <td>0.193298</td>
          <td>0.131914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.386383</td>
          <td>1.314606</td>
          <td>28.820641</td>
          <td>0.831281</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.066544</td>
          <td>0.135602</td>
          <td>25.076426</td>
          <td>0.108456</td>
          <td>24.311774</td>
          <td>0.124081</td>
          <td>0.103521</td>
          <td>0.084225</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.164302</td>
          <td>1.028453</td>
          <td>27.504150</td>
          <td>0.284790</td>
          <td>26.163079</td>
          <td>0.147361</td>
          <td>25.891821</td>
          <td>0.217883</td>
          <td>25.116706</td>
          <td>0.245519</td>
          <td>0.012434</td>
          <td>0.006387</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.364242</td>
          <td>0.338994</td>
          <td>26.168539</td>
          <td>0.103861</td>
          <td>25.807340</td>
          <td>0.066493</td>
          <td>25.708280</td>
          <td>0.099274</td>
          <td>25.671852</td>
          <td>0.181112</td>
          <td>25.024128</td>
          <td>0.227429</td>
          <td>0.013698</td>
          <td>0.010571</td>
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
          <td>26.811681</td>
          <td>0.477930</td>
          <td>26.409331</td>
          <td>0.128060</td>
          <td>25.416266</td>
          <td>0.046995</td>
          <td>25.084678</td>
          <td>0.057231</td>
          <td>24.727205</td>
          <td>0.079811</td>
          <td>24.966894</td>
          <td>0.216855</td>
          <td>0.048813</td>
          <td>0.037818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.615836</td>
          <td>0.412252</td>
          <td>26.567900</td>
          <td>0.146822</td>
          <td>25.968556</td>
          <td>0.076687</td>
          <td>25.238496</td>
          <td>0.065597</td>
          <td>24.687296</td>
          <td>0.077048</td>
          <td>24.202566</td>
          <td>0.112838</td>
          <td>0.146624</td>
          <td>0.080407</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.603261</td>
          <td>0.828941</td>
          <td>26.455953</td>
          <td>0.133328</td>
          <td>26.464611</td>
          <td>0.118503</td>
          <td>26.234598</td>
          <td>0.156682</td>
          <td>25.722368</td>
          <td>0.189015</td>
          <td>26.069165</td>
          <td>0.516861</td>
          <td>0.064109</td>
          <td>0.061100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.509797</td>
          <td>0.379906</td>
          <td>26.251609</td>
          <td>0.111668</td>
          <td>26.083032</td>
          <td>0.084837</td>
          <td>25.871376</td>
          <td>0.114481</td>
          <td>25.447207</td>
          <td>0.149536</td>
          <td>24.826195</td>
          <td>0.192729</td>
          <td>0.032825</td>
          <td>0.024223</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.471606</td>
          <td>0.368788</td>
          <td>26.694870</td>
          <td>0.163675</td>
          <td>26.665560</td>
          <td>0.141022</td>
          <td>26.492093</td>
          <td>0.194972</td>
          <td>26.465360</td>
          <td>0.347102</td>
          <td>26.582475</td>
          <td>0.740614</td>
          <td>0.160826</td>
          <td>0.154993</td>
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
          <td>27.069712</td>
          <td>0.644103</td>
          <td>26.444244</td>
          <td>0.155434</td>
          <td>25.980942</td>
          <td>0.093641</td>
          <td>25.156215</td>
          <td>0.074320</td>
          <td>24.579360</td>
          <td>0.084620</td>
          <td>23.930583</td>
          <td>0.107952</td>
          <td>0.108925</td>
          <td>0.065360</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.663095</td>
          <td>0.439358</td>
          <td>26.541316</td>
          <td>0.161178</td>
          <td>26.386801</td>
          <td>0.227665</td>
          <td>25.717409</td>
          <td>0.237752</td>
          <td>25.160131</td>
          <td>0.321260</td>
          <td>0.193298</td>
          <td>0.131914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.852847</td>
          <td>0.485290</td>
          <td>29.118102</td>
          <td>1.045166</td>
          <td>25.970614</td>
          <td>0.151703</td>
          <td>25.010073</td>
          <td>0.123678</td>
          <td>24.639332</td>
          <td>0.198899</td>
          <td>0.103521</td>
          <td>0.084225</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.915087</td>
          <td>0.976376</td>
          <td>26.873191</td>
          <td>0.196899</td>
          <td>26.743674</td>
          <td>0.281362</td>
          <td>25.833580</td>
          <td>0.241821</td>
          <td>25.140517</td>
          <td>0.292335</td>
          <td>0.012434</td>
          <td>0.006387</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.342334</td>
          <td>0.370147</td>
          <td>26.076979</td>
          <td>0.110585</td>
          <td>25.890579</td>
          <td>0.084257</td>
          <td>25.800963</td>
          <td>0.127274</td>
          <td>25.405982</td>
          <td>0.168984</td>
          <td>24.971939</td>
          <td>0.254920</td>
          <td>0.013698</td>
          <td>0.010571</td>
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
          <td>26.205591</td>
          <td>0.124317</td>
          <td>25.360474</td>
          <td>0.053024</td>
          <td>25.114145</td>
          <td>0.070119</td>
          <td>24.787002</td>
          <td>0.099525</td>
          <td>24.541270</td>
          <td>0.179006</td>
          <td>0.048813</td>
          <td>0.037818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.968556</td>
          <td>3.720840</td>
          <td>26.913693</td>
          <td>0.234489</td>
          <td>25.911666</td>
          <td>0.089740</td>
          <td>25.106790</td>
          <td>0.072510</td>
          <td>24.784471</td>
          <td>0.103180</td>
          <td>24.503192</td>
          <td>0.180039</td>
          <td>0.146624</td>
          <td>0.080407</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.647837</td>
          <td>0.934096</td>
          <td>26.562351</td>
          <td>0.169889</td>
          <td>26.326876</td>
          <td>0.124973</td>
          <td>26.287243</td>
          <td>0.195348</td>
          <td>25.768506</td>
          <td>0.231944</td>
          <td>25.641971</td>
          <td>0.438065</td>
          <td>0.064109</td>
          <td>0.061100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.053381</td>
          <td>0.294896</td>
          <td>26.356837</td>
          <td>0.141206</td>
          <td>25.954306</td>
          <td>0.089325</td>
          <td>25.716409</td>
          <td>0.118550</td>
          <td>25.592516</td>
          <td>0.198308</td>
          <td>25.115521</td>
          <td>0.287169</td>
          <td>0.032825</td>
          <td>0.024223</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.147527</td>
          <td>2.068544</td>
          <td>26.670036</td>
          <td>0.196987</td>
          <td>26.806494</td>
          <td>0.200570</td>
          <td>26.485907</td>
          <td>0.245633</td>
          <td>25.786179</td>
          <td>0.250162</td>
          <td>25.231783</td>
          <td>0.338133</td>
          <td>0.160826</td>
          <td>0.154993</td>
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
          <td>26.510793</td>
          <td>0.151946</td>
          <td>25.795014</td>
          <td>0.072528</td>
          <td>25.112482</td>
          <td>0.064985</td>
          <td>24.736537</td>
          <td>0.088683</td>
          <td>23.837871</td>
          <td>0.090622</td>
          <td>0.108925</td>
          <td>0.065360</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.151826</td>
          <td>0.300425</td>
          <td>26.636717</td>
          <td>0.178768</td>
          <td>26.686860</td>
          <td>0.297447</td>
          <td>25.744328</td>
          <td>0.248416</td>
          <td>25.327526</td>
          <td>0.374351</td>
          <td>0.193298</td>
          <td>0.131914</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.710222</td>
          <td>0.369760</td>
          <td>25.987726</td>
          <td>0.141461</td>
          <td>25.071694</td>
          <td>0.120153</td>
          <td>24.408187</td>
          <td>0.150427</td>
          <td>0.103521</td>
          <td>0.084225</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.552117</td>
          <td>0.696578</td>
          <td>27.434438</td>
          <td>0.269444</td>
          <td>26.506756</td>
          <td>0.197655</td>
          <td>25.538895</td>
          <td>0.161957</td>
          <td>25.235054</td>
          <td>0.270853</td>
          <td>0.012434</td>
          <td>0.006387</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.073815</td>
          <td>0.268900</td>
          <td>26.234053</td>
          <td>0.110165</td>
          <td>26.026784</td>
          <td>0.080898</td>
          <td>25.651084</td>
          <td>0.094617</td>
          <td>25.464294</td>
          <td>0.152045</td>
          <td>25.708912</td>
          <td>0.394829</td>
          <td>0.013698</td>
          <td>0.010571</td>
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
          <td>27.396177</td>
          <td>0.733125</td>
          <td>26.295997</td>
          <td>0.118616</td>
          <td>25.514353</td>
          <td>0.052598</td>
          <td>25.094197</td>
          <td>0.059282</td>
          <td>24.664949</td>
          <td>0.077481</td>
          <td>25.150665</td>
          <td>0.258708</td>
          <td>0.048813</td>
          <td>0.037818</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.433251</td>
          <td>0.395496</td>
          <td>26.663034</td>
          <td>0.181872</td>
          <td>25.985547</td>
          <td>0.090898</td>
          <td>25.152031</td>
          <td>0.071470</td>
          <td>24.979316</td>
          <td>0.116141</td>
          <td>24.425957</td>
          <td>0.160064</td>
          <td>0.146624</td>
          <td>0.080407</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.257251</td>
          <td>0.321580</td>
          <td>26.638342</td>
          <td>0.162797</td>
          <td>26.397358</td>
          <td>0.117507</td>
          <td>26.110997</td>
          <td>0.148401</td>
          <td>26.039550</td>
          <td>0.258135</td>
          <td>25.328005</td>
          <td>0.306018</td>
          <td>0.064109</td>
          <td>0.061100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.943555</td>
          <td>0.530084</td>
          <td>26.108790</td>
          <td>0.099531</td>
          <td>26.056037</td>
          <td>0.083777</td>
          <td>25.961997</td>
          <td>0.125310</td>
          <td>25.535889</td>
          <td>0.163092</td>
          <td>25.684390</td>
          <td>0.390666</td>
          <td>0.032825</td>
          <td>0.024223</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.949030</td>
          <td>1.914215</td>
          <td>26.538260</td>
          <td>0.178829</td>
          <td>26.152284</td>
          <td>0.116403</td>
          <td>26.305839</td>
          <td>0.214975</td>
          <td>25.823865</td>
          <td>0.261983</td>
          <td>25.891812</td>
          <td>0.565013</td>
          <td>0.160826</td>
          <td>0.154993</td>
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
