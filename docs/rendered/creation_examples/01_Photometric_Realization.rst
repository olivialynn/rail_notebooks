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

    <pzflow.flow.Flow at 0x7fe25a4bdb10>



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
    0      23.994413  0.129765  0.119171  
    1      25.391064  0.066020  0.057333  
    2      24.304707  0.066344  0.062461  
    3      25.291103  0.088926  0.061481  
    4      25.096743  0.000579  0.000462  
    ...          ...       ...       ...  
    99995  24.737946  0.063326  0.037664  
    99996  24.224169  0.124540  0.073121  
    99997  25.613836  0.015311  0.008687  
    99998  25.274899  0.274106  0.209156  
    99999  25.699642  0.116895  0.099593  
    
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
          <td>26.494707</td>
          <td>0.137861</td>
          <td>25.997183</td>
          <td>0.078651</td>
          <td>25.173688</td>
          <td>0.061934</td>
          <td>24.683310</td>
          <td>0.076777</td>
          <td>24.026498</td>
          <td>0.096736</td>
          <td>0.129765</td>
          <td>0.119171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.483254</td>
          <td>0.372150</td>
          <td>26.968402</td>
          <td>0.206247</td>
          <td>26.402223</td>
          <td>0.112236</td>
          <td>26.337404</td>
          <td>0.171048</td>
          <td>26.158543</td>
          <td>0.271443</td>
          <td>25.350834</td>
          <td>0.297098</td>
          <td>0.066020</td>
          <td>0.057333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.213547</td>
          <td>1.196241</td>
          <td>29.724677</td>
          <td>1.404166</td>
          <td>27.620529</td>
          <td>0.312746</td>
          <td>26.069207</td>
          <td>0.135914</td>
          <td>24.922396</td>
          <td>0.094772</td>
          <td>24.406362</td>
          <td>0.134670</td>
          <td>0.066344</td>
          <td>0.062461</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.233092</td>
          <td>1.071072</td>
          <td>27.649880</td>
          <td>0.320161</td>
          <td>26.264133</td>
          <td>0.160690</td>
          <td>25.479206</td>
          <td>0.153697</td>
          <td>24.707146</td>
          <td>0.174264</td>
          <td>0.088926</td>
          <td>0.061481</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.542303</td>
          <td>0.389589</td>
          <td>26.063154</td>
          <td>0.094709</td>
          <td>25.911819</td>
          <td>0.072936</td>
          <td>25.626540</td>
          <td>0.092402</td>
          <td>25.394125</td>
          <td>0.142866</td>
          <td>25.529877</td>
          <td>0.342687</td>
          <td>0.000579</td>
          <td>0.000462</td>
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
          <td>26.240668</td>
          <td>0.110609</td>
          <td>25.503449</td>
          <td>0.050777</td>
          <td>25.070969</td>
          <td>0.056539</td>
          <td>24.915210</td>
          <td>0.094176</td>
          <td>24.632424</td>
          <td>0.163523</td>
          <td>0.063326</td>
          <td>0.037664</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.774113</td>
          <td>0.464717</td>
          <td>26.653283</td>
          <td>0.157966</td>
          <td>25.861333</td>
          <td>0.069749</td>
          <td>25.093127</td>
          <td>0.057662</td>
          <td>24.997944</td>
          <td>0.101263</td>
          <td>24.279394</td>
          <td>0.120641</td>
          <td>0.124540</td>
          <td>0.073121</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.105077</td>
          <td>0.275460</td>
          <td>26.932778</td>
          <td>0.200178</td>
          <td>26.356555</td>
          <td>0.107851</td>
          <td>26.345549</td>
          <td>0.172237</td>
          <td>26.127441</td>
          <td>0.264646</td>
          <td>25.226530</td>
          <td>0.268635</td>
          <td>0.015311</td>
          <td>0.008687</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.039463</td>
          <td>0.261134</td>
          <td>26.091621</td>
          <td>0.097101</td>
          <td>26.107983</td>
          <td>0.086722</td>
          <td>25.813552</td>
          <td>0.108851</td>
          <td>25.654572</td>
          <td>0.178480</td>
          <td>25.352937</td>
          <td>0.297601</td>
          <td>0.274106</td>
          <td>0.209156</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.730709</td>
          <td>0.898809</td>
          <td>26.967042</td>
          <td>0.206012</td>
          <td>26.599500</td>
          <td>0.133207</td>
          <td>26.122546</td>
          <td>0.142311</td>
          <td>25.965555</td>
          <td>0.231650</td>
          <td>25.565626</td>
          <td>0.352473</td>
          <td>0.116895</td>
          <td>0.099593</td>
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
          <td>26.459318</td>
          <td>0.418885</td>
          <td>26.634555</td>
          <td>0.186546</td>
          <td>26.136292</td>
          <td>0.109832</td>
          <td>25.185683</td>
          <td>0.078173</td>
          <td>24.601327</td>
          <td>0.088335</td>
          <td>23.762860</td>
          <td>0.095491</td>
          <td>0.129765</td>
          <td>0.119171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.853183</td>
          <td>0.216963</td>
          <td>26.601678</td>
          <td>0.158297</td>
          <td>26.179210</td>
          <td>0.178252</td>
          <td>25.572859</td>
          <td>0.196937</td>
          <td>26.110320</td>
          <td>0.616655</td>
          <td>0.066020</td>
          <td>0.057333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>32.274643</td>
          <td>4.975045</td>
          <td>28.988147</td>
          <td>1.028821</td>
          <td>28.657808</td>
          <td>0.775342</td>
          <td>26.300415</td>
          <td>0.197673</td>
          <td>24.960230</td>
          <td>0.116641</td>
          <td>24.372070</td>
          <td>0.156144</td>
          <td>0.066344</td>
          <td>0.062461</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.775162</td>
          <td>1.718931</td>
          <td>28.276521</td>
          <td>0.652901</td>
          <td>27.597113</td>
          <td>0.361325</td>
          <td>26.468529</td>
          <td>0.228707</td>
          <td>25.643039</td>
          <td>0.210195</td>
          <td>24.963714</td>
          <td>0.257848</td>
          <td>0.088926</td>
          <td>0.061481</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.220599</td>
          <td>0.336309</td>
          <td>26.107369</td>
          <td>0.113498</td>
          <td>25.931389</td>
          <td>0.087294</td>
          <td>25.692679</td>
          <td>0.115791</td>
          <td>25.497390</td>
          <td>0.182526</td>
          <td>26.406125</td>
          <td>0.747277</td>
          <td>0.000579</td>
          <td>0.000462</td>
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
          <td>26.294028</td>
          <td>0.358534</td>
          <td>26.192332</td>
          <td>0.123192</td>
          <td>25.402885</td>
          <td>0.055208</td>
          <td>25.090096</td>
          <td>0.068836</td>
          <td>24.821316</td>
          <td>0.102835</td>
          <td>24.903443</td>
          <td>0.242971</td>
          <td>0.063326</td>
          <td>0.037664</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.739100</td>
          <td>0.200861</td>
          <td>26.030373</td>
          <td>0.098520</td>
          <td>25.116856</td>
          <td>0.072337</td>
          <td>24.803682</td>
          <td>0.103801</td>
          <td>24.265886</td>
          <td>0.145448</td>
          <td>0.124540</td>
          <td>0.073121</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.934262</td>
          <td>0.267362</td>
          <td>26.595631</td>
          <td>0.172840</td>
          <td>26.597575</td>
          <td>0.155860</td>
          <td>26.235210</td>
          <td>0.184623</td>
          <td>26.027093</td>
          <td>0.283314</td>
          <td>25.895490</td>
          <td>0.523234</td>
          <td>0.015311</td>
          <td>0.008687</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.784039</td>
          <td>0.267398</td>
          <td>26.362402</td>
          <td>0.164376</td>
          <td>25.950903</td>
          <td>0.104980</td>
          <td>25.782917</td>
          <td>0.148351</td>
          <td>26.311037</td>
          <td>0.411663</td>
          <td>25.097366</td>
          <td>0.329979</td>
          <td>0.274106</td>
          <td>0.209156</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.564536</td>
          <td>0.450148</td>
          <td>27.249702</td>
          <td>0.306658</td>
          <td>26.675137</td>
          <td>0.172783</td>
          <td>26.670805</td>
          <td>0.275062</td>
          <td>25.636979</td>
          <td>0.212962</td>
          <td>26.241047</td>
          <td>0.689215</td>
          <td>0.116895</td>
          <td>0.099593</td>
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
          <td>26.963799</td>
          <td>0.237857</td>
          <td>25.983154</td>
          <td>0.092563</td>
          <td>25.271424</td>
          <td>0.081127</td>
          <td>24.654514</td>
          <td>0.089188</td>
          <td>23.829534</td>
          <td>0.097461</td>
          <td>0.129765</td>
          <td>0.119171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.992244</td>
          <td>1.804163</td>
          <td>28.249327</td>
          <td>0.583210</td>
          <td>26.455379</td>
          <td>0.123431</td>
          <td>26.239972</td>
          <td>0.165508</td>
          <td>26.741892</td>
          <td>0.448597</td>
          <td>24.994438</td>
          <td>0.232865</td>
          <td>0.066020</td>
          <td>0.057333</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.375763</td>
          <td>0.580884</td>
          <td>25.985455</td>
          <td>0.133569</td>
          <td>25.157064</td>
          <td>0.122657</td>
          <td>24.230700</td>
          <td>0.122150</td>
          <td>0.066344</td>
          <td>0.062461</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.474568</td>
          <td>0.297317</td>
          <td>26.145774</td>
          <td>0.156411</td>
          <td>25.438493</td>
          <td>0.159344</td>
          <td>26.102628</td>
          <td>0.563608</td>
          <td>0.088926</td>
          <td>0.061481</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.076919</td>
          <td>0.269229</td>
          <td>26.185943</td>
          <td>0.105452</td>
          <td>25.858654</td>
          <td>0.069584</td>
          <td>25.500584</td>
          <td>0.082705</td>
          <td>25.467115</td>
          <td>0.152113</td>
          <td>25.524754</td>
          <td>0.341305</td>
          <td>0.000579</td>
          <td>0.000462</td>
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
          <td>27.174901</td>
          <td>0.633766</td>
          <td>26.527177</td>
          <td>0.146105</td>
          <td>25.375461</td>
          <td>0.046973</td>
          <td>25.027428</td>
          <td>0.056470</td>
          <td>24.911552</td>
          <td>0.097239</td>
          <td>24.710229</td>
          <td>0.181003</td>
          <td>0.063326</td>
          <td>0.037664</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.050318</td>
          <td>1.154156</td>
          <td>26.794649</td>
          <td>0.197447</td>
          <td>26.050890</td>
          <td>0.093121</td>
          <td>25.249519</td>
          <td>0.075251</td>
          <td>24.861655</td>
          <td>0.101404</td>
          <td>24.071265</td>
          <td>0.113964</td>
          <td>0.124540</td>
          <td>0.073121</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.143605</td>
          <td>1.151138</td>
          <td>26.454748</td>
          <td>0.133428</td>
          <td>26.436079</td>
          <td>0.115838</td>
          <td>26.269216</td>
          <td>0.161736</td>
          <td>25.892254</td>
          <td>0.218396</td>
          <td>26.216631</td>
          <td>0.576096</td>
          <td>0.015311</td>
          <td>0.008687</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.404441</td>
          <td>0.949852</td>
          <td>25.938721</td>
          <td>0.129640</td>
          <td>26.050039</td>
          <td>0.131104</td>
          <td>25.868743</td>
          <td>0.182872</td>
          <td>25.599472</td>
          <td>0.265118</td>
          <td>25.260573</td>
          <td>0.423933</td>
          <td>0.274106</td>
          <td>0.209156</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.958906</td>
          <td>0.577867</td>
          <td>26.477427</td>
          <td>0.153005</td>
          <td>26.309960</td>
          <td>0.118743</td>
          <td>26.180315</td>
          <td>0.172012</td>
          <td>25.668151</td>
          <td>0.206026</td>
          <td>25.886226</td>
          <td>0.509022</td>
          <td>0.116895</td>
          <td>0.099593</td>
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
