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

    <pzflow.flow.Flow at 0x7f473ed46290>



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
    0      23.994413  0.111575  0.059301  
    1      25.391064  0.043030  0.030527  
    2      24.304707  0.052204  0.047393  
    3      25.291103  0.116312  0.076747  
    4      25.096743  0.141420  0.135661  
    ...          ...       ...       ...  
    99995  24.737946  0.193390  0.159251  
    99996  24.224169  0.002281  0.001269  
    99997  25.613836  0.047248  0.026694  
    99998  25.274899  0.090507  0.066425  
    99999  25.699642  0.020561  0.011274  
    
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
          <td>26.076752</td>
          <td>0.269192</td>
          <td>26.832253</td>
          <td>0.183925</td>
          <td>25.942554</td>
          <td>0.074945</td>
          <td>25.102198</td>
          <td>0.058128</td>
          <td>24.464830</td>
          <td>0.063278</td>
          <td>24.123913</td>
          <td>0.105350</td>
          <td>0.111575</td>
          <td>0.059301</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.674422</td>
          <td>0.431078</td>
          <td>27.052612</td>
          <td>0.221262</td>
          <td>26.500081</td>
          <td>0.122213</td>
          <td>26.469372</td>
          <td>0.191275</td>
          <td>25.721940</td>
          <td>0.188946</td>
          <td>25.315197</td>
          <td>0.288680</td>
          <td>0.043030</td>
          <td>0.030527</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.203507</td>
          <td>1.052617</td>
          <td>28.821833</td>
          <td>0.757383</td>
          <td>26.267000</td>
          <td>0.161084</td>
          <td>24.890686</td>
          <td>0.092169</td>
          <td>24.331572</td>
          <td>0.126230</td>
          <td>0.052204</td>
          <td>0.047393</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.972151</td>
          <td>0.914930</td>
          <td>27.639703</td>
          <td>0.317573</td>
          <td>25.964440</td>
          <td>0.124130</td>
          <td>25.591684</td>
          <td>0.169195</td>
          <td>26.097042</td>
          <td>0.527500</td>
          <td>0.116312</td>
          <td>0.076747</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.503131</td>
          <td>0.377946</td>
          <td>26.278756</td>
          <td>0.114339</td>
          <td>25.891336</td>
          <td>0.071626</td>
          <td>25.490732</td>
          <td>0.081989</td>
          <td>25.764698</td>
          <td>0.195879</td>
          <td>25.060056</td>
          <td>0.234302</td>
          <td>0.141420</td>
          <td>0.135661</td>
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
          <td>27.860279</td>
          <td>0.973612</td>
          <td>26.568829</td>
          <td>0.146939</td>
          <td>25.460367</td>
          <td>0.048872</td>
          <td>25.187917</td>
          <td>0.062721</td>
          <td>24.970684</td>
          <td>0.098873</td>
          <td>24.736972</td>
          <td>0.178730</td>
          <td>0.193390</td>
          <td>0.159251</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.912250</td>
          <td>1.004669</td>
          <td>26.814166</td>
          <td>0.181133</td>
          <td>26.162233</td>
          <td>0.090962</td>
          <td>25.152119</td>
          <td>0.060761</td>
          <td>24.943626</td>
          <td>0.096555</td>
          <td>24.376062</td>
          <td>0.131188</td>
          <td>0.002281</td>
          <td>0.001269</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.311412</td>
          <td>0.682987</td>
          <td>26.883864</td>
          <td>0.192112</td>
          <td>26.294820</td>
          <td>0.102183</td>
          <td>25.897830</td>
          <td>0.117149</td>
          <td>25.891060</td>
          <td>0.217745</td>
          <td>26.770247</td>
          <td>0.837495</td>
          <td>0.047248</td>
          <td>0.026694</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.354872</td>
          <td>0.336494</td>
          <td>26.358804</td>
          <td>0.122575</td>
          <td>26.209641</td>
          <td>0.094831</td>
          <td>26.075157</td>
          <td>0.136614</td>
          <td>25.913557</td>
          <td>0.221863</td>
          <td>25.220048</td>
          <td>0.267219</td>
          <td>0.090507</td>
          <td>0.066425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.999416</td>
          <td>0.211666</td>
          <td>26.482776</td>
          <td>0.120389</td>
          <td>26.159518</td>
          <td>0.146911</td>
          <td>26.398638</td>
          <td>0.329261</td>
          <td>25.382781</td>
          <td>0.304825</td>
          <td>0.020561</td>
          <td>0.011274</td>
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
          <td>27.328056</td>
          <td>0.767094</td>
          <td>26.805553</td>
          <td>0.210929</td>
          <td>26.094323</td>
          <td>0.103388</td>
          <td>25.434460</td>
          <td>0.094927</td>
          <td>24.879527</td>
          <td>0.110056</td>
          <td>23.873201</td>
          <td>0.102637</td>
          <td>0.111575</td>
          <td>0.059301</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.493250</td>
          <td>0.361780</td>
          <td>26.720497</td>
          <td>0.173790</td>
          <td>26.177904</td>
          <td>0.176614</td>
          <td>25.815627</td>
          <td>0.239248</td>
          <td>25.374406</td>
          <td>0.353623</td>
          <td>0.043030</td>
          <td>0.030527</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.673475</td>
          <td>0.946537</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.901307</td>
          <td>0.452056</td>
          <td>25.875124</td>
          <td>0.136799</td>
          <td>25.011665</td>
          <td>0.121306</td>
          <td>24.212451</td>
          <td>0.135366</td>
          <td>0.052204</td>
          <td>0.047393</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.519172</td>
          <td>0.433054</td>
          <td>27.690934</td>
          <td>0.430544</td>
          <td>27.637967</td>
          <td>0.377149</td>
          <td>26.235985</td>
          <td>0.190585</td>
          <td>25.977263</td>
          <td>0.280066</td>
          <td>25.161983</td>
          <td>0.306364</td>
          <td>0.116312</td>
          <td>0.076747</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.485253</td>
          <td>0.866387</td>
          <td>26.020606</td>
          <td>0.111216</td>
          <td>26.067697</td>
          <td>0.104603</td>
          <td>25.432729</td>
          <td>0.098277</td>
          <td>25.477110</td>
          <td>0.190317</td>
          <td>25.030092</td>
          <td>0.283219</td>
          <td>0.141420</td>
          <td>0.135661</td>
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
          <td>27.416587</td>
          <td>0.846030</td>
          <td>26.523974</td>
          <td>0.176927</td>
          <td>25.486391</td>
          <td>0.064926</td>
          <td>25.051730</td>
          <td>0.072852</td>
          <td>24.657155</td>
          <td>0.097148</td>
          <td>24.563894</td>
          <td>0.199316</td>
          <td>0.193390</td>
          <td>0.159251</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.390407</td>
          <td>1.412823</td>
          <td>26.852229</td>
          <td>0.214419</td>
          <td>26.048468</td>
          <td>0.096752</td>
          <td>25.152988</td>
          <td>0.072083</td>
          <td>24.753141</td>
          <td>0.095995</td>
          <td>24.140400</td>
          <td>0.126111</td>
          <td>0.002281</td>
          <td>0.001269</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.443670</td>
          <td>0.401543</td>
          <td>27.279627</td>
          <td>0.305515</td>
          <td>26.432607</td>
          <td>0.135843</td>
          <td>26.097322</td>
          <td>0.164959</td>
          <td>25.483361</td>
          <td>0.181239</td>
          <td>26.947087</td>
          <td>1.053355</td>
          <td>0.047248</td>
          <td>0.026694</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.464973</td>
          <td>0.412533</td>
          <td>25.987392</td>
          <td>0.104171</td>
          <td>26.210030</td>
          <td>0.113778</td>
          <td>25.749975</td>
          <td>0.124344</td>
          <td>25.665756</td>
          <td>0.214528</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.090507</td>
          <td>0.066425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.521145</td>
          <td>0.424912</td>
          <td>27.192836</td>
          <td>0.283932</td>
          <td>26.403441</td>
          <td>0.131936</td>
          <td>26.399343</td>
          <td>0.212020</td>
          <td>26.569555</td>
          <td>0.433990</td>
          <td>25.317880</td>
          <td>0.337034</td>
          <td>0.020561</td>
          <td>0.011274</td>
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
          <td>26.581307</td>
          <td>0.161168</td>
          <td>25.942825</td>
          <td>0.082500</td>
          <td>25.125015</td>
          <td>0.065593</td>
          <td>24.679855</td>
          <td>0.084226</td>
          <td>24.089403</td>
          <td>0.112747</td>
          <td>0.111575</td>
          <td>0.059301</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.898290</td>
          <td>0.440573</td>
          <td>26.792325</td>
          <td>0.160099</td>
          <td>26.609511</td>
          <td>0.219139</td>
          <td>26.173208</td>
          <td>0.279436</td>
          <td>24.795340</td>
          <td>0.191278</td>
          <td>0.043030</td>
          <td>0.030527</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.250203</td>
          <td>1.835143</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.893181</td>
          <td>0.120698</td>
          <td>25.190635</td>
          <td>0.123729</td>
          <td>24.617657</td>
          <td>0.166876</td>
          <td>0.052204</td>
          <td>0.047393</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.688964</td>
          <td>0.404340</td>
          <td>27.205775</td>
          <td>0.248354</td>
          <td>26.402236</td>
          <td>0.202747</td>
          <td>25.316009</td>
          <td>0.149455</td>
          <td>26.185197</td>
          <td>0.618877</td>
          <td>0.116312</td>
          <td>0.076747</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.159981</td>
          <td>0.331087</td>
          <td>26.248323</td>
          <td>0.133718</td>
          <td>25.764861</td>
          <td>0.079000</td>
          <td>25.700090</td>
          <td>0.122243</td>
          <td>25.390016</td>
          <td>0.174319</td>
          <td>24.786040</td>
          <td>0.228635</td>
          <td>0.141420</td>
          <td>0.135661</td>
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
          <td>26.604999</td>
          <td>0.497222</td>
          <td>26.334380</td>
          <td>0.156773</td>
          <td>25.518029</td>
          <td>0.069883</td>
          <td>25.109080</td>
          <td>0.080295</td>
          <td>24.734648</td>
          <td>0.108740</td>
          <td>24.866962</td>
          <td>0.267542</td>
          <td>0.193390</td>
          <td>0.159251</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.182431</td>
          <td>0.624728</td>
          <td>27.122642</td>
          <td>0.234503</td>
          <td>26.043047</td>
          <td>0.081903</td>
          <td>25.229995</td>
          <td>0.065108</td>
          <td>24.828846</td>
          <td>0.087295</td>
          <td>24.240187</td>
          <td>0.116602</td>
          <td>0.002281</td>
          <td>0.001269</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.557262</td>
          <td>0.398843</td>
          <td>26.584224</td>
          <td>0.151382</td>
          <td>26.256746</td>
          <td>0.100774</td>
          <td>26.290511</td>
          <td>0.167646</td>
          <td>25.950258</td>
          <td>0.232972</td>
          <td>26.065032</td>
          <td>0.524058</td>
          <td>0.047248</td>
          <td>0.026694</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.536182</td>
          <td>0.407120</td>
          <td>26.129679</td>
          <td>0.107495</td>
          <td>26.035036</td>
          <td>0.087974</td>
          <td>25.820273</td>
          <td>0.118762</td>
          <td>25.944728</td>
          <td>0.245067</td>
          <td>25.046942</td>
          <td>0.250115</td>
          <td>0.090507</td>
          <td>0.066425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.614075</td>
          <td>0.412623</td>
          <td>26.585281</td>
          <td>0.149500</td>
          <td>26.495537</td>
          <td>0.122180</td>
          <td>26.383316</td>
          <td>0.178521</td>
          <td>26.217093</td>
          <td>0.285632</td>
          <td>27.255695</td>
          <td>1.127734</td>
          <td>0.020561</td>
          <td>0.011274</td>
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
