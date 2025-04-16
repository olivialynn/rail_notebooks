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

    <pzflow.flow.Flow at 0x7feb4d9c1210>



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
          <td>27.469639</td>
          <td>0.759684</td>
          <td>27.080200</td>
          <td>0.226392</td>
          <td>26.123064</td>
          <td>0.087881</td>
          <td>25.091608</td>
          <td>0.057584</td>
          <td>24.683631</td>
          <td>0.076799</td>
          <td>23.910739</td>
          <td>0.087380</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.886453</td>
          <td>1.688098</td>
          <td>27.976966</td>
          <td>0.461071</td>
          <td>26.595523</td>
          <td>0.132750</td>
          <td>26.080172</td>
          <td>0.137206</td>
          <td>26.159488</td>
          <td>0.271652</td>
          <td>26.037778</td>
          <td>0.505084</td>
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
          <td>28.144289</td>
          <td>0.469247</td>
          <td>26.172136</td>
          <td>0.148512</td>
          <td>25.054597</td>
          <td>0.106407</td>
          <td>24.098088</td>
          <td>0.102997</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.049135</td>
          <td>0.568446</td>
          <td>28.838663</td>
          <td>0.840957</td>
          <td>27.045482</td>
          <td>0.194940</td>
          <td>26.307674</td>
          <td>0.166773</td>
          <td>25.477349</td>
          <td>0.153452</td>
          <td>25.120312</td>
          <td>0.246249</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.479997</td>
          <td>0.371207</td>
          <td>26.141469</td>
          <td>0.101432</td>
          <td>25.936557</td>
          <td>0.074549</td>
          <td>25.647833</td>
          <td>0.094146</td>
          <td>25.383194</td>
          <td>0.141527</td>
          <td>24.906190</td>
          <td>0.206127</td>
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
          <td>28.634700</td>
          <td>1.494640</td>
          <td>26.426239</td>
          <td>0.129947</td>
          <td>25.482942</td>
          <td>0.049861</td>
          <td>25.252720</td>
          <td>0.066429</td>
          <td>24.796735</td>
          <td>0.084857</td>
          <td>24.626150</td>
          <td>0.162650</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.536631</td>
          <td>0.142929</td>
          <td>26.080399</td>
          <td>0.084641</td>
          <td>25.152076</td>
          <td>0.060758</td>
          <td>24.609166</td>
          <td>0.071906</td>
          <td>24.581013</td>
          <td>0.156494</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.296624</td>
          <td>1.252394</td>
          <td>26.506901</td>
          <td>0.139317</td>
          <td>26.342735</td>
          <td>0.106557</td>
          <td>26.608978</td>
          <td>0.215041</td>
          <td>25.702264</td>
          <td>0.185832</td>
          <td>25.425938</td>
          <td>0.315541</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.238816</td>
          <td>0.306816</td>
          <td>26.265497</td>
          <td>0.113027</td>
          <td>26.067909</td>
          <td>0.083714</td>
          <td>25.884078</td>
          <td>0.115755</td>
          <td>25.648101</td>
          <td>0.177503</td>
          <td>25.068030</td>
          <td>0.235853</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.619706</td>
          <td>0.413475</td>
          <td>27.221091</td>
          <td>0.254297</td>
          <td>26.548030</td>
          <td>0.127403</td>
          <td>26.104011</td>
          <td>0.140057</td>
          <td>25.708761</td>
          <td>0.186855</td>
          <td>26.154586</td>
          <td>0.549999</td>
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
          <td>35.020561</td>
          <td>7.698648</td>
          <td>26.755397</td>
          <td>0.197728</td>
          <td>25.878152</td>
          <td>0.083299</td>
          <td>25.138226</td>
          <td>0.071150</td>
          <td>24.717534</td>
          <td>0.093044</td>
          <td>24.057894</td>
          <td>0.117396</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.244600</td>
          <td>0.714672</td>
          <td>27.218989</td>
          <td>0.289830</td>
          <td>26.440294</td>
          <td>0.136110</td>
          <td>26.321759</td>
          <td>0.198539</td>
          <td>25.635916</td>
          <td>0.205156</td>
          <td>25.570741</td>
          <td>0.410164</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.801275</td>
          <td>1.741743</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.686265</td>
          <td>0.117813</td>
          <td>24.999114</td>
          <td>0.121655</td>
          <td>24.284323</td>
          <td>0.146047</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.968657</td>
          <td>1.161423</td>
          <td>27.187681</td>
          <td>0.298792</td>
          <td>29.033321</td>
          <td>1.019503</td>
          <td>26.479768</td>
          <td>0.241792</td>
          <td>25.433484</td>
          <td>0.184465</td>
          <td>25.237418</td>
          <td>0.336193</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.851179</td>
          <td>0.249783</td>
          <td>26.166016</td>
          <td>0.119471</td>
          <td>25.975132</td>
          <td>0.090750</td>
          <td>25.553206</td>
          <td>0.102557</td>
          <td>25.645338</td>
          <td>0.206805</td>
          <td>24.727996</td>
          <td>0.208237</td>
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
          <td>26.170701</td>
          <td>0.122194</td>
          <td>25.670240</td>
          <td>0.070811</td>
          <td>25.110419</td>
          <td>0.070960</td>
          <td>24.650081</td>
          <td>0.089552</td>
          <td>24.649108</td>
          <td>0.198899</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.710662</td>
          <td>0.966176</td>
          <td>26.627336</td>
          <td>0.178121</td>
          <td>26.117129</td>
          <td>0.103179</td>
          <td>25.280301</td>
          <td>0.081014</td>
          <td>24.797301</td>
          <td>0.100202</td>
          <td>24.214289</td>
          <td>0.135008</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.880013</td>
          <td>0.558516</td>
          <td>26.549184</td>
          <td>0.167944</td>
          <td>26.154008</td>
          <td>0.107477</td>
          <td>26.804929</td>
          <td>0.299158</td>
          <td>26.061402</td>
          <td>0.294582</td>
          <td>25.616916</td>
          <td>0.429675</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.282629</td>
          <td>0.360805</td>
          <td>26.417065</td>
          <td>0.152512</td>
          <td>26.283331</td>
          <td>0.122531</td>
          <td>25.960830</td>
          <td>0.150747</td>
          <td>25.715856</td>
          <td>0.225892</td>
          <td>25.524714</td>
          <td>0.407164</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.592252</td>
          <td>0.173786</td>
          <td>26.487003</td>
          <td>0.143076</td>
          <td>26.330731</td>
          <td>0.201995</td>
          <td>25.858351</td>
          <td>0.249064</td>
          <td>25.473464</td>
          <td>0.383914</td>
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
          <td>26.662744</td>
          <td>0.427303</td>
          <td>26.789681</td>
          <td>0.177434</td>
          <td>26.036822</td>
          <td>0.081461</td>
          <td>25.153673</td>
          <td>0.060853</td>
          <td>24.762164</td>
          <td>0.082321</td>
          <td>23.907749</td>
          <td>0.087162</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.018130</td>
          <td>0.215165</td>
          <td>26.571457</td>
          <td>0.130136</td>
          <td>26.430756</td>
          <td>0.185317</td>
          <td>26.021946</td>
          <td>0.242916</td>
          <td>25.569766</td>
          <td>0.353933</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.081428</td>
          <td>1.152913</td>
          <td>28.884723</td>
          <td>0.911722</td>
          <td>28.102162</td>
          <td>0.487390</td>
          <td>26.058145</td>
          <td>0.146404</td>
          <td>25.156549</td>
          <td>0.126100</td>
          <td>24.607148</td>
          <td>0.173738</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.992740</td>
          <td>0.624510</td>
          <td>27.978286</td>
          <td>0.545993</td>
          <td>27.449160</td>
          <td>0.334554</td>
          <td>26.198205</td>
          <td>0.190507</td>
          <td>25.845274</td>
          <td>0.259019</td>
          <td>24.953063</td>
          <td>0.266600</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.483681</td>
          <td>0.372598</td>
          <td>26.203916</td>
          <td>0.107251</td>
          <td>26.049531</td>
          <td>0.082486</td>
          <td>25.585152</td>
          <td>0.089233</td>
          <td>25.429604</td>
          <td>0.147497</td>
          <td>24.865164</td>
          <td>0.199435</td>
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
          <td>26.369676</td>
          <td>0.357903</td>
          <td>26.281215</td>
          <td>0.122681</td>
          <td>25.508822</td>
          <td>0.055256</td>
          <td>25.128188</td>
          <td>0.064651</td>
          <td>24.847546</td>
          <td>0.096006</td>
          <td>24.607589</td>
          <td>0.173272</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.954389</td>
          <td>0.535922</td>
          <td>26.812779</td>
          <td>0.183433</td>
          <td>25.867761</td>
          <td>0.071323</td>
          <td>25.159242</td>
          <td>0.062224</td>
          <td>24.691538</td>
          <td>0.078628</td>
          <td>24.197576</td>
          <td>0.114273</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.597369</td>
          <td>0.418837</td>
          <td>26.633877</td>
          <td>0.161960</td>
          <td>26.488071</td>
          <td>0.126933</td>
          <td>26.314437</td>
          <td>0.176262</td>
          <td>26.177863</td>
          <td>0.288446</td>
          <td>25.474630</td>
          <td>0.343354</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.121887</td>
          <td>0.301130</td>
          <td>26.144540</td>
          <td>0.112442</td>
          <td>26.133610</td>
          <td>0.099481</td>
          <td>25.822185</td>
          <td>0.123498</td>
          <td>25.819611</td>
          <td>0.228669</td>
          <td>25.812589</td>
          <td>0.472402</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.143143</td>
          <td>0.621104</td>
          <td>26.978728</td>
          <td>0.214853</td>
          <td>26.477165</td>
          <td>0.124503</td>
          <td>26.492815</td>
          <td>0.202850</td>
          <td>25.828893</td>
          <td>0.214511</td>
          <td>25.559123</td>
          <td>0.363541</td>
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
