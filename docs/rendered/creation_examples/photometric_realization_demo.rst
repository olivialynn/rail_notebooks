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

    <pzflow.flow.Flow at 0x7f3334540340>



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
          <td>26.933439</td>
          <td>0.522813</td>
          <td>26.774402</td>
          <td>0.175131</td>
          <td>26.165179</td>
          <td>0.091198</td>
          <td>25.084562</td>
          <td>0.057225</td>
          <td>24.649472</td>
          <td>0.074515</td>
          <td>23.926886</td>
          <td>0.088630</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.899978</td>
          <td>0.435062</td>
          <td>26.589705</td>
          <td>0.132084</td>
          <td>26.201239</td>
          <td>0.152268</td>
          <td>26.159758</td>
          <td>0.271712</td>
          <td>24.973932</td>
          <td>0.218131</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.486858</td>
          <td>1.236970</td>
          <td>28.285017</td>
          <td>0.520713</td>
          <td>26.172041</td>
          <td>0.148500</td>
          <td>25.055600</td>
          <td>0.106501</td>
          <td>24.433158</td>
          <td>0.137821</td>
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
          <td>27.601565</td>
          <td>0.308034</td>
          <td>26.256721</td>
          <td>0.159675</td>
          <td>25.657103</td>
          <td>0.178863</td>
          <td>25.405796</td>
          <td>0.310500</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.788725</td>
          <td>0.469821</td>
          <td>26.080230</td>
          <td>0.096137</td>
          <td>25.931621</td>
          <td>0.074224</td>
          <td>25.686951</td>
          <td>0.097435</td>
          <td>25.255513</td>
          <td>0.126743</td>
          <td>24.948373</td>
          <td>0.213529</td>
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
          <td>26.719519</td>
          <td>0.167149</td>
          <td>25.463338</td>
          <td>0.049001</td>
          <td>25.016622</td>
          <td>0.053876</td>
          <td>24.863933</td>
          <td>0.090027</td>
          <td>24.856319</td>
          <td>0.197678</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.583676</td>
          <td>0.402210</td>
          <td>26.906651</td>
          <td>0.195832</td>
          <td>26.047711</td>
          <td>0.082237</td>
          <td>25.229946</td>
          <td>0.065102</td>
          <td>24.913791</td>
          <td>0.094059</td>
          <td>24.180662</td>
          <td>0.110703</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.288049</td>
          <td>0.319118</td>
          <td>26.789055</td>
          <td>0.177321</td>
          <td>26.324708</td>
          <td>0.104891</td>
          <td>26.070065</td>
          <td>0.136015</td>
          <td>25.772750</td>
          <td>0.197210</td>
          <td>25.760128</td>
          <td>0.409936</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.976326</td>
          <td>0.247977</td>
          <td>26.187238</td>
          <td>0.105571</td>
          <td>26.051485</td>
          <td>0.082511</td>
          <td>25.895284</td>
          <td>0.116889</td>
          <td>25.659679</td>
          <td>0.179254</td>
          <td>25.751656</td>
          <td>0.407281</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.254542</td>
          <td>0.310700</td>
          <td>27.059539</td>
          <td>0.222540</td>
          <td>26.495372</td>
          <td>0.121714</td>
          <td>26.679147</td>
          <td>0.227971</td>
          <td>25.755725</td>
          <td>0.194405</td>
          <td>25.954877</td>
          <td>0.474992</td>
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
          <td>27.048352</td>
          <td>0.624446</td>
          <td>26.511525</td>
          <td>0.160828</td>
          <td>26.247020</td>
          <td>0.115087</td>
          <td>25.190516</td>
          <td>0.074516</td>
          <td>24.630394</td>
          <td>0.086183</td>
          <td>23.957157</td>
          <td>0.107529</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.873890</td>
          <td>0.482078</td>
          <td>26.704564</td>
          <td>0.170710</td>
          <td>26.118439</td>
          <td>0.167155</td>
          <td>26.386655</td>
          <td>0.376863</td>
          <td>25.878706</td>
          <td>0.516720</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.689063</td>
          <td>0.860403</td>
          <td>28.718336</td>
          <td>0.811656</td>
          <td>26.025838</td>
          <td>0.157920</td>
          <td>24.938667</td>
          <td>0.115427</td>
          <td>24.388977</td>
          <td>0.159751</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.391003</td>
          <td>0.729954</td>
          <td>27.155537</td>
          <td>0.265032</td>
          <td>26.356026</td>
          <td>0.218220</td>
          <td>25.617273</td>
          <td>0.215254</td>
          <td>25.499643</td>
          <td>0.412373</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.186232</td>
          <td>0.327360</td>
          <td>26.011461</td>
          <td>0.104428</td>
          <td>25.988202</td>
          <td>0.091798</td>
          <td>25.575991</td>
          <td>0.104622</td>
          <td>25.417424</td>
          <td>0.170612</td>
          <td>25.063596</td>
          <td>0.274687</td>
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
          <td>32.858723</td>
          <td>5.560619</td>
          <td>26.252738</td>
          <td>0.131187</td>
          <td>25.556228</td>
          <td>0.064013</td>
          <td>25.243978</td>
          <td>0.079845</td>
          <td>25.016305</td>
          <td>0.123325</td>
          <td>24.606362</td>
          <td>0.191872</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.839038</td>
          <td>0.539314</td>
          <td>26.567278</td>
          <td>0.169268</td>
          <td>25.986928</td>
          <td>0.092050</td>
          <td>25.165834</td>
          <td>0.073226</td>
          <td>24.984440</td>
          <td>0.117979</td>
          <td>24.003096</td>
          <td>0.112403</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.120418</td>
          <td>1.230780</td>
          <td>26.430235</td>
          <td>0.151725</td>
          <td>26.271891</td>
          <td>0.119104</td>
          <td>26.237524</td>
          <td>0.187261</td>
          <td>26.210829</td>
          <td>0.331962</td>
          <td>25.333896</td>
          <td>0.345088</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.878230</td>
          <td>0.261101</td>
          <td>26.314653</td>
          <td>0.139673</td>
          <td>26.236739</td>
          <td>0.117669</td>
          <td>26.052471</td>
          <td>0.163042</td>
          <td>25.353997</td>
          <td>0.166585</td>
          <td>25.619948</td>
          <td>0.437833</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.703058</td>
          <td>0.489993</td>
          <td>26.810912</td>
          <td>0.208942</td>
          <td>26.641921</td>
          <td>0.163390</td>
          <td>26.279090</td>
          <td>0.193413</td>
          <td>26.155075</td>
          <td>0.316772</td>
          <td>25.918522</td>
          <td>0.536427</td>
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
          <td>28.049188</td>
          <td>1.089402</td>
          <td>26.875123</td>
          <td>0.190723</td>
          <td>26.091682</td>
          <td>0.085498</td>
          <td>25.174013</td>
          <td>0.061961</td>
          <td>24.698744</td>
          <td>0.077841</td>
          <td>23.928742</td>
          <td>0.088787</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.806056</td>
          <td>2.468644</td>
          <td>28.582971</td>
          <td>0.711113</td>
          <td>26.732227</td>
          <td>0.149483</td>
          <td>26.159385</td>
          <td>0.147037</td>
          <td>26.524532</td>
          <td>0.363911</td>
          <td>25.579951</td>
          <td>0.356773</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.040457</td>
          <td>0.592398</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.201908</td>
          <td>0.524516</td>
          <td>26.023330</td>
          <td>0.142084</td>
          <td>25.110803</td>
          <td>0.121193</td>
          <td>24.372471</td>
          <td>0.142138</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.126878</td>
          <td>2.039910</td>
          <td>27.932304</td>
          <td>0.528064</td>
          <td>26.836822</td>
          <td>0.202881</td>
          <td>26.379045</td>
          <td>0.221667</td>
          <td>25.565355</td>
          <td>0.205413</td>
          <td>24.907310</td>
          <td>0.256813</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.004208</td>
          <td>0.253945</td>
          <td>26.059185</td>
          <td>0.094496</td>
          <td>25.998491</td>
          <td>0.078854</td>
          <td>25.618591</td>
          <td>0.091896</td>
          <td>25.957940</td>
          <td>0.230501</td>
          <td>25.000282</td>
          <td>0.223280</td>
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
          <td>27.434085</td>
          <td>0.773838</td>
          <td>26.382474</td>
          <td>0.133915</td>
          <td>25.348755</td>
          <td>0.047937</td>
          <td>25.120551</td>
          <td>0.064215</td>
          <td>24.680214</td>
          <td>0.082869</td>
          <td>24.696131</td>
          <td>0.186770</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.640715</td>
          <td>1.508920</td>
          <td>26.767216</td>
          <td>0.176491</td>
          <td>26.009306</td>
          <td>0.080825</td>
          <td>25.183826</td>
          <td>0.063596</td>
          <td>24.905624</td>
          <td>0.094937</td>
          <td>24.195204</td>
          <td>0.114037</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.278870</td>
          <td>0.326832</td>
          <td>26.650784</td>
          <td>0.164312</td>
          <td>26.401750</td>
          <td>0.117767</td>
          <td>26.091695</td>
          <td>0.145716</td>
          <td>26.177865</td>
          <td>0.288447</td>
          <td>27.602315</td>
          <td>1.402253</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.233355</td>
          <td>0.329132</td>
          <td>26.117002</td>
          <td>0.109777</td>
          <td>25.950735</td>
          <td>0.084715</td>
          <td>25.898886</td>
          <td>0.131983</td>
          <td>25.799731</td>
          <td>0.224926</td>
          <td>25.404812</td>
          <td>0.345379</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.093659</td>
          <td>0.599843</td>
          <td>26.937383</td>
          <td>0.207559</td>
          <td>26.569657</td>
          <td>0.134882</td>
          <td>26.167543</td>
          <td>0.153939</td>
          <td>26.180781</td>
          <td>0.286482</td>
          <td>25.540128</td>
          <td>0.358173</td>
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
