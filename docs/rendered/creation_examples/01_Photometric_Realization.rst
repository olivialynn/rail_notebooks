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

    <pzflow.flow.Flow at 0x7f704c2a1fc0>



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
    0      23.994413  0.012196  0.012098  
    1      25.391064  0.103104  0.057657  
    2      24.304707  0.058864  0.042653  
    3      25.291103  0.031872  0.018002  
    4      25.096743  0.035816  0.030326  
    ...          ...       ...       ...  
    99995  24.737946  0.040976  0.028929  
    99996  24.224169  0.183975  0.171946  
    99997  25.613836  0.058213  0.038503  
    99998  25.274899  0.252691  0.148108  
    99999  25.699642  0.122101  0.062592  
    
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
          <td>27.572975</td>
          <td>0.812883</td>
          <td>26.589891</td>
          <td>0.149620</td>
          <td>26.061068</td>
          <td>0.083211</td>
          <td>25.071394</td>
          <td>0.056560</td>
          <td>24.782806</td>
          <td>0.083822</td>
          <td>23.892591</td>
          <td>0.085995</td>
          <td>0.012196</td>
          <td>0.012098</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.618646</td>
          <td>1.482663</td>
          <td>27.757317</td>
          <td>0.390036</td>
          <td>26.593886</td>
          <td>0.132562</td>
          <td>26.263996</td>
          <td>0.160671</td>
          <td>25.718328</td>
          <td>0.188371</td>
          <td>24.971475</td>
          <td>0.217684</td>
          <td>0.103104</td>
          <td>0.057657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.471528</td>
          <td>2.173093</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.537187</td>
          <td>0.623766</td>
          <td>26.050254</td>
          <td>0.133707</td>
          <td>25.094942</td>
          <td>0.110223</td>
          <td>24.399283</td>
          <td>0.133849</td>
          <td>0.058864</td>
          <td>0.042653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.501190</td>
          <td>1.396434</td>
          <td>27.460313</td>
          <td>0.308712</td>
          <td>27.244771</td>
          <td>0.230258</td>
          <td>26.446502</td>
          <td>0.187619</td>
          <td>25.394331</td>
          <td>0.142891</td>
          <td>24.887272</td>
          <td>0.202884</td>
          <td>0.031872</td>
          <td>0.018002</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.479424</td>
          <td>0.371041</td>
          <td>26.231570</td>
          <td>0.109735</td>
          <td>26.005908</td>
          <td>0.079259</td>
          <td>25.707758</td>
          <td>0.099228</td>
          <td>25.690311</td>
          <td>0.183964</td>
          <td>25.231375</td>
          <td>0.269698</td>
          <td>0.035816</td>
          <td>0.030326</td>
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
          <td>26.392840</td>
          <td>0.346723</td>
          <td>26.468607</td>
          <td>0.134792</td>
          <td>25.502184</td>
          <td>0.050720</td>
          <td>25.029321</td>
          <td>0.054487</td>
          <td>25.040460</td>
          <td>0.105101</td>
          <td>24.530526</td>
          <td>0.149867</td>
          <td>0.040976</td>
          <td>0.028929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.305900</td>
          <td>0.323682</td>
          <td>26.600681</td>
          <td>0.151010</td>
          <td>25.941443</td>
          <td>0.074872</td>
          <td>25.296552</td>
          <td>0.069059</td>
          <td>24.786129</td>
          <td>0.084068</td>
          <td>24.023235</td>
          <td>0.096459</td>
          <td>0.183975</td>
          <td>0.171946</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.401619</td>
          <td>0.349125</td>
          <td>26.427867</td>
          <td>0.130131</td>
          <td>26.334499</td>
          <td>0.105792</td>
          <td>26.310858</td>
          <td>0.167226</td>
          <td>26.167224</td>
          <td>0.273367</td>
          <td>26.022521</td>
          <td>0.499435</td>
          <td>0.058213</td>
          <td>0.038503</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.484149</td>
          <td>0.372409</td>
          <td>26.260317</td>
          <td>0.112519</td>
          <td>26.043540</td>
          <td>0.081935</td>
          <td>25.939915</td>
          <td>0.121515</td>
          <td>25.504071</td>
          <td>0.157004</td>
          <td>25.692743</td>
          <td>0.389204</td>
          <td>0.252691</td>
          <td>0.148108</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.876610</td>
          <td>0.501469</td>
          <td>26.923209</td>
          <td>0.198576</td>
          <td>26.625812</td>
          <td>0.136270</td>
          <td>26.448024</td>
          <td>0.187860</td>
          <td>26.133157</td>
          <td>0.265884</td>
          <td>25.459100</td>
          <td>0.323994</td>
          <td>0.122101</td>
          <td>0.062592</td>
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
          <td>26.939745</td>
          <td>0.578489</td>
          <td>26.549756</td>
          <td>0.166224</td>
          <td>25.848944</td>
          <td>0.081220</td>
          <td>25.117370</td>
          <td>0.069884</td>
          <td>24.727514</td>
          <td>0.093907</td>
          <td>24.027472</td>
          <td>0.114383</td>
          <td>0.012196</td>
          <td>0.012098</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.148753</td>
          <td>0.279071</td>
          <td>26.492679</td>
          <td>0.145618</td>
          <td>26.301740</td>
          <td>0.199685</td>
          <td>26.180527</td>
          <td>0.327084</td>
          <td>25.056305</td>
          <td>0.279015</td>
          <td>0.103104</td>
          <td>0.057657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.984793</td>
          <td>0.600387</td>
          <td>29.482936</td>
          <td>1.353714</td>
          <td>31.683030</td>
          <td>3.127054</td>
          <td>26.255169</td>
          <td>0.189335</td>
          <td>25.063590</td>
          <td>0.126959</td>
          <td>24.140240</td>
          <td>0.127238</td>
          <td>0.058864</td>
          <td>0.042653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.204315</td>
          <td>0.613224</td>
          <td>27.573534</td>
          <td>0.349296</td>
          <td>26.565523</td>
          <td>0.243689</td>
          <td>26.029950</td>
          <td>0.284427</td>
          <td>25.213746</td>
          <td>0.310619</td>
          <td>0.031872</td>
          <td>0.018002</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.604165</td>
          <td>0.453282</td>
          <td>25.990212</td>
          <td>0.102820</td>
          <td>25.811902</td>
          <td>0.078866</td>
          <td>25.671827</td>
          <td>0.114145</td>
          <td>25.677264</td>
          <td>0.213084</td>
          <td>25.453630</td>
          <td>0.375880</td>
          <td>0.035816</td>
          <td>0.030326</td>
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
          <td>27.653510</td>
          <td>0.932910</td>
          <td>26.299378</td>
          <td>0.134557</td>
          <td>25.441230</td>
          <td>0.056836</td>
          <td>25.199182</td>
          <td>0.075421</td>
          <td>24.999334</td>
          <td>0.119524</td>
          <td>24.483358</td>
          <td>0.170047</td>
          <td>0.040976</td>
          <td>0.028929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.953804</td>
          <td>1.169500</td>
          <td>26.585284</td>
          <td>0.186532</td>
          <td>26.155294</td>
          <td>0.117025</td>
          <td>25.231659</td>
          <td>0.085487</td>
          <td>25.021180</td>
          <td>0.133527</td>
          <td>24.163585</td>
          <td>0.141942</td>
          <td>0.183975</td>
          <td>0.171946</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.949566</td>
          <td>0.585328</td>
          <td>26.746511</td>
          <td>0.197649</td>
          <td>26.397641</td>
          <td>0.132221</td>
          <td>26.058982</td>
          <td>0.160168</td>
          <td>26.056238</td>
          <td>0.292125</td>
          <td>24.981316</td>
          <td>0.258790</td>
          <td>0.058213</td>
          <td>0.038503</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.275806</td>
          <td>0.785518</td>
          <td>26.138872</td>
          <td>0.130520</td>
          <td>26.077928</td>
          <td>0.112374</td>
          <td>25.751168</td>
          <td>0.138238</td>
          <td>25.915053</td>
          <td>0.289992</td>
          <td>24.977216</td>
          <td>0.287944</td>
          <td>0.252691</td>
          <td>0.148108</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.500540</td>
          <td>0.426665</td>
          <td>26.871115</td>
          <td>0.223632</td>
          <td>26.417564</td>
          <td>0.137527</td>
          <td>26.279618</td>
          <td>0.197498</td>
          <td>25.819732</td>
          <td>0.245985</td>
          <td>25.496521</td>
          <td>0.398174</td>
          <td>0.122101</td>
          <td>0.062592</td>
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
          <td>27.041227</td>
          <td>0.565872</td>
          <td>26.751061</td>
          <td>0.171982</td>
          <td>25.890692</td>
          <td>0.071730</td>
          <td>25.186545</td>
          <td>0.062778</td>
          <td>24.769797</td>
          <td>0.083033</td>
          <td>23.931578</td>
          <td>0.089183</td>
          <td>0.012196</td>
          <td>0.012098</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.545433</td>
          <td>0.353225</td>
          <td>26.942598</td>
          <td>0.193880</td>
          <td>26.690820</td>
          <td>0.250147</td>
          <td>25.654677</td>
          <td>0.193631</td>
          <td>25.022662</td>
          <td>0.246686</td>
          <td>0.103104</td>
          <td>0.057657</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.572696</td>
          <td>0.407346</td>
          <td>28.730554</td>
          <td>0.801746</td>
          <td>28.675120</td>
          <td>0.704755</td>
          <td>25.946737</td>
          <td>0.126686</td>
          <td>25.089768</td>
          <td>0.113550</td>
          <td>24.205295</td>
          <td>0.117199</td>
          <td>0.058864</td>
          <td>0.042653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.743057</td>
          <td>0.456450</td>
          <td>27.788857</td>
          <td>0.402386</td>
          <td>27.259017</td>
          <td>0.234975</td>
          <td>26.234362</td>
          <td>0.158099</td>
          <td>25.561048</td>
          <td>0.166280</td>
          <td>25.847727</td>
          <td>0.441780</td>
          <td>0.031872</td>
          <td>0.018002</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.467821</td>
          <td>0.371059</td>
          <td>26.162169</td>
          <td>0.104610</td>
          <td>26.008229</td>
          <td>0.080607</td>
          <td>25.775037</td>
          <td>0.106882</td>
          <td>25.537581</td>
          <td>0.163894</td>
          <td>24.879032</td>
          <td>0.204452</td>
          <td>0.035816</td>
          <td>0.030326</td>
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
          <td>28.245893</td>
          <td>1.226901</td>
          <td>26.486298</td>
          <td>0.138838</td>
          <td>25.377469</td>
          <td>0.046183</td>
          <td>25.031697</td>
          <td>0.055584</td>
          <td>24.862411</td>
          <td>0.091428</td>
          <td>24.480946</td>
          <td>0.146086</td>
          <td>0.040976</td>
          <td>0.028929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.851259</td>
          <td>0.595622</td>
          <td>27.086784</td>
          <td>0.294253</td>
          <td>26.013374</td>
          <td>0.108463</td>
          <td>25.270863</td>
          <td>0.092968</td>
          <td>24.824090</td>
          <td>0.118022</td>
          <td>24.460509</td>
          <td>0.191719</td>
          <td>0.183975</td>
          <td>0.171946</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.951685</td>
          <td>2.622525</td>
          <td>26.566694</td>
          <td>0.150703</td>
          <td>26.309284</td>
          <td>0.106821</td>
          <td>26.347926</td>
          <td>0.178246</td>
          <td>25.726248</td>
          <td>0.195511</td>
          <td>25.359393</td>
          <td>0.308324</td>
          <td>0.058213</td>
          <td>0.038503</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.509539</td>
          <td>0.483918</td>
          <td>26.074273</td>
          <td>0.132586</td>
          <td>26.056944</td>
          <td>0.119128</td>
          <td>25.977060</td>
          <td>0.181056</td>
          <td>25.988766</td>
          <td>0.330040</td>
          <td>26.421751</td>
          <td>0.878692</td>
          <td>0.252691</td>
          <td>0.148108</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.153146</td>
          <td>0.651725</td>
          <td>26.723982</td>
          <td>0.184233</td>
          <td>26.499071</td>
          <td>0.136055</td>
          <td>26.382366</td>
          <td>0.198372</td>
          <td>25.649886</td>
          <td>0.197520</td>
          <td>25.552581</td>
          <td>0.385872</td>
          <td>0.122101</td>
          <td>0.062592</td>
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
