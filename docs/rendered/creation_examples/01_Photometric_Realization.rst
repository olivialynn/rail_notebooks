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

    <pzflow.flow.Flow at 0x7f73243de020>



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
    0      23.994413  0.027528  0.026674  
    1      25.391064  0.081924  0.062119  
    2      24.304707  0.053106  0.051798  
    3      25.291103  0.119962  0.097752  
    4      25.096743  0.071312  0.056581  
    ...          ...       ...       ...  
    99995  24.737946  0.052633  0.035794  
    99996  24.224169  0.030283  0.028515  
    99997  25.613836  0.019619  0.017985  
    99998  25.274899  0.050409  0.036797  
    99999  25.699642  0.232098  0.206833  
    
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
          <td>26.459348</td>
          <td>0.133719</td>
          <td>26.157501</td>
          <td>0.090584</td>
          <td>25.187868</td>
          <td>0.062718</td>
          <td>24.590488</td>
          <td>0.070728</td>
          <td>24.085257</td>
          <td>0.101847</td>
          <td>0.027528</td>
          <td>0.026674</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.673629</td>
          <td>3.272164</td>
          <td>27.682954</td>
          <td>0.368144</td>
          <td>26.667202</td>
          <td>0.141222</td>
          <td>26.223658</td>
          <td>0.155222</td>
          <td>25.987504</td>
          <td>0.235896</td>
          <td>25.517457</td>
          <td>0.339341</td>
          <td>0.081924</td>
          <td>0.062119</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.534177</td>
          <td>1.269363</td>
          <td>28.722718</td>
          <td>0.708747</td>
          <td>25.862504</td>
          <td>0.113600</td>
          <td>25.251513</td>
          <td>0.126304</td>
          <td>24.515268</td>
          <td>0.147916</td>
          <td>0.053106</td>
          <td>0.051798</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.750152</td>
          <td>0.387881</td>
          <td>27.207712</td>
          <td>0.223282</td>
          <td>26.517041</td>
          <td>0.199106</td>
          <td>25.410927</td>
          <td>0.144946</td>
          <td>24.960971</td>
          <td>0.215786</td>
          <td>0.119962</td>
          <td>0.097752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.487308</td>
          <td>0.373326</td>
          <td>26.033545</td>
          <td>0.092281</td>
          <td>26.106474</td>
          <td>0.086607</td>
          <td>25.804287</td>
          <td>0.107974</td>
          <td>25.755641</td>
          <td>0.194391</td>
          <td>24.944146</td>
          <td>0.212777</td>
          <td>0.071312</td>
          <td>0.056581</td>
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
          <td>26.480978</td>
          <td>0.371491</td>
          <td>26.315845</td>
          <td>0.118087</td>
          <td>25.450550</td>
          <td>0.048448</td>
          <td>25.034253</td>
          <td>0.054726</td>
          <td>24.885942</td>
          <td>0.091786</td>
          <td>24.673324</td>
          <td>0.169324</td>
          <td>0.052633</td>
          <td>0.035794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.255640</td>
          <td>1.224518</td>
          <td>26.706093</td>
          <td>0.165248</td>
          <td>25.942198</td>
          <td>0.074922</td>
          <td>25.175835</td>
          <td>0.062052</td>
          <td>24.841317</td>
          <td>0.088254</td>
          <td>24.135626</td>
          <td>0.106435</td>
          <td>0.030283</td>
          <td>0.028515</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.714864</td>
          <td>1.555106</td>
          <td>26.818803</td>
          <td>0.181845</td>
          <td>26.193007</td>
          <td>0.093456</td>
          <td>26.523005</td>
          <td>0.200106</td>
          <td>25.654172</td>
          <td>0.178419</td>
          <td>25.941390</td>
          <td>0.470234</td>
          <td>0.019619</td>
          <td>0.017985</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.940805</td>
          <td>0.240838</td>
          <td>26.311349</td>
          <td>0.117627</td>
          <td>26.493233</td>
          <td>0.121488</td>
          <td>25.874900</td>
          <td>0.114833</td>
          <td>25.708179</td>
          <td>0.186764</td>
          <td>25.126499</td>
          <td>0.247506</td>
          <td>0.050409</td>
          <td>0.036797</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.644200</td>
          <td>0.850980</td>
          <td>26.867612</td>
          <td>0.189499</td>
          <td>26.346737</td>
          <td>0.106930</td>
          <td>26.952432</td>
          <td>0.285222</td>
          <td>26.017600</td>
          <td>0.241833</td>
          <td>25.554871</td>
          <td>0.349504</td>
          <td>0.232098</td>
          <td>0.206833</td>
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
          <td>28.213151</td>
          <td>1.288277</td>
          <td>26.643683</td>
          <td>0.180336</td>
          <td>26.129704</td>
          <td>0.104143</td>
          <td>25.214365</td>
          <td>0.076299</td>
          <td>24.692062</td>
          <td>0.091210</td>
          <td>23.781589</td>
          <td>0.092437</td>
          <td>0.027528</td>
          <td>0.026674</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.119224</td>
          <td>0.271225</td>
          <td>26.521065</td>
          <td>0.148426</td>
          <td>26.270896</td>
          <td>0.193535</td>
          <td>26.029476</td>
          <td>0.288369</td>
          <td>25.388688</td>
          <td>0.361872</td>
          <td>0.081924</td>
          <td>0.062119</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.966854</td>
          <td>2.742611</td>
          <td>29.414912</td>
          <td>1.305957</td>
          <td>28.350105</td>
          <td>0.626891</td>
          <td>26.137591</td>
          <td>0.171451</td>
          <td>24.863940</td>
          <td>0.106755</td>
          <td>24.070214</td>
          <td>0.119783</td>
          <td>0.053106</td>
          <td>0.051798</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.123417</td>
          <td>0.673491</td>
          <td>27.391485</td>
          <td>0.343407</td>
          <td>27.048593</td>
          <td>0.236456</td>
          <td>26.169326</td>
          <td>0.181451</td>
          <td>25.394554</td>
          <td>0.173715</td>
          <td>24.890458</td>
          <td>0.247414</td>
          <td>0.119962</td>
          <td>0.097752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.684433</td>
          <td>0.484474</td>
          <td>25.980604</td>
          <td>0.102898</td>
          <td>25.987647</td>
          <td>0.093007</td>
          <td>25.804872</td>
          <td>0.129465</td>
          <td>25.635923</td>
          <td>0.207846</td>
          <td>24.816864</td>
          <td>0.227225</td>
          <td>0.071312</td>
          <td>0.056581</td>
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
          <td>26.326017</td>
          <td>0.137999</td>
          <td>25.397343</td>
          <td>0.054809</td>
          <td>25.071837</td>
          <td>0.067570</td>
          <td>25.010658</td>
          <td>0.121015</td>
          <td>24.380755</td>
          <td>0.156195</td>
          <td>0.052633</td>
          <td>0.035794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.557158</td>
          <td>0.167632</td>
          <td>26.027528</td>
          <td>0.095269</td>
          <td>25.160893</td>
          <td>0.072811</td>
          <td>25.018272</td>
          <td>0.121348</td>
          <td>23.962121</td>
          <td>0.108317</td>
          <td>0.030283</td>
          <td>0.028515</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.565863</td>
          <td>2.373555</td>
          <td>26.979751</td>
          <td>0.238594</td>
          <td>26.439549</td>
          <td>0.136152</td>
          <td>26.090987</td>
          <td>0.163448</td>
          <td>25.718892</td>
          <td>0.220080</td>
          <td>26.098482</td>
          <td>0.605704</td>
          <td>0.019619</td>
          <td>0.017985</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.416633</td>
          <td>0.393692</td>
          <td>26.210768</td>
          <td>0.124891</td>
          <td>25.992763</td>
          <td>0.092744</td>
          <td>26.011306</td>
          <td>0.153513</td>
          <td>26.167610</td>
          <td>0.318947</td>
          <td>25.691539</td>
          <td>0.452135</td>
          <td>0.050409</td>
          <td>0.036797</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.117963</td>
          <td>0.342662</td>
          <td>26.523420</td>
          <td>0.184068</td>
          <td>26.751333</td>
          <td>0.203389</td>
          <td>27.266575</td>
          <td>0.480939</td>
          <td>25.904087</td>
          <td>0.291933</td>
          <td>25.354439</td>
          <td>0.394095</td>
          <td>0.232098</td>
          <td>0.206833</td>
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
          <td>27.644095</td>
          <td>0.855166</td>
          <td>26.918060</td>
          <td>0.199348</td>
          <td>26.110998</td>
          <td>0.087819</td>
          <td>25.175242</td>
          <td>0.062674</td>
          <td>24.680700</td>
          <td>0.077365</td>
          <td>24.069268</td>
          <td>0.101466</td>
          <td>0.027528</td>
          <td>0.026674</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.360514</td>
          <td>1.334135</td>
          <td>27.263159</td>
          <td>0.277812</td>
          <td>26.423123</td>
          <td>0.122098</td>
          <td>26.353154</td>
          <td>0.185417</td>
          <td>25.536753</td>
          <td>0.172219</td>
          <td>25.013618</td>
          <td>0.240612</td>
          <td>0.081924</td>
          <td>0.062119</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.454985</td>
          <td>0.766753</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.538534</td>
          <td>0.302630</td>
          <td>26.054138</td>
          <td>0.139228</td>
          <td>25.036523</td>
          <td>0.108558</td>
          <td>24.286081</td>
          <td>0.125911</td>
          <td>0.053106</td>
          <td>0.051798</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.729143</td>
          <td>0.424991</td>
          <td>27.439848</td>
          <td>0.307270</td>
          <td>26.154080</td>
          <td>0.168484</td>
          <td>25.670961</td>
          <td>0.206825</td>
          <td>25.981684</td>
          <td>0.546457</td>
          <td>0.119962</td>
          <td>0.097752</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.198003</td>
          <td>0.307252</td>
          <td>26.055867</td>
          <td>0.098569</td>
          <td>25.881739</td>
          <td>0.074927</td>
          <td>25.625413</td>
          <td>0.097592</td>
          <td>25.377650</td>
          <td>0.148364</td>
          <td>25.074995</td>
          <td>0.249790</td>
          <td>0.071312</td>
          <td>0.056581</td>
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
          <td>26.839575</td>
          <td>0.495647</td>
          <td>26.173866</td>
          <td>0.106785</td>
          <td>25.445733</td>
          <td>0.049563</td>
          <td>25.127878</td>
          <td>0.061176</td>
          <td>24.857341</td>
          <td>0.091928</td>
          <td>24.828784</td>
          <td>0.198342</td>
          <td>0.052633</td>
          <td>0.035794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.635034</td>
          <td>0.421286</td>
          <td>26.782717</td>
          <td>0.178096</td>
          <td>26.039160</td>
          <td>0.082577</td>
          <td>25.358221</td>
          <td>0.073835</td>
          <td>24.770546</td>
          <td>0.083893</td>
          <td>24.286324</td>
          <td>0.122827</td>
          <td>0.030283</td>
          <td>0.028515</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.130912</td>
          <td>0.604100</td>
          <td>26.345562</td>
          <td>0.121674</td>
          <td>26.517657</td>
          <td>0.124680</td>
          <td>26.389159</td>
          <td>0.179602</td>
          <td>26.495426</td>
          <td>0.356928</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.019619</td>
          <td>0.017985</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.660653</td>
          <td>0.871150</td>
          <td>26.154060</td>
          <td>0.104866</td>
          <td>26.348870</td>
          <td>0.109904</td>
          <td>25.896091</td>
          <td>0.120137</td>
          <td>25.383807</td>
          <td>0.145210</td>
          <td>25.274007</td>
          <td>0.286165</td>
          <td>0.050409</td>
          <td>0.036797</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.724088</td>
          <td>0.583869</td>
          <td>26.690137</td>
          <td>0.233178</td>
          <td>26.723024</td>
          <td>0.220395</td>
          <td>26.380053</td>
          <td>0.265569</td>
          <td>26.213759</td>
          <td>0.410824</td>
          <td>25.750653</td>
          <td>0.581272</td>
          <td>0.232098</td>
          <td>0.206833</td>
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
