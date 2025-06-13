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

    <pzflow.flow.Flow at 0x7fba8ea158d0>



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
    0      23.994413  0.028902  0.024329  
    1      25.391064  0.169191  0.107850  
    2      24.304707  0.118175  0.087002  
    3      25.291103  0.026081  0.025933  
    4      25.096743  0.038854  0.034750  
    ...          ...       ...       ...  
    99995  24.737946  0.085891  0.077978  
    99996  24.224169  0.022108  0.015808  
    99997  25.613836  0.031797  0.024341  
    99998  25.274899  0.005309  0.004725  
    99999  25.699642  0.132987  0.130287  
    
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
          <td>31.091094</td>
          <td>3.672463</td>
          <td>26.503512</td>
          <td>0.138911</td>
          <td>26.003231</td>
          <td>0.079072</td>
          <td>25.168497</td>
          <td>0.061650</td>
          <td>24.806769</td>
          <td>0.085610</td>
          <td>23.947916</td>
          <td>0.090285</td>
          <td>0.028902</td>
          <td>0.024329</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.106163</td>
          <td>0.592030</td>
          <td>27.015751</td>
          <td>0.214571</td>
          <td>26.696383</td>
          <td>0.144815</td>
          <td>26.362271</td>
          <td>0.174701</td>
          <td>25.425556</td>
          <td>0.146781</td>
          <td>25.454265</td>
          <td>0.322749</td>
          <td>0.169191</td>
          <td>0.107850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.467442</td>
          <td>3.077077</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.319291</td>
          <td>0.168431</td>
          <td>24.852348</td>
          <td>0.089115</td>
          <td>24.451626</td>
          <td>0.140034</td>
          <td>0.118175</td>
          <td>0.087002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.162226</td>
          <td>0.214980</td>
          <td>26.159028</td>
          <td>0.146849</td>
          <td>25.442524</td>
          <td>0.148936</td>
          <td>25.804547</td>
          <td>0.424094</td>
          <td>0.026081</td>
          <td>0.025933</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.338395</td>
          <td>0.332135</td>
          <td>26.171006</td>
          <td>0.104085</td>
          <td>25.961490</td>
          <td>0.076210</td>
          <td>25.831793</td>
          <td>0.110598</td>
          <td>25.484134</td>
          <td>0.154347</td>
          <td>25.233247</td>
          <td>0.270109</td>
          <td>0.038854</td>
          <td>0.034750</td>
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
          <td>26.973222</td>
          <td>0.538173</td>
          <td>26.475753</td>
          <td>0.135626</td>
          <td>25.356579</td>
          <td>0.044570</td>
          <td>25.079288</td>
          <td>0.056958</td>
          <td>24.779890</td>
          <td>0.083607</td>
          <td>24.431997</td>
          <td>0.137683</td>
          <td>0.085891</td>
          <td>0.077978</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.575635</td>
          <td>0.399731</td>
          <td>26.624682</td>
          <td>0.154148</td>
          <td>25.938699</td>
          <td>0.074690</td>
          <td>25.202162</td>
          <td>0.063518</td>
          <td>24.701586</td>
          <td>0.078026</td>
          <td>23.992961</td>
          <td>0.093930</td>
          <td>0.022108</td>
          <td>0.015808</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.696481</td>
          <td>0.438345</td>
          <td>26.852616</td>
          <td>0.187116</td>
          <td>26.338911</td>
          <td>0.106201</td>
          <td>26.208742</td>
          <td>0.153251</td>
          <td>25.531769</td>
          <td>0.160767</td>
          <td>25.589502</td>
          <td>0.359140</td>
          <td>0.031797</td>
          <td>0.024341</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.028311</td>
          <td>0.258766</td>
          <td>26.375450</td>
          <td>0.124357</td>
          <td>26.156359</td>
          <td>0.090494</td>
          <td>25.852658</td>
          <td>0.112629</td>
          <td>25.793597</td>
          <td>0.200695</td>
          <td>25.019758</td>
          <td>0.226605</td>
          <td>0.005309</td>
          <td>0.004725</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.754179</td>
          <td>0.172149</td>
          <td>26.698553</td>
          <td>0.145085</td>
          <td>26.187777</td>
          <td>0.150520</td>
          <td>25.934750</td>
          <td>0.225806</td>
          <td>26.283538</td>
          <td>0.603079</td>
          <td>0.132987</td>
          <td>0.130287</td>
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
          <td>28.848813</td>
          <td>1.765949</td>
          <td>26.576625</td>
          <td>0.170354</td>
          <td>25.924623</td>
          <td>0.086987</td>
          <td>25.095238</td>
          <td>0.068666</td>
          <td>24.580924</td>
          <td>0.082707</td>
          <td>24.366484</td>
          <td>0.153616</td>
          <td>0.028902</td>
          <td>0.024329</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.462978</td>
          <td>2.178806</td>
          <td>26.449209</td>
          <td>0.145932</td>
          <td>26.358034</td>
          <td>0.217786</td>
          <td>25.852975</td>
          <td>0.260619</td>
          <td>25.272405</td>
          <td>0.344427</td>
          <td>0.169191</td>
          <td>0.107850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.518858</td>
          <td>0.433932</td>
          <td>30.430773</td>
          <td>2.125855</td>
          <td>27.364593</td>
          <td>0.304806</td>
          <td>25.969931</td>
          <td>0.152512</td>
          <td>25.030729</td>
          <td>0.126643</td>
          <td>24.196582</td>
          <td>0.137193</td>
          <td>0.118175</td>
          <td>0.087002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.084337</td>
          <td>1.958902</td>
          <td>30.007378</td>
          <td>1.746022</td>
          <td>27.169946</td>
          <td>0.252453</td>
          <td>26.329515</td>
          <td>0.200245</td>
          <td>25.250985</td>
          <td>0.148269</td>
          <td>26.502558</td>
          <td>0.797712</td>
          <td>0.026081</td>
          <td>0.025933</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.388362</td>
          <td>0.384675</td>
          <td>26.044376</td>
          <td>0.107882</td>
          <td>25.929195</td>
          <td>0.087530</td>
          <td>25.733472</td>
          <td>0.120540</td>
          <td>25.030294</td>
          <td>0.122823</td>
          <td>25.014976</td>
          <td>0.265093</td>
          <td>0.038854</td>
          <td>0.034750</td>
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
          <td>27.059316</td>
          <td>0.637829</td>
          <td>26.418747</td>
          <td>0.151515</td>
          <td>25.522595</td>
          <td>0.062220</td>
          <td>25.006240</td>
          <td>0.064802</td>
          <td>25.039543</td>
          <td>0.126006</td>
          <td>24.729876</td>
          <td>0.213106</td>
          <td>0.085891</td>
          <td>0.077978</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.124692</td>
          <td>0.268720</td>
          <td>26.039645</td>
          <td>0.096125</td>
          <td>25.128833</td>
          <td>0.070652</td>
          <td>24.893558</td>
          <td>0.108680</td>
          <td>24.397649</td>
          <td>0.157584</td>
          <td>0.022108</td>
          <td>0.015808</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.026152</td>
          <td>1.162026</td>
          <td>26.781242</td>
          <td>0.202532</td>
          <td>26.383956</td>
          <td>0.129959</td>
          <td>26.204065</td>
          <td>0.180215</td>
          <td>26.013164</td>
          <td>0.280701</td>
          <td>25.406035</td>
          <td>0.361841</td>
          <td>0.031797</td>
          <td>0.024341</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.197160</td>
          <td>0.330147</td>
          <td>26.163835</td>
          <td>0.119217</td>
          <td>26.235322</td>
          <td>0.113926</td>
          <td>25.953206</td>
          <td>0.145087</td>
          <td>25.820315</td>
          <td>0.239134</td>
          <td>25.806558</td>
          <td>0.489908</td>
          <td>0.005309</td>
          <td>0.004725</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.273123</td>
          <td>0.364178</td>
          <td>27.042548</td>
          <td>0.263086</td>
          <td>26.831115</td>
          <td>0.200395</td>
          <td>26.147747</td>
          <td>0.181132</td>
          <td>25.980090</td>
          <td>0.286940</td>
          <td>26.054320</td>
          <td>0.614055</td>
          <td>0.132987</td>
          <td>0.130287</td>
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
          <td>28.039538</td>
          <td>1.088025</td>
          <td>26.808178</td>
          <td>0.181670</td>
          <td>25.963205</td>
          <td>0.077067</td>
          <td>25.124037</td>
          <td>0.059873</td>
          <td>24.575029</td>
          <td>0.070445</td>
          <td>23.971568</td>
          <td>0.093107</td>
          <td>0.028902</td>
          <td>0.024329</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.362073</td>
          <td>0.795391</td>
          <td>28.277524</td>
          <td>0.667524</td>
          <td>26.528099</td>
          <td>0.153980</td>
          <td>26.339395</td>
          <td>0.211400</td>
          <td>25.753722</td>
          <td>0.237000</td>
          <td>25.107550</td>
          <td>0.298000</td>
          <td>0.169191</td>
          <td>0.107850</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.964081</td>
          <td>0.501863</td>
          <td>28.280452</td>
          <td>0.575893</td>
          <td>26.385395</td>
          <td>0.202243</td>
          <td>25.151438</td>
          <td>0.131192</td>
          <td>24.255302</td>
          <td>0.134432</td>
          <td>0.118175</td>
          <td>0.087002</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.457458</td>
          <td>2.167141</td>
          <td>28.355885</td>
          <td>0.611389</td>
          <td>27.068791</td>
          <td>0.200538</td>
          <td>26.620446</td>
          <td>0.219085</td>
          <td>25.627211</td>
          <td>0.175923</td>
          <td>25.303563</td>
          <td>0.288478</td>
          <td>0.026081</td>
          <td>0.025933</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.253570</td>
          <td>0.314040</td>
          <td>26.367150</td>
          <td>0.125400</td>
          <td>25.884557</td>
          <td>0.072511</td>
          <td>25.848818</td>
          <td>0.114391</td>
          <td>25.656591</td>
          <td>0.181932</td>
          <td>24.913894</td>
          <td>0.211212</td>
          <td>0.038854</td>
          <td>0.034750</td>
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
          <td>26.380628</td>
          <td>0.362128</td>
          <td>26.343455</td>
          <td>0.130032</td>
          <td>25.409346</td>
          <td>0.050839</td>
          <td>25.068805</td>
          <td>0.061655</td>
          <td>24.836426</td>
          <td>0.095543</td>
          <td>24.619764</td>
          <td>0.175937</td>
          <td>0.085891</td>
          <td>0.077978</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.326925</td>
          <td>1.275935</td>
          <td>26.618685</td>
          <td>0.154007</td>
          <td>26.189794</td>
          <td>0.093659</td>
          <td>25.388701</td>
          <td>0.075322</td>
          <td>24.844904</td>
          <td>0.088977</td>
          <td>24.291127</td>
          <td>0.122504</td>
          <td>0.022108</td>
          <td>0.015808</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.910849</td>
          <td>0.517459</td>
          <td>26.740745</td>
          <td>0.171742</td>
          <td>26.282268</td>
          <td>0.102158</td>
          <td>26.401583</td>
          <td>0.182603</td>
          <td>25.689195</td>
          <td>0.185703</td>
          <td>24.868472</td>
          <td>0.201854</td>
          <td>0.031797</td>
          <td>0.024341</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.919393</td>
          <td>0.236677</td>
          <td>26.210775</td>
          <td>0.107795</td>
          <td>26.024529</td>
          <td>0.080600</td>
          <td>25.768064</td>
          <td>0.104646</td>
          <td>25.480302</td>
          <td>0.153893</td>
          <td>27.699364</td>
          <td>1.431235</td>
          <td>0.005309</td>
          <td>0.004725</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.532264</td>
          <td>0.877055</td>
          <td>26.726489</td>
          <td>0.198048</td>
          <td>26.530968</td>
          <td>0.151506</td>
          <td>26.470216</td>
          <td>0.231388</td>
          <td>25.904752</td>
          <td>0.263562</td>
          <td>26.233047</td>
          <td>0.680796</td>
          <td>0.132987</td>
          <td>0.130287</td>
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
