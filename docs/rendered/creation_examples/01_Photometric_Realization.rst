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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7faf9da00410>



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
    0      23.994413  0.037106  0.018771  
    1      25.391064  0.081747  0.066400  
    2      24.304707  0.256743  0.213705  
    3      25.291103  0.145472  0.142331  
    4      25.096743  0.046312  0.038003  
    ...          ...       ...       ...  
    99995  24.737946  0.130094  0.100217  
    99996  24.224169  0.042725  0.024255  
    99997  25.613836  0.052177  0.036904  
    99998  25.274899  0.094564  0.060271  
    99999  25.699642  0.001180  0.000709  
    
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
          <td>28.624619</td>
          <td>1.487114</td>
          <td>26.682246</td>
          <td>0.161922</td>
          <td>26.000359</td>
          <td>0.078872</td>
          <td>25.139589</td>
          <td>0.060089</td>
          <td>24.705816</td>
          <td>0.078318</td>
          <td>23.881668</td>
          <td>0.085171</td>
          <td>0.037106</td>
          <td>0.018771</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.956047</td>
          <td>0.204123</td>
          <td>26.667445</td>
          <td>0.141252</td>
          <td>26.039128</td>
          <td>0.132427</td>
          <td>25.809057</td>
          <td>0.203316</td>
          <td>25.078589</td>
          <td>0.237920</td>
          <td>0.081747</td>
          <td>0.066400</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.665030</td>
          <td>0.324048</td>
          <td>25.858477</td>
          <td>0.113202</td>
          <td>24.922033</td>
          <td>0.094742</td>
          <td>24.361034</td>
          <td>0.129493</td>
          <td>0.256743</td>
          <td>0.213705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.404446</td>
          <td>0.628597</td>
          <td>27.130159</td>
          <td>0.209297</td>
          <td>26.303477</td>
          <td>0.166177</td>
          <td>25.754452</td>
          <td>0.194197</td>
          <td>25.423077</td>
          <td>0.314820</td>
          <td>0.145472</td>
          <td>0.142331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.433151</td>
          <td>0.357871</td>
          <td>26.274018</td>
          <td>0.113869</td>
          <td>25.967710</td>
          <td>0.076630</td>
          <td>25.540342</td>
          <td>0.085653</td>
          <td>25.416742</td>
          <td>0.145673</td>
          <td>24.735105</td>
          <td>0.178448</td>
          <td>0.046312</td>
          <td>0.038003</td>
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
          <td>27.764412</td>
          <td>0.917902</td>
          <td>26.781410</td>
          <td>0.176175</td>
          <td>25.399603</td>
          <td>0.046305</td>
          <td>25.151304</td>
          <td>0.060717</td>
          <td>24.825397</td>
          <td>0.087026</td>
          <td>24.892647</td>
          <td>0.203801</td>
          <td>0.130094</td>
          <td>0.100217</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.392613</td>
          <td>0.721628</td>
          <td>26.536222</td>
          <td>0.142879</td>
          <td>25.989229</td>
          <td>0.078100</td>
          <td>25.171447</td>
          <td>0.061811</td>
          <td>24.812648</td>
          <td>0.086055</td>
          <td>24.219870</td>
          <td>0.114552</td>
          <td>0.042725</td>
          <td>0.024255</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.102217</td>
          <td>0.274821</td>
          <td>26.794197</td>
          <td>0.178095</td>
          <td>26.275665</td>
          <td>0.100483</td>
          <td>26.135327</td>
          <td>0.143886</td>
          <td>25.818079</td>
          <td>0.204860</td>
          <td>25.667512</td>
          <td>0.381669</td>
          <td>0.052177</td>
          <td>0.036904</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.150731</td>
          <td>0.285832</td>
          <td>26.250822</td>
          <td>0.111592</td>
          <td>26.062885</td>
          <td>0.083344</td>
          <td>25.857928</td>
          <td>0.113147</td>
          <td>25.491773</td>
          <td>0.155360</td>
          <td>25.034863</td>
          <td>0.229464</td>
          <td>0.094564</td>
          <td>0.060271</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.362629</td>
          <td>0.338562</td>
          <td>26.584772</td>
          <td>0.148964</td>
          <td>26.361727</td>
          <td>0.108339</td>
          <td>26.077976</td>
          <td>0.136947</td>
          <td>26.809691</td>
          <td>0.452606</td>
          <td>25.940572</td>
          <td>0.469947</td>
          <td>0.001180</td>
          <td>0.000709</td>
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
          <td>26.271788</td>
          <td>0.350851</td>
          <td>27.445087</td>
          <td>0.347859</td>
          <td>25.847352</td>
          <td>0.081305</td>
          <td>25.246854</td>
          <td>0.078555</td>
          <td>24.732076</td>
          <td>0.094513</td>
          <td>23.993918</td>
          <td>0.111364</td>
          <td>0.037106</td>
          <td>0.018771</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.679639</td>
          <td>0.422419</td>
          <td>26.641045</td>
          <td>0.164614</td>
          <td>26.446598</td>
          <td>0.224374</td>
          <td>26.217188</td>
          <td>0.335359</td>
          <td>25.394371</td>
          <td>0.363771</td>
          <td>0.081747</td>
          <td>0.066400</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.371281</td>
          <td>0.715258</td>
          <td>25.865671</td>
          <td>0.157991</td>
          <td>24.990743</td>
          <td>0.138242</td>
          <td>24.445097</td>
          <td>0.191863</td>
          <td>0.256743</td>
          <td>0.213705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.909879</td>
          <td>1.122428</td>
          <td>30.137281</td>
          <td>1.905030</td>
          <td>27.213468</td>
          <td>0.277490</td>
          <td>26.215122</td>
          <td>0.193673</td>
          <td>25.495147</td>
          <td>0.194068</td>
          <td>24.985818</td>
          <td>0.274392</td>
          <td>0.145472</td>
          <td>0.142331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.645332</td>
          <td>0.211546</td>
          <td>26.025666</td>
          <td>0.106276</td>
          <td>25.849799</td>
          <td>0.081738</td>
          <td>25.552767</td>
          <td>0.103125</td>
          <td>25.486321</td>
          <td>0.181884</td>
          <td>24.849011</td>
          <td>0.231598</td>
          <td>0.046312</td>
          <td>0.038003</td>
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
          <td>27.440052</td>
          <td>0.833438</td>
          <td>26.286976</td>
          <td>0.137890</td>
          <td>25.469585</td>
          <td>0.060661</td>
          <td>25.109499</td>
          <td>0.072602</td>
          <td>24.731022</td>
          <td>0.098356</td>
          <td>24.753768</td>
          <td>0.221958</td>
          <td>0.130094</td>
          <td>0.100217</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.358298</td>
          <td>0.772639</td>
          <td>26.546666</td>
          <td>0.166309</td>
          <td>25.857616</td>
          <td>0.082140</td>
          <td>25.178655</td>
          <td>0.074052</td>
          <td>24.943741</td>
          <td>0.113861</td>
          <td>24.274041</td>
          <td>0.142129</td>
          <td>0.042725</td>
          <td>0.024255</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.761667</td>
          <td>0.997911</td>
          <td>27.165992</td>
          <td>0.279207</td>
          <td>26.406399</td>
          <td>0.133053</td>
          <td>26.256970</td>
          <td>0.189245</td>
          <td>25.754263</td>
          <td>0.227870</td>
          <td>26.862516</td>
          <td>1.002977</td>
          <td>0.052177</td>
          <td>0.036904</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.319091</td>
          <td>0.368558</td>
          <td>26.003290</td>
          <td>0.105619</td>
          <td>26.082128</td>
          <td>0.101746</td>
          <td>25.797043</td>
          <td>0.129508</td>
          <td>26.116478</td>
          <td>0.310183</td>
          <td>25.041008</td>
          <td>0.275002</td>
          <td>0.094564</td>
          <td>0.060271</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.743261</td>
          <td>0.501519</td>
          <td>26.771933</td>
          <td>0.200487</td>
          <td>26.806366</td>
          <td>0.186056</td>
          <td>26.279725</td>
          <td>0.191593</td>
          <td>25.947864</td>
          <td>0.265510</td>
          <td>26.039906</td>
          <td>0.580471</td>
          <td>0.001180</td>
          <td>0.000709</td>
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
          <td>28.538558</td>
          <td>1.430281</td>
          <td>26.550707</td>
          <td>0.146099</td>
          <td>26.043457</td>
          <td>0.082886</td>
          <td>25.215872</td>
          <td>0.065088</td>
          <td>24.805647</td>
          <td>0.086523</td>
          <td>23.859139</td>
          <td>0.084511</td>
          <td>0.037106</td>
          <td>0.018771</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.249769</td>
          <td>0.275519</td>
          <td>26.641820</td>
          <td>0.147950</td>
          <td>26.437343</td>
          <td>0.199685</td>
          <td>27.090650</td>
          <td>0.589214</td>
          <td>25.021711</td>
          <td>0.242973</td>
          <td>0.081747</td>
          <td>0.066400</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.798184</td>
          <td>1.941026</td>
          <td>29.233859</td>
          <td>1.386909</td>
          <td>27.552918</td>
          <td>0.442446</td>
          <td>25.679390</td>
          <td>0.153356</td>
          <td>25.017061</td>
          <td>0.160577</td>
          <td>24.240673</td>
          <td>0.183370</td>
          <td>0.256743</td>
          <td>0.213705</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.999122</td>
          <td>0.625822</td>
          <td>27.806973</td>
          <td>0.480113</td>
          <td>27.786600</td>
          <td>0.433237</td>
          <td>26.772542</td>
          <td>0.304692</td>
          <td>25.860348</td>
          <td>0.261308</td>
          <td>24.876050</td>
          <td>0.249402</td>
          <td>0.145472</td>
          <td>0.142331</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.640054</td>
          <td>0.426029</td>
          <td>26.051025</td>
          <td>0.095664</td>
          <td>26.041347</td>
          <td>0.083753</td>
          <td>25.749534</td>
          <td>0.105516</td>
          <td>25.721684</td>
          <td>0.193259</td>
          <td>25.449513</td>
          <td>0.328792</td>
          <td>0.046312</td>
          <td>0.038003</td>
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
          <td>27.365780</td>
          <td>0.771459</td>
          <td>26.437567</td>
          <td>0.149981</td>
          <td>25.529382</td>
          <td>0.060726</td>
          <td>25.018068</td>
          <td>0.063460</td>
          <td>24.891391</td>
          <td>0.107528</td>
          <td>24.719297</td>
          <td>0.205121</td>
          <td>0.130094</td>
          <td>0.100217</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.382377</td>
          <td>0.347343</td>
          <td>27.015542</td>
          <td>0.217395</td>
          <td>26.009260</td>
          <td>0.080788</td>
          <td>25.282103</td>
          <td>0.069351</td>
          <td>24.851130</td>
          <td>0.090462</td>
          <td>24.215478</td>
          <td>0.116019</td>
          <td>0.042725</td>
          <td>0.024255</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.160128</td>
          <td>0.624307</td>
          <td>26.689170</td>
          <td>0.166625</td>
          <td>26.534525</td>
          <td>0.129308</td>
          <td>25.895450</td>
          <td>0.120220</td>
          <td>25.887691</td>
          <td>0.222734</td>
          <td>25.596567</td>
          <td>0.370213</td>
          <td>0.052177</td>
          <td>0.036904</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.187627</td>
          <td>0.309706</td>
          <td>26.281328</td>
          <td>0.122589</td>
          <td>26.275956</td>
          <td>0.108592</td>
          <td>25.916193</td>
          <td>0.128980</td>
          <td>25.941322</td>
          <td>0.244231</td>
          <td>26.835004</td>
          <td>0.924250</td>
          <td>0.094564</td>
          <td>0.060271</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.035363</td>
          <td>0.562864</td>
          <td>26.916407</td>
          <td>0.197447</td>
          <td>26.559235</td>
          <td>0.128648</td>
          <td>26.282110</td>
          <td>0.163178</td>
          <td>25.735699</td>
          <td>0.191155</td>
          <td>25.458788</td>
          <td>0.323917</td>
          <td>0.001180</td>
          <td>0.000709</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_24_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_25_0.png


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
