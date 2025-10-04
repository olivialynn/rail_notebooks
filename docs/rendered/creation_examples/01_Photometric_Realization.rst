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

    <pzflow.flow.Flow at 0x7f2c4f3683d0>



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
    0      23.994413  0.238150  0.237689  
    1      25.391064  0.047976  0.031042  
    2      24.304707  0.243653  0.204165  
    3      25.291103  0.067061  0.053229  
    4      25.096743  0.038543  0.032375  
    ...          ...       ...       ...  
    99995  24.737946  0.019157  0.014468  
    99996  24.224169  0.017960  0.014912  
    99997  25.613836  0.046814  0.029746  
    99998  25.274899  0.014481  0.010420  
    99999  25.699642  0.104580  0.067344  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.884821</td>
          <td>0.192267</td>
          <td>26.064908</td>
          <td>0.083493</td>
          <td>25.178818</td>
          <td>0.062217</td>
          <td>24.781708</td>
          <td>0.083741</td>
          <td>24.006558</td>
          <td>0.095058</td>
          <td>0.238150</td>
          <td>0.237689</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.105462</td>
          <td>0.591736</td>
          <td>28.078253</td>
          <td>0.497173</td>
          <td>26.791460</td>
          <td>0.157123</td>
          <td>26.601536</td>
          <td>0.213709</td>
          <td>26.268180</td>
          <td>0.296643</td>
          <td>24.833905</td>
          <td>0.193985</td>
          <td>0.047976</td>
          <td>0.031042</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.515894</td>
          <td>1.256794</td>
          <td>28.786737</td>
          <td>0.739903</td>
          <td>26.140145</td>
          <td>0.144484</td>
          <td>25.013387</td>
          <td>0.102641</td>
          <td>24.105484</td>
          <td>0.103666</td>
          <td>0.243653</td>
          <td>0.204165</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.378498</td>
          <td>0.714802</td>
          <td>27.917245</td>
          <td>0.440790</td>
          <td>27.114098</td>
          <td>0.206502</td>
          <td>26.225526</td>
          <td>0.155470</td>
          <td>25.534594</td>
          <td>0.161155</td>
          <td>24.994989</td>
          <td>0.221989</td>
          <td>0.067061</td>
          <td>0.053229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.784039</td>
          <td>0.211491</td>
          <td>26.137790</td>
          <td>0.101106</td>
          <td>25.941943</td>
          <td>0.074905</td>
          <td>25.734889</td>
          <td>0.101615</td>
          <td>25.424061</td>
          <td>0.146592</td>
          <td>24.911343</td>
          <td>0.207019</td>
          <td>0.038543</td>
          <td>0.032375</td>
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
          <td>27.636814</td>
          <td>0.846975</td>
          <td>26.316107</td>
          <td>0.118114</td>
          <td>25.581831</td>
          <td>0.054437</td>
          <td>25.048546</td>
          <td>0.055424</td>
          <td>24.852201</td>
          <td>0.089103</td>
          <td>24.465472</td>
          <td>0.141714</td>
          <td>0.019157</td>
          <td>0.014468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.845314</td>
          <td>0.490011</td>
          <td>26.684730</td>
          <td>0.162265</td>
          <td>26.056830</td>
          <td>0.082901</td>
          <td>25.237978</td>
          <td>0.065567</td>
          <td>24.858513</td>
          <td>0.089599</td>
          <td>24.192870</td>
          <td>0.111888</td>
          <td>0.017960</td>
          <td>0.014912</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.873150</td>
          <td>0.981247</td>
          <td>26.532540</td>
          <td>0.142427</td>
          <td>26.267829</td>
          <td>0.099796</td>
          <td>26.155701</td>
          <td>0.146429</td>
          <td>25.611294</td>
          <td>0.172041</td>
          <td>25.455512</td>
          <td>0.323070</td>
          <td>0.046814</td>
          <td>0.029746</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.906089</td>
          <td>0.512454</td>
          <td>26.360218</td>
          <td>0.122725</td>
          <td>26.049362</td>
          <td>0.082357</td>
          <td>25.728993</td>
          <td>0.101092</td>
          <td>25.919816</td>
          <td>0.223020</td>
          <td>25.056355</td>
          <td>0.233586</td>
          <td>0.014481</td>
          <td>0.010420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.126190</td>
          <td>0.280215</td>
          <td>26.768270</td>
          <td>0.174222</td>
          <td>26.753775</td>
          <td>0.152132</td>
          <td>26.058364</td>
          <td>0.134647</td>
          <td>26.332555</td>
          <td>0.312371</td>
          <td>26.049155</td>
          <td>0.509328</td>
          <td>0.104580</td>
          <td>0.067344</td>
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
          <td>27.200872</td>
          <td>0.764706</td>
          <td>26.554418</td>
          <td>0.192460</td>
          <td>26.101090</td>
          <td>0.119025</td>
          <td>25.228168</td>
          <td>0.091093</td>
          <td>24.701404</td>
          <td>0.107828</td>
          <td>23.821311</td>
          <td>0.112632</td>
          <td>0.238150</td>
          <td>0.237689</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.538137</td>
          <td>0.374918</td>
          <td>27.053297</td>
          <td>0.229988</td>
          <td>26.265606</td>
          <td>0.190368</td>
          <td>25.538483</td>
          <td>0.189980</td>
          <td>25.988385</td>
          <td>0.562043</td>
          <td>0.047976</td>
          <td>0.031042</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.360820</td>
          <td>0.840740</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.537036</td>
          <td>0.385901</td>
          <td>26.010231</td>
          <td>0.176414</td>
          <td>25.079359</td>
          <td>0.147323</td>
          <td>24.448596</td>
          <td>0.190017</td>
          <td>0.243653</td>
          <td>0.204165</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.592769</td>
          <td>0.357812</td>
          <td>26.196168</td>
          <td>0.180737</td>
          <td>25.525161</td>
          <td>0.189086</td>
          <td>24.751577</td>
          <td>0.214879</td>
          <td>0.067061</td>
          <td>0.053229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.345556</td>
          <td>0.372026</td>
          <td>26.079467</td>
          <td>0.111201</td>
          <td>25.871079</td>
          <td>0.083137</td>
          <td>25.661252</td>
          <td>0.113163</td>
          <td>25.743348</td>
          <td>0.225259</td>
          <td>25.118745</td>
          <td>0.288319</td>
          <td>0.038543</td>
          <td>0.032375</td>
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
          <td>26.628119</td>
          <td>0.460681</td>
          <td>26.173903</td>
          <td>0.120359</td>
          <td>25.411704</td>
          <td>0.055183</td>
          <td>25.002174</td>
          <td>0.063138</td>
          <td>24.804202</td>
          <td>0.100486</td>
          <td>24.980616</td>
          <td>0.256855</td>
          <td>0.019157</td>
          <td>0.014468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.715077</td>
          <td>0.491482</td>
          <td>26.932401</td>
          <td>0.229377</td>
          <td>26.047537</td>
          <td>0.096761</td>
          <td>25.139615</td>
          <td>0.071304</td>
          <td>24.762314</td>
          <td>0.096859</td>
          <td>24.511414</td>
          <td>0.173574</td>
          <td>0.017960</td>
          <td>0.014912</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.982435</td>
          <td>0.278946</td>
          <td>26.505748</td>
          <td>0.160765</td>
          <td>26.440531</td>
          <td>0.136805</td>
          <td>25.932608</td>
          <td>0.143281</td>
          <td>25.657089</td>
          <td>0.209815</td>
          <td>25.468209</td>
          <td>0.380672</td>
          <td>0.046814</td>
          <td>0.029746</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.985411</td>
          <td>0.278704</td>
          <td>26.221431</td>
          <td>0.125373</td>
          <td>26.120632</td>
          <td>0.103119</td>
          <td>25.567629</td>
          <td>0.103879</td>
          <td>25.744534</td>
          <td>0.224683</td>
          <td>25.403246</td>
          <td>0.360331</td>
          <td>0.014481</td>
          <td>0.010420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.503670</td>
          <td>0.859203</td>
          <td>26.885803</td>
          <td>0.225370</td>
          <td>26.622668</td>
          <td>0.163164</td>
          <td>26.485215</td>
          <td>0.233270</td>
          <td>25.678256</td>
          <td>0.217720</td>
          <td>24.774138</td>
          <td>0.221794</td>
          <td>0.104580</td>
          <td>0.067344</td>
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
          <td>26.882425</td>
          <td>0.672755</td>
          <td>26.567121</td>
          <td>0.219390</td>
          <td>26.079600</td>
          <td>0.133458</td>
          <td>25.224108</td>
          <td>0.104160</td>
          <td>24.897534</td>
          <td>0.145977</td>
          <td>23.727991</td>
          <td>0.118975</td>
          <td>0.238150</td>
          <td>0.237689</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.942314</td>
          <td>0.456500</td>
          <td>26.643742</td>
          <td>0.141362</td>
          <td>26.538013</td>
          <td>0.207074</td>
          <td>25.859778</td>
          <td>0.216530</td>
          <td>26.733178</td>
          <td>0.831306</td>
          <td>0.047976</td>
          <td>0.031042</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.693795</td>
          <td>1.100523</td>
          <td>27.681399</td>
          <td>0.512115</td>
          <td>32.612409</td>
          <td>4.301792</td>
          <td>26.275662</td>
          <td>0.246338</td>
          <td>24.742540</td>
          <td>0.123340</td>
          <td>24.212294</td>
          <td>0.174157</td>
          <td>0.243653</td>
          <td>0.204165</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.703349</td>
          <td>0.904690</td>
          <td>28.776612</td>
          <td>0.832823</td>
          <td>27.179975</td>
          <td>0.228168</td>
          <td>26.134399</td>
          <td>0.150910</td>
          <td>25.730103</td>
          <td>0.199059</td>
          <td>25.101478</td>
          <td>0.253848</td>
          <td>0.067061</td>
          <td>0.053229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.157940</td>
          <td>0.619871</td>
          <td>26.050449</td>
          <td>0.095046</td>
          <td>26.001762</td>
          <td>0.080324</td>
          <td>25.828570</td>
          <td>0.112249</td>
          <td>25.482970</td>
          <td>0.156753</td>
          <td>24.905488</td>
          <td>0.209483</td>
          <td>0.038543</td>
          <td>0.032375</td>
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
          <td>26.388220</td>
          <td>0.126161</td>
          <td>25.466451</td>
          <td>0.049331</td>
          <td>24.903896</td>
          <td>0.048947</td>
          <td>24.850028</td>
          <td>0.089281</td>
          <td>24.598895</td>
          <td>0.159534</td>
          <td>0.019157</td>
          <td>0.014468</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.107776</td>
          <td>1.128671</td>
          <td>26.866404</td>
          <td>0.189888</td>
          <td>26.028129</td>
          <td>0.081129</td>
          <td>25.229852</td>
          <td>0.065352</td>
          <td>24.653502</td>
          <td>0.075059</td>
          <td>24.156417</td>
          <td>0.108799</td>
          <td>0.017960</td>
          <td>0.014912</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.943918</td>
          <td>0.533065</td>
          <td>26.575496</td>
          <td>0.150366</td>
          <td>26.263584</td>
          <td>0.101469</td>
          <td>26.132439</td>
          <td>0.146567</td>
          <td>26.079356</td>
          <td>0.259310</td>
          <td>25.567314</td>
          <td>0.359675</td>
          <td>0.046814</td>
          <td>0.029746</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.165925</td>
          <td>0.289757</td>
          <td>26.148546</td>
          <td>0.102252</td>
          <td>25.992048</td>
          <td>0.078465</td>
          <td>25.986458</td>
          <td>0.126807</td>
          <td>25.440030</td>
          <td>0.148931</td>
          <td>25.496861</td>
          <td>0.334531</td>
          <td>0.014481</td>
          <td>0.010420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.524913</td>
          <td>0.153365</td>
          <td>26.333693</td>
          <td>0.116034</td>
          <td>26.498808</td>
          <td>0.215268</td>
          <td>25.682340</td>
          <td>0.199901</td>
          <td>25.180294</td>
          <td>0.283009</td>
          <td>0.104580</td>
          <td>0.067344</td>
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
