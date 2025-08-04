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

    <pzflow.flow.Flow at 0x7f5d89645ea0>



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
    0      23.994413  0.171538  0.169567  
    1      25.391064  0.091219  0.049605  
    2      24.304707  0.198705  0.183209  
    3      25.291103  0.023586  0.012463  
    4      25.096743  0.076254  0.060439  
    ...          ...       ...       ...  
    99995  24.737946  0.004872  0.002754  
    99996  24.224169  0.113406  0.064965  
    99997  25.613836  0.065475  0.054797  
    99998  25.274899  0.084966  0.072724  
    99999  25.699642  0.038217  0.024966  
    
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
          <td>27.808296</td>
          <td>0.943148</td>
          <td>26.492894</td>
          <td>0.137646</td>
          <td>26.191940</td>
          <td>0.093368</td>
          <td>25.127933</td>
          <td>0.059471</td>
          <td>24.615924</td>
          <td>0.072337</td>
          <td>24.029629</td>
          <td>0.097002</td>
          <td>0.171538</td>
          <td>0.169567</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.643810</td>
          <td>0.357041</td>
          <td>26.535618</td>
          <td>0.126040</td>
          <td>26.280035</td>
          <td>0.162887</td>
          <td>25.690125</td>
          <td>0.183935</td>
          <td>25.477002</td>
          <td>0.328637</td>
          <td>0.091219</td>
          <td>0.049605</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.341095</td>
          <td>0.332846</td>
          <td>28.910241</td>
          <td>0.880118</td>
          <td>27.904055</td>
          <td>0.390892</td>
          <td>26.020215</td>
          <td>0.130277</td>
          <td>24.922600</td>
          <td>0.094789</td>
          <td>24.230824</td>
          <td>0.115650</td>
          <td>0.198705</td>
          <td>0.183209</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.630654</td>
          <td>2.312018</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.592297</td>
          <td>0.305754</td>
          <td>26.218702</td>
          <td>0.154564</td>
          <td>25.474484</td>
          <td>0.153076</td>
          <td>25.273323</td>
          <td>0.279056</td>
          <td>0.023586</td>
          <td>0.012463</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.197948</td>
          <td>0.296919</td>
          <td>26.189292</td>
          <td>0.105761</td>
          <td>26.025071</td>
          <td>0.080611</td>
          <td>25.802013</td>
          <td>0.107759</td>
          <td>25.290953</td>
          <td>0.130693</td>
          <td>25.226350</td>
          <td>0.268596</td>
          <td>0.076254</td>
          <td>0.060439</td>
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
          <td>26.516119</td>
          <td>0.381773</td>
          <td>26.589010</td>
          <td>0.149507</td>
          <td>25.410532</td>
          <td>0.046756</td>
          <td>25.008266</td>
          <td>0.053477</td>
          <td>25.060801</td>
          <td>0.106986</td>
          <td>24.648172</td>
          <td>0.165734</td>
          <td>0.004872</td>
          <td>0.002754</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.424409</td>
          <td>0.737172</td>
          <td>26.444976</td>
          <td>0.132069</td>
          <td>26.175047</td>
          <td>0.091992</td>
          <td>25.309940</td>
          <td>0.069882</td>
          <td>24.958222</td>
          <td>0.097799</td>
          <td>24.177843</td>
          <td>0.110431</td>
          <td>0.113406</td>
          <td>0.064965</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.102842</td>
          <td>0.590637</td>
          <td>26.463275</td>
          <td>0.134173</td>
          <td>26.348244</td>
          <td>0.107071</td>
          <td>26.505806</td>
          <td>0.197235</td>
          <td>25.768585</td>
          <td>0.196520</td>
          <td>25.276739</td>
          <td>0.279830</td>
          <td>0.065475</td>
          <td>0.054797</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.468356</td>
          <td>0.367855</td>
          <td>26.293068</td>
          <td>0.115772</td>
          <td>26.054845</td>
          <td>0.082756</td>
          <td>26.041300</td>
          <td>0.132676</td>
          <td>25.973814</td>
          <td>0.233240</td>
          <td>24.831276</td>
          <td>0.193556</td>
          <td>0.084966</td>
          <td>0.072724</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.428047</td>
          <td>0.356443</td>
          <td>26.610609</td>
          <td>0.152301</td>
          <td>26.639983</td>
          <td>0.137947</td>
          <td>26.223160</td>
          <td>0.155155</td>
          <td>26.333030</td>
          <td>0.312490</td>
          <td>25.670898</td>
          <td>0.382673</td>
          <td>0.038217</td>
          <td>0.024966</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.858007</td>
          <td>0.232806</td>
          <td>26.162832</td>
          <td>0.116947</td>
          <td>25.146645</td>
          <td>0.078720</td>
          <td>24.644382</td>
          <td>0.095480</td>
          <td>24.158464</td>
          <td>0.140288</td>
          <td>0.171538</td>
          <td>0.169567</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.215956</td>
          <td>0.293396</td>
          <td>26.862864</td>
          <td>0.198513</td>
          <td>26.369554</td>
          <td>0.210296</td>
          <td>26.182619</td>
          <td>0.326113</td>
          <td>26.767935</td>
          <td>0.954461</td>
          <td>0.091219</td>
          <td>0.049605</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.781467</td>
          <td>1.656028</td>
          <td>28.354582</td>
          <td>0.680991</td>
          <td>26.123172</td>
          <td>0.187322</td>
          <td>25.349446</td>
          <td>0.179133</td>
          <td>24.312429</td>
          <td>0.163388</td>
          <td>0.198705</td>
          <td>0.183209</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.171628</td>
          <td>0.680514</td>
          <td>31.496952</td>
          <td>3.058701</td>
          <td>27.381138</td>
          <td>0.299422</td>
          <td>26.508189</td>
          <td>0.232176</td>
          <td>25.674638</td>
          <td>0.212107</td>
          <td>24.799936</td>
          <td>0.221304</td>
          <td>0.023586</td>
          <td>0.012463</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.001508</td>
          <td>0.285412</td>
          <td>26.298511</td>
          <td>0.135847</td>
          <td>26.066563</td>
          <td>0.099866</td>
          <td>25.603556</td>
          <td>0.108896</td>
          <td>25.634459</td>
          <td>0.207976</td>
          <td>24.903345</td>
          <td>0.244527</td>
          <td>0.076254</td>
          <td>0.060439</td>
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
          <td>26.440042</td>
          <td>0.399112</td>
          <td>26.403898</td>
          <td>0.146676</td>
          <td>25.565752</td>
          <td>0.063201</td>
          <td>25.078165</td>
          <td>0.067469</td>
          <td>24.788585</td>
          <td>0.099029</td>
          <td>24.788424</td>
          <td>0.218946</td>
          <td>0.004872</td>
          <td>0.002754</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.043838</td>
          <td>1.188305</td>
          <td>26.502146</td>
          <td>0.163525</td>
          <td>26.089464</td>
          <td>0.103136</td>
          <td>25.203056</td>
          <td>0.077578</td>
          <td>24.919034</td>
          <td>0.114117</td>
          <td>24.068211</td>
          <td>0.121876</td>
          <td>0.113406</td>
          <td>0.064965</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.577424</td>
          <td>0.446728</td>
          <td>27.285151</td>
          <td>0.308713</td>
          <td>26.319218</td>
          <td>0.124032</td>
          <td>26.502024</td>
          <td>0.233496</td>
          <td>25.721245</td>
          <td>0.222830</td>
          <td>26.035872</td>
          <td>0.584743</td>
          <td>0.065475</td>
          <td>0.054797</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.977649</td>
          <td>0.280947</td>
          <td>26.240578</td>
          <td>0.129794</td>
          <td>25.927357</td>
          <td>0.088823</td>
          <td>25.739168</td>
          <td>0.123167</td>
          <td>25.679268</td>
          <td>0.216933</td>
          <td>24.900290</td>
          <td>0.245086</td>
          <td>0.084966</td>
          <td>0.072724</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.582159</td>
          <td>0.892018</td>
          <td>26.532308</td>
          <td>0.164207</td>
          <td>26.618269</td>
          <td>0.159109</td>
          <td>26.280109</td>
          <td>0.192329</td>
          <td>25.723515</td>
          <td>0.221416</td>
          <td>25.208728</td>
          <td>0.309739</td>
          <td>0.038217</td>
          <td>0.024966</td>
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
          <td>26.577456</td>
          <td>0.190146</td>
          <td>25.834313</td>
          <td>0.090995</td>
          <td>25.219619</td>
          <td>0.087170</td>
          <td>24.527814</td>
          <td>0.089394</td>
          <td>23.925937</td>
          <td>0.119014</td>
          <td>0.171538</td>
          <td>0.169567</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.326432</td>
          <td>0.715769</td>
          <td>27.359495</td>
          <td>0.300388</td>
          <td>26.614140</td>
          <td>0.144062</td>
          <td>26.416295</td>
          <td>0.195608</td>
          <td>26.309628</td>
          <td>0.325888</td>
          <td>24.841712</td>
          <td>0.208634</td>
          <td>0.091219</td>
          <td>0.049605</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.051390</td>
          <td>0.581558</td>
          <td>26.089190</td>
          <td>0.194618</td>
          <td>25.008407</td>
          <td>0.143005</td>
          <td>24.116646</td>
          <td>0.147838</td>
          <td>0.198705</td>
          <td>0.183209</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.074145</td>
          <td>1.672083</td>
          <td>27.364729</td>
          <td>0.255344</td>
          <td>26.053295</td>
          <td>0.134726</td>
          <td>25.674104</td>
          <td>0.182299</td>
          <td>25.451471</td>
          <td>0.323492</td>
          <td>0.023586</td>
          <td>0.012463</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.904452</td>
          <td>0.243240</td>
          <td>26.159185</td>
          <td>0.108555</td>
          <td>25.900061</td>
          <td>0.076695</td>
          <td>25.769530</td>
          <td>0.111520</td>
          <td>25.374420</td>
          <td>0.148981</td>
          <td>25.012809</td>
          <td>0.238953</td>
          <td>0.076254</td>
          <td>0.060439</td>
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
          <td>28.405876</td>
          <td>1.328442</td>
          <td>26.317176</td>
          <td>0.118246</td>
          <td>25.469244</td>
          <td>0.049269</td>
          <td>25.183603</td>
          <td>0.062496</td>
          <td>24.833631</td>
          <td>0.087678</td>
          <td>24.612514</td>
          <td>0.160802</td>
          <td>0.004872</td>
          <td>0.002754</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.146431</td>
          <td>1.206853</td>
          <td>26.738875</td>
          <td>0.185248</td>
          <td>26.059161</td>
          <td>0.091981</td>
          <td>25.217142</td>
          <td>0.071645</td>
          <td>24.615539</td>
          <td>0.080094</td>
          <td>24.149510</td>
          <td>0.119579</td>
          <td>0.113406</td>
          <td>0.064965</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.294333</td>
          <td>0.692619</td>
          <td>27.055558</td>
          <td>0.230580</td>
          <td>26.614759</td>
          <td>0.141362</td>
          <td>26.340675</td>
          <td>0.179897</td>
          <td>25.738976</td>
          <td>0.200509</td>
          <td>26.296929</td>
          <td>0.632988</td>
          <td>0.065475</td>
          <td>0.054797</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.204846</td>
          <td>0.314043</td>
          <td>26.353471</td>
          <td>0.130533</td>
          <td>26.116028</td>
          <td>0.094447</td>
          <td>25.880792</td>
          <td>0.125148</td>
          <td>25.790401</td>
          <td>0.215601</td>
          <td>25.413372</td>
          <td>0.336136</td>
          <td>0.084966</td>
          <td>0.072724</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.275671</td>
          <td>0.318775</td>
          <td>27.033167</td>
          <td>0.220219</td>
          <td>26.600206</td>
          <td>0.135127</td>
          <td>26.282895</td>
          <td>0.165619</td>
          <td>26.110414</td>
          <td>0.264386</td>
          <td>26.110499</td>
          <td>0.539104</td>
          <td>0.038217</td>
          <td>0.024966</td>
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
