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

    <pzflow.flow.Flow at 0x7fb6b4ff7c10>



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
    0      23.994413  0.098060  0.081005  
    1      25.391064  0.069606  0.048686  
    2      24.304707  0.057982  0.055577  
    3      25.291103  0.080679  0.048301  
    4      25.096743  0.106301  0.075124  
    ...          ...       ...       ...  
    99995  24.737946  0.069059  0.040387  
    99996  24.224169  0.138584  0.091203  
    99997  25.613836  0.005898  0.005095  
    99998  25.274899  0.265039  0.179951  
    99999  25.699642  0.013114  0.006913  
    
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
          <td>28.712886</td>
          <td>1.553601</td>
          <td>26.637152</td>
          <td>0.155802</td>
          <td>26.130917</td>
          <td>0.088491</td>
          <td>25.223971</td>
          <td>0.064758</td>
          <td>24.693102</td>
          <td>0.077444</td>
          <td>24.037301</td>
          <td>0.097657</td>
          <td>0.098060</td>
          <td>0.081005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.056283</td>
          <td>0.489157</td>
          <td>26.610318</td>
          <td>0.134459</td>
          <td>26.440638</td>
          <td>0.186692</td>
          <td>25.760680</td>
          <td>0.195217</td>
          <td>25.179545</td>
          <td>0.258519</td>
          <td>0.069606</td>
          <td>0.048686</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.560555</td>
          <td>0.699960</td>
          <td>27.339788</td>
          <td>0.249049</td>
          <td>26.026896</td>
          <td>0.131033</td>
          <td>25.141290</td>
          <td>0.114768</td>
          <td>24.388728</td>
          <td>0.132633</td>
          <td>0.057982</td>
          <td>0.055577</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.807053</td>
          <td>0.824035</td>
          <td>27.257315</td>
          <td>0.232664</td>
          <td>26.265378</td>
          <td>0.160861</td>
          <td>25.509551</td>
          <td>0.157742</td>
          <td>25.525702</td>
          <td>0.341559</td>
          <td>0.080679</td>
          <td>0.048301</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.615600</td>
          <td>0.412178</td>
          <td>26.275582</td>
          <td>0.114024</td>
          <td>26.066981</td>
          <td>0.083646</td>
          <td>25.744781</td>
          <td>0.102499</td>
          <td>25.346149</td>
          <td>0.137078</td>
          <td>25.224600</td>
          <td>0.268213</td>
          <td>0.106301</td>
          <td>0.075124</td>
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
          <td>28.838805</td>
          <td>1.650697</td>
          <td>26.045452</td>
          <td>0.093250</td>
          <td>25.466447</td>
          <td>0.049136</td>
          <td>25.066997</td>
          <td>0.056340</td>
          <td>24.873487</td>
          <td>0.090787</td>
          <td>24.523935</td>
          <td>0.149021</td>
          <td>0.069059</td>
          <td>0.040387</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.576633</td>
          <td>0.147927</td>
          <td>25.782840</td>
          <td>0.065065</td>
          <td>25.297016</td>
          <td>0.069087</td>
          <td>24.941684</td>
          <td>0.096390</td>
          <td>24.194170</td>
          <td>0.112015</td>
          <td>0.138584</td>
          <td>0.091203</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.202778</td>
          <td>0.633653</td>
          <td>26.589406</td>
          <td>0.149558</td>
          <td>26.479132</td>
          <td>0.120009</td>
          <td>26.418632</td>
          <td>0.183251</td>
          <td>25.682299</td>
          <td>0.182721</td>
          <td>26.143379</td>
          <td>0.545560</td>
          <td>0.005898</td>
          <td>0.005095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.589984</td>
          <td>0.404164</td>
          <td>26.239154</td>
          <td>0.110463</td>
          <td>26.092347</td>
          <td>0.085536</td>
          <td>25.774818</td>
          <td>0.105228</td>
          <td>25.746969</td>
          <td>0.192976</td>
          <td>25.850932</td>
          <td>0.439303</td>
          <td>0.265039</td>
          <td>0.179951</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.976164</td>
          <td>0.539322</td>
          <td>26.569314</td>
          <td>0.147001</td>
          <td>26.578435</td>
          <td>0.130802</td>
          <td>26.205677</td>
          <td>0.152848</td>
          <td>25.902306</td>
          <td>0.219794</td>
          <td>25.625728</td>
          <td>0.369459</td>
          <td>0.013114</td>
          <td>0.006913</td>
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
          <td>28.393270</td>
          <td>1.431894</td>
          <td>26.390380</td>
          <td>0.148450</td>
          <td>25.996707</td>
          <td>0.094965</td>
          <td>25.162896</td>
          <td>0.074776</td>
          <td>24.826888</td>
          <td>0.105168</td>
          <td>24.288214</td>
          <td>0.147193</td>
          <td>0.098060</td>
          <td>0.081005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.157598</td>
          <td>0.678389</td>
          <td>27.074400</td>
          <td>0.260275</td>
          <td>26.655478</td>
          <td>0.165620</td>
          <td>25.817947</td>
          <td>0.130700</td>
          <td>25.845139</td>
          <td>0.246846</td>
          <td>25.379962</td>
          <td>0.357590</td>
          <td>0.069606</td>
          <td>0.048686</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>33.112743</td>
          <td>5.804492</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.137835</td>
          <td>0.539576</td>
          <td>26.040958</td>
          <td>0.158141</td>
          <td>25.016210</td>
          <td>0.122084</td>
          <td>24.072466</td>
          <td>0.120211</td>
          <td>0.057982</td>
          <td>0.055577</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.037121</td>
          <td>0.228916</td>
          <td>26.372596</td>
          <td>0.210180</td>
          <td>25.581923</td>
          <td>0.198810</td>
          <td>24.632008</td>
          <td>0.194861</td>
          <td>0.080679</td>
          <td>0.048301</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.663313</td>
          <td>0.481302</td>
          <td>26.092403</td>
          <td>0.114874</td>
          <td>26.115199</td>
          <td>0.105472</td>
          <td>25.758471</td>
          <td>0.126158</td>
          <td>25.452571</td>
          <td>0.180524</td>
          <td>25.427856</td>
          <td>0.376615</td>
          <td>0.106301</td>
          <td>0.075124</td>
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
          <td>28.691255</td>
          <td>1.646882</td>
          <td>26.166274</td>
          <td>0.120613</td>
          <td>25.484140</td>
          <td>0.059430</td>
          <td>25.048807</td>
          <td>0.066477</td>
          <td>24.872558</td>
          <td>0.107718</td>
          <td>24.563605</td>
          <td>0.183202</td>
          <td>0.069059</td>
          <td>0.040387</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.342539</td>
          <td>1.406256</td>
          <td>26.482533</td>
          <td>0.163156</td>
          <td>26.005250</td>
          <td>0.097384</td>
          <td>25.164451</td>
          <td>0.076267</td>
          <td>24.929312</td>
          <td>0.117028</td>
          <td>23.979171</td>
          <td>0.114692</td>
          <td>0.138584</td>
          <td>0.091203</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.403771</td>
          <td>0.388115</td>
          <td>26.696472</td>
          <td>0.188170</td>
          <td>26.387338</td>
          <td>0.130004</td>
          <td>26.090145</td>
          <td>0.163149</td>
          <td>25.939658</td>
          <td>0.263762</td>
          <td>25.757093</td>
          <td>0.472220</td>
          <td>0.005898</td>
          <td>0.005095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.024063</td>
          <td>0.319452</td>
          <td>26.051102</td>
          <td>0.123451</td>
          <td>26.105728</td>
          <td>0.117700</td>
          <td>25.904181</td>
          <td>0.161204</td>
          <td>25.648356</td>
          <td>0.238104</td>
          <td>25.248483</td>
          <td>0.364707</td>
          <td>0.265039</td>
          <td>0.179951</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.522923</td>
          <td>0.857756</td>
          <td>27.292801</td>
          <td>0.307589</td>
          <td>26.654163</td>
          <td>0.163558</td>
          <td>26.253022</td>
          <td>0.187394</td>
          <td>25.619007</td>
          <td>0.202294</td>
          <td>25.851108</td>
          <td>0.506416</td>
          <td>0.013114</td>
          <td>0.006913</td>
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
          <td>27.354717</td>
          <td>0.742316</td>
          <td>26.950945</td>
          <td>0.220707</td>
          <td>26.171806</td>
          <td>0.101237</td>
          <td>25.208861</td>
          <td>0.070911</td>
          <td>24.824863</td>
          <td>0.096009</td>
          <td>24.079831</td>
          <td>0.112214</td>
          <td>0.098060</td>
          <td>0.081005</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544165</td>
          <td>0.342317</td>
          <td>26.606809</td>
          <td>0.140310</td>
          <td>26.132824</td>
          <td>0.150576</td>
          <td>25.863317</td>
          <td>0.222333</td>
          <td>26.830332</td>
          <td>0.900747</td>
          <td>0.069606</td>
          <td>0.048686</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.006068</td>
          <td>1.083219</td>
          <td>32.670237</td>
          <td>4.068861</td>
          <td>28.644902</td>
          <td>0.694416</td>
          <td>26.184020</td>
          <td>0.156611</td>
          <td>24.942927</td>
          <td>0.100619</td>
          <td>24.435909</td>
          <td>0.144175</td>
          <td>0.057982</td>
          <td>0.055577</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.743678</td>
          <td>0.819923</td>
          <td>27.285678</td>
          <td>0.251008</td>
          <td>25.862918</td>
          <td>0.120432</td>
          <td>25.962843</td>
          <td>0.243598</td>
          <td>25.123506</td>
          <td>0.260634</td>
          <td>0.080679</td>
          <td>0.048301</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.345004</td>
          <td>0.356511</td>
          <td>26.137743</td>
          <td>0.110561</td>
          <td>25.960295</td>
          <td>0.084368</td>
          <td>25.599404</td>
          <td>0.100387</td>
          <td>25.193614</td>
          <td>0.132845</td>
          <td>24.796065</td>
          <td>0.207901</td>
          <td>0.106301</td>
          <td>0.075124</td>
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
          <td>26.695553</td>
          <td>0.449086</td>
          <td>26.458401</td>
          <td>0.138419</td>
          <td>25.531924</td>
          <td>0.054300</td>
          <td>25.031426</td>
          <td>0.057030</td>
          <td>24.775853</td>
          <td>0.086827</td>
          <td>24.751938</td>
          <td>0.188618</td>
          <td>0.069059</td>
          <td>0.040387</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.798058</td>
          <td>0.203815</td>
          <td>26.128314</td>
          <td>0.103076</td>
          <td>25.220632</td>
          <td>0.075987</td>
          <td>24.865053</td>
          <td>0.105181</td>
          <td>24.290295</td>
          <td>0.142572</td>
          <td>0.138584</td>
          <td>0.091203</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.341180</td>
          <td>0.332954</td>
          <td>26.575986</td>
          <td>0.147897</td>
          <td>26.220789</td>
          <td>0.095803</td>
          <td>26.716942</td>
          <td>0.235319</td>
          <td>25.985571</td>
          <td>0.235611</td>
          <td>25.805599</td>
          <td>0.424592</td>
          <td>0.005898</td>
          <td>0.005095</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.313126</td>
          <td>0.434004</td>
          <td>26.330228</td>
          <td>0.173509</td>
          <td>25.986890</td>
          <td>0.118377</td>
          <td>25.899456</td>
          <td>0.179025</td>
          <td>26.272646</td>
          <td>0.431990</td>
          <td>25.662940</td>
          <td>0.548506</td>
          <td>0.265039</td>
          <td>0.179951</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.920018</td>
          <td>0.518154</td>
          <td>26.775717</td>
          <td>0.175545</td>
          <td>26.538852</td>
          <td>0.126580</td>
          <td>26.392888</td>
          <td>0.179570</td>
          <td>26.234406</td>
          <td>0.289066</td>
          <td>25.094247</td>
          <td>0.241366</td>
          <td>0.013114</td>
          <td>0.006913</td>
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
