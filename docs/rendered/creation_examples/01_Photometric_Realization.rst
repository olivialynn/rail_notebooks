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

    <pzflow.flow.Flow at 0x7ff9146f7550>



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
    0      23.994413  0.110061  0.080680  
    1      25.391064  0.242290  0.135166  
    2      24.304707  0.147938  0.127650  
    3      25.291103  0.004053  0.002921  
    4      25.096743  0.063258  0.032802  
    ...          ...       ...       ...  
    99995  24.737946  0.033789  0.027154  
    99996  24.224169  0.102273  0.068758  
    99997  25.613836  0.041323  0.036624  
    99998  25.274899  0.009185  0.006384  
    99999  25.699642  0.004419  0.004400  
    
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
          <td>26.478302</td>
          <td>0.135925</td>
          <td>26.112575</td>
          <td>0.087074</td>
          <td>25.139447</td>
          <td>0.060081</td>
          <td>24.708732</td>
          <td>0.078520</td>
          <td>23.796587</td>
          <td>0.079015</td>
          <td>0.110061</td>
          <td>0.080680</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.062235</td>
          <td>0.223039</td>
          <td>26.593842</td>
          <td>0.132557</td>
          <td>26.481419</td>
          <td>0.193227</td>
          <td>25.634807</td>
          <td>0.175512</td>
          <td>25.213479</td>
          <td>0.265791</td>
          <td>0.242290</td>
          <td>0.135166</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.417860</td>
          <td>0.733951</td>
          <td>28.413777</td>
          <td>0.632707</td>
          <td>27.609576</td>
          <td>0.310017</td>
          <td>26.177071</td>
          <td>0.149143</td>
          <td>25.176974</td>
          <td>0.118389</td>
          <td>24.205868</td>
          <td>0.113163</td>
          <td>0.147938</td>
          <td>0.127650</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.662729</td>
          <td>0.362372</td>
          <td>27.550223</td>
          <td>0.295586</td>
          <td>26.443455</td>
          <td>0.187137</td>
          <td>25.592242</td>
          <td>0.169275</td>
          <td>24.909195</td>
          <td>0.206647</td>
          <td>0.004053</td>
          <td>0.002921</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.861390</td>
          <td>0.225541</td>
          <td>25.891250</td>
          <td>0.081429</td>
          <td>25.902619</td>
          <td>0.072345</td>
          <td>25.674859</td>
          <td>0.096407</td>
          <td>25.832462</td>
          <td>0.207343</td>
          <td>25.384866</td>
          <td>0.305336</td>
          <td>0.063258</td>
          <td>0.032802</td>
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
          <td>26.988471</td>
          <td>0.544152</td>
          <td>26.339897</td>
          <td>0.120580</td>
          <td>25.337577</td>
          <td>0.043825</td>
          <td>25.051357</td>
          <td>0.055563</td>
          <td>24.818422</td>
          <td>0.086493</td>
          <td>24.822739</td>
          <td>0.192169</td>
          <td>0.033789</td>
          <td>0.027154</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.427826</td>
          <td>0.738856</td>
          <td>26.738854</td>
          <td>0.169921</td>
          <td>26.139150</td>
          <td>0.089134</td>
          <td>25.147896</td>
          <td>0.060533</td>
          <td>24.887752</td>
          <td>0.091932</td>
          <td>24.192082</td>
          <td>0.111811</td>
          <td>0.102273</td>
          <td>0.068758</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.627325</td>
          <td>0.415890</td>
          <td>26.660473</td>
          <td>0.158939</td>
          <td>26.533253</td>
          <td>0.125782</td>
          <td>26.154834</td>
          <td>0.146320</td>
          <td>26.022972</td>
          <td>0.242906</td>
          <td>25.389149</td>
          <td>0.306386</td>
          <td>0.041323</td>
          <td>0.036624</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.653013</td>
          <td>0.424118</td>
          <td>26.165800</td>
          <td>0.103613</td>
          <td>26.236378</td>
          <td>0.097082</td>
          <td>26.047812</td>
          <td>0.133425</td>
          <td>25.628159</td>
          <td>0.174524</td>
          <td>26.245607</td>
          <td>0.587080</td>
          <td>0.009185</td>
          <td>0.006384</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.635211</td>
          <td>0.418402</td>
          <td>26.623129</td>
          <td>0.153943</td>
          <td>26.797842</td>
          <td>0.157983</td>
          <td>26.238529</td>
          <td>0.157210</td>
          <td>25.819062</td>
          <td>0.205028</td>
          <td>25.766147</td>
          <td>0.411832</td>
          <td>0.004419</td>
          <td>0.004400</td>
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
          <td>26.950022</td>
          <td>0.238708</td>
          <td>26.096204</td>
          <td>0.104014</td>
          <td>25.199582</td>
          <td>0.077549</td>
          <td>24.646327</td>
          <td>0.090121</td>
          <td>24.015978</td>
          <td>0.116775</td>
          <td>0.110061</td>
          <td>0.080680</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.063818</td>
          <td>1.253194</td>
          <td>28.148303</td>
          <td>0.638989</td>
          <td>26.752469</td>
          <td>0.198117</td>
          <td>26.468528</td>
          <td>0.250327</td>
          <td>25.573997</td>
          <td>0.216817</td>
          <td>25.370044</td>
          <td>0.388784</td>
          <td>0.242290</td>
          <td>0.135166</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.684306</td>
          <td>0.980027</td>
          <td>29.436587</td>
          <td>1.358939</td>
          <td>29.188373</td>
          <td>1.111972</td>
          <td>26.227121</td>
          <td>0.194692</td>
          <td>25.269132</td>
          <td>0.159449</td>
          <td>24.376327</td>
          <td>0.164235</td>
          <td>0.147938</td>
          <td>0.127650</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.241097</td>
          <td>1.306279</td>
          <td>30.097650</td>
          <td>1.816921</td>
          <td>27.473085</td>
          <td>0.321944</td>
          <td>26.206157</td>
          <td>0.180052</td>
          <td>25.425600</td>
          <td>0.171750</td>
          <td>25.156255</td>
          <td>0.295986</td>
          <td>0.004053</td>
          <td>0.002921</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.419302</td>
          <td>0.395022</td>
          <td>26.065390</td>
          <td>0.110266</td>
          <td>25.865459</td>
          <td>0.083081</td>
          <td>25.801828</td>
          <td>0.128420</td>
          <td>25.156392</td>
          <td>0.137516</td>
          <td>25.180304</td>
          <td>0.304188</td>
          <td>0.063258</td>
          <td>0.032802</td>
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
          <td>26.366857</td>
          <td>0.377955</td>
          <td>26.458471</td>
          <td>0.154125</td>
          <td>25.367230</td>
          <td>0.053167</td>
          <td>25.138511</td>
          <td>0.071403</td>
          <td>24.672161</td>
          <td>0.089691</td>
          <td>24.868649</td>
          <td>0.234737</td>
          <td>0.033789</td>
          <td>0.027154</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.277285</td>
          <td>0.741187</td>
          <td>26.602955</td>
          <td>0.177693</td>
          <td>25.850775</td>
          <td>0.083389</td>
          <td>25.198008</td>
          <td>0.076998</td>
          <td>24.770138</td>
          <td>0.099909</td>
          <td>24.035541</td>
          <td>0.118113</td>
          <td>0.102273</td>
          <td>0.068758</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>30.287121</td>
          <td>3.037100</td>
          <td>26.960780</td>
          <td>0.235675</td>
          <td>26.312121</td>
          <td>0.122409</td>
          <td>25.965187</td>
          <td>0.147346</td>
          <td>26.768552</td>
          <td>0.505417</td>
          <td>25.637936</td>
          <td>0.433637</td>
          <td>0.041323</td>
          <td>0.036624</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.639950</td>
          <td>0.464550</td>
          <td>26.081028</td>
          <td>0.110946</td>
          <td>26.066697</td>
          <td>0.098330</td>
          <td>25.863856</td>
          <td>0.134351</td>
          <td>25.868521</td>
          <td>0.248851</td>
          <td>25.193431</td>
          <td>0.305013</td>
          <td>0.009185</td>
          <td>0.006384</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.266594</td>
          <td>0.348741</td>
          <td>26.897892</td>
          <td>0.222738</td>
          <td>26.954864</td>
          <td>0.210801</td>
          <td>26.085802</td>
          <td>0.162540</td>
          <td>25.795175</td>
          <td>0.234212</td>
          <td>25.471138</td>
          <td>0.379761</td>
          <td>0.004419</td>
          <td>0.004400</td>
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
          <td>27.501588</td>
          <td>0.823184</td>
          <td>26.671518</td>
          <td>0.176430</td>
          <td>25.888868</td>
          <td>0.079952</td>
          <td>25.335051</td>
          <td>0.080331</td>
          <td>24.510977</td>
          <td>0.073750</td>
          <td>24.056580</td>
          <td>0.111397</td>
          <td>0.110061</td>
          <td>0.080680</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.330015</td>
          <td>0.839064</td>
          <td>27.314512</td>
          <td>0.361273</td>
          <td>26.801687</td>
          <td>0.218635</td>
          <td>26.107462</td>
          <td>0.196498</td>
          <td>25.795857</td>
          <td>0.275377</td>
          <td>25.295477</td>
          <td>0.387287</td>
          <td>0.242290</td>
          <td>0.135166</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.965108</td>
          <td>0.605549</td>
          <td>28.431878</td>
          <td>0.738842</td>
          <td>28.834597</td>
          <td>0.888805</td>
          <td>26.185852</td>
          <td>0.185106</td>
          <td>25.156507</td>
          <td>0.142550</td>
          <td>24.306299</td>
          <td>0.152266</td>
          <td>0.147938</td>
          <td>0.127650</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.942413</td>
          <td>1.732574</td>
          <td>29.997034</td>
          <td>1.608442</td>
          <td>27.463427</td>
          <td>0.275580</td>
          <td>26.086525</td>
          <td>0.137985</td>
          <td>25.693944</td>
          <td>0.184560</td>
          <td>26.542639</td>
          <td>0.721201</td>
          <td>0.004053</td>
          <td>0.002921</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.449334</td>
          <td>0.369954</td>
          <td>26.011941</td>
          <td>0.093190</td>
          <td>26.016731</td>
          <td>0.082716</td>
          <td>25.744549</td>
          <td>0.106067</td>
          <td>25.615346</td>
          <td>0.178215</td>
          <td>25.075826</td>
          <td>0.245123</td>
          <td>0.063258</td>
          <td>0.032802</td>
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
          <td>27.086168</td>
          <td>0.587826</td>
          <td>26.340675</td>
          <td>0.121974</td>
          <td>25.393981</td>
          <td>0.046667</td>
          <td>25.066118</td>
          <td>0.057055</td>
          <td>24.874404</td>
          <td>0.092012</td>
          <td>24.873957</td>
          <td>0.203153</td>
          <td>0.033789</td>
          <td>0.027154</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.671559</td>
          <td>0.907778</td>
          <td>26.994962</td>
          <td>0.227738</td>
          <td>26.039390</td>
          <td>0.089567</td>
          <td>25.233139</td>
          <td>0.071971</td>
          <td>24.775223</td>
          <td>0.091339</td>
          <td>24.301876</td>
          <td>0.135187</td>
          <td>0.102273</td>
          <td>0.068758</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.630754</td>
          <td>0.422126</td>
          <td>26.565005</td>
          <td>0.149004</td>
          <td>26.229673</td>
          <td>0.098488</td>
          <td>26.159558</td>
          <td>0.150005</td>
          <td>25.806316</td>
          <td>0.206806</td>
          <td>26.214269</td>
          <td>0.584004</td>
          <td>0.041323</td>
          <td>0.036624</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.121590</td>
          <td>0.598818</td>
          <td>26.363354</td>
          <td>0.123149</td>
          <td>26.059359</td>
          <td>0.083157</td>
          <td>25.939223</td>
          <td>0.121549</td>
          <td>25.650454</td>
          <td>0.178003</td>
          <td>25.438108</td>
          <td>0.318876</td>
          <td>0.009185</td>
          <td>0.006384</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.406220</td>
          <td>1.328713</td>
          <td>26.708549</td>
          <td>0.165631</td>
          <td>26.535097</td>
          <td>0.126016</td>
          <td>26.304223</td>
          <td>0.166328</td>
          <td>26.167181</td>
          <td>0.273425</td>
          <td>25.348090</td>
          <td>0.296517</td>
          <td>0.004419</td>
          <td>0.004400</td>
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
