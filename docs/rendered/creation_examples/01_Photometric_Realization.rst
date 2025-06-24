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

    <pzflow.flow.Flow at 0x7f016059cf40>



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
    0      23.994413  0.085844  0.079148  
    1      25.391064  0.017760  0.010737  
    2      24.304707  0.095393  0.051267  
    3      25.291103  0.004861  0.002595  
    4      25.096743  0.038685  0.029540  
    ...          ...       ...       ...  
    99995  24.737946  0.066367  0.033539  
    99996  24.224169  0.156089  0.145370  
    99997  25.613836  0.045754  0.045460  
    99998  25.274899  0.070489  0.054690  
    99999  25.699642  0.009272  0.007639  
    
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
          <td>26.996920</td>
          <td>0.211225</td>
          <td>26.015803</td>
          <td>0.079954</td>
          <td>25.155936</td>
          <td>0.060967</td>
          <td>24.637812</td>
          <td>0.073751</td>
          <td>24.154918</td>
          <td>0.108243</td>
          <td>0.085844</td>
          <td>0.079148</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.453375</td>
          <td>0.751534</td>
          <td>27.398256</td>
          <td>0.293698</td>
          <td>26.445733</td>
          <td>0.116572</td>
          <td>26.200780</td>
          <td>0.152208</td>
          <td>25.788614</td>
          <td>0.199857</td>
          <td>25.053207</td>
          <td>0.232978</td>
          <td>0.017760</td>
          <td>0.010737</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.751458</td>
          <td>0.456892</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.628029</td>
          <td>0.314627</td>
          <td>25.914954</td>
          <td>0.118907</td>
          <td>24.989563</td>
          <td>0.100522</td>
          <td>24.308708</td>
          <td>0.123751</td>
          <td>0.095393</td>
          <td>0.051267</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.962846</td>
          <td>0.534133</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.305132</td>
          <td>0.242042</td>
          <td>26.183957</td>
          <td>0.150027</td>
          <td>25.434946</td>
          <td>0.147970</td>
          <td>24.751260</td>
          <td>0.180907</td>
          <td>0.004861</td>
          <td>0.002595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.517939</td>
          <td>0.382313</td>
          <td>26.152940</td>
          <td>0.102454</td>
          <td>25.957992</td>
          <td>0.075975</td>
          <td>25.843014</td>
          <td>0.111686</td>
          <td>25.572833</td>
          <td>0.166500</td>
          <td>25.032184</td>
          <td>0.228954</td>
          <td>0.038685</td>
          <td>0.029540</td>
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
          <td>27.004544</td>
          <td>0.550509</td>
          <td>26.261642</td>
          <td>0.112649</td>
          <td>25.399248</td>
          <td>0.046290</td>
          <td>25.066255</td>
          <td>0.056303</td>
          <td>24.730706</td>
          <td>0.080058</td>
          <td>24.595466</td>
          <td>0.158441</td>
          <td>0.066367</td>
          <td>0.033539</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.423401</td>
          <td>1.340718</td>
          <td>26.705347</td>
          <td>0.165143</td>
          <td>25.943324</td>
          <td>0.074996</td>
          <td>25.320979</td>
          <td>0.070569</td>
          <td>25.033714</td>
          <td>0.104483</td>
          <td>24.328801</td>
          <td>0.125927</td>
          <td>0.156089</td>
          <td>0.145370</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.766030</td>
          <td>0.918825</td>
          <td>26.667878</td>
          <td>0.159948</td>
          <td>26.443022</td>
          <td>0.116298</td>
          <td>26.039699</td>
          <td>0.132492</td>
          <td>25.865485</td>
          <td>0.213148</td>
          <td>25.468109</td>
          <td>0.326323</td>
          <td>0.045754</td>
          <td>0.045460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.342893</td>
          <td>0.333320</td>
          <td>26.337381</td>
          <td>0.120317</td>
          <td>26.054377</td>
          <td>0.082722</td>
          <td>26.051227</td>
          <td>0.133819</td>
          <td>25.566300</td>
          <td>0.165575</td>
          <td>25.076549</td>
          <td>0.237519</td>
          <td>0.070489</td>
          <td>0.054690</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.010982</td>
          <td>0.553072</td>
          <td>26.713880</td>
          <td>0.166348</td>
          <td>26.667553</td>
          <td>0.141265</td>
          <td>26.506553</td>
          <td>0.197358</td>
          <td>25.973498</td>
          <td>0.233178</td>
          <td>26.145776</td>
          <td>0.546507</td>
          <td>0.009272</td>
          <td>0.007639</td>
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
          <td>26.915554</td>
          <td>0.576462</td>
          <td>26.983503</td>
          <td>0.243732</td>
          <td>25.995763</td>
          <td>0.094498</td>
          <td>25.432301</td>
          <td>0.094400</td>
          <td>24.675503</td>
          <td>0.091728</td>
          <td>24.032856</td>
          <td>0.117546</td>
          <td>0.085844</td>
          <td>0.079148</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.483414</td>
          <td>0.836536</td>
          <td>27.339879</td>
          <td>0.319473</td>
          <td>26.416241</td>
          <td>0.133377</td>
          <td>26.385705</td>
          <td>0.209575</td>
          <td>25.749116</td>
          <td>0.225580</td>
          <td>25.393914</td>
          <td>0.357768</td>
          <td>0.017760</td>
          <td>0.010737</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.057277</td>
          <td>0.559305</td>
          <td>28.144851</td>
          <td>0.546180</td>
          <td>26.020622</td>
          <td>0.156765</td>
          <td>25.017788</td>
          <td>0.123293</td>
          <td>24.254110</td>
          <td>0.141891</td>
          <td>0.095393</td>
          <td>0.051267</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.124679</td>
          <td>0.658466</td>
          <td>28.849111</td>
          <td>0.937573</td>
          <td>27.834461</td>
          <td>0.426669</td>
          <td>26.080305</td>
          <td>0.161776</td>
          <td>25.235695</td>
          <td>0.146013</td>
          <td>25.900194</td>
          <td>0.524821</td>
          <td>0.004861</td>
          <td>0.002595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.669247</td>
          <td>0.475989</td>
          <td>26.132229</td>
          <td>0.116396</td>
          <td>26.006524</td>
          <td>0.093630</td>
          <td>25.612630</td>
          <td>0.108433</td>
          <td>25.292050</td>
          <td>0.153842</td>
          <td>24.925088</td>
          <td>0.246123</td>
          <td>0.038685</td>
          <td>0.029540</td>
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
          <td>28.402372</td>
          <td>1.427416</td>
          <td>26.421431</td>
          <td>0.150118</td>
          <td>25.436280</td>
          <td>0.056877</td>
          <td>25.105304</td>
          <td>0.069780</td>
          <td>24.908150</td>
          <td>0.110955</td>
          <td>24.575938</td>
          <td>0.184857</td>
          <td>0.066367</td>
          <td>0.033539</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.694504</td>
          <td>0.506858</td>
          <td>26.868430</td>
          <td>0.231153</td>
          <td>25.889641</td>
          <td>0.090431</td>
          <td>25.323906</td>
          <td>0.090298</td>
          <td>24.920985</td>
          <td>0.119376</td>
          <td>24.290868</td>
          <td>0.154336</td>
          <td>0.156089</td>
          <td>0.145370</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.979286</td>
          <td>0.278592</td>
          <td>26.664437</td>
          <td>0.184254</td>
          <td>26.259277</td>
          <td>0.117132</td>
          <td>26.062354</td>
          <td>0.160439</td>
          <td>25.787187</td>
          <td>0.234200</td>
          <td>25.837066</td>
          <td>0.504091</td>
          <td>0.045754</td>
          <td>0.045460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.091308</td>
          <td>0.306251</td>
          <td>26.216382</td>
          <td>0.126261</td>
          <td>25.985051</td>
          <td>0.092745</td>
          <td>25.770410</td>
          <td>0.125587</td>
          <td>26.203918</td>
          <td>0.330311</td>
          <td>25.448129</td>
          <td>0.377565</td>
          <td>0.070489</td>
          <td>0.054690</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.182223</td>
          <td>0.326295</td>
          <td>26.609120</td>
          <td>0.174788</td>
          <td>26.405322</td>
          <td>0.132061</td>
          <td>26.744463</td>
          <td>0.281519</td>
          <td>26.063627</td>
          <td>0.291730</td>
          <td>25.211736</td>
          <td>0.309530</td>
          <td>0.009272</td>
          <td>0.007639</td>
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
          <td>26.719622</td>
          <td>0.469589</td>
          <td>26.888467</td>
          <td>0.207032</td>
          <td>26.006471</td>
          <td>0.086338</td>
          <td>25.175624</td>
          <td>0.067849</td>
          <td>24.667222</td>
          <td>0.082415</td>
          <td>24.006833</td>
          <td>0.103782</td>
          <td>0.085844</td>
          <td>0.079148</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.959636</td>
          <td>0.533780</td>
          <td>27.445456</td>
          <td>0.305763</td>
          <td>26.745113</td>
          <td>0.151436</td>
          <td>26.228523</td>
          <td>0.156336</td>
          <td>25.714190</td>
          <td>0.188240</td>
          <td>25.092211</td>
          <td>0.241297</td>
          <td>0.017760</td>
          <td>0.010737</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.094795</td>
          <td>1.156370</td>
          <td>29.755886</td>
          <td>1.480036</td>
          <td>27.467661</td>
          <td>0.295492</td>
          <td>25.926093</td>
          <td>0.129365</td>
          <td>25.054976</td>
          <td>0.114356</td>
          <td>24.268863</td>
          <td>0.128714</td>
          <td>0.095393</td>
          <td>0.051267</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.195574</td>
          <td>0.630548</td>
          <td>27.987579</td>
          <td>0.464823</td>
          <td>27.396841</td>
          <td>0.261028</td>
          <td>26.356971</td>
          <td>0.173953</td>
          <td>25.691557</td>
          <td>0.184194</td>
          <td>25.464780</td>
          <td>0.325524</td>
          <td>0.004861</td>
          <td>0.002595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.095992</td>
          <td>0.276232</td>
          <td>26.066700</td>
          <td>0.096318</td>
          <td>25.817502</td>
          <td>0.068175</td>
          <td>26.007288</td>
          <td>0.130953</td>
          <td>25.721743</td>
          <td>0.191802</td>
          <td>25.060885</td>
          <td>0.238116</td>
          <td>0.038685</td>
          <td>0.029540</td>
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
          <td>26.433554</td>
          <td>0.134846</td>
          <td>25.441007</td>
          <td>0.049810</td>
          <td>25.022037</td>
          <td>0.056228</td>
          <td>24.936069</td>
          <td>0.099398</td>
          <td>25.005059</td>
          <td>0.231830</td>
          <td>0.066367</td>
          <td>0.033539</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.586044</td>
          <td>0.469260</td>
          <td>26.763428</td>
          <td>0.212736</td>
          <td>26.232545</td>
          <td>0.122618</td>
          <td>25.158399</td>
          <td>0.078447</td>
          <td>24.678343</td>
          <td>0.097062</td>
          <td>24.342290</td>
          <td>0.162063</td>
          <td>0.156089</td>
          <td>0.145370</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.260509</td>
          <td>0.669511</td>
          <td>26.931753</td>
          <td>0.204622</td>
          <td>26.149001</td>
          <td>0.092418</td>
          <td>26.413671</td>
          <td>0.187613</td>
          <td>26.102977</td>
          <td>0.266109</td>
          <td>25.465818</td>
          <td>0.334227</td>
          <td>0.045754</td>
          <td>0.045460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.338948</td>
          <td>0.343205</td>
          <td>26.465136</td>
          <td>0.140409</td>
          <td>26.140326</td>
          <td>0.093911</td>
          <td>25.832245</td>
          <td>0.116678</td>
          <td>25.654231</td>
          <td>0.187418</td>
          <td>25.932708</td>
          <td>0.488646</td>
          <td>0.070489</td>
          <td>0.054690</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.451590</td>
          <td>0.363290</td>
          <td>26.765031</td>
          <td>0.173886</td>
          <td>26.731885</td>
          <td>0.149444</td>
          <td>26.190250</td>
          <td>0.150991</td>
          <td>26.286931</td>
          <td>0.301421</td>
          <td>25.903750</td>
          <td>0.457556</td>
          <td>0.009272</td>
          <td>0.007639</td>
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
