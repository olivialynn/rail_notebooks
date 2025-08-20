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

    <pzflow.flow.Flow at 0x7f1966be6ef0>



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
    0      23.994413  0.225450  0.198391  
    1      25.391064  0.076859  0.063297  
    2      24.304707  0.059973  0.045407  
    3      25.291103  0.039289  0.028768  
    4      25.096743  0.067058  0.065382  
    ...          ...       ...       ...  
    99995  24.737946  0.012191  0.010724  
    99996  24.224169  0.048967  0.030811  
    99997  25.613836  0.048272  0.029316  
    99998  25.274899  0.077103  0.054582  
    99999  25.699642  0.045845  0.028671  
    
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
          <td>27.293269</td>
          <td>0.674560</td>
          <td>26.611623</td>
          <td>0.152433</td>
          <td>26.096787</td>
          <td>0.085872</td>
          <td>25.066560</td>
          <td>0.056318</td>
          <td>24.700132</td>
          <td>0.077926</td>
          <td>24.000927</td>
          <td>0.094589</td>
          <td>0.225450</td>
          <td>0.198391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.855062</td>
          <td>0.493557</td>
          <td>27.254069</td>
          <td>0.261256</td>
          <td>26.577083</td>
          <td>0.130649</td>
          <td>26.118546</td>
          <td>0.141822</td>
          <td>26.654336</td>
          <td>0.402129</td>
          <td>24.995369</td>
          <td>0.222059</td>
          <td>0.076859</td>
          <td>0.063297</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.253769</td>
          <td>1.084082</td>
          <td>29.235885</td>
          <td>0.984890</td>
          <td>25.934162</td>
          <td>0.120909</td>
          <td>25.064136</td>
          <td>0.107298</td>
          <td>24.220132</td>
          <td>0.114578</td>
          <td>0.059973</td>
          <td>0.045407</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.026812</td>
          <td>0.191898</td>
          <td>26.021010</td>
          <td>0.130367</td>
          <td>25.659455</td>
          <td>0.179220</td>
          <td>25.204011</td>
          <td>0.263744</td>
          <td>0.039289</td>
          <td>0.028768</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.586704</td>
          <td>0.403147</td>
          <td>26.156163</td>
          <td>0.102744</td>
          <td>25.931843</td>
          <td>0.074239</td>
          <td>25.580900</td>
          <td>0.088767</td>
          <td>25.280370</td>
          <td>0.129502</td>
          <td>24.793159</td>
          <td>0.187433</td>
          <td>0.067058</td>
          <td>0.065382</td>
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
          <td>26.323917</td>
          <td>0.118919</td>
          <td>25.432431</td>
          <td>0.047674</td>
          <td>25.038210</td>
          <td>0.054918</td>
          <td>24.752753</td>
          <td>0.081630</td>
          <td>24.604926</td>
          <td>0.159728</td>
          <td>0.012191</td>
          <td>0.010724</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.804824</td>
          <td>0.179706</td>
          <td>26.053303</td>
          <td>0.082643</td>
          <td>25.299524</td>
          <td>0.069241</td>
          <td>24.797102</td>
          <td>0.084884</td>
          <td>24.264160</td>
          <td>0.119054</td>
          <td>0.048967</td>
          <td>0.030811</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.318319</td>
          <td>0.326891</td>
          <td>26.711867</td>
          <td>0.166063</td>
          <td>26.335045</td>
          <td>0.105843</td>
          <td>26.279504</td>
          <td>0.162813</td>
          <td>25.580333</td>
          <td>0.167567</td>
          <td>25.928362</td>
          <td>0.465675</td>
          <td>0.048272</td>
          <td>0.029316</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.114648</td>
          <td>0.099079</td>
          <td>26.034213</td>
          <td>0.081263</td>
          <td>25.782007</td>
          <td>0.105892</td>
          <td>25.522056</td>
          <td>0.159438</td>
          <td>24.810453</td>
          <td>0.190188</td>
          <td>0.077103</td>
          <td>0.054582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.726802</td>
          <td>0.448497</td>
          <td>26.697053</td>
          <td>0.163980</td>
          <td>26.618253</td>
          <td>0.135383</td>
          <td>26.359139</td>
          <td>0.174237</td>
          <td>25.893968</td>
          <td>0.218273</td>
          <td>25.500858</td>
          <td>0.334914</td>
          <td>0.045845</td>
          <td>0.028671</td>
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
          <td>26.546909</td>
          <td>0.473456</td>
          <td>26.605494</td>
          <td>0.195806</td>
          <td>26.048743</td>
          <td>0.110494</td>
          <td>25.194823</td>
          <td>0.085859</td>
          <td>24.554531</td>
          <td>0.092111</td>
          <td>24.180653</td>
          <td>0.149267</td>
          <td>0.225450</td>
          <td>0.198391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.808189</td>
          <td>0.464809</td>
          <td>26.609785</td>
          <td>0.159979</td>
          <td>26.027062</td>
          <td>0.157176</td>
          <td>25.557418</td>
          <td>0.195093</td>
          <td>25.419507</td>
          <td>0.370330</td>
          <td>0.076859</td>
          <td>0.063297</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.934955</td>
          <td>0.993855</td>
          <td>28.435454</td>
          <td>0.665276</td>
          <td>26.240384</td>
          <td>0.187104</td>
          <td>24.898990</td>
          <td>0.110099</td>
          <td>24.250798</td>
          <td>0.140080</td>
          <td>0.059973</td>
          <td>0.045407</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.477022</td>
          <td>0.834633</td>
          <td>27.870517</td>
          <td>0.482261</td>
          <td>27.340749</td>
          <td>0.290583</td>
          <td>26.224194</td>
          <td>0.183550</td>
          <td>25.561743</td>
          <td>0.193461</td>
          <td>25.809856</td>
          <td>0.492782</td>
          <td>0.039289</td>
          <td>0.028768</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.977585</td>
          <td>0.279719</td>
          <td>26.138316</td>
          <td>0.118139</td>
          <td>25.928133</td>
          <td>0.088339</td>
          <td>25.699172</td>
          <td>0.118218</td>
          <td>25.268698</td>
          <td>0.152376</td>
          <td>25.770906</td>
          <td>0.483212</td>
          <td>0.067058</td>
          <td>0.065382</td>
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
          <td>26.117872</td>
          <td>0.114586</td>
          <td>25.410889</td>
          <td>0.055113</td>
          <td>25.067573</td>
          <td>0.066867</td>
          <td>24.762958</td>
          <td>0.096868</td>
          <td>24.556510</td>
          <td>0.180256</td>
          <td>0.012191</td>
          <td>0.010724</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.665547</td>
          <td>0.475184</td>
          <td>26.893167</td>
          <td>0.222927</td>
          <td>26.002173</td>
          <td>0.093427</td>
          <td>25.121443</td>
          <td>0.070515</td>
          <td>24.786923</td>
          <td>0.099440</td>
          <td>24.246695</td>
          <td>0.139040</td>
          <td>0.048967</td>
          <td>0.030811</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.854813</td>
          <td>0.545918</td>
          <td>27.605579</td>
          <td>0.394988</td>
          <td>26.244081</td>
          <td>0.115405</td>
          <td>26.240430</td>
          <td>0.186341</td>
          <td>25.953224</td>
          <td>0.268012</td>
          <td>25.378991</td>
          <td>0.355121</td>
          <td>0.048272</td>
          <td>0.029316</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.951990</td>
          <td>1.120700</td>
          <td>25.997394</td>
          <td>0.104516</td>
          <td>25.944948</td>
          <td>0.089676</td>
          <td>25.794162</td>
          <td>0.128406</td>
          <td>25.700818</td>
          <td>0.219632</td>
          <td>26.090886</td>
          <td>0.609329</td>
          <td>0.077103</td>
          <td>0.054582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.983541</td>
          <td>1.877252</td>
          <td>27.046230</td>
          <td>0.252808</td>
          <td>26.133597</td>
          <td>0.104754</td>
          <td>26.203499</td>
          <td>0.180530</td>
          <td>26.218278</td>
          <td>0.331570</td>
          <td>25.719254</td>
          <td>0.460988</td>
          <td>0.045845</td>
          <td>0.028671</td>
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
          <td>26.302762</td>
          <td>0.422291</td>
          <td>26.436653</td>
          <td>0.185482</td>
          <td>26.177328</td>
          <td>0.136178</td>
          <td>25.174320</td>
          <td>0.093300</td>
          <td>24.847096</td>
          <td>0.131097</td>
          <td>23.940697</td>
          <td>0.133977</td>
          <td>0.225450</td>
          <td>0.198391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.565566</td>
          <td>0.835799</td>
          <td>27.095584</td>
          <td>0.241477</td>
          <td>26.694690</td>
          <td>0.153773</td>
          <td>26.259369</td>
          <td>0.170601</td>
          <td>25.867365</td>
          <td>0.226578</td>
          <td>25.648416</td>
          <td>0.398281</td>
          <td>0.076859</td>
          <td>0.063297</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.486439</td>
          <td>1.407042</td>
          <td>30.157618</td>
          <td>1.763653</td>
          <td>28.062663</td>
          <td>0.455498</td>
          <td>25.840789</td>
          <td>0.115831</td>
          <td>25.056088</td>
          <td>0.110524</td>
          <td>24.185392</td>
          <td>0.115467</td>
          <td>0.059973</td>
          <td>0.045407</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.395402</td>
          <td>0.631499</td>
          <td>27.455268</td>
          <td>0.277750</td>
          <td>26.186827</td>
          <td>0.152856</td>
          <td>25.427137</td>
          <td>0.149270</td>
          <td>25.993059</td>
          <td>0.495482</td>
          <td>0.039289</td>
          <td>0.028768</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.353070</td>
          <td>0.348166</td>
          <td>26.165808</td>
          <td>0.108799</td>
          <td>25.718584</td>
          <td>0.065054</td>
          <td>25.657428</td>
          <td>0.100692</td>
          <td>25.571109</td>
          <td>0.175538</td>
          <td>25.255836</td>
          <td>0.290302</td>
          <td>0.067058</td>
          <td>0.065382</td>
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
          <td>26.941554</td>
          <td>0.526459</td>
          <td>26.415045</td>
          <td>0.128892</td>
          <td>25.544733</td>
          <td>0.052769</td>
          <td>24.929584</td>
          <td>0.049963</td>
          <td>24.963439</td>
          <td>0.098422</td>
          <td>24.265436</td>
          <td>0.119406</td>
          <td>0.012191</td>
          <td>0.010724</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.503367</td>
          <td>0.785781</td>
          <td>26.781336</td>
          <td>0.179467</td>
          <td>26.019228</td>
          <td>0.082001</td>
          <td>25.331174</td>
          <td>0.072895</td>
          <td>24.854923</td>
          <td>0.091317</td>
          <td>24.439500</td>
          <td>0.141738</td>
          <td>0.048967</td>
          <td>0.030811</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.855798</td>
          <td>0.499971</td>
          <td>26.892279</td>
          <td>0.196910</td>
          <td>26.330381</td>
          <td>0.107650</td>
          <td>26.011080</td>
          <td>0.132102</td>
          <td>26.610291</td>
          <td>0.395961</td>
          <td>25.049247</td>
          <td>0.237056</td>
          <td>0.048272</td>
          <td>0.029316</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.445750</td>
          <td>0.374501</td>
          <td>26.305883</td>
          <td>0.122961</td>
          <td>26.112037</td>
          <td>0.092128</td>
          <td>26.018749</td>
          <td>0.137949</td>
          <td>25.936936</td>
          <td>0.238632</td>
          <td>25.324269</td>
          <td>0.306901</td>
          <td>0.077103</td>
          <td>0.054582</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.939355</td>
          <td>0.530993</td>
          <td>26.471396</td>
          <td>0.137375</td>
          <td>26.339317</td>
          <td>0.108309</td>
          <td>26.063480</td>
          <td>0.137976</td>
          <td>26.135666</td>
          <td>0.271259</td>
          <td>25.991867</td>
          <td>0.496589</td>
          <td>0.045845</td>
          <td>0.028671</td>
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
