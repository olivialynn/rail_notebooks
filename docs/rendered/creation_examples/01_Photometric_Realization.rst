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

    <pzflow.flow.Flow at 0x7ff974eedc60>



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
    0      23.994413  0.085524  0.043456  
    1      25.391064  0.017812  0.015545  
    2      24.304707  0.074624  0.047490  
    3      25.291103  0.057055  0.043926  
    4      25.096743  0.024068  0.012510  
    ...          ...       ...       ...  
    99995  24.737946  0.107191  0.085472  
    99996  24.224169  0.068003  0.065315  
    99997  25.613836  0.075038  0.038283  
    99998  25.274899  0.045753  0.045502  
    99999  25.699642  0.046200  0.025062  
    
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
          <td>26.742255</td>
          <td>0.170413</td>
          <td>26.156961</td>
          <td>0.090541</td>
          <td>25.160706</td>
          <td>0.061225</td>
          <td>24.619846</td>
          <td>0.072589</td>
          <td>24.166619</td>
          <td>0.109355</td>
          <td>0.085524</td>
          <td>0.043456</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.634432</td>
          <td>0.354424</td>
          <td>26.539972</td>
          <td>0.126517</td>
          <td>26.685886</td>
          <td>0.229249</td>
          <td>25.686577</td>
          <td>0.183384</td>
          <td>25.401987</td>
          <td>0.309554</td>
          <td>0.017812</td>
          <td>0.015545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.723140</td>
          <td>0.379843</td>
          <td>28.538097</td>
          <td>0.624164</td>
          <td>25.802174</td>
          <td>0.107775</td>
          <td>24.996864</td>
          <td>0.101167</td>
          <td>24.171858</td>
          <td>0.109856</td>
          <td>0.074624</td>
          <td>0.047490</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.255756</td>
          <td>0.232364</td>
          <td>26.276406</td>
          <td>0.162383</td>
          <td>25.185925</td>
          <td>0.119314</td>
          <td>25.587190</td>
          <td>0.358490</td>
          <td>0.057055</td>
          <td>0.043926</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.244285</td>
          <td>0.308162</td>
          <td>26.256227</td>
          <td>0.112119</td>
          <td>25.744249</td>
          <td>0.062876</td>
          <td>25.748245</td>
          <td>0.102810</td>
          <td>25.499792</td>
          <td>0.156430</td>
          <td>24.923053</td>
          <td>0.209058</td>
          <td>0.024068</td>
          <td>0.012510</td>
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
          <td>27.799473</td>
          <td>0.938037</td>
          <td>26.278895</td>
          <td>0.114353</td>
          <td>25.470583</td>
          <td>0.049317</td>
          <td>25.059860</td>
          <td>0.055984</td>
          <td>24.774861</td>
          <td>0.083237</td>
          <td>24.640682</td>
          <td>0.164679</td>
          <td>0.107191</td>
          <td>0.085472</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.553379</td>
          <td>0.802606</td>
          <td>26.843051</td>
          <td>0.185611</td>
          <td>25.983954</td>
          <td>0.077737</td>
          <td>25.227484</td>
          <td>0.064960</td>
          <td>24.974866</td>
          <td>0.099236</td>
          <td>24.262883</td>
          <td>0.118922</td>
          <td>0.068003</td>
          <td>0.065315</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.025879</td>
          <td>0.216391</td>
          <td>26.268500</td>
          <td>0.099855</td>
          <td>26.177573</td>
          <td>0.149207</td>
          <td>26.611790</td>
          <td>0.389144</td>
          <td>26.246288</td>
          <td>0.587365</td>
          <td>0.075038</td>
          <td>0.038283</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.306894</td>
          <td>0.323938</td>
          <td>26.151038</td>
          <td>0.102284</td>
          <td>26.075945</td>
          <td>0.084309</td>
          <td>25.845855</td>
          <td>0.111963</td>
          <td>25.446345</td>
          <td>0.149425</td>
          <td>25.189727</td>
          <td>0.260682</td>
          <td>0.045753</td>
          <td>0.045502</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.849746</td>
          <td>0.967390</td>
          <td>26.770644</td>
          <td>0.174573</td>
          <td>26.285391</td>
          <td>0.101343</td>
          <td>26.268073</td>
          <td>0.161232</td>
          <td>25.850953</td>
          <td>0.210575</td>
          <td>26.434517</td>
          <td>0.669988</td>
          <td>0.046200</td>
          <td>0.025062</td>
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
          <td>27.320561</td>
          <td>0.758507</td>
          <td>26.938723</td>
          <td>0.233429</td>
          <td>25.963732</td>
          <td>0.091205</td>
          <td>25.130440</td>
          <td>0.071801</td>
          <td>24.668117</td>
          <td>0.090467</td>
          <td>24.055951</td>
          <td>0.119039</td>
          <td>0.085524</td>
          <td>0.043456</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.537318</td>
          <td>0.373312</td>
          <td>26.510303</td>
          <td>0.144675</td>
          <td>26.349454</td>
          <td>0.203352</td>
          <td>25.873660</td>
          <td>0.250077</td>
          <td>25.572000</td>
          <td>0.410823</td>
          <td>0.017812</td>
          <td>0.015545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.746254</td>
          <td>0.991904</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.727163</td>
          <td>1.446095</td>
          <td>25.996294</td>
          <td>0.152553</td>
          <td>24.986030</td>
          <td>0.119190</td>
          <td>24.291756</td>
          <td>0.145629</td>
          <td>0.074624</td>
          <td>0.047490</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.052491</td>
          <td>1.182856</td>
          <td>27.971462</td>
          <td>0.521374</td>
          <td>27.417950</td>
          <td>0.310527</td>
          <td>26.284232</td>
          <td>0.193999</td>
          <td>25.766231</td>
          <td>0.230543</td>
          <td>25.825089</td>
          <td>0.500398</td>
          <td>0.057055</td>
          <td>0.043926</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.783652</td>
          <td>0.517025</td>
          <td>25.975044</td>
          <td>0.101239</td>
          <td>25.945481</td>
          <td>0.088494</td>
          <td>25.766219</td>
          <td>0.123591</td>
          <td>25.896589</td>
          <td>0.254900</td>
          <td>24.982924</td>
          <td>0.257408</td>
          <td>0.024068</td>
          <td>0.012510</td>
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
          <td>27.199956</td>
          <td>0.706084</td>
          <td>26.537917</td>
          <td>0.169011</td>
          <td>25.457594</td>
          <td>0.059247</td>
          <td>25.072548</td>
          <td>0.069336</td>
          <td>24.814251</td>
          <td>0.104447</td>
          <td>24.844945</td>
          <td>0.236440</td>
          <td>0.107191</td>
          <td>0.085472</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.418812</td>
          <td>0.808767</td>
          <td>26.647323</td>
          <td>0.182857</td>
          <td>26.049135</td>
          <td>0.098261</td>
          <td>25.208039</td>
          <td>0.076862</td>
          <td>24.929622</td>
          <td>0.113683</td>
          <td>24.316515</td>
          <td>0.149028</td>
          <td>0.068003</td>
          <td>0.065315</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.230014</td>
          <td>1.305709</td>
          <td>26.589209</td>
          <td>0.173615</td>
          <td>26.224252</td>
          <td>0.114163</td>
          <td>26.036256</td>
          <td>0.157668</td>
          <td>25.833247</td>
          <td>0.244388</td>
          <td>25.156569</td>
          <td>0.299381</td>
          <td>0.075038</td>
          <td>0.038283</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.560617</td>
          <td>0.439588</td>
          <td>26.208449</td>
          <td>0.124691</td>
          <td>26.206644</td>
          <td>0.111885</td>
          <td>25.834033</td>
          <td>0.131846</td>
          <td>26.255313</td>
          <td>0.342073</td>
          <td>25.190899</td>
          <td>0.306351</td>
          <td>0.045753</td>
          <td>0.045502</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.166850</td>
          <td>0.679687</td>
          <td>26.985334</td>
          <td>0.240402</td>
          <td>26.457449</td>
          <td>0.138744</td>
          <td>26.206741</td>
          <td>0.180977</td>
          <td>25.665426</td>
          <td>0.211176</td>
          <td>26.565850</td>
          <td>0.832587</td>
          <td>0.046200</td>
          <td>0.025062</td>
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
          <td>27.459136</td>
          <td>0.777919</td>
          <td>26.739138</td>
          <td>0.178462</td>
          <td>26.272587</td>
          <td>0.106142</td>
          <td>25.164876</td>
          <td>0.065319</td>
          <td>24.703096</td>
          <td>0.082790</td>
          <td>24.125959</td>
          <td>0.112002</td>
          <td>0.085524</td>
          <td>0.043456</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.534593</td>
          <td>0.388169</td>
          <td>27.209938</td>
          <td>0.252754</td>
          <td>26.394083</td>
          <td>0.111862</td>
          <td>26.064328</td>
          <td>0.135872</td>
          <td>25.690900</td>
          <td>0.184726</td>
          <td>25.155106</td>
          <td>0.254324</td>
          <td>0.017812</td>
          <td>0.015545</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.857151</td>
          <td>0.509135</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.250716</td>
          <td>1.028190</td>
          <td>25.863858</td>
          <td>0.119792</td>
          <td>25.067356</td>
          <td>0.113075</td>
          <td>24.268072</td>
          <td>0.125743</td>
          <td>0.074624</td>
          <td>0.047490</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.476094</td>
          <td>0.377944</td>
          <td>28.881485</td>
          <td>0.882810</td>
          <td>27.330351</td>
          <td>0.255084</td>
          <td>26.519889</td>
          <td>0.206477</td>
          <td>25.330816</td>
          <td>0.139855</td>
          <td>26.106353</td>
          <td>0.546756</td>
          <td>0.057055</td>
          <td>0.043926</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.956518</td>
          <td>0.533199</td>
          <td>26.210119</td>
          <td>0.108164</td>
          <td>25.957866</td>
          <td>0.076347</td>
          <td>25.645772</td>
          <td>0.094468</td>
          <td>25.628465</td>
          <td>0.175408</td>
          <td>25.112593</td>
          <td>0.245875</td>
          <td>0.024068</td>
          <td>0.012510</td>
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
          <td>27.940868</td>
          <td>1.078961</td>
          <td>26.206045</td>
          <td>0.118420</td>
          <td>25.495186</td>
          <td>0.056502</td>
          <td>25.108328</td>
          <td>0.065834</td>
          <td>24.815265</td>
          <td>0.096546</td>
          <td>24.457444</td>
          <td>0.157730</td>
          <td>0.107191</td>
          <td>0.085472</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.384610</td>
          <td>0.131643</td>
          <td>26.135892</td>
          <td>0.094090</td>
          <td>25.128228</td>
          <td>0.063183</td>
          <td>24.857397</td>
          <td>0.094752</td>
          <td>24.326103</td>
          <td>0.133158</td>
          <td>0.068003</td>
          <td>0.065315</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.171359</td>
          <td>0.299255</td>
          <td>26.903972</td>
          <td>0.202919</td>
          <td>26.496390</td>
          <td>0.127391</td>
          <td>26.090624</td>
          <td>0.145036</td>
          <td>25.825174</td>
          <td>0.215138</td>
          <td>25.873891</td>
          <td>0.465296</td>
          <td>0.075038</td>
          <td>0.038283</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.154451</td>
          <td>0.291792</td>
          <td>26.435275</td>
          <td>0.134087</td>
          <td>26.353805</td>
          <td>0.110574</td>
          <td>25.998929</td>
          <td>0.131586</td>
          <td>25.612596</td>
          <td>0.176865</td>
          <td>24.654013</td>
          <td>0.171234</td>
          <td>0.045753</td>
          <td>0.045502</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.821458</td>
          <td>1.648486</td>
          <td>27.132770</td>
          <td>0.240031</td>
          <td>26.719323</td>
          <td>0.150367</td>
          <td>26.464368</td>
          <td>0.194013</td>
          <td>26.176794</td>
          <td>0.280198</td>
          <td>25.800543</td>
          <td>0.429824</td>
          <td>0.046200</td>
          <td>0.025062</td>
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
