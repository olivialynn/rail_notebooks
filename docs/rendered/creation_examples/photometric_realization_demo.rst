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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f713a9adcc0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.218408</td>
          <td>0.640585</td>
          <td>26.714770</td>
          <td>0.166474</td>
          <td>26.053967</td>
          <td>0.082692</td>
          <td>25.231459</td>
          <td>0.065189</td>
          <td>25.382247</td>
          <td>0.141412</td>
          <td>24.735889</td>
          <td>0.178566</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.517615</td>
          <td>0.615259</td>
          <td>28.207971</td>
          <td>0.727543</td>
          <td>26.174603</td>
          <td>0.275013</td>
          <td>26.036509</td>
          <td>0.504612</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.818684</td>
          <td>0.949187</td>
          <td>25.951986</td>
          <td>0.085900</td>
          <td>24.760132</td>
          <td>0.026346</td>
          <td>23.888342</td>
          <td>0.020032</td>
          <td>23.158161</td>
          <td>0.020089</td>
          <td>22.854415</td>
          <td>0.034290</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.076395</td>
          <td>0.579629</td>
          <td>28.618782</td>
          <td>0.728007</td>
          <td>27.428556</td>
          <td>0.267825</td>
          <td>26.456044</td>
          <td>0.189137</td>
          <td>26.267206</td>
          <td>0.296411</td>
          <td>25.066832</td>
          <td>0.235619</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.532159</td>
          <td>0.386545</td>
          <td>25.777864</td>
          <td>0.073681</td>
          <td>25.439219</td>
          <td>0.047963</td>
          <td>24.778236</td>
          <td>0.043599</td>
          <td>24.396315</td>
          <td>0.059548</td>
          <td>23.816282</td>
          <td>0.080401</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.762017</td>
          <td>0.460526</td>
          <td>26.187376</td>
          <td>0.105584</td>
          <td>26.076980</td>
          <td>0.084386</td>
          <td>25.823801</td>
          <td>0.109829</td>
          <td>25.968461</td>
          <td>0.232208</td>
          <td>25.525764</td>
          <td>0.341576</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.578487</td>
          <td>0.815790</td>
          <td>27.072454</td>
          <td>0.224941</td>
          <td>26.902991</td>
          <td>0.172803</td>
          <td>26.412769</td>
          <td>0.182344</td>
          <td>26.135501</td>
          <td>0.266393</td>
          <td>25.523983</td>
          <td>0.341096</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.732371</td>
          <td>1.568456</td>
          <td>27.049369</td>
          <td>0.220666</td>
          <td>26.947345</td>
          <td>0.179432</td>
          <td>26.316139</td>
          <td>0.167980</td>
          <td>26.491256</td>
          <td>0.354245</td>
          <td>26.206463</td>
          <td>0.570907</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.305132</td>
          <td>0.680062</td>
          <td>27.824400</td>
          <td>0.410705</td>
          <td>26.612012</td>
          <td>0.134655</td>
          <td>25.671668</td>
          <td>0.096137</td>
          <td>25.648165</td>
          <td>0.177513</td>
          <td>25.062533</td>
          <td>0.234783</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.437852</td>
          <td>0.359191</td>
          <td>26.482441</td>
          <td>0.136411</td>
          <td>26.207242</td>
          <td>0.094631</td>
          <td>25.483148</td>
          <td>0.081442</td>
          <td>25.352627</td>
          <td>0.137846</td>
          <td>24.689979</td>
          <td>0.171740</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>28.914257</td>
          <td>1.817146</td>
          <td>27.133086</td>
          <td>0.270287</td>
          <td>25.913310</td>
          <td>0.085919</td>
          <td>25.251940</td>
          <td>0.078670</td>
          <td>24.868611</td>
          <td>0.106209</td>
          <td>25.455254</td>
          <td>0.375089</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.324398</td>
          <td>0.753856</td>
          <td>28.771790</td>
          <td>0.893649</td>
          <td>27.923399</td>
          <td>0.456433</td>
          <td>27.271068</td>
          <td>0.426095</td>
          <td>26.333075</td>
          <td>0.361432</td>
          <td>25.997207</td>
          <td>0.563109</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.666908</td>
          <td>0.480836</td>
          <td>26.003568</td>
          <td>0.105783</td>
          <td>24.752740</td>
          <td>0.031474</td>
          <td>23.921015</td>
          <td>0.024876</td>
          <td>23.113134</td>
          <td>0.023151</td>
          <td>22.913096</td>
          <td>0.043765</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.834217</td>
          <td>0.559347</td>
          <td>29.354789</td>
          <td>1.305698</td>
          <td>27.631953</td>
          <td>0.387279</td>
          <td>26.606687</td>
          <td>0.268310</td>
          <td>25.865791</td>
          <td>0.264276</td>
          <td>24.827795</td>
          <td>0.241390</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>28.313735</td>
          <td>1.357790</td>
          <td>25.899734</td>
          <td>0.094704</td>
          <td>25.505289</td>
          <td>0.059921</td>
          <td>24.748829</td>
          <td>0.050398</td>
          <td>24.365541</td>
          <td>0.068237</td>
          <td>23.624673</td>
          <td>0.080336</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>25.767689</td>
          <td>0.236682</td>
          <td>26.300837</td>
          <td>0.136747</td>
          <td>26.039716</td>
          <td>0.098049</td>
          <td>26.220157</td>
          <td>0.186047</td>
          <td>25.548097</td>
          <td>0.194391</td>
          <td>25.741700</td>
          <td>0.475420</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.457738</td>
          <td>0.351698</td>
          <td>26.402682</td>
          <td>0.132274</td>
          <td>26.392164</td>
          <td>0.211431</td>
          <td>26.100151</td>
          <td>0.301527</td>
          <td>25.168601</td>
          <td>0.300114</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.910274</td>
          <td>1.092864</td>
          <td>27.102114</td>
          <td>0.266387</td>
          <td>27.081083</td>
          <td>0.236960</td>
          <td>26.513172</td>
          <td>0.235787</td>
          <td>25.795465</td>
          <td>0.237093</td>
          <td>27.031346</td>
          <td>1.112288</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.226643</td>
          <td>0.719116</td>
          <td>27.284666</td>
          <td>0.313509</td>
          <td>26.698852</td>
          <td>0.175090</td>
          <td>25.903360</td>
          <td>0.143484</td>
          <td>25.412401</td>
          <td>0.175068</td>
          <td>25.392209</td>
          <td>0.367464</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>30.117896</td>
          <td>2.882938</td>
          <td>26.335675</td>
          <td>0.139549</td>
          <td>26.060515</td>
          <td>0.098766</td>
          <td>25.390371</td>
          <td>0.089805</td>
          <td>25.220353</td>
          <td>0.145519</td>
          <td>24.858852</td>
          <td>0.234396</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>0.890625</td>
          <td>27.212638</td>
          <td>0.638066</td>
          <td>26.645009</td>
          <td>0.156869</td>
          <td>26.250940</td>
          <td>0.098342</td>
          <td>25.283184</td>
          <td>0.068256</td>
          <td>25.121269</td>
          <td>0.112798</td>
          <td>24.708707</td>
          <td>0.174518</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.851020</td>
          <td>0.968581</td>
          <td>27.701746</td>
          <td>0.373848</td>
          <td>27.654119</td>
          <td>0.321520</td>
          <td>27.095200</td>
          <td>0.320172</td>
          <td>26.517175</td>
          <td>0.361823</td>
          <td>26.073080</td>
          <td>0.518770</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.090624</td>
          <td>0.287351</td>
          <td>25.883319</td>
          <td>0.086919</td>
          <td>24.764547</td>
          <td>0.028704</td>
          <td>23.865718</td>
          <td>0.021363</td>
          <td>23.118901</td>
          <td>0.021045</td>
          <td>22.797207</td>
          <td>0.035514</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.674009</td>
          <td>0.875577</td>
          <td>27.303578</td>
          <td>0.297841</td>
          <td>26.426682</td>
          <td>0.230610</td>
          <td>25.747572</td>
          <td>0.239030</td>
          <td>24.979689</td>
          <td>0.272446</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.821914</td>
          <td>0.481980</td>
          <td>25.807825</td>
          <td>0.075749</td>
          <td>25.482283</td>
          <td>0.049904</td>
          <td>24.826503</td>
          <td>0.045576</td>
          <td>24.418034</td>
          <td>0.060793</td>
          <td>23.561858</td>
          <td>0.064295</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>2.147172</td>
          <td>26.599834</td>
          <td>0.427483</td>
          <td>26.389807</td>
          <td>0.134766</td>
          <td>26.038821</td>
          <td>0.088304</td>
          <td>26.069415</td>
          <td>0.147371</td>
          <td>25.560530</td>
          <td>0.177792</td>
          <td>25.337427</td>
          <td>0.316620</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.753972</td>
          <td>0.393956</td>
          <td>26.928901</td>
          <td>0.179478</td>
          <td>26.294189</td>
          <td>0.167654</td>
          <td>26.168817</td>
          <td>0.277917</td>
          <td>26.241765</td>
          <td>0.593627</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.315613</td>
          <td>0.285736</td>
          <td>26.900377</td>
          <td>0.180759</td>
          <td>26.593151</td>
          <td>0.222785</td>
          <td>25.874136</td>
          <td>0.224866</td>
          <td>24.967277</td>
          <td>0.227595</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.726143</td>
          <td>0.949713</td>
          <td>26.970487</td>
          <td>0.227353</td>
          <td>26.632508</td>
          <td>0.153342</td>
          <td>25.749262</td>
          <td>0.115914</td>
          <td>25.760809</td>
          <td>0.217759</td>
          <td>25.406295</td>
          <td>0.345783</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.055122</td>
          <td>1.112866</td>
          <td>26.528328</td>
          <td>0.146703</td>
          <td>26.265350</td>
          <td>0.103520</td>
          <td>25.537097</td>
          <td>0.088976</td>
          <td>25.314881</td>
          <td>0.138622</td>
          <td>24.592466</td>
          <td>0.164364</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
