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

    <pzflow.flow.Flow at 0x7efd8ef731f0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.953016</td>
          <td>0.203606</td>
          <td>25.860765</td>
          <td>0.069714</td>
          <td>25.314096</td>
          <td>0.070140</td>
          <td>24.796186</td>
          <td>0.084816</td>
          <td>25.012808</td>
          <td>0.225301</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.117283</td>
          <td>0.596713</td>
          <td>27.860684</td>
          <td>0.422255</td>
          <td>27.791736</td>
          <td>0.358161</td>
          <td>27.043361</td>
          <td>0.306898</td>
          <td>28.221487</td>
          <td>1.155152</td>
          <td>26.207987</td>
          <td>0.571531</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.179345</td>
          <td>0.292507</td>
          <td>25.981732</td>
          <td>0.088176</td>
          <td>24.806802</td>
          <td>0.027440</td>
          <td>23.864112</td>
          <td>0.019625</td>
          <td>23.099623</td>
          <td>0.019120</td>
          <td>22.865464</td>
          <td>0.034626</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.361565</td>
          <td>2.078667</td>
          <td>28.351842</td>
          <td>0.605800</td>
          <td>27.426290</td>
          <td>0.267331</td>
          <td>26.423759</td>
          <td>0.184047</td>
          <td>26.017279</td>
          <td>0.241769</td>
          <td>25.604465</td>
          <td>0.363372</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.064164</td>
          <td>0.266448</td>
          <td>25.836199</td>
          <td>0.077572</td>
          <td>25.470499</td>
          <td>0.049313</td>
          <td>24.807617</td>
          <td>0.044750</td>
          <td>24.189054</td>
          <td>0.049541</td>
          <td>23.696124</td>
          <td>0.072303</td>
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
          <td>28.671516</td>
          <td>1.522273</td>
          <td>26.272507</td>
          <td>0.113719</td>
          <td>26.100326</td>
          <td>0.086140</td>
          <td>25.938412</td>
          <td>0.121356</td>
          <td>26.148187</td>
          <td>0.269163</td>
          <td>25.529256</td>
          <td>0.342519</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.733903</td>
          <td>0.900607</td>
          <td>27.394020</td>
          <td>0.292697</td>
          <td>26.680122</td>
          <td>0.142802</td>
          <td>26.570942</td>
          <td>0.208313</td>
          <td>26.123367</td>
          <td>0.263767</td>
          <td>25.026131</td>
          <td>0.227807</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.093976</td>
          <td>1.117906</td>
          <td>26.854780</td>
          <td>0.187458</td>
          <td>26.782103</td>
          <td>0.155870</td>
          <td>26.472823</td>
          <td>0.191832</td>
          <td>25.705588</td>
          <td>0.186355</td>
          <td>25.020414</td>
          <td>0.226729</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.189593</td>
          <td>0.247805</td>
          <td>26.616933</td>
          <td>0.135229</td>
          <td>25.670976</td>
          <td>0.096079</td>
          <td>25.773697</td>
          <td>0.197367</td>
          <td>25.573764</td>
          <td>0.354734</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.731974</td>
          <td>0.450248</td>
          <td>26.565841</td>
          <td>0.146563</td>
          <td>26.028013</td>
          <td>0.080820</td>
          <td>25.570905</td>
          <td>0.087990</td>
          <td>25.209425</td>
          <td>0.121775</td>
          <td>24.759768</td>
          <td>0.182215</td>
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
          <td>28.128443</td>
          <td>1.228714</td>
          <td>26.845967</td>
          <td>0.213306</td>
          <td>25.972324</td>
          <td>0.090497</td>
          <td>25.221088</td>
          <td>0.076556</td>
          <td>25.150148</td>
          <td>0.135637</td>
          <td>25.605236</td>
          <td>0.421051</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.674457</td>
          <td>0.414758</td>
          <td>27.509507</td>
          <td>0.331458</td>
          <td>27.311951</td>
          <td>0.439527</td>
          <td>27.732934</td>
          <td>0.965469</td>
          <td>26.371133</td>
          <td>0.730147</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.779226</td>
          <td>0.522315</td>
          <td>25.975570</td>
          <td>0.103229</td>
          <td>24.782337</td>
          <td>0.032304</td>
          <td>23.859821</td>
          <td>0.023595</td>
          <td>23.133434</td>
          <td>0.023559</td>
          <td>22.828137</td>
          <td>0.040593</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.071278</td>
          <td>0.538586</td>
          <td>26.338032</td>
          <td>0.214970</td>
          <td>27.938794</td>
          <td>1.140190</td>
          <td>25.312335</td>
          <td>0.356636</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.081708</td>
          <td>0.301166</td>
          <td>25.732353</td>
          <td>0.081757</td>
          <td>25.366376</td>
          <td>0.052974</td>
          <td>24.841287</td>
          <td>0.054707</td>
          <td>24.408797</td>
          <td>0.070899</td>
          <td>23.662341</td>
          <td>0.083048</td>
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
          <td>26.368835</td>
          <td>0.383116</td>
          <td>26.174422</td>
          <td>0.122589</td>
          <td>26.177143</td>
          <td>0.110569</td>
          <td>25.775958</td>
          <td>0.127183</td>
          <td>26.045223</td>
          <td>0.292961</td>
          <td>25.257452</td>
          <td>0.327305</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.527985</td>
          <td>0.428048</td>
          <td>27.204599</td>
          <td>0.287425</td>
          <td>26.634718</td>
          <td>0.161465</td>
          <td>26.525282</td>
          <td>0.236171</td>
          <td>25.703014</td>
          <td>0.217806</td>
          <td>25.593099</td>
          <td>0.418724</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.129748</td>
          <td>0.272450</td>
          <td>27.207898</td>
          <td>0.262988</td>
          <td>26.325265</td>
          <td>0.201617</td>
          <td>26.614925</td>
          <td>0.453732</td>
          <td>25.695668</td>
          <td>0.456031</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.911594</td>
          <td>1.104041</td>
          <td>27.048587</td>
          <td>0.259020</td>
          <td>26.733951</td>
          <td>0.180379</td>
          <td>25.841678</td>
          <td>0.136056</td>
          <td>25.760387</td>
          <td>0.234385</td>
          <td>25.337796</td>
          <td>0.352129</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.679884</td>
          <td>0.481645</td>
          <td>26.878024</td>
          <td>0.220968</td>
          <td>25.947847</td>
          <td>0.089466</td>
          <td>25.566858</td>
          <td>0.104834</td>
          <td>25.075059</td>
          <td>0.128373</td>
          <td>25.212348</td>
          <td>0.312541</td>
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
          <td>28.290933</td>
          <td>1.248573</td>
          <td>26.873840</td>
          <td>0.190517</td>
          <td>25.966970</td>
          <td>0.076590</td>
          <td>25.431996</td>
          <td>0.077858</td>
          <td>24.906703</td>
          <td>0.093488</td>
          <td>25.047603</td>
          <td>0.231929</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.995728</td>
          <td>1.608025</td>
          <td>28.930989</td>
          <td>0.814129</td>
          <td>26.758882</td>
          <td>0.243738</td>
          <td>26.449695</td>
          <td>0.343132</td>
          <td>26.400260</td>
          <td>0.654858</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.988888</td>
          <td>0.095357</td>
          <td>24.748772</td>
          <td>0.028311</td>
          <td>23.868767</td>
          <td>0.021419</td>
          <td>23.147381</td>
          <td>0.021563</td>
          <td>22.865186</td>
          <td>0.037712</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.476800</td>
          <td>0.770895</td>
          <td>27.521684</td>
          <td>0.354247</td>
          <td>26.539863</td>
          <td>0.253171</td>
          <td>25.924568</td>
          <td>0.276319</td>
          <td>24.570435</td>
          <td>0.194110</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.668512</td>
          <td>0.192168</td>
          <td>25.852031</td>
          <td>0.078760</td>
          <td>25.448583</td>
          <td>0.048433</td>
          <td>24.774079</td>
          <td>0.043504</td>
          <td>24.307359</td>
          <td>0.055107</td>
          <td>23.781446</td>
          <td>0.078082</td>
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
          <td>26.423467</td>
          <td>0.373250</td>
          <td>26.610785</td>
          <td>0.162902</td>
          <td>25.998188</td>
          <td>0.085201</td>
          <td>26.217596</td>
          <td>0.167292</td>
          <td>25.934510</td>
          <td>0.243106</td>
          <td>25.032064</td>
          <td>0.247174</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.135697</td>
          <td>0.240249</td>
          <td>26.872569</td>
          <td>0.171096</td>
          <td>26.276260</td>
          <td>0.165111</td>
          <td>25.883101</td>
          <td>0.219706</td>
          <td>25.766996</td>
          <td>0.418280</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.025928</td>
          <td>0.225332</td>
          <td>27.038507</td>
          <td>0.203080</td>
          <td>26.461854</td>
          <td>0.199628</td>
          <td>26.060705</td>
          <td>0.262246</td>
          <td>25.838110</td>
          <td>0.454437</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.537879</td>
          <td>0.157897</td>
          <td>26.589390</td>
          <td>0.147773</td>
          <td>25.913597</td>
          <td>0.133672</td>
          <td>25.558499</td>
          <td>0.183737</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.842595</td>
          <td>0.500365</td>
          <td>26.447070</td>
          <td>0.136795</td>
          <td>26.191575</td>
          <td>0.097041</td>
          <td>25.554459</td>
          <td>0.090345</td>
          <td>25.310528</td>
          <td>0.138102</td>
          <td>24.941062</td>
          <td>0.220520</td>
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
