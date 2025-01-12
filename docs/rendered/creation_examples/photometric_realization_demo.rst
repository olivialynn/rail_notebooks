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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f730b7fb7c0>



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
          <td>27.795082</td>
          <td>0.935501</td>
          <td>26.906164</td>
          <td>0.195752</td>
          <td>26.032907</td>
          <td>0.081170</td>
          <td>25.262594</td>
          <td>0.067013</td>
          <td>25.046723</td>
          <td>0.105678</td>
          <td>25.009367</td>
          <td>0.224658</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.506708</td>
          <td>0.778485</td>
          <td>27.854607</td>
          <td>0.420302</td>
          <td>27.548197</td>
          <td>0.295104</td>
          <td>26.969652</td>
          <td>0.289222</td>
          <td>26.268809</td>
          <td>0.296793</td>
          <td>26.345081</td>
          <td>0.629728</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.560524</td>
          <td>0.395106</td>
          <td>25.754529</td>
          <td>0.072178</td>
          <td>24.770183</td>
          <td>0.026578</td>
          <td>23.850794</td>
          <td>0.019406</td>
          <td>23.134652</td>
          <td>0.019694</td>
          <td>22.787112</td>
          <td>0.032315</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.951765</td>
          <td>0.903372</td>
          <td>27.826853</td>
          <td>0.368137</td>
          <td>26.595918</td>
          <td>0.212709</td>
          <td>25.875674</td>
          <td>0.214969</td>
          <td>25.232344</td>
          <td>0.269911</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.547897</td>
          <td>0.391276</td>
          <td>25.992573</td>
          <td>0.089020</td>
          <td>25.395442</td>
          <td>0.046134</td>
          <td>24.781718</td>
          <td>0.043734</td>
          <td>24.297514</td>
          <td>0.054549</td>
          <td>23.731539</td>
          <td>0.074603</td>
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
          <td>28.376459</td>
          <td>1.307650</td>
          <td>26.313332</td>
          <td>0.117830</td>
          <td>26.040527</td>
          <td>0.081717</td>
          <td>26.111201</td>
          <td>0.140927</td>
          <td>26.321856</td>
          <td>0.309708</td>
          <td>25.694461</td>
          <td>0.389722</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.375457</td>
          <td>0.342007</td>
          <td>26.896055</td>
          <td>0.194094</td>
          <td>27.421765</td>
          <td>0.266346</td>
          <td>26.453509</td>
          <td>0.188732</td>
          <td>26.191900</td>
          <td>0.278903</td>
          <td>25.641928</td>
          <td>0.374153</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.730543</td>
          <td>0.449763</td>
          <td>27.001663</td>
          <td>0.212063</td>
          <td>26.619328</td>
          <td>0.135509</td>
          <td>26.466512</td>
          <td>0.190814</td>
          <td>25.955949</td>
          <td>0.229813</td>
          <td>25.556102</td>
          <td>0.349843</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.206331</td>
          <td>0.635224</td>
          <td>27.696692</td>
          <td>0.372109</td>
          <td>26.623832</td>
          <td>0.136037</td>
          <td>25.838303</td>
          <td>0.111228</td>
          <td>25.566806</td>
          <td>0.165647</td>
          <td>25.538452</td>
          <td>0.345013</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.030549</td>
          <td>0.560916</td>
          <td>26.733936</td>
          <td>0.169212</td>
          <td>26.193341</td>
          <td>0.093483</td>
          <td>25.764238</td>
          <td>0.104259</td>
          <td>25.128326</td>
          <td>0.113479</td>
          <td>24.510965</td>
          <td>0.147370</td>
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
          <td>32.249557</td>
          <td>4.938301</td>
          <td>26.460194</td>
          <td>0.153925</td>
          <td>26.019914</td>
          <td>0.094360</td>
          <td>25.189803</td>
          <td>0.074469</td>
          <td>25.033899</td>
          <td>0.122652</td>
          <td>24.712749</td>
          <td>0.205530</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.624598</td>
          <td>0.399188</td>
          <td>27.791162</td>
          <td>0.412862</td>
          <td>26.970654</td>
          <td>0.337433</td>
          <td>26.229426</td>
          <td>0.333095</td>
          <td>30.023137</td>
          <td>3.644902</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.963484</td>
          <td>0.596325</td>
          <td>25.831510</td>
          <td>0.090996</td>
          <td>24.826725</td>
          <td>0.033591</td>
          <td>23.877467</td>
          <td>0.023957</td>
          <td>23.145052</td>
          <td>0.023796</td>
          <td>22.847392</td>
          <td>0.041290</td>
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
          <td>28.118508</td>
          <td>0.557294</td>
          <td>26.645416</td>
          <td>0.276899</td>
          <td>25.991281</td>
          <td>0.292610</td>
          <td>25.558653</td>
          <td>0.431358</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.082222</td>
          <td>0.301290</td>
          <td>25.861579</td>
          <td>0.091589</td>
          <td>25.417552</td>
          <td>0.055435</td>
          <td>24.806268</td>
          <td>0.053033</td>
          <td>24.346192</td>
          <td>0.067078</td>
          <td>23.624411</td>
          <td>0.080317</td>
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
          <td>26.538620</td>
          <td>0.436331</td>
          <td>26.285546</td>
          <td>0.134956</td>
          <td>25.954101</td>
          <td>0.090952</td>
          <td>26.050644</td>
          <td>0.161093</td>
          <td>26.606791</td>
          <td>0.454141</td>
          <td>25.482403</td>
          <td>0.390431</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.559139</td>
          <td>0.879509</td>
          <td>27.594699</td>
          <td>0.391314</td>
          <td>26.919534</td>
          <td>0.205468</td>
          <td>26.493133</td>
          <td>0.229968</td>
          <td>25.994098</td>
          <td>0.276769</td>
          <td>25.443078</td>
          <td>0.372963</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.542619</td>
          <td>0.874498</td>
          <td>28.076686</td>
          <td>0.564377</td>
          <td>27.102241</td>
          <td>0.241136</td>
          <td>26.516288</td>
          <td>0.236395</td>
          <td>25.652654</td>
          <td>0.210556</td>
          <td>24.983027</td>
          <td>0.260294</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.660773</td>
          <td>0.481482</td>
          <td>28.479985</td>
          <td>0.755716</td>
          <td>26.632144</td>
          <td>0.165431</td>
          <td>25.658872</td>
          <td>0.116118</td>
          <td>25.696195</td>
          <td>0.222231</td>
          <td>25.741464</td>
          <td>0.479662</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.040257</td>
          <td>0.624706</td>
          <td>26.420615</td>
          <td>0.150114</td>
          <td>26.088411</td>
          <td>0.101209</td>
          <td>25.698808</td>
          <td>0.117619</td>
          <td>25.206131</td>
          <td>0.143750</td>
          <td>24.680010</td>
          <td>0.201943</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.770904</td>
          <td>0.174631</td>
          <td>25.893496</td>
          <td>0.071773</td>
          <td>25.420722</td>
          <td>0.077086</td>
          <td>25.033010</td>
          <td>0.104432</td>
          <td>24.713508</td>
          <td>0.175231</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.671081</td>
          <td>2.348372</td>
          <td>28.323131</td>
          <td>0.594014</td>
          <td>27.922687</td>
          <td>0.396886</td>
          <td>27.336363</td>
          <td>0.386991</td>
          <td>25.960957</td>
          <td>0.230975</td>
          <td>26.092080</td>
          <td>0.526023</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.082507</td>
          <td>0.285474</td>
          <td>25.891298</td>
          <td>0.087530</td>
          <td>24.789494</td>
          <td>0.029338</td>
          <td>23.871805</td>
          <td>0.021474</td>
          <td>23.117441</td>
          <td>0.021019</td>
          <td>22.870319</td>
          <td>0.037884</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.152096</td>
          <td>0.697034</td>
          <td>28.505514</td>
          <td>0.785581</td>
          <td>27.928845</td>
          <td>0.483649</td>
          <td>26.541053</td>
          <td>0.253418</td>
          <td>26.259945</td>
          <td>0.361118</td>
          <td>25.503627</td>
          <td>0.412317</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.265855</td>
          <td>0.313800</td>
          <td>25.747309</td>
          <td>0.071809</td>
          <td>25.387883</td>
          <td>0.045892</td>
          <td>24.813908</td>
          <td>0.045069</td>
          <td>24.359195</td>
          <td>0.057701</td>
          <td>23.878421</td>
          <td>0.085054</td>
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
          <td>27.020401</td>
          <td>0.582752</td>
          <td>26.269141</td>
          <td>0.121403</td>
          <td>26.117719</td>
          <td>0.094644</td>
          <td>25.900693</td>
          <td>0.127404</td>
          <td>25.468372</td>
          <td>0.164389</td>
          <td>25.674509</td>
          <td>0.412190</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.508224</td>
          <td>0.785962</td>
          <td>27.034139</td>
          <td>0.220862</td>
          <td>26.711661</td>
          <td>0.149112</td>
          <td>26.310510</td>
          <td>0.170000</td>
          <td>25.883578</td>
          <td>0.219793</td>
          <td>25.277389</td>
          <td>0.284429</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.812165</td>
          <td>1.660487</td>
          <td>27.263714</td>
          <td>0.273961</td>
          <td>26.982264</td>
          <td>0.193703</td>
          <td>26.374387</td>
          <td>0.185443</td>
          <td>25.648217</td>
          <td>0.186077</td>
          <td>25.796367</td>
          <td>0.440348</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.208795</td>
          <td>1.256227</td>
          <td>27.297706</td>
          <td>0.297083</td>
          <td>26.873146</td>
          <td>0.188181</td>
          <td>26.050837</td>
          <td>0.150441</td>
          <td>25.599184</td>
          <td>0.190160</td>
          <td>25.622628</td>
          <td>0.409142</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.336097</td>
          <td>0.339771</td>
          <td>26.561976</td>
          <td>0.151000</td>
          <td>26.132100</td>
          <td>0.092104</td>
          <td>25.681602</td>
          <td>0.101009</td>
          <td>25.409651</td>
          <td>0.150397</td>
          <td>24.798605</td>
          <td>0.195733</td>
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
