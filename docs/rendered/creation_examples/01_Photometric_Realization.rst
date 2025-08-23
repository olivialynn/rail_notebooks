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

    <pzflow.flow.Flow at 0x7ff7c1346a10>



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
    0      23.994413  0.039377  0.026321  
    1      25.391064  0.112394  0.079195  
    2      24.304707  0.026845  0.022993  
    3      25.291103  0.092090  0.059356  
    4      25.096743  0.198603  0.185528  
    ...          ...       ...       ...  
    99995  24.737946  0.040162  0.030846  
    99996  24.224169  0.135499  0.072105  
    99997  25.613836  0.024479  0.023240  
    99998  25.274899  0.006954  0.006267  
    99999  25.699642  0.114493  0.075957  
    
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
          <td>27.978334</td>
          <td>1.045019</td>
          <td>26.562388</td>
          <td>0.146129</td>
          <td>26.064858</td>
          <td>0.083489</td>
          <td>25.221117</td>
          <td>0.064595</td>
          <td>24.692956</td>
          <td>0.077434</td>
          <td>24.058773</td>
          <td>0.099512</td>
          <td>0.039377</td>
          <td>0.026321</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.331296</td>
          <td>0.692309</td>
          <td>27.799412</td>
          <td>0.402903</td>
          <td>26.495032</td>
          <td>0.121678</td>
          <td>26.298762</td>
          <td>0.165510</td>
          <td>26.229351</td>
          <td>0.287494</td>
          <td>25.050665</td>
          <td>0.232488</td>
          <td>0.112394</td>
          <td>0.079195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.377778</td>
          <td>0.556980</td>
          <td>26.021066</td>
          <td>0.130374</td>
          <td>24.984297</td>
          <td>0.100059</td>
          <td>24.302552</td>
          <td>0.123092</td>
          <td>0.026845</td>
          <td>0.022993</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.270005</td>
          <td>1.234250</td>
          <td>29.770449</td>
          <td>1.437566</td>
          <td>27.638659</td>
          <td>0.317309</td>
          <td>26.170425</td>
          <td>0.148294</td>
          <td>25.268174</td>
          <td>0.128141</td>
          <td>25.080109</td>
          <td>0.238219</td>
          <td>0.092090</td>
          <td>0.059356</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.250138</td>
          <td>0.309608</td>
          <td>26.268739</td>
          <td>0.113347</td>
          <td>25.945184</td>
          <td>0.075120</td>
          <td>25.716876</td>
          <td>0.100025</td>
          <td>26.045603</td>
          <td>0.247476</td>
          <td>24.983976</td>
          <td>0.219963</td>
          <td>0.198603</td>
          <td>0.185528</td>
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
          <td>26.304317</td>
          <td>0.116910</td>
          <td>25.504335</td>
          <td>0.050817</td>
          <td>25.058105</td>
          <td>0.055897</td>
          <td>24.891441</td>
          <td>0.092231</td>
          <td>24.386706</td>
          <td>0.132401</td>
          <td>0.040162</td>
          <td>0.030846</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.879874</td>
          <td>0.502676</td>
          <td>27.036029</td>
          <td>0.218229</td>
          <td>25.985117</td>
          <td>0.077817</td>
          <td>25.179380</td>
          <td>0.062248</td>
          <td>24.786405</td>
          <td>0.084088</td>
          <td>24.289439</td>
          <td>0.121698</td>
          <td>0.135499</td>
          <td>0.072105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.877319</td>
          <td>0.501731</td>
          <td>27.107525</td>
          <td>0.231579</td>
          <td>26.277481</td>
          <td>0.100643</td>
          <td>26.181245</td>
          <td>0.149678</td>
          <td>27.278425</td>
          <td>0.635953</td>
          <td>25.621803</td>
          <td>0.368329</td>
          <td>0.024479</td>
          <td>0.023240</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.260362</td>
          <td>0.312148</td>
          <td>26.147130</td>
          <td>0.101935</td>
          <td>26.249419</td>
          <td>0.098198</td>
          <td>26.039624</td>
          <td>0.132483</td>
          <td>25.742394</td>
          <td>0.192234</td>
          <td>25.051555</td>
          <td>0.232659</td>
          <td>0.006954</td>
          <td>0.006267</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.774997</td>
          <td>1.601172</td>
          <td>26.957136</td>
          <td>0.204310</td>
          <td>26.529495</td>
          <td>0.125373</td>
          <td>26.170507</td>
          <td>0.148304</td>
          <td>25.830927</td>
          <td>0.207077</td>
          <td>25.345083</td>
          <td>0.295725</td>
          <td>0.114493</td>
          <td>0.075957</td>
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
          <td>27.352179</td>
          <td>0.769405</td>
          <td>26.885744</td>
          <td>0.221205</td>
          <td>25.888278</td>
          <td>0.084365</td>
          <td>25.283566</td>
          <td>0.081214</td>
          <td>24.808475</td>
          <td>0.101147</td>
          <td>24.045028</td>
          <td>0.116536</td>
          <td>0.039377</td>
          <td>0.026321</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.745224</td>
          <td>0.512335</td>
          <td>27.295275</td>
          <td>0.316082</td>
          <td>26.476413</td>
          <td>0.144726</td>
          <td>26.619112</td>
          <td>0.261828</td>
          <td>25.791373</td>
          <td>0.240379</td>
          <td>25.831675</td>
          <td>0.512564</td>
          <td>0.112394</td>
          <td>0.079195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.993584</td>
          <td>0.601623</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.154727</td>
          <td>0.542300</td>
          <td>26.091912</td>
          <td>0.163728</td>
          <td>25.101419</td>
          <td>0.130311</td>
          <td>24.334342</td>
          <td>0.149395</td>
          <td>0.026845</td>
          <td>0.022993</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.937662</td>
          <td>0.512943</td>
          <td>27.054911</td>
          <td>0.233451</td>
          <td>25.928366</td>
          <td>0.144910</td>
          <td>25.401714</td>
          <td>0.171572</td>
          <td>25.856907</td>
          <td>0.517193</td>
          <td>0.092090</td>
          <td>0.059356</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.677678</td>
          <td>0.234928</td>
          <td>25.964896</td>
          <td>0.110919</td>
          <td>25.991209</td>
          <td>0.102871</td>
          <td>25.586123</td>
          <td>0.118305</td>
          <td>25.590406</td>
          <td>0.219563</td>
          <td>24.897778</td>
          <td>0.266756</td>
          <td>0.198603</td>
          <td>0.185528</td>
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
          <td>28.265312</td>
          <td>1.325896</td>
          <td>26.296648</td>
          <td>0.134250</td>
          <td>25.458889</td>
          <td>0.057738</td>
          <td>25.130139</td>
          <td>0.070962</td>
          <td>25.023646</td>
          <td>0.122085</td>
          <td>24.672463</td>
          <td>0.199544</td>
          <td>0.040162</td>
          <td>0.030846</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.322744</td>
          <td>0.769698</td>
          <td>26.664639</td>
          <td>0.189343</td>
          <td>25.894114</td>
          <td>0.087769</td>
          <td>25.296246</td>
          <td>0.085104</td>
          <td>24.674106</td>
          <td>0.093039</td>
          <td>23.958306</td>
          <td>0.111901</td>
          <td>0.135499</td>
          <td>0.072105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.982836</td>
          <td>0.597000</td>
          <td>26.530404</td>
          <td>0.163712</td>
          <td>26.250337</td>
          <td>0.115638</td>
          <td>26.337875</td>
          <td>0.201583</td>
          <td>29.417711</td>
          <td>2.226837</td>
          <td>26.098557</td>
          <td>0.606102</td>
          <td>0.024479</td>
          <td>0.023240</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.953307</td>
          <td>0.271464</td>
          <td>26.291504</td>
          <td>0.133158</td>
          <td>26.136581</td>
          <td>0.104527</td>
          <td>25.697757</td>
          <td>0.116322</td>
          <td>25.595221</td>
          <td>0.198251</td>
          <td>25.284560</td>
          <td>0.328009</td>
          <td>0.006954</td>
          <td>0.006267</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.182919</td>
          <td>0.698002</td>
          <td>26.785575</td>
          <td>0.208286</td>
          <td>26.366071</td>
          <td>0.131578</td>
          <td>26.732025</td>
          <td>0.286982</td>
          <td>26.323461</td>
          <td>0.368664</td>
          <td>25.579429</td>
          <td>0.424390</td>
          <td>0.114493</td>
          <td>0.075957</td>
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
          <td>29.192508</td>
          <td>1.946124</td>
          <td>26.560740</td>
          <td>0.147791</td>
          <td>26.011339</td>
          <td>0.080848</td>
          <td>25.378933</td>
          <td>0.075467</td>
          <td>24.657695</td>
          <td>0.076198</td>
          <td>23.864038</td>
          <td>0.085179</td>
          <td>0.039377</td>
          <td>0.026321</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.505114</td>
          <td>0.825646</td>
          <td>27.517629</td>
          <td>0.353306</td>
          <td>26.398739</td>
          <td>0.125094</td>
          <td>26.054809</td>
          <td>0.150630</td>
          <td>26.138770</td>
          <td>0.296291</td>
          <td>26.383542</td>
          <td>0.708264</td>
          <td>0.112394</td>
          <td>0.079195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.945591</td>
          <td>0.530027</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.874364</td>
          <td>0.384852</td>
          <td>25.923433</td>
          <td>0.120839</td>
          <td>24.930967</td>
          <td>0.096294</td>
          <td>24.420240</td>
          <td>0.137470</td>
          <td>0.026845</td>
          <td>0.022993</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544965</td>
          <td>0.350211</td>
          <td>26.846731</td>
          <td>0.177033</td>
          <td>26.417259</td>
          <td>0.197248</td>
          <td>25.652687</td>
          <td>0.191416</td>
          <td>25.415752</td>
          <td>0.335618</td>
          <td>0.092090</td>
          <td>0.059356</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.930673</td>
          <td>0.304130</td>
          <td>26.104860</td>
          <td>0.133466</td>
          <td>26.105676</td>
          <td>0.121797</td>
          <td>25.788568</td>
          <td>0.151164</td>
          <td>25.406040</td>
          <td>0.201085</td>
          <td>25.197540</td>
          <td>0.361657</td>
          <td>0.198603</td>
          <td>0.185528</td>
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
          <td>26.126715</td>
          <td>0.283426</td>
          <td>26.770952</td>
          <td>0.177143</td>
          <td>25.437639</td>
          <td>0.048733</td>
          <td>25.096798</td>
          <td>0.058911</td>
          <td>24.744234</td>
          <td>0.082422</td>
          <td>24.828656</td>
          <td>0.196444</td>
          <td>0.040162</td>
          <td>0.030846</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.165123</td>
          <td>0.666354</td>
          <td>27.322655</td>
          <td>0.307882</td>
          <td>26.146557</td>
          <td>0.102510</td>
          <td>25.184430</td>
          <td>0.071952</td>
          <td>24.626435</td>
          <td>0.083489</td>
          <td>24.582253</td>
          <td>0.179051</td>
          <td>0.135499</td>
          <td>0.072105</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.861152</td>
          <td>0.498007</td>
          <td>26.627047</td>
          <td>0.155467</td>
          <td>26.455678</td>
          <td>0.118484</td>
          <td>26.126591</td>
          <td>0.143941</td>
          <td>25.888915</td>
          <td>0.218943</td>
          <td>26.310604</td>
          <td>0.618626</td>
          <td>0.024479</td>
          <td>0.023240</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.573821</td>
          <td>0.813576</td>
          <td>26.115589</td>
          <td>0.099212</td>
          <td>26.154178</td>
          <td>0.090374</td>
          <td>25.753698</td>
          <td>0.103367</td>
          <td>26.294669</td>
          <td>0.303194</td>
          <td>25.192487</td>
          <td>0.261422</td>
          <td>0.006954</td>
          <td>0.006267</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.222913</td>
          <td>0.325935</td>
          <td>26.477927</td>
          <td>0.149678</td>
          <td>26.556250</td>
          <td>0.143254</td>
          <td>26.107884</td>
          <td>0.157548</td>
          <td>25.850871</td>
          <td>0.234090</td>
          <td>26.305213</td>
          <td>0.671143</td>
          <td>0.114493</td>
          <td>0.075957</td>
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
