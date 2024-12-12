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

    <pzflow.flow.Flow at 0x7f4caf576980>



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
          <td>27.508292</td>
          <td>0.779296</td>
          <td>26.383927</td>
          <td>0.125274</td>
          <td>25.965277</td>
          <td>0.076465</td>
          <td>25.308990</td>
          <td>0.069824</td>
          <td>25.019420</td>
          <td>0.103184</td>
          <td>24.896358</td>
          <td>0.204436</td>
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
          <td>27.501721</td>
          <td>0.284231</td>
          <td>27.447972</td>
          <td>0.421293</td>
          <td>26.969338</td>
          <td>0.509679</td>
          <td>25.020530</td>
          <td>0.226751</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.025054</td>
          <td>0.258078</td>
          <td>25.885414</td>
          <td>0.081011</td>
          <td>24.770025</td>
          <td>0.026574</td>
          <td>23.850316</td>
          <td>0.019398</td>
          <td>23.128935</td>
          <td>0.019599</td>
          <td>22.837887</td>
          <td>0.033793</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.432932</td>
          <td>0.641204</td>
          <td>27.026916</td>
          <td>0.191915</td>
          <td>26.316062</td>
          <td>0.167969</td>
          <td>26.351373</td>
          <td>0.317103</td>
          <td>25.012805</td>
          <td>0.225301</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.131076</td>
          <td>0.602561</td>
          <td>25.763985</td>
          <td>0.072784</td>
          <td>25.456974</td>
          <td>0.048725</td>
          <td>24.909060</td>
          <td>0.048968</td>
          <td>24.400963</td>
          <td>0.059794</td>
          <td>23.726525</td>
          <td>0.074273</td>
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
          <td>27.094840</td>
          <td>0.587289</td>
          <td>26.218331</td>
          <td>0.108475</td>
          <td>26.118423</td>
          <td>0.087523</td>
          <td>26.018044</td>
          <td>0.130033</td>
          <td>26.365484</td>
          <td>0.320692</td>
          <td>25.445693</td>
          <td>0.320553</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.385011</td>
          <td>1.313643</td>
          <td>27.077248</td>
          <td>0.225838</td>
          <td>26.877011</td>
          <td>0.169026</td>
          <td>26.669067</td>
          <td>0.226072</td>
          <td>26.420059</td>
          <td>0.334902</td>
          <td>25.667662</td>
          <td>0.381714</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.062983</td>
          <td>0.574106</td>
          <td>26.936765</td>
          <td>0.200849</td>
          <td>26.625250</td>
          <td>0.136204</td>
          <td>26.375605</td>
          <td>0.176690</td>
          <td>26.072382</td>
          <td>0.252982</td>
          <td>25.003897</td>
          <td>0.223639</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.084174</td>
          <td>0.582850</td>
          <td>26.959064</td>
          <td>0.204640</td>
          <td>26.519079</td>
          <td>0.124245</td>
          <td>25.675665</td>
          <td>0.096475</td>
          <td>25.630966</td>
          <td>0.174941</td>
          <td>26.572794</td>
          <td>0.735839</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.313178</td>
          <td>0.325559</td>
          <td>26.681180</td>
          <td>0.161774</td>
          <td>26.125751</td>
          <td>0.088090</td>
          <td>25.705641</td>
          <td>0.099044</td>
          <td>25.319506</td>
          <td>0.133960</td>
          <td>24.963415</td>
          <td>0.216227</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.904690</td>
          <td>0.223993</td>
          <td>26.006658</td>
          <td>0.093268</td>
          <td>25.350705</td>
          <td>0.085826</td>
          <td>25.087427</td>
          <td>0.128478</td>
          <td>25.111903</td>
          <td>0.285574</td>
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
          <td>27.596725</td>
          <td>0.355073</td>
          <td>27.769305</td>
          <td>0.613976</td>
          <td>26.901030</td>
          <td>0.554364</td>
          <td>27.630518</td>
          <td>1.520982</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.673415</td>
          <td>0.483166</td>
          <td>25.912061</td>
          <td>0.097652</td>
          <td>24.754325</td>
          <td>0.031518</td>
          <td>23.906928</td>
          <td>0.024574</td>
          <td>23.134483</td>
          <td>0.023580</td>
          <td>22.814076</td>
          <td>0.040091</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.436796</td>
          <td>0.364061</td>
          <td>27.135390</td>
          <td>0.260705</td>
          <td>26.387789</td>
          <td>0.224064</td>
          <td>25.747283</td>
          <td>0.239775</td>
          <td>25.137791</td>
          <td>0.310566</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.172003</td>
          <td>0.323682</td>
          <td>25.806791</td>
          <td>0.087287</td>
          <td>25.469339</td>
          <td>0.058041</td>
          <td>24.832654</td>
          <td>0.054290</td>
          <td>24.406506</td>
          <td>0.070755</td>
          <td>23.647377</td>
          <td>0.081960</td>
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
          <td>27.491256</td>
          <td>0.850279</td>
          <td>26.896566</td>
          <td>0.226511</td>
          <td>26.037927</td>
          <td>0.097896</td>
          <td>26.027409</td>
          <td>0.157926</td>
          <td>26.387407</td>
          <td>0.384065</td>
          <td>25.376887</td>
          <td>0.359648</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.070979</td>
          <td>2.834682</td>
          <td>27.051889</td>
          <td>0.253824</td>
          <td>27.010965</td>
          <td>0.221766</td>
          <td>26.459061</td>
          <td>0.223555</td>
          <td>25.981706</td>
          <td>0.273996</td>
          <td>25.601443</td>
          <td>0.421399</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.771822</td>
          <td>0.516344</td>
          <td>27.560706</td>
          <td>0.383773</td>
          <td>26.662250</td>
          <td>0.166690</td>
          <td>26.730038</td>
          <td>0.281605</td>
          <td>25.624806</td>
          <td>0.205706</td>
          <td>25.416953</td>
          <td>0.368322</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.860632</td>
          <td>1.071805</td>
          <td>27.781500</td>
          <td>0.460858</td>
          <td>26.790083</td>
          <td>0.189145</td>
          <td>25.843251</td>
          <td>0.136241</td>
          <td>25.525139</td>
          <td>0.192581</td>
          <td>26.560286</td>
          <td>0.846416</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.278591</td>
          <td>0.354446</td>
          <td>27.025919</td>
          <td>0.249703</td>
          <td>25.930429</td>
          <td>0.088106</td>
          <td>25.396199</td>
          <td>0.090266</td>
          <td>25.116045</td>
          <td>0.133006</td>
          <td>25.201422</td>
          <td>0.309821</td>
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
          <td>27.207304</td>
          <td>0.635701</td>
          <td>26.805326</td>
          <td>0.179802</td>
          <td>26.023374</td>
          <td>0.080501</td>
          <td>25.158921</td>
          <td>0.061137</td>
          <td>25.089236</td>
          <td>0.109690</td>
          <td>24.710312</td>
          <td>0.174756</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.962616</td>
          <td>0.534333</td>
          <td>28.798357</td>
          <td>0.819915</td>
          <td>27.708379</td>
          <td>0.335678</td>
          <td>27.085302</td>
          <td>0.317655</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.089113</td>
          <td>0.524886</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.502721</td>
          <td>0.810941</td>
          <td>25.924371</td>
          <td>0.090110</td>
          <td>24.790902</td>
          <td>0.029374</td>
          <td>23.883277</td>
          <td>0.021686</td>
          <td>23.138259</td>
          <td>0.021396</td>
          <td>22.807465</td>
          <td>0.035837</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.576157</td>
          <td>0.404421</td>
          <td>27.347621</td>
          <td>0.308561</td>
          <td>27.842711</td>
          <td>0.680294</td>
          <td>26.314952</td>
          <td>0.376955</td>
          <td>25.452166</td>
          <td>0.396329</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.919757</td>
          <td>0.236914</td>
          <td>25.878576</td>
          <td>0.080624</td>
          <td>25.452911</td>
          <td>0.048619</td>
          <td>24.906705</td>
          <td>0.048939</td>
          <td>24.348487</td>
          <td>0.057155</td>
          <td>23.723033</td>
          <td>0.074155</td>
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
          <td>26.658275</td>
          <td>0.446822</td>
          <td>26.355454</td>
          <td>0.130826</td>
          <td>26.110352</td>
          <td>0.094034</td>
          <td>25.935207</td>
          <td>0.131268</td>
          <td>26.486466</td>
          <td>0.378463</td>
          <td>26.212408</td>
          <td>0.612346</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.147692</td>
          <td>0.288158</td>
          <td>26.641125</td>
          <td>0.158527</td>
          <td>27.054612</td>
          <td>0.199563</td>
          <td>26.524361</td>
          <td>0.203668</td>
          <td>26.252696</td>
          <td>0.297416</td>
          <td>26.749483</td>
          <td>0.836792</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.773793</td>
          <td>0.478395</td>
          <td>27.135841</td>
          <td>0.246759</td>
          <td>26.861462</td>
          <td>0.174891</td>
          <td>26.655681</td>
          <td>0.234646</td>
          <td>25.788546</td>
          <td>0.209380</td>
          <td>26.604873</td>
          <td>0.780944</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.444418</td>
          <td>0.794537</td>
          <td>28.105401</td>
          <td>0.551438</td>
          <td>26.498865</td>
          <td>0.136692</td>
          <td>25.869401</td>
          <td>0.128658</td>
          <td>25.635392</td>
          <td>0.196050</td>
          <td>26.175589</td>
          <td>0.614664</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.630684</td>
          <td>0.426936</td>
          <td>26.766868</td>
          <td>0.179805</td>
          <td>26.121280</td>
          <td>0.091233</td>
          <td>25.500557</td>
          <td>0.086160</td>
          <td>25.393918</td>
          <td>0.148379</td>
          <td>24.781897</td>
          <td>0.192999</td>
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
