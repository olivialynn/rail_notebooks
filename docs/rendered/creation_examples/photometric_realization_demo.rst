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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f49d57803a0>



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
          <td>26.622954</td>
          <td>0.414503</td>
          <td>26.840257</td>
          <td>0.185173</td>
          <td>26.019173</td>
          <td>0.080192</td>
          <td>25.337045</td>
          <td>0.071579</td>
          <td>25.046690</td>
          <td>0.105675</td>
          <td>25.018654</td>
          <td>0.226398</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.827107</td>
          <td>0.954102</td>
          <td>30.379409</td>
          <td>1.914803</td>
          <td>27.448775</td>
          <td>0.272273</td>
          <td>27.643869</td>
          <td>0.488198</td>
          <td>26.557456</td>
          <td>0.373071</td>
          <td>26.809593</td>
          <td>0.858822</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.260703</td>
          <td>0.312233</td>
          <td>25.896661</td>
          <td>0.081818</td>
          <td>24.759570</td>
          <td>0.026333</td>
          <td>23.847645</td>
          <td>0.019354</td>
          <td>23.159982</td>
          <td>0.020120</td>
          <td>22.793484</td>
          <td>0.032496</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.614197</td>
          <td>0.725770</td>
          <td>27.479418</td>
          <td>0.279139</td>
          <td>26.280258</td>
          <td>0.162918</td>
          <td>25.923838</td>
          <td>0.223768</td>
          <td>25.342855</td>
          <td>0.295195</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.938374</td>
          <td>0.240356</td>
          <td>25.904269</td>
          <td>0.082367</td>
          <td>25.448715</td>
          <td>0.048369</td>
          <td>24.780087</td>
          <td>0.043670</td>
          <td>24.249843</td>
          <td>0.052288</td>
          <td>23.728621</td>
          <td>0.074411</td>
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
          <td>26.999564</td>
          <td>0.548534</td>
          <td>26.150388</td>
          <td>0.102226</td>
          <td>26.192040</td>
          <td>0.093376</td>
          <td>25.991393</td>
          <td>0.127066</td>
          <td>25.842078</td>
          <td>0.209018</td>
          <td>25.691521</td>
          <td>0.388837</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.116277</td>
          <td>1.132290</td>
          <td>27.095454</td>
          <td>0.229275</td>
          <td>27.001104</td>
          <td>0.187781</td>
          <td>26.655328</td>
          <td>0.223505</td>
          <td>25.954435</td>
          <td>0.229525</td>
          <td>25.447004</td>
          <td>0.320888</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.550839</td>
          <td>0.801281</td>
          <td>27.478033</td>
          <td>0.313120</td>
          <td>26.875262</td>
          <td>0.168774</td>
          <td>26.730621</td>
          <td>0.237898</td>
          <td>26.007291</td>
          <td>0.239784</td>
          <td>26.251525</td>
          <td>0.589555</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.510198</td>
          <td>0.380024</td>
          <td>27.037896</td>
          <td>0.218569</td>
          <td>26.574297</td>
          <td>0.130335</td>
          <td>25.701695</td>
          <td>0.098703</td>
          <td>25.741223</td>
          <td>0.192044</td>
          <td>25.410361</td>
          <td>0.311636</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.907429</td>
          <td>1.001763</td>
          <td>26.394030</td>
          <td>0.126375</td>
          <td>26.083011</td>
          <td>0.084836</td>
          <td>25.617975</td>
          <td>0.091709</td>
          <td>25.439370</td>
          <td>0.148533</td>
          <td>25.070800</td>
          <td>0.236393</td>
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
          <td>26.700096</td>
          <td>0.164406</td>
          <td>25.992199</td>
          <td>0.078305</td>
          <td>25.377697</td>
          <td>0.074200</td>
          <td>25.148633</td>
          <td>0.115504</td>
          <td>24.752521</td>
          <td>0.181100</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.525576</td>
          <td>0.325215</td>
          <td>27.312632</td>
          <td>0.243543</td>
          <td>26.763386</td>
          <td>0.244416</td>
          <td>26.950335</td>
          <td>0.502601</td>
          <td>26.837936</td>
          <td>0.874403</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.361321</td>
          <td>0.706558</td>
          <td>25.858604</td>
          <td>0.079120</td>
          <td>24.771000</td>
          <td>0.026596</td>
          <td>23.864432</td>
          <td>0.019631</td>
          <td>23.139274</td>
          <td>0.019771</td>
          <td>22.821760</td>
          <td>0.033316</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.360543</td>
          <td>2.976765</td>
          <td>28.708752</td>
          <td>0.772879</td>
          <td>27.373803</td>
          <td>0.256101</td>
          <td>27.096130</td>
          <td>0.320121</td>
          <td>25.990436</td>
          <td>0.236469</td>
          <td>25.417760</td>
          <td>0.313486</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.883706</td>
          <td>0.229749</td>
          <td>25.893486</td>
          <td>0.081589</td>
          <td>25.416941</td>
          <td>0.047023</td>
          <td>24.799578</td>
          <td>0.044432</td>
          <td>24.419346</td>
          <td>0.060777</td>
          <td>23.577742</td>
          <td>0.065109</td>
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
          <td>26.221121</td>
          <td>0.302496</td>
          <td>26.448966</td>
          <td>0.132525</td>
          <td>26.103152</td>
          <td>0.086354</td>
          <td>25.990507</td>
          <td>0.126968</td>
          <td>25.809470</td>
          <td>0.203386</td>
          <td>25.541438</td>
          <td>0.345826</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.618794</td>
          <td>0.413187</td>
          <td>26.703851</td>
          <td>0.164933</td>
          <td>26.938530</td>
          <td>0.178096</td>
          <td>26.854287</td>
          <td>0.263343</td>
          <td>25.781238</td>
          <td>0.198622</td>
          <td>25.439908</td>
          <td>0.319078</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.851969</td>
          <td>0.492430</td>
          <td>26.917637</td>
          <td>0.197649</td>
          <td>26.752622</td>
          <td>0.151982</td>
          <td>26.469364</td>
          <td>0.191274</td>
          <td>25.966839</td>
          <td>0.231896</td>
          <td>25.799158</td>
          <td>0.422356</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.861698</td>
          <td>0.495983</td>
          <td>26.776779</td>
          <td>0.175484</td>
          <td>26.637561</td>
          <td>0.137659</td>
          <td>25.773053</td>
          <td>0.105066</td>
          <td>25.432643</td>
          <td>0.147677</td>
          <td>26.160369</td>
          <td>0.552300</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.304359</td>
          <td>0.679702</td>
          <td>26.807028</td>
          <td>0.180041</td>
          <td>26.123123</td>
          <td>0.087886</td>
          <td>25.540206</td>
          <td>0.085643</td>
          <td>25.122289</td>
          <td>0.112884</td>
          <td>24.734533</td>
          <td>0.178361</td>
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
          <td>27.232672</td>
          <td>0.646960</td>
          <td>26.512713</td>
          <td>0.140017</td>
          <td>25.914227</td>
          <td>0.073091</td>
          <td>25.296809</td>
          <td>0.069075</td>
          <td>24.912460</td>
          <td>0.093949</td>
          <td>24.989861</td>
          <td>0.221043</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.925229</td>
          <td>1.012518</td>
          <td>30.568650</td>
          <td>2.073848</td>
          <td>27.948165</td>
          <td>0.404411</td>
          <td>27.172165</td>
          <td>0.340035</td>
          <td>27.579627</td>
          <td>0.779827</td>
          <td>25.609403</td>
          <td>0.364778</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.794919</td>
          <td>0.471998</td>
          <td>26.014043</td>
          <td>0.090714</td>
          <td>24.798963</td>
          <td>0.027253</td>
          <td>23.874856</td>
          <td>0.019804</td>
          <td>23.121393</td>
          <td>0.019474</td>
          <td>22.782987</td>
          <td>0.032197</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.167611</td>
          <td>1.165795</td>
          <td>29.120912</td>
          <td>1.002102</td>
          <td>28.003378</td>
          <td>0.421875</td>
          <td>26.394482</td>
          <td>0.179541</td>
          <td>25.815420</td>
          <td>0.204404</td>
          <td>25.373106</td>
          <td>0.302467</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.242084</td>
          <td>0.307620</td>
          <td>25.840644</td>
          <td>0.077877</td>
          <td>25.411814</td>
          <td>0.046810</td>
          <td>24.826673</td>
          <td>0.045514</td>
          <td>24.369617</td>
          <td>0.058154</td>
          <td>23.699751</td>
          <td>0.072536</td>
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
          <td>26.647136</td>
          <td>0.422224</td>
          <td>26.463595</td>
          <td>0.134210</td>
          <td>26.305516</td>
          <td>0.103144</td>
          <td>26.095363</td>
          <td>0.139016</td>
          <td>25.439787</td>
          <td>0.148586</td>
          <td>25.215475</td>
          <td>0.266224</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.163098</td>
          <td>1.162827</td>
          <td>26.945441</td>
          <td>0.202317</td>
          <td>27.330130</td>
          <td>0.247078</td>
          <td>26.349927</td>
          <td>0.172879</td>
          <td>26.211431</td>
          <td>0.283355</td>
          <td>25.783864</td>
          <td>0.417453</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.458592</td>
          <td>0.754142</td>
          <td>26.865875</td>
          <td>0.189221</td>
          <td>26.821166</td>
          <td>0.161165</td>
          <td>26.837889</td>
          <td>0.259835</td>
          <td>25.680298</td>
          <td>0.182412</td>
          <td>25.341636</td>
          <td>0.294905</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.512774</td>
          <td>0.781592</td>
          <td>27.162024</td>
          <td>0.242244</td>
          <td>26.594502</td>
          <td>0.132633</td>
          <td>26.212782</td>
          <td>0.153782</td>
          <td>25.378258</td>
          <td>0.140926</td>
          <td>26.050341</td>
          <td>0.509772</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.820557</td>
          <td>0.481095</td>
          <td>26.558521</td>
          <td>0.145644</td>
          <td>26.086900</td>
          <td>0.085127</td>
          <td>25.801958</td>
          <td>0.107754</td>
          <td>25.116727</td>
          <td>0.112338</td>
          <td>24.835190</td>
          <td>0.194195</td>
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
