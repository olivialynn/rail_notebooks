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

    <pzflow.flow.Flow at 0x7f8ee688b640>



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
          <td>27.550088</td>
          <td>0.800889</td>
          <td>26.499352</td>
          <td>0.138414</td>
          <td>25.972786</td>
          <td>0.076974</td>
          <td>25.220384</td>
          <td>0.064553</td>
          <td>25.048614</td>
          <td>0.105853</td>
          <td>25.000155</td>
          <td>0.222945</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.196652</td>
          <td>0.630952</td>
          <td>28.200410</td>
          <td>0.543638</td>
          <td>27.927231</td>
          <td>0.397948</td>
          <td>27.202140</td>
          <td>0.348171</td>
          <td>26.474343</td>
          <td>0.349566</td>
          <td>25.454114</td>
          <td>0.322710</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.054989</td>
          <td>0.264463</td>
          <td>26.037363</td>
          <td>0.092590</td>
          <td>24.778750</td>
          <td>0.026777</td>
          <td>23.880343</td>
          <td>0.019897</td>
          <td>23.153511</td>
          <td>0.020010</td>
          <td>22.861225</td>
          <td>0.034496</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.106090</td>
          <td>0.507477</td>
          <td>27.399917</td>
          <td>0.261635</td>
          <td>27.202447</td>
          <td>0.348255</td>
          <td>26.868359</td>
          <td>0.472954</td>
          <td>25.900716</td>
          <td>0.456116</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.673919</td>
          <td>0.430913</td>
          <td>25.761202</td>
          <td>0.072605</td>
          <td>25.503369</td>
          <td>0.050774</td>
          <td>24.785616</td>
          <td>0.043885</td>
          <td>24.397753</td>
          <td>0.059624</td>
          <td>23.776026</td>
          <td>0.077594</td>
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
          <td>26.600762</td>
          <td>0.407520</td>
          <td>26.397840</td>
          <td>0.126793</td>
          <td>26.110816</td>
          <td>0.086939</td>
          <td>25.941834</td>
          <td>0.121718</td>
          <td>25.598958</td>
          <td>0.170245</td>
          <td>25.488391</td>
          <td>0.331621</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.582664</td>
          <td>0.817997</td>
          <td>26.964325</td>
          <td>0.205544</td>
          <td>26.638479</td>
          <td>0.137768</td>
          <td>26.463214</td>
          <td>0.190284</td>
          <td>25.859617</td>
          <td>0.212106</td>
          <td>25.752668</td>
          <td>0.407597</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.543521</td>
          <td>0.329883</td>
          <td>26.672586</td>
          <td>0.141878</td>
          <td>26.224263</td>
          <td>0.155302</td>
          <td>25.779550</td>
          <td>0.198341</td>
          <td>25.132890</td>
          <td>0.248810</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.852402</td>
          <td>0.492588</td>
          <td>26.885001</td>
          <td>0.192296</td>
          <td>26.720675</td>
          <td>0.147871</td>
          <td>26.161301</td>
          <td>0.147136</td>
          <td>25.713503</td>
          <td>0.187605</td>
          <td>25.588635</td>
          <td>0.358896</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.808267</td>
          <td>0.476717</td>
          <td>26.501025</td>
          <td>0.138614</td>
          <td>26.113675</td>
          <td>0.087158</td>
          <td>25.672532</td>
          <td>0.096210</td>
          <td>24.977623</td>
          <td>0.099476</td>
          <td>25.162402</td>
          <td>0.254913</td>
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
          <td>27.197131</td>
          <td>0.691975</td>
          <td>26.955205</td>
          <td>0.233571</td>
          <td>26.098939</td>
          <td>0.101129</td>
          <td>25.367071</td>
          <td>0.087071</td>
          <td>24.929694</td>
          <td>0.112024</td>
          <td>25.369710</td>
          <td>0.350807</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.892541</td>
          <td>2.667277</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.408820</td>
          <td>0.305881</td>
          <td>26.667940</td>
          <td>0.264525</td>
          <td>26.645141</td>
          <td>0.459193</td>
          <td>26.557065</td>
          <td>0.825127</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.317274</td>
          <td>0.368405</td>
          <td>25.785814</td>
          <td>0.087418</td>
          <td>24.772226</td>
          <td>0.032018</td>
          <td>23.838590</td>
          <td>0.023168</td>
          <td>23.146258</td>
          <td>0.023820</td>
          <td>22.898087</td>
          <td>0.043187</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.754299</td>
          <td>0.527913</td>
          <td>28.576375</td>
          <td>0.824605</td>
          <td>27.741763</td>
          <td>0.421383</td>
          <td>26.614461</td>
          <td>0.270015</td>
          <td>27.243026</td>
          <td>0.741155</td>
          <td>25.176792</td>
          <td>0.320390</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.961355</td>
          <td>0.587405</td>
          <td>25.674189</td>
          <td>0.077675</td>
          <td>25.466366</td>
          <td>0.057888</td>
          <td>24.770747</td>
          <td>0.051388</td>
          <td>24.445792</td>
          <td>0.073256</td>
          <td>23.764058</td>
          <td>0.090824</td>
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
          <td>26.149699</td>
          <td>0.322569</td>
          <td>26.585414</td>
          <td>0.174443</td>
          <td>26.032630</td>
          <td>0.097442</td>
          <td>26.260046</td>
          <td>0.192415</td>
          <td>25.801595</td>
          <td>0.240141</td>
          <td>24.932124</td>
          <td>0.251630</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.339658</td>
          <td>0.320324</td>
          <td>26.968002</td>
          <td>0.213967</td>
          <td>26.088179</td>
          <td>0.163549</td>
          <td>25.760966</td>
          <td>0.228557</td>
          <td>25.007066</td>
          <td>0.263284</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.162085</td>
          <td>2.031925</td>
          <td>27.035671</td>
          <td>0.252295</td>
          <td>27.190334</td>
          <td>0.259237</td>
          <td>26.461198</td>
          <td>0.225847</td>
          <td>25.876686</td>
          <td>0.253492</td>
          <td>25.281030</td>
          <td>0.330958</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.101224</td>
          <td>0.312613</td>
          <td>26.769216</td>
          <td>0.205532</td>
          <td>26.446713</td>
          <td>0.141125</td>
          <td>25.830335</td>
          <td>0.134730</td>
          <td>25.714652</td>
          <td>0.225666</td>
          <td>25.890901</td>
          <td>0.535391</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.344669</td>
          <td>0.140634</td>
          <td>26.127727</td>
          <td>0.104751</td>
          <td>25.578142</td>
          <td>0.105873</td>
          <td>25.076324</td>
          <td>0.128513</td>
          <td>24.890459</td>
          <td>0.240597</td>
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
          <td>28.321675</td>
          <td>1.269669</td>
          <td>26.638447</td>
          <td>0.155992</td>
          <td>26.102771</td>
          <td>0.086337</td>
          <td>25.302029</td>
          <td>0.069404</td>
          <td>25.072314</td>
          <td>0.108081</td>
          <td>24.843004</td>
          <td>0.195502</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.784718</td>
          <td>0.929959</td>
          <td>28.097993</td>
          <td>0.504809</td>
          <td>27.363896</td>
          <td>0.254253</td>
          <td>27.663856</td>
          <td>0.495890</td>
          <td>26.670982</td>
          <td>0.407642</td>
          <td>26.219277</td>
          <td>0.576624</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.936094</td>
          <td>0.253417</td>
          <td>25.895442</td>
          <td>0.087850</td>
          <td>24.794515</td>
          <td>0.029467</td>
          <td>23.886050</td>
          <td>0.021738</td>
          <td>23.119527</td>
          <td>0.021057</td>
          <td>22.878545</td>
          <td>0.038160</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.145311</td>
          <td>1.278403</td>
          <td>28.991883</td>
          <td>1.062824</td>
          <td>27.269992</td>
          <td>0.289885</td>
          <td>26.948488</td>
          <td>0.351640</td>
          <td>26.229500</td>
          <td>0.352597</td>
          <td>26.354772</td>
          <td>0.758703</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.863503</td>
          <td>0.226146</td>
          <td>25.689625</td>
          <td>0.068240</td>
          <td>25.414923</td>
          <td>0.047007</td>
          <td>24.814249</td>
          <td>0.045083</td>
          <td>24.363480</td>
          <td>0.057921</td>
          <td>23.578475</td>
          <td>0.065249</td>
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
          <td>26.501885</td>
          <td>0.396611</td>
          <td>26.373482</td>
          <td>0.132880</td>
          <td>26.238733</td>
          <td>0.105229</td>
          <td>25.877049</td>
          <td>0.124819</td>
          <td>25.845096</td>
          <td>0.225770</td>
          <td>25.279020</td>
          <td>0.302153</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.271164</td>
          <td>0.670385</td>
          <td>27.285367</td>
          <td>0.271597</td>
          <td>26.929375</td>
          <td>0.179550</td>
          <td>26.321896</td>
          <td>0.171654</td>
          <td>26.108341</td>
          <td>0.264564</td>
          <td>25.549034</td>
          <td>0.353269</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.028690</td>
          <td>1.101191</td>
          <td>27.598770</td>
          <td>0.358037</td>
          <td>26.633831</td>
          <td>0.143961</td>
          <td>26.368044</td>
          <td>0.184451</td>
          <td>26.115336</td>
          <td>0.274190</td>
          <td>25.020462</td>
          <td>0.237841</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.717775</td>
          <td>0.944844</td>
          <td>27.020024</td>
          <td>0.236869</td>
          <td>26.564075</td>
          <td>0.144592</td>
          <td>25.840060</td>
          <td>0.125428</td>
          <td>25.173629</td>
          <td>0.132182</td>
          <td>25.786114</td>
          <td>0.463140</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.143074</td>
          <td>0.621074</td>
          <td>26.748225</td>
          <td>0.176987</td>
          <td>25.970910</td>
          <td>0.079915</td>
          <td>25.500612</td>
          <td>0.086164</td>
          <td>25.024081</td>
          <td>0.107694</td>
          <td>24.677355</td>
          <td>0.176673</td>
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
