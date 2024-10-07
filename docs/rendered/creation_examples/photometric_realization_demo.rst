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

    <pzflow.flow.Flow at 0x7f2d36761d20>



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
          <td>27.798624</td>
          <td>0.937547</td>
          <td>26.862352</td>
          <td>0.188660</td>
          <td>26.098172</td>
          <td>0.085976</td>
          <td>25.369329</td>
          <td>0.073653</td>
          <td>25.108265</td>
          <td>0.111512</td>
          <td>25.028551</td>
          <td>0.228265</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.655950</td>
          <td>1.510561</td>
          <td>28.656545</td>
          <td>0.746614</td>
          <td>27.985073</td>
          <td>0.416018</td>
          <td>27.722541</td>
          <td>0.517341</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.021325</td>
          <td>0.498995</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.073550</td>
          <td>0.578454</td>
          <td>25.821258</td>
          <td>0.076557</td>
          <td>24.825404</td>
          <td>0.027890</td>
          <td>23.846611</td>
          <td>0.019337</td>
          <td>23.129748</td>
          <td>0.019612</td>
          <td>22.789833</td>
          <td>0.032392</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.811283</td>
          <td>0.944882</td>
          <td>28.738107</td>
          <td>0.787923</td>
          <td>28.261323</td>
          <td>0.511750</td>
          <td>26.469408</td>
          <td>0.191281</td>
          <td>27.496526</td>
          <td>0.738050</td>
          <td>25.068572</td>
          <td>0.235958</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.765028</td>
          <td>0.208162</td>
          <td>25.807263</td>
          <td>0.075618</td>
          <td>25.361005</td>
          <td>0.044746</td>
          <td>24.861240</td>
          <td>0.046932</td>
          <td>24.348748</td>
          <td>0.057087</td>
          <td>23.727344</td>
          <td>0.074327</td>
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
          <td>27.439954</td>
          <td>0.744856</td>
          <td>26.986968</td>
          <td>0.209475</td>
          <td>26.074837</td>
          <td>0.084227</td>
          <td>26.161418</td>
          <td>0.147151</td>
          <td>25.787843</td>
          <td>0.199728</td>
          <td>26.190727</td>
          <td>0.564502</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.147927</td>
          <td>0.609764</td>
          <td>27.914297</td>
          <td>0.439808</td>
          <td>27.336823</td>
          <td>0.248442</td>
          <td>26.482477</td>
          <td>0.193399</td>
          <td>25.972174</td>
          <td>0.232923</td>
          <td>25.480432</td>
          <td>0.329533</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.345831</td>
          <td>0.281515</td>
          <td>27.105578</td>
          <td>0.205033</td>
          <td>26.288088</td>
          <td>0.164010</td>
          <td>26.227532</td>
          <td>0.287071</td>
          <td>24.754230</td>
          <td>0.181363</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.474919</td>
          <td>1.377492</td>
          <td>27.624431</td>
          <td>0.351650</td>
          <td>26.490627</td>
          <td>0.121213</td>
          <td>25.771903</td>
          <td>0.104960</td>
          <td>25.659339</td>
          <td>0.179203</td>
          <td>25.695018</td>
          <td>0.389890</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>32.850567</td>
          <td>5.401925</td>
          <td>26.708659</td>
          <td>0.165610</td>
          <td>26.068218</td>
          <td>0.083737</td>
          <td>25.695355</td>
          <td>0.098155</td>
          <td>25.135297</td>
          <td>0.114171</td>
          <td>25.146846</td>
          <td>0.251680</td>
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
          <td>28.604389</td>
          <td>1.572584</td>
          <td>26.592010</td>
          <td>0.172235</td>
          <td>26.387860</td>
          <td>0.130054</td>
          <td>25.402841</td>
          <td>0.089854</td>
          <td>25.004292</td>
          <td>0.119539</td>
          <td>25.139160</td>
          <td>0.291933</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.453821</td>
          <td>1.459471</td>
          <td>27.571162</td>
          <td>0.383041</td>
          <td>27.492791</td>
          <td>0.327089</td>
          <td>27.736666</td>
          <td>0.600003</td>
          <td>26.592376</td>
          <td>0.441293</td>
          <td>25.342769</td>
          <td>0.343505</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.765229</td>
          <td>0.516998</td>
          <td>25.982785</td>
          <td>0.103881</td>
          <td>24.815238</td>
          <td>0.033253</td>
          <td>23.872673</td>
          <td>0.023858</td>
          <td>23.089839</td>
          <td>0.022692</td>
          <td>22.877648</td>
          <td>0.042412</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.624557</td>
          <td>0.479836</td>
          <td>27.461159</td>
          <td>0.371050</td>
          <td>27.055040</td>
          <td>0.244065</td>
          <td>26.458231</td>
          <td>0.237531</td>
          <td>26.609777</td>
          <td>0.473396</td>
          <td>25.157320</td>
          <td>0.315452</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.002530</td>
          <td>0.282556</td>
          <td>25.600621</td>
          <td>0.072796</td>
          <td>25.289855</td>
          <td>0.049497</td>
          <td>24.931249</td>
          <td>0.059252</td>
          <td>24.341932</td>
          <td>0.066825</td>
          <td>23.831273</td>
          <td>0.096344</td>
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
          <td>26.782967</td>
          <td>0.523295</td>
          <td>26.440896</td>
          <td>0.154229</td>
          <td>26.161382</td>
          <td>0.109059</td>
          <td>25.943678</td>
          <td>0.146988</td>
          <td>25.542293</td>
          <td>0.193443</td>
          <td>27.017281</td>
          <td>1.109421</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.194330</td>
          <td>2.949426</td>
          <td>27.067818</td>
          <td>0.257158</td>
          <td>27.600296</td>
          <td>0.357332</td>
          <td>26.429664</td>
          <td>0.218152</td>
          <td>26.020710</td>
          <td>0.282808</td>
          <td>25.206381</td>
          <td>0.309350</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.296485</td>
          <td>0.360123</td>
          <td>26.969066</td>
          <td>0.238839</td>
          <td>26.700987</td>
          <td>0.172277</td>
          <td>26.470447</td>
          <td>0.227588</td>
          <td>26.203478</td>
          <td>0.330032</td>
          <td>24.892691</td>
          <td>0.241680</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.077129</td>
          <td>1.212524</td>
          <td>27.322822</td>
          <td>0.323193</td>
          <td>26.536105</td>
          <td>0.152392</td>
          <td>25.978918</td>
          <td>0.153103</td>
          <td>25.682149</td>
          <td>0.219649</td>
          <td>25.349407</td>
          <td>0.355355</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.165953</td>
          <td>0.324292</td>
          <td>26.236253</td>
          <td>0.128074</td>
          <td>26.024860</td>
          <td>0.095726</td>
          <td>25.568939</td>
          <td>0.105025</td>
          <td>25.455074</td>
          <td>0.177816</td>
          <td>24.906338</td>
          <td>0.243768</td>
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
          <td>27.094791</td>
          <td>0.587312</td>
          <td>26.644989</td>
          <td>0.156867</td>
          <td>25.919933</td>
          <td>0.073471</td>
          <td>25.294120</td>
          <td>0.068920</td>
          <td>24.898745</td>
          <td>0.092837</td>
          <td>24.862979</td>
          <td>0.198814</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.203816</td>
          <td>0.634441</td>
          <td>30.398045</td>
          <td>1.931041</td>
          <td>27.340455</td>
          <td>0.249406</td>
          <td>27.635460</td>
          <td>0.485569</td>
          <td>26.357701</td>
          <td>0.318981</td>
          <td>27.228980</td>
          <td>1.108455</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.926648</td>
          <td>0.251463</td>
          <td>26.052208</td>
          <td>0.100793</td>
          <td>24.763425</td>
          <td>0.028676</td>
          <td>23.883076</td>
          <td>0.021682</td>
          <td>23.135084</td>
          <td>0.021338</td>
          <td>22.752587</td>
          <td>0.034143</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.396737</td>
          <td>1.458331</td>
          <td>27.529507</td>
          <td>0.390141</td>
          <td>27.295305</td>
          <td>0.295863</td>
          <td>26.680627</td>
          <td>0.283953</td>
          <td>25.926369</td>
          <td>0.276723</td>
          <td>25.569346</td>
          <td>0.433502</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.904016</td>
          <td>0.233856</td>
          <td>25.788291</td>
          <td>0.074454</td>
          <td>25.487339</td>
          <td>0.050128</td>
          <td>24.804821</td>
          <td>0.044707</td>
          <td>24.485220</td>
          <td>0.064524</td>
          <td>23.815399</td>
          <td>0.080457</td>
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
          <td>26.929673</td>
          <td>0.546023</td>
          <td>26.292617</td>
          <td>0.123900</td>
          <td>26.052146</td>
          <td>0.089345</td>
          <td>25.995872</td>
          <td>0.138330</td>
          <td>25.742508</td>
          <td>0.207259</td>
          <td>26.267721</td>
          <td>0.636528</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.836233</td>
          <td>0.967119</td>
          <td>26.617879</td>
          <td>0.155407</td>
          <td>26.925159</td>
          <td>0.178910</td>
          <td>26.527459</td>
          <td>0.204198</td>
          <td>26.429087</td>
          <td>0.342329</td>
          <td>25.228803</td>
          <td>0.273432</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.134189</td>
          <td>0.620761</td>
          <td>27.190551</td>
          <td>0.258089</td>
          <td>27.013470</td>
          <td>0.198856</td>
          <td>26.615298</td>
          <td>0.226923</td>
          <td>25.843337</td>
          <td>0.219178</td>
          <td>26.839106</td>
          <td>0.907344</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.277278</td>
          <td>0.711072</td>
          <td>27.305768</td>
          <td>0.299016</td>
          <td>26.542379</td>
          <td>0.141918</td>
          <td>25.735213</td>
          <td>0.114504</td>
          <td>25.354352</td>
          <td>0.154426</td>
          <td>24.976968</td>
          <td>0.244567</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.307922</td>
          <td>0.695915</td>
          <td>26.517336</td>
          <td>0.145324</td>
          <td>26.126717</td>
          <td>0.091670</td>
          <td>25.872025</td>
          <td>0.119271</td>
          <td>25.146391</td>
          <td>0.119805</td>
          <td>24.693402</td>
          <td>0.179093</td>
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
