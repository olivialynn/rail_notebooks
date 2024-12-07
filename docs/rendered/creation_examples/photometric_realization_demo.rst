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

    <pzflow.flow.Flow at 0x7f61dc16d9c0>



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
          <td>27.517458</td>
          <td>0.783996</td>
          <td>26.879575</td>
          <td>0.191419</td>
          <td>25.928034</td>
          <td>0.073989</td>
          <td>25.391744</td>
          <td>0.075127</td>
          <td>25.062646</td>
          <td>0.107159</td>
          <td>25.091028</td>
          <td>0.240376</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.935304</td>
          <td>0.446845</td>
          <td>27.122332</td>
          <td>0.207930</td>
          <td>27.230049</td>
          <td>0.355894</td>
          <td>26.812739</td>
          <td>0.453645</td>
          <td>26.718507</td>
          <td>0.809991</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.738500</td>
          <td>0.452465</td>
          <td>25.830949</td>
          <td>0.077214</td>
          <td>24.786891</td>
          <td>0.026968</td>
          <td>23.836572</td>
          <td>0.019174</td>
          <td>23.112594</td>
          <td>0.019330</td>
          <td>22.830900</td>
          <td>0.033586</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.140419</td>
          <td>1.893111</td>
          <td>27.719022</td>
          <td>0.378630</td>
          <td>27.774603</td>
          <td>0.353377</td>
          <td>26.348089</td>
          <td>0.172609</td>
          <td>26.093878</td>
          <td>0.257480</td>
          <td>25.599404</td>
          <td>0.361936</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.618902</td>
          <td>0.413221</td>
          <td>25.781387</td>
          <td>0.073910</td>
          <td>25.439679</td>
          <td>0.047982</td>
          <td>24.805256</td>
          <td>0.044657</td>
          <td>24.487803</td>
          <td>0.064580</td>
          <td>23.716203</td>
          <td>0.073599</td>
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
          <td>26.084518</td>
          <td>0.270898</td>
          <td>26.164561</td>
          <td>0.103501</td>
          <td>26.243566</td>
          <td>0.097696</td>
          <td>25.858484</td>
          <td>0.113202</td>
          <td>26.209784</td>
          <td>0.282977</td>
          <td>25.650582</td>
          <td>0.376682</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.102334</td>
          <td>0.590424</td>
          <td>27.053334</td>
          <td>0.221395</td>
          <td>27.078960</td>
          <td>0.200505</td>
          <td>26.498559</td>
          <td>0.196036</td>
          <td>25.873442</td>
          <td>0.214569</td>
          <td>25.292709</td>
          <td>0.283476</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.718173</td>
          <td>0.891772</td>
          <td>27.222297</td>
          <td>0.254549</td>
          <td>26.698063</td>
          <td>0.145024</td>
          <td>26.528780</td>
          <td>0.201079</td>
          <td>26.194631</td>
          <td>0.279522</td>
          <td>25.668867</td>
          <td>0.382071</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.976902</td>
          <td>1.760047</td>
          <td>27.735596</td>
          <td>0.383532</td>
          <td>26.647251</td>
          <td>0.138814</td>
          <td>25.645427</td>
          <td>0.093948</td>
          <td>25.560223</td>
          <td>0.164719</td>
          <td>25.190404</td>
          <td>0.260827</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.299503</td>
          <td>1.254365</td>
          <td>26.656319</td>
          <td>0.158376</td>
          <td>26.140990</td>
          <td>0.089279</td>
          <td>25.431841</td>
          <td>0.077836</td>
          <td>25.475029</td>
          <td>0.153147</td>
          <td>25.022417</td>
          <td>0.227106</td>
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
          <td>30.121604</td>
          <td>2.878315</td>
          <td>26.620472</td>
          <td>0.176446</td>
          <td>25.899673</td>
          <td>0.084893</td>
          <td>25.303531</td>
          <td>0.082332</td>
          <td>24.783022</td>
          <td>0.098545</td>
          <td>25.205444</td>
          <td>0.307913</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.145867</td>
          <td>0.587526</td>
          <td>27.075747</td>
          <td>0.233140</td>
          <td>27.440391</td>
          <td>0.483971</td>
          <td>27.111113</td>
          <td>0.643267</td>
          <td>25.650935</td>
          <td>0.436023</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.336768</td>
          <td>0.374039</td>
          <td>26.123726</td>
          <td>0.117451</td>
          <td>24.787917</td>
          <td>0.032463</td>
          <td>23.864463</td>
          <td>0.023690</td>
          <td>23.143971</td>
          <td>0.023774</td>
          <td>22.838263</td>
          <td>0.040958</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.312644</td>
          <td>0.777615</td>
          <td>27.812690</td>
          <td>0.484904</td>
          <td>28.563696</td>
          <td>0.758331</td>
          <td>26.883839</td>
          <td>0.335260</td>
          <td>26.497478</td>
          <td>0.435045</td>
          <td>25.496562</td>
          <td>0.411401</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.100756</td>
          <td>0.305799</td>
          <td>25.774988</td>
          <td>0.084881</td>
          <td>25.381554</td>
          <td>0.053692</td>
          <td>24.873248</td>
          <td>0.056281</td>
          <td>24.289935</td>
          <td>0.063818</td>
          <td>23.663869</td>
          <td>0.083160</td>
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
          <td>26.735819</td>
          <td>0.505528</td>
          <td>26.489310</td>
          <td>0.160744</td>
          <td>26.236766</td>
          <td>0.116464</td>
          <td>26.027810</td>
          <td>0.157980</td>
          <td>26.317644</td>
          <td>0.363759</td>
          <td>25.619077</td>
          <td>0.433527</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.971459</td>
          <td>0.237569</td>
          <td>26.532994</td>
          <td>0.147994</td>
          <td>26.680681</td>
          <td>0.268310</td>
          <td>25.832502</td>
          <td>0.242485</td>
          <td>25.471790</td>
          <td>0.381383</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.611901</td>
          <td>0.458630</td>
          <td>26.755068</td>
          <td>0.199866</td>
          <td>26.524366</td>
          <td>0.148144</td>
          <td>26.418981</td>
          <td>0.218054</td>
          <td>25.933390</td>
          <td>0.265531</td>
          <td>25.220167</td>
          <td>0.315308</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.441385</td>
          <td>0.354939</td>
          <td>26.536120</td>
          <td>0.152394</td>
          <td>25.794441</td>
          <td>0.130614</td>
          <td>26.020629</td>
          <td>0.289962</td>
          <td>24.982099</td>
          <td>0.264764</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.970888</td>
          <td>0.594930</td>
          <td>26.352608</td>
          <td>0.141598</td>
          <td>26.080843</td>
          <td>0.100541</td>
          <td>25.588345</td>
          <td>0.106821</td>
          <td>25.034669</td>
          <td>0.123957</td>
          <td>25.081471</td>
          <td>0.281283</td>
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
          <td>26.131952</td>
          <td>0.281548</td>
          <td>26.637815</td>
          <td>0.155907</td>
          <td>26.043222</td>
          <td>0.081923</td>
          <td>25.383699</td>
          <td>0.074605</td>
          <td>25.104691</td>
          <td>0.111179</td>
          <td>25.167622</td>
          <td>0.256039</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.355124</td>
          <td>0.607604</td>
          <td>27.347775</td>
          <td>0.250911</td>
          <td>27.052372</td>
          <td>0.309402</td>
          <td>27.984430</td>
          <td>1.006542</td>
          <td>25.736684</td>
          <td>0.402969</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.494873</td>
          <td>0.395403</td>
          <td>25.856610</td>
          <td>0.084901</td>
          <td>24.784774</td>
          <td>0.029217</td>
          <td>23.859534</td>
          <td>0.021251</td>
          <td>23.139842</td>
          <td>0.021425</td>
          <td>22.825780</td>
          <td>0.036421</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.185191</td>
          <td>0.712825</td>
          <td>30.822466</td>
          <td>2.496431</td>
          <td>27.613790</td>
          <td>0.380664</td>
          <td>26.610327</td>
          <td>0.268190</td>
          <td>26.130441</td>
          <td>0.326043</td>
          <td>25.554892</td>
          <td>0.428768</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.315733</td>
          <td>0.326510</td>
          <td>25.688953</td>
          <td>0.068200</td>
          <td>25.517965</td>
          <td>0.051510</td>
          <td>24.790322</td>
          <td>0.044135</td>
          <td>24.357397</td>
          <td>0.057609</td>
          <td>23.656223</td>
          <td>0.069899</td>
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
          <td>28.188672</td>
          <td>1.222174</td>
          <td>26.360089</td>
          <td>0.131351</td>
          <td>26.116703</td>
          <td>0.094559</td>
          <td>26.227227</td>
          <td>0.168670</td>
          <td>25.813729</td>
          <td>0.219958</td>
          <td>25.285501</td>
          <td>0.303729</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.056336</td>
          <td>0.576741</td>
          <td>27.175242</td>
          <td>0.248202</td>
          <td>26.608579</td>
          <td>0.136449</td>
          <td>26.314398</td>
          <td>0.170563</td>
          <td>26.052478</td>
          <td>0.252736</td>
          <td>25.850764</td>
          <td>0.445758</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.131575</td>
          <td>0.245895</td>
          <td>27.631485</td>
          <td>0.329787</td>
          <td>26.221066</td>
          <td>0.162796</td>
          <td>26.579722</td>
          <td>0.396273</td>
          <td>25.924036</td>
          <td>0.484581</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.992158</td>
          <td>0.583461</td>
          <td>27.328309</td>
          <td>0.304478</td>
          <td>26.615765</td>
          <td>0.151157</td>
          <td>25.999575</td>
          <td>0.143959</td>
          <td>25.868802</td>
          <td>0.238172</td>
          <td>25.041274</td>
          <td>0.257831</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.455743</td>
          <td>0.137822</td>
          <td>26.106701</td>
          <td>0.090071</td>
          <td>25.524332</td>
          <td>0.087982</td>
          <td>25.194384</td>
          <td>0.124902</td>
          <td>24.742435</td>
          <td>0.186680</td>
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
