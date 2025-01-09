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

    <pzflow.flow.Flow at 0x7fb85d130430>



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
          <td>26.900027</td>
          <td>0.510180</td>
          <td>26.558465</td>
          <td>0.145637</td>
          <td>26.104882</td>
          <td>0.086486</td>
          <td>25.368752</td>
          <td>0.073615</td>
          <td>25.148702</td>
          <td>0.115511</td>
          <td>25.408570</td>
          <td>0.311190</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>31.506463</td>
          <td>4.076039</td>
          <td>27.327247</td>
          <td>0.277304</td>
          <td>27.561621</td>
          <td>0.298311</td>
          <td>27.536698</td>
          <td>0.450612</td>
          <td>26.467548</td>
          <td>0.347701</td>
          <td>25.848250</td>
          <td>0.438412</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.773586</td>
          <td>0.464534</td>
          <td>25.895016</td>
          <td>0.081699</td>
          <td>24.850719</td>
          <td>0.028515</td>
          <td>23.875027</td>
          <td>0.019807</td>
          <td>23.113822</td>
          <td>0.019350</td>
          <td>22.852051</td>
          <td>0.034218</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.153751</td>
          <td>1.156695</td>
          <td>27.828320</td>
          <td>0.411940</td>
          <td>27.193676</td>
          <td>0.220690</td>
          <td>26.871118</td>
          <td>0.266986</td>
          <td>26.100760</td>
          <td>0.258935</td>
          <td>26.155375</td>
          <td>0.550312</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.873782</td>
          <td>0.500425</td>
          <td>25.814028</td>
          <td>0.076070</td>
          <td>25.391050</td>
          <td>0.045955</td>
          <td>24.823596</td>
          <td>0.045390</td>
          <td>24.319861</td>
          <td>0.055642</td>
          <td>23.688022</td>
          <td>0.071787</td>
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
          <td>26.597418</td>
          <td>0.406476</td>
          <td>26.823694</td>
          <td>0.182599</td>
          <td>26.134798</td>
          <td>0.088794</td>
          <td>26.053908</td>
          <td>0.134130</td>
          <td>25.839481</td>
          <td>0.208565</td>
          <td>25.403801</td>
          <td>0.310004</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.549336</td>
          <td>0.331408</td>
          <td>26.977138</td>
          <td>0.184015</td>
          <td>26.430848</td>
          <td>0.185154</td>
          <td>25.848268</td>
          <td>0.210103</td>
          <td>25.269112</td>
          <td>0.278103</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.828620</td>
          <td>0.483985</td>
          <td>27.048908</td>
          <td>0.220581</td>
          <td>27.075679</td>
          <td>0.199953</td>
          <td>26.360117</td>
          <td>0.174382</td>
          <td>25.825632</td>
          <td>0.206160</td>
          <td>25.076126</td>
          <td>0.237436</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.656027</td>
          <td>2.334403</td>
          <td>27.477243</td>
          <td>0.312922</td>
          <td>26.645222</td>
          <td>0.138571</td>
          <td>25.896831</td>
          <td>0.117047</td>
          <td>25.811790</td>
          <td>0.203782</td>
          <td>25.250961</td>
          <td>0.274032</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.083935</td>
          <td>0.270769</td>
          <td>26.678033</td>
          <td>0.161341</td>
          <td>26.023209</td>
          <td>0.080478</td>
          <td>25.649766</td>
          <td>0.094306</td>
          <td>25.308146</td>
          <td>0.132651</td>
          <td>24.622179</td>
          <td>0.162099</td>
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
          <td>27.102910</td>
          <td>0.648623</td>
          <td>26.958108</td>
          <td>0.234133</td>
          <td>25.990092</td>
          <td>0.091921</td>
          <td>25.261130</td>
          <td>0.079310</td>
          <td>24.953806</td>
          <td>0.114402</td>
          <td>24.451409</td>
          <td>0.164786</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.184761</td>
          <td>0.326951</td>
          <td>27.339546</td>
          <td>0.319259</td>
          <td>27.917200</td>
          <td>0.454310</td>
          <td>27.974083</td>
          <td>0.707167</td>
          <td>29.814107</td>
          <td>2.577899</td>
          <td>25.764495</td>
          <td>0.474890</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.166691</td>
          <td>0.327256</td>
          <td>26.056362</td>
          <td>0.110766</td>
          <td>24.792757</td>
          <td>0.032601</td>
          <td>23.890087</td>
          <td>0.024219</td>
          <td>23.157961</td>
          <td>0.024062</td>
          <td>22.850262</td>
          <td>0.041396</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.549516</td>
          <td>0.453660</td>
          <td>27.946196</td>
          <td>0.534871</td>
          <td>26.715265</td>
          <td>0.183766</td>
          <td>26.437296</td>
          <td>0.233455</td>
          <td>26.579011</td>
          <td>0.462629</td>
          <td>26.215164</td>
          <td>0.692728</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.593978</td>
          <td>0.448832</td>
          <td>25.831113</td>
          <td>0.089172</td>
          <td>25.375681</td>
          <td>0.053413</td>
          <td>24.857311</td>
          <td>0.055490</td>
          <td>24.442284</td>
          <td>0.073029</td>
          <td>23.800262</td>
          <td>0.093758</td>
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
          <td>26.059784</td>
          <td>0.300213</td>
          <td>26.300678</td>
          <td>0.136728</td>
          <td>26.124350</td>
          <td>0.105588</td>
          <td>26.114180</td>
          <td>0.170059</td>
          <td>26.339713</td>
          <td>0.370083</td>
          <td>24.560583</td>
          <td>0.184600</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.216974</td>
          <td>1.291960</td>
          <td>27.012199</td>
          <td>0.245682</td>
          <td>27.374933</td>
          <td>0.298749</td>
          <td>26.284521</td>
          <td>0.193173</td>
          <td>26.548244</td>
          <td>0.428227</td>
          <td>25.784917</td>
          <td>0.483830</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.620165</td>
          <td>0.401802</td>
          <td>26.675598</td>
          <td>0.168596</td>
          <td>26.868613</td>
          <td>0.314826</td>
          <td>26.119995</td>
          <td>0.308781</td>
          <td>25.138746</td>
          <td>0.295373</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.509251</td>
          <td>0.374273</td>
          <td>26.595475</td>
          <td>0.160334</td>
          <td>25.738051</td>
          <td>0.124387</td>
          <td>25.464938</td>
          <td>0.183038</td>
          <td>25.695516</td>
          <td>0.463484</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.767357</td>
          <td>1.002984</td>
          <td>26.689573</td>
          <td>0.188701</td>
          <td>25.972489</td>
          <td>0.091425</td>
          <td>25.558353</td>
          <td>0.104057</td>
          <td>25.197119</td>
          <td>0.142639</td>
          <td>24.845190</td>
          <td>0.231760</td>
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
          <td>26.631080</td>
          <td>0.155011</td>
          <td>26.235187</td>
          <td>0.096993</td>
          <td>25.249609</td>
          <td>0.066256</td>
          <td>24.995514</td>
          <td>0.101060</td>
          <td>25.040140</td>
          <td>0.230499</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.556737</td>
          <td>0.333604</td>
          <td>26.878032</td>
          <td>0.169328</td>
          <td>27.161690</td>
          <td>0.337532</td>
          <td>26.248266</td>
          <td>0.292173</td>
          <td>27.491668</td>
          <td>1.283634</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.232799</td>
          <td>0.322026</td>
          <td>26.028880</td>
          <td>0.098756</td>
          <td>24.755899</td>
          <td>0.028488</td>
          <td>23.896176</td>
          <td>0.021927</td>
          <td>23.163589</td>
          <td>0.021864</td>
          <td>22.812592</td>
          <td>0.035999</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.691179</td>
          <td>0.985521</td>
          <td>29.494849</td>
          <td>1.402844</td>
          <td>27.747075</td>
          <td>0.421785</td>
          <td>26.669894</td>
          <td>0.281495</td>
          <td>26.713580</td>
          <td>0.509682</td>
          <td>25.315646</td>
          <td>0.356398</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.700435</td>
          <td>0.440031</td>
          <td>25.925287</td>
          <td>0.084009</td>
          <td>25.466491</td>
          <td>0.049209</td>
          <td>24.896586</td>
          <td>0.048502</td>
          <td>24.428862</td>
          <td>0.061380</td>
          <td>23.656319</td>
          <td>0.069905</td>
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
          <td>26.465429</td>
          <td>0.385604</td>
          <td>26.436601</td>
          <td>0.140313</td>
          <td>26.282902</td>
          <td>0.109368</td>
          <td>26.151061</td>
          <td>0.158055</td>
          <td>25.776560</td>
          <td>0.213245</td>
          <td>25.936112</td>
          <td>0.501782</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.493735</td>
          <td>1.400485</td>
          <td>27.145228</td>
          <td>0.242145</td>
          <td>27.138182</td>
          <td>0.214031</td>
          <td>26.709065</td>
          <td>0.237526</td>
          <td>25.837813</td>
          <td>0.211561</td>
          <td>24.964993</td>
          <td>0.220052</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.077567</td>
          <td>0.596489</td>
          <td>27.201295</td>
          <td>0.260368</td>
          <td>26.709889</td>
          <td>0.153677</td>
          <td>26.589558</td>
          <td>0.222120</td>
          <td>25.849531</td>
          <td>0.220311</td>
          <td>25.933492</td>
          <td>0.487994</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.744696</td>
          <td>0.421852</td>
          <td>26.826288</td>
          <td>0.180871</td>
          <td>25.919847</td>
          <td>0.134396</td>
          <td>25.415209</td>
          <td>0.162675</td>
          <td>26.022457</td>
          <td>0.551121</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.593847</td>
          <td>0.415114</td>
          <td>26.881464</td>
          <td>0.198053</td>
          <td>26.130107</td>
          <td>0.091943</td>
          <td>25.649630</td>
          <td>0.098219</td>
          <td>25.161371</td>
          <td>0.121374</td>
          <td>24.832089</td>
          <td>0.201320</td>
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
