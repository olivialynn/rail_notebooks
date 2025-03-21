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

    <pzflow.flow.Flow at 0x7fe69810bd30>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>27.609667</td>
          <td>0.832363</td>
          <td>26.348105</td>
          <td>0.121442</td>
          <td>26.113493</td>
          <td>0.087144</td>
          <td>25.132130</td>
          <td>0.059693</td>
          <td>24.757255</td>
          <td>0.081955</td>
          <td>23.973970</td>
          <td>0.092376</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.320060</td>
          <td>1.268485</td>
          <td>27.583324</td>
          <td>0.340443</td>
          <td>26.685366</td>
          <td>0.143448</td>
          <td>26.289067</td>
          <td>0.164148</td>
          <td>25.980106</td>
          <td>0.234457</td>
          <td>25.769618</td>
          <td>0.412928</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.991741</td>
          <td>1.053322</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.560526</td>
          <td>0.298048</td>
          <td>26.163866</td>
          <td>0.147461</td>
          <td>24.982212</td>
          <td>0.099877</td>
          <td>24.315081</td>
          <td>0.124437</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.983830</td>
          <td>1.048418</td>
          <td>28.822518</td>
          <td>0.832286</td>
          <td>27.870345</td>
          <td>0.380816</td>
          <td>26.368148</td>
          <td>0.175575</td>
          <td>25.591763</td>
          <td>0.169206</td>
          <td>25.407725</td>
          <td>0.310980</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.901389</td>
          <td>0.233133</td>
          <td>26.094478</td>
          <td>0.097345</td>
          <td>25.901912</td>
          <td>0.072300</td>
          <td>25.684891</td>
          <td>0.097259</td>
          <td>25.387673</td>
          <td>0.142074</td>
          <td>25.075945</td>
          <td>0.237401</td>
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
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.374226</td>
          <td>0.124225</td>
          <td>25.446290</td>
          <td>0.048265</td>
          <td>25.039406</td>
          <td>0.054977</td>
          <td>24.847609</td>
          <td>0.088744</td>
          <td>24.639364</td>
          <td>0.164494</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.421970</td>
          <td>1.339705</td>
          <td>26.572089</td>
          <td>0.147351</td>
          <td>26.084779</td>
          <td>0.084968</td>
          <td>25.209269</td>
          <td>0.063920</td>
          <td>24.672699</td>
          <td>0.076061</td>
          <td>24.185504</td>
          <td>0.111172</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.545027</td>
          <td>0.798253</td>
          <td>26.226905</td>
          <td>0.109290</td>
          <td>26.602184</td>
          <td>0.133517</td>
          <td>26.324439</td>
          <td>0.169171</td>
          <td>26.246556</td>
          <td>0.291517</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.846389</td>
          <td>0.490401</td>
          <td>26.711792</td>
          <td>0.166052</td>
          <td>26.177966</td>
          <td>0.092229</td>
          <td>25.912832</td>
          <td>0.118688</td>
          <td>25.986862</td>
          <td>0.235771</td>
          <td>25.046231</td>
          <td>0.231636</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.992453</td>
          <td>1.053764</td>
          <td>27.007242</td>
          <td>0.213053</td>
          <td>26.709432</td>
          <td>0.146449</td>
          <td>26.335050</td>
          <td>0.170706</td>
          <td>26.266044</td>
          <td>0.296133</td>
          <td>25.563742</td>
          <td>0.351951</td>
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
          <td>1.398944</td>
          <td>27.392547</td>
          <td>0.788388</td>
          <td>26.651274</td>
          <td>0.181110</td>
          <td>26.105466</td>
          <td>0.101708</td>
          <td>25.042734</td>
          <td>0.065383</td>
          <td>24.765970</td>
          <td>0.097084</td>
          <td>23.882693</td>
          <td>0.100751</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.510641</td>
          <td>0.365416</td>
          <td>26.728085</td>
          <td>0.174157</td>
          <td>25.894840</td>
          <td>0.137996</td>
          <td>25.781163</td>
          <td>0.231548</td>
          <td>25.441793</td>
          <td>0.371246</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.643447</td>
          <td>1.617567</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.707010</td>
          <td>0.805710</td>
          <td>25.981103</td>
          <td>0.151985</td>
          <td>25.178613</td>
          <td>0.142081</td>
          <td>24.363256</td>
          <td>0.156275</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.624711</td>
          <td>0.479891</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.916973</td>
          <td>0.217680</td>
          <td>26.164033</td>
          <td>0.185747</td>
          <td>25.480024</td>
          <td>0.191855</td>
          <td>24.742677</td>
          <td>0.224968</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.917111</td>
          <td>0.263621</td>
          <td>26.041728</td>
          <td>0.107223</td>
          <td>25.918304</td>
          <td>0.086326</td>
          <td>25.749525</td>
          <td>0.121702</td>
          <td>25.661352</td>
          <td>0.209594</td>
          <td>24.989521</td>
          <td>0.258580</td>
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
          <td>0.389450</td>
          <td>27.681210</td>
          <td>0.957461</td>
          <td>26.441014</td>
          <td>0.154245</td>
          <td>25.444603</td>
          <td>0.057982</td>
          <td>25.132951</td>
          <td>0.072388</td>
          <td>24.813935</td>
          <td>0.103390</td>
          <td>24.845388</td>
          <td>0.234269</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.847551</td>
          <td>1.766171</td>
          <td>26.999164</td>
          <td>0.243060</td>
          <td>26.151141</td>
          <td>0.106293</td>
          <td>25.202643</td>
          <td>0.075647</td>
          <td>24.821802</td>
          <td>0.102374</td>
          <td>24.108727</td>
          <td>0.123218</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.537296</td>
          <td>0.871559</td>
          <td>26.852101</td>
          <td>0.216761</td>
          <td>26.107369</td>
          <td>0.103184</td>
          <td>25.902224</td>
          <td>0.140663</td>
          <td>25.865941</td>
          <td>0.251266</td>
          <td>26.276225</td>
          <td>0.691629</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.140036</td>
          <td>0.322431</td>
          <td>26.190089</td>
          <td>0.125429</td>
          <td>26.009349</td>
          <td>0.096472</td>
          <td>26.068340</td>
          <td>0.165264</td>
          <td>25.484907</td>
          <td>0.186154</td>
          <td>24.823492</td>
          <td>0.232390</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.576596</td>
          <td>0.445808</td>
          <td>27.042970</td>
          <td>0.253222</td>
          <td>26.728663</td>
          <td>0.175906</td>
          <td>26.512636</td>
          <td>0.235053</td>
          <td>25.782865</td>
          <td>0.234031</td>
          <td>25.601641</td>
          <td>0.423672</td>
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
          <td>1.398944</td>
          <td>26.629573</td>
          <td>0.416638</td>
          <td>27.259595</td>
          <td>0.262466</td>
          <td>25.982913</td>
          <td>0.077676</td>
          <td>25.245580</td>
          <td>0.066019</td>
          <td>24.706479</td>
          <td>0.078374</td>
          <td>24.003848</td>
          <td>0.094845</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.692500</td>
          <td>0.371165</td>
          <td>26.693073</td>
          <td>0.144537</td>
          <td>26.309063</td>
          <td>0.167131</td>
          <td>25.531998</td>
          <td>0.160946</td>
          <td>24.961216</td>
          <td>0.216032</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.905547</td>
          <td>0.537750</td>
          <td>28.248744</td>
          <td>0.596748</td>
          <td>27.637229</td>
          <td>0.341261</td>
          <td>26.231476</td>
          <td>0.169802</td>
          <td>25.187133</td>
          <td>0.129485</td>
          <td>24.341755</td>
          <td>0.138425</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.098386</td>
          <td>2.747616</td>
          <td>27.172085</td>
          <td>0.267745</td>
          <td>26.253021</td>
          <td>0.199502</td>
          <td>25.690608</td>
          <td>0.228023</td>
          <td>24.873120</td>
          <td>0.249709</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.824036</td>
          <td>0.482741</td>
          <td>26.243344</td>
          <td>0.111003</td>
          <td>25.902020</td>
          <td>0.072410</td>
          <td>25.847783</td>
          <td>0.112317</td>
          <td>25.644404</td>
          <td>0.177190</td>
          <td>25.557003</td>
          <td>0.350554</td>
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
          <td>0.389450</td>
          <td>26.228422</td>
          <td>0.320129</td>
          <td>26.202883</td>
          <td>0.114612</td>
          <td>25.584153</td>
          <td>0.059075</td>
          <td>25.146664</td>
          <td>0.065718</td>
          <td>24.988483</td>
          <td>0.108609</td>
          <td>24.687217</td>
          <td>0.185368</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.559997</td>
          <td>0.812923</td>
          <td>26.902481</td>
          <td>0.197840</td>
          <td>26.058440</td>
          <td>0.084403</td>
          <td>25.369994</td>
          <td>0.074991</td>
          <td>24.900863</td>
          <td>0.094541</td>
          <td>24.400651</td>
          <td>0.136281</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.834547</td>
          <td>0.500409</td>
          <td>26.810015</td>
          <td>0.188071</td>
          <td>26.404118</td>
          <td>0.118010</td>
          <td>26.096301</td>
          <td>0.146294</td>
          <td>25.711171</td>
          <td>0.196221</td>
          <td>25.432750</td>
          <td>0.332166</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.883619</td>
          <td>0.248168</td>
          <td>26.238136</td>
          <td>0.121971</td>
          <td>26.077309</td>
          <td>0.094689</td>
          <td>25.745780</td>
          <td>0.115563</td>
          <td>25.674412</td>
          <td>0.202583</td>
          <td>25.065863</td>
          <td>0.263069</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.856780</td>
          <td>0.989973</td>
          <td>26.634052</td>
          <td>0.160604</td>
          <td>26.514799</td>
          <td>0.128631</td>
          <td>26.547844</td>
          <td>0.212414</td>
          <td>25.373939</td>
          <td>0.145854</td>
          <td>25.176536</td>
          <td>0.267750</td>
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
