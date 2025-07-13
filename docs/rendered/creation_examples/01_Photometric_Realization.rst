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

    <pzflow.flow.Flow at 0x7ff0d8699630>



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
    0      23.994413  0.002614  0.002511  
    1      25.391064  0.006463  0.005453  
    2      24.304707  0.002459  0.001559  
    3      25.291103  0.015208  0.008209  
    4      25.096743  0.215902  0.141471  
    ...          ...       ...       ...  
    99995  24.737946  0.077677  0.076822  
    99996  24.224169  0.004232  0.003448  
    99997  25.613836  0.010017  0.007454  
    99998  25.274899  0.032051  0.016944  
    99999  25.699642  0.029954  0.028916  
    
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
          <td>28.120223</td>
          <td>1.134845</td>
          <td>26.666057</td>
          <td>0.159699</td>
          <td>26.011464</td>
          <td>0.079649</td>
          <td>25.278872</td>
          <td>0.067986</td>
          <td>24.790605</td>
          <td>0.084400</td>
          <td>23.898062</td>
          <td>0.086410</td>
          <td>0.002614</td>
          <td>0.002511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.094011</td>
          <td>1.117928</td>
          <td>27.663317</td>
          <td>0.362539</td>
          <td>26.777949</td>
          <td>0.155316</td>
          <td>26.143230</td>
          <td>0.144867</td>
          <td>25.632353</td>
          <td>0.175147</td>
          <td>25.699028</td>
          <td>0.391100</td>
          <td>0.006463</td>
          <td>0.005453</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.424389</td>
          <td>0.575920</td>
          <td>25.872365</td>
          <td>0.114580</td>
          <td>25.114186</td>
          <td>0.112089</td>
          <td>24.343838</td>
          <td>0.127579</td>
          <td>0.002459</td>
          <td>0.001559</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.806557</td>
          <td>0.476110</td>
          <td>29.528772</td>
          <td>1.265641</td>
          <td>27.520897</td>
          <td>0.288674</td>
          <td>26.234207</td>
          <td>0.156630</td>
          <td>25.624134</td>
          <td>0.173928</td>
          <td>25.358493</td>
          <td>0.298935</td>
          <td>0.015208</td>
          <td>0.008209</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.165077</td>
          <td>0.289162</td>
          <td>26.002918</td>
          <td>0.089832</td>
          <td>25.892389</td>
          <td>0.071693</td>
          <td>25.721845</td>
          <td>0.100461</td>
          <td>25.562045</td>
          <td>0.164976</td>
          <td>25.358568</td>
          <td>0.298953</td>
          <td>0.215902</td>
          <td>0.141471</td>
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
          <td>28.239675</td>
          <td>1.213751</td>
          <td>26.833330</td>
          <td>0.184092</td>
          <td>25.521828</td>
          <td>0.051613</td>
          <td>25.167441</td>
          <td>0.061592</td>
          <td>24.807368</td>
          <td>0.085655</td>
          <td>24.972520</td>
          <td>0.217874</td>
          <td>0.077677</td>
          <td>0.076822</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.704152</td>
          <td>0.164975</td>
          <td>26.100693</td>
          <td>0.086167</td>
          <td>25.198431</td>
          <td>0.063308</td>
          <td>24.724552</td>
          <td>0.079624</td>
          <td>24.324220</td>
          <td>0.125428</td>
          <td>0.004232</td>
          <td>0.003448</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.032846</td>
          <td>0.561843</td>
          <td>26.688482</td>
          <td>0.162786</td>
          <td>26.376888</td>
          <td>0.109783</td>
          <td>26.351420</td>
          <td>0.173098</td>
          <td>25.631535</td>
          <td>0.175025</td>
          <td>27.195436</td>
          <td>1.086411</td>
          <td>0.010017</td>
          <td>0.007454</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.411439</td>
          <td>0.351829</td>
          <td>26.352201</td>
          <td>0.121875</td>
          <td>26.118836</td>
          <td>0.087555</td>
          <td>25.881444</td>
          <td>0.115490</td>
          <td>25.837169</td>
          <td>0.208161</td>
          <td>25.297221</td>
          <td>0.284513</td>
          <td>0.032051</td>
          <td>0.016944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.746136</td>
          <td>0.907516</td>
          <td>26.711547</td>
          <td>0.166018</td>
          <td>26.451640</td>
          <td>0.117173</td>
          <td>26.219207</td>
          <td>0.154631</td>
          <td>26.246238</td>
          <td>0.291442</td>
          <td>26.016928</td>
          <td>0.497377</td>
          <td>0.029954</td>
          <td>0.028916</td>
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
          <td>28.758867</td>
          <td>1.692661</td>
          <td>26.860489</td>
          <td>0.215903</td>
          <td>26.011303</td>
          <td>0.093648</td>
          <td>25.135692</td>
          <td>0.070990</td>
          <td>24.743579</td>
          <td>0.095195</td>
          <td>24.055480</td>
          <td>0.117148</td>
          <td>0.002614</td>
          <td>0.002511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.577562</td>
          <td>0.384911</td>
          <td>26.831727</td>
          <td>0.190104</td>
          <td>26.257408</td>
          <td>0.188042</td>
          <td>25.911755</td>
          <td>0.257816</td>
          <td>25.412722</td>
          <td>0.362874</td>
          <td>0.006463</td>
          <td>0.005453</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.247137</td>
          <td>1.939855</td>
          <td>28.597292</td>
          <td>0.737051</td>
          <td>25.882483</td>
          <td>0.136501</td>
          <td>24.914425</td>
          <td>0.110540</td>
          <td>24.342615</td>
          <td>0.150140</td>
          <td>0.002459</td>
          <td>0.001559</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.318431</td>
          <td>0.750992</td>
          <td>30.473239</td>
          <td>2.131637</td>
          <td>27.359235</td>
          <td>0.293999</td>
          <td>26.159685</td>
          <td>0.173172</td>
          <td>25.842126</td>
          <td>0.243571</td>
          <td>25.908651</td>
          <td>0.528275</td>
          <td>0.015208</td>
          <td>0.008209</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.008976</td>
          <td>1.208216</td>
          <td>26.315854</td>
          <td>0.148732</td>
          <td>25.990641</td>
          <td>0.101676</td>
          <td>25.649228</td>
          <td>0.123546</td>
          <td>25.225398</td>
          <td>0.159620</td>
          <td>24.655856</td>
          <td>0.216147</td>
          <td>0.215902</td>
          <td>0.141471</td>
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
          <td>29.853249</td>
          <td>2.646979</td>
          <td>26.242465</td>
          <td>0.129899</td>
          <td>25.341217</td>
          <td>0.052842</td>
          <td>25.126565</td>
          <td>0.071897</td>
          <td>24.952207</td>
          <td>0.116517</td>
          <td>25.304230</td>
          <td>0.339318</td>
          <td>0.077677</td>
          <td>0.076822</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.201836</td>
          <td>0.694201</td>
          <td>26.983813</td>
          <td>0.239161</td>
          <td>25.980711</td>
          <td>0.091168</td>
          <td>25.154194</td>
          <td>0.072163</td>
          <td>24.878895</td>
          <td>0.107169</td>
          <td>24.215570</td>
          <td>0.134591</td>
          <td>0.004232</td>
          <td>0.003448</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.170831</td>
          <td>0.323361</td>
          <td>26.408755</td>
          <td>0.147316</td>
          <td>26.210345</td>
          <td>0.111493</td>
          <td>26.584937</td>
          <td>0.247133</td>
          <td>25.788137</td>
          <td>0.232895</td>
          <td>25.532381</td>
          <td>0.398258</td>
          <td>0.010017</td>
          <td>0.007454</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.420430</td>
          <td>0.393699</td>
          <td>26.064249</td>
          <td>0.109534</td>
          <td>25.987450</td>
          <td>0.091910</td>
          <td>25.733681</td>
          <td>0.120269</td>
          <td>25.794337</td>
          <td>0.234527</td>
          <td>25.382335</td>
          <td>0.355021</td>
          <td>0.032051</td>
          <td>0.016944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.418370</td>
          <td>2.244993</td>
          <td>26.927608</td>
          <td>0.228860</td>
          <td>26.576122</td>
          <td>0.153383</td>
          <td>26.448032</td>
          <td>0.221241</td>
          <td>25.936240</td>
          <td>0.263722</td>
          <td>25.031773</td>
          <td>0.268324</td>
          <td>0.029954</td>
          <td>0.028916</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.822168</td>
          <td>0.182377</td>
          <td>25.970875</td>
          <td>0.076851</td>
          <td>25.125559</td>
          <td>0.059351</td>
          <td>24.686458</td>
          <td>0.076998</td>
          <td>23.981086</td>
          <td>0.092964</td>
          <td>0.002614</td>
          <td>0.002511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.013182</td>
          <td>0.473900</td>
          <td>26.589042</td>
          <td>0.132071</td>
          <td>26.531343</td>
          <td>0.201611</td>
          <td>25.689794</td>
          <td>0.183969</td>
          <td>25.073469</td>
          <td>0.237028</td>
          <td>0.006463</td>
          <td>0.005453</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.210609</td>
          <td>1.776923</td>
          <td>28.229001</td>
          <td>0.499744</td>
          <td>26.135777</td>
          <td>0.143950</td>
          <td>25.018936</td>
          <td>0.103146</td>
          <td>24.392855</td>
          <td>0.133115</td>
          <td>0.002459</td>
          <td>0.001559</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.063074</td>
          <td>0.198230</td>
          <td>26.251918</td>
          <td>0.159351</td>
          <td>25.319487</td>
          <td>0.134225</td>
          <td>24.889480</td>
          <td>0.203669</td>
          <td>0.015208</td>
          <td>0.008209</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.466087</td>
          <td>0.450742</td>
          <td>25.882829</td>
          <td>0.106812</td>
          <td>25.828123</td>
          <td>0.092447</td>
          <td>25.791230</td>
          <td>0.146475</td>
          <td>25.608470</td>
          <td>0.230619</td>
          <td>25.447465</td>
          <td>0.425611</td>
          <td>0.215902</td>
          <td>0.141471</td>
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
          <td>26.341794</td>
          <td>0.128862</td>
          <td>25.468008</td>
          <td>0.053086</td>
          <td>25.028269</td>
          <td>0.058934</td>
          <td>24.884267</td>
          <td>0.098774</td>
          <td>24.495359</td>
          <td>0.156864</td>
          <td>0.077677</td>
          <td>0.076822</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.813379</td>
          <td>0.946192</td>
          <td>26.606009</td>
          <td>0.151728</td>
          <td>25.982220</td>
          <td>0.077634</td>
          <td>25.159845</td>
          <td>0.061192</td>
          <td>24.918157</td>
          <td>0.094440</td>
          <td>24.222639</td>
          <td>0.114853</td>
          <td>0.004232</td>
          <td>0.003448</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.138196</td>
          <td>1.147072</td>
          <td>27.045527</td>
          <td>0.220153</td>
          <td>26.509893</td>
          <td>0.123387</td>
          <td>26.495677</td>
          <td>0.195769</td>
          <td>25.944847</td>
          <td>0.227934</td>
          <td>25.076936</td>
          <td>0.237841</td>
          <td>0.010017</td>
          <td>0.007454</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.488309</td>
          <td>0.375649</td>
          <td>26.446367</td>
          <td>0.133229</td>
          <td>25.987830</td>
          <td>0.078700</td>
          <td>25.810875</td>
          <td>0.109602</td>
          <td>25.641678</td>
          <td>0.178047</td>
          <td>25.076268</td>
          <td>0.239520</td>
          <td>0.032051</td>
          <td>0.016944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.437268</td>
          <td>0.361621</td>
          <td>27.007361</td>
          <td>0.215129</td>
          <td>26.708950</td>
          <td>0.148064</td>
          <td>26.113504</td>
          <td>0.142909</td>
          <td>26.025363</td>
          <td>0.246052</td>
          <td>25.963711</td>
          <td>0.483046</td>
          <td>0.029954</td>
          <td>0.028916</td>
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
