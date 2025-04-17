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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1e104d1f90>



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
          <td>26.477394</td>
          <td>0.370455</td>
          <td>26.719284</td>
          <td>0.167115</td>
          <td>25.974473</td>
          <td>0.077089</td>
          <td>25.269416</td>
          <td>0.067419</td>
          <td>24.803215</td>
          <td>0.085343</td>
          <td>24.015828</td>
          <td>0.095834</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.158603</td>
          <td>0.241562</td>
          <td>26.832028</td>
          <td>0.162667</td>
          <td>26.014062</td>
          <td>0.129585</td>
          <td>26.351270</td>
          <td>0.317077</td>
          <td>25.926644</td>
          <td>0.465076</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.611466</td>
          <td>1.323219</td>
          <td>33.635470</td>
          <td>4.835726</td>
          <td>26.009583</td>
          <td>0.129084</td>
          <td>25.123343</td>
          <td>0.112987</td>
          <td>24.566420</td>
          <td>0.154551</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.181434</td>
          <td>0.292999</td>
          <td>29.004533</td>
          <td>0.933484</td>
          <td>26.993393</td>
          <td>0.186562</td>
          <td>26.227848</td>
          <td>0.155780</td>
          <td>25.730920</td>
          <td>0.190383</td>
          <td>25.295693</td>
          <td>0.284161</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.903107</td>
          <td>0.233464</td>
          <td>26.090614</td>
          <td>0.097016</td>
          <td>25.904921</td>
          <td>0.072492</td>
          <td>25.765579</td>
          <td>0.104381</td>
          <td>25.618399</td>
          <td>0.173083</td>
          <td>25.102935</td>
          <td>0.242749</td>
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
          <td>27.241625</td>
          <td>0.650985</td>
          <td>26.388711</td>
          <td>0.125794</td>
          <td>25.364246</td>
          <td>0.044874</td>
          <td>25.054496</td>
          <td>0.055718</td>
          <td>24.915830</td>
          <td>0.094228</td>
          <td>24.773806</td>
          <td>0.184392</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.508788</td>
          <td>0.379609</td>
          <td>26.659556</td>
          <td>0.158815</td>
          <td>25.960463</td>
          <td>0.076141</td>
          <td>25.332190</td>
          <td>0.071272</td>
          <td>24.831913</td>
          <td>0.087527</td>
          <td>24.139080</td>
          <td>0.106756</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.447753</td>
          <td>0.748732</td>
          <td>26.657339</td>
          <td>0.158514</td>
          <td>26.294880</td>
          <td>0.102188</td>
          <td>26.371322</td>
          <td>0.176049</td>
          <td>26.208627</td>
          <td>0.282712</td>
          <td>25.131012</td>
          <td>0.248426</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.436230</td>
          <td>0.358735</td>
          <td>26.458512</td>
          <td>0.133623</td>
          <td>26.104760</td>
          <td>0.086477</td>
          <td>25.966412</td>
          <td>0.124343</td>
          <td>25.843165</td>
          <td>0.209208</td>
          <td>25.768965</td>
          <td>0.412722</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.642549</td>
          <td>0.420750</td>
          <td>27.095772</td>
          <td>0.229335</td>
          <td>26.559649</td>
          <td>0.128692</td>
          <td>26.519824</td>
          <td>0.199572</td>
          <td>25.662812</td>
          <td>0.179731</td>
          <td>25.594506</td>
          <td>0.360551</td>
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
          <td>29.828252</td>
          <td>2.608495</td>
          <td>27.101038</td>
          <td>0.263316</td>
          <td>26.085310</td>
          <td>0.099929</td>
          <td>25.156052</td>
          <td>0.072281</td>
          <td>24.538596</td>
          <td>0.079486</td>
          <td>23.974204</td>
          <td>0.109141</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.209018</td>
          <td>0.333297</td>
          <td>28.334359</td>
          <td>0.670270</td>
          <td>26.574658</td>
          <td>0.152787</td>
          <td>26.141437</td>
          <td>0.170460</td>
          <td>25.790890</td>
          <td>0.233420</td>
          <td>26.440484</td>
          <td>0.764643</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.911094</td>
          <td>0.917364</td>
          <td>25.858475</td>
          <td>0.136770</td>
          <td>25.002333</td>
          <td>0.121995</td>
          <td>24.608547</td>
          <td>0.192472</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.362826</td>
          <td>0.343519</td>
          <td>26.813199</td>
          <td>0.199578</td>
          <td>26.072048</td>
          <td>0.171816</td>
          <td>25.367701</td>
          <td>0.174465</td>
          <td>24.390962</td>
          <td>0.167330</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.898993</td>
          <td>0.259752</td>
          <td>26.217440</td>
          <td>0.124920</td>
          <td>25.970769</td>
          <td>0.090403</td>
          <td>25.747296</td>
          <td>0.121466</td>
          <td>25.769324</td>
          <td>0.229313</td>
          <td>25.472401</td>
          <td>0.380234</td>
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
          <td>27.227002</td>
          <td>0.714890</td>
          <td>26.419693</td>
          <td>0.151454</td>
          <td>25.440134</td>
          <td>0.057753</td>
          <td>24.937925</td>
          <td>0.060908</td>
          <td>24.761572</td>
          <td>0.098758</td>
          <td>24.750680</td>
          <td>0.216547</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.735321</td>
          <td>0.980767</td>
          <td>26.947613</td>
          <td>0.232932</td>
          <td>26.079795</td>
          <td>0.099862</td>
          <td>25.120706</td>
          <td>0.070362</td>
          <td>24.824507</td>
          <td>0.102617</td>
          <td>24.179524</td>
          <td>0.131012</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.062676</td>
          <td>0.635620</td>
          <td>26.450940</td>
          <td>0.154439</td>
          <td>26.445064</td>
          <td>0.138371</td>
          <td>26.128260</td>
          <td>0.170698</td>
          <td>26.494243</td>
          <td>0.414022</td>
          <td>25.500741</td>
          <td>0.393078</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.799680</td>
          <td>0.533233</td>
          <td>26.185505</td>
          <td>0.124932</td>
          <td>26.206116</td>
          <td>0.114575</td>
          <td>25.819582</td>
          <td>0.133484</td>
          <td>25.405661</td>
          <td>0.174069</td>
          <td>24.989973</td>
          <td>0.266470</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.812762</td>
          <td>0.531076</td>
          <td>26.515225</td>
          <td>0.162763</td>
          <td>26.585713</td>
          <td>0.155727</td>
          <td>26.369083</td>
          <td>0.208592</td>
          <td>26.245857</td>
          <td>0.340451</td>
          <td>25.169740</td>
          <td>0.302049</td>
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
          <td>27.379375</td>
          <td>0.715275</td>
          <td>26.478708</td>
          <td>0.135987</td>
          <td>25.853211</td>
          <td>0.069259</td>
          <td>25.295815</td>
          <td>0.069024</td>
          <td>24.522755</td>
          <td>0.066620</td>
          <td>23.881319</td>
          <td>0.085157</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.547912</td>
          <td>0.800143</td>
          <td>27.574892</td>
          <td>0.338432</td>
          <td>26.507919</td>
          <td>0.123163</td>
          <td>26.369280</td>
          <td>0.175913</td>
          <td>25.768378</td>
          <td>0.196664</td>
          <td>25.059905</td>
          <td>0.234490</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.464931</td>
          <td>0.206802</td>
          <td>24.832597</td>
          <td>0.095052</td>
          <td>24.140074</td>
          <td>0.116231</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.942528</td>
          <td>1.142270</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.857357</td>
          <td>0.206403</td>
          <td>26.391663</td>
          <td>0.224005</td>
          <td>25.775001</td>
          <td>0.244498</td>
          <td>25.964042</td>
          <td>0.579786</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.904124</td>
          <td>0.233877</td>
          <td>26.068012</td>
          <td>0.095230</td>
          <td>25.973859</td>
          <td>0.077158</td>
          <td>25.822457</td>
          <td>0.109863</td>
          <td>25.475618</td>
          <td>0.153437</td>
          <td>25.218206</td>
          <td>0.267184</td>
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
          <td>26.952813</td>
          <td>0.555217</td>
          <td>26.054658</td>
          <td>0.100709</td>
          <td>25.369359</td>
          <td>0.048822</td>
          <td>25.066135</td>
          <td>0.061191</td>
          <td>24.807111</td>
          <td>0.092658</td>
          <td>24.634353</td>
          <td>0.177254</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.338712</td>
          <td>0.335672</td>
          <td>26.894836</td>
          <td>0.196573</td>
          <td>26.054330</td>
          <td>0.084098</td>
          <td>25.276598</td>
          <td>0.069044</td>
          <td>24.884886</td>
          <td>0.093224</td>
          <td>24.184999</td>
          <td>0.113027</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.608265</td>
          <td>0.422331</td>
          <td>26.534299</td>
          <td>0.148732</td>
          <td>26.327907</td>
          <td>0.110429</td>
          <td>26.375513</td>
          <td>0.185619</td>
          <td>25.767760</td>
          <td>0.205768</td>
          <td>25.859385</td>
          <td>0.461757</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.024754</td>
          <td>0.278444</td>
          <td>26.052482</td>
          <td>0.103768</td>
          <td>26.041285</td>
          <td>0.091740</td>
          <td>26.018073</td>
          <td>0.146267</td>
          <td>25.767554</td>
          <td>0.218986</td>
          <td>25.272034</td>
          <td>0.310806</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.144436</td>
          <td>1.170951</td>
          <td>27.220655</td>
          <td>0.262363</td>
          <td>26.426102</td>
          <td>0.119102</td>
          <td>26.231492</td>
          <td>0.162593</td>
          <td>25.906104</td>
          <td>0.228744</td>
          <td>25.630549</td>
          <td>0.384334</td>
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
