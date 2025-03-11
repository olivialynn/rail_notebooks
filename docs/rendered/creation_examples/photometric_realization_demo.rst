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

    <pzflow.flow.Flow at 0x7fe6fc3caf20>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.674873</td>
          <td>0.160906</td>
          <td>26.062302</td>
          <td>0.083302</td>
          <td>25.206126</td>
          <td>0.063742</td>
          <td>24.762561</td>
          <td>0.082339</td>
          <td>23.998419</td>
          <td>0.094381</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.622642</td>
          <td>0.839326</td>
          <td>27.445607</td>
          <td>0.305095</td>
          <td>26.853068</td>
          <td>0.165613</td>
          <td>26.156463</td>
          <td>0.146525</td>
          <td>25.879346</td>
          <td>0.215629</td>
          <td>25.498040</td>
          <td>0.334167</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.657798</td>
          <td>1.511949</td>
          <td>29.452058</td>
          <td>1.213433</td>
          <td>28.221944</td>
          <td>0.497123</td>
          <td>26.152388</td>
          <td>0.146013</td>
          <td>25.087722</td>
          <td>0.109531</td>
          <td>24.323448</td>
          <td>0.125344</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.975073</td>
          <td>1.043006</td>
          <td>28.342781</td>
          <td>0.601937</td>
          <td>27.663986</td>
          <td>0.323778</td>
          <td>26.624230</td>
          <td>0.217794</td>
          <td>25.785660</td>
          <td>0.199362</td>
          <td>25.346920</td>
          <td>0.296163</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.583298</td>
          <td>0.178675</td>
          <td>26.104008</td>
          <td>0.098160</td>
          <td>25.929773</td>
          <td>0.074103</td>
          <td>25.547829</td>
          <td>0.086220</td>
          <td>25.530643</td>
          <td>0.160612</td>
          <td>25.016654</td>
          <td>0.226022</td>
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
          <td>27.692568</td>
          <td>0.877511</td>
          <td>26.404032</td>
          <td>0.127474</td>
          <td>25.448492</td>
          <td>0.048359</td>
          <td>25.012458</td>
          <td>0.053677</td>
          <td>25.041319</td>
          <td>0.105180</td>
          <td>24.685900</td>
          <td>0.171145</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.600124</td>
          <td>0.150938</td>
          <td>26.170510</td>
          <td>0.091626</td>
          <td>25.168311</td>
          <td>0.061640</td>
          <td>24.746352</td>
          <td>0.081171</td>
          <td>24.178336</td>
          <td>0.110479</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.555689</td>
          <td>0.803813</td>
          <td>26.589459</td>
          <td>0.149564</td>
          <td>26.459015</td>
          <td>0.117927</td>
          <td>26.279210</td>
          <td>0.162772</td>
          <td>25.700562</td>
          <td>0.185565</td>
          <td>25.790761</td>
          <td>0.419658</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.768903</td>
          <td>0.920466</td>
          <td>25.942530</td>
          <td>0.085188</td>
          <td>25.950314</td>
          <td>0.075461</td>
          <td>25.748402</td>
          <td>0.102824</td>
          <td>25.851961</td>
          <td>0.210753</td>
          <td>25.363931</td>
          <td>0.300245</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.387002</td>
          <td>0.718909</td>
          <td>26.850532</td>
          <td>0.186787</td>
          <td>26.708733</td>
          <td>0.146361</td>
          <td>25.941015</td>
          <td>0.121631</td>
          <td>26.067580</td>
          <td>0.251987</td>
          <td>26.302525</td>
          <td>0.611210</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.864041</td>
          <td>0.216545</td>
          <td>26.005195</td>
          <td>0.093149</td>
          <td>25.100619</td>
          <td>0.068822</td>
          <td>24.871023</td>
          <td>0.106433</td>
          <td>24.140434</td>
          <td>0.126118</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.078275</td>
          <td>0.258506</td>
          <td>26.823068</td>
          <td>0.188741</td>
          <td>26.128745</td>
          <td>0.168629</td>
          <td>26.114346</td>
          <td>0.303881</td>
          <td>25.635196</td>
          <td>0.430845</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>31.528573</td>
          <td>4.247095</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.775601</td>
          <td>0.415896</td>
          <td>26.054130</td>
          <td>0.161784</td>
          <td>24.836322</td>
          <td>0.105570</td>
          <td>24.401351</td>
          <td>0.161448</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.521662</td>
          <td>0.444243</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.543532</td>
          <td>0.361517</td>
          <td>26.251495</td>
          <td>0.199949</td>
          <td>25.622246</td>
          <td>0.216149</td>
          <td>26.300138</td>
          <td>0.733625</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.132368</td>
          <td>0.662086</td>
          <td>26.164163</td>
          <td>0.119279</td>
          <td>25.914675</td>
          <td>0.086050</td>
          <td>25.909425</td>
          <td>0.139759</td>
          <td>25.312558</td>
          <td>0.156010</td>
          <td>24.796564</td>
          <td>0.220501</td>
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
          <td>28.004924</td>
          <td>1.158526</td>
          <td>26.373801</td>
          <td>0.145608</td>
          <td>25.466055</td>
          <td>0.059096</td>
          <td>25.170407</td>
          <td>0.074824</td>
          <td>24.846227</td>
          <td>0.106350</td>
          <td>24.951042</td>
          <td>0.255566</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.583285</td>
          <td>0.446347</td>
          <td>26.642659</td>
          <td>0.180447</td>
          <td>26.033407</td>
          <td>0.095883</td>
          <td>25.269398</td>
          <td>0.080239</td>
          <td>24.698182</td>
          <td>0.091859</td>
          <td>24.405939</td>
          <td>0.159174</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.277388</td>
          <td>0.735944</td>
          <td>26.916619</td>
          <td>0.228700</td>
          <td>26.588342</td>
          <td>0.156496</td>
          <td>26.246550</td>
          <td>0.188693</td>
          <td>26.138834</td>
          <td>0.313470</td>
          <td>25.006373</td>
          <td>0.265307</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.501017</td>
          <td>0.860559</td>
          <td>26.447075</td>
          <td>0.156479</td>
          <td>26.081660</td>
          <td>0.102780</td>
          <td>25.933626</td>
          <td>0.147267</td>
          <td>25.888478</td>
          <td>0.260432</td>
          <td>24.775740</td>
          <td>0.223364</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.337157</td>
          <td>0.764547</td>
          <td>26.846931</td>
          <td>0.215321</td>
          <td>26.676448</td>
          <td>0.168269</td>
          <td>26.103525</td>
          <td>0.166682</td>
          <td>26.200006</td>
          <td>0.328308</td>
          <td>25.611554</td>
          <td>0.426883</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.653905</td>
          <td>0.158067</td>
          <td>25.988745</td>
          <td>0.078077</td>
          <td>25.177444</td>
          <td>0.062150</td>
          <td>24.701316</td>
          <td>0.078018</td>
          <td>24.158549</td>
          <td>0.108602</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.144333</td>
          <td>0.608542</td>
          <td>27.898690</td>
          <td>0.434945</td>
          <td>26.690769</td>
          <td>0.144250</td>
          <td>26.234970</td>
          <td>0.156884</td>
          <td>25.800717</td>
          <td>0.202081</td>
          <td>25.068722</td>
          <td>0.236206</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.857526</td>
          <td>0.448208</td>
          <td>28.727843</td>
          <td>0.756915</td>
          <td>26.063929</td>
          <td>0.147134</td>
          <td>25.024439</td>
          <td>0.112419</td>
          <td>24.378241</td>
          <td>0.142846</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.994200</td>
          <td>0.231323</td>
          <td>25.978404</td>
          <td>0.158065</td>
          <td>25.838979</td>
          <td>0.257688</td>
          <td>25.783969</td>
          <td>0.508895</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.514923</td>
          <td>0.381751</td>
          <td>26.025212</td>
          <td>0.091721</td>
          <td>25.880532</td>
          <td>0.071046</td>
          <td>25.720582</td>
          <td>0.100499</td>
          <td>25.357437</td>
          <td>0.138612</td>
          <td>24.778716</td>
          <td>0.185422</td>
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
          <td>26.840536</td>
          <td>0.511701</td>
          <td>26.534243</td>
          <td>0.152586</td>
          <td>25.377694</td>
          <td>0.049185</td>
          <td>25.062944</td>
          <td>0.061018</td>
          <td>24.917480</td>
          <td>0.102073</td>
          <td>24.973868</td>
          <td>0.235589</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.241227</td>
          <td>0.310640</td>
          <td>27.011957</td>
          <td>0.216821</td>
          <td>25.987151</td>
          <td>0.079260</td>
          <td>25.275239</td>
          <td>0.068961</td>
          <td>24.772052</td>
          <td>0.084415</td>
          <td>24.192552</td>
          <td>0.113774</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.771544</td>
          <td>0.477595</td>
          <td>26.568131</td>
          <td>0.153109</td>
          <td>26.327068</td>
          <td>0.110348</td>
          <td>26.383981</td>
          <td>0.186952</td>
          <td>25.681238</td>
          <td>0.191336</td>
          <td>25.172355</td>
          <td>0.269412</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.868800</td>
          <td>0.245165</td>
          <td>26.231687</td>
          <td>0.121290</td>
          <td>26.134441</td>
          <td>0.099553</td>
          <td>26.069571</td>
          <td>0.152878</td>
          <td>26.056386</td>
          <td>0.277725</td>
          <td>25.335274</td>
          <td>0.326883</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.085170</td>
          <td>0.596251</td>
          <td>26.860190</td>
          <td>0.194542</td>
          <td>26.643162</td>
          <td>0.143707</td>
          <td>26.051179</td>
          <td>0.139286</td>
          <td>26.346894</td>
          <td>0.327296</td>
          <td>26.275498</td>
          <td>0.619382</td>
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
