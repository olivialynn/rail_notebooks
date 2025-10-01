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

    <pzflow.flow.Flow at 0x7f60686ddba0>



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
    0      23.994413  0.173845  0.110514  
    1      25.391064  0.082512  0.056458  
    2      24.304707  0.020406  0.012861  
    3      25.291103  0.053841  0.044787  
    4      25.096743  0.048765  0.044240  
    ...          ...       ...       ...  
    99995  24.737946  0.072753  0.062262  
    99996  24.224169  0.014903  0.007727  
    99997  25.613836  0.076345  0.053893  
    99998  25.274899  0.024394  0.013072  
    99999  25.699642  0.011350  0.005693  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>27.333562</td>
          <td>0.693377</td>
          <td>26.644107</td>
          <td>0.156731</td>
          <td>26.067239</td>
          <td>0.083665</td>
          <td>25.268876</td>
          <td>0.067387</td>
          <td>24.717764</td>
          <td>0.079149</td>
          <td>23.912467</td>
          <td>0.087513</td>
          <td>0.173845</td>
          <td>0.110514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.571584</td>
          <td>2.076347</td>
          <td>26.675573</td>
          <td>0.142244</td>
          <td>26.395435</td>
          <td>0.179686</td>
          <td>25.536981</td>
          <td>0.161484</td>
          <td>25.378483</td>
          <td>0.303776</td>
          <td>0.082512</td>
          <td>0.056458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.816480</td>
          <td>0.365166</td>
          <td>26.088291</td>
          <td>0.138171</td>
          <td>25.146807</td>
          <td>0.115321</td>
          <td>24.174095</td>
          <td>0.110071</td>
          <td>0.020406</td>
          <td>0.012861</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.766825</td>
          <td>0.462188</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.386116</td>
          <td>0.258697</td>
          <td>26.304202</td>
          <td>0.166280</td>
          <td>25.139697</td>
          <td>0.114609</td>
          <td>25.175859</td>
          <td>0.257740</td>
          <td>0.053841</td>
          <td>0.044787</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.036248</td>
          <td>0.260449</td>
          <td>26.208974</td>
          <td>0.107594</td>
          <td>25.962846</td>
          <td>0.076301</td>
          <td>25.896463</td>
          <td>0.117009</td>
          <td>25.471555</td>
          <td>0.152692</td>
          <td>25.387718</td>
          <td>0.306035</td>
          <td>0.048765</td>
          <td>0.044240</td>
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
          <td>26.584382</td>
          <td>0.402428</td>
          <td>26.476751</td>
          <td>0.135743</td>
          <td>25.461230</td>
          <td>0.048909</td>
          <td>25.074018</td>
          <td>0.056692</td>
          <td>25.004799</td>
          <td>0.101872</td>
          <td>24.517320</td>
          <td>0.148177</td>
          <td>0.072753</td>
          <td>0.062262</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.395800</td>
          <td>0.723175</td>
          <td>27.059853</td>
          <td>0.222598</td>
          <td>26.026610</td>
          <td>0.080720</td>
          <td>25.228057</td>
          <td>0.064993</td>
          <td>24.743367</td>
          <td>0.080957</td>
          <td>24.379639</td>
          <td>0.131595</td>
          <td>0.014903</td>
          <td>0.007727</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.557013</td>
          <td>0.394038</td>
          <td>26.976941</td>
          <td>0.207726</td>
          <td>26.644044</td>
          <td>0.138431</td>
          <td>26.088137</td>
          <td>0.138153</td>
          <td>25.730641</td>
          <td>0.190339</td>
          <td>25.839051</td>
          <td>0.435366</td>
          <td>0.076345</td>
          <td>0.053893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.714938</td>
          <td>0.444503</td>
          <td>26.222475</td>
          <td>0.108868</td>
          <td>26.127523</td>
          <td>0.088227</td>
          <td>25.925980</td>
          <td>0.120052</td>
          <td>25.728190</td>
          <td>0.189945</td>
          <td>25.149542</td>
          <td>0.252237</td>
          <td>0.024394</td>
          <td>0.013072</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.870666</td>
          <td>0.499277</td>
          <td>26.810014</td>
          <td>0.180497</td>
          <td>26.416488</td>
          <td>0.113641</td>
          <td>26.316639</td>
          <td>0.168051</td>
          <td>26.205744</td>
          <td>0.282052</td>
          <td>25.191225</td>
          <td>0.261002</td>
          <td>0.011350</td>
          <td>0.005693</td>
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
          <td>28.767983</td>
          <td>1.746421</td>
          <td>26.786556</td>
          <td>0.214989</td>
          <td>25.998189</td>
          <td>0.098948</td>
          <td>25.283820</td>
          <td>0.086698</td>
          <td>24.703795</td>
          <td>0.098252</td>
          <td>24.038742</td>
          <td>0.123531</td>
          <td>0.173845</td>
          <td>0.110514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.066039</td>
          <td>0.561688</td>
          <td>26.568573</td>
          <td>0.154454</td>
          <td>26.151464</td>
          <td>0.174776</td>
          <td>25.609763</td>
          <td>0.203883</td>
          <td>26.156330</td>
          <td>0.638821</td>
          <td>0.082512</td>
          <td>0.056458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.301560</td>
          <td>0.655641</td>
          <td>28.475020</td>
          <td>0.679045</td>
          <td>25.900995</td>
          <td>0.138836</td>
          <td>25.134240</td>
          <td>0.133912</td>
          <td>24.077883</td>
          <td>0.119570</td>
          <td>0.020406</td>
          <td>0.012861</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.353752</td>
          <td>0.772174</td>
          <td>28.748936</td>
          <td>0.885485</td>
          <td>28.308205</td>
          <td>0.608227</td>
          <td>26.228322</td>
          <td>0.184975</td>
          <td>25.966189</td>
          <td>0.271582</td>
          <td>26.998062</td>
          <td>1.087722</td>
          <td>0.053841</td>
          <td>0.044787</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.856866</td>
          <td>0.252197</td>
          <td>25.910073</td>
          <td>0.096169</td>
          <td>25.929088</td>
          <td>0.087761</td>
          <td>25.595412</td>
          <td>0.107180</td>
          <td>25.666299</td>
          <td>0.211863</td>
          <td>25.547672</td>
          <td>0.405540</td>
          <td>0.048765</td>
          <td>0.044240</td>
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
          <td>28.530583</td>
          <td>1.526603</td>
          <td>26.509701</td>
          <td>0.162754</td>
          <td>25.360492</td>
          <td>0.053506</td>
          <td>25.061869</td>
          <td>0.067576</td>
          <td>24.803335</td>
          <td>0.101860</td>
          <td>25.367579</td>
          <td>0.355188</td>
          <td>0.072753</td>
          <td>0.062262</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.828222</td>
          <td>0.533846</td>
          <td>26.592893</td>
          <td>0.172432</td>
          <td>25.948982</td>
          <td>0.088699</td>
          <td>25.156914</td>
          <td>0.072369</td>
          <td>24.812051</td>
          <td>0.101128</td>
          <td>24.260897</td>
          <td>0.140018</td>
          <td>0.014903</td>
          <td>0.007727</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.076000</td>
          <td>0.642243</td>
          <td>26.589443</td>
          <td>0.174065</td>
          <td>26.405844</td>
          <td>0.134005</td>
          <td>26.032605</td>
          <td>0.157611</td>
          <td>25.785036</td>
          <td>0.235460</td>
          <td>25.692581</td>
          <td>0.455701</td>
          <td>0.076345</td>
          <td>0.053893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.542227</td>
          <td>1.526188</td>
          <td>26.302545</td>
          <td>0.134571</td>
          <td>26.231469</td>
          <td>0.113682</td>
          <td>26.136415</td>
          <td>0.169915</td>
          <td>25.911387</td>
          <td>0.258022</td>
          <td>25.919713</td>
          <td>0.532908</td>
          <td>0.024394</td>
          <td>0.013072</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.929117</td>
          <td>1.829401</td>
          <td>26.465797</td>
          <td>0.154698</td>
          <td>26.578944</td>
          <td>0.153355</td>
          <td>26.117726</td>
          <td>0.167060</td>
          <td>25.625048</td>
          <td>0.203302</td>
          <td>25.142107</td>
          <td>0.292694</td>
          <td>0.011350</td>
          <td>0.005693</td>
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
          <td>27.064327</td>
          <td>0.268439</td>
          <td>25.996254</td>
          <td>0.097932</td>
          <td>25.115781</td>
          <td>0.074077</td>
          <td>24.645243</td>
          <td>0.092531</td>
          <td>23.920993</td>
          <td>0.110516</td>
          <td>0.173845</td>
          <td>0.110514</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.020646</td>
          <td>0.226998</td>
          <td>26.625085</td>
          <td>0.144872</td>
          <td>26.373091</td>
          <td>0.187886</td>
          <td>25.838220</td>
          <td>0.221183</td>
          <td>25.974907</td>
          <td>0.509356</td>
          <td>0.082512</td>
          <td>0.056458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.097651</td>
          <td>0.990466</td>
          <td>27.924828</td>
          <td>0.398584</td>
          <td>25.869415</td>
          <td>0.114755</td>
          <td>24.954754</td>
          <td>0.097885</td>
          <td>24.151567</td>
          <td>0.108367</td>
          <td>0.020406</td>
          <td>0.012861</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.327495</td>
          <td>1.151941</td>
          <td>26.937212</td>
          <td>0.183504</td>
          <td>26.208497</td>
          <td>0.158343</td>
          <td>25.334268</td>
          <td>0.140033</td>
          <td>24.961941</td>
          <td>0.222910</td>
          <td>0.053841</td>
          <td>0.044787</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.565396</td>
          <td>0.403584</td>
          <td>26.226863</td>
          <td>0.112020</td>
          <td>25.919250</td>
          <td>0.075562</td>
          <td>25.908299</td>
          <td>0.121782</td>
          <td>25.498364</td>
          <td>0.160643</td>
          <td>25.256519</td>
          <td>0.282920</td>
          <td>0.048765</td>
          <td>0.044240</td>
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
          <td>27.469671</td>
          <td>0.783546</td>
          <td>26.504695</td>
          <td>0.146196</td>
          <td>25.467300</td>
          <td>0.052177</td>
          <td>25.051948</td>
          <td>0.059150</td>
          <td>24.966209</td>
          <td>0.104397</td>
          <td>24.686495</td>
          <td>0.181566</td>
          <td>0.072753</td>
          <td>0.062262</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.013921</td>
          <td>1.068084</td>
          <td>26.928285</td>
          <td>0.199741</td>
          <td>26.006174</td>
          <td>0.079430</td>
          <td>25.123929</td>
          <td>0.059380</td>
          <td>24.796871</td>
          <td>0.085030</td>
          <td>24.187437</td>
          <td>0.111579</td>
          <td>0.014903</td>
          <td>0.007727</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.267694</td>
          <td>0.325350</td>
          <td>26.726881</td>
          <td>0.176297</td>
          <td>26.287700</td>
          <td>0.107334</td>
          <td>26.263639</td>
          <td>0.169964</td>
          <td>25.679886</td>
          <td>0.192351</td>
          <td>25.438307</td>
          <td>0.335728</td>
          <td>0.076345</td>
          <td>0.053893</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.891613</td>
          <td>0.508558</td>
          <td>26.211776</td>
          <td>0.108339</td>
          <td>25.989813</td>
          <td>0.078548</td>
          <td>25.836161</td>
          <td>0.111620</td>
          <td>25.777562</td>
          <td>0.198990</td>
          <td>25.510358</td>
          <td>0.339077</td>
          <td>0.024394</td>
          <td>0.013072</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.753442</td>
          <td>0.457870</td>
          <td>26.926052</td>
          <td>0.199232</td>
          <td>26.822458</td>
          <td>0.161514</td>
          <td>26.758050</td>
          <td>0.243606</td>
          <td>25.363144</td>
          <td>0.139252</td>
          <td>25.776664</td>
          <td>0.415572</td>
          <td>0.011350</td>
          <td>0.005693</td>
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
