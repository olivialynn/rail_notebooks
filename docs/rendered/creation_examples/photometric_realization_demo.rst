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

    <pzflow.flow.Flow at 0x7fd79e43d540>



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
          <td>29.294857</td>
          <td>2.022061</td>
          <td>26.461136</td>
          <td>0.133926</td>
          <td>26.153037</td>
          <td>0.090230</td>
          <td>25.274338</td>
          <td>0.067714</td>
          <td>24.984057</td>
          <td>0.100038</td>
          <td>25.258703</td>
          <td>0.275762</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.290993</td>
          <td>1.248544</td>
          <td>27.484771</td>
          <td>0.314810</td>
          <td>27.447325</td>
          <td>0.271952</td>
          <td>27.166697</td>
          <td>0.338568</td>
          <td>26.354739</td>
          <td>0.317956</td>
          <td>25.998955</td>
          <td>0.490809</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.859738</td>
          <td>0.225232</td>
          <td>25.975519</td>
          <td>0.087696</td>
          <td>24.820345</td>
          <td>0.027767</td>
          <td>23.870193</td>
          <td>0.019726</td>
          <td>23.110482</td>
          <td>0.019296</td>
          <td>22.858603</td>
          <td>0.034417</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.216534</td>
          <td>0.550015</td>
          <td>27.908957</td>
          <td>0.392376</td>
          <td>26.533072</td>
          <td>0.201805</td>
          <td>25.793835</td>
          <td>0.200735</td>
          <td>25.860427</td>
          <td>0.442470</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.196783</td>
          <td>0.631009</td>
          <td>25.794366</td>
          <td>0.074762</td>
          <td>25.472663</td>
          <td>0.049408</td>
          <td>24.842406</td>
          <td>0.046154</td>
          <td>24.408637</td>
          <td>0.060202</td>
          <td>23.705891</td>
          <td>0.072931</td>
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
          <td>26.598132</td>
          <td>0.406699</td>
          <td>26.168724</td>
          <td>0.103878</td>
          <td>26.220998</td>
          <td>0.095781</td>
          <td>26.241412</td>
          <td>0.157599</td>
          <td>25.847781</td>
          <td>0.210018</td>
          <td>25.913522</td>
          <td>0.460524</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.604951</td>
          <td>0.408831</td>
          <td>26.888782</td>
          <td>0.192909</td>
          <td>26.720684</td>
          <td>0.147872</td>
          <td>26.342092</td>
          <td>0.171731</td>
          <td>26.339824</td>
          <td>0.314192</td>
          <td>25.239520</td>
          <td>0.271493</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.526772</td>
          <td>1.415003</td>
          <td>27.445786</td>
          <td>0.305139</td>
          <td>27.186018</td>
          <td>0.219287</td>
          <td>26.557240</td>
          <td>0.205936</td>
          <td>26.399670</td>
          <td>0.329531</td>
          <td>25.932005</td>
          <td>0.466946</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.931450</td>
          <td>1.016293</td>
          <td>26.961550</td>
          <td>0.205067</td>
          <td>26.600641</td>
          <td>0.133339</td>
          <td>26.006562</td>
          <td>0.128747</td>
          <td>25.748559</td>
          <td>0.193235</td>
          <td>26.176940</td>
          <td>0.558935</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.191110</td>
          <td>0.628514</td>
          <td>26.598800</td>
          <td>0.150767</td>
          <td>25.954903</td>
          <td>0.075768</td>
          <td>25.650672</td>
          <td>0.094381</td>
          <td>25.306594</td>
          <td>0.132473</td>
          <td>24.792085</td>
          <td>0.187263</td>
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
          <td>27.040938</td>
          <td>0.621213</td>
          <td>26.773803</td>
          <td>0.200807</td>
          <td>26.208474</td>
          <td>0.111286</td>
          <td>25.337558</td>
          <td>0.084838</td>
          <td>24.970066</td>
          <td>0.116033</td>
          <td>24.654855</td>
          <td>0.195780</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.360367</td>
          <td>0.682322</td>
          <td>27.525566</td>
          <td>0.335703</td>
          <td>28.273741</td>
          <td>0.860886</td>
          <td>27.312763</td>
          <td>0.738037</td>
          <td>26.497633</td>
          <td>0.793902</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.634559</td>
          <td>0.469388</td>
          <td>26.124990</td>
          <td>0.117580</td>
          <td>24.745535</td>
          <td>0.031276</td>
          <td>23.919732</td>
          <td>0.024848</td>
          <td>23.111985</td>
          <td>0.023128</td>
          <td>22.859178</td>
          <td>0.041724</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.266298</td>
          <td>0.754178</td>
          <td>26.955464</td>
          <td>0.247369</td>
          <td>27.600644</td>
          <td>0.377987</td>
          <td>26.551988</td>
          <td>0.256580</td>
          <td>26.556685</td>
          <td>0.454939</td>
          <td>24.940983</td>
          <td>0.264883</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.314977</td>
          <td>0.362292</td>
          <td>25.806265</td>
          <td>0.087247</td>
          <td>25.359037</td>
          <td>0.052630</td>
          <td>24.868140</td>
          <td>0.056026</td>
          <td>24.277685</td>
          <td>0.063129</td>
          <td>23.700359</td>
          <td>0.085876</td>
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
          <td>26.342198</td>
          <td>0.375279</td>
          <td>26.266061</td>
          <td>0.132705</td>
          <td>26.020776</td>
          <td>0.096435</td>
          <td>26.280208</td>
          <td>0.195709</td>
          <td>25.674796</td>
          <td>0.216163</td>
          <td>26.539460</td>
          <td>0.828687</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.468190</td>
          <td>0.829991</td>
          <td>26.848008</td>
          <td>0.214434</td>
          <td>27.284621</td>
          <td>0.277720</td>
          <td>26.308878</td>
          <td>0.197173</td>
          <td>25.624551</td>
          <td>0.203980</td>
          <td>25.526186</td>
          <td>0.397770</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.025386</td>
          <td>0.619269</td>
          <td>27.308840</td>
          <td>0.314761</td>
          <td>27.045096</td>
          <td>0.230006</td>
          <td>26.210288</td>
          <td>0.182999</td>
          <td>25.772515</td>
          <td>0.232635</td>
          <td>24.968088</td>
          <td>0.257131</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.286857</td>
          <td>0.748674</td>
          <td>27.172079</td>
          <td>0.286386</td>
          <td>26.716751</td>
          <td>0.177769</td>
          <td>25.884791</td>
          <td>0.141209</td>
          <td>25.885787</td>
          <td>0.259859</td>
          <td>25.619163</td>
          <td>0.437573</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.700351</td>
          <td>1.653517</td>
          <td>26.535633</td>
          <td>0.165618</td>
          <td>25.933237</td>
          <td>0.088324</td>
          <td>25.736865</td>
          <td>0.121574</td>
          <td>25.301205</td>
          <td>0.155969</td>
          <td>25.237419</td>
          <td>0.318861</td>
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
          <td>27.127915</td>
          <td>0.601261</td>
          <td>26.675531</td>
          <td>0.161014</td>
          <td>25.979013</td>
          <td>0.077409</td>
          <td>25.371182</td>
          <td>0.073784</td>
          <td>25.055701</td>
          <td>0.106524</td>
          <td>24.772259</td>
          <td>0.184175</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.093629</td>
          <td>0.503190</td>
          <td>27.879521</td>
          <td>0.383857</td>
          <td>27.905216</td>
          <td>0.590693</td>
          <td>25.931264</td>
          <td>0.225354</td>
          <td>27.072343</td>
          <td>1.010898</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.373772</td>
          <td>0.359910</td>
          <td>25.916750</td>
          <td>0.089509</td>
          <td>24.794383</td>
          <td>0.029464</td>
          <td>23.901710</td>
          <td>0.022031</td>
          <td>23.139868</td>
          <td>0.021425</td>
          <td>22.824291</td>
          <td>0.036373</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.773613</td>
          <td>1.035420</td>
          <td>28.013055</td>
          <td>0.559859</td>
          <td>27.143731</td>
          <td>0.261618</td>
          <td>26.559183</td>
          <td>0.257213</td>
          <td>25.616102</td>
          <td>0.214317</td>
          <td>25.438761</td>
          <td>0.392250</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.680497</td>
          <td>0.433437</td>
          <td>25.746810</td>
          <td>0.071777</td>
          <td>25.409111</td>
          <td>0.046765</td>
          <td>24.780526</td>
          <td>0.043753</td>
          <td>24.367485</td>
          <td>0.058127</td>
          <td>23.725053</td>
          <td>0.074287</td>
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
          <td>26.618659</td>
          <td>0.433636</td>
          <td>26.600385</td>
          <td>0.161463</td>
          <td>26.268746</td>
          <td>0.108025</td>
          <td>25.944534</td>
          <td>0.132331</td>
          <td>25.860679</td>
          <td>0.228709</td>
          <td>25.596663</td>
          <td>0.388211</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.118442</td>
          <td>0.602742</td>
          <td>27.129452</td>
          <td>0.239014</td>
          <td>26.781342</td>
          <td>0.158288</td>
          <td>26.608556</td>
          <td>0.218520</td>
          <td>26.576988</td>
          <td>0.384326</td>
          <td>25.694598</td>
          <td>0.395666</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.985773</td>
          <td>2.666221</td>
          <td>27.329581</td>
          <td>0.288980</td>
          <td>26.823503</td>
          <td>0.169337</td>
          <td>26.213876</td>
          <td>0.161800</td>
          <td>26.119129</td>
          <td>0.275037</td>
          <td>25.300313</td>
          <td>0.298823</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.816734</td>
          <td>1.706827</td>
          <td>27.129318</td>
          <td>0.259138</td>
          <td>26.817508</td>
          <td>0.179530</td>
          <td>25.823022</td>
          <td>0.123588</td>
          <td>25.913687</td>
          <td>0.247149</td>
          <td>25.657284</td>
          <td>0.420136</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.281294</td>
          <td>0.683405</td>
          <td>26.209279</td>
          <td>0.111311</td>
          <td>26.096305</td>
          <td>0.089251</td>
          <td>25.710283</td>
          <td>0.103577</td>
          <td>25.354323</td>
          <td>0.143413</td>
          <td>24.815985</td>
          <td>0.198615</td>
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
