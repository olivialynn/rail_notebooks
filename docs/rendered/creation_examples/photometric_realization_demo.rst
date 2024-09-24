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

    <pzflow.flow.Flow at 0x7f798d869900>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.555093</td>
          <td>0.145216</td>
          <td>26.096150</td>
          <td>0.085823</td>
          <td>25.335444</td>
          <td>0.071478</td>
          <td>24.995667</td>
          <td>0.101061</td>
          <td>24.953711</td>
          <td>0.214483</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.890547</td>
          <td>0.386828</td>
          <td>27.364205</td>
          <td>0.395064</td>
          <td>27.033183</td>
          <td>0.534031</td>
          <td>31.319456</td>
          <td>4.726751</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.558945</td>
          <td>0.805516</td>
          <td>26.002067</td>
          <td>0.089765</td>
          <td>24.730269</td>
          <td>0.025670</td>
          <td>23.888773</td>
          <td>0.020039</td>
          <td>23.132563</td>
          <td>0.019659</td>
          <td>22.783679</td>
          <td>0.032217</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.205417</td>
          <td>0.251048</td>
          <td>27.589788</td>
          <td>0.305139</td>
          <td>27.144666</td>
          <td>0.332714</td>
          <td>26.439911</td>
          <td>0.340202</td>
          <td>25.597941</td>
          <td>0.361522</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.766230</td>
          <td>0.461982</td>
          <td>25.851254</td>
          <td>0.078609</td>
          <td>25.318195</td>
          <td>0.043078</td>
          <td>24.936175</td>
          <td>0.050161</td>
          <td>24.469833</td>
          <td>0.063559</td>
          <td>23.696530</td>
          <td>0.072329</td>
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
          <td>26.633588</td>
          <td>0.417884</td>
          <td>26.583564</td>
          <td>0.148810</td>
          <td>26.164485</td>
          <td>0.091142</td>
          <td>26.401164</td>
          <td>0.180561</td>
          <td>26.254913</td>
          <td>0.293489</td>
          <td>25.827139</td>
          <td>0.431447</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.434747</td>
          <td>0.302448</td>
          <td>26.793270</td>
          <td>0.157367</td>
          <td>26.237709</td>
          <td>0.157100</td>
          <td>25.787078</td>
          <td>0.199599</td>
          <td>25.499150</td>
          <td>0.334461</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.333144</td>
          <td>0.278634</td>
          <td>27.070064</td>
          <td>0.199012</td>
          <td>26.319117</td>
          <td>0.168406</td>
          <td>26.026833</td>
          <td>0.243680</td>
          <td>25.244682</td>
          <td>0.272636</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.689197</td>
          <td>1.535627</td>
          <td>27.715357</td>
          <td>0.377554</td>
          <td>26.825100</td>
          <td>0.161707</td>
          <td>25.744951</td>
          <td>0.102514</td>
          <td>25.567283</td>
          <td>0.165714</td>
          <td>25.186523</td>
          <td>0.260000</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.573642</td>
          <td>0.147548</td>
          <td>26.116390</td>
          <td>0.087367</td>
          <td>25.666399</td>
          <td>0.095694</td>
          <td>25.194335</td>
          <td>0.120189</td>
          <td>24.752889</td>
          <td>0.181157</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.430690</td>
          <td>0.150083</td>
          <td>26.098922</td>
          <td>0.101127</td>
          <td>25.396206</td>
          <td>0.089332</td>
          <td>24.924323</td>
          <td>0.111501</td>
          <td>24.871578</td>
          <td>0.234588</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.053035</td>
          <td>1.060214</td>
          <td>27.723237</td>
          <td>0.391844</td>
          <td>26.966580</td>
          <td>0.336347</td>
          <td>26.611440</td>
          <td>0.447694</td>
          <td>26.352350</td>
          <td>0.720995</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.368623</td>
          <td>0.383401</td>
          <td>25.886093</td>
          <td>0.095457</td>
          <td>24.758246</td>
          <td>0.031627</td>
          <td>23.860285</td>
          <td>0.023604</td>
          <td>23.137057</td>
          <td>0.023632</td>
          <td>22.850338</td>
          <td>0.041398</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.446376</td>
          <td>0.366796</td>
          <td>27.760321</td>
          <td>0.427384</td>
          <td>26.397196</td>
          <td>0.225822</td>
          <td>25.671080</td>
          <td>0.225114</td>
          <td>25.249896</td>
          <td>0.339527</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.386611</td>
          <td>0.383060</td>
          <td>25.737833</td>
          <td>0.082152</td>
          <td>25.317619</td>
          <td>0.050731</td>
          <td>24.897895</td>
          <td>0.057525</td>
          <td>24.224060</td>
          <td>0.060199</td>
          <td>23.618836</td>
          <td>0.079924</td>
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
          <td>26.781775</td>
          <td>0.522840</td>
          <td>26.327194</td>
          <td>0.139888</td>
          <td>26.257970</td>
          <td>0.118631</td>
          <td>26.072838</td>
          <td>0.164174</td>
          <td>25.961312</td>
          <td>0.273714</td>
          <td>25.277222</td>
          <td>0.332480</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.941209</td>
          <td>0.231701</td>
          <td>26.772186</td>
          <td>0.181487</td>
          <td>26.624938</td>
          <td>0.256360</td>
          <td>26.074969</td>
          <td>0.295479</td>
          <td>25.653970</td>
          <td>0.438564</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.161837</td>
          <td>0.680640</td>
          <td>27.328559</td>
          <td>0.319751</td>
          <td>26.824633</td>
          <td>0.191286</td>
          <td>26.435418</td>
          <td>0.221059</td>
          <td>26.191935</td>
          <td>0.327021</td>
          <td>25.344642</td>
          <td>0.348023</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>30.573345</td>
          <td>3.329661</td>
          <td>27.601774</td>
          <td>0.402047</td>
          <td>26.464421</td>
          <td>0.143292</td>
          <td>25.948489</td>
          <td>0.149159</td>
          <td>25.356732</td>
          <td>0.166973</td>
          <td>25.296456</td>
          <td>0.340844</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.935198</td>
          <td>0.580033</td>
          <td>26.594666</td>
          <td>0.174142</td>
          <td>25.996560</td>
          <td>0.093378</td>
          <td>25.742008</td>
          <td>0.122118</td>
          <td>25.029098</td>
          <td>0.123359</td>
          <td>25.487008</td>
          <td>0.387963</td>
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
          <td>27.465214</td>
          <td>0.757512</td>
          <td>26.576502</td>
          <td>0.147927</td>
          <td>26.190079</td>
          <td>0.093228</td>
          <td>25.265798</td>
          <td>0.067213</td>
          <td>25.030102</td>
          <td>0.104166</td>
          <td>25.029413</td>
          <td>0.228458</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.748840</td>
          <td>0.456250</td>
          <td>28.375474</td>
          <td>0.616368</td>
          <td>28.229737</td>
          <td>0.500389</td>
          <td>26.882659</td>
          <td>0.269759</td>
          <td>26.590794</td>
          <td>0.383185</td>
          <td>26.174768</td>
          <td>0.558512</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.098514</td>
          <td>0.617168</td>
          <td>25.935667</td>
          <td>0.091008</td>
          <td>24.758892</td>
          <td>0.028562</td>
          <td>23.875694</td>
          <td>0.021546</td>
          <td>23.099624</td>
          <td>0.020703</td>
          <td>22.893423</td>
          <td>0.038666</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.373907</td>
          <td>0.807685</td>
          <td>27.915334</td>
          <td>0.521564</td>
          <td>27.811487</td>
          <td>0.442928</td>
          <td>26.506949</td>
          <td>0.246415</td>
          <td>25.823300</td>
          <td>0.254398</td>
          <td>25.508398</td>
          <td>0.413826</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.163916</td>
          <td>0.289152</td>
          <td>25.782037</td>
          <td>0.074045</td>
          <td>25.421284</td>
          <td>0.047273</td>
          <td>24.791026</td>
          <td>0.044163</td>
          <td>24.395621</td>
          <td>0.059596</td>
          <td>23.682864</td>
          <td>0.071567</td>
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
          <td>27.068196</td>
          <td>0.602840</td>
          <td>26.356949</td>
          <td>0.130995</td>
          <td>26.089578</td>
          <td>0.092333</td>
          <td>25.946621</td>
          <td>0.132570</td>
          <td>25.905084</td>
          <td>0.237273</td>
          <td>25.417690</td>
          <td>0.337473</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.106013</td>
          <td>0.597469</td>
          <td>27.197433</td>
          <td>0.252767</td>
          <td>26.718091</td>
          <td>0.149938</td>
          <td>26.182579</td>
          <td>0.152400</td>
          <td>25.965043</td>
          <td>0.235168</td>
          <td>26.055174</td>
          <td>0.518934</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.860839</td>
          <td>0.510181</td>
          <td>27.273705</td>
          <td>0.276195</td>
          <td>26.874130</td>
          <td>0.176781</td>
          <td>26.475333</td>
          <td>0.201900</td>
          <td>25.864440</td>
          <td>0.223061</td>
          <td>25.951812</td>
          <td>0.494659</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.476644</td>
          <td>0.398057</td>
          <td>27.607478</td>
          <td>0.379573</td>
          <td>26.482316</td>
          <td>0.134753</td>
          <td>25.779337</td>
          <td>0.118986</td>
          <td>25.628055</td>
          <td>0.194843</td>
          <td>26.231196</td>
          <td>0.639043</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.864631</td>
          <td>0.994675</td>
          <td>26.538884</td>
          <td>0.148038</td>
          <td>26.228049</td>
          <td>0.100194</td>
          <td>25.525241</td>
          <td>0.088052</td>
          <td>25.278806</td>
          <td>0.134372</td>
          <td>25.201944</td>
          <td>0.273348</td>
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
