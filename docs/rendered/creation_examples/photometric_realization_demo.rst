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

    <pzflow.flow.Flow at 0x7f2c306d1630>



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
          <td>27.036847</td>
          <td>0.563459</td>
          <td>26.578429</td>
          <td>0.148155</td>
          <td>25.958253</td>
          <td>0.075992</td>
          <td>25.265267</td>
          <td>0.067172</td>
          <td>25.064920</td>
          <td>0.107372</td>
          <td>24.518862</td>
          <td>0.148373</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.697424</td>
          <td>0.880205</td>
          <td>28.023897</td>
          <td>0.477529</td>
          <td>27.422330</td>
          <td>0.266469</td>
          <td>27.581121</td>
          <td>0.465900</td>
          <td>26.817218</td>
          <td>0.455176</td>
          <td>25.420598</td>
          <td>0.314197</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.439789</td>
          <td>0.359736</td>
          <td>25.922720</td>
          <td>0.083716</td>
          <td>24.791314</td>
          <td>0.027072</td>
          <td>23.915287</td>
          <td>0.020496</td>
          <td>23.148967</td>
          <td>0.019934</td>
          <td>22.870543</td>
          <td>0.034781</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.786365</td>
          <td>2.450331</td>
          <td>28.813653</td>
          <td>0.827550</td>
          <td>27.426381</td>
          <td>0.267351</td>
          <td>26.941160</td>
          <td>0.282631</td>
          <td>26.708809</td>
          <td>0.419274</td>
          <td>25.733512</td>
          <td>0.401641</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.254759</td>
          <td>0.310754</td>
          <td>25.762934</td>
          <td>0.072716</td>
          <td>25.440553</td>
          <td>0.048019</td>
          <td>24.828846</td>
          <td>0.045602</td>
          <td>24.377022</td>
          <td>0.058537</td>
          <td>23.718980</td>
          <td>0.073780</td>
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
          <td>26.675100</td>
          <td>0.431300</td>
          <td>26.272961</td>
          <td>0.113764</td>
          <td>26.155107</td>
          <td>0.090394</td>
          <td>26.001473</td>
          <td>0.128180</td>
          <td>25.979198</td>
          <td>0.234281</td>
          <td>25.803936</td>
          <td>0.423897</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.269638</td>
          <td>0.314468</td>
          <td>27.032132</td>
          <td>0.217522</td>
          <td>26.721704</td>
          <td>0.148001</td>
          <td>25.945868</td>
          <td>0.122145</td>
          <td>26.040226</td>
          <td>0.246383</td>
          <td>25.823834</td>
          <td>0.430365</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.127365</td>
          <td>0.235412</td>
          <td>26.890849</td>
          <td>0.171028</td>
          <td>26.779853</td>
          <td>0.247752</td>
          <td>26.255349</td>
          <td>0.293592</td>
          <td>25.560639</td>
          <td>0.351094</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.313116</td>
          <td>2.037502</td>
          <td>27.181316</td>
          <td>0.246123</td>
          <td>26.430694</td>
          <td>0.115056</td>
          <td>25.808687</td>
          <td>0.108389</td>
          <td>25.641997</td>
          <td>0.176586</td>
          <td>25.360763</td>
          <td>0.299481</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.965311</td>
          <td>0.535091</td>
          <td>26.459712</td>
          <td>0.133761</td>
          <td>26.124163</td>
          <td>0.087966</td>
          <td>25.438792</td>
          <td>0.078315</td>
          <td>25.000083</td>
          <td>0.101452</td>
          <td>24.918214</td>
          <td>0.208213</td>
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
          <td>26.736322</td>
          <td>0.498970</td>
          <td>27.111451</td>
          <td>0.265563</td>
          <td>25.993786</td>
          <td>0.092220</td>
          <td>25.481383</td>
          <td>0.096271</td>
          <td>25.206776</td>
          <td>0.142423</td>
          <td>25.067053</td>
          <td>0.275376</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.807271</td>
          <td>0.458677</td>
          <td>27.329531</td>
          <td>0.286961</td>
          <td>27.392498</td>
          <td>0.466996</td>
          <td>27.389116</td>
          <td>0.776360</td>
          <td>25.544756</td>
          <td>0.402060</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.083196</td>
          <td>0.648482</td>
          <td>25.934613</td>
          <td>0.099598</td>
          <td>24.804038</td>
          <td>0.032926</td>
          <td>23.836880</td>
          <td>0.023134</td>
          <td>23.125099</td>
          <td>0.023390</td>
          <td>22.857399</td>
          <td>0.041658</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.274426</td>
          <td>0.758253</td>
          <td>27.795188</td>
          <td>0.478637</td>
          <td>27.225049</td>
          <td>0.280453</td>
          <td>26.238738</td>
          <td>0.197817</td>
          <td>26.114109</td>
          <td>0.322876</td>
          <td>25.374451</td>
          <td>0.374375</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.560794</td>
          <td>0.437726</td>
          <td>25.699502</td>
          <td>0.079426</td>
          <td>25.508735</td>
          <td>0.060104</td>
          <td>24.860743</td>
          <td>0.055660</td>
          <td>24.253396</td>
          <td>0.061785</td>
          <td>23.742409</td>
          <td>0.089112</td>
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
          <td>26.091316</td>
          <td>0.307895</td>
          <td>26.690371</td>
          <td>0.190633</td>
          <td>26.222038</td>
          <td>0.114980</td>
          <td>25.861948</td>
          <td>0.137000</td>
          <td>26.374104</td>
          <td>0.380121</td>
          <td>25.660151</td>
          <td>0.447214</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.059654</td>
          <td>0.255444</td>
          <td>27.075474</td>
          <td>0.233959</td>
          <td>26.393318</td>
          <td>0.211635</td>
          <td>25.955099</td>
          <td>0.268124</td>
          <td>25.187523</td>
          <td>0.304710</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.820335</td>
          <td>0.534937</td>
          <td>27.820315</td>
          <td>0.467685</td>
          <td>26.802695</td>
          <td>0.187779</td>
          <td>26.083814</td>
          <td>0.164357</td>
          <td>26.141074</td>
          <td>0.314032</td>
          <td>25.362705</td>
          <td>0.353002</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.346460</td>
          <td>0.329321</td>
          <td>26.473674</td>
          <td>0.144438</td>
          <td>26.237808</td>
          <td>0.190803</td>
          <td>25.280831</td>
          <td>0.156497</td>
          <td>25.217580</td>
          <td>0.320167</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.800854</td>
          <td>0.526491</td>
          <td>26.534263</td>
          <td>0.165425</td>
          <td>25.979602</td>
          <td>0.091998</td>
          <td>25.545475</td>
          <td>0.102892</td>
          <td>25.296018</td>
          <td>0.155278</td>
          <td>24.482630</td>
          <td>0.170926</td>
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
          <td>28.216056</td>
          <td>1.197985</td>
          <td>26.583872</td>
          <td>0.148866</td>
          <td>26.109923</td>
          <td>0.086882</td>
          <td>25.196278</td>
          <td>0.063196</td>
          <td>25.115099</td>
          <td>0.112193</td>
          <td>24.681843</td>
          <td>0.170578</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.289145</td>
          <td>0.672999</td>
          <td>27.668116</td>
          <td>0.364168</td>
          <td>27.739740</td>
          <td>0.344101</td>
          <td>27.321341</td>
          <td>0.382511</td>
          <td>27.476245</td>
          <td>0.728621</td>
          <td>25.473512</td>
          <td>0.328019</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.598536</td>
          <td>0.428052</td>
          <td>25.972189</td>
          <td>0.093972</td>
          <td>24.739989</td>
          <td>0.028094</td>
          <td>23.874276</td>
          <td>0.021520</td>
          <td>23.155683</td>
          <td>0.021716</td>
          <td>22.812894</td>
          <td>0.036009</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.232331</td>
          <td>0.602683</td>
          <td>26.759150</td>
          <td>0.302514</td>
          <td>26.060682</td>
          <td>0.308389</td>
          <td>25.362606</td>
          <td>0.369734</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.918124</td>
          <td>0.236595</td>
          <td>25.710373</td>
          <td>0.069503</td>
          <td>25.360637</td>
          <td>0.044795</td>
          <td>24.808970</td>
          <td>0.044872</td>
          <td>24.345745</td>
          <td>0.057016</td>
          <td>23.679063</td>
          <td>0.071327</td>
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
          <td>26.387571</td>
          <td>0.362948</td>
          <td>26.316932</td>
          <td>0.126538</td>
          <td>26.183612</td>
          <td>0.100273</td>
          <td>25.949759</td>
          <td>0.132930</td>
          <td>25.426380</td>
          <td>0.158599</td>
          <td>24.944548</td>
          <td>0.229939</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.517205</td>
          <td>0.790595</td>
          <td>26.931791</td>
          <td>0.202767</td>
          <td>27.044768</td>
          <td>0.197919</td>
          <td>26.490805</td>
          <td>0.198010</td>
          <td>26.239888</td>
          <td>0.294364</td>
          <td>25.174552</td>
          <td>0.261598</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.364714</td>
          <td>0.727114</td>
          <td>26.810686</td>
          <td>0.188177</td>
          <td>26.861080</td>
          <td>0.174834</td>
          <td>26.722322</td>
          <td>0.247907</td>
          <td>26.287750</td>
          <td>0.315073</td>
          <td>25.368536</td>
          <td>0.315619</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.485841</td>
          <td>0.400884</td>
          <td>27.111914</td>
          <td>0.255471</td>
          <td>26.659622</td>
          <td>0.156944</td>
          <td>25.891343</td>
          <td>0.131125</td>
          <td>25.778740</td>
          <td>0.221035</td>
          <td>27.324753</td>
          <td>1.260783</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.650669</td>
          <td>0.871489</td>
          <td>26.545436</td>
          <td>0.148873</td>
          <td>26.050902</td>
          <td>0.085755</td>
          <td>25.543799</td>
          <td>0.089502</td>
          <td>25.289913</td>
          <td>0.135667</td>
          <td>24.937457</td>
          <td>0.219859</td>
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
