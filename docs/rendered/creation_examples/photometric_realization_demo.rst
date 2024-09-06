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

    <pzflow.flow.Flow at 0x7f9f46b44340>



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
          <td>27.111392</td>
          <td>0.594229</td>
          <td>26.650655</td>
          <td>0.157611</td>
          <td>26.182562</td>
          <td>0.092602</td>
          <td>25.299361</td>
          <td>0.069231</td>
          <td>25.069312</td>
          <td>0.107784</td>
          <td>24.620038</td>
          <td>0.161803</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.876462</td>
          <td>1.680226</td>
          <td>27.707236</td>
          <td>0.375177</td>
          <td>27.340576</td>
          <td>0.249210</td>
          <td>26.860571</td>
          <td>0.264698</td>
          <td>29.288863</td>
          <td>1.958290</td>
          <td>27.241510</td>
          <td>1.115768</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.714195</td>
          <td>0.444254</td>
          <td>25.798881</td>
          <td>0.075060</td>
          <td>24.800439</td>
          <td>0.027288</td>
          <td>23.866862</td>
          <td>0.019671</td>
          <td>23.185104</td>
          <td>0.020554</td>
          <td>22.825371</td>
          <td>0.033422</td>
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
          <td>27.678088</td>
          <td>0.327430</td>
          <td>26.748304</td>
          <td>0.241396</td>
          <td>26.319431</td>
          <td>0.309107</td>
          <td>25.134192</td>
          <td>0.249076</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.012334</td>
          <td>0.553611</td>
          <td>25.753894</td>
          <td>0.072138</td>
          <td>25.482145</td>
          <td>0.049826</td>
          <td>24.829798</td>
          <td>0.045640</td>
          <td>24.382168</td>
          <td>0.058805</td>
          <td>23.678341</td>
          <td>0.071175</td>
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
          <td>26.336579</td>
          <td>0.331658</td>
          <td>26.325892</td>
          <td>0.119123</td>
          <td>26.239594</td>
          <td>0.097356</td>
          <td>25.929047</td>
          <td>0.120373</td>
          <td>25.848870</td>
          <td>0.210209</td>
          <td>25.550281</td>
          <td>0.348244</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.819988</td>
          <td>0.480892</td>
          <td>26.671601</td>
          <td>0.160457</td>
          <td>26.877088</td>
          <td>0.169037</td>
          <td>26.632065</td>
          <td>0.219220</td>
          <td>25.978362</td>
          <td>0.234119</td>
          <td>26.533809</td>
          <td>0.716829</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.526811</td>
          <td>0.384949</td>
          <td>27.317948</td>
          <td>0.275218</td>
          <td>26.832431</td>
          <td>0.162723</td>
          <td>27.231083</td>
          <td>0.356183</td>
          <td>25.828406</td>
          <td>0.206640</td>
          <td>25.047460</td>
          <td>0.231872</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.328732</td>
          <td>0.329602</td>
          <td>27.508807</td>
          <td>0.320903</td>
          <td>26.518961</td>
          <td>0.124232</td>
          <td>25.803710</td>
          <td>0.107919</td>
          <td>25.723379</td>
          <td>0.189176</td>
          <td>25.371333</td>
          <td>0.302036</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.688378</td>
          <td>0.435664</td>
          <td>26.699670</td>
          <td>0.164346</td>
          <td>26.035024</td>
          <td>0.081322</td>
          <td>25.688738</td>
          <td>0.097587</td>
          <td>25.187281</td>
          <td>0.119454</td>
          <td>24.857732</td>
          <td>0.197913</td>
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
          <td>27.909075</td>
          <td>1.085096</td>
          <td>26.495116</td>
          <td>0.158591</td>
          <td>25.922977</td>
          <td>0.086653</td>
          <td>25.376870</td>
          <td>0.087825</td>
          <td>25.078481</td>
          <td>0.127486</td>
          <td>25.224907</td>
          <td>0.312747</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.286733</td>
          <td>0.735178</td>
          <td>28.351080</td>
          <td>0.678000</td>
          <td>27.231691</td>
          <td>0.265032</td>
          <td>27.173177</td>
          <td>0.395294</td>
          <td>26.896463</td>
          <td>0.552541</td>
          <td>26.692508</td>
          <td>0.899306</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.476716</td>
          <td>0.416638</td>
          <td>25.838075</td>
          <td>0.091522</td>
          <td>24.838586</td>
          <td>0.033944</td>
          <td>23.888912</td>
          <td>0.024195</td>
          <td>23.147316</td>
          <td>0.023842</td>
          <td>22.840205</td>
          <td>0.041029</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.492519</td>
          <td>0.780809</td>
          <td>27.949267</td>
          <td>0.492509</td>
          <td>26.771180</td>
          <td>0.306477</td>
          <td>25.597822</td>
          <td>0.211788</td>
          <td>25.029094</td>
          <td>0.284551</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.605266</td>
          <td>0.452661</td>
          <td>25.765336</td>
          <td>0.084164</td>
          <td>25.459907</td>
          <td>0.057557</td>
          <td>24.958730</td>
          <td>0.060713</td>
          <td>24.478189</td>
          <td>0.075383</td>
          <td>23.534196</td>
          <td>0.074169</td>
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
          <td>26.585204</td>
          <td>0.451945</td>
          <td>26.379119</td>
          <td>0.146274</td>
          <td>26.123613</td>
          <td>0.105520</td>
          <td>26.212523</td>
          <td>0.184850</td>
          <td>26.132301</td>
          <td>0.314173</td>
          <td>25.394404</td>
          <td>0.364613</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.416043</td>
          <td>0.392886</td>
          <td>27.388310</td>
          <td>0.332948</td>
          <td>26.781087</td>
          <td>0.182859</td>
          <td>26.151356</td>
          <td>0.172588</td>
          <td>25.864409</td>
          <td>0.248938</td>
          <td>25.501375</td>
          <td>0.390224</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.661288</td>
          <td>0.414678</td>
          <td>27.183465</td>
          <td>0.257784</td>
          <td>26.577930</td>
          <td>0.248718</td>
          <td>25.815479</td>
          <td>0.241043</td>
          <td>25.404761</td>
          <td>0.364831</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.518056</td>
          <td>0.869912</td>
          <td>28.235868</td>
          <td>0.640200</td>
          <td>26.374789</td>
          <td>0.132633</td>
          <td>25.890141</td>
          <td>0.141861</td>
          <td>25.390797</td>
          <td>0.171884</td>
          <td>24.933051</td>
          <td>0.254348</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.168469</td>
          <td>1.261941</td>
          <td>26.526360</td>
          <td>0.164315</td>
          <td>25.992827</td>
          <td>0.093073</td>
          <td>25.734384</td>
          <td>0.121312</td>
          <td>25.009839</td>
          <td>0.121314</td>
          <td>24.828984</td>
          <td>0.228668</td>
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
          <td>28.481067</td>
          <td>1.381987</td>
          <td>26.340841</td>
          <td>0.120693</td>
          <td>26.048069</td>
          <td>0.082274</td>
          <td>25.400313</td>
          <td>0.075709</td>
          <td>24.855983</td>
          <td>0.089412</td>
          <td>25.036283</td>
          <td>0.229763</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.994032</td>
          <td>1.055207</td>
          <td>28.851890</td>
          <td>0.848612</td>
          <td>27.449543</td>
          <td>0.272682</td>
          <td>27.963765</td>
          <td>0.615639</td>
          <td>26.412731</td>
          <td>0.333246</td>
          <td>25.384730</td>
          <td>0.305577</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.902433</td>
          <td>0.536536</td>
          <td>25.934426</td>
          <td>0.090909</td>
          <td>24.744809</td>
          <td>0.028213</td>
          <td>23.849393</td>
          <td>0.021068</td>
          <td>23.141741</td>
          <td>0.021459</td>
          <td>22.869842</td>
          <td>0.037868</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.690477</td>
          <td>0.884720</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.328133</td>
          <td>0.470530</td>
          <td>25.956828</td>
          <td>0.283644</td>
          <td>25.019082</td>
          <td>0.281303</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.135596</td>
          <td>0.282612</td>
          <td>25.638624</td>
          <td>0.065231</td>
          <td>25.421042</td>
          <td>0.047263</td>
          <td>24.839638</td>
          <td>0.046110</td>
          <td>24.460628</td>
          <td>0.063133</td>
          <td>23.637841</td>
          <td>0.068771</td>
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
          <td>26.617415</td>
          <td>0.433228</td>
          <td>26.387165</td>
          <td>0.134459</td>
          <td>26.124013</td>
          <td>0.095168</td>
          <td>26.076699</td>
          <td>0.148296</td>
          <td>26.175774</td>
          <td>0.295937</td>
          <td>25.092501</td>
          <td>0.259740</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.917438</td>
          <td>0.200340</td>
          <td>26.832788</td>
          <td>0.165397</td>
          <td>26.561635</td>
          <td>0.210126</td>
          <td>26.254272</td>
          <td>0.297794</td>
          <td>25.586545</td>
          <td>0.363811</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.825221</td>
          <td>0.426497</td>
          <td>26.984216</td>
          <td>0.194022</td>
          <td>26.109745</td>
          <td>0.147994</td>
          <td>26.596055</td>
          <td>0.401291</td>
          <td>25.242121</td>
          <td>0.285117</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.641427</td>
          <td>0.901154</td>
          <td>27.542391</td>
          <td>0.360791</td>
          <td>26.642579</td>
          <td>0.154671</td>
          <td>25.978145</td>
          <td>0.141327</td>
          <td>25.523768</td>
          <td>0.178412</td>
          <td>25.192489</td>
          <td>0.291560</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.532583</td>
          <td>0.807951</td>
          <td>26.595195</td>
          <td>0.155357</td>
          <td>26.147514</td>
          <td>0.093360</td>
          <td>25.475678</td>
          <td>0.084292</td>
          <td>25.169611</td>
          <td>0.122246</td>
          <td>24.933663</td>
          <td>0.219166</td>
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
