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

    <pzflow.flow.Flow at 0x7fb76e1020e0>



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
          <td>26.850363</td>
          <td>0.491845</td>
          <td>26.784325</td>
          <td>0.176611</td>
          <td>26.029924</td>
          <td>0.080957</td>
          <td>25.496825</td>
          <td>0.082431</td>
          <td>24.973151</td>
          <td>0.099087</td>
          <td>24.759475</td>
          <td>0.182170</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.116162</td>
          <td>1.873139</td>
          <td>27.514602</td>
          <td>0.322388</td>
          <td>27.423419</td>
          <td>0.266706</td>
          <td>27.245685</td>
          <td>0.360284</td>
          <td>29.101970</td>
          <td>1.804483</td>
          <td>26.715073</td>
          <td>0.808188</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.003399</td>
          <td>0.550055</td>
          <td>26.048428</td>
          <td>0.093493</td>
          <td>24.776454</td>
          <td>0.026723</td>
          <td>23.915995</td>
          <td>0.020508</td>
          <td>23.132225</td>
          <td>0.019653</td>
          <td>22.840126</td>
          <td>0.033860</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.467641</td>
          <td>1.372267</td>
          <td>27.377953</td>
          <td>0.288927</td>
          <td>27.656322</td>
          <td>0.321809</td>
          <td>26.801042</td>
          <td>0.252104</td>
          <td>25.875251</td>
          <td>0.214893</td>
          <td>25.509398</td>
          <td>0.337186</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.390323</td>
          <td>0.346036</td>
          <td>25.855574</td>
          <td>0.078909</td>
          <td>25.405750</td>
          <td>0.046558</td>
          <td>24.768603</td>
          <td>0.043228</td>
          <td>24.320386</td>
          <td>0.055668</td>
          <td>23.621844</td>
          <td>0.067703</td>
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
          <td>25.994575</td>
          <td>0.251718</td>
          <td>26.254152</td>
          <td>0.111916</td>
          <td>26.272473</td>
          <td>0.100203</td>
          <td>25.971698</td>
          <td>0.124914</td>
          <td>25.580187</td>
          <td>0.167546</td>
          <td>25.330014</td>
          <td>0.292154</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.736202</td>
          <td>0.451683</td>
          <td>27.919426</td>
          <td>0.441517</td>
          <td>26.692845</td>
          <td>0.144375</td>
          <td>26.778691</td>
          <td>0.247515</td>
          <td>26.177886</td>
          <td>0.275747</td>
          <td>25.841263</td>
          <td>0.436097</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.509085</td>
          <td>0.779701</td>
          <td>27.046801</td>
          <td>0.220195</td>
          <td>26.895231</td>
          <td>0.171666</td>
          <td>26.613300</td>
          <td>0.215818</td>
          <td>26.017382</td>
          <td>0.241789</td>
          <td>25.265053</td>
          <td>0.277188</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.442633</td>
          <td>0.304368</td>
          <td>26.739149</td>
          <td>0.150235</td>
          <td>25.586685</td>
          <td>0.089220</td>
          <td>25.500474</td>
          <td>0.156522</td>
          <td>25.637310</td>
          <td>0.372810</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>25.924292</td>
          <td>0.237583</td>
          <td>26.563770</td>
          <td>0.146302</td>
          <td>26.283019</td>
          <td>0.101133</td>
          <td>25.697605</td>
          <td>0.098349</td>
          <td>25.015683</td>
          <td>0.102847</td>
          <td>24.777355</td>
          <td>0.184946</td>
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
          <td>26.827761</td>
          <td>0.533518</td>
          <td>26.688024</td>
          <td>0.186822</td>
          <td>26.028207</td>
          <td>0.095049</td>
          <td>25.293723</td>
          <td>0.081624</td>
          <td>25.105378</td>
          <td>0.130490</td>
          <td>25.206266</td>
          <td>0.308116</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.963241</td>
          <td>1.119735</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.691585</td>
          <td>0.382359</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.991206</td>
          <td>0.591311</td>
          <td>25.946749</td>
          <td>0.542980</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.538552</td>
          <td>0.436696</td>
          <td>25.999468</td>
          <td>0.105405</td>
          <td>24.792704</td>
          <td>0.032600</td>
          <td>23.919093</td>
          <td>0.024835</td>
          <td>23.109696</td>
          <td>0.023082</td>
          <td>22.800566</td>
          <td>0.039614</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.630053</td>
          <td>1.505315</td>
          <td>27.299225</td>
          <td>0.297771</td>
          <td>26.819222</td>
          <td>0.318479</td>
          <td>26.134921</td>
          <td>0.328262</td>
          <td>24.934556</td>
          <td>0.263497</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.661371</td>
          <td>0.472087</td>
          <td>25.842973</td>
          <td>0.090105</td>
          <td>25.396029</td>
          <td>0.054386</td>
          <td>24.790209</td>
          <td>0.052283</td>
          <td>24.449866</td>
          <td>0.073520</td>
          <td>23.626239</td>
          <td>0.080447</td>
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
          <td>27.643584</td>
          <td>0.935584</td>
          <td>26.158912</td>
          <td>0.120951</td>
          <td>26.228785</td>
          <td>0.115658</td>
          <td>25.955897</td>
          <td>0.148539</td>
          <td>25.985143</td>
          <td>0.279064</td>
          <td>25.253857</td>
          <td>0.326371</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.365322</td>
          <td>1.397249</td>
          <td>27.244413</td>
          <td>0.296803</td>
          <td>27.010089</td>
          <td>0.221605</td>
          <td>26.777094</td>
          <td>0.290142</td>
          <td>26.713782</td>
          <td>0.484968</td>
          <td>24.940650</td>
          <td>0.249338</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.207786</td>
          <td>0.290232</td>
          <td>26.932508</td>
          <td>0.209423</td>
          <td>26.299539</td>
          <td>0.197307</td>
          <td>26.128026</td>
          <td>0.310772</td>
          <td>25.224197</td>
          <td>0.316324</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.635051</td>
          <td>0.472352</td>
          <td>27.480300</td>
          <td>0.365920</td>
          <td>26.929378</td>
          <td>0.212610</td>
          <td>25.916410</td>
          <td>0.145104</td>
          <td>25.306546</td>
          <td>0.159976</td>
          <td>25.547185</td>
          <td>0.414236</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.366977</td>
          <td>0.779712</td>
          <td>26.635497</td>
          <td>0.180274</td>
          <td>25.916397</td>
          <td>0.087025</td>
          <td>25.640068</td>
          <td>0.111754</td>
          <td>25.097964</td>
          <td>0.130943</td>
          <td>25.027761</td>
          <td>0.269269</td>
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
          <td>27.618480</td>
          <td>0.837144</td>
          <td>26.820647</td>
          <td>0.182148</td>
          <td>25.975973</td>
          <td>0.077201</td>
          <td>25.419630</td>
          <td>0.077012</td>
          <td>25.007246</td>
          <td>0.102104</td>
          <td>25.147138</td>
          <td>0.251772</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.511061</td>
          <td>0.321719</td>
          <td>27.401813</td>
          <td>0.262271</td>
          <td>26.887703</td>
          <td>0.270869</td>
          <td>26.531368</td>
          <td>0.365861</td>
          <td>25.334156</td>
          <td>0.293397</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.439843</td>
          <td>0.378927</td>
          <td>25.847236</td>
          <td>0.084204</td>
          <td>24.795305</td>
          <td>0.029488</td>
          <td>23.885697</td>
          <td>0.021731</td>
          <td>23.132176</td>
          <td>0.021285</td>
          <td>22.801172</td>
          <td>0.035638</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.232324</td>
          <td>0.735749</td>
          <td>33.381167</td>
          <td>4.951710</td>
          <td>27.821458</td>
          <td>0.446276</td>
          <td>26.164507</td>
          <td>0.185164</td>
          <td>25.679465</td>
          <td>0.225924</td>
          <td>24.977202</td>
          <td>0.271895</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.070568</td>
          <td>0.577685</td>
          <td>25.930552</td>
          <td>0.084399</td>
          <td>25.553806</td>
          <td>0.053176</td>
          <td>24.813673</td>
          <td>0.045060</td>
          <td>24.467278</td>
          <td>0.063506</td>
          <td>23.571687</td>
          <td>0.064857</td>
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
          <td>26.470186</td>
          <td>0.387025</td>
          <td>26.405017</td>
          <td>0.136545</td>
          <td>26.165819</td>
          <td>0.098722</td>
          <td>26.004036</td>
          <td>0.139307</td>
          <td>26.014128</td>
          <td>0.259534</td>
          <td>25.141295</td>
          <td>0.270296</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.564858</td>
          <td>0.815486</td>
          <td>26.693147</td>
          <td>0.165722</td>
          <td>26.629046</td>
          <td>0.138880</td>
          <td>26.291192</td>
          <td>0.167226</td>
          <td>26.555091</td>
          <td>0.377848</td>
          <td>25.283363</td>
          <td>0.285807</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.892285</td>
          <td>1.016294</td>
          <td>27.435248</td>
          <td>0.314577</td>
          <td>26.598162</td>
          <td>0.139606</td>
          <td>26.353158</td>
          <td>0.182142</td>
          <td>25.923073</td>
          <td>0.234178</td>
          <td>25.321226</td>
          <td>0.303888</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.966181</td>
          <td>0.226542</td>
          <td>26.577298</td>
          <td>0.146246</td>
          <td>25.730454</td>
          <td>0.114031</td>
          <td>25.516156</td>
          <td>0.177264</td>
          <td>27.625610</td>
          <td>1.476061</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.876087</td>
          <td>0.512838</td>
          <td>26.602750</td>
          <td>0.156365</td>
          <td>26.052741</td>
          <td>0.085894</td>
          <td>25.533896</td>
          <td>0.088726</td>
          <td>25.185215</td>
          <td>0.123913</td>
          <td>24.789145</td>
          <td>0.194181</td>
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
