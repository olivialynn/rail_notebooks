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

    <pzflow.flow.Flow at 0x7fe23522a680>



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
          <td>26.885003</td>
          <td>0.192296</td>
          <td>25.912079</td>
          <td>0.072953</td>
          <td>25.239156</td>
          <td>0.065636</td>
          <td>24.998708</td>
          <td>0.101330</td>
          <td>24.674068</td>
          <td>0.169431</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.515300</td>
          <td>0.782888</td>
          <td>28.034723</td>
          <td>0.481391</td>
          <td>27.722885</td>
          <td>0.339263</td>
          <td>26.684040</td>
          <td>0.228899</td>
          <td>26.478280</td>
          <td>0.350650</td>
          <td>26.341787</td>
          <td>0.628280</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.790879</td>
          <td>0.470577</td>
          <td>26.042556</td>
          <td>0.093013</td>
          <td>24.779158</td>
          <td>0.026786</td>
          <td>23.880145</td>
          <td>0.019893</td>
          <td>23.138398</td>
          <td>0.019756</td>
          <td>22.762736</td>
          <td>0.031629</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.354309</td>
          <td>0.606855</td>
          <td>27.409937</td>
          <td>0.263786</td>
          <td>26.501472</td>
          <td>0.196517</td>
          <td>25.971898</td>
          <td>0.232870</td>
          <td>24.909351</td>
          <td>0.206674</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>28.382940</td>
          <td>1.312191</td>
          <td>25.654437</td>
          <td>0.066068</td>
          <td>25.399480</td>
          <td>0.046300</td>
          <td>24.796049</td>
          <td>0.044293</td>
          <td>24.356397</td>
          <td>0.057475</td>
          <td>23.741024</td>
          <td>0.075231</td>
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
          <td>26.523622</td>
          <td>0.384000</td>
          <td>26.497131</td>
          <td>0.138149</td>
          <td>26.060966</td>
          <td>0.083203</td>
          <td>25.957906</td>
          <td>0.123428</td>
          <td>25.842585</td>
          <td>0.209107</td>
          <td>25.570368</td>
          <td>0.353789</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.634441</td>
          <td>0.418156</td>
          <td>26.998242</td>
          <td>0.211458</td>
          <td>27.042278</td>
          <td>0.194415</td>
          <td>26.516911</td>
          <td>0.199084</td>
          <td>25.747231</td>
          <td>0.193019</td>
          <td>25.380698</td>
          <td>0.304316</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.678151</td>
          <td>0.432299</td>
          <td>27.400038</td>
          <td>0.294121</td>
          <td>27.189708</td>
          <td>0.219962</td>
          <td>26.455514</td>
          <td>0.189052</td>
          <td>26.229515</td>
          <td>0.287531</td>
          <td>25.879769</td>
          <td>0.448980</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.380825</td>
          <td>0.289597</td>
          <td>27.052511</td>
          <td>0.196096</td>
          <td>25.662812</td>
          <td>0.095393</td>
          <td>25.843697</td>
          <td>0.209302</td>
          <td>25.360784</td>
          <td>0.299486</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.242819</td>
          <td>0.651523</td>
          <td>26.416859</td>
          <td>0.128897</td>
          <td>26.098686</td>
          <td>0.086015</td>
          <td>25.530692</td>
          <td>0.084928</td>
          <td>25.390404</td>
          <td>0.142409</td>
          <td>24.967296</td>
          <td>0.216927</td>
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
          <td>28.426581</td>
          <td>1.439275</td>
          <td>27.012522</td>
          <td>0.244883</td>
          <td>26.133244</td>
          <td>0.104210</td>
          <td>25.347711</td>
          <td>0.085600</td>
          <td>24.981083</td>
          <td>0.117151</td>
          <td>25.923206</td>
          <td>0.533687</td>
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
          <td>27.337413</td>
          <td>0.288795</td>
          <td>27.457735</td>
          <td>0.490238</td>
          <td>26.689998</td>
          <td>0.474866</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.245912</td>
          <td>0.348387</td>
          <td>25.974323</td>
          <td>0.103116</td>
          <td>24.779334</td>
          <td>0.032219</td>
          <td>23.909941</td>
          <td>0.024639</td>
          <td>23.080784</td>
          <td>0.022517</td>
          <td>22.873419</td>
          <td>0.042253</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.889328</td>
          <td>0.513117</td>
          <td>28.153381</td>
          <td>0.571423</td>
          <td>26.820790</td>
          <td>0.318878</td>
          <td>26.018426</td>
          <td>0.299078</td>
          <td>25.859389</td>
          <td>0.539346</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.663040</td>
          <td>0.472675</td>
          <td>25.711145</td>
          <td>0.080245</td>
          <td>25.485841</td>
          <td>0.058897</td>
          <td>24.854600</td>
          <td>0.055357</td>
          <td>24.321509</td>
          <td>0.065628</td>
          <td>23.698296</td>
          <td>0.085720</td>
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
          <td>26.382113</td>
          <td>0.387073</td>
          <td>26.393649</td>
          <td>0.148110</td>
          <td>26.108446</td>
          <td>0.104130</td>
          <td>26.220045</td>
          <td>0.186029</td>
          <td>25.408471</td>
          <td>0.172737</td>
          <td>25.531917</td>
          <td>0.405616</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.477366</td>
          <td>0.411840</td>
          <td>27.742739</td>
          <td>0.438237</td>
          <td>26.647181</td>
          <td>0.163192</td>
          <td>26.279499</td>
          <td>0.192357</td>
          <td>25.738850</td>
          <td>0.224399</td>
          <td>24.844344</td>
          <td>0.230286</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.123555</td>
          <td>0.662993</td>
          <td>26.919810</td>
          <td>0.229306</td>
          <td>26.520491</td>
          <td>0.147652</td>
          <td>26.813653</td>
          <td>0.301264</td>
          <td>26.174558</td>
          <td>0.322532</td>
          <td>24.784550</td>
          <td>0.220970</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.630884</td>
          <td>1.613984</td>
          <td>27.327024</td>
          <td>0.324275</td>
          <td>26.474144</td>
          <td>0.144496</td>
          <td>25.971390</td>
          <td>0.152118</td>
          <td>26.258936</td>
          <td>0.350631</td>
          <td>25.561521</td>
          <td>0.418801</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.103635</td>
          <td>0.652863</td>
          <td>26.455300</td>
          <td>0.154641</td>
          <td>26.097765</td>
          <td>0.102041</td>
          <td>25.695610</td>
          <td>0.117292</td>
          <td>25.446074</td>
          <td>0.176464</td>
          <td>24.964046</td>
          <td>0.255607</td>
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
          <td>26.568875</td>
          <td>0.397688</td>
          <td>26.609528</td>
          <td>0.152177</td>
          <td>26.023691</td>
          <td>0.080523</td>
          <td>25.484005</td>
          <td>0.081515</td>
          <td>25.019697</td>
          <td>0.103223</td>
          <td>24.559771</td>
          <td>0.153693</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.729391</td>
          <td>3.325961</td>
          <td>30.036236</td>
          <td>1.639450</td>
          <td>27.748964</td>
          <td>0.346613</td>
          <td>27.369155</td>
          <td>0.396922</td>
          <td>26.596321</td>
          <td>0.384831</td>
          <td>25.443298</td>
          <td>0.320227</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.534252</td>
          <td>0.407554</td>
          <td>26.029521</td>
          <td>0.098812</td>
          <td>24.829278</td>
          <td>0.030380</td>
          <td>23.857566</td>
          <td>0.021215</td>
          <td>23.177516</td>
          <td>0.022126</td>
          <td>22.783102</td>
          <td>0.035074</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.388822</td>
          <td>0.349602</td>
          <td>27.826055</td>
          <td>0.447827</td>
          <td>26.747429</td>
          <td>0.299678</td>
          <td>26.996976</td>
          <td>0.624637</td>
          <td>26.256131</td>
          <td>0.710238</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.301267</td>
          <td>0.322779</td>
          <td>25.778566</td>
          <td>0.073818</td>
          <td>25.431776</td>
          <td>0.047715</td>
          <td>24.799088</td>
          <td>0.044480</td>
          <td>24.428826</td>
          <td>0.061378</td>
          <td>23.714899</td>
          <td>0.073623</td>
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
          <td>26.815891</td>
          <td>0.502516</td>
          <td>26.320263</td>
          <td>0.126903</td>
          <td>26.130549</td>
          <td>0.095715</td>
          <td>26.149736</td>
          <td>0.157876</td>
          <td>26.099065</td>
          <td>0.278138</td>
          <td>25.052003</td>
          <td>0.251259</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.306008</td>
          <td>0.686564</td>
          <td>26.894375</td>
          <td>0.196496</td>
          <td>26.691692</td>
          <td>0.146576</td>
          <td>26.244217</td>
          <td>0.160657</td>
          <td>26.169399</td>
          <td>0.278048</td>
          <td>25.897577</td>
          <td>0.461739</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.818241</td>
          <td>0.494424</td>
          <td>28.509209</td>
          <td>0.698641</td>
          <td>27.028320</td>
          <td>0.201352</td>
          <td>26.582676</td>
          <td>0.220852</td>
          <td>25.695113</td>
          <td>0.193586</td>
          <td>25.275611</td>
          <td>0.292936</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.124851</td>
          <td>0.301847</td>
          <td>28.003846</td>
          <td>0.512149</td>
          <td>26.737089</td>
          <td>0.167674</td>
          <td>25.837074</td>
          <td>0.125104</td>
          <td>25.885007</td>
          <td>0.241379</td>
          <td>25.969332</td>
          <td>0.530292</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.488529</td>
          <td>0.785066</td>
          <td>26.405695</td>
          <td>0.131996</td>
          <td>26.101919</td>
          <td>0.089693</td>
          <td>25.651527</td>
          <td>0.098382</td>
          <td>25.181098</td>
          <td>0.123471</td>
          <td>25.371263</td>
          <td>0.313346</td>
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
