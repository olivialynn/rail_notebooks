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

    <pzflow.flow.Flow at 0x7efef50c20e0>



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
          <td>28.553069</td>
          <td>1.434215</td>
          <td>26.676452</td>
          <td>0.161123</td>
          <td>26.055057</td>
          <td>0.082771</td>
          <td>25.264623</td>
          <td>0.067133</td>
          <td>24.949088</td>
          <td>0.097018</td>
          <td>24.770465</td>
          <td>0.183872</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.198125</td>
          <td>0.542739</td>
          <td>27.442011</td>
          <td>0.270778</td>
          <td>27.160763</td>
          <td>0.336982</td>
          <td>26.365886</td>
          <td>0.320795</td>
          <td>27.367744</td>
          <td>1.198488</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.096964</td>
          <td>0.273651</td>
          <td>25.838914</td>
          <td>0.077758</td>
          <td>24.767812</td>
          <td>0.026523</td>
          <td>23.890708</td>
          <td>0.020072</td>
          <td>23.133014</td>
          <td>0.019666</td>
          <td>22.852279</td>
          <td>0.034225</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.985365</td>
          <td>3.570502</td>
          <td>29.235121</td>
          <td>1.072345</td>
          <td>27.421132</td>
          <td>0.266208</td>
          <td>26.702362</td>
          <td>0.232401</td>
          <td>26.170230</td>
          <td>0.274036</td>
          <td>25.098304</td>
          <td>0.241823</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.740734</td>
          <td>0.453226</td>
          <td>25.991132</td>
          <td>0.088907</td>
          <td>25.444367</td>
          <td>0.048182</td>
          <td>24.821345</td>
          <td>0.045299</td>
          <td>24.362031</td>
          <td>0.057763</td>
          <td>23.650702</td>
          <td>0.069455</td>
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
          <td>26.353615</td>
          <td>0.336159</td>
          <td>26.183905</td>
          <td>0.105265</td>
          <td>26.299320</td>
          <td>0.102586</td>
          <td>26.406966</td>
          <td>0.181450</td>
          <td>25.901657</td>
          <td>0.219676</td>
          <td>25.566011</td>
          <td>0.352580</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.183398</td>
          <td>0.625135</td>
          <td>27.230154</td>
          <td>0.256193</td>
          <td>26.922581</td>
          <td>0.175702</td>
          <td>26.711602</td>
          <td>0.234186</td>
          <td>26.295373</td>
          <td>0.303201</td>
          <td>25.180664</td>
          <td>0.258756</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.990563</td>
          <td>0.210106</td>
          <td>27.055174</td>
          <td>0.196536</td>
          <td>26.903366</td>
          <td>0.274092</td>
          <td>26.200539</td>
          <td>0.280865</td>
          <td>26.159499</td>
          <td>0.551953</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.183865</td>
          <td>0.246640</td>
          <td>26.770916</td>
          <td>0.154383</td>
          <td>25.897843</td>
          <td>0.117150</td>
          <td>25.322780</td>
          <td>0.134340</td>
          <td>25.997912</td>
          <td>0.490430</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.891482</td>
          <td>0.506987</td>
          <td>26.271372</td>
          <td>0.113607</td>
          <td>26.103370</td>
          <td>0.086371</td>
          <td>25.627425</td>
          <td>0.092474</td>
          <td>25.348171</td>
          <td>0.137317</td>
          <td>24.828912</td>
          <td>0.193171</td>
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
          <td>27.667553</td>
          <td>0.938905</td>
          <td>26.968804</td>
          <td>0.236212</td>
          <td>26.042602</td>
          <td>0.096257</td>
          <td>25.325171</td>
          <td>0.083918</td>
          <td>24.854139</td>
          <td>0.104874</td>
          <td>25.354503</td>
          <td>0.346632</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.150089</td>
          <td>0.670158</td>
          <td>28.724457</td>
          <td>0.867361</td>
          <td>27.535930</td>
          <td>0.338467</td>
          <td>27.482347</td>
          <td>0.499241</td>
          <td>26.332828</td>
          <td>0.361362</td>
          <td>25.529264</td>
          <td>0.397293</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.334385</td>
          <td>0.373346</td>
          <td>26.026469</td>
          <td>0.107918</td>
          <td>24.782017</td>
          <td>0.032295</td>
          <td>23.905133</td>
          <td>0.024536</td>
          <td>23.162165</td>
          <td>0.024150</td>
          <td>22.834483</td>
          <td>0.040821</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.347937</td>
          <td>0.389042</td>
          <td>28.695878</td>
          <td>0.889803</td>
          <td>26.900143</td>
          <td>0.214647</td>
          <td>26.741327</td>
          <td>0.299217</td>
          <td>26.011926</td>
          <td>0.297518</td>
          <td>25.101986</td>
          <td>0.301777</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.810875</td>
          <td>0.241647</td>
          <td>25.841760</td>
          <td>0.090009</td>
          <td>25.483060</td>
          <td>0.058751</td>
          <td>24.788732</td>
          <td>0.052215</td>
          <td>24.387378</td>
          <td>0.069568</td>
          <td>23.650827</td>
          <td>0.082210</td>
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
          <td>25.758987</td>
          <td>0.234988</td>
          <td>26.266950</td>
          <td>0.132807</td>
          <td>26.025719</td>
          <td>0.096854</td>
          <td>26.098339</td>
          <td>0.167781</td>
          <td>26.213326</td>
          <td>0.335089</td>
          <td>25.799830</td>
          <td>0.496380</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.040753</td>
          <td>0.622704</td>
          <td>26.908844</td>
          <td>0.225568</td>
          <td>26.966188</td>
          <td>0.213643</td>
          <td>26.545571</td>
          <td>0.240162</td>
          <td>26.131393</td>
          <td>0.309178</td>
          <td>25.869730</td>
          <td>0.515080</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.954548</td>
          <td>1.121154</td>
          <td>27.459246</td>
          <td>0.354570</td>
          <td>27.089629</td>
          <td>0.238639</td>
          <td>26.612778</td>
          <td>0.255936</td>
          <td>25.971033</td>
          <td>0.273800</td>
          <td>26.361526</td>
          <td>0.732641</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.924414</td>
          <td>0.583292</td>
          <td>27.843593</td>
          <td>0.482722</td>
          <td>26.714923</td>
          <td>0.177494</td>
          <td>25.654960</td>
          <td>0.115723</td>
          <td>25.965595</td>
          <td>0.277324</td>
          <td>26.603887</td>
          <td>0.870234</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.246473</td>
          <td>0.345613</td>
          <td>26.395714</td>
          <td>0.146942</td>
          <td>26.053808</td>
          <td>0.098187</td>
          <td>25.826268</td>
          <td>0.131368</td>
          <td>25.225315</td>
          <td>0.146140</td>
          <td>24.618055</td>
          <td>0.191691</td>
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
          <td>27.694252</td>
          <td>0.878501</td>
          <td>26.672537</td>
          <td>0.160603</td>
          <td>25.995572</td>
          <td>0.078549</td>
          <td>25.258121</td>
          <td>0.066757</td>
          <td>25.023226</td>
          <td>0.103542</td>
          <td>25.149825</td>
          <td>0.252328</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.638989</td>
          <td>2.320021</td>
          <td>28.749888</td>
          <td>0.794500</td>
          <td>32.226612</td>
          <td>3.461000</td>
          <td>27.169594</td>
          <td>0.339648</td>
          <td>26.400718</td>
          <td>0.330087</td>
          <td>25.895140</td>
          <td>0.454590</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.279679</td>
          <td>0.699380</td>
          <td>26.115072</td>
          <td>0.106483</td>
          <td>24.790272</td>
          <td>0.029358</td>
          <td>23.882596</td>
          <td>0.021673</td>
          <td>23.130499</td>
          <td>0.021254</td>
          <td>22.832901</td>
          <td>0.036651</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.002607</td>
          <td>0.628833</td>
          <td>27.809672</td>
          <td>0.482489</td>
          <td>27.419023</td>
          <td>0.326650</td>
          <td>26.811897</td>
          <td>0.315566</td>
          <td>26.298603</td>
          <td>0.372188</td>
          <td>24.739084</td>
          <td>0.223523</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.179310</td>
          <td>0.292762</td>
          <td>25.743799</td>
          <td>0.071586</td>
          <td>25.402862</td>
          <td>0.046506</td>
          <td>24.789626</td>
          <td>0.044108</td>
          <td>24.515941</td>
          <td>0.066305</td>
          <td>23.690124</td>
          <td>0.072028</td>
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
          <td>27.191548</td>
          <td>0.657067</td>
          <td>26.698634</td>
          <td>0.175541</td>
          <td>26.243822</td>
          <td>0.105698</td>
          <td>25.990411</td>
          <td>0.137680</td>
          <td>26.119461</td>
          <td>0.282776</td>
          <td>25.880663</td>
          <td>0.481599</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.807638</td>
          <td>1.636568</td>
          <td>27.024047</td>
          <td>0.219015</td>
          <td>26.667163</td>
          <td>0.143516</td>
          <td>26.883844</td>
          <td>0.274123</td>
          <td>26.354260</td>
          <td>0.322611</td>
          <td>25.428483</td>
          <td>0.321129</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.124224</td>
          <td>1.912088</td>
          <td>27.156263</td>
          <td>0.250936</td>
          <td>26.997209</td>
          <td>0.196156</td>
          <td>26.222857</td>
          <td>0.163045</td>
          <td>25.901760</td>
          <td>0.230080</td>
          <td>26.007762</td>
          <td>0.515460</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.140236</td>
          <td>1.209710</td>
          <td>27.004949</td>
          <td>0.233936</td>
          <td>26.718864</td>
          <td>0.165090</td>
          <td>25.870526</td>
          <td>0.128783</td>
          <td>25.593093</td>
          <td>0.189186</td>
          <td>25.238807</td>
          <td>0.302637</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.757974</td>
          <td>0.931982</td>
          <td>26.468484</td>
          <td>0.139343</td>
          <td>26.094745</td>
          <td>0.089129</td>
          <td>25.421770</td>
          <td>0.080379</td>
          <td>25.136940</td>
          <td>0.118825</td>
          <td>24.974281</td>
          <td>0.226695</td>
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
