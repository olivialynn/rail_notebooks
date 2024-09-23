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

    <pzflow.flow.Flow at 0x7ff439366860>



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
          <td>27.300387</td>
          <td>0.677857</td>
          <td>26.768010</td>
          <td>0.174183</td>
          <td>25.944762</td>
          <td>0.075092</td>
          <td>25.338200</td>
          <td>0.071652</td>
          <td>25.098027</td>
          <td>0.110520</td>
          <td>24.885579</td>
          <td>0.202596</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.280769</td>
          <td>2.010174</td>
          <td>28.564036</td>
          <td>0.701615</td>
          <td>27.874472</td>
          <td>0.382038</td>
          <td>26.866529</td>
          <td>0.265988</td>
          <td>26.161603</td>
          <td>0.272120</td>
          <td>26.083654</td>
          <td>0.522370</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.329978</td>
          <td>0.329928</td>
          <td>25.966544</td>
          <td>0.087006</td>
          <td>24.796381</td>
          <td>0.027192</td>
          <td>23.869264</td>
          <td>0.019711</td>
          <td>23.163517</td>
          <td>0.020181</td>
          <td>22.839777</td>
          <td>0.033850</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.735123</td>
          <td>0.901294</td>
          <td>27.551653</td>
          <td>0.332017</td>
          <td>27.515169</td>
          <td>0.287340</td>
          <td>26.706848</td>
          <td>0.233266</td>
          <td>26.217684</td>
          <td>0.284793</td>
          <td>25.374631</td>
          <td>0.302838</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.820217</td>
          <td>0.217961</td>
          <td>25.747951</td>
          <td>0.071760</td>
          <td>25.447146</td>
          <td>0.048301</td>
          <td>24.814852</td>
          <td>0.045039</td>
          <td>24.502610</td>
          <td>0.065433</td>
          <td>23.651259</td>
          <td>0.069489</td>
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
          <td>28.306000</td>
          <td>1.258818</td>
          <td>26.329824</td>
          <td>0.119530</td>
          <td>26.139468</td>
          <td>0.089159</td>
          <td>25.854892</td>
          <td>0.112848</td>
          <td>25.616074</td>
          <td>0.172741</td>
          <td>26.303408</td>
          <td>0.611590</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.860692</td>
          <td>0.495615</td>
          <td>27.058348</td>
          <td>0.222320</td>
          <td>26.692066</td>
          <td>0.144278</td>
          <td>26.345391</td>
          <td>0.172213</td>
          <td>26.218235</td>
          <td>0.284920</td>
          <td>25.201837</td>
          <td>0.263276</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.006830</td>
          <td>0.551418</td>
          <td>27.064740</td>
          <td>0.223504</td>
          <td>27.191144</td>
          <td>0.220225</td>
          <td>26.559730</td>
          <td>0.206367</td>
          <td>26.561925</td>
          <td>0.374372</td>
          <td>24.989751</td>
          <td>0.221023</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.666286</td>
          <td>0.428422</td>
          <td>26.901820</td>
          <td>0.195038</td>
          <td>26.683976</td>
          <td>0.143277</td>
          <td>25.685362</td>
          <td>0.097299</td>
          <td>25.880041</td>
          <td>0.215754</td>
          <td>25.817506</td>
          <td>0.428299</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.121146</td>
          <td>0.598347</td>
          <td>26.664350</td>
          <td>0.159467</td>
          <td>26.194034</td>
          <td>0.093540</td>
          <td>25.614089</td>
          <td>0.091396</td>
          <td>25.255154</td>
          <td>0.126704</td>
          <td>24.555924</td>
          <td>0.153167</td>
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

.. parsed-literal::

    




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
          <td>26.825204</td>
          <td>0.532527</td>
          <td>26.590760</td>
          <td>0.172053</td>
          <td>25.891550</td>
          <td>0.084288</td>
          <td>25.271151</td>
          <td>0.080015</td>
          <td>24.947738</td>
          <td>0.113799</td>
          <td>24.998752</td>
          <td>0.260460</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.821043</td>
          <td>0.530986</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.492264</td>
          <td>0.326952</td>
          <td>27.127503</td>
          <td>0.381567</td>
          <td>26.330219</td>
          <td>0.360625</td>
          <td>25.700170</td>
          <td>0.452546</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.432033</td>
          <td>0.402621</td>
          <td>25.929287</td>
          <td>0.099135</td>
          <td>24.795330</td>
          <td>0.032675</td>
          <td>23.852286</td>
          <td>0.023442</td>
          <td>23.152154</td>
          <td>0.023942</td>
          <td>22.882718</td>
          <td>0.042603</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.342503</td>
          <td>0.387411</td>
          <td>27.828728</td>
          <td>0.490705</td>
          <td>27.733304</td>
          <td>0.418671</td>
          <td>26.543541</td>
          <td>0.254810</td>
          <td>25.883104</td>
          <td>0.268036</td>
          <td>25.388744</td>
          <td>0.378559</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.763978</td>
          <td>0.232480</td>
          <td>25.728243</td>
          <td>0.081462</td>
          <td>25.353720</td>
          <td>0.052383</td>
          <td>24.799718</td>
          <td>0.052726</td>
          <td>24.315511</td>
          <td>0.065280</td>
          <td>23.720615</td>
          <td>0.087420</td>
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
          <td>26.380094</td>
          <td>0.386469</td>
          <td>26.491349</td>
          <td>0.161024</td>
          <td>26.056638</td>
          <td>0.099514</td>
          <td>26.062772</td>
          <td>0.162770</td>
          <td>25.715387</td>
          <td>0.223592</td>
          <td>25.152560</td>
          <td>0.300989</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.313303</td>
          <td>0.750028</td>
          <td>27.094263</td>
          <td>0.262780</td>
          <td>27.197022</td>
          <td>0.258578</td>
          <td>26.423142</td>
          <td>0.216969</td>
          <td>26.360510</td>
          <td>0.370566</td>
          <td>24.929595</td>
          <td>0.247081</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.823327</td>
          <td>1.038531</td>
          <td>27.067492</td>
          <td>0.258959</td>
          <td>27.142317</td>
          <td>0.249226</td>
          <td>25.997129</td>
          <td>0.152616</td>
          <td>26.421871</td>
          <td>0.391609</td>
          <td>25.684636</td>
          <td>0.452262</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.640520</td>
          <td>0.414175</td>
          <td>26.560637</td>
          <td>0.155628</td>
          <td>25.768607</td>
          <td>0.127726</td>
          <td>25.453150</td>
          <td>0.181221</td>
          <td>25.356917</td>
          <td>0.357455</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.134759</td>
          <td>0.316342</td>
          <td>26.430031</td>
          <td>0.151331</td>
          <td>26.041054</td>
          <td>0.097096</td>
          <td>25.721101</td>
          <td>0.119920</td>
          <td>25.150083</td>
          <td>0.136973</td>
          <td>24.827358</td>
          <td>0.228360</td>
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
          <td>26.768241</td>
          <td>0.462715</td>
          <td>26.508830</td>
          <td>0.139565</td>
          <td>26.092731</td>
          <td>0.085577</td>
          <td>25.320858</td>
          <td>0.070571</td>
          <td>25.017681</td>
          <td>0.103041</td>
          <td>24.907040</td>
          <td>0.206301</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.908303</td>
          <td>1.002738</td>
          <td>28.186715</td>
          <td>0.538632</td>
          <td>27.263257</td>
          <td>0.234020</td>
          <td>27.790451</td>
          <td>0.544020</td>
          <td>26.246282</td>
          <td>0.291705</td>
          <td>25.697579</td>
          <td>0.391001</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.568669</td>
          <td>0.418426</td>
          <td>25.942693</td>
          <td>0.091571</td>
          <td>24.824020</td>
          <td>0.030240</td>
          <td>23.887371</td>
          <td>0.021762</td>
          <td>23.139649</td>
          <td>0.021421</td>
          <td>22.790903</td>
          <td>0.035316</td>
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
          <td>28.086937</td>
          <td>0.543134</td>
          <td>26.196683</td>
          <td>0.190263</td>
          <td>25.545289</td>
          <td>0.201986</td>
          <td>25.380468</td>
          <td>0.374915</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.192757</td>
          <td>0.295948</td>
          <td>25.733459</td>
          <td>0.070935</td>
          <td>25.461229</td>
          <td>0.048979</td>
          <td>24.808489</td>
          <td>0.044853</td>
          <td>24.295984</td>
          <td>0.054553</td>
          <td>23.592665</td>
          <td>0.066074</td>
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
          <td>26.387653</td>
          <td>0.362971</td>
          <td>26.349178</td>
          <td>0.130118</td>
          <td>26.255012</td>
          <td>0.106737</td>
          <td>26.049898</td>
          <td>0.144919</td>
          <td>25.650386</td>
          <td>0.191825</td>
          <td>26.741973</td>
          <td>0.872626</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.175948</td>
          <td>1.179916</td>
          <td>27.224618</td>
          <td>0.258460</td>
          <td>26.826336</td>
          <td>0.164489</td>
          <td>26.372784</td>
          <td>0.179233</td>
          <td>26.049410</td>
          <td>0.252100</td>
          <td>27.667685</td>
          <td>1.421999</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.048356</td>
          <td>1.113761</td>
          <td>27.116972</td>
          <td>0.242956</td>
          <td>27.053548</td>
          <td>0.205657</td>
          <td>26.568372</td>
          <td>0.218237</td>
          <td>26.040398</td>
          <td>0.257925</td>
          <td>26.715494</td>
          <td>0.839075</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.891006</td>
          <td>0.542590</td>
          <td>27.261397</td>
          <td>0.288512</td>
          <td>26.604962</td>
          <td>0.149762</td>
          <td>25.954489</td>
          <td>0.138475</td>
          <td>25.704736</td>
          <td>0.207796</td>
          <td>25.032394</td>
          <td>0.255962</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.136167</td>
          <td>0.289663</td>
          <td>26.570632</td>
          <td>0.152124</td>
          <td>26.097415</td>
          <td>0.089338</td>
          <td>25.512799</td>
          <td>0.087093</td>
          <td>25.092366</td>
          <td>0.114303</td>
          <td>24.809239</td>
          <td>0.197492</td>
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
