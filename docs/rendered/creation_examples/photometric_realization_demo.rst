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

    <pzflow.flow.Flow at 0x7fc2a88bb250>



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
          <td>26.753292</td>
          <td>0.457522</td>
          <td>26.792313</td>
          <td>0.177811</td>
          <td>26.061947</td>
          <td>0.083275</td>
          <td>25.326294</td>
          <td>0.070901</td>
          <td>24.959588</td>
          <td>0.097916</td>
          <td>25.321334</td>
          <td>0.290114</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.890508</td>
          <td>1.691297</td>
          <td>29.288867</td>
          <td>1.106377</td>
          <td>27.477325</td>
          <td>0.278665</td>
          <td>27.371696</td>
          <td>0.397354</td>
          <td>27.295500</td>
          <td>0.643554</td>
          <td>25.826326</td>
          <td>0.431181</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.623626</td>
          <td>0.414716</td>
          <td>26.094116</td>
          <td>0.097314</td>
          <td>24.790537</td>
          <td>0.027054</td>
          <td>23.877412</td>
          <td>0.019847</td>
          <td>23.121416</td>
          <td>0.019475</td>
          <td>22.800877</td>
          <td>0.032709</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.871787</td>
          <td>0.425842</td>
          <td>27.122423</td>
          <td>0.207946</td>
          <td>27.320341</td>
          <td>0.381879</td>
          <td>26.427485</td>
          <td>0.336876</td>
          <td>25.335756</td>
          <td>0.293510</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.333440</td>
          <td>0.330834</td>
          <td>25.761308</td>
          <td>0.072612</td>
          <td>25.433818</td>
          <td>0.047733</td>
          <td>24.828588</td>
          <td>0.045591</td>
          <td>24.433851</td>
          <td>0.061563</td>
          <td>23.872946</td>
          <td>0.084520</td>
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
          <td>26.576473</td>
          <td>0.399988</td>
          <td>26.213110</td>
          <td>0.107983</td>
          <td>26.204949</td>
          <td>0.094441</td>
          <td>26.271370</td>
          <td>0.161686</td>
          <td>26.225270</td>
          <td>0.286546</td>
          <td>25.342863</td>
          <td>0.295197</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.672745</td>
          <td>0.430529</td>
          <td>27.617846</td>
          <td>0.349834</td>
          <td>26.618104</td>
          <td>0.135366</td>
          <td>26.188209</td>
          <td>0.150575</td>
          <td>25.661032</td>
          <td>0.179460</td>
          <td>25.826194</td>
          <td>0.431138</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.307744</td>
          <td>0.681277</td>
          <td>27.866218</td>
          <td>0.424040</td>
          <td>26.683441</td>
          <td>0.143211</td>
          <td>26.387552</td>
          <td>0.178490</td>
          <td>26.170476</td>
          <td>0.274091</td>
          <td>26.127390</td>
          <td>0.539275</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.984773</td>
          <td>0.542697</td>
          <td>27.036061</td>
          <td>0.218235</td>
          <td>26.385623</td>
          <td>0.110623</td>
          <td>25.925530</td>
          <td>0.120005</td>
          <td>25.704899</td>
          <td>0.186247</td>
          <td>25.337969</td>
          <td>0.294035</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.665884</td>
          <td>0.428291</td>
          <td>26.649774</td>
          <td>0.157492</td>
          <td>26.105246</td>
          <td>0.086514</td>
          <td>25.624089</td>
          <td>0.092203</td>
          <td>25.261485</td>
          <td>0.127401</td>
          <td>24.725768</td>
          <td>0.177040</td>
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
          <td>26.692132</td>
          <td>0.482918</td>
          <td>26.791152</td>
          <td>0.203749</td>
          <td>26.263308</td>
          <td>0.116730</td>
          <td>25.217460</td>
          <td>0.076311</td>
          <td>24.986295</td>
          <td>0.117683</td>
          <td>24.882166</td>
          <td>0.236651</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.882091</td>
          <td>0.485024</td>
          <td>28.011069</td>
          <td>0.487311</td>
          <td>27.122841</td>
          <td>0.380188</td>
          <td>26.561873</td>
          <td>0.431206</td>
          <td>26.389504</td>
          <td>0.739177</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.395830</td>
          <td>0.391552</td>
          <td>25.807806</td>
          <td>0.089123</td>
          <td>24.814959</td>
          <td>0.033245</td>
          <td>23.867603</td>
          <td>0.023754</td>
          <td>23.168089</td>
          <td>0.024273</td>
          <td>22.864211</td>
          <td>0.041910</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.123649</td>
          <td>1.265650</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.273252</td>
          <td>0.291604</td>
          <td>26.402056</td>
          <td>0.226735</td>
          <td>25.752223</td>
          <td>0.240755</td>
          <td>24.868752</td>
          <td>0.249667</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.548965</td>
          <td>0.872039</td>
          <td>25.828669</td>
          <td>0.088981</td>
          <td>25.505500</td>
          <td>0.059932</td>
          <td>24.832211</td>
          <td>0.054268</td>
          <td>24.364126</td>
          <td>0.068151</td>
          <td>23.671921</td>
          <td>0.083752</td>
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
          <td>27.023423</td>
          <td>0.621524</td>
          <td>26.149989</td>
          <td>0.120018</td>
          <td>26.138527</td>
          <td>0.106904</td>
          <td>25.781275</td>
          <td>0.127770</td>
          <td>26.112940</td>
          <td>0.309345</td>
          <td>26.079244</td>
          <td>0.607381</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.046428</td>
          <td>0.625182</td>
          <td>26.737419</td>
          <td>0.195465</td>
          <td>26.663375</td>
          <td>0.165462</td>
          <td>26.108567</td>
          <td>0.166416</td>
          <td>25.614506</td>
          <td>0.202269</td>
          <td>26.369117</td>
          <td>0.731430</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.887614</td>
          <td>1.078547</td>
          <td>27.035087</td>
          <td>0.252174</td>
          <td>26.719030</td>
          <td>0.174938</td>
          <td>26.022708</td>
          <td>0.155996</td>
          <td>26.177578</td>
          <td>0.323308</td>
          <td>26.616659</td>
          <td>0.865296</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.749979</td>
          <td>0.450067</td>
          <td>26.371256</td>
          <td>0.132228</td>
          <td>25.604273</td>
          <td>0.110724</td>
          <td>25.715672</td>
          <td>0.225857</td>
          <td>25.516434</td>
          <td>0.404583</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.472809</td>
          <td>0.835182</td>
          <td>26.619923</td>
          <td>0.177912</td>
          <td>25.923057</td>
          <td>0.087536</td>
          <td>25.689039</td>
          <td>0.116623</td>
          <td>24.985694</td>
          <td>0.118796</td>
          <td>25.210757</td>
          <td>0.312143</td>
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
          <td>27.606027</td>
          <td>0.830472</td>
          <td>26.921453</td>
          <td>0.198305</td>
          <td>26.086665</td>
          <td>0.085121</td>
          <td>25.235807</td>
          <td>0.065450</td>
          <td>25.073602</td>
          <td>0.108203</td>
          <td>25.514316</td>
          <td>0.338541</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.334527</td>
          <td>2.056292</td>
          <td>28.445400</td>
          <td>0.647199</td>
          <td>27.674856</td>
          <td>0.326869</td>
          <td>26.990243</td>
          <td>0.294336</td>
          <td>26.808084</td>
          <td>0.452424</td>
          <td>26.868199</td>
          <td>0.891869</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.256839</td>
          <td>0.688602</td>
          <td>26.090109</td>
          <td>0.104188</td>
          <td>24.763401</td>
          <td>0.028675</td>
          <td>23.882536</td>
          <td>0.021672</td>
          <td>23.144190</td>
          <td>0.021504</td>
          <td>22.870402</td>
          <td>0.037887</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.885258</td>
          <td>0.578848</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.088481</td>
          <td>0.250036</td>
          <td>26.302100</td>
          <td>0.207883</td>
          <td>27.026593</td>
          <td>0.637689</td>
          <td>26.135485</td>
          <td>0.654001</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.261428</td>
          <td>0.312693</td>
          <td>25.698979</td>
          <td>0.068807</td>
          <td>25.396727</td>
          <td>0.046253</td>
          <td>24.829224</td>
          <td>0.045686</td>
          <td>24.285841</td>
          <td>0.054064</td>
          <td>23.567255</td>
          <td>0.064603</td>
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
          <td>26.527287</td>
          <td>0.404435</td>
          <td>26.133179</td>
          <td>0.107859</td>
          <td>26.293193</td>
          <td>0.110355</td>
          <td>25.930784</td>
          <td>0.130766</td>
          <td>25.804145</td>
          <td>0.218209</td>
          <td>25.664709</td>
          <td>0.409106</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.959015</td>
          <td>2.618000</td>
          <td>27.315085</td>
          <td>0.278236</td>
          <td>26.946274</td>
          <td>0.182138</td>
          <td>27.168020</td>
          <td>0.344225</td>
          <td>25.795364</td>
          <td>0.204175</td>
          <td>25.463207</td>
          <td>0.330119</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.812830</td>
          <td>0.492450</td>
          <td>27.082666</td>
          <td>0.236175</td>
          <td>26.672514</td>
          <td>0.148829</td>
          <td>26.529689</td>
          <td>0.211304</td>
          <td>25.289281</td>
          <td>0.136932</td>
          <td>25.230864</td>
          <td>0.282530</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.330223</td>
          <td>0.646568</td>
          <td>26.607035</td>
          <td>0.150029</td>
          <td>25.846389</td>
          <td>0.126118</td>
          <td>25.497868</td>
          <td>0.174535</td>
          <td>25.007323</td>
          <td>0.250750</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.400548</td>
          <td>1.346499</td>
          <td>26.737204</td>
          <td>0.175341</td>
          <td>26.006319</td>
          <td>0.082451</td>
          <td>25.744994</td>
          <td>0.106769</td>
          <td>25.105780</td>
          <td>0.115646</td>
          <td>24.805715</td>
          <td>0.196908</td>
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
