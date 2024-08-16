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

    <pzflow.flow.Flow at 0x7fb60dc03c10>



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
          <td>27.078557</td>
          <td>0.580523</td>
          <td>26.659729</td>
          <td>0.158838</td>
          <td>26.074795</td>
          <td>0.084224</td>
          <td>25.272454</td>
          <td>0.067601</td>
          <td>25.243596</td>
          <td>0.125440</td>
          <td>24.720149</td>
          <td>0.176198</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.549239</td>
          <td>1.431409</td>
          <td>28.548208</td>
          <td>0.694112</td>
          <td>27.777418</td>
          <td>0.354159</td>
          <td>27.315346</td>
          <td>0.380401</td>
          <td>26.770346</td>
          <td>0.439359</td>
          <td>25.826612</td>
          <td>0.431275</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.638764</td>
          <td>0.419538</td>
          <td>25.970159</td>
          <td>0.087283</td>
          <td>24.775883</td>
          <td>0.026710</td>
          <td>23.860863</td>
          <td>0.019571</td>
          <td>23.144153</td>
          <td>0.019852</td>
          <td>22.822669</td>
          <td>0.033343</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.657911</td>
          <td>2.336068</td>
          <td>29.613309</td>
          <td>1.324518</td>
          <td>27.387283</td>
          <td>0.258944</td>
          <td>26.898829</td>
          <td>0.273082</td>
          <td>25.816392</td>
          <td>0.204570</td>
          <td>24.849912</td>
          <td>0.196616</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.705779</td>
          <td>0.441438</td>
          <td>25.815907</td>
          <td>0.076196</td>
          <td>25.510314</td>
          <td>0.051088</td>
          <td>24.832449</td>
          <td>0.045748</td>
          <td>24.359180</td>
          <td>0.057618</td>
          <td>23.654185</td>
          <td>0.069670</td>
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
          <td>27.602589</td>
          <td>0.828582</td>
          <td>26.190037</td>
          <td>0.105830</td>
          <td>26.162097</td>
          <td>0.090951</td>
          <td>26.422423</td>
          <td>0.183839</td>
          <td>25.872692</td>
          <td>0.214434</td>
          <td>25.329345</td>
          <td>0.291997</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.429293</td>
          <td>0.739580</td>
          <td>27.361738</td>
          <td>0.285164</td>
          <td>26.829690</td>
          <td>0.162342</td>
          <td>26.622511</td>
          <td>0.217482</td>
          <td>26.200563</td>
          <td>0.280870</td>
          <td>25.389693</td>
          <td>0.306520</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.376179</td>
          <td>0.713685</td>
          <td>26.902070</td>
          <td>0.195079</td>
          <td>26.819602</td>
          <td>0.160950</td>
          <td>26.387007</td>
          <td>0.178407</td>
          <td>25.930061</td>
          <td>0.224928</td>
          <td>25.335582</td>
          <td>0.293469</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.037969</td>
          <td>1.082248</td>
          <td>26.947386</td>
          <td>0.202647</td>
          <td>26.787017</td>
          <td>0.156527</td>
          <td>25.673501</td>
          <td>0.096292</td>
          <td>25.398102</td>
          <td>0.143355</td>
          <td>25.421948</td>
          <td>0.314536</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.693430</td>
          <td>0.437334</td>
          <td>26.529530</td>
          <td>0.142058</td>
          <td>26.026082</td>
          <td>0.080683</td>
          <td>25.760391</td>
          <td>0.103909</td>
          <td>25.511270</td>
          <td>0.157974</td>
          <td>25.128163</td>
          <td>0.247845</td>
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
          <td>27.480962</td>
          <td>0.765393</td>
          <td>26.580739</td>
          <td>0.148449</td>
          <td>26.158271</td>
          <td>0.090646</td>
          <td>25.193695</td>
          <td>0.063043</td>
          <td>25.143103</td>
          <td>0.114949</td>
          <td>24.949668</td>
          <td>0.213760</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.777169</td>
          <td>0.925197</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.844392</td>
          <td>0.373207</td>
          <td>27.349819</td>
          <td>0.390699</td>
          <td>26.424503</td>
          <td>0.336082</td>
          <td>26.404118</td>
          <td>0.656099</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.257833</td>
          <td>0.658318</td>
          <td>25.883363</td>
          <td>0.080865</td>
          <td>24.754859</td>
          <td>0.026225</td>
          <td>23.885603</td>
          <td>0.019986</td>
          <td>23.166574</td>
          <td>0.020233</td>
          <td>22.813095</td>
          <td>0.033063</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.805464</td>
          <td>0.404781</td>
          <td>27.435179</td>
          <td>0.269275</td>
          <td>26.945230</td>
          <td>0.283564</td>
          <td>26.108428</td>
          <td>0.260565</td>
          <td>25.145067</td>
          <td>0.251312</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.160410</td>
          <td>0.288075</td>
          <td>25.816231</td>
          <td>0.076218</td>
          <td>25.509421</td>
          <td>0.051047</td>
          <td>24.847226</td>
          <td>0.046352</td>
          <td>24.356856</td>
          <td>0.057499</td>
          <td>23.731022</td>
          <td>0.074569</td>
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
          <td>26.901891</td>
          <td>0.510878</td>
          <td>26.345689</td>
          <td>0.121188</td>
          <td>26.028759</td>
          <td>0.080873</td>
          <td>26.029543</td>
          <td>0.131333</td>
          <td>25.829909</td>
          <td>0.206900</td>
          <td>25.293429</td>
          <td>0.283641</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.025130</td>
          <td>0.558735</td>
          <td>27.347056</td>
          <td>0.281795</td>
          <td>26.866903</td>
          <td>0.167577</td>
          <td>26.237310</td>
          <td>0.157046</td>
          <td>26.137146</td>
          <td>0.266751</td>
          <td>25.779460</td>
          <td>0.416050</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.042068</td>
          <td>0.565574</td>
          <td>27.167250</td>
          <td>0.243289</td>
          <td>26.960923</td>
          <td>0.181507</td>
          <td>26.290779</td>
          <td>0.164387</td>
          <td>25.826374</td>
          <td>0.206288</td>
          <td>25.751333</td>
          <td>0.407180</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.565161</td>
          <td>1.443091</td>
          <td>27.326166</td>
          <td>0.277061</td>
          <td>26.897116</td>
          <td>0.171942</td>
          <td>26.095334</td>
          <td>0.139013</td>
          <td>25.475344</td>
          <td>0.153189</td>
          <td>25.453364</td>
          <td>0.322518</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.758448</td>
          <td>0.459295</td>
          <td>26.510997</td>
          <td>0.139810</td>
          <td>26.068778</td>
          <td>0.083778</td>
          <td>25.632195</td>
          <td>0.092862</td>
          <td>25.283624</td>
          <td>0.129867</td>
          <td>24.844701</td>
          <td>0.195756</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.883255</td>
          <td>0.192013</td>
          <td>26.018971</td>
          <td>0.080178</td>
          <td>25.265010</td>
          <td>0.067157</td>
          <td>25.095394</td>
          <td>0.110267</td>
          <td>24.630338</td>
          <td>0.163232</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.050342</td>
          <td>0.568938</td>
          <td>28.964800</td>
          <td>0.910752</td>
          <td>27.662116</td>
          <td>0.323297</td>
          <td>28.516267</td>
          <td>0.888988</td>
          <td>26.310247</td>
          <td>0.306841</td>
          <td>25.598677</td>
          <td>0.361730</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.177134</td>
          <td>0.291986</td>
          <td>25.923087</td>
          <td>0.083743</td>
          <td>24.833620</td>
          <td>0.028091</td>
          <td>23.862658</td>
          <td>0.019601</td>
          <td>23.107671</td>
          <td>0.019250</td>
          <td>22.772968</td>
          <td>0.031915</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.651640</td>
          <td>0.423675</td>
          <td>27.625534</td>
          <td>0.351955</td>
          <td>27.085755</td>
          <td>0.201652</td>
          <td>26.239969</td>
          <td>0.157404</td>
          <td>25.941178</td>
          <td>0.227014</td>
          <td>25.013501</td>
          <td>0.225431</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.203205</td>
          <td>0.298176</td>
          <td>25.836412</td>
          <td>0.077587</td>
          <td>25.483924</td>
          <td>0.049905</td>
          <td>24.831265</td>
          <td>0.045700</td>
          <td>24.363378</td>
          <td>0.057833</td>
          <td>23.636823</td>
          <td>0.068607</td>
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
          <td>26.819089</td>
          <td>0.480570</td>
          <td>26.580937</td>
          <td>0.148475</td>
          <td>26.048818</td>
          <td>0.082317</td>
          <td>26.338821</td>
          <td>0.171254</td>
          <td>25.638186</td>
          <td>0.176016</td>
          <td>25.166049</td>
          <td>0.255676</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.788806</td>
          <td>0.469850</td>
          <td>27.097731</td>
          <td>0.229708</td>
          <td>26.861187</td>
          <td>0.166763</td>
          <td>26.260146</td>
          <td>0.160143</td>
          <td>26.213038</td>
          <td>0.283724</td>
          <td>26.491082</td>
          <td>0.696395</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.664788</td>
          <td>0.427935</td>
          <td>27.526957</td>
          <td>0.325572</td>
          <td>27.040011</td>
          <td>0.194044</td>
          <td>26.441359</td>
          <td>0.186806</td>
          <td>26.131388</td>
          <td>0.265500</td>
          <td>25.655492</td>
          <td>0.378122</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.014760</td>
          <td>0.554580</td>
          <td>27.199307</td>
          <td>0.249791</td>
          <td>26.537119</td>
          <td>0.126204</td>
          <td>25.690212</td>
          <td>0.097714</td>
          <td>25.413477</td>
          <td>0.145264</td>
          <td>26.399755</td>
          <td>0.654123</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.064882</td>
          <td>0.574886</td>
          <td>26.543148</td>
          <td>0.143732</td>
          <td>26.220850</td>
          <td>0.095768</td>
          <td>25.595714</td>
          <td>0.089931</td>
          <td>24.955853</td>
          <td>0.097596</td>
          <td>25.418332</td>
          <td>0.313629</td>
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
