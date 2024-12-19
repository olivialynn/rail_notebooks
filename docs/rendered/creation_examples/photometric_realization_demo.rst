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

    <pzflow.flow.Flow at 0x7fcc60d05bd0>



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
          <td>27.310342</td>
          <td>0.682488</td>
          <td>26.723777</td>
          <td>0.167756</td>
          <td>25.965561</td>
          <td>0.076485</td>
          <td>25.524715</td>
          <td>0.084482</td>
          <td>24.977837</td>
          <td>0.099495</td>
          <td>25.346401</td>
          <td>0.296039</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.997325</td>
          <td>0.468154</td>
          <td>27.436144</td>
          <td>0.269487</td>
          <td>26.781175</td>
          <td>0.248022</td>
          <td>27.744968</td>
          <td>0.867667</td>
          <td>26.237045</td>
          <td>0.583513</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.691589</td>
          <td>0.436725</td>
          <td>26.040386</td>
          <td>0.092836</td>
          <td>24.820799</td>
          <td>0.027778</td>
          <td>23.858658</td>
          <td>0.019535</td>
          <td>23.149438</td>
          <td>0.019941</td>
          <td>22.777636</td>
          <td>0.032046</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.828887</td>
          <td>1.642956</td>
          <td>27.783669</td>
          <td>0.398050</td>
          <td>27.104791</td>
          <td>0.204898</td>
          <td>26.657926</td>
          <td>0.223989</td>
          <td>25.830460</td>
          <td>0.206996</td>
          <td>25.479033</td>
          <td>0.329168</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.730265</td>
          <td>0.202198</td>
          <td>25.775522</td>
          <td>0.073529</td>
          <td>25.473554</td>
          <td>0.049447</td>
          <td>24.795383</td>
          <td>0.044267</td>
          <td>24.423134</td>
          <td>0.060981</td>
          <td>23.740081</td>
          <td>0.075169</td>
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
          <td>26.799893</td>
          <td>0.473752</td>
          <td>26.390987</td>
          <td>0.126043</td>
          <td>26.219574</td>
          <td>0.095661</td>
          <td>25.920294</td>
          <td>0.119460</td>
          <td>25.843718</td>
          <td>0.209305</td>
          <td>25.402276</td>
          <td>0.309626</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.736541</td>
          <td>0.451798</td>
          <td>26.776143</td>
          <td>0.175389</td>
          <td>26.995819</td>
          <td>0.186944</td>
          <td>26.360697</td>
          <td>0.174468</td>
          <td>25.801114</td>
          <td>0.201966</td>
          <td>25.473014</td>
          <td>0.327598</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.170360</td>
          <td>0.619452</td>
          <td>27.551143</td>
          <td>0.331883</td>
          <td>26.773035</td>
          <td>0.154664</td>
          <td>26.130168</td>
          <td>0.143248</td>
          <td>26.125584</td>
          <td>0.264245</td>
          <td>25.806707</td>
          <td>0.424793</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.397199</td>
          <td>1.322209</td>
          <td>27.190701</td>
          <td>0.248031</td>
          <td>26.645060</td>
          <td>0.138552</td>
          <td>26.078675</td>
          <td>0.137029</td>
          <td>25.300081</td>
          <td>0.131729</td>
          <td>25.693929</td>
          <td>0.389561</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.943283</td>
          <td>0.526582</td>
          <td>26.387250</td>
          <td>0.125635</td>
          <td>26.094843</td>
          <td>0.085725</td>
          <td>25.529042</td>
          <td>0.084805</td>
          <td>25.256805</td>
          <td>0.126885</td>
          <td>24.646898</td>
          <td>0.165554</td>
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
          <td>27.930774</td>
          <td>1.098852</td>
          <td>26.555764</td>
          <td>0.167008</td>
          <td>25.956639</td>
          <td>0.089258</td>
          <td>25.494037</td>
          <td>0.097345</td>
          <td>25.131668</td>
          <td>0.133490</td>
          <td>25.429710</td>
          <td>0.367694</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.501687</td>
          <td>0.846153</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.318368</td>
          <td>0.284381</td>
          <td>27.628553</td>
          <td>0.555426</td>
          <td>26.111505</td>
          <td>0.303189</td>
          <td>25.550665</td>
          <td>0.403891</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.521867</td>
          <td>0.431208</td>
          <td>25.849418</td>
          <td>0.092437</td>
          <td>24.801771</td>
          <td>0.032861</td>
          <td>23.878617</td>
          <td>0.023981</td>
          <td>23.124262</td>
          <td>0.023373</td>
          <td>22.898598</td>
          <td>0.043206</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.482429</td>
          <td>0.775648</td>
          <td>27.311503</td>
          <td>0.300726</td>
          <td>26.680096</td>
          <td>0.284795</td>
          <td>25.721707</td>
          <td>0.234762</td>
          <td>25.315836</td>
          <td>0.357617</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.131250</td>
          <td>0.313346</td>
          <td>25.656942</td>
          <td>0.076503</td>
          <td>25.489850</td>
          <td>0.059106</td>
          <td>24.844715</td>
          <td>0.054874</td>
          <td>24.447141</td>
          <td>0.073343</td>
          <td>23.639593</td>
          <td>0.081400</td>
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
          <td>25.861467</td>
          <td>0.255635</td>
          <td>26.493479</td>
          <td>0.161317</td>
          <td>26.102717</td>
          <td>0.103610</td>
          <td>26.131660</td>
          <td>0.172605</td>
          <td>26.699394</td>
          <td>0.486674</td>
          <td>25.404783</td>
          <td>0.367581</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.710942</td>
          <td>0.966340</td>
          <td>27.089526</td>
          <td>0.261765</td>
          <td>26.911796</td>
          <td>0.204140</td>
          <td>26.790528</td>
          <td>0.293304</td>
          <td>26.948149</td>
          <td>0.575294</td>
          <td>25.360831</td>
          <td>0.349704</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.879680</td>
          <td>0.221794</td>
          <td>27.032239</td>
          <td>0.227567</td>
          <td>26.664111</td>
          <td>0.266907</td>
          <td>26.284671</td>
          <td>0.351890</td>
          <td>25.064055</td>
          <td>0.278061</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.049233</td>
          <td>0.561032</td>
          <td>26.425795</td>
          <td>0.138603</td>
          <td>25.810076</td>
          <td>0.132392</td>
          <td>25.792883</td>
          <td>0.240762</td>
          <td>25.230711</td>
          <td>0.323533</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.800020</td>
          <td>0.526171</td>
          <td>26.375810</td>
          <td>0.144451</td>
          <td>26.096720</td>
          <td>0.101948</td>
          <td>25.643998</td>
          <td>0.112137</td>
          <td>25.099358</td>
          <td>0.131101</td>
          <td>24.677274</td>
          <td>0.201480</td>
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
          <td>27.738141</td>
          <td>0.903055</td>
          <td>26.834816</td>
          <td>0.184344</td>
          <td>26.164127</td>
          <td>0.091126</td>
          <td>25.297555</td>
          <td>0.069130</td>
          <td>24.963063</td>
          <td>0.098227</td>
          <td>24.792529</td>
          <td>0.187357</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.483689</td>
          <td>0.664553</td>
          <td>27.847064</td>
          <td>0.374298</td>
          <td>26.619555</td>
          <td>0.217151</td>
          <td>26.278296</td>
          <td>0.299327</td>
          <td>25.584354</td>
          <td>0.358008</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.768793</td>
          <td>0.486423</td>
          <td>26.103256</td>
          <td>0.105391</td>
          <td>24.764344</td>
          <td>0.028699</td>
          <td>23.885950</td>
          <td>0.021736</td>
          <td>23.106902</td>
          <td>0.020832</td>
          <td>22.889586</td>
          <td>0.038535</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.880909</td>
          <td>0.994903</td>
          <td>27.423334</td>
          <td>0.327771</td>
          <td>26.038934</td>
          <td>0.166447</td>
          <td>25.976902</td>
          <td>0.288287</td>
          <td>25.896238</td>
          <td>0.552255</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.979751</td>
          <td>0.248904</td>
          <td>25.723310</td>
          <td>0.070302</td>
          <td>25.487218</td>
          <td>0.050123</td>
          <td>24.811405</td>
          <td>0.044969</td>
          <td>24.372643</td>
          <td>0.058394</td>
          <td>23.733891</td>
          <td>0.074870</td>
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
          <td>26.239835</td>
          <td>0.323048</td>
          <td>26.453048</td>
          <td>0.142313</td>
          <td>26.061751</td>
          <td>0.090103</td>
          <td>26.237505</td>
          <td>0.170152</td>
          <td>25.933895</td>
          <td>0.242983</td>
          <td>25.252481</td>
          <td>0.295771</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.596752</td>
          <td>0.410357</td>
          <td>27.053079</td>
          <td>0.224367</td>
          <td>26.911051</td>
          <td>0.176782</td>
          <td>26.541217</td>
          <td>0.206566</td>
          <td>26.168753</td>
          <td>0.277903</td>
          <td>26.110070</td>
          <td>0.540112</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.621840</td>
          <td>0.860140</td>
          <td>27.534506</td>
          <td>0.340383</td>
          <td>27.203048</td>
          <td>0.232931</td>
          <td>26.494624</td>
          <td>0.205193</td>
          <td>26.028578</td>
          <td>0.255439</td>
          <td>25.522025</td>
          <td>0.356400</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.197037</td>
          <td>0.673284</td>
          <td>27.037577</td>
          <td>0.240326</td>
          <td>26.610041</td>
          <td>0.150417</td>
          <td>25.776044</td>
          <td>0.118646</td>
          <td>25.541480</td>
          <td>0.181110</td>
          <td>25.647765</td>
          <td>0.417092</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.352942</td>
          <td>0.344314</td>
          <td>26.810766</td>
          <td>0.186604</td>
          <td>26.014403</td>
          <td>0.083041</td>
          <td>25.575319</td>
          <td>0.092017</td>
          <td>25.080741</td>
          <td>0.113151</td>
          <td>24.999565</td>
          <td>0.231498</td>
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
