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

    <pzflow.flow.Flow at 0x7faa31bdbe50>



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
          <td>27.771440</td>
          <td>0.921916</td>
          <td>26.698208</td>
          <td>0.164141</td>
          <td>26.099015</td>
          <td>0.086040</td>
          <td>25.417670</td>
          <td>0.076868</td>
          <td>24.984951</td>
          <td>0.100117</td>
          <td>24.806251</td>
          <td>0.189515</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.179586</td>
          <td>0.623469</td>
          <td>28.800254</td>
          <td>0.820426</td>
          <td>27.719828</td>
          <td>0.338444</td>
          <td>27.366602</td>
          <td>0.395796</td>
          <td>26.340597</td>
          <td>0.314386</td>
          <td>25.368323</td>
          <td>0.301307</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>30.321840</td>
          <td>2.940600</td>
          <td>25.789885</td>
          <td>0.074467</td>
          <td>24.783296</td>
          <td>0.026883</td>
          <td>23.864039</td>
          <td>0.019624</td>
          <td>23.132289</td>
          <td>0.019654</td>
          <td>22.846157</td>
          <td>0.034041</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.963496</td>
          <td>1.035876</td>
          <td>27.687831</td>
          <td>0.369548</td>
          <td>27.248830</td>
          <td>0.231034</td>
          <td>26.670431</td>
          <td>0.226328</td>
          <td>25.683595</td>
          <td>0.182922</td>
          <td>25.326609</td>
          <td>0.291353</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.348341</td>
          <td>0.334760</td>
          <td>25.779334</td>
          <td>0.073777</td>
          <td>25.392286</td>
          <td>0.046005</td>
          <td>24.801960</td>
          <td>0.044526</td>
          <td>24.240363</td>
          <td>0.051850</td>
          <td>23.743065</td>
          <td>0.075367</td>
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
          <td>26.676222</td>
          <td>0.431667</td>
          <td>26.380024</td>
          <td>0.124851</td>
          <td>26.021020</td>
          <td>0.080323</td>
          <td>25.993584</td>
          <td>0.127307</td>
          <td>25.878473</td>
          <td>0.215471</td>
          <td>25.383828</td>
          <td>0.305082</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.947618</td>
          <td>1.736618</td>
          <td>27.658761</td>
          <td>0.361248</td>
          <td>26.665196</td>
          <td>0.140978</td>
          <td>26.397988</td>
          <td>0.180075</td>
          <td>26.576003</td>
          <td>0.378494</td>
          <td>24.989920</td>
          <td>0.221054</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.115584</td>
          <td>0.595996</td>
          <td>27.201433</td>
          <td>0.250228</td>
          <td>26.897210</td>
          <td>0.171956</td>
          <td>26.184417</td>
          <td>0.150086</td>
          <td>26.473319</td>
          <td>0.349284</td>
          <td>26.098537</td>
          <td>0.528076</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.374710</td>
          <td>0.712978</td>
          <td>28.321488</td>
          <td>0.592930</td>
          <td>26.662676</td>
          <td>0.140672</td>
          <td>25.879933</td>
          <td>0.115338</td>
          <td>25.574912</td>
          <td>0.166795</td>
          <td>25.491319</td>
          <td>0.332392</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.599375</td>
          <td>0.407086</td>
          <td>26.597514</td>
          <td>0.150601</td>
          <td>26.179981</td>
          <td>0.092392</td>
          <td>25.477374</td>
          <td>0.081028</td>
          <td>25.365211</td>
          <td>0.139351</td>
          <td>25.086456</td>
          <td>0.239471</td>
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
          <td>26.353399</td>
          <td>0.140442</td>
          <td>26.020698</td>
          <td>0.094425</td>
          <td>25.344161</td>
          <td>0.085333</td>
          <td>24.939755</td>
          <td>0.113011</td>
          <td>25.260811</td>
          <td>0.321837</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.479916</td>
          <td>0.834430</td>
          <td>28.279240</td>
          <td>0.645239</td>
          <td>27.184144</td>
          <td>0.254920</td>
          <td>27.886271</td>
          <td>0.666032</td>
          <td>26.346910</td>
          <td>0.365365</td>
          <td>26.347896</td>
          <td>0.718836</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.333984</td>
          <td>0.373230</td>
          <td>25.847737</td>
          <td>0.092301</td>
          <td>24.785232</td>
          <td>0.032386</td>
          <td>23.865776</td>
          <td>0.023716</td>
          <td>23.152530</td>
          <td>0.023950</td>
          <td>22.803568</td>
          <td>0.039720</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.733284</td>
          <td>2.575727</td>
          <td>27.459402</td>
          <td>0.370542</td>
          <td>27.623373</td>
          <td>0.384714</td>
          <td>26.388415</td>
          <td>0.224181</td>
          <td>25.870285</td>
          <td>0.265248</td>
          <td>25.871379</td>
          <td>0.544054</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.471680</td>
          <td>0.409007</td>
          <td>25.888454</td>
          <td>0.093773</td>
          <td>25.522025</td>
          <td>0.060817</td>
          <td>24.809831</td>
          <td>0.053201</td>
          <td>24.433720</td>
          <td>0.072478</td>
          <td>23.602883</td>
          <td>0.078807</td>
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
          <td>26.332552</td>
          <td>0.372474</td>
          <td>26.151984</td>
          <td>0.120226</td>
          <td>26.225445</td>
          <td>0.115322</td>
          <td>25.902695</td>
          <td>0.141897</td>
          <td>25.896696</td>
          <td>0.259660</td>
          <td>25.192189</td>
          <td>0.310708</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.415904</td>
          <td>0.392844</td>
          <td>26.779832</td>
          <td>0.202551</td>
          <td>27.222461</td>
          <td>0.264013</td>
          <td>27.296829</td>
          <td>0.436061</td>
          <td>26.439660</td>
          <td>0.394036</td>
          <td>25.068197</td>
          <td>0.276726</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.602755</td>
          <td>0.908150</td>
          <td>26.972101</td>
          <td>0.239438</td>
          <td>26.607943</td>
          <td>0.159141</td>
          <td>26.452641</td>
          <td>0.224248</td>
          <td>25.908265</td>
          <td>0.260135</td>
          <td>25.036103</td>
          <td>0.271814</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.801045</td>
          <td>0.533762</td>
          <td>28.289716</td>
          <td>0.664507</td>
          <td>26.844969</td>
          <td>0.198093</td>
          <td>25.657550</td>
          <td>0.115984</td>
          <td>25.489873</td>
          <td>0.186936</td>
          <td>28.086987</td>
          <td>1.911578</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.195273</td>
          <td>0.123609</td>
          <td>26.039551</td>
          <td>0.096968</td>
          <td>25.578712</td>
          <td>0.105926</td>
          <td>25.196209</td>
          <td>0.142527</td>
          <td>24.685376</td>
          <td>0.202854</td>
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
          <td>26.595290</td>
          <td>0.405845</td>
          <td>26.544489</td>
          <td>0.143914</td>
          <td>25.924958</td>
          <td>0.073798</td>
          <td>25.260501</td>
          <td>0.066898</td>
          <td>25.002446</td>
          <td>0.101676</td>
          <td>24.770860</td>
          <td>0.183957</td>
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
          <td>28.167354</td>
          <td>0.477777</td>
          <td>28.515990</td>
          <td>0.889465</td>
          <td>26.572637</td>
          <td>0.377821</td>
          <td>27.943466</td>
          <td>1.615739</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.456587</td>
          <td>0.383878</td>
          <td>25.942996</td>
          <td>0.091596</td>
          <td>24.773998</td>
          <td>0.028942</td>
          <td>23.824388</td>
          <td>0.020624</td>
          <td>23.174318</td>
          <td>0.022066</td>
          <td>22.767790</td>
          <td>0.034604</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.534230</td>
          <td>0.894697</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.356419</td>
          <td>0.310742</td>
          <td>26.352348</td>
          <td>0.216793</td>
          <td>26.287777</td>
          <td>0.369060</td>
          <td>25.528539</td>
          <td>0.420245</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.211227</td>
          <td>0.300374</td>
          <td>25.683777</td>
          <td>0.067888</td>
          <td>25.386800</td>
          <td>0.045848</td>
          <td>24.824440</td>
          <td>0.045492</td>
          <td>24.324584</td>
          <td>0.055956</td>
          <td>23.652227</td>
          <td>0.069653</td>
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
          <td>27.572453</td>
          <td>0.846516</td>
          <td>26.193846</td>
          <td>0.113714</td>
          <td>26.051797</td>
          <td>0.089317</td>
          <td>25.956905</td>
          <td>0.133754</td>
          <td>26.023025</td>
          <td>0.261430</td>
          <td>25.892155</td>
          <td>0.485728</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.219510</td>
          <td>0.305288</td>
          <td>27.751704</td>
          <td>0.393267</td>
          <td>26.680110</td>
          <td>0.145124</td>
          <td>26.828878</td>
          <td>0.262109</td>
          <td>26.094931</td>
          <td>0.261681</td>
          <td>25.621774</td>
          <td>0.373952</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.946980</td>
          <td>0.543246</td>
          <td>27.482624</td>
          <td>0.326676</td>
          <td>27.099481</td>
          <td>0.213712</td>
          <td>26.115833</td>
          <td>0.148770</td>
          <td>26.586060</td>
          <td>0.398214</td>
          <td>25.771325</td>
          <td>0.432067</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.940324</td>
          <td>0.221728</td>
          <td>26.568325</td>
          <td>0.145122</td>
          <td>25.800571</td>
          <td>0.121202</td>
          <td>25.757031</td>
          <td>0.217074</td>
          <td>25.765153</td>
          <td>0.455910</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.450896</td>
          <td>0.371768</td>
          <td>26.474590</td>
          <td>0.140078</td>
          <td>26.097052</td>
          <td>0.089310</td>
          <td>25.566763</td>
          <td>0.091327</td>
          <td>25.114586</td>
          <td>0.116536</td>
          <td>25.234841</td>
          <td>0.280749</td>
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
