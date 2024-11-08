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

    <pzflow.flow.Flow at 0x7fbc1347e8c0>



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
          <td>26.867494</td>
          <td>0.498110</td>
          <td>26.557959</td>
          <td>0.145574</td>
          <td>25.949327</td>
          <td>0.075395</td>
          <td>25.289171</td>
          <td>0.068609</td>
          <td>25.012724</td>
          <td>0.102581</td>
          <td>25.127416</td>
          <td>0.247692</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.868547</td>
          <td>1.674001</td>
          <td>28.969577</td>
          <td>0.913466</td>
          <td>27.303080</td>
          <td>0.241633</td>
          <td>27.365879</td>
          <td>0.395575</td>
          <td>26.171922</td>
          <td>0.274414</td>
          <td>26.744305</td>
          <td>0.823629</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.709060</td>
          <td>0.442534</td>
          <td>25.929662</td>
          <td>0.084229</td>
          <td>24.836831</td>
          <td>0.028170</td>
          <td>23.887100</td>
          <td>0.020011</td>
          <td>23.161774</td>
          <td>0.020151</td>
          <td>22.783556</td>
          <td>0.032214</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.787541</td>
          <td>1.610857</td>
          <td>28.246909</td>
          <td>0.562183</td>
          <td>27.706290</td>
          <td>0.334837</td>
          <td>26.825114</td>
          <td>0.257131</td>
          <td>26.140909</td>
          <td>0.267570</td>
          <td>26.093700</td>
          <td>0.526216</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.045217</td>
          <td>0.262363</td>
          <td>25.775816</td>
          <td>0.073548</td>
          <td>25.411265</td>
          <td>0.046787</td>
          <td>24.743040</td>
          <td>0.042258</td>
          <td>24.413071</td>
          <td>0.060439</td>
          <td>23.612390</td>
          <td>0.067139</td>
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
          <td>26.780046</td>
          <td>0.466784</td>
          <td>26.228539</td>
          <td>0.109445</td>
          <td>26.035493</td>
          <td>0.081355</td>
          <td>26.447602</td>
          <td>0.187794</td>
          <td>26.612677</td>
          <td>0.389411</td>
          <td>26.575536</td>
          <td>0.737189</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.956876</td>
          <td>1.031813</td>
          <td>26.822331</td>
          <td>0.182388</td>
          <td>26.927506</td>
          <td>0.176438</td>
          <td>26.315080</td>
          <td>0.167828</td>
          <td>26.124371</td>
          <td>0.263983</td>
          <td>25.790070</td>
          <td>0.419437</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.311213</td>
          <td>0.273715</td>
          <td>27.225638</td>
          <td>0.226632</td>
          <td>26.377380</td>
          <td>0.176956</td>
          <td>25.852308</td>
          <td>0.210814</td>
          <td>25.537060</td>
          <td>0.344634</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.816558</td>
          <td>0.479667</td>
          <td>27.293917</td>
          <td>0.269890</td>
          <td>26.820020</td>
          <td>0.161007</td>
          <td>25.732263</td>
          <td>0.101382</td>
          <td>25.530392</td>
          <td>0.160578</td>
          <td>25.382610</td>
          <td>0.304784</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>31.292313</td>
          <td>3.867422</td>
          <td>26.503852</td>
          <td>0.138952</td>
          <td>26.143799</td>
          <td>0.089499</td>
          <td>25.779027</td>
          <td>0.105616</td>
          <td>25.396322</td>
          <td>0.143136</td>
          <td>24.738694</td>
          <td>0.178991</td>
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
          <td>25.823024</td>
          <td>0.244018</td>
          <td>27.337524</td>
          <td>0.318691</td>
          <td>26.270971</td>
          <td>0.117511</td>
          <td>25.378804</td>
          <td>0.087975</td>
          <td>25.380846</td>
          <td>0.165328</td>
          <td>24.872562</td>
          <td>0.234779</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.966729</td>
          <td>0.516261</td>
          <td>27.389494</td>
          <td>0.301172</td>
          <td>27.456019</td>
          <td>0.489615</td>
          <td>26.796409</td>
          <td>0.513748</td>
          <td>26.002628</td>
          <td>0.565305</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.318987</td>
          <td>0.368897</td>
          <td>26.149895</td>
          <td>0.120150</td>
          <td>24.795526</td>
          <td>0.032681</td>
          <td>23.881924</td>
          <td>0.024049</td>
          <td>23.073466</td>
          <td>0.022376</td>
          <td>22.866916</td>
          <td>0.042011</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>33.326303</td>
          <td>6.066439</td>
          <td>28.891180</td>
          <td>1.003308</td>
          <td>27.939940</td>
          <td>0.489119</td>
          <td>26.624204</td>
          <td>0.272166</td>
          <td>25.771468</td>
          <td>0.244604</td>
          <td>25.186994</td>
          <td>0.323004</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.058998</td>
          <td>0.295722</td>
          <td>25.871899</td>
          <td>0.092421</td>
          <td>25.463141</td>
          <td>0.057723</td>
          <td>24.819111</td>
          <td>0.053641</td>
          <td>24.295230</td>
          <td>0.064118</td>
          <td>23.682128</td>
          <td>0.084508</td>
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
          <td>27.832464</td>
          <td>1.048582</td>
          <td>26.608819</td>
          <td>0.177939</td>
          <td>26.148569</td>
          <td>0.107846</td>
          <td>26.559502</td>
          <td>0.246932</td>
          <td>26.043205</td>
          <td>0.292485</td>
          <td>25.977603</td>
          <td>0.565007</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.756227</td>
          <td>0.507681</td>
          <td>27.341561</td>
          <td>0.320810</td>
          <td>27.046810</td>
          <td>0.228469</td>
          <td>26.202583</td>
          <td>0.180255</td>
          <td>25.977045</td>
          <td>0.272959</td>
          <td>26.742596</td>
          <td>0.930458</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.148402</td>
          <td>0.276611</td>
          <td>26.698696</td>
          <td>0.171942</td>
          <td>26.974060</td>
          <td>0.342325</td>
          <td>25.786531</td>
          <td>0.235349</td>
          <td>24.973119</td>
          <td>0.258192</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.781712</td>
          <td>1.022991</td>
          <td>27.368836</td>
          <td>0.335213</td>
          <td>26.791523</td>
          <td>0.189375</td>
          <td>25.775186</td>
          <td>0.128455</td>
          <td>25.717933</td>
          <td>0.226282</td>
          <td>25.631360</td>
          <td>0.441631</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.641686</td>
          <td>0.928999</td>
          <td>26.184497</td>
          <td>0.122459</td>
          <td>25.905271</td>
          <td>0.086177</td>
          <td>25.429608</td>
          <td>0.092955</td>
          <td>25.020478</td>
          <td>0.122440</td>
          <td>25.252186</td>
          <td>0.322634</td>
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
          <td>28.518821</td>
          <td>1.409294</td>
          <td>26.921367</td>
          <td>0.198291</td>
          <td>26.011772</td>
          <td>0.079681</td>
          <td>25.488849</td>
          <td>0.081864</td>
          <td>25.044663</td>
          <td>0.105501</td>
          <td>24.607685</td>
          <td>0.160126</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.425202</td>
          <td>0.300363</td>
          <td>27.686734</td>
          <td>0.329968</td>
          <td>26.660687</td>
          <td>0.224714</td>
          <td>26.379691</td>
          <td>0.324618</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.388432</td>
          <td>0.364059</td>
          <td>25.970215</td>
          <td>0.093809</td>
          <td>24.841560</td>
          <td>0.030709</td>
          <td>23.874864</td>
          <td>0.021531</td>
          <td>23.150022</td>
          <td>0.021612</td>
          <td>22.850845</td>
          <td>0.037237</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.310587</td>
          <td>0.774943</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.912279</td>
          <td>0.477726</td>
          <td>26.449979</td>
          <td>0.235101</td>
          <td>26.415684</td>
          <td>0.407461</td>
          <td>27.726567</td>
          <td>1.651463</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.785119</td>
          <td>0.211879</td>
          <td>25.681205</td>
          <td>0.067734</td>
          <td>25.472607</td>
          <td>0.049477</td>
          <td>24.798679</td>
          <td>0.044464</td>
          <td>24.300130</td>
          <td>0.054754</td>
          <td>23.841607</td>
          <td>0.082339</td>
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
          <td>26.798870</td>
          <td>0.496248</td>
          <td>26.422571</td>
          <td>0.138627</td>
          <td>26.161303</td>
          <td>0.098332</td>
          <td>26.209542</td>
          <td>0.166148</td>
          <td>25.971073</td>
          <td>0.250532</td>
          <td>25.503685</td>
          <td>0.361105</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.256134</td>
          <td>0.265204</td>
          <td>26.748826</td>
          <td>0.153942</td>
          <td>26.464340</td>
          <td>0.193649</td>
          <td>26.346184</td>
          <td>0.320543</td>
          <td>25.418937</td>
          <td>0.318695</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.876899</td>
          <td>2.567136</td>
          <td>26.840700</td>
          <td>0.192998</td>
          <td>26.667847</td>
          <td>0.148233</td>
          <td>26.245545</td>
          <td>0.166231</td>
          <td>26.159298</td>
          <td>0.284147</td>
          <td>25.839556</td>
          <td>0.454932</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.611710</td>
          <td>0.380822</td>
          <td>26.459799</td>
          <td>0.132157</td>
          <td>25.902596</td>
          <td>0.132407</td>
          <td>25.567704</td>
          <td>0.185173</td>
          <td>24.533753</td>
          <td>0.168677</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.383857</td>
          <td>0.732483</td>
          <td>26.276923</td>
          <td>0.118059</td>
          <td>26.174337</td>
          <td>0.095585</td>
          <td>25.832637</td>
          <td>0.115253</td>
          <td>25.280197</td>
          <td>0.134534</td>
          <td>24.744540</td>
          <td>0.187012</td>
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
