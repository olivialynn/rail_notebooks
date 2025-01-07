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

    <pzflow.flow.Flow at 0x7f201fb65d80>



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
          <td>27.349657</td>
          <td>0.700998</td>
          <td>26.689137</td>
          <td>0.162876</td>
          <td>26.024495</td>
          <td>0.080570</td>
          <td>25.337258</td>
          <td>0.071593</td>
          <td>25.034325</td>
          <td>0.104538</td>
          <td>24.932357</td>
          <td>0.210691</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.123302</td>
          <td>0.599260</td>
          <td>27.468211</td>
          <td>0.310670</td>
          <td>27.320390</td>
          <td>0.245105</td>
          <td>27.084074</td>
          <td>0.317057</td>
          <td>26.375569</td>
          <td>0.323278</td>
          <td>24.992940</td>
          <td>0.221610</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.837176</td>
          <td>0.487066</td>
          <td>25.903934</td>
          <td>0.082343</td>
          <td>24.793501</td>
          <td>0.027124</td>
          <td>23.842728</td>
          <td>0.019274</td>
          <td>23.120311</td>
          <td>0.019456</td>
          <td>22.789032</td>
          <td>0.032369</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.577774</td>
          <td>0.815414</td>
          <td>29.914630</td>
          <td>1.545205</td>
          <td>27.984394</td>
          <td>0.415802</td>
          <td>26.121855</td>
          <td>0.142227</td>
          <td>25.574828</td>
          <td>0.166783</td>
          <td>25.718813</td>
          <td>0.397120</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.606750</td>
          <td>0.409394</td>
          <td>25.784148</td>
          <td>0.074091</td>
          <td>25.389621</td>
          <td>0.045897</td>
          <td>24.771334</td>
          <td>0.043333</td>
          <td>24.424080</td>
          <td>0.061032</td>
          <td>23.668685</td>
          <td>0.070569</td>
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
          <td>26.063787</td>
          <td>0.266366</td>
          <td>26.275609</td>
          <td>0.114027</td>
          <td>26.230713</td>
          <td>0.096601</td>
          <td>26.151953</td>
          <td>0.145958</td>
          <td>25.839254</td>
          <td>0.208525</td>
          <td>26.474325</td>
          <td>0.688496</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.381345</td>
          <td>0.343598</td>
          <td>27.323836</td>
          <td>0.276537</td>
          <td>26.606547</td>
          <td>0.134021</td>
          <td>26.598436</td>
          <td>0.213157</td>
          <td>25.925662</td>
          <td>0.224107</td>
          <td>25.832774</td>
          <td>0.433297</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.464939</td>
          <td>0.366875</td>
          <td>27.137341</td>
          <td>0.237360</td>
          <td>26.886141</td>
          <td>0.170344</td>
          <td>26.343959</td>
          <td>0.172004</td>
          <td>26.275076</td>
          <td>0.298294</td>
          <td>25.390196</td>
          <td>0.306644</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.910242</td>
          <td>2.561875</td>
          <td>26.820451</td>
          <td>0.182098</td>
          <td>26.694191</td>
          <td>0.144542</td>
          <td>25.912443</td>
          <td>0.118648</td>
          <td>25.456966</td>
          <td>0.150794</td>
          <td>25.909842</td>
          <td>0.459254</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.261145</td>
          <td>0.312343</td>
          <td>26.537595</td>
          <td>0.143047</td>
          <td>26.177477</td>
          <td>0.092189</td>
          <td>25.682740</td>
          <td>0.097075</td>
          <td>25.405183</td>
          <td>0.144232</td>
          <td>24.584118</td>
          <td>0.156911</td>
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
          <td>28.211663</td>
          <td>1.285767</td>
          <td>26.747369</td>
          <td>0.196398</td>
          <td>26.021637</td>
          <td>0.094503</td>
          <td>25.413596</td>
          <td>0.090708</td>
          <td>24.900611</td>
          <td>0.109218</td>
          <td>24.857564</td>
          <td>0.231883</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.946471</td>
          <td>1.108979</td>
          <td>28.515750</td>
          <td>0.757562</td>
          <td>27.608973</td>
          <td>0.358501</td>
          <td>28.496027</td>
          <td>0.988150</td>
          <td>26.556210</td>
          <td>0.429355</td>
          <td>25.693192</td>
          <td>0.450174</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.495763</td>
          <td>0.422734</td>
          <td>26.052298</td>
          <td>0.110375</td>
          <td>24.821417</td>
          <td>0.033434</td>
          <td>23.878643</td>
          <td>0.023981</td>
          <td>23.089442</td>
          <td>0.022684</td>
          <td>22.776936</td>
          <td>0.038795</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.284538</td>
          <td>1.257041</td>
          <td>27.847255</td>
          <td>0.456426</td>
          <td>27.056315</td>
          <td>0.383780</td>
          <td>25.957787</td>
          <td>0.284799</td>
          <td>25.420559</td>
          <td>0.388015</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.407104</td>
          <td>0.389182</td>
          <td>26.000620</td>
          <td>0.103444</td>
          <td>25.437083</td>
          <td>0.056404</td>
          <td>24.793039</td>
          <td>0.052414</td>
          <td>24.407892</td>
          <td>0.070842</td>
          <td>23.722454</td>
          <td>0.087562</td>
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
          <td>27.861473</td>
          <td>1.066630</td>
          <td>26.497663</td>
          <td>0.161894</td>
          <td>26.490493</td>
          <td>0.145055</td>
          <td>26.127054</td>
          <td>0.171931</td>
          <td>26.498987</td>
          <td>0.418502</td>
          <td>25.407105</td>
          <td>0.368248</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.902932</td>
          <td>0.564747</td>
          <td>27.533930</td>
          <td>0.373297</td>
          <td>26.780679</td>
          <td>0.182796</td>
          <td>26.689970</td>
          <td>0.270348</td>
          <td>25.958787</td>
          <td>0.268932</td>
          <td>25.362595</td>
          <td>0.350189</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.672661</td>
          <td>2.477884</td>
          <td>27.809025</td>
          <td>0.463750</td>
          <td>26.517182</td>
          <td>0.147233</td>
          <td>26.081478</td>
          <td>0.164030</td>
          <td>25.876514</td>
          <td>0.253456</td>
          <td>25.948206</td>
          <td>0.549364</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.503473</td>
          <td>0.427799</td>
          <td>27.461522</td>
          <td>0.360585</td>
          <td>26.482163</td>
          <td>0.145496</td>
          <td>25.955504</td>
          <td>0.150059</td>
          <td>26.469929</td>
          <td>0.413028</td>
          <td>25.979320</td>
          <td>0.570656</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.612989</td>
          <td>0.912600</td>
          <td>26.565309</td>
          <td>0.169854</td>
          <td>26.060260</td>
          <td>0.098744</td>
          <td>25.697716</td>
          <td>0.117507</td>
          <td>25.224760</td>
          <td>0.146071</td>
          <td>24.549289</td>
          <td>0.180873</td>
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
          <td>27.794182</td>
          <td>0.935041</td>
          <td>26.852808</td>
          <td>0.187167</td>
          <td>25.944252</td>
          <td>0.075068</td>
          <td>25.434902</td>
          <td>0.078058</td>
          <td>25.034606</td>
          <td>0.104578</td>
          <td>25.495944</td>
          <td>0.333654</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.043467</td>
          <td>0.484865</td>
          <td>28.015795</td>
          <td>0.426236</td>
          <td>27.506253</td>
          <td>0.440747</td>
          <td>26.386788</td>
          <td>0.326455</td>
          <td>26.306886</td>
          <td>0.613571</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.560184</td>
          <td>0.841535</td>
          <td>25.960153</td>
          <td>0.092985</td>
          <td>24.797324</td>
          <td>0.029540</td>
          <td>23.874366</td>
          <td>0.021522</td>
          <td>23.114274</td>
          <td>0.020963</td>
          <td>22.887321</td>
          <td>0.038458</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.314976</td>
          <td>1.398529</td>
          <td>27.997363</td>
          <td>0.553568</td>
          <td>27.704590</td>
          <td>0.408297</td>
          <td>26.489262</td>
          <td>0.242851</td>
          <td>27.909521</td>
          <td>1.118641</td>
          <td>25.016134</td>
          <td>0.280631</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.236193</td>
          <td>0.306447</td>
          <td>25.755912</td>
          <td>0.072356</td>
          <td>25.472341</td>
          <td>0.049465</td>
          <td>24.802306</td>
          <td>0.044607</td>
          <td>24.468951</td>
          <td>0.063601</td>
          <td>23.734173</td>
          <td>0.074889</td>
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
          <td>26.148257</td>
          <td>0.300258</td>
          <td>26.418972</td>
          <td>0.138198</td>
          <td>26.154270</td>
          <td>0.097728</td>
          <td>26.095258</td>
          <td>0.150677</td>
          <td>25.946937</td>
          <td>0.245607</td>
          <td>25.119647</td>
          <td>0.265567</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.687197</td>
          <td>0.439601</td>
          <td>26.893729</td>
          <td>0.196390</td>
          <td>26.609146</td>
          <td>0.136516</td>
          <td>26.414478</td>
          <td>0.185671</td>
          <td>25.692054</td>
          <td>0.187174</td>
          <td>25.781168</td>
          <td>0.422829</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.055886</td>
          <td>0.587386</td>
          <td>27.409996</td>
          <td>0.308288</td>
          <td>26.706090</td>
          <td>0.153178</td>
          <td>26.468073</td>
          <td>0.200673</td>
          <td>25.952994</td>
          <td>0.240040</td>
          <td>25.819933</td>
          <td>0.448258</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.638298</td>
          <td>0.899391</td>
          <td>27.844462</td>
          <td>0.454957</td>
          <td>26.666710</td>
          <td>0.157899</td>
          <td>25.889212</td>
          <td>0.130883</td>
          <td>25.649556</td>
          <td>0.198398</td>
          <td>25.434349</td>
          <td>0.353501</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.067605</td>
          <td>1.120884</td>
          <td>26.689784</td>
          <td>0.168417</td>
          <td>26.051984</td>
          <td>0.085836</td>
          <td>25.653950</td>
          <td>0.098592</td>
          <td>25.133369</td>
          <td>0.118456</td>
          <td>24.631716</td>
          <td>0.169953</td>
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
