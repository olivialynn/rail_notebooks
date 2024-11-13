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

    <pzflow.flow.Flow at 0x7f1ecbaff6a0>



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
          <td>28.791680</td>
          <td>1.614058</td>
          <td>26.652335</td>
          <td>0.157838</td>
          <td>26.007386</td>
          <td>0.079362</td>
          <td>25.346697</td>
          <td>0.072193</td>
          <td>24.913671</td>
          <td>0.094049</td>
          <td>25.604554</td>
          <td>0.363398</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.018511</td>
          <td>0.556080</td>
          <td>29.296494</td>
          <td>1.111256</td>
          <td>27.882406</td>
          <td>0.384396</td>
          <td>27.313157</td>
          <td>0.379755</td>
          <td>28.144366</td>
          <td>1.105275</td>
          <td>26.577954</td>
          <td>0.738381</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.664018</td>
          <td>0.427684</td>
          <td>25.928464</td>
          <td>0.084141</td>
          <td>24.813390</td>
          <td>0.027599</td>
          <td>23.896322</td>
          <td>0.020168</td>
          <td>23.170296</td>
          <td>0.020297</td>
          <td>22.847789</td>
          <td>0.034090</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.112855</td>
          <td>0.594845</td>
          <td>27.872857</td>
          <td>0.426190</td>
          <td>27.245807</td>
          <td>0.230456</td>
          <td>26.510032</td>
          <td>0.197937</td>
          <td>26.152400</td>
          <td>0.270088</td>
          <td>25.750429</td>
          <td>0.406897</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.713505</td>
          <td>0.444022</td>
          <td>25.821929</td>
          <td>0.076602</td>
          <td>25.405752</td>
          <td>0.046558</td>
          <td>24.732314</td>
          <td>0.041858</td>
          <td>24.354546</td>
          <td>0.057381</td>
          <td>23.664214</td>
          <td>0.070291</td>
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
          <td>26.463624</td>
          <td>0.366499</td>
          <td>26.264872</td>
          <td>0.112966</td>
          <td>26.117281</td>
          <td>0.087435</td>
          <td>25.793122</td>
          <td>0.106926</td>
          <td>26.043590</td>
          <td>0.247066</td>
          <td>25.419750</td>
          <td>0.313984</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.274359</td>
          <td>0.665857</td>
          <td>27.373224</td>
          <td>0.287825</td>
          <td>27.124036</td>
          <td>0.208227</td>
          <td>26.459487</td>
          <td>0.189687</td>
          <td>26.206772</td>
          <td>0.282287</td>
          <td>25.941657</td>
          <td>0.470328</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.405273</td>
          <td>0.727789</td>
          <td>27.809421</td>
          <td>0.406013</td>
          <td>26.856876</td>
          <td>0.166151</td>
          <td>26.749776</td>
          <td>0.241689</td>
          <td>25.939577</td>
          <td>0.226713</td>
          <td>26.371309</td>
          <td>0.641345</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.961494</td>
          <td>0.533609</td>
          <td>27.173296</td>
          <td>0.244504</td>
          <td>26.433056</td>
          <td>0.115293</td>
          <td>26.043320</td>
          <td>0.132908</td>
          <td>25.716794</td>
          <td>0.188127</td>
          <td>25.974811</td>
          <td>0.482094</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.876764</td>
          <td>0.501526</td>
          <td>26.478730</td>
          <td>0.135975</td>
          <td>26.068345</td>
          <td>0.083746</td>
          <td>25.747873</td>
          <td>0.102777</td>
          <td>25.175714</td>
          <td>0.118259</td>
          <td>24.855262</td>
          <td>0.197503</td>
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
          <td>27.489414</td>
          <td>0.839436</td>
          <td>26.652951</td>
          <td>0.181367</td>
          <td>25.972446</td>
          <td>0.090507</td>
          <td>25.370683</td>
          <td>0.087349</td>
          <td>25.071852</td>
          <td>0.126756</td>
          <td>24.738866</td>
          <td>0.210072</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.259234</td>
          <td>0.721748</td>
          <td>28.087444</td>
          <td>0.563505</td>
          <td>28.267639</td>
          <td>0.587188</td>
          <td>29.523549</td>
          <td>1.708647</td>
          <td>25.950124</td>
          <td>0.266058</td>
          <td>25.860869</td>
          <td>0.510003</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.853446</td>
          <td>0.551224</td>
          <td>25.885329</td>
          <td>0.095393</td>
          <td>24.870439</td>
          <td>0.034911</td>
          <td>23.877301</td>
          <td>0.023953</td>
          <td>23.131818</td>
          <td>0.023526</td>
          <td>22.804379</td>
          <td>0.039748</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.539403</td>
          <td>0.899389</td>
          <td>27.287657</td>
          <td>0.323663</td>
          <td>27.360248</td>
          <td>0.312710</td>
          <td>26.383222</td>
          <td>0.223215</td>
          <td>26.691166</td>
          <td>0.502845</td>
          <td>25.814759</td>
          <td>0.522099</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.247811</td>
          <td>0.343692</td>
          <td>25.561651</td>
          <td>0.070335</td>
          <td>25.383012</td>
          <td>0.053762</td>
          <td>24.744651</td>
          <td>0.050212</td>
          <td>24.366344</td>
          <td>0.068285</td>
          <td>23.742625</td>
          <td>0.089129</td>
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
          <td>26.188679</td>
          <td>0.332699</td>
          <td>26.191811</td>
          <td>0.124451</td>
          <td>25.995169</td>
          <td>0.094292</td>
          <td>26.186299</td>
          <td>0.180794</td>
          <td>26.044375</td>
          <td>0.292761</td>
          <td>25.771879</td>
          <td>0.486213</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.796307</td>
          <td>1.017433</td>
          <td>27.113818</td>
          <td>0.267007</td>
          <td>26.416958</td>
          <td>0.133916</td>
          <td>26.875388</td>
          <td>0.313981</td>
          <td>26.052727</td>
          <td>0.290226</td>
          <td>25.759040</td>
          <td>0.474600</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.523589</td>
          <td>0.429052</td>
          <td>27.804215</td>
          <td>0.462081</td>
          <td>26.872395</td>
          <td>0.199132</td>
          <td>26.572378</td>
          <td>0.247585</td>
          <td>26.917484</td>
          <td>0.566759</td>
          <td>25.059270</td>
          <td>0.276982</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.976085</td>
          <td>0.532122</td>
          <td>26.305433</td>
          <td>0.124903</td>
          <td>25.816984</td>
          <td>0.133185</td>
          <td>25.938473</td>
          <td>0.271276</td>
          <td>25.460160</td>
          <td>0.387398</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.805469</td>
          <td>0.528264</td>
          <td>26.254751</td>
          <td>0.130139</td>
          <td>25.990696</td>
          <td>0.092899</td>
          <td>25.543723</td>
          <td>0.102734</td>
          <td>25.485403</td>
          <td>0.182443</td>
          <td>24.726625</td>
          <td>0.209984</td>
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
          <td>26.927658</td>
          <td>0.199341</td>
          <td>26.194508</td>
          <td>0.093591</td>
          <td>25.277555</td>
          <td>0.067916</td>
          <td>24.951804</td>
          <td>0.097262</td>
          <td>24.798736</td>
          <td>0.188342</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.927275</td>
          <td>0.444458</td>
          <td>27.496560</td>
          <td>0.283292</td>
          <td>27.229313</td>
          <td>0.356005</td>
          <td>27.036851</td>
          <td>0.535875</td>
          <td>26.524376</td>
          <td>0.712819</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.025237</td>
          <td>0.272531</td>
          <td>26.027337</td>
          <td>0.098623</td>
          <td>24.823668</td>
          <td>0.030230</td>
          <td>23.884814</td>
          <td>0.021715</td>
          <td>23.117444</td>
          <td>0.021019</td>
          <td>22.813442</td>
          <td>0.036026</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.848065</td>
          <td>0.496410</td>
          <td>27.744483</td>
          <td>0.420952</td>
          <td>26.520310</td>
          <td>0.249138</td>
          <td>26.110451</td>
          <td>0.320897</td>
          <td>25.183177</td>
          <td>0.320958</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.787706</td>
          <td>0.469856</td>
          <td>25.886954</td>
          <td>0.081221</td>
          <td>25.463922</td>
          <td>0.049097</td>
          <td>24.844021</td>
          <td>0.046290</td>
          <td>24.464072</td>
          <td>0.063326</td>
          <td>23.643181</td>
          <td>0.069097</td>
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
          <td>28.854367</td>
          <td>1.712938</td>
          <td>26.165143</td>
          <td>0.110907</td>
          <td>26.304190</td>
          <td>0.111418</td>
          <td>25.917488</td>
          <td>0.129270</td>
          <td>26.232384</td>
          <td>0.309701</td>
          <td>26.250228</td>
          <td>0.628806</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.171684</td>
          <td>0.293785</td>
          <td>27.297948</td>
          <td>0.274391</td>
          <td>26.929183</td>
          <td>0.179521</td>
          <td>26.415228</td>
          <td>0.185789</td>
          <td>25.919512</td>
          <td>0.226460</td>
          <td>25.498484</td>
          <td>0.339473</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.452677</td>
          <td>0.374658</td>
          <td>27.060552</td>
          <td>0.231894</td>
          <td>26.691693</td>
          <td>0.151299</td>
          <td>26.658213</td>
          <td>0.235138</td>
          <td>26.150605</td>
          <td>0.282154</td>
          <td>26.044029</td>
          <td>0.529305</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.751091</td>
          <td>1.655153</td>
          <td>27.124128</td>
          <td>0.258040</td>
          <td>26.607923</td>
          <td>0.150143</td>
          <td>25.807224</td>
          <td>0.121905</td>
          <td>26.073852</td>
          <td>0.281688</td>
          <td>25.111178</td>
          <td>0.272970</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.172195</td>
          <td>0.633844</td>
          <td>26.706534</td>
          <td>0.170833</td>
          <td>26.073061</td>
          <td>0.087444</td>
          <td>25.606108</td>
          <td>0.094539</td>
          <td>25.427030</td>
          <td>0.152655</td>
          <td>24.897928</td>
          <td>0.212730</td>
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
