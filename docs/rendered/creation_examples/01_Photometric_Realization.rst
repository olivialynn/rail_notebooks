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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f01da714730>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.689675</td>
          <td>0.162951</td>
          <td>26.126557</td>
          <td>0.088152</td>
          <td>25.272089</td>
          <td>0.067579</td>
          <td>24.731671</td>
          <td>0.080126</td>
          <td>23.981904</td>
          <td>0.093022</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.960666</td>
          <td>0.533288</td>
          <td>27.498315</td>
          <td>0.318231</td>
          <td>26.731675</td>
          <td>0.149274</td>
          <td>26.301157</td>
          <td>0.165849</td>
          <td>25.753093</td>
          <td>0.193975</td>
          <td>25.458789</td>
          <td>0.323913</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.866496</td>
          <td>1.672390</td>
          <td>28.753426</td>
          <td>0.795853</td>
          <td>28.035277</td>
          <td>0.432243</td>
          <td>25.903114</td>
          <td>0.117689</td>
          <td>25.024985</td>
          <td>0.103688</td>
          <td>24.444242</td>
          <td>0.139145</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.462297</td>
          <td>0.755997</td>
          <td>29.270812</td>
          <td>1.094876</td>
          <td>27.380640</td>
          <td>0.257540</td>
          <td>26.262891</td>
          <td>0.160520</td>
          <td>25.417878</td>
          <td>0.145815</td>
          <td>25.514778</td>
          <td>0.338624</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.585403</td>
          <td>0.178993</td>
          <td>26.066844</td>
          <td>0.095015</td>
          <td>25.940040</td>
          <td>0.074779</td>
          <td>25.571622</td>
          <td>0.088045</td>
          <td>25.671579</td>
          <td>0.181070</td>
          <td>25.187077</td>
          <td>0.260118</td>
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
          <td>0.389450</td>
          <td>27.871327</td>
          <td>0.980164</td>
          <td>26.327296</td>
          <td>0.119268</td>
          <td>25.388026</td>
          <td>0.045832</td>
          <td>25.005442</td>
          <td>0.053343</td>
          <td>24.933999</td>
          <td>0.095742</td>
          <td>24.818799</td>
          <td>0.191532</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>32.705733</td>
          <td>5.258160</td>
          <td>26.863362</td>
          <td>0.188820</td>
          <td>25.948909</td>
          <td>0.075367</td>
          <td>25.230301</td>
          <td>0.065122</td>
          <td>24.801303</td>
          <td>0.085199</td>
          <td>24.038424</td>
          <td>0.097753</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.659957</td>
          <td>0.426366</td>
          <td>26.694360</td>
          <td>0.163604</td>
          <td>26.299002</td>
          <td>0.102558</td>
          <td>26.319271</td>
          <td>0.168428</td>
          <td>25.468286</td>
          <td>0.152265</td>
          <td>24.974949</td>
          <td>0.218316</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.605593</td>
          <td>0.409032</td>
          <td>26.287249</td>
          <td>0.115188</td>
          <td>26.111854</td>
          <td>0.087019</td>
          <td>25.976168</td>
          <td>0.125399</td>
          <td>25.604366</td>
          <td>0.171030</td>
          <td>25.087628</td>
          <td>0.239703</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.648958</td>
          <td>0.853566</td>
          <td>26.679842</td>
          <td>0.161590</td>
          <td>26.626191</td>
          <td>0.136314</td>
          <td>26.257166</td>
          <td>0.159736</td>
          <td>25.718395</td>
          <td>0.188382</td>
          <td>25.345083</td>
          <td>0.295725</td>
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
          <td>1.398944</td>
          <td>27.742474</td>
          <td>0.982877</td>
          <td>26.886923</td>
          <td>0.220709</td>
          <td>25.994608</td>
          <td>0.092286</td>
          <td>25.281555</td>
          <td>0.080752</td>
          <td>24.719531</td>
          <td>0.093208</td>
          <td>24.035035</td>
          <td>0.115084</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.699745</td>
          <td>0.485718</td>
          <td>28.372326</td>
          <td>0.687916</td>
          <td>26.690980</td>
          <td>0.168749</td>
          <td>26.049593</td>
          <td>0.157615</td>
          <td>25.715336</td>
          <td>0.219228</td>
          <td>25.452191</td>
          <td>0.374265</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.017999</td>
          <td>1.918201</td>
          <td>29.398727</td>
          <td>1.304025</td>
          <td>29.114216</td>
          <td>1.037815</td>
          <td>25.799501</td>
          <td>0.129974</td>
          <td>25.322190</td>
          <td>0.160698</td>
          <td>24.303776</td>
          <td>0.148508</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.012413</td>
          <td>1.190352</td>
          <td>28.863533</td>
          <td>0.986722</td>
          <td>26.907721</td>
          <td>0.216008</td>
          <td>25.993350</td>
          <td>0.160672</td>
          <td>25.636648</td>
          <td>0.218759</td>
          <td>25.414347</td>
          <td>0.386153</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.835392</td>
          <td>0.246567</td>
          <td>26.011430</td>
          <td>0.104425</td>
          <td>25.870859</td>
          <td>0.082792</td>
          <td>25.640156</td>
          <td>0.110651</td>
          <td>25.346473</td>
          <td>0.160600</td>
          <td>24.969061</td>
          <td>0.254281</td>
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
          <td>0.389450</td>
          <td>26.924257</td>
          <td>0.579454</td>
          <td>26.454608</td>
          <td>0.156049</td>
          <td>25.413698</td>
          <td>0.056415</td>
          <td>25.109644</td>
          <td>0.070911</td>
          <td>24.808980</td>
          <td>0.102943</td>
          <td>24.846618</td>
          <td>0.234507</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.357229</td>
          <td>0.375395</td>
          <td>26.643705</td>
          <td>0.180607</td>
          <td>25.913320</td>
          <td>0.086280</td>
          <td>25.222067</td>
          <td>0.076956</td>
          <td>24.842421</td>
          <td>0.104237</td>
          <td>24.380335</td>
          <td>0.155726</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.661456</td>
          <td>0.475934</td>
          <td>26.645498</td>
          <td>0.182241</td>
          <td>26.344049</td>
          <td>0.126801</td>
          <td>26.163652</td>
          <td>0.175909</td>
          <td>27.249195</td>
          <td>0.714017</td>
          <td>25.314851</td>
          <td>0.339940</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.045994</td>
          <td>0.299090</td>
          <td>25.985308</td>
          <td>0.104969</td>
          <td>26.064053</td>
          <td>0.101209</td>
          <td>25.708255</td>
          <td>0.121212</td>
          <td>25.666062</td>
          <td>0.216725</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.029655</td>
          <td>1.168514</td>
          <td>26.173832</td>
          <td>0.121332</td>
          <td>26.565838</td>
          <td>0.153098</td>
          <td>26.466527</td>
          <td>0.226241</td>
          <td>25.830340</td>
          <td>0.243387</td>
          <td>25.298862</td>
          <td>0.334816</td>
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
          <td>1.398944</td>
          <td>26.865734</td>
          <td>0.497501</td>
          <td>26.662553</td>
          <td>0.159239</td>
          <td>25.975906</td>
          <td>0.077197</td>
          <td>25.116683</td>
          <td>0.058888</td>
          <td>24.710860</td>
          <td>0.078678</td>
          <td>24.015821</td>
          <td>0.095847</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.237133</td>
          <td>0.257859</td>
          <td>26.634518</td>
          <td>0.137425</td>
          <td>26.246412</td>
          <td>0.158427</td>
          <td>25.889763</td>
          <td>0.217705</td>
          <td>25.435310</td>
          <td>0.318194</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.978845</td>
          <td>0.444408</td>
          <td>26.029874</td>
          <td>0.142887</td>
          <td>25.035892</td>
          <td>0.113546</td>
          <td>24.379449</td>
          <td>0.142995</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.954796</td>
          <td>1.039825</td>
          <td>27.127664</td>
          <td>0.258202</td>
          <td>26.584660</td>
          <td>0.262630</td>
          <td>25.534903</td>
          <td>0.200233</td>
          <td>28.053518</td>
          <td>1.914896</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.245388</td>
          <td>0.308710</td>
          <td>26.085886</td>
          <td>0.096733</td>
          <td>25.932238</td>
          <td>0.074371</td>
          <td>25.658572</td>
          <td>0.095180</td>
          <td>25.676158</td>
          <td>0.182022</td>
          <td>24.878635</td>
          <td>0.201703</td>
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
          <td>0.389450</td>
          <td>27.581920</td>
          <td>0.851649</td>
          <td>26.451869</td>
          <td>0.142168</td>
          <td>25.560476</td>
          <td>0.057847</td>
          <td>25.132197</td>
          <td>0.064881</td>
          <td>25.049765</td>
          <td>0.114571</td>
          <td>24.565632</td>
          <td>0.167196</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.089691</td>
          <td>0.590598</td>
          <td>26.536579</td>
          <td>0.144941</td>
          <td>26.119091</td>
          <td>0.089033</td>
          <td>25.115219</td>
          <td>0.059841</td>
          <td>24.695928</td>
          <td>0.078934</td>
          <td>24.168950</td>
          <td>0.111457</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.153581</td>
          <td>0.295695</td>
          <td>26.958995</td>
          <td>0.213120</td>
          <td>26.539566</td>
          <td>0.132719</td>
          <td>26.022861</td>
          <td>0.137327</td>
          <td>25.874642</td>
          <td>0.224960</td>
          <td>25.399571</td>
          <td>0.323526</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.643368</td>
          <td>0.203353</td>
          <td>26.039683</td>
          <td>0.102614</td>
          <td>26.058489</td>
          <td>0.093137</td>
          <td>26.091669</td>
          <td>0.155799</td>
          <td>25.337130</td>
          <td>0.152164</td>
          <td>25.320902</td>
          <td>0.323167</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.706148</td>
          <td>0.452015</td>
          <td>26.357964</td>
          <td>0.126657</td>
          <td>26.681606</td>
          <td>0.148535</td>
          <td>26.230174</td>
          <td>0.162410</td>
          <td>25.872006</td>
          <td>0.222356</td>
          <td>26.104682</td>
          <td>0.548420</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
