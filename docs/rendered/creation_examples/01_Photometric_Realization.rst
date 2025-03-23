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

    <pzflow.flow.Flow at 0x7f8178f6fc70>



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
          <td>26.903050</td>
          <td>0.511313</td>
          <td>26.834274</td>
          <td>0.184239</td>
          <td>26.008485</td>
          <td>0.079439</td>
          <td>25.204104</td>
          <td>0.063628</td>
          <td>24.819963</td>
          <td>0.086611</td>
          <td>24.179584</td>
          <td>0.110599</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.535598</td>
          <td>0.793357</td>
          <td>27.512070</td>
          <td>0.321738</td>
          <td>26.456178</td>
          <td>0.117637</td>
          <td>26.372992</td>
          <td>0.176299</td>
          <td>25.481143</td>
          <td>0.153952</td>
          <td>25.932692</td>
          <td>0.467186</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.236086</td>
          <td>2.664805</td>
          <td>28.645761</td>
          <td>0.672543</td>
          <td>26.250648</td>
          <td>0.158849</td>
          <td>25.172817</td>
          <td>0.117961</td>
          <td>24.310943</td>
          <td>0.123991</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.602422</td>
          <td>0.828492</td>
          <td>27.671346</td>
          <td>0.364822</td>
          <td>27.181402</td>
          <td>0.218445</td>
          <td>26.088725</td>
          <td>0.138223</td>
          <td>25.542701</td>
          <td>0.162275</td>
          <td>25.359698</td>
          <td>0.299225</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.448416</td>
          <td>0.362172</td>
          <td>26.008369</td>
          <td>0.090263</td>
          <td>25.983695</td>
          <td>0.077720</td>
          <td>25.529510</td>
          <td>0.084840</td>
          <td>25.269637</td>
          <td>0.128304</td>
          <td>24.883408</td>
          <td>0.202228</td>
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
          <td>26.614074</td>
          <td>0.411697</td>
          <td>26.418648</td>
          <td>0.129097</td>
          <td>25.438698</td>
          <td>0.047940</td>
          <td>25.116360</td>
          <td>0.058863</td>
          <td>24.779207</td>
          <td>0.083556</td>
          <td>24.759406</td>
          <td>0.182159</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.650721</td>
          <td>0.423379</td>
          <td>26.698066</td>
          <td>0.164121</td>
          <td>26.032965</td>
          <td>0.081174</td>
          <td>25.196003</td>
          <td>0.063172</td>
          <td>24.844726</td>
          <td>0.088519</td>
          <td>24.486743</td>
          <td>0.144333</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.460115</td>
          <td>0.754904</td>
          <td>26.656981</td>
          <td>0.158466</td>
          <td>26.235161</td>
          <td>0.096978</td>
          <td>26.147927</td>
          <td>0.145454</td>
          <td>26.181843</td>
          <td>0.276635</td>
          <td>24.972963</td>
          <td>0.217954</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.255031</td>
          <td>0.657046</td>
          <td>26.278788</td>
          <td>0.114343</td>
          <td>26.181238</td>
          <td>0.092494</td>
          <td>25.796708</td>
          <td>0.107261</td>
          <td>25.524822</td>
          <td>0.159815</td>
          <td>26.294604</td>
          <td>0.607808</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.901679</td>
          <td>0.510799</td>
          <td>26.810142</td>
          <td>0.180517</td>
          <td>26.648902</td>
          <td>0.139012</td>
          <td>25.961323</td>
          <td>0.123795</td>
          <td>25.892957</td>
          <td>0.218089</td>
          <td>25.642388</td>
          <td>0.374287</td>
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
          <td>26.762832</td>
          <td>0.508800</td>
          <td>26.356620</td>
          <td>0.140832</td>
          <td>26.056439</td>
          <td>0.097432</td>
          <td>25.148828</td>
          <td>0.071820</td>
          <td>24.591799</td>
          <td>0.083303</td>
          <td>24.025624</td>
          <td>0.114145</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.792548</td>
          <td>0.453630</td>
          <td>27.019040</td>
          <td>0.222424</td>
          <td>26.237047</td>
          <td>0.184857</td>
          <td>26.293390</td>
          <td>0.350350</td>
          <td>25.126671</td>
          <td>0.289060</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.170432</td>
          <td>0.607525</td>
          <td>27.560545</td>
          <td>0.352005</td>
          <td>26.259378</td>
          <td>0.192554</td>
          <td>25.011393</td>
          <td>0.122959</td>
          <td>24.422664</td>
          <td>0.164411</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.830026</td>
          <td>1.796605</td>
          <td>29.779058</td>
          <td>1.618933</td>
          <td>27.595631</td>
          <td>0.376517</td>
          <td>26.493924</td>
          <td>0.244629</td>
          <td>25.429792</td>
          <td>0.183890</td>
          <td>25.121668</td>
          <td>0.306581</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.639382</td>
          <td>0.209630</td>
          <td>26.215391</td>
          <td>0.124698</td>
          <td>25.746060</td>
          <td>0.074158</td>
          <td>25.566126</td>
          <td>0.103723</td>
          <td>25.718480</td>
          <td>0.219828</td>
          <td>25.034830</td>
          <td>0.268330</td>
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
          <td>26.871500</td>
          <td>0.557969</td>
          <td>26.537677</td>
          <td>0.167509</td>
          <td>25.445413</td>
          <td>0.058024</td>
          <td>25.053843</td>
          <td>0.067495</td>
          <td>24.914223</td>
          <td>0.112850</td>
          <td>24.782145</td>
          <td>0.222296</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.916929</td>
          <td>0.570439</td>
          <td>26.722301</td>
          <td>0.192994</td>
          <td>26.029690</td>
          <td>0.095571</td>
          <td>25.147745</td>
          <td>0.072065</td>
          <td>24.689652</td>
          <td>0.091173</td>
          <td>24.336782</td>
          <td>0.150021</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.022787</td>
          <td>0.289736</td>
          <td>26.899426</td>
          <td>0.225462</td>
          <td>26.566683</td>
          <td>0.153620</td>
          <td>25.824362</td>
          <td>0.131520</td>
          <td>26.064239</td>
          <td>0.295256</td>
          <td>25.280270</td>
          <td>0.330759</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.396479</td>
          <td>0.394156</td>
          <td>26.121666</td>
          <td>0.118202</td>
          <td>26.126863</td>
          <td>0.106923</td>
          <td>25.909477</td>
          <td>0.144241</td>
          <td>25.649198</td>
          <td>0.213697</td>
          <td>25.284183</td>
          <td>0.337554</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.850227</td>
          <td>0.545705</td>
          <td>26.705902</td>
          <td>0.191316</td>
          <td>26.660176</td>
          <td>0.165953</td>
          <td>26.600232</td>
          <td>0.252645</td>
          <td>25.881421</td>
          <td>0.253827</td>
          <td>25.692208</td>
          <td>0.453750</td>
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
          <td>27.616948</td>
          <td>0.836321</td>
          <td>26.509100</td>
          <td>0.139597</td>
          <td>26.057325</td>
          <td>0.082948</td>
          <td>25.178555</td>
          <td>0.062211</td>
          <td>24.785984</td>
          <td>0.084068</td>
          <td>24.039540</td>
          <td>0.097862</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.911026</td>
          <td>0.514593</td>
          <td>27.157168</td>
          <td>0.241462</td>
          <td>26.838864</td>
          <td>0.163769</td>
          <td>26.237029</td>
          <td>0.157161</td>
          <td>25.839911</td>
          <td>0.208827</td>
          <td>25.667613</td>
          <td>0.382031</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.769790</td>
          <td>0.486782</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.685632</td>
          <td>0.354521</td>
          <td>26.428462</td>
          <td>0.200573</td>
          <td>24.999380</td>
          <td>0.109989</td>
          <td>24.404446</td>
          <td>0.146102</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.108471</td>
          <td>0.321634</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.803445</td>
          <td>0.440242</td>
          <td>26.266710</td>
          <td>0.201808</td>
          <td>25.250517</td>
          <td>0.157338</td>
          <td>25.107785</td>
          <td>0.302174</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.164502</td>
          <td>0.289289</td>
          <td>26.029444</td>
          <td>0.092062</td>
          <td>25.988643</td>
          <td>0.078172</td>
          <td>25.648737</td>
          <td>0.094362</td>
          <td>25.433763</td>
          <td>0.148025</td>
          <td>25.080684</td>
          <td>0.238663</td>
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
          <td>27.336014</td>
          <td>0.724993</td>
          <td>26.508520</td>
          <td>0.149258</td>
          <td>25.462336</td>
          <td>0.053022</td>
          <td>25.063315</td>
          <td>0.061038</td>
          <td>24.786429</td>
          <td>0.090989</td>
          <td>24.607517</td>
          <td>0.173262</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.564738</td>
          <td>3.181232</td>
          <td>27.049652</td>
          <td>0.223729</td>
          <td>26.162833</td>
          <td>0.092524</td>
          <td>25.228417</td>
          <td>0.066159</td>
          <td>24.857402</td>
          <td>0.091000</td>
          <td>24.177400</td>
          <td>0.112281</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.413243</td>
          <td>0.363317</td>
          <td>26.708805</td>
          <td>0.172628</td>
          <td>26.379501</td>
          <td>0.115509</td>
          <td>26.204495</td>
          <td>0.160509</td>
          <td>26.615441</td>
          <td>0.407314</td>
          <td>25.703100</td>
          <td>0.410149</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.125274</td>
          <td>0.301950</td>
          <td>26.224588</td>
          <td>0.120545</td>
          <td>26.192312</td>
          <td>0.104726</td>
          <td>25.935018</td>
          <td>0.136168</td>
          <td>25.673210</td>
          <td>0.202378</td>
          <td>25.748623</td>
          <td>0.450274</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.762819</td>
          <td>0.179189</td>
          <td>26.487316</td>
          <td>0.125604</td>
          <td>26.329739</td>
          <td>0.176773</td>
          <td>26.093103</td>
          <td>0.266788</td>
          <td>26.563223</td>
          <td>0.753880</td>
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
