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

    <pzflow.flow.Flow at 0x7fd9a02d3280>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>27.642783</td>
          <td>0.850210</td>
          <td>26.603302</td>
          <td>0.151350</td>
          <td>26.026135</td>
          <td>0.080686</td>
          <td>25.334050</td>
          <td>0.071390</td>
          <td>24.635761</td>
          <td>0.073618</td>
          <td>24.014814</td>
          <td>0.095749</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.650560</td>
          <td>0.358936</td>
          <td>26.653337</td>
          <td>0.139545</td>
          <td>26.214629</td>
          <td>0.154026</td>
          <td>25.918533</td>
          <td>0.222783</td>
          <td>25.242529</td>
          <td>0.272159</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.669569</td>
          <td>0.429492</td>
          <td>29.228243</td>
          <td>1.068035</td>
          <td>28.001874</td>
          <td>0.421391</td>
          <td>25.783551</td>
          <td>0.106035</td>
          <td>25.046899</td>
          <td>0.105694</td>
          <td>24.297392</td>
          <td>0.122541</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.729913</td>
          <td>0.341152</td>
          <td>26.393944</td>
          <td>0.179459</td>
          <td>25.609743</td>
          <td>0.171814</td>
          <td>25.138054</td>
          <td>0.249868</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.312903</td>
          <td>0.325488</td>
          <td>26.012587</td>
          <td>0.090598</td>
          <td>25.917414</td>
          <td>0.073298</td>
          <td>25.501230</td>
          <td>0.082751</td>
          <td>25.396838</td>
          <td>0.143200</td>
          <td>25.145004</td>
          <td>0.251299</td>
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
          <td>27.652394</td>
          <td>0.855437</td>
          <td>26.212059</td>
          <td>0.107884</td>
          <td>25.387751</td>
          <td>0.045820</td>
          <td>25.142544</td>
          <td>0.060247</td>
          <td>24.857847</td>
          <td>0.089547</td>
          <td>25.304200</td>
          <td>0.286124</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.592854</td>
          <td>1.463517</td>
          <td>26.887489</td>
          <td>0.192699</td>
          <td>26.129297</td>
          <td>0.088365</td>
          <td>25.213236</td>
          <td>0.064145</td>
          <td>24.926539</td>
          <td>0.095118</td>
          <td>24.225449</td>
          <td>0.115110</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.000602</td>
          <td>0.252964</td>
          <td>26.920233</td>
          <td>0.198081</td>
          <td>26.262044</td>
          <td>0.099291</td>
          <td>26.457022</td>
          <td>0.189293</td>
          <td>25.767369</td>
          <td>0.196319</td>
          <td>25.924312</td>
          <td>0.464264</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.064703</td>
          <td>0.574812</td>
          <td>26.019432</td>
          <td>0.091144</td>
          <td>26.092868</td>
          <td>0.085576</td>
          <td>25.955441</td>
          <td>0.123164</td>
          <td>25.916281</td>
          <td>0.222366</td>
          <td>24.871614</td>
          <td>0.200235</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.204880</td>
          <td>0.298578</td>
          <td>26.829277</td>
          <td>0.183463</td>
          <td>26.384142</td>
          <td>0.110480</td>
          <td>26.420290</td>
          <td>0.183508</td>
          <td>25.744316</td>
          <td>0.192546</td>
          <td>25.533285</td>
          <td>0.343610</td>
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
          <td>1.398945</td>
          <td>27.593034</td>
          <td>0.896419</td>
          <td>26.465596</td>
          <td>0.154638</td>
          <td>26.098852</td>
          <td>0.101121</td>
          <td>25.152541</td>
          <td>0.072056</td>
          <td>24.696740</td>
          <td>0.091360</td>
          <td>24.104569</td>
          <td>0.122255</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.614223</td>
          <td>1.580249</td>
          <td>27.368864</td>
          <td>0.326794</td>
          <td>26.886991</td>
          <td>0.199178</td>
          <td>26.393512</td>
          <td>0.210845</td>
          <td>25.906904</td>
          <td>0.256821</td>
          <td>24.938036</td>
          <td>0.247856</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.558807</td>
          <td>2.383604</td>
          <td>28.168015</td>
          <td>0.606490</td>
          <td>28.608517</td>
          <td>0.755238</td>
          <td>26.322861</td>
          <td>0.203109</td>
          <td>24.792104</td>
          <td>0.101566</td>
          <td>24.276117</td>
          <td>0.145020</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.156647</td>
          <td>0.621590</td>
          <td>27.667884</td>
          <td>0.398177</td>
          <td>26.446209</td>
          <td>0.235183</td>
          <td>25.457143</td>
          <td>0.188188</td>
          <td>25.171018</td>
          <td>0.318919</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.320884</td>
          <td>0.363968</td>
          <td>25.840232</td>
          <td>0.089889</td>
          <td>25.960439</td>
          <td>0.089586</td>
          <td>25.794301</td>
          <td>0.126522</td>
          <td>25.219107</td>
          <td>0.143988</td>
          <td>25.376215</td>
          <td>0.352710</td>
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
          <td>27.801760</td>
          <td>1.029677</td>
          <td>26.193188</td>
          <td>0.124599</td>
          <td>25.498708</td>
          <td>0.060832</td>
          <td>24.981599</td>
          <td>0.063311</td>
          <td>24.847359</td>
          <td>0.106455</td>
          <td>24.797083</td>
          <td>0.225073</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.252444</td>
          <td>0.720119</td>
          <td>26.389519</td>
          <td>0.145407</td>
          <td>26.089250</td>
          <td>0.100692</td>
          <td>25.196496</td>
          <td>0.075238</td>
          <td>24.623883</td>
          <td>0.086049</td>
          <td>24.175312</td>
          <td>0.130535</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.116225</td>
          <td>0.659653</td>
          <td>26.855921</td>
          <td>0.217452</td>
          <td>26.625226</td>
          <td>0.161509</td>
          <td>26.156173</td>
          <td>0.174796</td>
          <td>25.942463</td>
          <td>0.267503</td>
          <td>26.333128</td>
          <td>0.718802</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.292328</td>
          <td>0.363550</td>
          <td>26.195383</td>
          <td>0.126005</td>
          <td>26.046547</td>
          <td>0.099669</td>
          <td>25.802285</td>
          <td>0.131503</td>
          <td>25.725689</td>
          <td>0.227743</td>
          <td>25.023804</td>
          <td>0.273915</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.438245</td>
          <td>0.401229</td>
          <td>26.848822</td>
          <td>0.215660</td>
          <td>26.338429</td>
          <td>0.125843</td>
          <td>26.229616</td>
          <td>0.185506</td>
          <td>25.997794</td>
          <td>0.279107</td>
          <td>25.151034</td>
          <td>0.297540</td>
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
          <td>1.398945</td>
          <td>27.249808</td>
          <td>0.654727</td>
          <td>26.691510</td>
          <td>0.163224</td>
          <td>25.939949</td>
          <td>0.074783</td>
          <td>25.140607</td>
          <td>0.060152</td>
          <td>24.644789</td>
          <td>0.074217</td>
          <td>23.835834</td>
          <td>0.081811</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.242422</td>
          <td>0.651681</td>
          <td>27.773712</td>
          <td>0.395291</td>
          <td>26.432480</td>
          <td>0.115343</td>
          <td>26.082478</td>
          <td>0.137614</td>
          <td>25.982044</td>
          <td>0.235043</td>
          <td>25.408368</td>
          <td>0.311418</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.024816</td>
          <td>0.585855</td>
          <td>31.189073</td>
          <td>2.696616</td>
          <td>28.959677</td>
          <td>0.879443</td>
          <td>25.827507</td>
          <td>0.119939</td>
          <td>25.261030</td>
          <td>0.138023</td>
          <td>24.370833</td>
          <td>0.141938</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.643107</td>
          <td>0.389410</td>
          <td>26.376807</td>
          <td>0.221255</td>
          <td>25.492192</td>
          <td>0.193169</td>
          <td>25.602295</td>
          <td>0.444453</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.166471</td>
          <td>0.289749</td>
          <td>26.179181</td>
          <td>0.104960</td>
          <td>26.042466</td>
          <td>0.081974</td>
          <td>25.588645</td>
          <td>0.089508</td>
          <td>25.361737</td>
          <td>0.139127</td>
          <td>24.943066</td>
          <td>0.212883</td>
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
          <td>27.055674</td>
          <td>0.597527</td>
          <td>26.420347</td>
          <td>0.138362</td>
          <td>25.534163</td>
          <td>0.056512</td>
          <td>25.109352</td>
          <td>0.063581</td>
          <td>24.903046</td>
          <td>0.100792</td>
          <td>24.526609</td>
          <td>0.161723</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.454902</td>
          <td>0.367744</td>
          <td>26.811995</td>
          <td>0.183311</td>
          <td>25.804750</td>
          <td>0.067453</td>
          <td>25.239033</td>
          <td>0.066785</td>
          <td>24.825325</td>
          <td>0.088468</td>
          <td>24.328101</td>
          <td>0.127993</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.737215</td>
          <td>0.465517</td>
          <td>26.887795</td>
          <td>0.200792</td>
          <td>26.277259</td>
          <td>0.105651</td>
          <td>26.624474</td>
          <td>0.228657</td>
          <td>25.727079</td>
          <td>0.198863</td>
          <td>25.246913</td>
          <td>0.286224</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.483985</td>
          <td>0.815237</td>
          <td>26.277535</td>
          <td>0.126207</td>
          <td>26.022712</td>
          <td>0.090255</td>
          <td>25.879715</td>
          <td>0.129812</td>
          <td>25.374153</td>
          <td>0.157066</td>
          <td>24.985870</td>
          <td>0.246366</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.364197</td>
          <td>0.347378</td>
          <td>26.776228</td>
          <td>0.181235</td>
          <td>26.591011</td>
          <td>0.137391</td>
          <td>26.050308</td>
          <td>0.139182</td>
          <td>25.691325</td>
          <td>0.191129</td>
          <td>25.522393</td>
          <td>0.353222</td>
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
