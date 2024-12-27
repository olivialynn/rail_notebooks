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

    <pzflow.flow.Flow at 0x7faee09d55d0>



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
          <td>26.866468</td>
          <td>0.497733</td>
          <td>26.786299</td>
          <td>0.176907</td>
          <td>26.008688</td>
          <td>0.079454</td>
          <td>25.302022</td>
          <td>0.069394</td>
          <td>25.124152</td>
          <td>0.113067</td>
          <td>24.880564</td>
          <td>0.201746</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.797722</td>
          <td>0.402379</td>
          <td>27.342397</td>
          <td>0.249583</td>
          <td>28.270198</td>
          <td>0.758370</td>
          <td>27.051766</td>
          <td>0.541286</td>
          <td>26.854349</td>
          <td>0.883510</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.825669</td>
          <td>0.482925</td>
          <td>25.929327</td>
          <td>0.084204</td>
          <td>24.794191</td>
          <td>0.027140</td>
          <td>23.909615</td>
          <td>0.020397</td>
          <td>23.143908</td>
          <td>0.019848</td>
          <td>22.827537</td>
          <td>0.033486</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.245414</td>
          <td>0.259414</td>
          <td>27.087941</td>
          <td>0.202022</td>
          <td>26.774167</td>
          <td>0.246596</td>
          <td>25.795068</td>
          <td>0.200943</td>
          <td>25.743123</td>
          <td>0.404620</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.138664</td>
          <td>0.283058</td>
          <td>25.695345</td>
          <td>0.068501</td>
          <td>25.409585</td>
          <td>0.046717</td>
          <td>24.796262</td>
          <td>0.044302</td>
          <td>24.387635</td>
          <td>0.059091</td>
          <td>23.700106</td>
          <td>0.072559</td>
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
          <td>26.252246</td>
          <td>0.310130</td>
          <td>26.173884</td>
          <td>0.104347</td>
          <td>26.150078</td>
          <td>0.089995</td>
          <td>26.139724</td>
          <td>0.144431</td>
          <td>25.795357</td>
          <td>0.200992</td>
          <td>25.049480</td>
          <td>0.232260</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.136766</td>
          <td>0.237248</td>
          <td>26.900278</td>
          <td>0.172405</td>
          <td>26.574904</td>
          <td>0.209005</td>
          <td>26.283252</td>
          <td>0.300263</td>
          <td>25.565141</td>
          <td>0.352339</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.483038</td>
          <td>2.183054</td>
          <td>27.395736</td>
          <td>0.293103</td>
          <td>27.073162</td>
          <td>0.199531</td>
          <td>26.477830</td>
          <td>0.192644</td>
          <td>26.150810</td>
          <td>0.269739</td>
          <td>25.414837</td>
          <td>0.312754</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.373692</td>
          <td>0.287934</td>
          <td>26.511141</td>
          <td>0.123392</td>
          <td>26.006468</td>
          <td>0.128736</td>
          <td>25.747205</td>
          <td>0.193015</td>
          <td>25.092180</td>
          <td>0.240605</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.775602</td>
          <td>0.465235</td>
          <td>26.533157</td>
          <td>0.142502</td>
          <td>26.163608</td>
          <td>0.091072</td>
          <td>25.606452</td>
          <td>0.090785</td>
          <td>25.272491</td>
          <td>0.128621</td>
          <td>24.905776</td>
          <td>0.206056</td>
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
          <td>29.155656</td>
          <td>2.017138</td>
          <td>26.716797</td>
          <td>0.191410</td>
          <td>26.064894</td>
          <td>0.098157</td>
          <td>25.420904</td>
          <td>0.091292</td>
          <td>25.014447</td>
          <td>0.120598</td>
          <td>24.819881</td>
          <td>0.224747</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.547708</td>
          <td>2.356716</td>
          <td>27.507109</td>
          <td>0.364409</td>
          <td>27.475262</td>
          <td>0.322560</td>
          <td>27.723900</td>
          <td>0.594604</td>
          <td>26.280196</td>
          <td>0.346730</td>
          <td>26.006500</td>
          <td>0.566878</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.693242</td>
          <td>0.490319</td>
          <td>25.918049</td>
          <td>0.098165</td>
          <td>24.752412</td>
          <td>0.031465</td>
          <td>23.809095</td>
          <td>0.022588</td>
          <td>23.135579</td>
          <td>0.023602</td>
          <td>22.811702</td>
          <td>0.040007</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.568140</td>
          <td>0.403083</td>
          <td>27.332670</td>
          <td>0.305880</td>
          <td>26.561249</td>
          <td>0.258533</td>
          <td>26.092467</td>
          <td>0.317354</td>
          <td>25.852954</td>
          <td>0.536833</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.918369</td>
          <td>0.263891</td>
          <td>25.969180</td>
          <td>0.100640</td>
          <td>25.509049</td>
          <td>0.060121</td>
          <td>24.799287</td>
          <td>0.052706</td>
          <td>24.410580</td>
          <td>0.071010</td>
          <td>23.773043</td>
          <td>0.091544</td>
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
          <td>26.527547</td>
          <td>0.432685</td>
          <td>26.350159</td>
          <td>0.142679</td>
          <td>25.997044</td>
          <td>0.094448</td>
          <td>25.761126</td>
          <td>0.125559</td>
          <td>25.759781</td>
          <td>0.231982</td>
          <td>25.117701</td>
          <td>0.292659</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>31.708699</td>
          <td>4.408384</td>
          <td>26.573881</td>
          <td>0.170221</td>
          <td>26.871983</td>
          <td>0.197429</td>
          <td>26.140582</td>
          <td>0.171014</td>
          <td>25.810804</td>
          <td>0.238182</td>
          <td>26.785511</td>
          <td>0.955360</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.334377</td>
          <td>0.764350</td>
          <td>27.429580</td>
          <td>0.346395</td>
          <td>26.731644</td>
          <td>0.176820</td>
          <td>26.453971</td>
          <td>0.224496</td>
          <td>25.974588</td>
          <td>0.274592</td>
          <td>25.573893</td>
          <td>0.415809</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.259276</td>
          <td>0.307205</td>
          <td>26.713983</td>
          <td>0.177353</td>
          <td>25.787951</td>
          <td>0.129883</td>
          <td>25.805744</td>
          <td>0.243329</td>
          <td>25.691813</td>
          <td>0.462199</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.567230</td>
          <td>0.442668</td>
          <td>26.589904</td>
          <td>0.173440</td>
          <td>25.976447</td>
          <td>0.091743</td>
          <td>25.722859</td>
          <td>0.120104</td>
          <td>25.122293</td>
          <td>0.133726</td>
          <td>24.762543</td>
          <td>0.216377</td>
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
          <td>27.562407</td>
          <td>0.807384</td>
          <td>26.546590</td>
          <td>0.144174</td>
          <td>26.134454</td>
          <td>0.088778</td>
          <td>25.328729</td>
          <td>0.071064</td>
          <td>25.121196</td>
          <td>0.112791</td>
          <td>25.059885</td>
          <td>0.234299</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.388124</td>
          <td>0.719813</td>
          <td>28.214147</td>
          <td>0.549438</td>
          <td>28.237983</td>
          <td>0.503440</td>
          <td>26.663716</td>
          <td>0.225281</td>
          <td>26.224613</td>
          <td>0.286643</td>
          <td>26.195203</td>
          <td>0.566773</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.341313</td>
          <td>0.350867</td>
          <td>25.912063</td>
          <td>0.089142</td>
          <td>24.786736</td>
          <td>0.029267</td>
          <td>23.857435</td>
          <td>0.021213</td>
          <td>23.135802</td>
          <td>0.021351</td>
          <td>22.854626</td>
          <td>0.037362</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.578299</td>
          <td>0.462459</td>
          <td>28.059048</td>
          <td>0.578613</td>
          <td>26.983289</td>
          <td>0.229241</td>
          <td>26.498134</td>
          <td>0.244633</td>
          <td>26.210357</td>
          <td>0.347326</td>
          <td>25.153821</td>
          <td>0.313528</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.594488</td>
          <td>0.405912</td>
          <td>25.803894</td>
          <td>0.075487</td>
          <td>25.448969</td>
          <td>0.048449</td>
          <td>24.769440</td>
          <td>0.043325</td>
          <td>24.427259</td>
          <td>0.061292</td>
          <td>23.712769</td>
          <td>0.073485</td>
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
          <td>26.467363</td>
          <td>0.386181</td>
          <td>26.311263</td>
          <td>0.125918</td>
          <td>26.107043</td>
          <td>0.093761</td>
          <td>26.171649</td>
          <td>0.160861</td>
          <td>26.413264</td>
          <td>0.357440</td>
          <td>25.131790</td>
          <td>0.268211</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.241981</td>
          <td>0.262156</td>
          <td>27.229855</td>
          <td>0.230990</td>
          <td>26.463716</td>
          <td>0.193547</td>
          <td>26.057274</td>
          <td>0.253733</td>
          <td>25.509699</td>
          <td>0.342493</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.215166</td>
          <td>1.223631</td>
          <td>27.051155</td>
          <td>0.230096</td>
          <td>26.717259</td>
          <td>0.154651</td>
          <td>26.557350</td>
          <td>0.216241</td>
          <td>25.728269</td>
          <td>0.199062</td>
          <td>25.260779</td>
          <td>0.289450</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.255300</td>
          <td>0.700575</td>
          <td>27.688117</td>
          <td>0.403971</td>
          <td>26.742169</td>
          <td>0.168401</td>
          <td>25.680499</td>
          <td>0.109170</td>
          <td>25.760709</td>
          <td>0.217741</td>
          <td>25.157815</td>
          <td>0.283501</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.548270</td>
          <td>0.816208</td>
          <td>26.404223</td>
          <td>0.131828</td>
          <td>25.947543</td>
          <td>0.078284</td>
          <td>25.720251</td>
          <td>0.104485</td>
          <td>25.226372</td>
          <td>0.128414</td>
          <td>25.022200</td>
          <td>0.235876</td>
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
