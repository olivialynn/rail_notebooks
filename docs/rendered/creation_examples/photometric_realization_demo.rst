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

    <pzflow.flow.Flow at 0x7f685451c340>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.722835</td>
          <td>0.167621</td>
          <td>26.132107</td>
          <td>0.088584</td>
          <td>25.446919</td>
          <td>0.078879</td>
          <td>25.107175</td>
          <td>0.111406</td>
          <td>24.991436</td>
          <td>0.221333</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.471171</td>
          <td>1.374800</td>
          <td>29.147162</td>
          <td>1.017995</td>
          <td>27.604422</td>
          <td>0.308740</td>
          <td>27.728600</td>
          <td>0.519640</td>
          <td>26.240456</td>
          <td>0.290085</td>
          <td>26.595894</td>
          <td>0.747268</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>29.337926</td>
          <td>2.058548</td>
          <td>25.854039</td>
          <td>0.078802</td>
          <td>24.771691</td>
          <td>0.026612</td>
          <td>23.883879</td>
          <td>0.019956</td>
          <td>23.129340</td>
          <td>0.019605</td>
          <td>22.851163</td>
          <td>0.034191</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.014809</td>
          <td>0.474306</td>
          <td>27.551611</td>
          <td>0.295917</td>
          <td>26.574449</td>
          <td>0.208925</td>
          <td>25.718783</td>
          <td>0.188443</td>
          <td>25.290118</td>
          <td>0.282881</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.477859</td>
          <td>0.370590</td>
          <td>25.775409</td>
          <td>0.073521</td>
          <td>25.324050</td>
          <td>0.043302</td>
          <td>24.810560</td>
          <td>0.044868</td>
          <td>24.431076</td>
          <td>0.061412</td>
          <td>23.678724</td>
          <td>0.071199</td>
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
          <td>26.180698</td>
          <td>0.292826</td>
          <td>26.396489</td>
          <td>0.126645</td>
          <td>26.122350</td>
          <td>0.087826</td>
          <td>26.049359</td>
          <td>0.133603</td>
          <td>25.847230</td>
          <td>0.209921</td>
          <td>25.672992</td>
          <td>0.383295</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.520375</td>
          <td>2.215463</td>
          <td>27.300320</td>
          <td>0.271301</td>
          <td>26.943753</td>
          <td>0.178886</td>
          <td>26.013289</td>
          <td>0.129499</td>
          <td>25.979658</td>
          <td>0.234370</td>
          <td>25.566644</td>
          <td>0.352755</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.405281</td>
          <td>0.727793</td>
          <td>26.754129</td>
          <td>0.172142</td>
          <td>27.052290</td>
          <td>0.196060</td>
          <td>26.365573</td>
          <td>0.175192</td>
          <td>25.772786</td>
          <td>0.197216</td>
          <td>25.379377</td>
          <td>0.303994</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.703016</td>
          <td>0.373946</td>
          <td>26.649980</td>
          <td>0.139141</td>
          <td>26.038131</td>
          <td>0.132313</td>
          <td>25.710665</td>
          <td>0.187156</td>
          <td>25.901182</td>
          <td>0.456276</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.800942</td>
          <td>0.474123</td>
          <td>26.347139</td>
          <td>0.121341</td>
          <td>26.052696</td>
          <td>0.082599</td>
          <td>25.591552</td>
          <td>0.089603</td>
          <td>25.157287</td>
          <td>0.116378</td>
          <td>24.626206</td>
          <td>0.162658</td>
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
          <td>27.126977</td>
          <td>0.659504</td>
          <td>26.373859</td>
          <td>0.142936</td>
          <td>25.985328</td>
          <td>0.091537</td>
          <td>25.272693</td>
          <td>0.080124</td>
          <td>24.947661</td>
          <td>0.113792</td>
          <td>25.462767</td>
          <td>0.377287</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.696539</td>
          <td>0.852095</td>
          <td>27.585254</td>
          <td>0.351888</td>
          <td>28.003480</td>
          <td>0.721333</td>
          <td>26.007376</td>
          <td>0.278747</td>
          <td>27.868073</td>
          <td>1.704639</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.601367</td>
          <td>0.457870</td>
          <td>26.013072</td>
          <td>0.106664</td>
          <td>24.773367</td>
          <td>0.032050</td>
          <td>23.886740</td>
          <td>0.024149</td>
          <td>23.135271</td>
          <td>0.023596</td>
          <td>22.759202</td>
          <td>0.038191</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.376169</td>
          <td>0.810541</td>
          <td>28.287680</td>
          <td>0.680629</td>
          <td>27.830823</td>
          <td>0.450818</td>
          <td>26.294427</td>
          <td>0.207278</td>
          <td>26.009836</td>
          <td>0.297018</td>
          <td>25.769317</td>
          <td>0.504983</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.029460</td>
          <td>0.288770</td>
          <td>25.946538</td>
          <td>0.098667</td>
          <td>25.554818</td>
          <td>0.062610</td>
          <td>24.790370</td>
          <td>0.052290</td>
          <td>24.238531</td>
          <td>0.060976</td>
          <td>23.621956</td>
          <td>0.080144</td>
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
          <td>26.413855</td>
          <td>0.396673</td>
          <td>26.210949</td>
          <td>0.126530</td>
          <td>26.013991</td>
          <td>0.095863</td>
          <td>25.890479</td>
          <td>0.140411</td>
          <td>25.849864</td>
          <td>0.249878</td>
          <td>25.242758</td>
          <td>0.323503</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.058587</td>
          <td>1.184288</td>
          <td>27.237841</td>
          <td>0.295237</td>
          <td>26.783008</td>
          <td>0.183156</td>
          <td>26.584905</td>
          <td>0.248072</td>
          <td>26.156970</td>
          <td>0.315567</td>
          <td>25.900951</td>
          <td>0.526974</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.999580</td>
          <td>0.244922</td>
          <td>26.479573</td>
          <td>0.142547</td>
          <td>26.527682</td>
          <td>0.238631</td>
          <td>25.937525</td>
          <td>0.266428</td>
          <td>25.609593</td>
          <td>0.427289</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.599568</td>
          <td>0.915570</td>
          <td>26.971463</td>
          <td>0.243129</td>
          <td>27.026644</td>
          <td>0.230531</td>
          <td>25.735335</td>
          <td>0.124095</td>
          <td>25.676903</td>
          <td>0.218691</td>
          <td>25.407486</td>
          <td>0.371869</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.317739</td>
          <td>0.365469</td>
          <td>26.368276</td>
          <td>0.143519</td>
          <td>26.063267</td>
          <td>0.099005</td>
          <td>25.480967</td>
          <td>0.097240</td>
          <td>25.134216</td>
          <td>0.135110</td>
          <td>24.627187</td>
          <td>0.193172</td>
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
          <td>28.050286</td>
          <td>1.090097</td>
          <td>26.731465</td>
          <td>0.168875</td>
          <td>26.057462</td>
          <td>0.082958</td>
          <td>25.322215</td>
          <td>0.070656</td>
          <td>24.953519</td>
          <td>0.097409</td>
          <td>25.154253</td>
          <td>0.253247</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.260697</td>
          <td>0.568154</td>
          <td>27.668325</td>
          <td>0.325176</td>
          <td>27.147249</td>
          <td>0.333695</td>
          <td>26.486108</td>
          <td>0.353113</td>
          <td>25.476397</td>
          <td>0.328771</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.713447</td>
          <td>0.466787</td>
          <td>25.917193</td>
          <td>0.089544</td>
          <td>24.773248</td>
          <td>0.028923</td>
          <td>23.878782</td>
          <td>0.021603</td>
          <td>23.141676</td>
          <td>0.021458</td>
          <td>22.861321</td>
          <td>0.037583</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.088062</td>
          <td>0.667194</td>
          <td>27.516109</td>
          <td>0.386118</td>
          <td>27.151988</td>
          <td>0.263389</td>
          <td>26.391179</td>
          <td>0.223915</td>
          <td>26.253712</td>
          <td>0.359359</td>
          <td>25.660478</td>
          <td>0.464336</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.473015</td>
          <td>0.369516</td>
          <td>25.587745</td>
          <td>0.062360</td>
          <td>25.426816</td>
          <td>0.047506</td>
          <td>24.788000</td>
          <td>0.044045</td>
          <td>24.508151</td>
          <td>0.065849</td>
          <td>23.710181</td>
          <td>0.073317</td>
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
          <td>26.208543</td>
          <td>0.315098</td>
          <td>26.566341</td>
          <td>0.156835</td>
          <td>26.300105</td>
          <td>0.111022</td>
          <td>26.166426</td>
          <td>0.160145</td>
          <td>25.712335</td>
          <td>0.202084</td>
          <td>26.636042</td>
          <td>0.815409</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.625227</td>
          <td>0.419386</td>
          <td>27.533743</td>
          <td>0.331592</td>
          <td>26.856524</td>
          <td>0.168776</td>
          <td>26.155546</td>
          <td>0.148906</td>
          <td>25.777312</td>
          <td>0.201107</td>
          <td>25.446685</td>
          <td>0.325815</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.936922</td>
          <td>0.209226</td>
          <td>26.628255</td>
          <td>0.143272</td>
          <td>27.058914</td>
          <td>0.325534</td>
          <td>25.718075</td>
          <td>0.197364</td>
          <td>25.349462</td>
          <td>0.310843</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.806610</td>
          <td>1.698814</td>
          <td>27.413983</td>
          <td>0.326035</td>
          <td>26.977351</td>
          <td>0.205418</td>
          <td>26.148473</td>
          <td>0.163550</td>
          <td>25.830854</td>
          <td>0.230810</td>
          <td>25.392218</td>
          <td>0.341964</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.685493</td>
          <td>0.167803</td>
          <td>26.004973</td>
          <td>0.082353</td>
          <td>25.619310</td>
          <td>0.095641</td>
          <td>25.370286</td>
          <td>0.145396</td>
          <td>25.393206</td>
          <td>0.318884</td>
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
