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

    <pzflow.flow.Flow at 0x7f4ddfac6740>



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
          <td>26.926625</td>
          <td>0.520217</td>
          <td>26.680511</td>
          <td>0.161682</td>
          <td>26.049388</td>
          <td>0.082358</td>
          <td>25.382234</td>
          <td>0.074498</td>
          <td>25.349916</td>
          <td>0.137524</td>
          <td>25.014350</td>
          <td>0.225590</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.066009</td>
          <td>0.969355</td>
          <td>27.774035</td>
          <td>0.353219</td>
          <td>27.148004</td>
          <td>0.333595</td>
          <td>26.241142</td>
          <td>0.290245</td>
          <td>26.913804</td>
          <td>0.917010</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.572922</td>
          <td>0.398897</td>
          <td>25.882547</td>
          <td>0.080807</td>
          <td>24.777220</td>
          <td>0.026741</td>
          <td>23.887803</td>
          <td>0.020023</td>
          <td>23.149260</td>
          <td>0.019938</td>
          <td>22.816272</td>
          <td>0.033155</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.398621</td>
          <td>0.626042</td>
          <td>27.395633</td>
          <td>0.260720</td>
          <td>26.823961</td>
          <td>0.256889</td>
          <td>26.481098</td>
          <td>0.351428</td>
          <td>25.701888</td>
          <td>0.391966</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.678376</td>
          <td>0.193586</td>
          <td>25.777306</td>
          <td>0.073645</td>
          <td>25.514032</td>
          <td>0.051257</td>
          <td>24.742897</td>
          <td>0.042253</td>
          <td>24.369707</td>
          <td>0.058158</td>
          <td>23.584907</td>
          <td>0.065524</td>
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
          <td>27.040728</td>
          <td>0.565031</td>
          <td>26.360653</td>
          <td>0.122772</td>
          <td>26.069828</td>
          <td>0.083856</td>
          <td>26.258535</td>
          <td>0.159923</td>
          <td>25.754971</td>
          <td>0.194281</td>
          <td>26.245609</td>
          <td>0.587081</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.293513</td>
          <td>0.320509</td>
          <td>27.140161</td>
          <td>0.237914</td>
          <td>26.868252</td>
          <td>0.167770</td>
          <td>26.395363</td>
          <td>0.179675</td>
          <td>25.729032</td>
          <td>0.190080</td>
          <td>25.553089</td>
          <td>0.349014</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.740503</td>
          <td>0.453147</td>
          <td>27.050072</td>
          <td>0.220795</td>
          <td>26.928743</td>
          <td>0.176623</td>
          <td>26.596206</td>
          <td>0.212760</td>
          <td>25.690097</td>
          <td>0.183931</td>
          <td>25.214511</td>
          <td>0.266015</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.834228</td>
          <td>0.486002</td>
          <td>27.356263</td>
          <td>0.283904</td>
          <td>26.627454</td>
          <td>0.136463</td>
          <td>25.817099</td>
          <td>0.109188</td>
          <td>25.576613</td>
          <td>0.167037</td>
          <td>25.776653</td>
          <td>0.415157</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.478541</td>
          <td>0.370786</td>
          <td>26.603550</td>
          <td>0.151382</td>
          <td>26.018785</td>
          <td>0.080165</td>
          <td>25.572538</td>
          <td>0.088116</td>
          <td>25.080948</td>
          <td>0.108885</td>
          <td>24.851477</td>
          <td>0.196875</td>
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
          <td>28.249888</td>
          <td>1.312430</td>
          <td>26.742668</td>
          <td>0.195624</td>
          <td>26.094750</td>
          <td>0.100758</td>
          <td>25.351412</td>
          <td>0.085879</td>
          <td>24.934554</td>
          <td>0.112500</td>
          <td>25.377191</td>
          <td>0.352876</td>
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
          <td>27.719819</td>
          <td>0.390811</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.917959</td>
          <td>0.561162</td>
          <td>25.972900</td>
          <td>0.553342</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.404436</td>
          <td>0.804639</td>
          <td>25.962386</td>
          <td>0.102046</td>
          <td>24.816431</td>
          <td>0.033288</td>
          <td>23.852650</td>
          <td>0.023450</td>
          <td>23.153371</td>
          <td>0.023967</td>
          <td>22.828309</td>
          <td>0.040599</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.881886</td>
          <td>0.510322</td>
          <td>27.075750</td>
          <td>0.248261</td>
          <td>26.588994</td>
          <td>0.264465</td>
          <td>28.017138</td>
          <td>1.191754</td>
          <td>25.214182</td>
          <td>0.330059</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.235294</td>
          <td>0.340317</td>
          <td>25.696674</td>
          <td>0.079229</td>
          <td>25.424833</td>
          <td>0.055794</td>
          <td>24.776909</td>
          <td>0.051670</td>
          <td>24.258322</td>
          <td>0.062055</td>
          <td>23.637168</td>
          <td>0.081226</td>
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
          <td>25.834378</td>
          <td>0.250025</td>
          <td>26.295461</td>
          <td>0.136115</td>
          <td>26.054109</td>
          <td>0.099294</td>
          <td>26.003072</td>
          <td>0.154671</td>
          <td>26.326152</td>
          <td>0.366186</td>
          <td>25.240264</td>
          <td>0.322861</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.867932</td>
          <td>0.218024</td>
          <td>26.725045</td>
          <td>0.174375</td>
          <td>26.515875</td>
          <td>0.234340</td>
          <td>26.595858</td>
          <td>0.443967</td>
          <td>25.689941</td>
          <td>0.450644</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.177392</td>
          <td>0.687906</td>
          <td>27.037334</td>
          <td>0.252640</td>
          <td>26.910089</td>
          <td>0.205530</td>
          <td>26.818094</td>
          <td>0.302340</td>
          <td>25.827576</td>
          <td>0.243460</td>
          <td>25.406039</td>
          <td>0.365195</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.249256</td>
          <td>1.331214</td>
          <td>26.783694</td>
          <td>0.208037</td>
          <td>26.672390</td>
          <td>0.171198</td>
          <td>25.864748</td>
          <td>0.138791</td>
          <td>25.806740</td>
          <td>0.243529</td>
          <td>25.588938</td>
          <td>0.427646</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.659978</td>
          <td>0.474564</td>
          <td>26.328187</td>
          <td>0.138652</td>
          <td>26.110609</td>
          <td>0.103194</td>
          <td>25.570184</td>
          <td>0.105139</td>
          <td>25.265508</td>
          <td>0.151271</td>
          <td>24.822905</td>
          <td>0.227518</td>
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
          <td>27.168381</td>
          <td>0.618637</td>
          <td>26.726087</td>
          <td>0.168104</td>
          <td>25.959138</td>
          <td>0.076062</td>
          <td>25.508031</td>
          <td>0.083261</td>
          <td>24.988451</td>
          <td>0.100437</td>
          <td>25.262681</td>
          <td>0.276690</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.589311</td>
          <td>0.821913</td>
          <td>30.491482</td>
          <td>2.009247</td>
          <td>27.906628</td>
          <td>0.391997</td>
          <td>27.619279</td>
          <td>0.479764</td>
          <td>26.851893</td>
          <td>0.467546</td>
          <td>25.486161</td>
          <td>0.331329</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.893956</td>
          <td>0.244804</td>
          <td>26.003487</td>
          <td>0.096585</td>
          <td>24.815392</td>
          <td>0.030012</td>
          <td>23.825773</td>
          <td>0.020648</td>
          <td>23.134902</td>
          <td>0.021334</td>
          <td>22.902173</td>
          <td>0.038967</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.794778</td>
          <td>1.048472</td>
          <td>29.606148</td>
          <td>1.484675</td>
          <td>27.297365</td>
          <td>0.296355</td>
          <td>26.969112</td>
          <td>0.357381</td>
          <td>26.143223</td>
          <td>0.329370</td>
          <td>25.826317</td>
          <td>0.524928</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.043363</td>
          <td>0.262206</td>
          <td>25.825466</td>
          <td>0.076937</td>
          <td>25.457985</td>
          <td>0.048839</td>
          <td>24.822139</td>
          <td>0.045400</td>
          <td>24.389617</td>
          <td>0.059280</td>
          <td>23.572400</td>
          <td>0.064898</td>
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
          <td>26.352087</td>
          <td>0.353001</td>
          <td>26.104036</td>
          <td>0.105150</td>
          <td>26.017527</td>
          <td>0.086664</td>
          <td>26.010277</td>
          <td>0.140059</td>
          <td>25.957884</td>
          <td>0.247830</td>
          <td>25.034841</td>
          <td>0.247739</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.232457</td>
          <td>0.308469</td>
          <td>27.012925</td>
          <td>0.216996</td>
          <td>26.923086</td>
          <td>0.178595</td>
          <td>26.505499</td>
          <td>0.200470</td>
          <td>25.753860</td>
          <td>0.197183</td>
          <td>26.274285</td>
          <td>0.607438</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.084778</td>
          <td>0.236587</td>
          <td>27.139355</td>
          <td>0.220935</td>
          <td>26.718633</td>
          <td>0.247156</td>
          <td>26.519174</td>
          <td>0.378124</td>
          <td>25.091866</td>
          <td>0.252248</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.656304</td>
          <td>0.909564</td>
          <td>27.808887</td>
          <td>0.442919</td>
          <td>26.586809</td>
          <td>0.147446</td>
          <td>25.908737</td>
          <td>0.133112</td>
          <td>25.594444</td>
          <td>0.189402</td>
          <td>25.367029</td>
          <td>0.335221</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.994196</td>
          <td>0.558770</td>
          <td>26.432222</td>
          <td>0.135054</td>
          <td>26.020820</td>
          <td>0.083512</td>
          <td>25.622320</td>
          <td>0.095894</td>
          <td>25.249039</td>
          <td>0.130958</td>
          <td>24.891288</td>
          <td>0.211553</td>
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
