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

    <pzflow.flow.Flow at 0x7f1b47ec6bf0>



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
          <td>27.421123</td>
          <td>0.735555</td>
          <td>26.655255</td>
          <td>0.158232</td>
          <td>26.153837</td>
          <td>0.090293</td>
          <td>25.177211</td>
          <td>0.062128</td>
          <td>24.765931</td>
          <td>0.082584</td>
          <td>23.938425</td>
          <td>0.089534</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.694167</td>
          <td>2.368166</td>
          <td>28.245715</td>
          <td>0.561701</td>
          <td>26.514528</td>
          <td>0.123755</td>
          <td>26.287312</td>
          <td>0.163902</td>
          <td>25.692081</td>
          <td>0.184240</td>
          <td>25.086422</td>
          <td>0.239464</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.350514</td>
          <td>0.701405</td>
          <td>27.810354</td>
          <td>0.406304</td>
          <td>28.277460</td>
          <td>0.517841</td>
          <td>25.986703</td>
          <td>0.126550</td>
          <td>25.281661</td>
          <td>0.129646</td>
          <td>24.214496</td>
          <td>0.114017</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.577027</td>
          <td>2.264912</td>
          <td>28.449279</td>
          <td>0.648523</td>
          <td>27.366282</td>
          <td>0.254526</td>
          <td>25.921484</td>
          <td>0.119584</td>
          <td>25.595929</td>
          <td>0.169807</td>
          <td>24.903546</td>
          <td>0.205671</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.434625</td>
          <td>0.358285</td>
          <td>26.016918</td>
          <td>0.090943</td>
          <td>25.914895</td>
          <td>0.073135</td>
          <td>25.620410</td>
          <td>0.091905</td>
          <td>25.671804</td>
          <td>0.181105</td>
          <td>25.566509</td>
          <td>0.352718</td>
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
          <td>29.321513</td>
          <td>2.044617</td>
          <td>26.709897</td>
          <td>0.165784</td>
          <td>25.481128</td>
          <td>0.049781</td>
          <td>24.995481</td>
          <td>0.052874</td>
          <td>24.745077</td>
          <td>0.081079</td>
          <td>24.960937</td>
          <td>0.215780</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.906098</td>
          <td>1.703620</td>
          <td>26.612405</td>
          <td>0.152535</td>
          <td>26.039226</td>
          <td>0.081624</td>
          <td>25.270522</td>
          <td>0.067485</td>
          <td>24.868123</td>
          <td>0.090360</td>
          <td>24.223881</td>
          <td>0.114953</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.116906</td>
          <td>0.278115</td>
          <td>26.799549</td>
          <td>0.178904</td>
          <td>26.333066</td>
          <td>0.105660</td>
          <td>26.321704</td>
          <td>0.168778</td>
          <td>25.842859</td>
          <td>0.209155</td>
          <td>25.555044</td>
          <td>0.349552</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.511451</td>
          <td>0.380394</td>
          <td>26.097183</td>
          <td>0.097575</td>
          <td>25.989841</td>
          <td>0.078143</td>
          <td>25.879882</td>
          <td>0.115333</td>
          <td>26.161616</td>
          <td>0.272123</td>
          <td>25.943797</td>
          <td>0.471080</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.984953</td>
          <td>0.542768</td>
          <td>26.880994</td>
          <td>0.191648</td>
          <td>26.528932</td>
          <td>0.125312</td>
          <td>26.085389</td>
          <td>0.137825</td>
          <td>25.954475</td>
          <td>0.229532</td>
          <td>25.526378</td>
          <td>0.341742</td>
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
          <td>26.561852</td>
          <td>0.437984</td>
          <td>26.380471</td>
          <td>0.143751</td>
          <td>26.180060</td>
          <td>0.108561</td>
          <td>25.130665</td>
          <td>0.070676</td>
          <td>24.725824</td>
          <td>0.093724</td>
          <td>24.110656</td>
          <td>0.122903</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.786389</td>
          <td>0.451532</td>
          <td>26.717802</td>
          <td>0.172642</td>
          <td>26.364638</td>
          <td>0.205811</td>
          <td>25.987172</td>
          <td>0.274209</td>
          <td>25.020509</td>
          <td>0.265184</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.130189</td>
          <td>0.590471</td>
          <td>27.629044</td>
          <td>0.371395</td>
          <td>25.873312</td>
          <td>0.138532</td>
          <td>24.908371</td>
          <td>0.112421</td>
          <td>24.334711</td>
          <td>0.152501</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.693291</td>
          <td>1.687392</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.602372</td>
          <td>0.777939</td>
          <td>26.450544</td>
          <td>0.236027</td>
          <td>25.637579</td>
          <td>0.218929</td>
          <td>26.516777</td>
          <td>0.845406</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>30.208862</td>
          <td>2.959895</td>
          <td>26.137522</td>
          <td>0.116551</td>
          <td>25.813960</td>
          <td>0.078741</td>
          <td>25.834065</td>
          <td>0.130954</td>
          <td>25.628895</td>
          <td>0.203975</td>
          <td>24.820046</td>
          <td>0.224848</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.558474</td>
          <td>0.170498</td>
          <td>25.348814</td>
          <td>0.053259</td>
          <td>25.070063</td>
          <td>0.068471</td>
          <td>24.890073</td>
          <td>0.110499</td>
          <td>25.118287</td>
          <td>0.292797</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.337718</td>
          <td>0.762266</td>
          <td>26.638579</td>
          <td>0.179825</td>
          <td>25.991395</td>
          <td>0.092411</td>
          <td>25.162221</td>
          <td>0.072993</td>
          <td>24.784821</td>
          <td>0.099112</td>
          <td>24.128387</td>
          <td>0.125337</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.617018</td>
          <td>0.460393</td>
          <td>26.650741</td>
          <td>0.183051</td>
          <td>26.458356</td>
          <td>0.139966</td>
          <td>26.353001</td>
          <td>0.206360</td>
          <td>26.097597</td>
          <td>0.303285</td>
          <td>25.712037</td>
          <td>0.461670</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.712848</td>
          <td>0.500397</td>
          <td>26.141188</td>
          <td>0.120223</td>
          <td>26.199193</td>
          <td>0.113886</td>
          <td>25.722428</td>
          <td>0.122712</td>
          <td>26.448142</td>
          <td>0.406185</td>
          <td>25.575585</td>
          <td>0.423319</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.426375</td>
          <td>0.810528</td>
          <td>26.650112</td>
          <td>0.182517</td>
          <td>26.668576</td>
          <td>0.167145</td>
          <td>26.393612</td>
          <td>0.212913</td>
          <td>25.887547</td>
          <td>0.255105</td>
          <td>25.259274</td>
          <td>0.324460</td>
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
          <td>27.331298</td>
          <td>0.692359</td>
          <td>26.569727</td>
          <td>0.147069</td>
          <td>25.995592</td>
          <td>0.078551</td>
          <td>25.081029</td>
          <td>0.057054</td>
          <td>24.723231</td>
          <td>0.079542</td>
          <td>23.937040</td>
          <td>0.089438</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.513332</td>
          <td>1.405774</td>
          <td>26.886466</td>
          <td>0.192685</td>
          <td>26.654253</td>
          <td>0.139785</td>
          <td>26.475967</td>
          <td>0.192525</td>
          <td>25.795700</td>
          <td>0.201231</td>
          <td>25.230002</td>
          <td>0.269642</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.759962</td>
          <td>0.953756</td>
          <td>30.711950</td>
          <td>2.268384</td>
          <td>27.979245</td>
          <td>0.444542</td>
          <td>25.877399</td>
          <td>0.125248</td>
          <td>24.767951</td>
          <td>0.089804</td>
          <td>24.486902</td>
          <td>0.156808</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.460430</td>
          <td>0.762607</td>
          <td>27.157100</td>
          <td>0.264491</td>
          <td>26.144898</td>
          <td>0.182119</td>
          <td>25.466137</td>
          <td>0.188973</td>
          <td>25.425263</td>
          <td>0.388178</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.118292</td>
          <td>0.278680</td>
          <td>26.118495</td>
          <td>0.099536</td>
          <td>25.828797</td>
          <td>0.067866</td>
          <td>25.899024</td>
          <td>0.117444</td>
          <td>25.340539</td>
          <td>0.136606</td>
          <td>25.263363</td>
          <td>0.277186</td>
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
          <td>27.021457</td>
          <td>0.583191</td>
          <td>26.538011</td>
          <td>0.153079</td>
          <td>25.327237</td>
          <td>0.047031</td>
          <td>24.998910</td>
          <td>0.057648</td>
          <td>24.863585</td>
          <td>0.097366</td>
          <td>24.750404</td>
          <td>0.195514</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.704975</td>
          <td>2.389353</td>
          <td>26.663936</td>
          <td>0.161646</td>
          <td>26.032824</td>
          <td>0.082519</td>
          <td>25.265255</td>
          <td>0.068354</td>
          <td>24.874127</td>
          <td>0.092347</td>
          <td>24.337500</td>
          <td>0.129039</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.739339</td>
          <td>0.466257</td>
          <td>26.599791</td>
          <td>0.157313</td>
          <td>26.153926</td>
          <td>0.094831</td>
          <td>26.135458</td>
          <td>0.151297</td>
          <td>25.943531</td>
          <td>0.238172</td>
          <td>25.658455</td>
          <td>0.396308</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.154833</td>
          <td>0.309183</td>
          <td>26.240513</td>
          <td>0.122222</td>
          <td>26.138042</td>
          <td>0.099868</td>
          <td>25.842865</td>
          <td>0.125734</td>
          <td>25.598959</td>
          <td>0.190124</td>
          <td>25.676555</td>
          <td>0.426354</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.649506</td>
          <td>0.162735</td>
          <td>26.622596</td>
          <td>0.141184</td>
          <td>26.397244</td>
          <td>0.187169</td>
          <td>25.978826</td>
          <td>0.242921</td>
          <td>26.981280</td>
          <td>0.983108</td>
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
