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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f642509b820>



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
    0      23.994413  0.198503  0.188511  
    1      25.391064  0.069449  0.059440  
    2      24.304707  0.089956  0.071945  
    3      25.291103  0.087398  0.060801  
    4      25.096743  0.070950  0.046760  
    ...          ...       ...       ...  
    99995  24.737946  0.032196  0.022952  
    99996  24.224169  0.195619  0.189500  
    99997  25.613836  0.004192  0.004074  
    99998  25.274899  0.028249  0.027749  
    99999  25.699642  0.050744  0.031302  
    
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
          <td>27.002824</td>
          <td>0.212269</td>
          <td>26.009143</td>
          <td>0.079485</td>
          <td>25.130782</td>
          <td>0.059621</td>
          <td>24.838164</td>
          <td>0.088009</td>
          <td>24.010293</td>
          <td>0.095370</td>
          <td>0.198503</td>
          <td>0.188511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.540281</td>
          <td>0.795786</td>
          <td>28.232945</td>
          <td>0.556564</td>
          <td>26.626882</td>
          <td>0.136396</td>
          <td>26.652377</td>
          <td>0.222958</td>
          <td>25.744605</td>
          <td>0.192592</td>
          <td>25.320564</td>
          <td>0.289934</td>
          <td>0.069449</td>
          <td>0.059440</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.356792</td>
          <td>0.704395</td>
          <td>30.088610</td>
          <td>1.679732</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.969760</td>
          <td>0.124704</td>
          <td>25.087070</td>
          <td>0.109469</td>
          <td>24.216223</td>
          <td>0.114189</td>
          <td>0.089956</td>
          <td>0.071945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.456403</td>
          <td>1.364218</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.390633</td>
          <td>0.259655</td>
          <td>26.399003</td>
          <td>0.180230</td>
          <td>25.608900</td>
          <td>0.171691</td>
          <td>24.887571</td>
          <td>0.202935</td>
          <td>0.087398</td>
          <td>0.060801</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.127519</td>
          <td>0.280516</td>
          <td>26.030250</td>
          <td>0.092014</td>
          <td>25.905632</td>
          <td>0.072538</td>
          <td>25.746413</td>
          <td>0.102646</td>
          <td>25.137638</td>
          <td>0.114404</td>
          <td>25.499016</td>
          <td>0.334426</td>
          <td>0.070950</td>
          <td>0.046760</td>
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
          <td>29.501392</td>
          <td>2.198967</td>
          <td>26.602748</td>
          <td>0.151278</td>
          <td>25.380884</td>
          <td>0.045542</td>
          <td>25.080636</td>
          <td>0.057026</td>
          <td>24.901041</td>
          <td>0.093012</td>
          <td>24.845760</td>
          <td>0.195931</td>
          <td>0.032196</td>
          <td>0.022952</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.166361</td>
          <td>0.617717</td>
          <td>26.660725</td>
          <td>0.158973</td>
          <td>26.065046</td>
          <td>0.083503</td>
          <td>25.167197</td>
          <td>0.061579</td>
          <td>24.726330</td>
          <td>0.079749</td>
          <td>24.294403</td>
          <td>0.122224</td>
          <td>0.195619</td>
          <td>0.189500</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.296920</td>
          <td>0.676250</td>
          <td>26.563076</td>
          <td>0.146215</td>
          <td>26.394873</td>
          <td>0.111519</td>
          <td>26.554567</td>
          <td>0.205476</td>
          <td>25.467811</td>
          <td>0.152203</td>
          <td>25.500699</td>
          <td>0.334872</td>
          <td>0.004192</td>
          <td>0.004074</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.968307</td>
          <td>0.246349</td>
          <td>26.046774</td>
          <td>0.093358</td>
          <td>26.250604</td>
          <td>0.098301</td>
          <td>25.810128</td>
          <td>0.108526</td>
          <td>25.982235</td>
          <td>0.234871</td>
          <td>25.015452</td>
          <td>0.225797</td>
          <td>0.028249</td>
          <td>0.027749</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.209092</td>
          <td>0.299590</td>
          <td>26.772617</td>
          <td>0.174865</td>
          <td>26.432431</td>
          <td>0.115230</td>
          <td>26.387059</td>
          <td>0.178415</td>
          <td>25.551425</td>
          <td>0.163488</td>
          <td>25.989034</td>
          <td>0.487213</td>
          <td>0.050744</td>
          <td>0.031302</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.026347</td>
          <td>0.272625</td>
          <td>26.083576</td>
          <td>0.111674</td>
          <td>25.136891</td>
          <td>0.079931</td>
          <td>24.695465</td>
          <td>0.102173</td>
          <td>24.082158</td>
          <td>0.134431</td>
          <td>0.198503</td>
          <td>0.188511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.548871</td>
          <td>0.878540</td>
          <td>27.083514</td>
          <td>0.262640</td>
          <td>26.796633</td>
          <td>0.187038</td>
          <td>25.885476</td>
          <td>0.138818</td>
          <td>25.671277</td>
          <td>0.214103</td>
          <td>26.898149</td>
          <td>1.029689</td>
          <td>0.069449</td>
          <td>0.059440</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.130698</td>
          <td>0.590566</td>
          <td>29.110353</td>
          <td>1.035259</td>
          <td>26.041302</td>
          <td>0.159981</td>
          <td>24.975378</td>
          <td>0.119142</td>
          <td>24.256052</td>
          <td>0.142502</td>
          <td>0.089956</td>
          <td>0.071945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.017941</td>
          <td>0.618336</td>
          <td>30.801883</td>
          <td>2.436569</td>
          <td>27.056157</td>
          <td>0.233457</td>
          <td>26.042354</td>
          <td>0.159614</td>
          <td>25.986309</td>
          <td>0.278780</td>
          <td>25.672747</td>
          <td>0.450634</td>
          <td>0.087398</td>
          <td>0.060801</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.973727</td>
          <td>0.278337</td>
          <td>26.187087</td>
          <td>0.122961</td>
          <td>26.020402</td>
          <td>0.095549</td>
          <td>25.797345</td>
          <td>0.128390</td>
          <td>25.228835</td>
          <td>0.146878</td>
          <td>25.737355</td>
          <td>0.470219</td>
          <td>0.070950</td>
          <td>0.046760</td>
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
          <td>29.468066</td>
          <td>2.288304</td>
          <td>26.434292</td>
          <td>0.150896</td>
          <td>25.373935</td>
          <td>0.053455</td>
          <td>24.970229</td>
          <td>0.061483</td>
          <td>24.818001</td>
          <td>0.101877</td>
          <td>25.384801</td>
          <td>0.355847</td>
          <td>0.032196</td>
          <td>0.022952</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.756370</td>
          <td>0.218109</td>
          <td>25.970934</td>
          <td>0.101108</td>
          <td>25.281753</td>
          <td>0.090708</td>
          <td>24.951380</td>
          <td>0.127545</td>
          <td>24.369586</td>
          <td>0.171806</td>
          <td>0.195619</td>
          <td>0.189500</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.381602</td>
          <td>0.381500</td>
          <td>26.850282</td>
          <td>0.214080</td>
          <td>26.352447</td>
          <td>0.126129</td>
          <td>26.571980</td>
          <td>0.244462</td>
          <td>25.905396</td>
          <td>0.256461</td>
          <td>25.438774</td>
          <td>0.370312</td>
          <td>0.004192</td>
          <td>0.004074</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.616953</td>
          <td>0.457338</td>
          <td>26.018529</td>
          <td>0.105293</td>
          <td>26.052786</td>
          <td>0.097377</td>
          <td>26.186822</td>
          <td>0.177590</td>
          <td>25.799324</td>
          <td>0.235592</td>
          <td>24.650326</td>
          <td>0.195541</td>
          <td>0.028249</td>
          <td>0.027749</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.638073</td>
          <td>1.602427</td>
          <td>26.745325</td>
          <td>0.197076</td>
          <td>26.478292</td>
          <td>0.141443</td>
          <td>26.188414</td>
          <td>0.178426</td>
          <td>25.676185</td>
          <td>0.213352</td>
          <td>25.646220</td>
          <td>0.436696</td>
          <td>0.050744</td>
          <td>0.031302</td>
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
          <td>27.869257</td>
          <td>1.166886</td>
          <td>26.657900</td>
          <td>0.214203</td>
          <td>26.109559</td>
          <td>0.122653</td>
          <td>25.115071</td>
          <td>0.084434</td>
          <td>24.663136</td>
          <td>0.106690</td>
          <td>23.844696</td>
          <td>0.117642</td>
          <td>0.198503</td>
          <td>0.188511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.474024</td>
          <td>0.382028</td>
          <td>27.637258</td>
          <td>0.370318</td>
          <td>26.711092</td>
          <td>0.154541</td>
          <td>26.560740</td>
          <td>0.217867</td>
          <td>25.708411</td>
          <td>0.196646</td>
          <td>25.669425</td>
          <td>0.401381</td>
          <td>0.069449</td>
          <td>0.059440</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.757613</td>
          <td>0.841495</td>
          <td>27.866984</td>
          <td>0.408295</td>
          <td>26.151441</td>
          <td>0.158686</td>
          <td>24.933460</td>
          <td>0.103883</td>
          <td>24.450327</td>
          <td>0.152056</td>
          <td>0.089956</td>
          <td>0.071945</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.982734</td>
          <td>0.564694</td>
          <td>28.597587</td>
          <td>0.752403</td>
          <td>27.817784</td>
          <td>0.389184</td>
          <td>26.014136</td>
          <td>0.139400</td>
          <td>25.389578</td>
          <td>0.152511</td>
          <td>25.685226</td>
          <td>0.412659</td>
          <td>0.087398</td>
          <td>0.060801</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.819271</td>
          <td>0.224624</td>
          <td>26.071217</td>
          <td>0.099318</td>
          <td>26.123867</td>
          <td>0.092128</td>
          <td>25.903723</td>
          <td>0.123552</td>
          <td>25.773559</td>
          <td>0.206275</td>
          <td>25.169333</td>
          <td>0.268101</td>
          <td>0.070950</td>
          <td>0.046760</td>
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
          <td>28.823909</td>
          <td>1.645537</td>
          <td>26.312858</td>
          <td>0.118849</td>
          <td>25.351336</td>
          <td>0.044838</td>
          <td>25.039918</td>
          <td>0.055619</td>
          <td>24.860435</td>
          <td>0.090699</td>
          <td>24.660796</td>
          <td>0.169307</td>
          <td>0.032196</td>
          <td>0.022952</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.013560</td>
          <td>1.262449</td>
          <td>26.254501</td>
          <td>0.151960</td>
          <td>26.207997</td>
          <td>0.133248</td>
          <td>25.092393</td>
          <td>0.082560</td>
          <td>24.685680</td>
          <td>0.108549</td>
          <td>23.991989</td>
          <td>0.133335</td>
          <td>0.195619</td>
          <td>0.189500</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.459136</td>
          <td>0.365270</td>
          <td>26.771144</td>
          <td>0.174681</td>
          <td>26.467707</td>
          <td>0.118850</td>
          <td>26.091791</td>
          <td>0.138622</td>
          <td>25.812517</td>
          <td>0.203952</td>
          <td>25.723338</td>
          <td>0.398591</td>
          <td>0.004192</td>
          <td>0.004074</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.968172</td>
          <td>0.248015</td>
          <td>26.295972</td>
          <td>0.117122</td>
          <td>26.041414</td>
          <td>0.082652</td>
          <td>25.851165</td>
          <td>0.113725</td>
          <td>25.842749</td>
          <td>0.211241</td>
          <td>25.021148</td>
          <td>0.229225</td>
          <td>0.028249</td>
          <td>0.027749</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.221792</td>
          <td>0.650443</td>
          <td>26.726135</td>
          <td>0.171448</td>
          <td>26.544413</td>
          <td>0.129977</td>
          <td>26.497490</td>
          <td>0.200525</td>
          <td>25.580406</td>
          <td>0.171425</td>
          <td>25.373047</td>
          <td>0.309250</td>
          <td>0.050744</td>
          <td>0.031302</td>
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
