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

    <pzflow.flow.Flow at 0x7f573588f490>



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
          <td>27.441402</td>
          <td>0.745574</td>
          <td>26.536767</td>
          <td>0.142946</td>
          <td>26.165526</td>
          <td>0.091226</td>
          <td>25.139204</td>
          <td>0.060068</td>
          <td>24.642387</td>
          <td>0.074050</td>
          <td>23.974113</td>
          <td>0.092388</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.037912</td>
          <td>1.082211</td>
          <td>27.249611</td>
          <td>0.260306</td>
          <td>26.625461</td>
          <td>0.136229</td>
          <td>26.107688</td>
          <td>0.140501</td>
          <td>26.119005</td>
          <td>0.262828</td>
          <td>25.461178</td>
          <td>0.324530</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.940197</td>
          <td>1.021617</td>
          <td>28.465612</td>
          <td>0.655895</td>
          <td>29.743235</td>
          <td>1.314859</td>
          <td>26.079941</td>
          <td>0.137179</td>
          <td>25.098282</td>
          <td>0.110545</td>
          <td>24.529149</td>
          <td>0.149690</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.139515</td>
          <td>1.147389</td>
          <td>28.017292</td>
          <td>0.475185</td>
          <td>27.595667</td>
          <td>0.306582</td>
          <td>26.248441</td>
          <td>0.158549</td>
          <td>25.556847</td>
          <td>0.164246</td>
          <td>24.970644</td>
          <td>0.217534</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.686186</td>
          <td>0.434941</td>
          <td>26.018313</td>
          <td>0.091055</td>
          <td>25.927208</td>
          <td>0.073935</td>
          <td>25.580724</td>
          <td>0.088753</td>
          <td>25.433831</td>
          <td>0.147828</td>
          <td>25.444796</td>
          <td>0.320324</td>
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
          <td>28.617861</td>
          <td>1.482078</td>
          <td>26.381988</td>
          <td>0.125064</td>
          <td>25.459028</td>
          <td>0.048814</td>
          <td>25.141191</td>
          <td>0.060174</td>
          <td>24.965935</td>
          <td>0.098462</td>
          <td>24.857564</td>
          <td>0.197885</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.801822</td>
          <td>0.939396</td>
          <td>26.666016</td>
          <td>0.159694</td>
          <td>26.160078</td>
          <td>0.090790</td>
          <td>25.194640</td>
          <td>0.063096</td>
          <td>24.912129</td>
          <td>0.093922</td>
          <td>24.300550</td>
          <td>0.122878</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.799304</td>
          <td>0.473544</td>
          <td>26.711468</td>
          <td>0.166006</td>
          <td>26.276055</td>
          <td>0.100518</td>
          <td>26.205772</td>
          <td>0.152861</td>
          <td>25.956236</td>
          <td>0.229868</td>
          <td>25.904995</td>
          <td>0.457585</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.950924</td>
          <td>0.242853</td>
          <td>26.155251</td>
          <td>0.102662</td>
          <td>26.144416</td>
          <td>0.089548</td>
          <td>26.029918</td>
          <td>0.131376</td>
          <td>26.047416</td>
          <td>0.247845</td>
          <td>25.335626</td>
          <td>0.293480</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.413464</td>
          <td>0.297318</td>
          <td>26.506130</td>
          <td>0.122856</td>
          <td>26.508016</td>
          <td>0.197601</td>
          <td>26.462396</td>
          <td>0.346292</td>
          <td>25.815250</td>
          <td>0.427565</td>
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
          <td>26.494066</td>
          <td>0.415982</td>
          <td>27.152607</td>
          <td>0.274612</td>
          <td>26.189648</td>
          <td>0.109473</td>
          <td>25.160683</td>
          <td>0.072577</td>
          <td>24.734134</td>
          <td>0.094410</td>
          <td>24.129665</td>
          <td>0.124946</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.091930</td>
          <td>0.565323</td>
          <td>27.068331</td>
          <td>0.231713</td>
          <td>26.519796</td>
          <td>0.234195</td>
          <td>26.192631</td>
          <td>0.323502</td>
          <td>26.041633</td>
          <td>0.581300</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.302633</td>
          <td>0.666129</td>
          <td>27.708460</td>
          <td>0.394990</td>
          <td>25.958347</td>
          <td>0.149047</td>
          <td>25.161604</td>
          <td>0.140015</td>
          <td>24.416694</td>
          <td>0.163576</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.698589</td>
          <td>1.691573</td>
          <td>29.081467</td>
          <td>1.121980</td>
          <td>27.226673</td>
          <td>0.280822</td>
          <td>26.172849</td>
          <td>0.187135</td>
          <td>25.618406</td>
          <td>0.215458</td>
          <td>25.204224</td>
          <td>0.327460</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.986169</td>
          <td>0.278839</td>
          <td>26.086456</td>
          <td>0.111486</td>
          <td>25.777212</td>
          <td>0.076227</td>
          <td>25.693789</td>
          <td>0.115945</td>
          <td>25.612392</td>
          <td>0.201171</td>
          <td>25.078079</td>
          <td>0.277938</td>
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
          <td>27.383496</td>
          <td>0.793130</td>
          <td>26.675871</td>
          <td>0.188317</td>
          <td>25.515951</td>
          <td>0.061769</td>
          <td>25.039770</td>
          <td>0.066659</td>
          <td>24.825471</td>
          <td>0.104438</td>
          <td>24.678787</td>
          <td>0.203915</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.704685</td>
          <td>0.488740</td>
          <td>26.391823</td>
          <td>0.145695</td>
          <td>26.047172</td>
          <td>0.097047</td>
          <td>25.105124</td>
          <td>0.069398</td>
          <td>24.934775</td>
          <td>0.112988</td>
          <td>24.219738</td>
          <td>0.135644</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.322049</td>
          <td>0.367386</td>
          <td>26.451087</td>
          <td>0.154458</td>
          <td>26.375402</td>
          <td>0.130290</td>
          <td>26.379346</td>
          <td>0.210959</td>
          <td>25.558508</td>
          <td>0.194566</td>
          <td>25.325820</td>
          <td>0.342897</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.592248</td>
          <td>0.457466</td>
          <td>26.084711</td>
          <td>0.114466</td>
          <td>26.228298</td>
          <td>0.116809</td>
          <td>26.107797</td>
          <td>0.170910</td>
          <td>26.425698</td>
          <td>0.399235</td>
          <td>24.924333</td>
          <td>0.252536</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.286535</td>
          <td>0.739273</td>
          <td>26.994528</td>
          <td>0.243338</td>
          <td>26.481057</td>
          <td>0.142345</td>
          <td>26.293798</td>
          <td>0.195823</td>
          <td>26.168116</td>
          <td>0.320084</td>
          <td>25.417423</td>
          <td>0.367535</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.563847</td>
          <td>0.146328</td>
          <td>26.010355</td>
          <td>0.079581</td>
          <td>25.273779</td>
          <td>0.067690</td>
          <td>24.661955</td>
          <td>0.075352</td>
          <td>23.810734</td>
          <td>0.080019</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.402069</td>
          <td>0.294824</td>
          <td>26.587754</td>
          <td>0.131984</td>
          <td>26.349484</td>
          <td>0.172980</td>
          <td>25.635111</td>
          <td>0.175718</td>
          <td>25.915929</td>
          <td>0.461744</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.445833</td>
          <td>0.781399</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.143092</td>
          <td>0.985044</td>
          <td>25.793467</td>
          <td>0.116440</td>
          <td>25.099655</td>
          <td>0.120025</td>
          <td>24.301203</td>
          <td>0.133663</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.370786</td>
          <td>1.439219</td>
          <td>28.406465</td>
          <td>0.735723</td>
          <td>27.853685</td>
          <td>0.457239</td>
          <td>25.890240</td>
          <td>0.146559</td>
          <td>25.941972</td>
          <td>0.280250</td>
          <td>24.916640</td>
          <td>0.258783</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.057048</td>
          <td>0.265149</td>
          <td>25.968494</td>
          <td>0.087263</td>
          <td>25.941064</td>
          <td>0.074954</td>
          <td>25.647672</td>
          <td>0.094274</td>
          <td>25.224101</td>
          <td>0.123510</td>
          <td>25.198695</td>
          <td>0.262961</td>
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
          <td>26.248002</td>
          <td>0.119196</td>
          <td>25.429967</td>
          <td>0.051521</td>
          <td>25.052729</td>
          <td>0.060467</td>
          <td>24.835912</td>
          <td>0.095031</td>
          <td>24.379727</td>
          <td>0.142586</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.584383</td>
          <td>0.151012</td>
          <td>25.997627</td>
          <td>0.079996</td>
          <td>25.121150</td>
          <td>0.060157</td>
          <td>24.682455</td>
          <td>0.078000</td>
          <td>24.123792</td>
          <td>0.107149</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.689506</td>
          <td>0.169819</td>
          <td>26.323187</td>
          <td>0.109975</td>
          <td>26.288385</td>
          <td>0.172405</td>
          <td>26.084399</td>
          <td>0.267369</td>
          <td>27.211587</td>
          <td>1.133676</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.135020</td>
          <td>0.111514</td>
          <td>26.117130</td>
          <td>0.098055</td>
          <td>25.816367</td>
          <td>0.122876</td>
          <td>26.067833</td>
          <td>0.280316</td>
          <td>25.363787</td>
          <td>0.334361</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.765593</td>
          <td>0.472597</td>
          <td>27.010742</td>
          <td>0.220660</td>
          <td>26.850971</td>
          <td>0.171670</td>
          <td>26.294220</td>
          <td>0.171519</td>
          <td>26.063458</td>
          <td>0.260405</td>
          <td>25.594369</td>
          <td>0.373680</td>
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
